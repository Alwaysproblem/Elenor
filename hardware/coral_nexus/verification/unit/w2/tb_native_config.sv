// tb_native_config.sv — C027 / W3 native per-Context frontend config snapshot TB.
//
// Exercises the REAL RvvFrontEnd (Verilog, not a stand-in array or source-text
// check) with CONTEXT_COUNT=2. Context identity is the low CTX_BITS of the
// opaque native rob_tag; keys of Context A and Context B that encode the SAME
// scalar slot differ only in that low bit (KEY_A1 vs KEY_B1).
//
// Covers the W3 native-config-snapshot acceptance list:
//   1. Contexts A/B program different values under distinct native keys that
//      encode the same scalar slot; each emits its own immutable cfg payload.
//   2. A's committed result stays immutable while B dispatches different
//      values and the shared inputs (frm, selector) churn; pending lanes are
//      handled across ctx_select switches (per-Context config_state_valid).
//   3. A younger same-Context config op emits its OWN payload and wins the
//      working state.
//   4. vsetvli with rd=x0 and computed VL=0 still emits cfg_valid with
//      wr_vl=1 and vl=0 (validity is independent of rd and of nonzero data).
//   5. Masks mirror actual ISA field writes: vset writes VL+VTYPE and
//      implicitly zeroes MTYPE (committing the zero); msetmtypei writes
//      VL(=0)+VTYPE+MTYPE; msettn writes VL only; msettm writes MTYPE only
//      and preserves VL/VTYPE (asserted on the wr_* masks and values).
//   6. An accepted vector op keeps the accept-time CSR view (vstart update,
//      vxrm, vxsat, frm) in its emitted RVVCmd even though ctx_select
//      switches and the shared inputs change to the other Context's banked
//      values before emission. vstart updates are owner-qualified per
//      Context (A's channel lands on A even while B is selected); the
//      selected Context's vxrm/xsat/frm still land.
//   7. ctx_reset_i[0] restores Context A's working snapshot only; Context B's
//      snapshot and dispatch behavior survive unchanged.
//   8. Capacity backpressure (queue_capacity_i=0 blocks accept) and the vill
//      trap path (trap_valid_o, no cmd, no cfg, no reg write) are preserved.
//   9. Mixed-Context same-cycle dispatch: adjacent lanes of different
//      Contexts each emit their own payload (per-lane snapshots, no
//      lane-state aliasing).
//  10. An idle successor lane retaining a foreign-Context tag does not alias
//      lane 0's payload.
//  11. A trap in one Context stops the whole batch: younger lanes of OTHER
//      Contexts are suppressed AND do not publish to their working snapshot
//      (global trap prefix for emission and publication alike).
//
// Built and executed with the locked toolchain by test_w2_c027.py.
`timescale 1ns/1ps

`include "rvv_backend.svh"

module tb_native_config;
  localparam int N = 2;
  localparam int CONTEXT_COUNT = 2;
  localparam int CTX_BITS = 1;
  localparam int TAGW = `ROB_TAG_WIDTH;

  // Native keys. A keys carry ctx=0 in bit 0, B keys ctx=1. KEY_A1 and KEY_B1
  // share every other bit: the same scalar slot in two distinct native keys.
  localparam logic [TAGW-1:0] KEY_A1 = 64'h0010_0000_0042_0004;
  localparam logic [TAGW-1:0] KEY_B1 = 64'h0010_0000_0042_0005;
  localparam logic [TAGW-1:0] KEY_A2 = 64'h0020_0000_0084_0006;
  localparam logic [TAGW-1:0] KEY_B2 = 64'h0020_0000_0084_0007;
  localparam logic [TAGW-1:0] KEY_A3 = 64'h0030_0000_00C6_0008;

  localparam logic [2:0] FRM_AT_ACCEPT = 3'd2;  // FRDN
  localparam logic [2:0] FRM_LATER     = 3'd1;  // FRTZ (must NOT leak into the cmd)

  logic clk;
  logic rstn;

  logic [CONTEXT_COUNT-1:0]              vstart_update_valid_i;
  logic [CONTEXT_COUNT-1:0][`VSTART_WIDTH-1:0] vstart_update_data_i;
  logic [`VCSR_VXRM_WIDTH-1:0] vxrm_i;
  logic [`VCSR_VXSAT_WIDTH-1:0] vxsat_i;
  logic [2:0]                  frm_i;
  logic                        flush_i;
  logic [CTX_BITS-1:0]         ctx_select_i;
  logic [CONTEXT_COUNT-1:0]    ctx_reset_i;

  logic [N-1:0]      inst_valid_i;
  RVVInstruction [N-1:0] inst_data_i;
  logic [N-1:0]      inst_ready_o;
  logic [(2*N)-1:0]  reg_read_valid_i;
  logic [(2*N)-1:0][31:0] reg_read_data_i;
  logic [N-1:0][31:0]     freg_read_data_i;

  logic [N-1:0]      reg_write_valid_o;
  logic [N-1:0][4:0] reg_write_addr_o;
  logic [N-1:0][31:0] reg_write_data_o;
  logic [N-1:0][TAGW-1:0] reg_write_tag_o;

  logic [N-1:0]            cfg_valid_o;
  logic [N-1:0][TAGW-1:0]  cfg_tag_o;
  logic [N-1:0]            cfg_wr_vl_o;
  logic [N-1:0]            cfg_wr_vtype_o;
  logic [N-1:0]            cfg_wr_mtype_o;
  logic [N-1:0][`VL_WIDTH-1:0] cfg_vl_o;
  logic [N-1:0][31:0]      cfg_vtype_o;
  logic [N-1:0][31:0]      cfg_mtype_o;

  logic [N-1:0] cmd_valid_o;
  RVVCmd [N-1:0] cmd_data_o;
  logic [2:0]    queue_capacity_i;  // CAPACITYBITS = $clog2(2*N+1) = 3
  logic [2:0]    queue_capacity_o;

  logic trap_valid_o;
  RVVInstruction trap_data_o;

  logic config_state_valid;
  RVVConfigState config_state;

  logic [N-1:0] vme_mtype_valid_o;
  logic [N-1:0][31:0] vme_mtype_data_o;
  logic [N-1:0][TAGW-1:0] vme_mtype_tag_o;

  int errors;
  RVVConfigState b_snapshot;

  RvvFrontEnd#(.N(N),
               .CONTEXT_COUNT(CONTEXT_COUNT),
               .CTX_BITS(CTX_BITS)) dut (
      .clk(clk),
      .rstn(rstn),
      .vstart_update_valid_i(vstart_update_valid_i),
      .vstart_update_data_i(vstart_update_data_i),
      .vxrm_i(vxrm_i),
      .vxsat_i(vxsat_i),
      .frm_i(frm_i),
      .flush_i(flush_i),
      .ctx_select_i(ctx_select_i),
      .ctx_reset_i(ctx_reset_i),
      .inst_valid_i(inst_valid_i),
      .inst_data_i(inst_data_i),
      .inst_ready_o(inst_ready_o),
      .reg_read_valid_i(reg_read_valid_i),
      .reg_read_data_i(reg_read_data_i),
      .freg_read_data_i(freg_read_data_i),
      .reg_write_valid_o(reg_write_valid_o),
      .reg_write_addr_o(reg_write_addr_o),
      .reg_write_data_o(reg_write_data_o),
      .reg_write_tag_o(reg_write_tag_o),
      .cfg_valid_o(cfg_valid_o),
      .cfg_tag_o(cfg_tag_o),
      .cfg_wr_vl_o(cfg_wr_vl_o),
      .cfg_wr_vtype_o(cfg_wr_vtype_o),
      .cfg_wr_mtype_o(cfg_wr_mtype_o),
      .cfg_vl_o(cfg_vl_o),
      .cfg_vtype_o(cfg_vtype_o),
      .cfg_mtype_o(cfg_mtype_o),
      .cmd_valid_o(cmd_valid_o),
      .cmd_data_o(cmd_data_o),
      .queue_capacity_i(queue_capacity_i),
      .queue_capacity_o(queue_capacity_o),
      .trap_valid_o(trap_valid_o),
      .trap_data_o(trap_data_o),
      .config_state_valid(config_state_valid),
      .config_state(config_state),
      .vme_mtype_valid_o(vme_mtype_valid_o),
      .vme_mtype_data_o(vme_mtype_data_o),
      .vme_mtype_tag_o(vme_mtype_tag_o)
  );

  always #5 clk = ~clk;

  // ---------------------------------------------------------------- helpers

  task automatic check_bit(input logic got, input logic exp, input string what);
    begin
      if (got !== exp) begin
        errors = errors + 1;
        $error("FAIL %s: got %0b expected %0b", what, got, exp);
      end
    end
  endtask

  task automatic check_val(input logic [63:0] got, input logic [63:0] exp,
                           input string what);
    begin
      if (got !== exp) begin
        errors = errors + 1;
        $error("FAIL %s: got 0x%0h expected 0x%0h", what, got, exp);
      end
    end
  endtask

  task automatic check_ctx_state(input int ctx, input bit exp_valid,
                                 input logic [31:0] exp_vl,
                                 input logic [31:0] exp_sew,
                                 input bit exp_vill, input string what);
    begin
      ctx_select_i = CTX_BITS'(ctx);
      #1;
      check_bit(config_state_valid, exp_valid, {what, " valid"});
      check_val(32'(config_state.vl), exp_vl, {what, " vl"});
      check_val(32'(config_state.sew), exp_sew, {what, " sew"});
      check_bit(config_state.vill, exp_vill, {what, " vill"});
    end
  endtask

  // Dispatch one instruction on the given lane with the given accept-time
  // frm. On return (one delta past the accept edge) the combinational cfg
  // outputs for this lane are stable for the remainder of the cycle.
  task automatic dispatch_lane(input int lane, input RVVInstruction inst,
                               input logic [2:0] frm);
    begin
      @(negedge clk);
      inst_valid_i = '0;
      inst_valid_i[lane] = 1'b1;
      inst_data_i[lane] = inst;
      frm_i = frm;
      @(posedge clk);
      #1;
      inst_valid_i = '0;
    end
  endtask

  // Lane-0 convenience wrapper used by the single-lane cases.
  task automatic dispatch(input RVVInstruction inst, input logic [2:0] frm);
    begin
      dispatch_lane(0, inst, frm);
    end
  endtask

  // Dispatch one instruction on EACH lane in the same cycle (the shared frm
  // input carries one value for both lanes, as the wrapper contract allows).
  task automatic dispatch_dual(input RVVInstruction inst0,
                               input RVVInstruction inst1,
                               input logic [2:0] frm);
    begin
      @(negedge clk);
      inst_valid_i = '0;
      inst_valid_i[0] = 1'b1;
      inst_valid_i[1] = 1'b1;
      inst_data_i[0] = inst0;
      inst_data_i[1] = inst1;
      frm_i = frm;
      @(posedge clk);
      #1;
      inst_valid_i = '0;
    end
  endtask

  // Advance one cycle so the accepted op's working-state update commits and
  // the lane drains.
  task automatic settle;
    begin
      @(posedge clk);
      #1;
    end
  endtask

  function automatic RVVInstruction make_inst(input logic [TAGW-1:0] tag,
                                              input logic [24:0] bits_);
    RVVInstruction inst;
    inst.rob_tag = tag;
    inst.pc = '0;
    inst.opcode = RVV;
    inst.bits = bits_;
    return inst;
  endfunction

  // vsetvli rd, rs1, sew, lmul, ta, ma (funct3=111; zimm in bits[20:13]).
  function automatic logic [24:0] vsetvli_bits(input logic [4:0] rd,
                                                input logic [4:0] rs1,
                                                input logic [2:0] sew,
                                                input logic [2:0] lmul,
                                                input logic ta,
                                                input logic ma);
    logic [24:0] b;
    b = '0;
    b[4:0] = rd;
    b[7:5] = 3'b111;
    b[12:8] = rs1;
    b[15:13] = lmul;
    b[18:16] = sew;
    b[19] = ta;
    b[20] = ma;
    vsetvli_bits = b;
  endfunction

  // msetmtypei with altfmt=0 (Zvt §12.1.1.4).
  function automatic logic [24:0] msetmtypei_bits(input logic [1:0] mtwiden,
                                                  input logic [1:0] sew2);
    logic [24:0] b;
    b = '0;
    b[7:5] = 3'b111;
    b[24:22] = 3'b100;
    b[20:18] = 3'b100;
    b[15:13] = 3'b100;
    b[12:10] = 3'b000;
    b[9:8] = mtwiden;
    b[17:16] = sew2;
    msetmtypei_bits = b;
  endfunction

  // msettn rd, rs1 (bits[24:18]=1000100, bits[17:15]=000, bits[14:13]=00).
  function automatic logic [24:0] msettn_bits(input logic [4:0] rd,
                                              input logic [4:0] rs1);
    logic [24:0] b;
    b = '0;
    b[7:5] = 3'b111;
    b[24:18] = 7'b1000100;
    b[17:15] = 3'b000;
    b[14:13] = 2'b00;
    b[12:8] = rs1;
    b[4:0] = rd;
    msettn_bits = b;
  endfunction

  // msettm rd, rs1 (bits[24:18]=1000100, bits[17:15]=000, bits[14:13]=01).
  function automatic logic [24:0] msettm_bits(input logic [4:0] rd,
                                              input logic [4:0] rs1);
    logic [24:0] b;
    b = '0;
    b[7:5] = 3'b111;
    b[24:18] = 7'b1000100;
    b[17:15] = 3'b000;
    b[14:13] = 2'b01;
    b[12:8] = rs1;
    b[4:0] = rd;
    msettm_bits = b;
  endfunction

  // vadd.vv vd, vs2, vs1, vm=1 (funct6=000000, funct3=OPIVV=000).
  function automatic logic [24:0] vadd_vv_bits(input logic [4:0] vd,
                                               input logic [4:0] vs2,
                                               input logic [4:0] vs1);
    logic [24:0] b;
    b = '0;
    b[7:5] = 3'b000;
    b[24:19] = 6'b000000;
    b[18] = 1'b1;
    b[17:13] = vs2;
    b[12:8] = vs1;
    b[4:0] = vd;
    vadd_vv_bits = b;
  endfunction

  // ------------------------------------------------------------------- test

  initial begin
    errors = 0;
    clk = 0;
    rstn = 0;
    vstart_update_valid_i = '0;
    vstart_update_data_i = '0;
    vxrm_i = '0;
    vxsat_i = 0;
    frm_i = 3'd0;
    flush_i = 0;
    ctx_select_i = '0;
    ctx_reset_i = '0;
    inst_valid_i = '0;
    inst_data_i[0] = '0;
    inst_data_i[1] = '0;
    reg_read_valid_i = '0;
    for (int i = 0; i < 2*N; i++) reg_read_data_i[i] = 32'd0;
    for (int i = 0; i < N; i++) freg_read_data_i[i] = 32'd0;
    queue_capacity_i = 3'(N);

    repeat (4) @(negedge clk);
    rstn = 1;

    // Working state after reset: vill set, vl 0, SEW8 (RVV §3.11).
    check_ctx_state(0, 1'b1, 32'd0, 32'(SEW8), 1'b1, "post-reset A");
    check_ctx_state(1, 1'b1, 32'd0, 32'(SEW8), 1'b1, "post-reset B");

    // ---- 1. A and B program different values under keys that encode the
    //         same scalar slot; each emits its own immutable payload.
    $display("TB_NATIVE_CONFIG: contexts A B distinct keys");
    reg_read_data_i[0] = 32'd8;  // x5 = 8 (avl for A's vsetvli)
    dispatch(make_inst(KEY_A1, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "A1 cfg_valid");
    check_val(cfg_tag_o[0], KEY_A1, "A1 cfg_tag");
    check_val(reg_write_tag_o[0], KEY_A1, "A1 reg_write_tag");
    check_bit(cfg_wr_vl_o[0], 1'b1, "A1 cfg_wr_vl");
    check_bit(cfg_wr_vtype_o[0], 1'b1, "A1 cfg_wr_vtype");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "A1 cfg_wr_mtype");
    // Zvt §12.1.1.4: vset implicitly zeroes mtype — an architectural write.
    check_val(cfg_mtype_o[0], 32'h0, "A1 cfg_mtype implicit zero");
    check_val(32'(cfg_vl_o[0]), 32'd8, "A1 cfg_vl");
    // vtype = {vill=0, 0, altfmt=0, ma=1, ta=1, sew=000, lmul_orig=000}
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "A1 cfg_vtype");
    check_bit(reg_write_valid_o[0], 1'b1, "A1 reg_write_valid");
    check_val(32'(reg_write_addr_o[0]), 32'd5, "A1 reg_write_addr");
    check_val(reg_write_data_o[0], 32'd8, "A1 reg_write_data");
    settle();
    check_ctx_state(0, 1'b1, 32'd8, 32'(SEW8), 1'b0, "A after A1");
    check_ctx_state(1, 1'b1, 32'd0, 32'(SEW8), 1'b1, "B untouched by A1");

    reg_read_data_i[0] = 32'd4;  // x5 = 4 (avl for B's vsetvli)
    dispatch(make_inst(KEY_B1, vsetvli_bits(5'd6, 5'd5, SEW32, LMUL1, 1'b0, 1'b0)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "B1 cfg_valid");
    check_val(cfg_tag_o[0], KEY_B1, "B1 cfg_tag");
    check_val(reg_write_tag_o[0], KEY_B1, "B1 reg_write_tag");
    check_bit(cfg_wr_vl_o[0], 1'b1, "B1 cfg_wr_vl");
    check_bit(cfg_wr_vtype_o[0], 1'b1, "B1 cfg_wr_vtype");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "B1 cfg_wr_mtype");
    check_val(cfg_mtype_o[0], 32'h0, "B1 cfg_mtype implicit zero");
    check_val(32'(cfg_vl_o[0]), 32'd4, "B1 cfg_vl");
    // vtype = {vill=0, 0, altfmt=0, ma=0, ta=0, sew=010, lmul_orig=000}
    check_val(cfg_vtype_o[0], 32'h0000_0010, "B1 cfg_vtype");

    // ---- 2. While B's op is pending: shared inputs churn (frm flips, the
    //         selector flips) and A's snapshot must stay immutable; the
    //         pending Context reads invalid while the idle Context reads valid.
    frm_i = FRM_LATER;  // later scalar value on the shared input
    check_ctx_state(0, 1'b1, 32'd8, 32'(SEW8), 1'b0, "A while B pending");
    check_ctx_state(1, 1'b0, 32'd0, 32'(SEW8), 1'b1, "B while B pending");
    settle();
    check_ctx_state(0, 1'b1, 32'd8, 32'(SEW8), 1'b0, "A after B1 immutable");
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW32), 1'b0, "B after B1");
    frm_i = 3'd0;

    // ---- 3. Younger same-Context config emits its own payload and wins.
    $display("TB_NATIVE_CONFIG: younger same-context config");
    reg_read_data_i[0] = 32'd3;
    dispatch(make_inst(KEY_A2, vsetvli_bits(5'd5, 5'd5, SEW16, LMUL1, 1'b0, 1'b0)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "A2 cfg_valid");
    check_val(cfg_tag_o[0], KEY_A2, "A2 cfg_tag own key");
    check_val(32'(cfg_vl_o[0]), 32'd3, "A2 cfg_vl own value");
    check_val(cfg_vtype_o[0], 32'h0000_0008, "A2 cfg_vtype own value");
    settle();
    check_ctx_state(0, 1'b1, 32'd3, 32'(SEW16), 1'b0, "A after younger A2");

    // ---- 4. rd=x0 with computed VL=0 still emits a valid payload.
    $display("TB_NATIVE_CONFIG: rd x0 and vl 0 still valid");
    reg_read_data_i[0] = 32'd0;  // x5 = 0 -> avl 0 -> vl 0
    dispatch(make_inst(KEY_B2, vsetvli_bits(5'd0, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "B2 cfg_valid");
    check_val(cfg_tag_o[0], KEY_B2, "B2 cfg_tag");
    check_bit(cfg_wr_vl_o[0], 1'b1, "B2 cfg_wr_vl");
    check_bit(cfg_wr_vtype_o[0], 1'b1, "B2 cfg_wr_vtype");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "B2 cfg_wr_mtype");
    check_val(cfg_mtype_o[0], 32'h0, "B2 cfg_mtype implicit zero");
    check_val(32'(cfg_vl_o[0]), 32'd0, "B2 cfg_vl zero");
    // Upstream behavior preserved: rd=x0 writeback is still emitted (dropped
    // by the x0 register) — and now carries the producer identity.
    check_bit(reg_write_valid_o[0], 1'b1, "B2 reg_write_valid");
    check_val(32'(reg_write_addr_o[0]), 32'd0, "B2 reg_write_addr x0");
    check_val(reg_write_tag_o[0], KEY_B2, "B2 reg_write_tag");
    settle();
    check_ctx_state(1, 1'b1, 32'd0, 32'(SEW8), 1'b0, "B after B2 vl0");

    // ---- 5. mset-family field masks.
    $display("TB_NATIVE_CONFIG: mset and vset field masks");
    // msetmtypei (Zvt §12.1.1.4): writes MTYPE, VTYPE (mtwiden!=0 forces
    // ta=ma=1, SEW8->LMUL1) and clears VL to 0 — all three mask bits set.
    dispatch(make_inst(KEY_A1, msetmtypei_bits(2'b01, 2'b00)), 3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "msetmtypei cfg_valid");
    check_bit(cfg_wr_vl_o[0], 1'b1, "msetmtypei cfg_wr_vl cleared");
    check_val(32'(cfg_vl_o[0]), 32'd0, "msetmtypei cfg_vl zero");
    check_bit(cfg_wr_vtype_o[0], 1'b1, "msetmtypei cfg_wr_vtype");
    // vtype = {vill=0, 0, altfmt=0, ma=1, ta=1, sew=000, lmul_orig=000}
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "msetmtypei cfg_vtype");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "msetmtypei cfg_wr_mtype");
    // mtype = {0, tm=0, 0, tk=0, 0, mtwiden=01}
    check_val(cfg_mtype_o[0], 32'h0000_0001, "msetmtypei cfg_mtype");
    check_bit(vme_mtype_valid_o[0], 1'b1, "msetmtypei vme_mtype_valid");
    check_val(vme_mtype_tag_o[0], KEY_A1, "msetmtypei vme_mtype_tag");
    check_bit(reg_write_valid_o[0], 1'b0, "msetmtypei no rd write");
    settle();
    // msettn: VL only (TN maps to vector length). After msetmtypei the
    // Context has mtwiden=01, so msettn takes x[rs1] directly (Zvt
    // §12.1.1.5); SEW8 LMUL1 vlmax=16 leaves vl = 12 unclamped.
    reg_read_data_i[0] = 32'd12;
    dispatch(make_inst(KEY_A2, msettn_bits(5'd7, 5'd5)), 3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "msettn cfg_valid");
    check_val(cfg_tag_o[0], KEY_A2, "msettn cfg_tag");
    check_bit(cfg_wr_vl_o[0], 1'b1, "msettn cfg_wr_vl");
    check_bit(cfg_wr_vtype_o[0], 1'b0, "msettn cfg_wr_vtype masked");
    check_bit(cfg_wr_mtype_o[0], 1'b0, "msettn cfg_wr_mtype masked");
    check_val(32'(cfg_vl_o[0]), 32'd12, "msettn cfg_vl");
    check_bit(reg_write_valid_o[0], 1'b1, "msettn reg_write_valid");
    check_val(32'(reg_write_addr_o[0]), 32'd7, "msettn reg_write_addr");
    check_val(reg_write_data_o[0], 32'd12, "msettn reg_write_data");
    check_bit(vme_mtype_valid_o[0], 1'b0, "msettn vme_mtype_valid");
    settle();
    // msettm: MTYPE only (Zvt §12.1.1.5); preserves VL and VTYPE even in the
    // working snapshot (vconfig_change stays 0 for this op). Context still
    // has mtwiden=01 (msettn preserved it), so the write is legal.
    reg_read_data_i[0] = 32'd20;  // x5 = 20 -> tm = min(20, TE=16) = 16
    dispatch(make_inst(KEY_A1, msettm_bits(5'd8, 5'd5)), 3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "msettm cfg_valid");
    check_val(cfg_tag_o[0], KEY_A1, "msettm cfg_tag");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "msettm cfg_wr_mtype");
    check_bit(cfg_wr_vl_o[0], 1'b0, "msettm cfg_wr_vl preserved");
    check_bit(cfg_wr_vtype_o[0], 1'b0, "msettm cfg_wr_vtype preserved");
    // mtype = {0, tm=16, 0, tk=0, 0, mtwiden=01}
    check_val(cfg_mtype_o[0], 32'h0000_4001, "msettm cfg_mtype");
    // Masked-off payload fields still carry the PRESERVED values.
    check_val(32'(cfg_vl_o[0]), 32'd12, "msettm cfg_vl preserved");
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "msettm cfg_vtype preserved");
    check_bit(reg_write_valid_o[0], 1'b1, "msettm reg_write_valid");
    check_val(32'(reg_write_addr_o[0]), 32'd8, "msettm reg_write_addr");
    check_val(reg_write_data_o[0], 32'd16, "msettm reg_write_data");
    check_val(reg_write_tag_o[0], KEY_A1, "msettm reg_write_tag");
    check_bit(vme_mtype_valid_o[0], 1'b1, "msettm vme_mtype_valid");
    check_val(vme_mtype_data_o[0], 32'h0000_4001, "msettm vme_mtype_data");
    settle();
    check_ctx_state(0, 1'b1, 32'd12, 32'(SEW8), 1'b0, "A after msettm vl kept");
    // vset (Zvt §12.1.1.4): writes VL+VTYPE AND implicitly zeroes MTYPE —
    // the mask must carry the zero write so retire-point commit clears a
    // previously nonzero mtype (tm=16 above).
    reg_read_data_i[0] = 32'd8;
    dispatch(make_inst(KEY_A1, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "vset-after-mset cfg_valid");
    check_bit(cfg_wr_vl_o[0], 1'b1, "vset-after-mset cfg_wr_vl");
    check_bit(cfg_wr_vtype_o[0], 1'b1, "vset-after-mset cfg_wr_vtype");
    check_bit(cfg_wr_mtype_o[0], 1'b1, "vset-after-mset cfg_wr_mtype zero");
    check_val(cfg_mtype_o[0], 32'h0, "vset-after-mset cfg_mtype zeroed");
    check_val(32'(cfg_vl_o[0]), 32'd8, "vset-after-mset cfg_vl");
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "vset-after-mset cfg_vtype");
    settle();
    check_ctx_state(0, 1'b1, 32'd8, 32'(SEW8), 1'b0, "A after mask sequence");

    // ---- 6. Accept-time CSR capture: the vstart write, vxrm, vxsat and frm
    //          visible at accept survive a selector switch that rebanks the
    //          shared inputs to Context B's values before the command is
    //          sampled; B's banked vstart write still lands on B.
    $display("TB_NATIVE_CONFIG: accept-time csr capture");
    vxrm_i = 2'b10;        // A's banked values while A is selected
    vxsat_i = 1'b1;
    vstart_update_valid_i = 2'b01;   // A's channel: write for the accepted op
    vstart_update_data_i[0] = 'd3;
    dispatch(make_inst(KEY_A2, vadd_vv_bits(5'd1, 5'd2, 5'd3)), FRM_AT_ACCEPT);
    // Selector flips A->B and every shared input changes to B's banked
    // values BEFORE the emitted command is sampled.
    ctx_select_i = CTX_BITS'(1);
    vxrm_i = 2'b01;
    vxsat_i = 1'b0;
    frm_i = FRM_LATER;
    vstart_update_valid_i = 2'b10;   // B's channel: B's vstart write
    vstart_update_data_i[1] = 'd5;
    #1;
    check_bit(cmd_valid_o[0], 1'b1, "vadd cmd_valid");
    check_val(cmd_data_o[0].rob_tag, KEY_A2, "vadd cmd rob_tag");
    check_val(32'(cmd_data_o[0].arch_state.frm), 32'(FRM_AT_ACCEPT),
              "vadd cmd frm captured at accept");
    check_val(32'(cmd_data_o[0].arch_state.xrm), 32'd2, "vadd cmd xrm captured");
    check_val(32'(cmd_data_o[0].arch_state.xsat), 32'd1, "vadd cmd xsat captured");
    check_val(32'(cmd_data_o[0].arch_state.vstart), 32'd3,
              "vadd cmd vstart captured");
    check_bit(trap_valid_o, 1'b0, "vadd no trap");
    settle();
    // B's banked vstart write landed on the SELECTED Context (valid CSR
    // writes are not suppressed by the capture mechanism).
    check_ctx_state(1, 1'b1, 32'd0, 32'(SEW8), 1'b0, "B valid after churn");
    ctx_select_i = CTX_BITS'(1);
    #1;
    check_val(32'(config_state.vstart), 32'd5, "B registered vstart from write");
    check_val(32'(config_state.xrm), 32'd1, "B registered xrm banked");
    // A's working snapshot kept A's values throughout the churn.
    ctx_select_i = CTX_BITS'(0);
    #1;
    check_val(32'(config_state.xrm), 32'd2, "A xrm kept through churn");
    check_val(32'(config_state.xsat), 32'd1, "A xsat kept through churn");
    // Restore the quiescent input view for later sections.
    vstart_update_valid_i = '0;
    vstart_update_data_i = '0;
    vxrm_i = '0;
    vxsat_i = 1'b0;
    frm_i = 3'd0;

    // Idle-successor CSR staleness: prime lane 1 with an A op carrying
    // distinct captures (xrm=2'b11 and a vstart write), let it drain — the
    // idle lane retains A's tag and those stale captures — then a fresh A op
    // on lane 0 must own the publication, and a later ctx_reset must not be
    // undone by the idle lane.
    $display("TB_NATIVE_CONFIG: idle-successor csr staleness");
    reg_read_data_i[2] = 32'd8;  // lane1 rs1 for the prime vsetvli
    vxrm_i = 2'b11;
    vxsat_i = 1'b0;
    vstart_update_valid_i = 2'b01;   // prime's A-channel write (consumed on retire)
    vstart_update_data_i[0] = 'd4;
    dispatch_lane(1, make_inst(KEY_A1, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
                  3'd0);
    check_val(32'(cfg_vl_o[1]), 32'd8, "csr prime lane1 vl");
    settle();
    // Fresh A op on lane 0 with different banked CSR values.
    vstart_update_valid_i = '0;
    vstart_update_data_i = '0;
    vxrm_i = 2'b01;
    reg_read_data_i[0] = 32'd8;
    dispatch(make_inst(KEY_A1, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)), 3'd0);
    check_val(32'(cfg_vl_o[0]), 32'd8, "csr fresh lane0 vl");
    settle();
    ctx_select_i = CTX_BITS'(0);
    #1;
    check_val(32'(config_state.xrm), 32'd1,
              "A xrm from lane0 fresh not lane1 stale");
    check_val(32'(config_state.vstart), 32'd0, "A vstart consumed not resurrected");
    // ctx_reset must stick: the idle lane's stale captures must not leak
    // back into the reset snapshot on later cycles.
    vxrm_i = '0;
    @(negedge clk);
    ctx_reset_i = 2'b01;
    @(posedge clk);
    #1;
    ctx_reset_i = 2'b00;
    settle();
    ctx_select_i = CTX_BITS'(0);
    #1;
    check_val(32'(config_state.xrm), 32'(RNU), "A xrm at reset despite idle lane");
    check_val(32'(config_state.vstart), 32'd0, "A vstart stays 0 after reset");

    // Owner-qualified vstart routing: A's update channel lands on A while
    // the selector queries and accepts a B instruction in the same beat —
    // vstart is NOT routed by ctx_select.
    $display("TB_NATIVE_CONFIG: a update while selecting b");
    ctx_select_i = CTX_BITS'(1);
    vstart_update_valid_i = 2'b01;   // A's channel only; B's is silent
    vstart_update_data_i[0] = 'd6;
    vstart_update_data_i[1] = 'd7;   // ignored: B's channel is silent
    reg_read_data_i[0] = 32'd4;
    dispatch(make_inst(KEY_B1, vsetvli_bits(5'd6, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "ab-update B cfg_valid");
    check_val(cfg_tag_o[0], KEY_B1, "ab-update B cfg_tag");
    check_val(32'(cfg_vl_o[0]), 32'd4, "ab-update B cfg_vl");
    settle();
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW8), 1'b0, "B after ab-update vset");
    check_val(32'(config_state.vstart), 32'd0,
              "B vstart not polluted by A update");
    ctx_select_i = CTX_BITS'(0);
    #1;
    check_val(32'(config_state.vstart), 32'd6,
              "A vstart updated while selecting B");
    // Restore quiescent inputs.
    vstart_update_valid_i = '0;
    vstart_update_data_i = '0;

    // flush must NOT write vstart by itself: the shim drives silent channels
    // with valid=false, bits=0, so a flush-qualified write would clear B's
    // vstart on an A-only update beat. Only a Context's OWN valid channel
    // writes its vstart.
    $display("TB_NATIVE_CONFIG: a update with flush keeps b vstart");
    vstart_update_valid_i = 2'b10;   // give B a nonzero vstart
    vstart_update_data_i[1] = 'd5;
    settle();
    @(negedge clk);
    vstart_update_valid_i = 2'b01;   // A's channel only, flush in the same beat
    vstart_update_data_i[0] = 'd2;
    flush_i = 1'b1;
    @(posedge clk);
    #1;
    flush_i = 1'b0;
    vstart_update_valid_i = '0;
    vstart_update_data_i = '0;
    settle();  // an extra idle beat must also preserve both
    ctx_select_i = CTX_BITS'(0);
    #1;
    check_val(32'(config_state.vstart), 32'd2, "A vstart from update with flush");
    ctx_select_i = CTX_BITS'(1);
    #1;
    check_val(32'(config_state.vstart), 32'd5, "B vstart preserved across A flush");

    // ---- 7. ctx_reset_i restores only the named Context.
    $display("TB_NATIVE_CONFIG: per-context reset isolation");
    ctx_select_i = CTX_BITS'(1);
    #1;
    b_snapshot = config_state;
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW8), 1'b0, "B pre-reset");
    @(negedge clk);
    ctx_reset_i = 2'b01;  // reuse Context A only
    @(posedge clk);
    #1;
    ctx_reset_i = 2'b00;
    check_ctx_state(0, 1'b1, 32'd0, 32'(SEW8), 1'b1, "A after ctx_reset");
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW8), 1'b0, "B intact after A reset");
    ctx_select_i = CTX_BITS'(1);
    #1;
    check_val(32'(config_state.vl), 32'(b_snapshot.vl), "B struct vl equal");
    check_val(32'(config_state.lmul_orig), 32'(b_snapshot.lmul_orig),
              "B struct lmul_orig equal");
    check_bit(config_state.vill, b_snapshot.vill, "B struct vill equal");
    // B still dispatches and computes after A's reset.
    reg_read_data_i[0] = 32'd6;
    dispatch(make_inst(KEY_B2, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
             3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "B post-A-reset cfg_valid");
    check_val(cfg_tag_o[0], KEY_B2, "B post-A-reset cfg_tag");
    check_val(32'(cfg_vl_o[0]), 32'd6, "B post-A-reset cfg_vl");
    settle();
    check_ctx_state(1, 1'b1, 32'd6, 32'(SEW8), 1'b0, "B state post-A-reset");

    // ---- 8. Backpressure and trap behavior preserved.
    $display("TB_NATIVE_CONFIG: backpressure and trap");
    @(negedge clk);
    queue_capacity_i = 3'd0;
    inst_valid_i = '0;
    inst_valid_i[0] = 1'b1;
    inst_data_i[0] = make_inst(KEY_A3, vadd_vv_bits(5'd1, 5'd2, 5'd3));
    @(posedge clk);
    #1;
    check_bit(inst_ready_o[0], 1'b0, "backpressure inst_ready low");
    check_bit(cfg_valid_o[0], 1'b0, "backpressure no cfg");
    check_bit(cmd_valid_o[0], 1'b0, "backpressure no cmd");
    inst_valid_i = '0;
    queue_capacity_i = 3'(N);
    settle();
    // A is vill (ctx_reset above, never reconfigured): a vector op traps.
    dispatch(make_inst(KEY_A3, vadd_vv_bits(5'd1, 5'd2, 5'd3)), 3'd0);
    check_bit(trap_valid_o, 1'b1, "vill trap valid");
    check_val(trap_data_o.rob_tag, KEY_A3, "vill trap rob_tag");
    check_bit(cmd_valid_o[0], 1'b0, "vill trap no cmd");
    check_bit(cfg_valid_o[0], 1'b0, "vill trap no cfg");
    check_bit(reg_write_valid_o[0], 1'b0, "vill trap no reg write");
    settle();
    check_ctx_state(0, 1'b1, 32'd0, 32'(SEW8), 1'b1, "A stays reset after trap");
    check_ctx_state(1, 1'b1, 32'd6, 32'(SEW8), 1'b0, "B stays live after A trap");

    // ---- 9. Mixed-Context same-cycle dispatch: each lane keeps its OWN
    //         Context's computation. Per-lane snapshots (lane_in/lane_out)
    //         guarantee an adjacent Context-B lane never aliases lane 0's
    //         Context-A payload.
    $display("TB_NATIVE_CONFIG: mixed-context same cycle");
    reg_read_data_i[0] = 32'd2;  // lane0 rs1 (x5) = 2, Context A
    reg_read_data_i[2] = 32'd5;  // lane1 rs1 (x5) = 5, Context B
    dispatch_dual(
        make_inst(KEY_A1, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
        make_inst(KEY_B1, vsetvli_bits(5'd6, 5'd5, SEW16, LMUL1, 1'b0, 1'b0)),
        3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "dual lane0 cfg_valid");
    check_val(cfg_tag_o[0], KEY_A1, "dual lane0 cfg_tag");
    check_val(32'(cfg_vl_o[0]), 32'd2, "dual lane0 cfg_vl A own");
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "dual lane0 cfg_vtype A own");
    check_val(reg_write_tag_o[0], KEY_A1, "dual lane0 reg_write_tag");
    check_val(reg_write_data_o[0], 32'd2, "dual lane0 reg_write_data");
    check_bit(cfg_valid_o[1], 1'b1, "dual lane1 cfg_valid");
    check_val(cfg_tag_o[1], KEY_B1, "dual lane1 cfg_tag");
    check_val(32'(cfg_vl_o[1]), 32'd5, "dual lane1 cfg_vl B own");
    check_val(cfg_vtype_o[1], 32'h0000_0008, "dual lane1 cfg_vtype B own");
    check_val(reg_write_tag_o[1], KEY_B1, "dual lane1 reg_write_tag");
    check_val(reg_write_data_o[1], 32'd5, "dual lane1 reg_write_data");
    settle();
    check_ctx_state(0, 1'b1, 32'd2, 32'(SEW8), 1'b0, "A after dual");
    check_ctx_state(1, 1'b1, 32'd5, 32'(SEW16), 1'b0, "B after dual");

    // ---- 10. Idle successor lane retains a foreign-Context tag: lane 0's
    //          payload must not read the idle lane's Context state.
    $display("TB_NATIVE_CONFIG: idle successor lane");
    reg_read_data_i[2] = 32'd7;  // lane1 rs1 = 7, Context B (SEW32 vl clamps to 4)
    dispatch_lane(1, make_inst(KEY_B2, vsetvli_bits(5'd5, 5'd5, SEW32, LMUL1, 1'b1, 1'b1)),
                  3'd0);
    check_bit(cfg_valid_o[1], 1'b1, "prime lane1 cfg_valid");
    check_val(32'(cfg_vl_o[1]), 32'd4, "prime lane1 B cfg_vl");
    settle();
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW32), 1'b0, "B after prime");
    // Lane 1 is now idle but still holds a Context-B tag; A dispatches on
    // lane 0 and must emit A's own computed payload.
    reg_read_data_i[0] = 32'd9;
    dispatch(make_inst(KEY_A2, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)), 3'd0);
    check_bit(cfg_valid_o[0], 1'b1, "idle-successor lane0 cfg_valid");
    check_val(cfg_tag_o[0], KEY_A2, "idle-successor lane0 cfg_tag");
    check_val(32'(cfg_vl_o[0]), 32'd9, "idle-successor lane0 cfg_vl A own");
    check_val(cfg_vtype_o[0], 32'h0000_00C0, "idle-successor lane0 cfg_vtype");
    settle();
    check_ctx_state(0, 1'b1, 32'd9, 32'(SEW8), 1'b0, "A after idle-successor");

    // ---- 11. A trap in one Context stops the whole batch: lane 0's A op
    //          traps (vill), lane 1's B vset is suppressed AND must not
    //          publish to B's working snapshot (global trap-prefix for both
    //          emission and publication — no shadow advance without emit).
    $display("TB_NATIVE_CONFIG: trap stops batch globally");
    // Reuse Context A: reset its snapshot so it is vill again.
    @(negedge clk);
    ctx_reset_i = 2'b01;
    @(posedge clk);
    #1;
    ctx_reset_i = 2'b00;
    check_ctx_state(0, 1'b1, 32'd0, 32'(SEW8), 1'b1, "A re-reset for trap test");
    reg_read_data_i[2] = 32'd6;  // lane1 rs1 = 6 for B's vsetvli
    dispatch_dual(
        make_inst(KEY_A3, vadd_vv_bits(5'd1, 5'd2, 5'd3)),  // lane0: A vill -> trap
        make_inst(KEY_B2, vsetvli_bits(5'd5, 5'd5, SEW8, LMUL1, 1'b1, 1'b1)),
        3'd0);
    check_bit(trap_valid_o, 1'b1, "mixed trap valid");
    check_val(trap_data_o.rob_tag, KEY_A3, "mixed trap rob_tag");
    check_bit(cfg_valid_o[1], 1'b0, "mixed trap lane1 cfg suppressed");
    check_bit(reg_write_valid_o[1], 1'b0, "mixed trap lane1 reg_write suppressed");
    check_bit(cmd_valid_o[0], 1'b0, "mixed trap lane0 cmd suppressed");
    check_bit(cmd_valid_o[1], 1'b0, "mixed trap lane1 cmd suppressed");
    settle();
    // B's working snapshot must be UNCHANGED (vl=4 SEW32 from the prime):
    // publication stopped at the trap exactly like emission did.
    check_ctx_state(1, 1'b1, 32'd4, 32'(SEW32), 1'b0, "B shadow unchanged after trap");
    check_ctx_state(0, 1'b1, 32'd0, 32'(SEW8), 1'b1, "A stays reset after mixed trap");

    if (errors == 0) begin
      $display("TB_NATIVE_CONFIG ALL PASS");
    end else begin
      $display("TB_NATIVE_CONFIG: %0d CHECK(S) FAILED", errors);
    end
    $finish;
  end

  initial begin
    #100000;
    $error("TB_NATIVE_CONFIG: TIMEOUT");
    $finish;
  end
endmodule
