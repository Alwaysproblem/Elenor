// C027: real native retire stage, per-parent deltas with architectural FFLAGS bits.
`timescale 1ns/1ps
`include "rvv_backend.svh"
module tb_native_flags;
  localparam logic [4:0] NV = 5'b10000, DZ = 5'b01000, OF = 5'b00100,
                         UF = 5'b00010, NX = 5'b00001;
  // Same scalar slot (bits 4:1), different Context and generation/op-sequence fields.
  localparam logic [`ROB_TAG_WIDTH-1:0] A = 64'h0010_0000_0042_0008;
  localparam logic [`ROB_TAG_WIDTH-1:0] B = 64'h0020_0000_0084_0009;
  logic [`NUM_RT_UOP-1:0] valid, ready, xready, fready;
  ROB2RT_t [`NUM_RT_UOP-1:0] data;
  logic vcsr_ready;
  logic [`NUM_RT_UOP-1:0] flags_valid;
  logic [`NUM_RT_UOP-1:0][`ROB_TAG_WIDTH-1:0] flags_tag;
  logic [`NUM_RT_UOP-1:0][4:0] flags;
  logic [`NUM_RT_UOP-1:0] sat;
`ifdef ZVT_ON
  logic [`NUM_BLK-1:0] matrix_valid, matrix_ready, matrix_out_valid;
  RVFEXP_t [`NUM_BLK-1:0] matrix_exp;
  logic [`NUM_BLK-1:0][`ROB_TAG_WIDTH-1:0] matrix_tag, matrix_out_tag;
  logic [`NUM_BLK-1:0][4:0] matrix_out_flags;
`endif
  rvv_backend_retire dut (
    .rob2rt_write_valid(valid), .rob2rt_write_data(data), .rt2rob_write_ready(ready),
    .rt2xrf_write_valid(), .rt2rvs_write_data(), .rvs2rt_write_ready(xready),
    .rt2frf_write_valid(), .frf2rt_write_ready(fready),
    .rt2vrf_write_valid(), .rt2vrf_write_data(),
    .rt2vcsr_write_valid(), .rt2vcsr_write_data(), .vcsr2rt_write_ready(vcsr_ready),
    .rvvrob2rt_valid(), .rvvrob2rt_data(),
    .cn_flags_valid_o(flags_valid), .cn_flags_tag_o(flags_tag),
    .cn_fflags_o(flags), .cn_vxsat_o(sat)
`ifdef ZVT_ON
    ,.zvtFpexpVld(matrix_valid), .zvtFpexp(matrix_exp), .zvtFpexpRdy(matrix_ready),
    .zvtFpexpTag(matrix_tag), .cn_zvt_flags_valid_o(matrix_out_valid),
    .cn_zvt_flags_tag_o(matrix_out_tag), .cn_zvt_fflags_o(matrix_out_flags)
`endif
  );

  task automatic check(input logic condition, input string message);
    if (condition !== 1'b1) $fatal(1, "%s", message);
  endtask
  task automatic clear_inputs;
    valid = '0; data = '0; xready = '1; fready = '1; vcsr_ready = 1;
`ifdef ZVT_ON
    matrix_valid = '0; matrix_exp = '0; matrix_tag = '0;
`endif
  endtask
  task automatic lane(input int i, input logic [`ROB_TAG_WIDTH-1:0] tag,
                      input logic [4:0] bits, input logic saturation);
    valid[i] = 1;
    data[i].rob_tag = tag;
    data[i].w_valid = 1;
    data[i].w_type = VRF;
    data[i].w_index = `REGIDX_WIDTH'(i + 1);
    data[i].res_updating_end = 1;
    data[i].last_uop_valid = 1;
    for (int e = 0; e < `VLENB; e++) data[i].vd_type[e] = BODY_ACTIVE;
    data[i].fpexp[0].nv = bits[4]; data[i].fpexp[0].dz = bits[3];
    data[i].fpexp[0].of = bits[2]; data[i].fpexp[0].uf = bits[1];
    data[i].fpexp[0].nx = bits[0];
    data[i].vxsaturate[0] = saturation;
  endtask

  initial begin
    check(`ROB_TAG_WIDTH == 64, "native identity word was truncated");
    check(A[4:1] == B[4:1] && A[0] != B[0], "fixture must alias only scalar slot");
    clear_inputs(); lane(0, A, NX, 1); lane(2, B, NV | DZ, 0); #1;
    check(flags_valid == 4'b0101, "independent same-beat lanes");
    check(flags_tag[0] == A && flags_tag[2] == B, "full parent keys preserved");
    check(flags[0] == NX && flags[2] == (NV | DZ), "flags smeared across parents");
    check(sat[0] == 1 && sat[2] == 0, "saturation belongs only to its parent");

    clear_inputs(); lane(0, A, NV | OF, 1);
    data[0].vd_type[0] = TAIL; #1;
    check(!flags_valid[0] && flags[0] == 0 && !sat[0], "tail bytes raised flags");
    data[0].vd_type[0] = BODY_INACTIVE; #1;
    check(!flags_valid[0], "masked bytes raised flags");
    data[0].vd_type[0] = BODY_ACTIVE; #1;
    check(flags_valid[0] && flags[0] == (NV | OF) && sat[0], "active positive control");

    // Native scalar-result backpressure is part of the real retirement handshake.
    clear_inputs(); lane(0, A, UF, 0); lane(1, B, OF, 1);
    data[0].w_type = XRF; xready[0] = 0; #1;
    check(flags_valid == 0, "stalled retirement emitted a delta");
    xready[0] = 1; #1;
    check(flags_valid[1:0] == 2'b11 && flags[0] == UF && flags[1] == OF,
          "release lost a same-cycle delta");

    clear_inputs(); lane(0, A, DZ, 0); lane(1, B, NV, 1); lane(2, A, OF, 1);
    data[1].trap_flag = 1; #1;
    check(flags_valid == 4'b0001, "trapping and younger lanes must be suppressed");
    check(flags[0] == 5'b01000, "RISC-V DZ is bit 3");
    // An invalid lane's stale payload is not a trap on a younger valid lane.
    valid[1] = 0; #1;
    check(flags_valid == 4'b0101, "invalid trap payload suppressed a live instruction");

    clear_inputs(); lane(0, A, NX, 0);
    data[0].res_updating_end = 0; data[0].last_uop_valid = 0; data[0].w_valid = 0; #1;
    check(flags_valid[0] && flags[0] == NX, "early non-writing uop flags were lost");
    lane(0, A, DZ, 0); #1;
    check(flags_valid[0] && flags_tag[0] == A && flags[0] == DZ,
          "later same-parent delta was not independently exported");
`ifdef ZVT_ON
    clear_inputs();
    matrix_valid[0] = 1; matrix_tag[0] = A; matrix_exp[0].nv = 1;
    matrix_valid[3] = 1; matrix_tag[3] = B; matrix_exp[3].of = 1;
    #1;
    check(matrix_ready == '1 && matrix_out_valid == 4'b1001, "matrix delta lanes dropped");
    check(matrix_out_tag[0] == A && matrix_out_tag[3] == B &&
          matrix_out_flags[0] == NV && matrix_out_flags[3] == OF,
          "matrix blocks merged distinct parents");
`endif
    $display("TB_NATIVE_FLAGS ALL PASS");
    $finish;
  end
endmodule
