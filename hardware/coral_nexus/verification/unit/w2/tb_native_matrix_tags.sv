// C027: the real matrix PE pipeline must not retag an older exception with a
// younger command's identity. Arithmetic datapaths are the pinned native ones.
`timescale 1ns/1ps
`include "rvv_backend.svh"
module tb_native_matrix_tags;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n, command_valid, command_ready, flush, busy;
  BLKCMD_t command;
  logic [3:1] multiply_valid;
  MULBULKTAG_t [3:1] multiply_tag;
  logic [`NUM_BLKPORT-1:0][$clog2(`NUM_MT)-1:0] read_mt, write_mt;
  logic [`NUM_BLKPORT-1:0][$clog2(`NUM_SUBTILE)-1:0] read_sub, write_sub;
  logic [`NUM_BLKPORT-1:0][`SUBTILE_SIZE*8-1:0] read_data, write_data;
  logic [`NUM_BLKPORT-1:0][`SUBTILE_SIZE-1:0] write_enable;
  logic exception_valid, exception_ready, retired;
  RVFEXP_t exception_bits;
  logic [`ROB_TAG_WIDTH-1:0] exception_tag;
  localparam logic [`ROB_TAG_WIDTH-1:0] A = 64'h0010_0000_0042_0008;
  localparam logic [`ROB_TAG_WIDTH-1:0] B = 64'h0020_0000_0084_0009;
  int seen_a, seen_b;
  zvt_pe_block #(.MULBULKPIPENUM(3), .ADDERPIPENUM(3), .BLKID(0)) dut (
    .clk(clk), .rst_n(rst_n), .blkCmdVld(command_valid), .blkCmd(command),
    .blkCmdRdy(command_ready), .mulbulkTagOutVld(multiply_valid), .mulbulkTagOut(multiply_tag),
    .readMtIdx(read_mt), .readSubIdx(read_sub), .readData(read_data),
    .writeEn(write_enable), .writeMtIdx(write_mt), .writeSubIdx(write_sub), .writeData(write_data),
    .fpexpVld(exception_valid), .fpexp(exception_bits), .fpexpTag(exception_tag),
    .fpexpRdy(exception_ready), .blkRtVld(retired), .flush(flush), .busy(busy)
  );

  task automatic submit(input logic [`ROB_TAG_WIDTH-1:0] key,
                        input logic [31:0] a, input logic [31:0] b, input int mt);
    @(negedge clk);
    command = '0;
    command.vaInfo.rob_tag = key;
    command.vaInfo.op = FPMUL;
    command.vaInfo.rndMode = fpnew_pkg::RNE;
    command.vaInfo.fsrcFmt = fpnew_pkg::FP32;
    command.vaInfo.fdstFmt = fpnew_pkg::FP32;
    command.vaInfo.isrcFmt = fpnew_pkg::INT32;
    command.vaInfo.idstFmt = fpnew_pkg::INT32;
    command.vaInfo.readMtIdx = mt;
    command.vaInfo.readMtId = 0;
    command.vaInfo.lastUopVld = 1;
    for (int r = 0; r < `TE/2*`COMPRATIO; r++) begin
      command.vaInfo.va[r] = a;
      command.vaInfo.vaMask[r] = '1;
    end
    for (int c = 0; c < `TE/2; c++) begin
      command.vbInfo.vb[c] = b;
      command.vbInfo.vbMask[c] = '1;
    end
    command_valid = 1;
    do @(posedge clk); while (!command_ready);
    @(negedge clk); command_valid = 0;
  endtask

  always @(posedge clk) begin
    if (rst_n && exception_valid && exception_ready) begin
      case (exception_tag)
        A: begin
          if (!exception_bits.nv || exception_bits.of) $fatal(1, "A must carry invalid, not B overflow");
          seen_a++;
        end
        B: begin
          if (!exception_bits.of || exception_bits.nv) $fatal(1, "B must carry overflow, not A invalid");
          seen_b++;
        end
        default: $fatal(1, "unrecognized matrix exception owner %h", exception_tag);
      endcase
    end
  end

  initial begin
    rst_n = 0; command_valid = 0; command = '0; flush = 0;
    read_data = '0; exception_ready = 1; seen_a = 0; seen_b = 0;
    repeat (3) @(posedge clk);
    @(negedge clk); rst_n = 1;
    // Distinct operations overlap in the existing pipelined PE block.
    submit(A, 32'h7f800000, 32'h00000000, 0); // infinity * zero -> NV
    submit(B, 32'h7f7fffff, 32'h7f7fffff, 1); // max finite squared -> OF
    exception_ready = 0;
    wait (exception_valid);
    if (exception_tag !== A) $fatal(1, "older exception was retagged by younger command");
    repeat (3) begin
      @(negedge clk);
      if (!exception_valid || exception_tag !== A || !exception_bits.nv)
        $fatal(1, "backpressured exception lost its payload or identity");
    end
    exception_ready = 1;
    wait (seen_a == 1 && seen_b == 1);
    repeat (12) @(posedge clk);
    if (seen_a != 1 || seen_b != 1) $fatal(1, "duplicate matrix exception after completion");
    $display("TB_NATIVE_MATRIX_TAGS ALL PASS");
    $finish;
  end
  initial begin
    #20000;
    $fatal(1, "matrix exception pipeline failed to make progress");
  end
endmodule
