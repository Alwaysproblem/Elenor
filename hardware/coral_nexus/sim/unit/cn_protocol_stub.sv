/* cn_protocol_stub.sv — unit-only stub Tile endpoint for the CN C ABI (C025 / W1-03).
 *
 * boundary_kind = "stub": NEVER admissible as CS1 evidence (W1 plan §7.6); the real Tile top
 * replaces this module with the same interface in C040.
 *
 * Models the boundary contract the ABI driver must honor:
 *   * one CTRL-channel beat link: G2T payload in, T2G echo out;
 *   * deterministic ready lowering from cfg_ready_seed (an LFSR; ready is guaranteed to open at
 *     least once every 16 cycles so bounded tests always drain);
 *   * accepted beats echo on T2G exactly once (P3 commits only the P1-accepted set);
 *   * posedge_count exports the rising-edge count (driver asserts +1 per tick);
 *   * inject_fatal_at_tick raises fatal_code (catchable);
 *   * inject_abort_at_tick executes $fatal (process-level crash, uncatchable);
 *   * glitch_enable deliberately corrupts the G2T payload after the edge so the driver's
 *     post-edge payload-stability check (CN_EP_PAYLOAD_READY_MISMATCH) can be exercised.
 */
`timescale 1ns/1ps

module cn_protocol_stub (
  input  wire        clk,
  input  wire        rst_n,

  // G2T (Group -> Tile) control beat
  input  wire        g2t_valid,
  output wire        g2t_ready,
  input  wire [63:0] g2t_payload,

  // T2G (Tile -> Group) echo beat
  output wire        t2g_valid,
  input  wire        t2g_ready,
  output wire [63:0] t2g_payload,

  // configuration / status
  input  wire [31:0] cfg_ready_seed,
  input  wire [31:0] cfg_inject_fatal_at_tick,
  input  wire [31:0] cfg_inject_abort_at_tick,
  input  wire        glitch_enable,
  output reg  [63:0] posedge_count,
  output reg  [31:0] fatal_code,
  output wire [63:0] dbg_payload_seen
);

  reg [31:0] lfsr;
  reg        echo_valid;
  reg [63:0] echo_payload;
  reg [63:0] g2t_payload_seen;  // payload as of the pre-edge sample point

  wire ready_open = lfsr[0] | (posedge_count[3:0] == 4'hF);
  assign g2t_ready = ready_open & rst_n;
  assign t2g_valid = echo_valid;
  assign t2g_payload = echo_payload;
  assign dbg_payload_seen = g2t_payload_seen;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      posedge_count   <= 64'd0;
      lfsr            <= (cfg_ready_seed == 32'd0) ? 32'hACE1 : cfg_ready_seed;
      echo_valid      <= 1'b0;
      echo_payload    <= 64'd0;
      fatal_code      <= 32'd0;
      g2t_payload_seen <= 64'd0;
    end else begin
      posedge_count <= posedge_count + 64'd1;

      // deterministic LFSR advance (x^32+x^22+x^2+x+1)
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

      if (cfg_inject_abort_at_tick == posedge_count) begin
        $fatal(1, "cn_protocol_stub: injected process-level crash");
      end
      if (cfg_inject_fatal_at_tick == posedge_count) begin
        fatal_code <= 32'd1;
      end

      // P1-accepted beat echoes exactly once (commit only the pre-edge accepted set)
      if (g2t_valid && g2t_ready) begin
        g2t_payload_seen <= g2t_payload;
        echo_valid       <= 1'b1;
        echo_payload     <= g2t_payload;
      end else if (glitch_enable) begin
        // test hook: mutate the post-edge view so the driver's payload-stability check trips
        g2t_payload_seen <= g2t_payload_seen ^ 64'h1;
      end

      if (t2g_valid && t2g_ready) begin
        echo_valid <= 1'b0;
      end
    end
  end

endmodule
