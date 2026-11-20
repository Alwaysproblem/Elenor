// completion_bus.sv — C033 fair fan-in of event completions (W6-02).
//
// One producer per cycle is granted: round-robin, no starvation (a
// continuously-valid source is granted at latest SOURCES cycles after it
// raises valid). Producers hold valid until their ready fires; the bus never
// withdraws a grant. The arbitration has NO downstream stall input: an
// accepted completion is always sinkable by the event table (it either
// updates a live entry or only changes counters), so the bus stays fair and
// keeps servicing completions even while the table is full or while
// allocation / WAIT / terminal paths stall (frozen C033 contract).
//
// The grant index stays internal (only the payload and the per-source ready
// are consumed by the table); the pointer advances only on a grant, to the
// slot after the granted one. Reset (synchronous, active-low, cold) clears
// the pointer only; it is not a functional protocol reset.
`timescale 1ns/1ps
module completion_bus
  import c033_event_pkg::*;
#(
  parameter int unsigned SOURCES,  // producer port count, explicit at instantiation
  localparam int unsigned SRC_W = (SOURCES <= 1) ? 1 : $clog2(SOURCES)
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic [SOURCES-1:0]        req_valid_i,
  input  completion_t [SOURCES-1:0] comp_i,
  output logic [SOURCES-1:0]        grant_ready_o,

  output logic                      grant_valid_o,
  output completion_t               grant_comp_o
);

  logic [SRC_W-1:0]   ptr_q;
  logic [SOURCES-1:0] rotated;
  logic [SRC_W-1:0]   grant_off;
  logic [SRC_W-1:0]   grant_idx;
  int                 rot_idx;

  // Rotate the request vector so offset 0 is the slot just past the last
  // grant: fixed-priority pick inside the rotated view is round-robin
  // globally. All index arithmetic is done in 32-bit int before the modulo;
  // loop bounds are elaboration constants (no unbounded scan).
  always_comb begin
    for (int unsigned k = 0; k < SOURCES; k++) begin
      rot_idx     = (int'(ptr_q) + int'(k)) % int'(SOURCES);
      rotated[k]  = req_valid_i[rot_idx];
    end
  end

  // Offset of the first requesting source in rotated order (descending
  // overwrite keeps the lowest offset).
  always_comb begin
    grant_off = '0;
    for (int unsigned k = SOURCES; k > 0; k--) begin
      if (rotated[k - 1]) begin
        grant_off = SRC_W'(k - 1);
      end
    end
  end

  always_comb begin
    grant_valid_o = |rotated;
    grant_idx     = SRC_W'((int'(ptr_q) + int'(grant_off)) % int'(SOURCES));
    grant_ready_o = '0;
    grant_ready_o[grant_idx] = grant_valid_o;
    grant_comp_o  = comp_i[grant_idx];
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      ptr_q <= '0;
    end else if (grant_valid_o) begin
      ptr_q <= SRC_W'((int'(grant_idx) + 1) % int'(SOURCES));
    end
  end

endmodule
