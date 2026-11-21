// c034_reset_pkg.sv — C034 (W6-03) reset-segment shared types.
//
// References c033_event_pkg types (word_t / owner_t) directly; no shadow
// word or owner aliases are defined here. Every container is the C017
// 64-bit transport word, not a silicon width. Encoded values follow the
// sealed orders: reset phases use the C017 frozen phase order 0..6; the
// wire fault code is 0..10 (protocol/generated/cn_layout.json
// enums.fault_code) and is DISTINCT from the Python FaultCode IntEnum
// values 1..11 (runtime/fault_ring.py) — same names, shifted by one, and
// the two encodings never mix.
`timescale 1ns/1ps
package c034_reset_pkg;
  import c033_event_pkg::*;

  // Reset segment phases, C017 frozen order (w6_terminal_reset
  // #reset_endpoint.phases: FREEZE_ISSUE, CANCEL_UNISSUED, DRAIN,
  // MARK_EVENTS, INVALIDATE_RESIDENCY, REBUILD_CREDIT, RESUME_DONE).
  typedef enum logic [2:0] {
    RP_FREEZE     = 3'd0,
    RP_CANCEL     = 3'd1,
    RP_DRAIN      = 3'd2,
    RP_MARK       = 3'd3,
    RP_INVALIDATE = 3'd4,
    RP_REBUILD    = 3'd5,
    RP_RESUME     = 3'd6
  } reset_phase_t;

  // CANCEL_TASK classification (w6_terminal_reset #cancel.classification).
  typedef enum logic [1:0] {
    CC_APPLY     = 2'd0,  // matches the live task binding
    CC_STALE     = 2'd1,  // no live binding: count + idempotent ACK
    CC_DUPLICATE = 2'd2,  // duplicate command: idempotent
    CC_CONFLICT  = 2'd3   // live task, conflicting generation: protocol error
  } cancel_class_t;

  // Full first-fault record, C015 #first_fault_latch.record_fields. The
  // task/context/tile attribution is the OwnerTag BY REFERENCE: owner.ctx
  // and op_seq are the actual owner tag fields, with no duplicate
  // tile/ctx/Task copy beside them. `code` carries the wire encoding 0..10.
  typedef struct packed {
    owner_t owner;
    word_t  op_seq;
    word_t  code;
    word_t  source_id;
    word_t  severity;
    word_t  command_id;
    word_t  event_id;
    word_t  program_id;
    word_t  desc_id;
    word_t  group_id;
    word_t  queue_id;
    word_t  slot_id;
    word_t  patch_id;
    word_t  engine_id;
    word_t  offending_addr;
    word_t  aux0;
    word_t  aux1;
    word_t  cycle;
  } fault_t;

  // Wire fault codes, C017 enums.fault_code order 0..10. Named word
  // constants, not a new enum width: fault_t.code is a word container.
  localparam word_t FC_INVALID_DESCRIPTOR       = 64'd0;
  localparam word_t FC_UNSUPPORTED_ABI          = 64'd1;
  localparam word_t FC_ADDRESS_FAULT            = 64'd2;
  localparam word_t FC_DMA_TIMEOUT              = 64'd3;
  localparam word_t FC_EVENT_DEPENDENCY_TIMEOUT = 64'd4;
  localparam word_t FC_STREAM_PROTOCOL_ERROR    = 64'd5;
  localparam word_t FC_ENGINE_INTERNAL_FAULT    = 64'd6;
  localparam word_t FC_RESET_DURING_EXECUTION   = 64'd7;
  localparam word_t FC_L2_CAPACITY_FAULT        = 64'd8;
  localparam word_t FC_INVALID_LAYOUT_FAULT     = 64'd9;
  localparam word_t FC_SLOT_PERMISSION_FAULT    = 64'd10;
endpackage
