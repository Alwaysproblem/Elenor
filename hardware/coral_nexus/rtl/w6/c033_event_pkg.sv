// c033_event_pkg.sv — C033 (W6-02) local event facility types.
//
// Shared type records for event_table.sv / wait_record.sv /
// completion_bus.sv / terminal_store.sv and tb_c033. These are LOCAL typed
// records, not a wire packing or redefinition of the frozen C017 ABI: every
// width here is the C017 64-bit transport word container, not a silicon
// width. Encoded enum orders follow the sealed C017 layout
// (protocol/generated/cn_layout.json, cosim/wire.py):
//   event_status  [PENDING, DONE, ERROR, TIMEOUT, RESET]
//   event_scope   [task_private, tile_shared, group_l2, external]
//   task_terminal [DONE, ERROR, TIMEOUT, RESET]
//   phase_direction [input_released, output_ready]
// The C015 fragments (w6_event_facility.contract.yaml,
// w6_terminal_reset.contract.yaml) remain the semantic authority; this file
// adds no capacity, timer or silicon dimension.
`timescale 1ns/1ps
package c033_event_pkg;
  // C017 transport word (64-bit little-endian container). Not a silicon width.
  typedef logic [63:0] word_t;

  // Event terminal status, 3-bit, C017 order. PENDING is the only
  // non-terminal value; terminal states are sticky (EVT-02/EVT-16).
  typedef enum logic [2:0] {
    EV_PENDING = 3'd0,
    EV_DONE    = 3'd1,
    EV_ERROR   = 3'd2,
    EV_TIMEOUT = 3'd3,
    EV_RESET   = 3'd4
  } event_status_t;

  // Task terminal status. DISTINCT 2-bit enum (never interchanged with
  // event_status_t), C017 order [DONE, ERROR, TIMEOUT, RESET]; there is no
  // SUCCESS value (w6_terminal_reset.contract.yaml#task_terminal).
  typedef enum logic [1:0] {
    TS_DONE    = 2'd0,
    TS_ERROR   = 2'd1,
    TS_TIMEOUT = 2'd2,
    TS_RESET   = 2'd3
  } task_status_t;

  // Terminal-store lifecycle command kinds.
  typedef enum logic [1:0] {
    CMD_START  = 2'd0,
    CMD_DRAIN  = 2'd1,
    CMD_RETIRE = 2'd2,
    CMD_FINISH = 2'd3
  } cmd_kind_t;

  // PHASE_SIGNAL direction, sealed C017 order ['input_released',
  // 'output_ready'] (1 bit). INPUT_RELEASED/OUTPUT_READY are PHASE_SIGNAL
  // message semantics, not local event kinds (C015 #event_kinds).
  localparam logic PHASE_INPUT_RELEASED = 1'b0;
  localparam logic PHASE_OUTPUT_READY   = 1'b1;

  // Event scope encoding, sealed C017 order. Scope is part of the full match
  // key; a name string never matches (EVT-16) and "tile_shared" is not a
  // string identity.
  localparam logic [1:0] EV_SCOPE_TASK_PRIVATE = 2'd0;
  localparam logic [1:0] EV_SCOPE_TILE_SHARED  = 2'd1;
  localparam logic [1:0] EV_SCOPE_GROUP_L2     = 2'd2;
  localparam logic [1:0] EV_SCOPE_EXTERNAL     = 2'd3;

  // Local waitable event kinds, C015 order [compute_done, transfer_done,
  // store_done, region_done]; kind is fixed at registration.
  localparam logic [1:0] EV_KIND_COMPUTE_DONE = 2'd0;
  localparam logic [1:0] EV_KIND_TRANSFER_DONE = 2'd1;
  localparam logic [1:0] EV_KIND_STORE_DONE    = 2'd2;
  localparam logic [1:0] EV_KIND_REGION_DONE   = 2'd3;

  // Physical context address: which Tile, which Context slot, which
  // generation of that slot. All identities are FULL words (upper bits
  // included in every compare).
  typedef struct packed {
    word_t tile_id;
    word_t ctx_slot;
    word_t ctx_generation;
  } context_key_t;

  // Full local event key: the physical name plus the event-side generation
  // domains (event_generation and seq are distinct from ctx_generation,
  // C015 #identity.generation_domains_distinct) and the 2-bit scope.
  typedef struct packed {
    context_key_t ctx;
    word_t local_event_id;
    word_t event_generation;
    word_t seq;
    logic [1:0] scope;
  } event_key_t;

  // Task identity carried by RETIRE/RETIRE_READY/TASK_COMPLETION/
  // PHASE_SIGNAL (C017 word fields, same order).
  typedef struct packed {
    word_t launch_generation;
    word_t root_launch_generation;
    word_t dispatch_ordinal;
    word_t task_id;
    word_t context_handle;
  } task_key_t;

  // Full owner = context key + task key. The terminal store validates the
  // FULL owner on every command/abort/phase input.
  typedef struct packed {
    context_key_t ctx;
    task_key_t task_key;
  } owner_t;

  // One producer completion arriving on the completion bus.
  typedef struct packed {
    event_key_t key;
    event_status_t status;
    word_t producer_id;
    word_t error_code;
  } completion_t;

  // Committed-state view of one event slot (query output; absent => all
  // zero including live).
  typedef struct packed {
    logic live;
    event_status_t status;
    logic [1:0] kind;
    word_t producer_id;
    word_t error_code;
    word_t consumer_refs;
    word_t future_uses;
  } event_view_t;

  // PMU counters. The first six are the C015 aggregates (check order per
  // w6_event_facility.contract.yaml#signal: unknown_event_notify,
  // wrong_owner, stale_sequence, duplicate_signal, plus capacity_reject and
  // protocol_error); wrong_context/wrong_scope/wrong_sequence/
  // wrong_event_generation are per-field detail buckets, NOT replacements
  // for the aggregates.
  typedef struct packed {
    word_t unknown_event_notify;
    word_t wrong_owner;
    word_t stale_sequence;
    word_t duplicate_signal;
    word_t capacity_reject;
    word_t protocol_error;
    word_t wrong_context;
    word_t wrong_scope;
    word_t wrong_sequence;
    word_t wrong_event_generation;
  } counts_t;

  // Sticky per-counter seen flags (C015 saturation.sticky_bit_per_counter,
  // wrap forbidden). Field names mirror counts_t exactly; a flag is 1 once
  // its counter has ever been nonzero (derived, never a second register).
  typedef struct packed {
    logic unknown_event_notify;
    logic wrong_owner;
    logic stale_sequence;
    logic duplicate_signal;
    logic capacity_reject;
    logic protocol_error;
    logic wrong_context;
    logic wrong_scope;
    logic wrong_sequence;
    logic wrong_event_generation;
  } counter_flags_t;

  // One PHASE_SIGNAL message emitted for one accepted phase instruction
  // (1:1, duplicates included; message_seq is the link-side sequencer's
  // value, not allocated here).
  typedef struct packed {
    task_key_t task_key;
    logic direction;
    word_t message_seq;
  } phase_msg_t;

  // RETIRE_READY (is_completion=0) or TASK_COMPLETION (is_completion=1)
  // message. RETIRE_READY is not a task success: it carries the drain
  // evidence sequence and neutral status/ref fields; TASK_COMPLETION carries
  // the terminal status and first-fault reference.
  typedef struct packed {
    logic is_completion;
    owner_t owner;
    task_status_t status;
    word_t drain_evidence_seq;
    word_t first_fault_ref;
  } terminal_msg_t;
endpackage
