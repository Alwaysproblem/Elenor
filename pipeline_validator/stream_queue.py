"""ELENOR Stream Queue — hardware producer-consumer contract.

Implements the contract in design/elenor_stream_queue/ELENOR_Stream_Queue_Design.md
and Architecture doc section 16.2:

  token lifecycle:  producer acquire credit -> fill payload -> push token
                    -> consumer pop -> consumer release.
  credit invariant: credit_available + tokens_in_fifo + tokens_popped_not_released == depth
  backpressure:     queue full  -> producer stall (stream_queue_full)
                    queue empty -> consumer stall (stream_queue_empty)
  EOS:              per-producer EOS bitmap; all-EOS policy closes the role.
  error:            error token carries fault record index, propagates.
  reset/drain:      credit reconciled to depth, occupancy cleared.

The queue is a cycle-accurate model: each call advances one cycle and
accumulates PMU counters (occupancy-weighted cycles, credit-empty cycles,
queue-empty cycles).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag

from .pmu import PMUCounter, StallReason


class TokenFlags(IntFlag):
    VALID = 1 << 0
    EOS = 1 << 1
    ERROR = 1 << 2
    FENCE = 1 << 3


class QueueKind(IntEnum):
    SPSC = 0
    MPSC = 1
    BROADCAST = 2


class EOSPolicy(IntEnum):
    SINGLE_PRODUCER = 0
    ALL_PRODUCERS = 1
    PER_PRODUCER = 2


@dataclass
class StreamToken:
    """elenor_stream_token_v0_t (Stream Queue design 4.1)."""
    token_id: int
    payload_addr: int = 0
    payload_bytes: int = 0
    flags: TokenFlags = TokenFlags.VALID
    producer_id: int = 0
    sequence_id: int = 0
    fault_record_index: int = 0
    user_metadata: int = 0
    # simulation bookkeeping
    pushed_cycle: int = 0
    popped_cycle: int = 0
    released: bool = False

    @property
    def is_eos(self) -> bool:
        return bool(self.flags & TokenFlags.EOS)

    @property
    def is_error(self) -> bool:
        return bool(self.flags & TokenFlags.ERROR)

    @property
    def is_valid(self) -> bool:
        return bool(self.flags & TokenFlags.VALID)


@dataclass
class StreamQueue:
    """One producer-consumer stream queue.

    `producers`/`consumers` are the set of producer/consumer ids (from masks).
    The model tracks per-producer `sequence_id` monotonicity, the all-EOS
    bitmap, and the credit invariant.
    """

    queue_id: int
    depth: int
    producers: frozenset
    consumers: frozenset
    kind: QueueKind = QueueKind.SPSC
    eos_policy: EOSPolicy = EOSPolicy.ALL_PRODUCERS
    pmu: PMUCounter = field(default_factory=PMUCounter)

    # state
    _fifo: deque = field(default_factory=deque)
    _credit_available: int = 0
    _credit_leased: int = 0  # acquired but not yet pushed (in-flight credit)
    _producer_eos_bitmap: int = 0
    _producer_seq: dict = field(
        default_factory=dict)  # producer_id -> next seq
    _popped_unreleased: int = 0
    _first_fault_index: int = -1
    _faulted: bool = False
    _next_token_id: int = 0
    # last-cycle stall tracking for unique attribution
    _last_producer_stall_cycle: int = -1
    _last_consumer_stall_cycle: int = -1

    # ---- lifecycle -------------------------------------------------------

    def init(self) -> None:
        self._credit_available = self.depth
        self._credit_leased = 0
        self._fifo.clear()
        self._producer_eos_bitmap = 0
        self._popped_unreleased = 0
        self._first_fault_index = -1
        self._faulted = False
        self._producer_seq = dict.fromkeys(self.producers, 0)

    @property
    def occupancy(self) -> int:
        return len(self._fifo)

    @property
    def is_empty(self) -> bool:
        return len(self._fifo) == 0

    @property
    def is_full(self) -> bool:
        return self._credit_available == 0

    @property
    def all_eos_seen(self) -> bool:
        """True when the EOS policy is satisfied (role done for consumers)."""
        if self.eos_policy == EOSPolicy.SINGLE_PRODUCER:
            return self._producer_eos_bitmap != 0
        if self.eos_policy == EOSPolicy.ALL_PRODUCERS:
            return self._producer_eos_bitmap == self._all_producer_mask()
        return self._producer_eos_bitmap != 0  # per-producer

    def _all_producer_mask(self) -> int:
        m = 0
        for p in self.producers:
            m |= (1 << p)
        return m

    def _valid_fifo_count(self) -> int:
        """Number of non-EOS tokens in the FIFO (EOS carries no payload credit)."""
        return sum(1 for t in self._fifo if not t.is_eos)

    def credit_invariant_holds(self) -> bool:
        """credit_available + credit_leased + valid_tokens_in_fifo
        + popped_not_released == depth.

        EOS tokens carry no payload and do not consume credit, so they are
        excluded from the FIFO count.  credit_leased counts slots acquired
        by a producer but not yet pushed, holding the invariant between
        acquire and push.
        """
        return (self._credit_available + self._credit_leased +
                self._valid_fifo_count() +
                self._popped_unreleased) == self.depth

    # ---- producer side ---------------------------------------------------

    def acquire(self, cycle: int) -> bool:
        """Try to acquire one credit slot.  Returns True on success.

        On failure (queue full) the producer stalls this cycle and the PMU
        records `stream_credit_empty_or_full`.
        """
        if self._credit_available > 0 and not self._faulted:
            self._credit_available -= 1
            self._credit_leased += 1
            return True
        self._last_producer_stall_cycle = cycle
        self.pmu.add(StallReason.STREAM_CREDIT, 1)
        self.pmu.add_cycle("queue_full", 1)
        return False

    def push(self, token: StreamToken, cycle: int) -> bool:
        """Push a token.  Assumes credit already acquired for valid tokens.

        EOS tokens carry no payload and do not consume FIFO credit, so they
        are excluded from the credit invariant.  Error tokens bypass to the
        fault fabric and do not consume FIFO credit either.
        """
        if token.is_error:
            if self._first_fault_index < 0:
                self._first_fault_index = token.fault_record_index
            self._faulted = True
            return True
        if token.is_eos:
            # EOS carries no payload and consumes no credit slot: mark the
            # producer as end-of-stream, append exactly one EOS token, and
            # return before the valid-token path.  Credit leased by an
            # explicit acquire() must NOT be decremented for an EOS push.
            self._producer_eos_bitmap |= (1 << token.producer_id)
            token.pushed_cycle = cycle
            self._fifo.append(token)
            return True
        # valid token (credit already acquired by the producer)
        token.sequence_id = self._producer_seq.get(token.producer_id, 0)
        self._producer_seq[token.producer_id] = token.sequence_id + 1
        token.pushed_cycle = cycle
        if self._credit_leased > 0:
            self._credit_leased -= 1
        self._fifo.append(token)
        return True

    def push_eos(self, producer_id: int, cycle: int) -> None:
        tok = StreamToken(
            token_id=self._next_token_id,
            flags=TokenFlags.EOS,
            producer_id=producer_id,
            pushed_cycle=cycle,
        )
        self._next_token_id += 1
        # EOS does not consume credit in this model; just push.
        self.push(tok, cycle)

    # ---- consumer side ----------------------------------------------------

    def pop(self, cycle: int) -> StreamToken | None:
        """Pop the head token.  Returns None (consumer stalls) if empty."""
        if self._fifo:
            tok = self._fifo.popleft()
            tok.popped_cycle = cycle
            if tok.is_valid:
                self._popped_unreleased += 1
            return tok
        self._last_consumer_stall_cycle = cycle
        self.pmu.add(StallReason.STREAM_CREDIT, 1)
        self.pmu.add_cycle("queue_empty", 1)
        return None

    def release(self, token: StreamToken, cycle: int) -> None:
        """Release a popped token, returning its credit."""
        if token.released:
            # double release -> fault
            self.pmu.add_cycle("credit_fault", 1)
            return
        token.released = True
        if token.is_valid:
            self._popped_unreleased -= 1
            self._credit_available += 1
        # EOS and error tokens did not consume FIFO credit; nothing to return.

    # ---- per-cycle accounting -------------------------------------------

    def tick(self, cycle: int) -> None:
        """Advance one cycle: update occupancy-weighted PMU counters."""
        self.pmu.add_cycle("occupancy", self.occupancy)
        if self.is_full:
            self.pmu.add_cycle("credit_full", 1)
        if self.is_empty:
            self.pmu.add_cycle("credit_empty", 1)

    # ---- reset / drain ---------------------------------------------------

    def reset(self) -> None:
        self._fifo.clear()
        self._credit_available = self.depth
        self._credit_leased = 0
        self._popped_unreleased = 0
        self._producer_eos_bitmap = 0
        self._first_fault_index = -1
        self._faulted = False
        self._producer_seq = dict.fromkeys(self.producers, 0)

    def snapshot(self) -> dict:
        return {
            "queue_id": self.queue_id,
            "depth": self.depth,
            "occupancy": self.occupancy,
            "credit_available": self._credit_available,
            "popped_unreleased": self._popped_unreleased,
            "producer_eos_bitmap": self._producer_eos_bitmap,
            "all_eos_seen": self.all_eos_seen,
            "faulted": self._faulted,
            "first_fault_index": self._first_fault_index,
            "credit_invariant_holds": self.credit_invariant_holds(),
        }
