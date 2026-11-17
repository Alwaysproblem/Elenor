"""C031 oracle checks; these do not execute or emulate the RTL LSU.

Native RTL/real MemoryService integration lives in test_w4_c031.py. Here the
existing bank-stage and external-service contracts supply independent oracles.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_validator.cosim.memory_service import MemoryService, MemoryServiceError, MemRequest
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.hbm_region import HBMRegion
from pipeline_validator.memory.transfer import (
  StageRequest,
  StageResult,
  StageWait,
  StageWaitReason,
  TransferStage,
)


def test_blocked_compound_access_leaves_other_bank_available() -> None:
  bank = TransferStage("l1", StageWaitReason.L1_BANK, 1, 32, 16, burst_bytes=32)
  first = bank.try_issue("ctx0:seq7", [StageRequest("3", 256)], 0)
  assert isinstance(first, StageResult)
  blocked = bank.try_issue("ctx1:seq7", [StageRequest("10", 32), StageRequest("3", 32)], 1)
  assert isinstance(blocked, StageWait) and blocked.reason is StageWaitReason.L1_BANK
  peer = bank.try_issue("ctx1:seq8", [StageRequest("10", 32)], 1)
  assert isinstance(peer, StageResult)
  assert peer.resources == (10,) and peer.completion_cycle < first.completion_cycle
  bank.release_outstanding("ctx1:seq8")
  assert isinstance(bank.try_issue("ctx1:seq9", [StageRequest("3", 32)], 2), StageWait)
  bank.release_outstanding("ctx0:seq7")
  resumed = bank.try_issue("ctx1:seq9", [StageRequest("10", 32), StageRequest("3", 32)], 2)
  assert isinstance(resumed, StageResult) and resumed.resources == (10, 3)


def service_and_request() -> tuple[MemoryService, MemRequest]:
  store = ByteStore()
  hbm = HBMRegion(byte_store=store)
  hbm.bind_external(GlobalBinding("input", 0x100000, 256, "r"))
  store.seed_hbm(0x100000, bytes(range(256)))
  service = MemoryService(hbm, store)
  service.register_binding("input", 0x100000, 256, "r", generation=3)
  request = MemRequest(
    tile_id=0, ctx_slot=0, ctx_generation=1, txn_id=10, slice_id=0,
    slice_count=1, arena_alloc_gen=9, binding_name="input", binding_generation=3,
    binding_offset=0, scope_epoch=7, mem_op=0, length=64,
  )
  return service, request


def test_peer_refill_ack_cannot_release_other_context_obligation() -> None:
  service, a = service_and_request()
  b = replace(a, txn_id=11, ctx_slot=1, binding_offset=64)
  assert service.submit_mem_req(a).data == bytes(range(64))
  assert service.submit_mem_req(b).data == bytes(range(64, 128))
  service.terminal_ack(11, 0, 1, "COMPLETED")
  before = service.ledger_snapshot()
  assert before["terminal_refs"] == [(10, 0)]
  with pytest.raises(MemoryServiceError, match="TERMINAL_ACK_UNKNOWN"):
    service.terminal_ack(11, 0, 1, "COMPLETED")
  assert service.ledger_snapshot() == before
  with pytest.raises(MemoryServiceError, match="TERMINAL_REFS_OUTSTANDING"):
    service.assert_closed()
  service.terminal_ack(10, 0, 1, "ISOLATED")
  service.assert_closed()


def test_old_binding_generation_rejects_before_refill_or_ledger_mutation() -> None:
  service, req = service_and_request()
  before = service.ledger_snapshot()
  with pytest.raises(MemoryServiceError, match="stale binding generation"):
    service.submit_mem_req(replace(req, binding_generation=2))
  assert service.ledger_snapshot() == before
  assert service.submit_mem_req(req).data == bytes(range(64))
  service.terminal_ack(req.txn_id, 0, 1, "COMPLETED")
  service.assert_closed()
