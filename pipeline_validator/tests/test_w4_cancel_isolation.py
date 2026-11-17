"""C032: cancellation settles only on real completion or isolation confirmation.

These tests pin the Python side of the MFE/transfer settlement contract
(MEM-09/MEM-12/MEM-13, W4 plan §5.4):

* CANCEL_REQUESTED is never a terminal by itself;
* a cancelled transaction whose accepted write leg then completes is COMPLETED
  (the effect happened) - never ISOLATED/CANCELLED;
* a read-only accepted leg may settle as CANCELLED when it really finishes;
* an isolation failure poisons new work until an explicit quiescent clear;
* unknown/duplicate isolation confirmations are refused.
"""

from __future__ import annotations

import pytest

from pipeline_validator.memory.allocator import MemoryInvariantError


def _task_owner(tile: int = 0, ctx: int = 0, task: int = 0, buf: str = "l1"):
  from pipeline_validator.memory.allocator import TaskBufferOwner

  return TaskBufferOwner("ctx", 0, f"role:{buf}", task, tile, ctx, buf)


def _transaction(transaction_id, op, *, src=None, dst=None, owner=None, bytes_total=128):
  from pipeline_validator.memory.transfer import MemoryTransaction

  return MemoryTransaction(
    transaction_id=transaction_id,
    op=op,
    issuer=_task_owner() if owner is None else owner,
    src=src,
    dst=dst,
    bytes_total=bytes_total,
    completion_event=transaction_id,
    tile_id=0,
  )


def _view(space: str, owner=None):
  from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
  from pipeline_validator.memory.transfer import ResolvedMemoryView

  actual_owner = _task_owner() if owner is None else owner
  segments = (BankSegment(0, 0, 64), BankSegment(1, 64, 64))
  handle = AllocationHandle(
    allocation_id=f"{space}:0:1",
    memory_space=space,
    owner=actual_owner,
    base_address=0,
    size_bytes=128,
    alignment=1,
    bank_segments=segments,
    generation=0,
    allocate_cycle=0,
  )
  return ResolvedMemoryView(
    handle=handle, offset_bytes=0, size_bytes=128, address=0, segments=segments, permissions="rw"
  )


def _manager():
  from pipeline_validator.config import HardwareConfig
  from pipeline_validator.memory.noc import NoCRouter
  from pipeline_validator.memory.transfer import TransferManager

  config = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=4, hbm_outstanding_limit=8)
  noc = NoCRouter(vc_depth=config.noc_vc_depth, router_latency_cycles=config.noc_router_latency_cycles)
  manager = TransferManager(config, full_memory=True, noc=noc)
  return config, noc, manager


def _drive_until_write_leg(manager, noc, txn, limit: int = 400) -> None:
  """Advance the real stage machinery until an external write leg is accepted."""
  from pipeline_validator.memory.transfer import TransferLegKind, TransferStatus

  write_legs = (TransferLegKind.GLOBAL_DMA, TransferLegKind.HBM_WRITE)
  for cycle in range(limit):
    traversed = noc.step(cycle)
    manager.note_traversed(traversed, cycle)
    manager.step(cycle)
    if txn.status is not TransferStatus.RUNNING:
      break
    if (
      txn.current_leg < len(txn.legs)
      and txn.legs[txn.current_leg].kind in write_legs
      and txn.leg_start_cycle >= 0
    ):
      return
  raise AssertionError("write leg never accepted")


def test_write_after_cancel_completes_with_the_real_effect() -> None:
  from pipeline_validator.memory.transfer import TransferOp, TransferStatus

  _config, noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction(
    "scatter", TransferOp.SCATTER_WRITE, src=_view("l1", owner), owner=owner, bytes_total=128
  )
  manager.submit(txn, cycle=0)
  _drive_until_write_leg(manager, noc, txn)

  assert manager.cancel_owner(owner, cycle=txn.leg_start_cycle) is False
  assert manager.status("scatter") is TransferStatus.CANCEL_REQUESTED
  assert txn.destination_committed_cycle < 0

  for cycle in range(txn.leg_start_cycle + 1, txn.leg_start_cycle + 400):
    traversed = noc.step(cycle)
    manager.note_traversed(traversed, cycle)
    manager.step(cycle)
    if manager.status("scatter") in (TransferStatus.DONE, TransferStatus.CANCELLED):
      break

  # The write was already accepted, so the only truthful terminal is COMPLETED.
  assert manager.status("scatter") is TransferStatus.DONE
  assert txn.destination_committed_cycle >= 0
  assert manager.inflight_count == 0


def test_read_only_cancel_has_exactly_one_terminal() -> None:
  from pipeline_validator.memory.transfer import TransferOp, TransferStatus

  _config, _noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction("index", TransferOp.INDEX_READ, dst=_view("l1", owner), owner=owner)
  manager.submit(txn, cycle=0)
  manager.step(cycle=0)
  completion = txn.leg_completion_cycle
  assert completion > 0

  assert manager.cancel_owner(owner, cycle=0) is False
  assert manager.status("index") is TransferStatus.CANCEL_REQUESTED
  manager.step(cycle=completion - 1)
  assert manager.status("index") is TransferStatus.CANCEL_REQUESTED
  manager.step(cycle=completion)
  assert manager.status("index") is TransferStatus.CANCELLED

  with pytest.raises(MemoryInvariantError):
    manager.confirm_isolation("index", cycle=completion + 1)
  with pytest.raises(MemoryInvariantError):
    manager.confirm_isolation("missing", cycle=completion + 1)


def test_isolation_failure_poisons_new_work_until_quiescent_clear() -> None:
  from pipeline_validator.memory.transfer import TransferOp, TransferStatus

  _config, _noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction("dest", TransferOp.GATHER_DEST_WRITE, dst=_view("l1", owner), owner=owner)
  manager.submit(txn, cycle=0)
  manager.step(cycle=0)
  assert manager.cancel_owner(owner, cycle=0) is False
  assert manager.status("dest") is TransferStatus.CANCEL_REQUESTED

  manager.isolation_failed("dest", cycle=1, reason="peer cannot prove quiescence")
  assert manager.status("dest") is TransferStatus.CANCEL_REQUESTED, "an isolation failure is not a terminal"
  assert manager.poisoned_reason

  with pytest.raises(MemoryInvariantError, match="poison"):
    manager.submit(_transaction("new", TransferOp.GATHER_L1_LOOKUP, owner=owner), cycle=2)
  with pytest.raises(MemoryInvariantError, match="poison"):
    manager.begin_run(2)
  with pytest.raises(MemoryInvariantError, match="poison"):
    manager.clear_poison()

  manager.confirm_isolation("dest", cycle=2)
  assert manager.status("dest") is TransferStatus.CANCELLED
  assert manager.inflight_count == 0
  manager.clear_poison()
  assert manager.poisoned_reason is None
  manager.submit(_transaction("after", TransferOp.GATHER_L1_LOOKUP, owner=owner), cycle=3)
  manager.step(cycle=3)


def test_completed_terminal_ack_binds_a_cancelled_transaction_to_completion() -> None:
  from pipeline_validator.memory.transfer import TransferOp, TransferStatus

  _config, _noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction("acked", TransferOp.INDEX_READ, dst=_view("l1", owner), owner=owner)
  manager.submit(txn, cycle=0)
  manager.step(cycle=0)
  completion = txn.leg_completion_cycle
  assert manager.cancel_owner(owner, cycle=0) is False
  assert manager.status("acked") is TransferStatus.CANCEL_REQUESTED

  # The boundary really finished the work: that terminal is authoritative.
  manager.terminal_ack("acked", "COMPLETED", cycle=0)
  manager.step(cycle=completion)
  assert manager.status("acked") is TransferStatus.DONE
  with pytest.raises(MemoryInvariantError):
    manager.terminal_ack("acked", "ISOLATED", cycle=completion + 1)


def test_faulted_terminal_ack_is_never_isolation() -> None:
  from pipeline_validator.memory.transfer import TransferOp, TransferStatus

  _config, _noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction("faulted", TransferOp.GATHER_L1_LOOKUP, owner=owner)
  manager.submit(txn, cycle=0)
  manager.step(cycle=0)
  manager.cancel_owner(owner, cycle=0)
  assert manager.status("faulted") is TransferStatus.CANCEL_REQUESTED
  assert manager.terminal_ack("faulted", "FAULTED", cycle=1) is TransferStatus.FAULTED
  assert manager.status("faulted") is TransferStatus.FAULTED
  with pytest.raises(MemoryInvariantError):
    manager.terminal_ack("faulted", "ISOLATED", cycle=2)


def test_isolation_confirmation_requires_a_cancelled_transaction() -> None:
  from pipeline_validator.memory.transfer import TransferOp

  _config, _noc, manager = _manager()
  owner = _task_owner()
  txn = _transaction("running", TransferOp.GATHER_L1_LOOKUP, owner=owner)
  manager.submit(txn, cycle=0)
  with pytest.raises(MemoryInvariantError, match="CANCEL_REQUESTED"):
    manager.confirm_isolation("running", cycle=0)
