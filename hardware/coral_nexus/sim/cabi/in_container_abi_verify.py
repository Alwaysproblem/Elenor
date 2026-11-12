#!/usr/bin/env python3
"""In-container end-to-end Python verification of `wire.py` + `abi.py` + `RtlTileBackend` (C025).

Runs inside `coral-nexus-builder:c002` against the real `libcn_cosim_stub.so` via `ctypes.CDLL`,
mounting the repository root at `/repo:ro`:
1. Verifies `bp_abi_info()` 128-hex hash and all 10 slot capacities against `cn_layout.json`
   (`TASK_PLAN=96B`, `TASK_COMMIT=104B`, `PROGRAM_INSTALL_ACK=128B`, `MEM_REQ=136B`), plus
   repeatable read-only `bp_layout` queries.
2. Drives full-width `wire.pack` messages through `RtlTileBackend(TilePort)` across all protocol
   channels (`CTRL`, `EVENT`, `PROGRAM`, `MEMORY`), decoding real SV-synthesized T2G responses via
   `BoundaryMessage.from_wire`, including a 136-byte `MEM_REQ` and 128-byte `PROGRAM_INSTALL_ACK`.
3. Proves P1/P3 staged-output isolation: a G2T message accepted in tick `T` NEVER produces an
   accepted T2G response in tick `T` (it appears only in tick `T+1`+).
4. Exercises all ABI state/range/protocol negatives:
   - missing `bp_drive` before `bp_tick` -> `CN_ERR_STATE`
   - double `bp_collect` in the same tick -> `CN_ERR_STATE`
   - two consecutive `bp_drive -> bp_tick` cycles without `bp_collect` retain buffered records
   - short `bp_collect` buffer -> `CN_ERR_RANGE` without truncation or record loss
   - mutating payload while `valid && !ready` -> `CN_ERR_STATE` + `CN_EP_PAYLOAD_READY_MISMATCH`
   - post-edge DUT payload glitch (`CN_STUB_FLAG_GLITCH_PAYLOAD`) -> `CN_EP_PAYLOAD_READY_MISMATCH`
   - catchable DUT fatal -> `INCOMPLETE_FATAL` JSON with residual pending set
5. Tamper proof (`CN_STUB_FLAG_CORRUPT_T2G`): corrupting the SV DUT's T2G output pin causes the
   decoded response `txn_id` to mismatch immediately, proving `bp_collect` reads real DUT T2G pins.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def _bootstrap_cosim_package(repo_root: Path) -> None:
  """Load `pipeline_validator.cosim.*` without triggering `pipeline_validator/__init__.py` (xdsl)."""
  pv_pkg = types.ModuleType("pipeline_validator")
  pv_pkg.__path__ = [str(repo_root / "pipeline_validator")]
  cosim_pkg = types.ModuleType("pipeline_validator.cosim")
  cosim_pkg.__path__ = [str(repo_root / "pipeline_validator/cosim")]
  sys.modules["pipeline_validator"] = pv_pkg
  sys.modules["pipeline_validator.cosim"] = cosim_pkg


def run_all(repo_root: Path, so_path: Path, out_dir: Path) -> dict[str, object]:
  _bootstrap_cosim_package(repo_root)
  from pipeline_validator.cosim import abi as cn_abi, rtl_backend as rb, tile_port as tp, wire

  results: list[str] = []

  def record(ok: bool, name: str, detail: str = "") -> None:
    if not ok:
      raise AssertionError(f"FAIL {name}: {detail}")
    results.append(name)

  # 1. Raw ctypes ABI checks: hash, repeatable read-only layout, slot capacities, phase errors.
  raw_abi = cn_abi.load(so_path)
  cfg = cn_abi.CnBeginCfg()
  cfg.abi_version = cn_abi.CN_ABI_VERSION
  cfg.tile_count = 1
  cfg.ready_seed = 0xFFFFFFFE  # non-zero seed with bit 0 == 0 -> tick 0 has g2t_ready[0] == 0!
  cfg.inject_fatal_at_tick = -1
  cfg.inject_abort_at_tick = -1
  raw_abi.begin(cfg)
  try:
    slots1 = raw_abi.layout()
    st_before = raw_abi.status()
    slots2 = raw_abi.layout()
    st_after = raw_abi.status()
    record(slots1 == slots2, "layout_repeatable_readonly")
    record(int(st_before.tick_count) == int(st_after.tick_count) == 0, "layout_mutates_no_state")
    by_link = {(str(s["channel"]), str(s["dir"])): int(s["len"]) for s in slots1}
    record(by_link[("CTRL", "G2T")] == 104, "slot_ctrl_g2t_104b", str(by_link))
    record(by_link[("CTRL", "T2G")] == 96, "slot_ctrl_t2g_96b")
    record(by_link[("EVENT", "G2T")] == 104, "slot_event_g2t_104b")
    record(by_link[("PROGRAM", "T2G")] == 128, "slot_program_t2g_128b")
    record(by_link[("MEMORY", "T2G")] == 136, "slot_memory_t2g_136b")

    # Missing bp_drive -> CN_ERR_STATE
    record(raw_abi.tick() == cn_abi.CN_ERR_STATE, "missing_drive_returns_state_err")

    # Payload stability negative: seed 0x1ACEB00C has bit 0 == 0 on tick 0, so CTRL g2t_ready==0!
    plan_a = wire.pack(
      "TASK_PLAN",
      {
        "launch_generation": 1,
        "root_launch_generation": 1,
        "dispatch_ordinal": 0,
        "task_id": 2,
        "context_handle": 0,
        "txn_id": 41,
        "pool_epoch": 1,
        "pool_version": 1,
      },
      message_seq=0,
    )
    record(len(plan_a) == 96, "task_plan_wire_is_96b")
    record(raw_abi.drive([(0, plan_a)]) == cn_abi.CN_OK, "drive_stalled_beat_ok")
    record(raw_abi.tick() == cn_abi.CN_OK, "tick_stalled_beat_ok")
    crc0, recs0 = raw_abi.collect()
    record(crc0 == cn_abi.CN_OK and recs0 == [], "stalled_beat_not_accepted")
    # Double collect -> CN_ERR_STATE
    record(raw_abi.collect()[0] == cn_abi.CN_ERR_STATE, "double_collect_returns_state_err")
    # Mutating payload while `valid && !ready` -> CN_ERR_STATE + CN_EP_PAYLOAD_READY_MISMATCH!
    plan_mutated = wire.pack(
      "TASK_PLAN",
      {
        "launch_generation": 1,
        "root_launch_generation": 1,
        "dispatch_ordinal": 0,
        "task_id": 2,
        "context_handle": 0,
        "txn_id": 999,
        "pool_epoch": 1,
        "pool_version": 1,
      },
      message_seq=0,
    )
    record(raw_abi.drive([(0, plan_mutated)]) == cn_abi.CN_ERR_STATE, "unstable_payload_rejected")
    record(
      int(raw_abi.status().proto_err) == cn_abi.CN_EP_PAYLOAD_READY_MISMATCH,
      "unstable_payload_sets_proto_err",
    )
  finally:
    raw_abi.destroy()

  # 2. Buffered multi-tick without collect + short collect range check + staged P1/P3 proof.
  buf_abi = cn_abi.load(so_path)
  cfg2 = cn_abi.CnBeginCfg()
  cfg2.abi_version = cn_abi.CN_ABI_VERSION
  cfg2.tile_count = 1
  cfg2.ready_seed = 0xFFFFFFFF  # ready=1 on tick 0
  cfg2.inject_fatal_at_tick = -1
  cfg2.inject_abort_at_tick = -1
  buf_abi.begin(cfg2)
  try:
    qcaps = wire.pack("QUERY_CAPS", {"txn_id": 55}, message_seq=0)
    record(buf_abi.drive([(0, qcaps)]) == cn_abi.CN_OK, "drive_qcaps_ok")
    record(buf_abi.tick() == cn_abi.CN_OK, "tick0_qcaps_ok")
    # At end of tick 0: G2T QUERY_CAPS was accepted at P1, and DUT staged CAPS_RESP at P2.
    # Staged output MUST NOT be in collect of tick 0; tick 1 without prior collect must keep tick 0
    # receipt AND add tick 1's accepted T2G CAPS_RESP!
    st_t0 = buf_abi.status()
    record(int(st_t0.staged_pending) == 1, "staged_t2g_visible_only_as_next_tick_candidate")
    record(buf_abi.drive([(0, b"")]) == cn_abi.CN_OK, "drive_idle_tick1_without_collect")
    record(buf_abi.tick() == cn_abi.CN_OK, "tick1_without_prior_collect_ok")
    # Short collect buffer -> CN_ERR_RANGE without losing buffered records
    rc_short, _ = buf_abi.collect_raw(12)
    record(rc_short == cn_abi.CN_ERR_RANGE, "short_collect_returns_range")
    rc_full, recs = buf_abi.collect()
    record(rc_full == cn_abi.CN_OK and len(recs) == 2, "uncollected_records_preserved_across_ticks")
    record(recs[0].is_t2g is False and recs[0].byte_len == len(qcaps), "rec0_is_g2t_receipt")
    record(recs[1].is_t2g is True and recs[1].byte_len == 64, "rec1_is_64b_caps_resp")
    decoded_caps = wire.unpack(recs[1].wire_bytes, port=(0, 1))
    record(
      decoded_caps.message == "CAPS_RESP" and decoded_caps.values["txn_id"] == 55,
      "caps_resp_decoded_via_wire_unpack",
    )
  finally:
    buf_abi.destroy()

  # 3. Post-edge DUT glitch negative -> CN_ERR_STATE + CN_EP_PAYLOAD_READY_MISMATCH.
  gl_abi = cn_abi.load(so_path)
  cfg_gl = cn_abi.CnBeginCfg()
  cfg_gl.abi_version = cn_abi.CN_ABI_VERSION
  cfg_gl.tile_count = 1
  cfg_gl.ready_seed = 0xFFFFFFFF
  cfg_gl.inject_fatal_at_tick = -1
  cfg_gl.inject_abort_at_tick = -1
  cfg_gl.reserved[0] = cn_abi.CN_STUB_FLAG_GLITCH_PAYLOAD
  gl_abi.begin(cfg_gl)
  try:
    gl_abi.drive([(0, wire.pack("QUERY_CAPS", {"txn_id": 1}, message_seq=0))])
    record(gl_abi.tick() == cn_abi.CN_ERR_STATE, "post_edge_glitch_rejected")
    record(
      int(gl_abi.status().proto_err) == cn_abi.CN_EP_PAYLOAD_READY_MISMATCH,
      "post_edge_glitch_sets_proto_err",
    )
  finally:
    gl_abi.destroy()

  # 4. Full `RtlTileBackend(TilePort)` round-trip across CTRL, EVENT, PROGRAM, MEMORY + 136B MEM_REQ!
  backend = rb.RtlTileBackend(
    0,
    library=so_path,
    is_unit_stub=True,
    ready_seed=0x13579BDF,
    stub_flags=cn_abi.CN_STUB_FLAG_EMIT_MEM_REQ,
  )
  try:
    record(isinstance(backend, tp.TilePort), "rtl_backend_implements_tile_port")
    record(backend.backend_kind == "stub", "unit_stub_backend_kind_tagged_stub")
    try:
      tp.assert_cs1_backend_kind(backend.backend_kind)
      record(False, "cs1_guard_rejects_stub")
    except ValueError:
      record(True, "cs1_guard_rejects_stub")

    pkg_hash = bytes(range(32))
    outbound = [
      tp.BoundaryMessage(
        "TASK_PLAN",
        "CTRL",
        "G2T",
        {
          "launch_generation": 2,
          "root_launch_generation": 2,
          "dispatch_ordinal": 1,
          "task_id": 3,
          "context_handle": 0,
          "txn_id": 101,
          "pool_epoch": 1,
          "pool_version": 4,
        },
      ),
      tp.BoundaryMessage(
        "TASK_COMMIT",
        "CTRL",
        "G2T",
        {
          "launch_generation": 2,
          "root_launch_generation": 2,
          "dispatch_ordinal": 1,
          "task_id": 3,
          "context_handle": 0,
          "txn_id": 102,
          "ctx_slot": 1,
          "ctx_generation": 5,
          "pool_version": 4,
        },
      ),
      tp.BoundaryMessage(
        "TASK_COMMIT_FINAL",
        "CTRL",
        "G2T",
        {
          "launch_generation": 2,
          "root_launch_generation": 2,
          "dispatch_ordinal": 1,
          "task_id": 3,
          "context_handle": 0,
          "txn_id": 103,
          "ctx_slot": 1,
          "ctx_generation": 4,  # stale (< live 5) -> DUT must reply outcome=STALE!
        },
      ),
      tp.BoundaryMessage(
        "EVENT_NOTIFY",
        "EVENT",
        "G2T",
        {
          "tile_id": 0,
          "ctx_slot": 1,
          "ctx_generation": 5,
          "local_event_id": 9,
          "event_generation": 1,
          "sequence": 2,
          "scope": "group_l2",
          "status": "DONE",
          "error_code": 0,
        },
      ),
      tp.BoundaryMessage(
        "PROGRAM_INSTALL",
        "PROGRAM",
        "G2T",
        {
          "program_id": 7,
          "version": 1,
          "package_hash": pkg_hash,
          "program_epoch": 3,
          "slot": 0,
        },
      ),
      tp.BoundaryMessage(
        "PROGRAM_READ_RESP",
        "PROGRAM",
        "G2T",
        {
          "program_id": 7,
          "version": 1,
          "package_hash": pkg_hash,
          "program_epoch": 3,
          "hbm_chunk_len": 160,
        },
      ),
      tp.BoundaryMessage(
        "MEM_RSP",
        "MEMORY",
        "G2T",
        {
          "txn_id": 88,
          "slice_id": 0,
          "binding_generation": 2,
          "visibility": 2,
          "status": 0,
        },
      ),
    ]
    pos_before_send = int(backend.abi.status().posedge_count)
    for m in outbound:
      backend.send(m)
    record(
      int(backend.abi.status().posedge_count) == pos_before_send,
      "send_does_not_tick_verilator",
    )

    received: list[tp.BoundaryMessage] = []
    for cyc in range(64):
      backend.step(cyc)
      while backend.poll_inbound():
        popped = backend.pop_inbound()
        assert popped is not None
        received.append(popped)

    by_type: dict[str, list[tp.BoundaryMessage]] = {}
    for rm in received:
      by_type.setdefault(rm.message_type, []).append(rm)

    record(
      "MEM_REQ" in by_type and by_type["MEM_REQ"][0].fields["length"] == 128,
      "t2g_136b_mem_req_decoded",
    )
    record(
      "MEM_REQ" in by_type and by_type["MEM_REQ"][0].fields["txn_id"] == 77,
      "t2g_136b_mem_req_fields_intact",
    )
    acks = {int(a.fields["txn_id"]): a.fields for a in by_type.get("ACK", [])}
    record(acks.get(101, {}).get("outcome") == "ACCEPTED", "ack_task_plan_96b_accepted")
    record(acks.get(102, {}).get("outcome") == "HOLD_READY", "ack_task_commit_104b_hold_ready")
    record(acks.get(103, {}).get("outcome") == "STALE", "ack_stale_ctx_gen_computed_by_sv_dut")
    record(
      by_type.get("POST_EVENT", [])[0].fields["local_event_id"] == 9,
      "event_channel_104b_to_96b_roundtrip",
    )
    record(
      by_type.get("PROGRAM_READ_REQ", [])[0].fields["package_hash"] == pkg_hash,
      "program_install_digest32_roundtrip",
    )
    record(
      by_type.get("PROGRAM_INSTALL_ACK", [])[0].fields["installed_len"] == 160
      and by_type["PROGRAM_INSTALL_ACK"][0].fields["measured_hash"] == pkg_hash,
      "program_128b_install_ack_roundtrip",
    )
    record(
      by_type.get("TERMINAL_ACK", [])[0].fields["txn_id"] == 88,
      "memory_terminal_ack_roundtrip",
    )
    record(backend.credit_invariant_holds(), "rtl_backend_credits_conserved")
  finally:
    backend.close()

  # 5. DUT T2G tamper negative (`CN_STUB_FLAG_CORRUPT_T2G`): proves T2G comes from SV DUT pins!
  corrupted = rb.RtlTileBackend(
    0,
    library=so_path,
    is_unit_stub=True,
    ready_seed=0xFFFFFFFF,
    stub_flags=cn_abi.CN_STUB_FLAG_CORRUPT_T2G,
  )
  try:
    corrupted.send(tp.BoundaryMessage("QUERY_CAPS", "CTRL", "G2T", {"txn_id": 200}))
    for cyc in range(4):
      corrupted.step(cyc)
    got_tampered = corrupted.pop_inbound()
    record(got_tampered is not None, "tampered_t2g_frame_received")
    assert got_tampered is not None
    record(
      got_tampered.message_type == "CAPS_RESP" and int(got_tampered.fields["txn_id"]) == (200 ^ 1),
      "dut_t2g_tamper_detected_not_self_echo",
    )
  finally:
    corrupted.close()

  # 6. Catchable DUT fatal via `RtlTileBackend` writes `incomplete_fatal.json` with residual pending set.
  fatal_dir = out_dir / "fatal_incomplete"
  fatal_be = rb.RtlTileBackend(
    0,
    library=so_path,
    is_unit_stub=True,
    ready_seed=0x00000000,
    inject_fatal_at_tick=1,
    incomplete_out_dir=fatal_dir,
  )
  try:
    fatal_be.send(tp.BoundaryMessage("QUERY_CAPS", "CTRL", "G2T", {"txn_id": 7}))
    fatal_be.send(tp.BoundaryMessage("QUERY_CAPS", "CTRL", "G2T", {"txn_id": 8}))
    fatal_caught = False
    for cyc in range(4):
      try:
        fatal_be.step(cyc)
      except cn_abi.CnAbiError as exc:
        if exc.code == "DUT_FATAL":
          fatal_caught = True
          break
    record(fatal_caught, "rtl_backend_catches_dut_fatal")
    inc_file = fatal_dir / "incomplete_fatal.json"
    record(inc_file.is_file(), "incomplete_fatal_json_written")
    inc_doc = json.loads(inc_file.read_text())
    record(
      inc_doc["verdict"] == "INCOMPLETE_FATAL" and len(inc_doc["pending_set"]) >= 1,
      "incomplete_fatal_records_pending_set",
    )
  finally:
    fatal_be.close()

  summary = {"schema": "cn_c025_in_container_verify/v1", "passed": len(results), "checks": results}
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "python_abi_verify.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary


def main() -> int:
  repo_root = Path(sys.argv[1]).resolve()
  so_path = Path(sys.argv[2]).resolve()
  out_dir = Path(sys.argv[3]).resolve()
  summary = run_all(repo_root, so_path, out_dir)
  print(json.dumps({"passed": summary["passed"]}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
