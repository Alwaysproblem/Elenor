# W3 VME region declaration ABI v1 (C012) — DRAFT

Machine-readable contract: `w3_vme_decl_abi_v1.json`. Kernel mode tables: `w3_vme_mode_tables_v1.json`.
Request to W1-01: `field_requests/w3.yaml`. Executable checks: `scripts/lib/validate_vme_decl.py`,
reference model `scripts/lib/vme_decl_ref.py`.

This is a **contract check**, not an RTL result. Nothing here says W2/W3 hardware already does this.
The real progress proofs are C044 and C053.

## What is NOT decided here

| Item                                   | State          | Owner                                                    |
| -------------------------------------- | -------------- | -------------------------------------------------------- |
| Declaration CSR bit positions          | `unresolved`   | W1-01 CSR allocator (C017)                               |
| Declaration CSR width                  | `unresolved`   | the plan's "32 bits" is not a sourced bound              |
| Declaration CSR access level           | `unspecified`  | write-only per Context is the intent, not a frozen fact  |
| END instruction encoding               | undecided      | W1/W2 helper; the plan's `Decode.scala:1194` is `csrrw`  |
| Real CSR addresses                     | none assigned  | C017; `0x7c0` is never assumed free                      |

## Two events

```text
              E1 request/park                          E2 consume
   target at issue head, valid,               the beat the front end ACCEPTS the
   NOT yet in the shared RVV queue            granted target instruction
   ┌───────────────────────────────┐          ┌──────────────────────────────┐
   │ free instance  -> GRANTED     │          │ pending is consumed exactly  │
   │ none free      -> PARKED      │  ─────▶  │ once; the only consume point │
   │   (ONE waiter, queue untouched)│         └──────────────────────────────┘
   └───────────────────────────────┘
   may READ pending, never consume it
   the waiter does NOT depend on the target being accepted
```

Two bad implementations are excluded. Queue first, then park: a parked instruction sits in the shared
queue and blocks other Contexts. Gate first, register the waiter only on accept: the waiter can never be
created, so nothing is ever granted (livelock). `events.E1.requires_accepted` is therefore pinned `false`.

Properties U1–U6 (no early bind, no early consume, no late bind, stall keeps pending, redirect/cancel
clears pending, event-driven wake) are in the JSON and each is exercised by a scenario.

## Release order

`state_free` → `region_done` → `wake`. `state_free` recycles state but does **not** hand the instance over.
Only a reliably stored `region_done` wakes and grants a waiter. END does not release, FINAL does not drain.

## Cleanup is not one path

| Situation                              | Outcome                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------ |
| grant taken, target redirected         | the grant is **settled** (instance released, waiters woken); "not accepted" does not mean "no side effect" |
| waiter registered, redirected          | waiter removed, head eligible again                                      |
| declaration ARMED, redirected          | pending cleared; no later instruction inherits it                        |
| ops accepted, then cancel/fault        | accepted ops still settle for real; the region ends FAILED and never publishes a success `region_done` |

## Faults

24 coded faults in 7 groups (F1 touch without binding … F7 chain order). The validator holds the required
set; adding or removing one in the JSON alone is rejected. `static: true` faults are rejectable at
build/load; the rest are runtime protocol faults. Any fault is fail-stop: the Context ends FAILED.

## Mode tables are per kernel

A table travels with the sealed kernel (W5 manifest). `reference` is the K-VME-BF16-ACC template and is
explicitly **not** a global whitelist; `tail` is a second, different table (tm = 9) accepted by the same
schema. The checker recomputes `mtype = tm<<10 | tk<<5 | mtwiden`, refuses a `tk`/`tm` above what the front
end silently clamps to, refuses a disabled mode, requires every mode to be reachable from `NONE`, and
requires every `CLEARED -> mode` row to carry `restore: true`. `CLEARED` is the state after an inter-block
`vset*`, never a compute mode.

## Corrections to the design text (found by reading the pinned tree)

- **vtype.** The design lists 18 / 265 / 256. The upstream constants are `0xD2` / `0x1C9` / `0xC0`: they
  carry the `ta/ma` bits that `msetmtype` forces to 1 when `mtwiden != 0`. 18 and 265 are the same values
  with `ta/ma` cleared. 256 is not the INT8 vtype in any form.
- **Anchors.** The plan's `RvvFrontEnd.sv:131-135` accepted beat is at line 116 (registered at 134). Its
  `Decode.scala:1194` END decode is the `csrrw` BitPat.
- **tk clamp.** `msetmtype` clamps `tk` silently by SEW (SEW8→4, SEW16→2, SEW32→1) and `tm` to 16.
- **Encoding.** The compiler's `16384 | (tk<<5) | w` has `tm = 16` shifted by 10 as its `16384`; it is not a flag.

## Evidence not obtained

No RTL run exercised E1/E2 or the mode table: they live in W2/W3 hardware that does not exist yet. The
numeric side of the same modes is covered by C011 (`w3_numeric_profile_v1.yaml`).
