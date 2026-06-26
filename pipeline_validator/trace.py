"""Profiling trace collector for Perfetto / Chrome `chrome://tracing`.

Produces a Chrome Trace Format JSON (`trace_event` schema) that can be
loaded directly into Perfetto (perfetto.dev) or Chrome's built-in
`chrome://tracing` viewer.  An optional standalone HTML wrapper embeds
the same JSON with a minimal `catapult:trace_viewer` shim so it can be
opened in any browser without a server.

The trace has three kinds of events:

  * **slice**  (ph=B/E):  engine jobs (BOA/EVU/MFE/USE), UCE instruction
    phases (wait/issue/stream), region-sequencer stages, DMA jobs.
    These show up as horizontal bars on a Gantt timeline.
  * **counter** (ph=C):   stream-queue occupancy and credit, sampled
    per cycle.  These render as line graphs in Perfetto/Chrome.
  * **instant** (ph=i):   stream push/pop/release/EOS events, dispatch,
    tile_done, region_done — markers on the timeline.

Cycle → time mapping: one cycle = `hw.cycle_ns()` nanoseconds.  Chrome
trace uses microseconds, so cycles are converted to µs with 3 decimal
places to preserve sub-µs resolution at 1 GHz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import HardwareConfig


@dataclass
class Tracer:
    """Collects Chrome trace events during a simulation run.

    Call `begin`/`end` for slice events, `counter` for sampled values,
    `instant` for point-in-time markers.  After the run, `to_chrome_json`
    produces the Perfetto-loadable JSON.
    """

    hw: HardwareConfig
    _events: list[dict] = field(default_factory=list)
    # track open slices: (pid, tid, name) -> (start_us, args)
    _open: dict[tuple, tuple] = field(default_factory=dict)
    _pid_counter: int = 0
    _tid_counter: int = 0
    _pids: dict[str, int] = field(default_factory=dict)  # track_name -> pid
    _tids: dict[tuple[int, str], int] = field(default_factory=dict)

    # ---- helpers ---------------------------------------------------------

    def cycle_to_us(self, cycle: int) -> float:
        """Convert a cycle number to wall-clock microseconds."""
        ns = cycle * self.hw.cycle_ns()
        return round(ns / 1000.0, 3)

    def _pid(self, track: str) -> int:
        if track not in self._pids:
            self._pid_counter += 1
            pid = self._pid_counter
            self._pids[track] = pid
            self._events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {
                    "name": track
                },
            })
        return self._pids[track]

    def _tid(self, pid: int, thread: str) -> int:
        key = (pid, thread)
        if key not in self._tids:
            self._tid_counter += 1
            tid = self._tid_counter
            self._tids[key] = tid
            self._events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": thread
                },
            })
        return self._tids[key]

    # ---- slice events (Gantt bars) --------------------------------------

    def begin(self,
              track: str,
              thread: str,
              name: str,
              cycle: int,
              args: dict | None = None) -> None:
        """Start a slice.  Pair with `end` using the same (track, thread, name)."""
        pid = self._pid(track)
        tid = self._tid(pid, thread)
        key = (pid, tid, name)
        self._open[key] = (self.cycle_to_us(cycle), args or {})
        self._events.append({
            "name": name,
            "ph": "B",
            "pid": pid,
            "tid": tid,
            "ts": self.cycle_to_us(cycle),
            "cat": thread,
            "args": dict(args) if args else {},
        })

    def end(self, track: str, thread: str, name: str, cycle: int) -> None:
        """End a previously-started slice."""
        pid = self._pid(track)
        tid = self._tid(pid, thread)
        key = (pid, tid, name)
        self._events.append({
            "name": name,
            "ph": "E",
            "pid": pid,
            "tid": tid,
            "ts": self.cycle_to_us(cycle),
            "cat": thread,
        })
        self._open.pop(key, None)

    def complete(self,
                 track: str,
                 thread: str,
                 name: str,
                 start_cycle: int,
                 end_cycle: int,
                 args: dict | None = None) -> None:
        """Emit a complete slice (X event) with known start and end."""
        pid = self._pid(track)
        tid = self._tid(pid, thread)
        dur = self.cycle_to_us(end_cycle) - self.cycle_to_us(start_cycle)
        self._events.append({
            "name": name,
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "ts": self.cycle_to_us(start_cycle),
            "dur": max(dur, 0.001),
            "cat": thread,
            "args": dict(args) if args else {},
        })

    # ---- counter events (line graphs) -----------------------------------

    def counter(self,
                track: str,
                name: str,
                cycle: int,
                value: float,
                unit: str = "") -> None:
        """Sample a counter value at a given cycle."""
        pid = self._pid(track)
        tid = self._tid(pid, name)
        self._events.append({
            "name": name,
            "ph": "C",
            "pid": pid,
            "tid": tid,
            "ts": self.cycle_to_us(cycle),
            "args": {
                name: value,
                "unit": unit
            } if unit else {
                name: value
            },
        })

    # ---- instant events (markers) ---------------------------------------

    def instant(self,
                track: str,
                thread: str,
                name: str,
                cycle: int,
                args: dict | None = None) -> None:
        """Emit a point-in-time marker."""
        pid = self._pid(track)
        tid = self._tid(pid, thread)
        self._events.append({
            "name": name,
            "ph": "i",
            "pid": pid,
            "tid": tid,
            "ts": self.cycle_to_us(cycle),
            "cat": thread,
            "s": "t",
            "args": dict(args) if args else {},
        })

    # ---- output ---------------------------------------------------------

    def to_chrome_json(self) -> str:
        """Produce the Perfetto/Chrome-loadable JSON trace."""
        # flush any open slices
        for (pid, tid, name), (_, _) in list(self._open.items()):
            self._events.append({
                "name":
                name,
                "ph":
                "E",
                "pid":
                pid,
                "tid":
                tid,
                "ts":
                self.cycle_to_us(9999999),
                "cat":
                "open",
            })
        return json.dumps({"traceEvents": self._events},
                          indent=None,
                          separators=(",", ":"))

    def to_chrome_json_pretty(self) -> str:
        return json.dumps({"traceEvents": self._events}, indent=2)

    @property
    def event_count(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Standalone HTML wrapper
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ELENOR Pipeline Validator — Trace</title>
<style>
body {{ margin:0; background:#1a1a1a; color:#eee; font-family:monospace; }}
#header {{ padding:8px 12px; background:#222; border-bottom:1px solid #333; }}
#header h2 {{ margin:0; font-size:14px; font-weight:normal; }}
#chart {{ width:100%; height:90vh; }}
.bar {{ position:absolute; height:14px; border-radius:2px; overflow:hidden;
        white-space:nowrap; font-size:10px; color:#fff; padding:1px 3px;
        box-sizing:border-box; }}
.bar:hover {{ outline:1px solid #fff; }}
#tooltip {{ position:fixed; background:#333; border:1px solid #555; padding:6px 8px;
            font-size:11px; border-radius:3px; pointer-events:none; z-index:100;
            max-width:400px; display:none; }}
.axis {{ position:absolute; color:#888; font-size:10px; }}
.grid {{ position:absolute; border-left:1px solid #2a2a2a; }}
</style>
</head>
<body>
<div id="header"><h2>ELENOR Pipeline Validator — Gantt Trace (Chrome Trace Format embedded)</h2></div>
<div id="chart"></div>
<div id="tooltip"></div>
<script>
const TRACE = __TRACE_JSON__;
// --- build tracks and render Gantt ---
(function() {{
  const events = TRACE.traceEvents || [];
  const slices = events.filter(e => e.ph === "X" || e.ph === "B");
  // collect complete slices (X) and pair B/E
  const pairs = [];
  const openSlices = {{}};
  for (const e of events) {{
    if (e.ph === "X") pairs.push(e);
    else if (e.ph === "B") openSlices[e.pid + ":" + e.tid + ":" + e.name] = e;
    else if (e.ph === "E") {{
      const key = e.pid + ":" + e.tid + ":" + e.name;
      if (openSlices[key]) {{
        const b = openSlices[key];
        pairs.push({{name:b.name, cat:b.cat, pid:e.pid, tid:e.tid,
                     ts:b.ts, dur:Math.max(e.ts - b.ts, 0.01), args:b.args||{{}}}});
        delete openSlices[key];
      }}
    }}
  }}
  // group by (pid, tid) → tracks
  const tracks = {{}};
  const trackNames = {{}};
  for (const m of events.filter(e => e.ph === "M" && e.name === "thread_name"))
    trackNames[m.pid + ":" + m.tid] = m.args.name;
  const procNames = {{}};
  for (const m of events.filter(e => e.ph === "M" && e.name === "process_name"))
    procNames[m.pid] = m.args.name;
  for (const s of pairs) {{
    const k = s.pid + ":" + s.tid;
    if (!tracks[k]) tracks[k] = [];
    tracks[k].push(s);
  }}
  const trackKeys = Object.keys(tracks).sort();
  const allTs = pairs.map(s => s.ts);
  const allEnd = pairs.map(s => s.ts + (s.dur || 0));
  const minTs = Math.min(...allTs, 0);
  const maxTs = Math.max(...allEnd, 1);
  const range = maxTs - minTs;
  const chart = document.getElementById("chart");
  const barH = 16, rowH = 22, leftPad = 180;
  const colors = {{BOA:"#e74c3c", EVU:"#27ae60", MFE:"#3498db", USE:"#f39c12",
                   UCE:"#9b59b6", DMA:"#1abc9c", Region:"#e67e22",
                   Stream:"#95a5a6", Stage:"#2ecc71", "Global DMA":"#16a085",
                   Collective:"#d35400", default:"#888"}};
  let y = 4;
  const tooltip = document.getElementById("tooltip");
  for (const k of trackKeys) {{
    const sl = tracks[k];
    const [pidStr, tidStr] = k.split(":");
    const pid = parseInt(pidStr), tid = parseInt(tidStr);
    const pn = procNames[pid] || ("proc " + pid);
    const tn = trackNames[k] || ("thread " + tid);
    // label
    const lbl = document.createElement("div");
    lbl.className = "axis"; lbl.style.left = "4px"; lbl.style.top = (y + 2) + "px";
    lbl.style.width = (leftPad - 8) + "px";
    lbl.textContent = pn + " / " + tn;
    chart.appendChild(lbl);
    // bars
    for (const s of sl) {{
      const x = leftPad + ((s.ts - minTs) / range) * (chart.clientWidth - leftPad - 10);
      const w = Math.max((s.dur / range) * (chart.clientWidth - leftPad - 10), 1);
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.left = x + "px"; bar.style.top = y + "px";
      bar.style.width = w + "px";
      bar.style.background = colors[s.cat] || colors.default;
      bar.textContent = s.name + (w > 50 ? "" : "");
      chart.appendChild(bar);
      bar.addEventListener("mouseenter", (ev) => {{
        const a = s.args || {{}};
        let txt = s.name + "\\n" + (s.cat||"") + "  " + s.ts.toFixed(1) + "µs";
        txt += "  dur=" + (s.dur||0).toFixed(1) + "µs";
        for (const [kk, vv] of Object.entries(a)) txt += "\\n  " + kk + ": " + vv;
        tooltip.innerHTML = txt.replace(/\\n/g, "<br>");
        tooltip.style.display = "block";
        tooltip.style.left = (ev.clientX + 12) + "px";
        tooltip.style.top = (ev.clientY + 12) + "px";
      }});
      bar.addEventListener("mouseleave", () => tooltip.style.display = "none");
    }}
    y += rowH;
  }}
  // time axis
  const axisDiv = document.createElement("div");
  axisDiv.style.position = "absolute"; axisDiv.style.left = (leftPad) + "px";
  axisDiv.style.top = (y + 4) + "px"; axisDiv.style.width = "100%";
  axisDiv.className = "axis";
  axisDiv.textContent = "0µs".padEnd(20) + " ... " + maxTs.toFixed(0) + "µs";
  chart.appendChild(axisDiv);
}})();
</script>
</body>
</html>
"""


def trace_to_html(tracer: Tracer) -> str:
    """Wrap the Chrome trace JSON in a standalone HTML page with a Gantt chart."""
    json_str = tracer.to_chrome_json()
    return _HTML_TEMPLATE.replace("__TRACE_JSON__", json_str)
