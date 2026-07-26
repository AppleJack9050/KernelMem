#!/usr/bin/env python3
"""Head-to-head: KernelMem best (agent_best) vs torch.compile max-autotune,
alongside the other candidates. Same-session reference anchor (see RESULTS.md /
kernelmem-single-workload-overfit for why the stored t_b is not trusted)."""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent
CANDS = ["agent_best", "compile_maxautotune", "compile_unlimited",
         "nhwc_triton_fused", "compile_default", "compile_cudagraph", "nhwc_eager"]


def load(cand):
    rows = {}
    p = BASE / f"out_{cand}" / "traces.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        perf = ev.get("performance") or {}
        rows[t["workload"]["uuid"]] = {
            "status": ev["status"],
            "lat": perf.get("latency_ms"),
            "ref": perf.get("reference_latency_ms"),
            "axes": t["workload"]["axes"],
        }
    return rows


all_rows = {c: load(c) for c in CANDS}

# ---- geomean table (same-session anchor: ref / lat) ----
print("Geometric-mean speedup over 20 workloads (same-session reference anchor)\n")
hdr = f"{'candidate':<22}{'pass':>6}{'geo speedup':>14}{'min shape':>12}{'max shape':>12}"
print(hdr); print("-" * len(hdr))
summ = {}
for c in CANDS:
    sp, n = [], 0
    for u, r in all_rows[c].items():
        if r["status"] != "PASSED" or not r["lat"] or not r["ref"]:
            continue
        n += 1
        sp.append(r["ref"] / r["lat"])
    if not n:
        print(f"{c:<22}{'--':>6}  (no traces yet)")
        continue
    g = statistics.geometric_mean(sp)
    summ[c] = g
    print(f"{c:<22}{n:>4}/20{g:>13.3f}x{min(sp):>11.3f}x{max(sp):>11.3f}x")

# ---- per-shape head-to-head: agent_best vs compile_maxautotune ----
A, M = "agent_best", "compile_maxautotune"
if all_rows.get(A) and all_rows.get(M):
    print(f"\nPer-shape head-to-head:  {A}  vs  {M}   (speedup = ref/lat)\n")
    h = f"{'batch':>6}{'H':>6}{'W':>6}{'agent_best':>13}{'max-autotune':>15}{'winner':>14}"
    print(h); print("-" * len(h))
    keys = sorted(all_rows[A].keys(),
                  key=lambda u: (all_rows[A][u]["axes"].get("batch_size", 0),
                                 all_rows[A][u]["axes"].get("height", 0),
                                 all_rows[A][u]["axes"].get("width", 0)))
    for u in keys:
        ra, rm = all_rows[A][u], all_rows[M].get(u)
        if not rm:
            continue
        ax = ra["axes"]
        sa = ra["ref"] / ra["lat"] if ra["lat"] and ra["ref"] and ra["status"] == "PASSED" else None
        sm = rm["ref"] / rm["lat"] if rm["lat"] and rm["ref"] and rm["status"] == "PASSED" else None
        wa = f"{sa:.3f}x" if sa else ra["status"][:9]
        wm = f"{sm:.3f}x" if sm else rm["status"][:9]
        if sa and sm:
            win = "max-autotune" if sm > sa else "agent_best"
            win += f" +{abs(sm-sa)/min(sa,sm)*100:.0f}%"
        else:
            win = "-"
        print(f"{ax.get('batch_size',0):>6}{ax.get('height',0):>6}{ax.get('width',0):>6}"
              f"{wa:>13}{wm:>15}{win:>18}")

    if A in summ and M in summ:
        print(f"\nGeomean: agent_best {summ[A]:.3f}x  vs  max-autotune {summ[M]:.3f}x  "
              f"-> {'max-autotune' if summ[M]>summ[A] else 'agent_best'} wins by "
              f"{abs(summ[M]-summ[A])/min(summ[A],summ[M])*100:.1f}%")
