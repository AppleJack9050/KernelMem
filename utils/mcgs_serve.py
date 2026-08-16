"""Interactive browser view of the MCGS graph, live while a run builds it.

    python -m utils.mcgs_serve run/<stamp>_<task>_<tag>
    -> http://127.0.0.1:8747

Same data source as ``utils.mcgs_view`` -- the graph a run has already written to
disk -- but rendered as a pannable, zoomable, clickable DAG instead of an ASCII
tree. Read-only: it never imports the loop and never writes to the run folder, so
starting or killing it cannot affect a run in progress.

Binds 127.0.0.1 by default. The page serves the contents of a run directory, so
it is not something to expose on a network without meaning to; ``--host`` exists
for the container/remote-desktop case and warns when it is not a loopback
address.

Design notes for the picture itself:

* Node fill is a **sequential** blue ramp over Q -- one hue, light to dark,
  because Q is a magnitude. The ramp stops at the ordinal floor (light step 250,
  dark step 600) so the palest node is still a visible mark rather than a hole in
  the surface. Q is printed on every node too: fill is the fast read, the number
  is the true one.
* ``best``, ``dead`` and ``back-edge`` are **status**, not series, so each ships
  an icon and a label and never leans on hue alone.
* Layers are BFS depth from the root, matching where ``utils.mcgs_view`` expands
  a node, so the two views agree about a transposition's home.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from utils.mcgs_view import find_graph_file, load_graph

DEFAULT_PORT = 8747


def graph_payload(root_path: Path, lam: float = 0.7) -> Dict[str, Any]:
    """The graph as JSON for the page. Re-resolves the file every call.

    Re-resolving matters: a watcher is normally started before the run's first
    round boundary, so the file it must render does not exist yet when the page
    first loads.
    """
    f = find_graph_file(root_path)
    if f is None:
        return {"ok": False, "note": f"no checkpoint.json or graph.json under {root_path} yet",
                "source": str(root_path)}
    graph, note = load_graph(f)
    if graph is None:
        return {"ok": False, "note": note, "source": str(f)}
    best = graph.best()
    nodes = {}
    for k, n in graph.nodes.items():
        nodes[k] = {
            "key": k, "depth": n.depth, "N": n.N, "W": n.W, "M": n.M,
            "q": n.q(lam), "value": n.rep_value, "rep": n.rep,
            # The representative's source, so a state can be read as code and not
            # only as statistics. Served through /api/code, which re-checks the
            # path -- this field is a hint, never an authorisation.
            "rep_path": n.rep_path,
            "members": n.members, "parents": n.parents, "children": n.children,
            "via": n.via, "failures": n.failures,
            # `rep is None` is what makes a state unexpandable; `runnable` is
            # ORed to True by observe() and cannot express it.
            "dead": n.rep is None,
            "tried": n.tried[-12:],
        }
    try:
        mtime = f.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "ok": True, "note": "", "source": str(f), "mtime": mtime,
        "root": graph.root, "stats": graph.stats(), "lam": lam,
        "best": best.key if best is not None else None,
        "nodes": nodes,
    }


_CODE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
_MAX_CODE_BYTES = 4_000_000


def read_code(root_path: Path, requested: str) -> Dict[str, Any]:
    """Source of a kernel file, if and only if it lives under the served root.

    The path arrives from the page, so it is attacker-controlled in exactly the
    way a `../../etc/passwd` is: the node's `rep_path` is a hint about where to
    look, never permission to read it. Both sides are resolved before the
    containment test so symlinks and `..` cannot escape, and the suffix list
    keeps this to source files rather than a general file server.
    """
    if not requested:
        return {"ok": False, "note": "no path given"}
    root = root_path if root_path.is_dir() else root_path.parent
    try:
        root = root.resolve()
        target = Path(requested).resolve()
    except OSError as exc:
        return {"ok": False, "note": f"unresolvable path: {exc}"}
    if not target.is_relative_to(root):
        return {"ok": False, "note": f"refused: {target} is outside the served root {root}"}
    if target.suffix not in _CODE_SUFFIXES:
        return {"ok": False, "note": f"refused: {target.suffix or 'no'} suffix is not source"}
    if not target.is_file():
        return {"ok": False, "note": "the kernel file is gone (cleaned run dir?)"}
    try:
        size = target.stat().st_size
        if size > _MAX_CODE_BYTES:
            return {"ok": False, "note": f"file is {size} bytes, over the {_MAX_CODE_BYTES} cap"}
        return {"ok": True, "path": str(target), "bytes": size,
                "code": target.read_text(encoding="utf-8", errors="replace")}
    except OSError as exc:
        return {"ok": False, "note": f"unreadable: {exc}"}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCGS graph</title>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  /* sequential blue, ordinal floor at step 250 (2.06:1 on this surface) */
  --seq-0:#86b6ef; --seq-1:#5598e7; --seq-2:#2a78d6; --seq-3:#1c5cab; --seq-4:#0d366b;
  /* Edges are structure, not chrome: a gridline token is too recessive to
     carry an arrowhead, so they sit at the muted-ink step instead. */
  --edge:#898781;
  --status-good:#0ca30c; --status-serious:#ec835a; --status-critical:#d03b3b;
  --on-fill-light:#0b0b0b; --on-fill-dark:#ffffff;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  /* dark ramp: ceiling at step 600 (2.15:1 on the dark surface) */
  --seq-0:#cde2fb; --seq-1:#9ec5f4; --seq-2:#6da7ec; --seq-3:#3987e5; --seq-4:#184f95;
  --edge:#898781;   /* muted ink is mode-invariant in this palette */
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --seq-0:#cde2fb; --seq-1:#9ec5f4; --seq-2:#6da7ec; --seq-3:#3987e5; --seq-4:#184f95;
  --edge:#898781;   /* muted ink is mode-invariant in this palette */
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--text-primary);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;height:100vh;
  display:flex;flex-direction:column;overflow:hidden}
header{padding:10px 16px;background:var(--surface-1);border-bottom:1px solid var(--border);
  display:flex;gap:18px;align-items:center;flex-wrap:wrap;flex:0 0 auto}
h1{font-size:14px;font-weight:600;margin:0}
.stat{display:flex;flex-direction:column;line-height:1.25}
.stat b{font-size:15px;font-variant-numeric:tabular-nums;font-weight:600}
.stat span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.spacer{flex:1}
button,select{font:inherit;padding:4px 10px;border:1px solid var(--border);border-radius:6px;
  background:var(--surface-1);color:var(--text-primary);cursor:pointer}
button[aria-pressed="true"]{background:var(--seq-2);color:#fff;border-color:transparent}
main{flex:1;display:flex;min-height:0}
#wrap{flex:1;position:relative;background:var(--surface-1);overflow:hidden}
svg{width:100%;height:100%;display:block;cursor:grab}
svg.drag{cursor:grabbing}
aside{width:430px;flex:0 0 430px;border-left:1px solid var(--border);background:var(--surface-1);
  padding:14px;overflow:auto}
aside h2{font-size:13px;margin:0 0 8px;font-weight:600}
aside dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:0 0 12px;font-size:12px}
aside dt{color:var(--muted)}
aside dd{margin:0;font-variant-numeric:tabular-nums;word-break:break-all}
.k{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
.chip{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:1px 7px;
  border-radius:999px;border:1px solid var(--border);margin:0 4px 4px 0}
.tried{font-size:11px;border-top:1px solid var(--border);padding:5px 0;color:var(--text-secondary)}
.codehead{display:flex;align-items:center;gap:8px;justify-content:space-between;
  font-size:11px;color:var(--muted);margin:10px 0 4px}
pre.code{margin:0 0 12px;padding:10px;background:var(--plane);border:1px solid var(--border);
  border-radius:6px;max-height:46vh;overflow:auto;font-size:11px;line-height:1.45;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;color:var(--text-primary)}
.legend{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:11px;
  color:var(--text-secondary);padding:7px 16px;background:var(--surface-1);
  border-top:1px solid var(--border);flex:0 0 auto}
.legend i{font-style:normal}
.ramp{display:inline-flex;height:9px;width:76px;border-radius:2px;overflow:hidden;
  border:1px solid var(--border)}
.ramp s{flex:1;display:block}
.note{padding:22px;color:var(--text-secondary);max-width:60ch}
.hint{color:var(--muted);font-size:11px}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
</style></head><body>
<header>
  <h1>MCGS graph</h1>
  <div class="stat"><b id="s-states">–</b><span>states</span></div>
  <div class="stat"><b id="s-kernels">–</b><span>kernels</span></div>
  <div class="stat"><b id="s-merged">–</b><span>merged</span></div>
  <div class="stat"><b id="s-visits">–</b><span>visits</span></div>
  <div class="stat"><b id="s-depth">–</b><span>max depth</span></div>
  <div class="stat"><b id="s-best">–</b><span>best value</span></div>
  <div class="spacer"></div>
  <button id="live" aria-pressed="true">● Live</button>
  <button id="fit">Fit</button>
  <button id="theme">Theme</button>
</header>
<main>
  <div id="wrap"><svg id="svg"><g id="scene"></g></svg><div id="empty" class="note"></div></div>
  <aside id="side"><h2>No state selected</h2>
    <p class="hint">Click a node to see its members, its statistics, and what has
    already been tried from it. Drag to pan, scroll to zoom.</p></aside>
</main>
<div class="legend">
  <span>Q&nbsp;<span class="ramp"><s style="background:var(--seq-0)"></s><s style="background:var(--seq-1)"></s><s style="background:var(--seq-2)"></s><s style="background:var(--seq-3)"></s><s style="background:var(--seq-4)"></s></span>&nbsp;low → high</span>
  <span><i style="color:var(--status-good)">★</i> best state</span>
  <span><i style="color:var(--status-critical)">✕</i> dead (no runnable kernel)</span>
  <span><i style="color:var(--status-serious)">⇠ dashed</i> back-edge (cycle)</span>
  <span><i>◎</i> transposition (&gt;1 parent)</span>
  <span id="src" class="hint"></span>
</div>
<script>
const SVGNS="http://www.w3.org/2000/svg";
const $=s=>document.querySelector(s);
let DATA=null, sel=null, live=true;
let view={x:40,y:40,k:1};

function ramp(q){ // sequential: 5 steps, clamped
  const i=Math.max(0,Math.min(4,Math.floor(q*5)));
  return `var(--seq-${i})`;
}
function inkOn(q){ return q>=0.6 ? "var(--on-fill-dark)" : "var(--on-fill-light)"; }
function shortKey(k){ return k.length>16 ? k.slice(0,15)+"…" : k; }

// An edge is a back-edge iff it points to its own layer or a shallower one.
//
// The tempting test -- "c can reach p, so this edge is in a cycle" -- marks
// EVERY edge of the cycle, including the ordinary forward ones that merely
// participate. What is worth seeing is the single edge that CLOSES the loop,
// and in a BFS layering that is exactly the edge that fails to descend.
function isBack(depth,p,c){ return depth[c]!==undefined && depth[p]!==undefined
  && depth[c] <= depth[p]; }

function layout(d){
  const nodes=d.nodes, root=d.root;
  // BFS from the root fixes each node's layer, so a node sits at its shallowest
  // route -- the same rule the CLI view uses to decide where to expand it.
  const depth={}, order=[];
  if(root&&nodes[root]){ depth[root]=0; const q=[root];
    while(q.length){ const k=q.shift(); order.push(k);
      for(const c of nodes[k].children) if(nodes[c]&&depth[c]===undefined){depth[c]=depth[k]+1;q.push(c);}
    }
  }
  let maxd=0; for(const k in depth) maxd=Math.max(maxd,depth[k]);
  const orphans=Object.keys(nodes).filter(k=>depth[k]===undefined);
  orphans.forEach(k=>depth[k]=maxd+1);
  const levels={}; for(const k in depth){ (levels[depth[k]] ||= []).push(k); }
  // Barycentre pass: order each level by the mean row of its parents so edges
  // cross as little as possible without a full Sugiyama.
  const row={}; const keys=Object.keys(levels).map(Number).sort((a,b)=>a-b);
  for(const d0 of keys){
    const lv=levels[d0];
    if(d0>0) lv.sort((a,b)=>{
      const m=k=>{const ps=(nodes[k].parents||[]).filter(p=>row[p]!==undefined);
        return ps.length?ps.reduce((s,p)=>s+row[p],0)/ps.length:1e9;};
      return m(a)-m(b) || a.localeCompare(b);
    });
    lv.forEach((k,i)=>row[k]=i);
  }
  const COLW=250, ROWH=74;
  const pos={};
  for(const k in depth) pos[k]={x:depth[k]*COLW, y:row[k]*ROWH};
  return {pos,depth,orphans,maxd};
}

function render(){
  const d=DATA; const scene=$("#scene"); scene.textContent="";
  if(!d||!d.ok){ $("#empty").textContent=d?d.note:"loading…"; $("#empty").style.display="block"; return; }
  $("#empty").style.display="none";
  const {pos,depth}=layout(d), nodes=d.nodes;
  const NW=190,NH=44;
  // Arrowheads: an edge is an ordered relation (parent produced child) and a
  // bare line does not say which way. Two markers so the return edge keeps its
  // own colour rather than inheriting the forward one.
  const defs=document.createElementNS(SVGNS,"defs");
  defs.innerHTML=`
    <marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="var(--edge)"/></marker>
    <marker id="ah-back" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="var(--status-serious)"/></marker>`;
  const edges=document.createElementNS(SVGNS,"g");
  const nodeg=document.createElementNS(SVGNS,"g");
  scene.append(defs,edges,nodeg);

  for(const k in nodes) for(const c of nodes[k].children){
    if(!pos[c]||!pos[k]) continue;
    const back=isBack(depth,k,c);
    const p=document.createElementNS(SVGNS,"path");
    if(back){
      // A return edge routed like a forward one swings right before coming
      // back, which reads as a tangle. Arc it UNDER both nodes instead, so the
      // shape itself says "this goes backwards".
      const x1=pos[k].x+NW/2, y1=pos[k].y+NH, x2=pos[c].x+NW/2, y2=pos[c].y+NH;
      const dip=Math.max(38,Math.abs(x1-x2)*0.16);
      p.setAttribute("d",`M${x1},${y1} C${x1},${y1+dip} ${x2},${y2+dip} ${x2},${y2}`);
      p.setAttribute("stroke","var(--status-serious)");
      p.setAttribute("stroke-width",2);
      p.setAttribute("stroke-dasharray","5 4");
      p.setAttribute("marker-end","url(#ah-back)");
    } else {
      const x1=pos[k].x+NW, y1=pos[k].y+NH/2, x2=pos[c].x, y2=pos[c].y+NH/2;
      const mx=(x1+x2)/2;
      p.setAttribute("d",`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
      p.setAttribute("stroke","var(--edge)");
      p.setAttribute("stroke-width",1.5);
      p.setAttribute("marker-end","url(#ah)");
    }
    p.setAttribute("fill","none");
    edges.appendChild(p);
  }

  for(const k in nodes){
    const n=nodes[k], p=pos[k]; if(!p) continue;
    const g=document.createElementNS(SVGNS,"g");
    g.setAttribute("transform",`translate(${p.x},${p.y})`);
    g.style.cursor="pointer";
    const r=document.createElementNS(SVGNS,"rect");
    r.setAttribute("width",NW); r.setAttribute("height",NH); r.setAttribute("rx",6);
    r.setAttribute("fill",ramp(n.q));
    r.setAttribute("stroke", sel===k?"var(--text-primary)":(n.dead?"var(--status-critical)":"var(--border)"));
    r.setAttribute("stroke-width", sel===k?2.5:(n.dead?2:1));
    g.appendChild(r);
    const t1=document.createElementNS(SVGNS,"text");
    t1.setAttribute("x",9); t1.setAttribute("y",18); t1.setAttribute("fill",inkOn(n.q));
    t1.setAttribute("font-size",12); t1.setAttribute("font-weight",600);
    t1.textContent=(d.best===k?"★ ":"")+(n.dead?"✕ ":"")+((n.parents||[]).length>1?"◎ ":"")+shortKey(k);
    const t2=document.createElementNS(SVGNS,"text");
    t2.setAttribute("x",9); t2.setAttribute("y",34); t2.setAttribute("fill",inkOn(n.q));
    t2.setAttribute("font-size",11); t2.setAttribute("opacity",.92);
    t2.setAttribute("font-family","ui-monospace,SFMono-Regular,Menlo,monospace");
    t2.textContent=`Q ${n.q.toFixed(3)}  N ${n.N}  v ${n.value.toFixed(4)}`;
    g.append(t1,t2);
    if(n.via){
      const t3=document.createElementNS(SVGNS,"text");
      t3.setAttribute("x",9); t3.setAttribute("y",NH+13);
      t3.setAttribute("fill","var(--muted)"); t3.setAttribute("font-size",10);
      t3.textContent="← "+n.via;
      g.appendChild(t3);
    }
    g.addEventListener("click",e=>{
      e.stopPropagation();
      if(sel!==k) codeOpenFor=null;   // a new state starts with its source closed
      sel=k; render(); detail(k);
    });
    nodeg.appendChild(g);
  }
  apply();
}

function detail(k){
  const n=DATA.nodes[k]; const s=$("#side");
  const chips=[];
  if(DATA.best===k) chips.push(`<span class="chip" style="color:var(--status-good)">★ best</span>`);
  if(n.dead) chips.push(`<span class="chip" style="color:var(--status-critical)">✕ dead</span>`);
  if((n.parents||[]).length>1) chips.push(`<span class="chip">◎ transposition</span>`);
  const tried=(n.tried||[]).map(t=>`<div class="tried"><b>${t.mechanism||"(unnamed)"}</b> — ${
    t.runnable?("scored "+Number(t.value||0).toFixed(4)):"FAILED to compile/run"}${
    t.note?" · "+t.note:""}</div>`).join("")||`<div class="tried">nothing tried yet</div>`;
  s.innerHTML=`<h2 class="k">${k}</h2><div>${chips.join("")}</div>
   <dl><dt>Q</dt><dd>${n.q.toFixed(4)}</dd>
   <dt>N</dt><dd>${n.N}</dd><dt>W / M</dt><dd>${n.W.toFixed(3)} / ${n.M.toFixed(3)}</dd>
   <dt>value</dt><dd>${n.value.toFixed(4)}</dd><dt>depth</dt><dd>${n.depth}</dd>
   <dt>via</dt><dd>${n.via||"—"}</dd><dt>rep</dt><dd>${n.rep||"—"}</dd>
   <dt>failures</dt><dd>${n.failures}</dd>
   <dt>parents</dt><dd>${(n.parents||[]).length}</dd>
   <dt>children</dt><dd>${(n.children||[]).length}</dd>
   <dt>members</dt><dd>${(n.members||[]).join("<br>")||"—"}</dd></dl>
   ${n.rep_path?`<button id="showcode">View kernel source</button>
     <div id="codebox"></div>`:`<p class="hint">No source on record for this state.</p>`}
   <h2>Tried from here</h2>${tried}`;
  const btn=document.getElementById("showcode");
  if(btn) btn.onclick=()=>loadCode(n.rep_path,n.rep);
  // A live poll re-renders this panel every time the checkpoint changes. Without
  // this the open source would silently vanish mid-read, once per round.
  if(codeOpenFor===k&&n.rep_path) loadCode(n.rep_path,n.rep);
}

let codeOpenFor=null;

function esc(s){ return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

async function loadCode(path,name){
  const box=document.getElementById("codebox");
  if(!box) return;
  codeOpenFor=sel;
  box.innerHTML=`<p class="hint">loading ${esc(name||"")} …</p>`;
  try{
    const r=await fetch("/api/code?path="+encodeURIComponent(path),{cache:"no-store"});
    const d=await r.json();
    if(!d.ok){ box.innerHTML=`<p class="hint">${esc(d.note)}</p>`; return; }
    // Escaped, not innerHTML'd raw: this is generated CUDA/Python that routinely
    // contains "<" and "&", and one unescaped file would rewrite the page.
    box.innerHTML=`<div class="codehead">${esc(d.path.split("/").pop())} · ${d.bytes} bytes
      <button id="copycode">Copy</button></div><pre class="code">${esc(d.code)}</pre>`;
    document.getElementById("copycode").onclick=()=>navigator.clipboard.writeText(d.code);
  }catch(e){ box.innerHTML=`<p class="hint">could not load source</p>`; }
}

function apply(){ $("#scene").setAttribute("transform",
  `translate(${view.x},${view.y}) scale(${view.k})`); }
function fit(){
  const b=$("#scene").getBBox(); const w=$("#wrap").clientWidth, h=$("#wrap").clientHeight;
  if(!b.width||!b.height) return;
  view.k=Math.min(w/(b.width+80),h/(b.height+80),1.4);
  view.x=(w-b.width*view.k)/2-b.x*view.k; view.y=(h-b.height*view.k)/2-b.y*view.k;
  apply();
}
const svg=$("#svg");
let drag=null;
svg.addEventListener("mousedown",e=>{drag={x:e.clientX-view.x,y:e.clientY-view.y};svg.classList.add("drag");});
addEventListener("mousemove",e=>{if(drag){view.x=e.clientX-drag.x;view.y=e.clientY-drag.y;apply();}});
addEventListener("mouseup",()=>{drag=null;svg.classList.remove("drag");});
svg.addEventListener("wheel",e=>{e.preventDefault();
  const r=svg.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const f=e.deltaY<0?1.12:1/1.12, nk=Math.max(.15,Math.min(3,view.k*f));
  view.x=mx-(mx-view.x)*(nk/view.k); view.y=my-(my-view.y)*(nk/view.k); view.k=nk; apply();
},{passive:false});
$("#fit").onclick=fit;
$("#live").onclick=e=>{live=!live;e.target.setAttribute("aria-pressed",live);
  e.target.textContent=live?"● Live":"‖ Paused";};
$("#theme").onclick=()=>{const r=document.documentElement;
  const cur=r.getAttribute("data-theme");
  r.setAttribute("data-theme", cur==="dark"?"light":"dark");};

// ?theme=light|dark pins the theme for this load, so a screenshot or a shared
// link is not at the mercy of the viewer's OS setting.
const qTheme=new URLSearchParams(location.search).get("theme");
if(qTheme==="light"||qTheme==="dark") document.documentElement.setAttribute("data-theme",qTheme);

let lastM=-1, firstDraw=true;
async function poll(){
  try{
    const r=await fetch("/api/graph",{cache:"no-store"}); const d=await r.json();
    $("#src").textContent=d.source||"";
    if(d.ok){
      $("#s-states").textContent=d.stats.states;
      $("#s-kernels").textContent=d.stats.kernels;
      $("#s-merged").textContent=d.stats.merged_states;
      $("#s-visits").textContent=d.stats.total_visits;
      $("#s-depth").textContent=d.stats.max_depth_seen;
      const b=d.best&&d.nodes[d.best];
      $("#s-best").textContent=b?b.value.toFixed(4):"–";
      if(d.mtime!==lastM){ lastM=d.mtime; DATA=d; render();
        if(firstDraw){
          firstDraw=false; fit();
          // ?select=<key> deep-links a state (and &code=1 opens its source), so a
          // particular kernel can be linked to rather than described.
          const qs=new URLSearchParams(location.search);
          const want=qs.get("select");
          if(want&&d.nodes[want]){
            sel=want; render();
            // Mark it open and let the single detail() call below do the load,
            // rather than calling detail() twice and wiping the first result.
            if(qs.get("code")==="1"&&d.nodes[want].rep_path) codeOpenFor=want;
          }
        }
        if(sel&&d.nodes[sel]) detail(sel);
      }
    } else { DATA=d; render(); }
  }catch(e){ /* server restarting or file mid-write; next tick retries */ }
}
poll(); setInterval(()=>{ if(live) poll(); },3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    root_path: Path = Path(".")
    lam: float = 0.7

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/code":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            payload = read_code(self.root_path, (q.get("path") or [""])[0])
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif path == "/api/graph":
            try:
                payload = graph_payload(self.root_path, self.lam)
            except Exception as exc:                     # never take the page down
                payload = {"ok": False, "note": f"{exc.__class__.__name__}: {exc}"}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, *_args) -> None:
        pass                                             # one line per 3s poll is noise


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive browser view of the MCGS graph.")
    ap.add_argument("path", type=Path, help="batch folder, task folder, or graph file")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; anything else exposes the run "
                         "directory's contents to that interface)")
    ap.add_argument("--lam", type=float, default=0.7,
                    help="lambda for Q = (1-lam)*mean + lam*max; match the run's --mcgs_lam")
    ap.add_argument("--no_open", action="store_true", help="do not open a browser")
    a = ap.parse_args(argv)

    if not a.path.exists():
        print(f"[mcgs-serve] no such path: {a.path}", file=sys.stderr)
        return 2
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[mcgs-serve] WARNING: binding {a.host}, not loopback. This serves the "
              f"contents of {a.path} to anything that can reach that address.")

    Handler.root_path = a.path
    Handler.lam = a.lam
    try:
        httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as exc:
        print(f"[mcgs-serve] cannot bind {a.host}:{a.port} -- {exc}", file=sys.stderr)
        return 2
    url = f"http://{a.host}:{a.port}"
    print(f"[mcgs-serve] {a.path}\n[mcgs-serve] {url}   (Ctrl-C to stop)", flush=True)
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mcgs-serve] stopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
