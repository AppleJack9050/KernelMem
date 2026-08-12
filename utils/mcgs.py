"""Monte Carlo Graph Search over kernel states, replacing the one-way ratchet.

Why a GRAPH and not a tree
--------------------------
The ratchet in `main_memory_latest.py` keeps exactly one node -- `base_kernel` --
and branches from it forever. Every candidate it rejects becomes a node that is
visited once and never returned to. Measured on the 18 saved trees under `run/`:
169 nodes, 113 of them (66.9%) never used as a parent.

Tree search would let those nodes be revisited, but a tree cannot afford it here.
Generation is ~95% of the wall clock (16-20 min per draw), so a 10-hour run buys
~30-40 node evaluations TOTAL, and UCT's statistics need many visits per node to
mean anything. A tree spreads a tiny budget over distinct paths and leaves every
node at N=1, where the exploration term dominates and selection degenerates to
round robin.

A graph is meant to fix the budget problem rather than work around it. Kernel
edits should commute -- {vectorize, fuse} applied in either order ought to land on
the same structure -- and in a graph those are ONE state with two visits instead
of two nodes with one each. Pooling transpositions is the only mechanism that
raises N per node without spending more evaluations.

HOW WELL THAT ACTUALLY WORKS HERE, measured before trusting it
--------------------------------------------------------------
Replaying all 145 scored kernels in `run/` that still have readable sources:

* ``mechanisms`` (path multiset): 85 states, 1.71 kernels/state. 27 states hold
  more than one kernel -- but only 2 involved more than one ORDERING, and both of
  those were an artifact of set-based keying collapsing "X then X again" into
  {X}. Fixed here by counting repeats. So GENUINE commuting transpositions are
  ~absent from the recorded history; the real merging is re-derivation of an
  identical recipe, which is still worth pooling (one run re-derived the same
  migration three times) but is deduplication, not graph structure.
* ``features`` (the obvious choice, and wrong): 11 states for 145 kernels, one of
  them holding 47 kernels spanning a 457% speedup range against a 516% total
  range. It pools nearly everything, and a Q averaged over that is noise. The
  cause is visible in the data -- `is_aligned_vector_access` and `is_pointwise`
  never vary at all, and has_reuse / has_shared_memory_tile / cudagraph_eligible
  are ~constant, so the vector carries about two bits.
* ``code``: 104 states, 1.39 kernels/state, ~0% internal spread. Honest, and
  barely more than a tree.

So the default is ``mechanisms``, NOT ``features``, and the expected benefit over
a tree is modest -- roughly 1.7 kernels per state, from recipe dedup rather than
from commuting edits. That is worth having and is not the transformative win the
graph framing suggests. `merge_tolerance` exists because of the `features`
result: any merge whose value disagrees with the state's representative beyond it
is refused and split, so a bad abstraction degrades toward a tree instead of
corrupting Q.

What a "state" is
-----------------
Not a kernel -- a kernel is a sample OF a state. `state_key_mode` selects the
abstraction:

* ``mechanisms`` -- order-independent MULTISET of method_names applied from the
                    root (default). An empty set is not a state: kernels whose
                    change the catalog cannot name each get their own key, or
                    they all pile into one bucket (measured: 37 kernels, 457%
                    internal spread).
* ``features``   -- hash of the canonical `code_features_used` vector. Kept
                    because it needs no LLM call, but see above: too coarse on
                    this task. Use only with a tight `merge_tolerance`.
* ``code``       -- sha1 of the normalised source. Merges only identical kernels;
                    the graph degenerates to a tree. This is the A/B control, so
                    "did merging help" can be answered without swapping the
                    search out.

Reward, and why it is not the score
-----------------------------------
`score` is ``T_ref / T_k`` with a separately measured ``T_ref``, timed in blocked
fashion. Two defects make it unfit for backup. It carries cross-round drift
(+0.9..+1.7% measured on unchanged kernels, against a 0.5% margin), and its
denominator is corruptible -- a kernel that starves the reference's L2 read
+9.35% while running 1.28x slower. `utils/paired_bench` says it outright: rank on
``test_ms``, never on ``score``.

Backing up a max over that quantity would compound the bias at every level, and
the search would chase drift while looking like it was working. So the reward
here is the PAIRED relative gain against the node that was branched from --
drift-cancelled by construction -- and `value` accumulates as
``parent.value * (1 + rel_pct/100)``: a chain of verified gains anchored at the
seed, on one basis throughout.

Backup is max-dominant on purpose
---------------------------------
The gain distribution measured across those runs is bimodal, not centred: 42% of
edges regress past -1%, 33% win past +1%, and only 25% land inside the +-1% noise
band. You keep the best kernel found, not the average one, and the ratchet
discards losses at zero cost. Mean-backup would bury a state that produced one
+6% child among four regressions -- which is the shape of every real win in the
data. Hence ``Q = (1-lam)*mean + lam*max`` with lam high.

Depth is capped from measurement, not taste
-------------------------------------------
Win rate by round index over those runs: 41% for rounds 0-4, 41% for 5-9, then
0% for 10-14, 0% for 15-19, 0% for 20-24 (0 wins in 22 edges past round 10).
Depth past ~10 edits from the seed has never paid on this task, so `max_depth`
makes selection walk elsewhere instead of deeper. Raise it if a task proves
otherwise; the cliff is measured on vae_block_002 and is not a law of nature.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Features that define a STATE. Deliberately a subset of `code_features_used`:
# fields that describe the kernel's structure, not its numerics or its accidental
# properties. Adding a field makes the abstraction finer (fewer merges, more
# nodes at N=1); removing one makes it coarser (more merges, risk of pooling
# kernels that behave differently). This list is the main tuning knob of the
# whole method, so it is explicit rather than "everything in the dict".
CANONICAL_FEATURES: Tuple[str, ...] = (
    "kernel_structure_id",
    "has_reuse",
    "has_shared_memory_tile",
    "uses_vector_types",
    "has_vector_load_store",
    "is_aligned_vector_access",
    "tc_eligible",
    "has_k_loop",
    "is_pointwise",
    "has_multiple_kernels_in_forward",
    "cudagraph_eligible",
)


def _norm_code(code: str) -> str:
    """Source with comments and whitespace removed, for the `code` key mode.

    Without this, a reformatting-only reply is a different state and gets its own
    visit budget. Not a semantic normaliser -- it cannot tell that two different
    tilings are equivalent -- which is precisely why `features` is the default.
    """
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"#.*?$", "", code, flags=re.M)
    return re.sub(r"\s+", "", code)


def state_key(*, mode: str = "features",
              features: Optional[Dict[str, Any]] = None,
              mechanisms: Optional[Iterable[str]] = None,
              code: Optional[str] = None,
              fallback: str = "") -> str:
    """Identity of a kernel's STATE, under the selected abstraction.

    Falls back to hashing *fallback* (normally the kernel name) when the inputs
    the chosen mode needs are missing -- a round whose machine_check failed must
    become its own state rather than silently merging into whatever state hashes
    to the empty vector, which would pool unrelated kernels and corrupt Q.
    """
    if mode == "features" and features:
        vec = [f"{k}={features.get(k)!r}" for k in CANONICAL_FEATURES]
        return "f:" + hashlib.sha1("|".join(vec).encode()).hexdigest()[:16]
    if mode == "mechanisms" and mechanisms is not None:
        # MULTISET, not set. A set collapses "X then X again" into {X}, which
        # measured as the only two "transpositions" in the whole run history --
        # both false, and one pooled kernels from 0.3404 to 1.1635 speedup into a
        # single state. Counting repeats keeps applying a method twice distinct
        # from applying it once, which is what it is.
        counts: Dict[str, int] = {}
        for m in mechanisms:
            m = str(m).strip()
            if m:
                counts[m] = counts.get(m, 0) + 1
        if not counts:
            # An empty mechanism list is NOT a state. Treated as one, every kernel
            # whose change the catalog could not name lands in the same bucket:
            # measured at 37 kernels spanning 457%, which makes Q an average over
            # unrelated kernels -- strictly worse than not merging at all.
            return "x:" + hashlib.sha1(f"m-empty:{fallback}".encode()).hexdigest()[:16]
        ms = [f"{k}x{v}" for k, v in sorted(counts.items())]
        return "m:" + hashlib.sha1("|".join(ms).encode()).hexdigest()[:16]
    if mode == "code" and code:
        return "c:" + hashlib.sha1(_norm_code(code).encode()).hexdigest()[:16]
    return "x:" + hashlib.sha1(f"{mode}:{fallback}".encode()).hexdigest()[:16]


def reward_from_gain(rel_pct: Optional[float], *, scale: float = 3.0,
                     failed: bool = False) -> float:
    """Map a paired relative gain (%) onto a bounded reward in [0, 1].

    UCB's exploration constant is only interpretable against a bounded reward, so
    an unbounded speedup ratio cannot be used directly. tanh gives a smooth,
    saturating map that keeps the interesting range linear-ish: at scale=3,
    0% -> 0.50, +1% -> 0.58, +3% -> 0.88, +10% -> 1.00, and symmetrically below.

    A failed candidate scores 0 rather than 0.5. It is strictly worse than a
    measured no-op: it consumed a full evaluation and produced nothing, and a
    state that keeps emitting uncompilable code should fall out of contention on
    its own rather than needing a separate rule.
    """
    if failed or rel_pct is None:
        return 0.0
    return 0.5 * (1.0 + math.tanh(float(rel_pct) / max(scale, 1e-9)))


@dataclass
class MechanismPrior:
    """Historical advantage per optimization mechanism, as a PUCT prior.

    Why this exists
    ---------------
    Selection had no predictive signal about where the next gain would come from.
    Measured over the saved runs, a node's OWN score correlates with the gain it
    yields at rho = +0.02 -- i.e. not at all. So UCT was ranking branch points on
    a quantity that does not predict what it is being used to predict.

    A per-mechanism average does carry signal. Leave-one-RUN-out over the 108
    scored edges that name a method (a held-out run has no memory of itself):

        global mechanism prior   rho = +0.485, top-third win rate 50% vs base 34%
        state-conditioned (kNN)  rho = +0.373, top-third 18% vs base 19% -- no lift

    The state-conditioned form is JitRL's Eq. 4-7 and it buys nothing here,
    because the code-feature vector carries about two bits (see the state-key
    measurement above). The simple global average is the useful object, so that
    is what this is.

    Two corrections the raw average needs
    -------------------------------------
    * SUPPORT FLOOR, not shrinkage. The obvious correction for "the mechanisms
      driving the top third have n=3" is to shrink each estimate toward zero by
      n/(n+kappa). Swept leave-one-run-out, that makes it monotonically WORSE:

          kappa    0     1     2     5    10
          rho   +.324 +.214 +.209 +.169 +.118

      The reason is visible in the table it produces. The n=3 mechanisms really
      are good (3/3 wins each) while the n=18 one -- CUDA_Graph_Capture_Replay_
      StaticBuffers, median -0.56% -- really is mediocre, so shrinking by support
      pulls the informative estimates toward the mean and leaves the uninformative
      one at full size. `kappa` is kept as a knob and defaults to 0. What DOES
      help is a hard floor: dropping n=1 mechanisms (min_support=2) raises
      top-third precision from 58%/47% to 59%/42% at a coverage cost of 7 points.
    * ROUND CONFOUND. Win rate is 42% in rounds 0-9 and 0% after, so part of the
      raw rho is the exhaustion cliff rather than anything about mechanisms.
      Controlling for it (rounds 0-9 only) the signal drops to rho = +0.324 and a
      +11 point precision lift. `max_depth` already handles the cliff, so the
      prior must not be credited with it twice -- fit with `max_round` set to the
      productive window rather than over everything.

    Coverage is 73%: 47 distinct mechanism names over 108 edges, 27 of them
    appearing exactly once. On the rest the prior has no estimate and returns
    0.0, which is silence, not a negative opinion.
    """
    # mechanism -> (shrunk advantage in %, support count)
    table: Dict[str, Tuple[float, int]] = field(default_factory=dict)
    global_mean: float = 0.0
    kappa: float = 0.0        # shrinkage; 0 = off, measured best (see docstring)
    tau: float = 2.0          # softmax temperature over advantages, in % units
    repeat_penalty: float = 0.25   # multiplier for a mechanism already on the path
    fitted_on: int = 0
    note: str = ""

    @classmethod
    def fit(cls, observations: Sequence[Tuple[str, float]], *, kappa: float = 0.0,
            tau: float = 2.0, min_support: int = 2,
            repeat_penalty: float = 0.25, note: str = "") -> "MechanismPrior":
        """Fit from (mechanism, gain_pct) pairs.

        *min_support* drops mechanisms seen fewer times than this outright rather
        than shrinking them: a single observation carries no information about a
        mean, and including it only adds names whose advantage is one sample of a
        distribution with a 5% standard deviation.
        """
        by: Dict[str, List[float]] = {}
        for mech, gain in observations:
            m = str(mech).strip()
            if m and isinstance(gain, (int, float)) and math.isfinite(gain):
                by.setdefault(m, []).append(float(gain))
        if not by:
            return cls(note=note)
        allg = [g for v in by.values() for g in v]
        gm = sum(allg) / len(allg)
        table: Dict[str, Tuple[float, int]] = {}
        for m, v in by.items():
            n = len(v)
            if n < max(1, min_support):
                continue
            raw = (sum(v) / n) - gm
            table[m] = (raw * (n / (n + max(kappa, 0.0))), n)
        return cls(table=table, global_mean=gm, kappa=kappa, tau=tau,
                   repeat_penalty=repeat_penalty, fitted_on=len(allg), note=note)

    def advantage(self, mechanism: Optional[str]) -> float:
        """Shrunk historical advantage of a mechanism, in percent. 0.0 if unknown."""
        if not mechanism:
            return 0.0
        got = self.table.get(str(mechanism).strip())
        return got[0] if got else 0.0

    def support(self, mechanism: Optional[str]) -> int:
        got = self.table.get(str(mechanism).strip()) if mechanism else None
        return got[1] if got else 0

    def weights(self, mechanisms: Sequence[Optional[str]],
                already_applied: Sequence[str] = ()) -> List[float]:
        """Normalised PUCT priors over a candidate list, summing to 1.

        A mechanism already on the path is damped rather than removed. Removing
        it outright would forbid a legitimate second application (a wider tile
        after a different change made it fit); damping only makes it earn its way
        back. The measured case for damping: the only repeat-mechanism states in
        the run history pooled kernels from 0.3404 to 1.1635 speedup, i.e.
        reapplying the same method behaved erratically.
        """
        seen = {str(m).strip() for m in already_applied if m}
        adv = []
        for m in mechanisms:
            a = self.advantage(m)
            if m and str(m).strip() in seen:
                a = a * self.repeat_penalty if a > 0 else a
            adv.append(a)
        t = max(self.tau, 1e-6)
        mx = max(adv) if adv else 0.0
        ex = [math.exp((a - mx) / t) for a in adv]
        s = sum(ex)
        return [e / s for e in ex] if s > 0 else [1.0 / max(len(adv), 1)] * len(adv)

    def ranked(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        return sorted(((m, a, n) for m, (a, n) in self.table.items()),
                      key=lambda t: -t[1])[:limit]

    def hint(self, already_applied: Sequence[str] = (), limit: int = 5) -> str:
        """Prompt-side view: what has paid here, and what is already spent.

        Not wired into the optimization prompt by this module. Selection can only
        re-rank children that already exist, so steering GENERATION is where the
        prior would have more leverage -- but that changes what the model is asked
        for, which belongs in an A/B of its own rather than riding along.
        """
        if not self.table:
            return "(no mechanism history available)"
        seen = {str(m).strip() for m in already_applied if m}
        up = [(m, a, n) for m, a, n in self.ranked(limit * 3) if a > 0 and m not in seen][:limit]
        down = [(m, a, n) for m, a, n in reversed(self.ranked(10_000)) if a < 0][:limit]
        out = []
        if up:
            out.append("Historically paid on this task: " +
                       ", ".join(f"{m} ({a:+.2f}%, n={n})" for m, a, n in up))
        if down:
            out.append("Historically lost: " +
                       ", ".join(f"{m} ({a:+.2f}%, n={n})" for m, a, n in down))
        if seen:
            out.append("Already applied on this path: " + ", ".join(sorted(seen)))
        return "\n".join(out) or "(no mechanism history available)"

    def to_dict(self) -> Dict[str, Any]:
        return {"table": {m: [a, n] for m, (a, n) in self.table.items()},
                "global_mean": self.global_mean, "kappa": self.kappa, "tau": self.tau,
                "repeat_penalty": self.repeat_penalty, "fitted_on": self.fitted_on,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["MechanismPrior"]:
        if not d or not isinstance(d, dict):
            return None
        tbl = {}
        for m, v in (d.get("table") or {}).items():
            try:
                tbl[str(m)] = (float(v[0]), int(v[1]))
            except Exception:
                continue
        return cls(table=tbl, global_mean=float(d.get("global_mean") or 0.0),
                   kappa=float(d.get("kappa", 0.0)), tau=float(d.get("tau", 2.0)),
                   repeat_penalty=float(d.get("repeat_penalty", 0.25)),
                   fitted_on=int(d.get("fitted_on") or 0), note=str(d.get("note") or ""))


@dataclass
class StateNode:
    """One state in the DAG. Statistics are pooled over every kernel in it."""
    key: str
    depth: int = 0
    # Kernels observed in this state. `rep` is the one branched from: the
    # best-scoring member, since a state's samples differ in quality even though
    # they share an abstraction.
    members: List[str] = field(default_factory=list)
    rep: Optional[str] = None
    rep_path: Optional[str] = None
    rep_value: float = 0.0            # chained verified score of `rep`
    # Visit statistics, pooled across transpositions -- the point of the graph.
    N: int = 0
    W: float = 0.0                    # sum of rewards, for the mean term
    M: float = 0.0                    # max reward, for the max term
    failures: int = 0                 # children that never ran
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    # What has already been tried FROM here, so expansion can be told to do
    # something else. The loop's history block was mtime-ordered and global,
    # which is how one run re-derived the same migration three times.
    tried: List[Dict[str, Any]] = field(default_factory=list)
    runnable: bool = True
    # The method_name of the edge that first produced this state. Needed to build
    # the path's mechanism multiset for `mechanisms` keying -- `tried` cannot serve,
    # it lists what was attempted FROM here, not what led TO here.
    via: Optional[str] = None

    def q(self, lam: float = 0.7) -> float:
        if self.N <= 0:
            return 0.0
        return (1.0 - lam) * (self.W / self.N) + lam * self.M

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "depth": self.depth, "members": self.members,
            "rep": self.rep, "rep_path": self.rep_path, "rep_value": self.rep_value,
            "N": self.N, "W": self.W, "M": self.M, "failures": self.failures,
            "parents": self.parents, "children": self.children,
            "tried": self.tried, "runnable": self.runnable, "via": self.via,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateNode":
        n = cls(key=str(d["key"]))
        n.depth = int(d.get("depth") or 0)
        n.members = list(d.get("members") or [])
        n.rep = d.get("rep")
        n.rep_path = d.get("rep_path")
        n.rep_value = float(d.get("rep_value") or 0.0)
        n.N = int(d.get("N") or 0)
        n.W = float(d.get("W") or 0.0)
        n.M = float(d.get("M") or 0.0)
        n.failures = int(d.get("failures") or 0)
        n.parents = list(d.get("parents") or [])
        n.children = list(d.get("children") or [])
        n.tried = list(d.get("tried") or [])
        n.runnable = bool(d.get("runnable", True))
        n.via = d.get("via")
        return n


@dataclass
class Selection:
    """What the search chose, and why -- the `why` is printed every round.

    A search whose choices are not legible cannot be debugged: the ratchet at
    least printed "[base] Keeping base_kernel ...". This carries the equivalent.
    """
    node: StateNode
    path: List[str]
    reason: str


class MonteCarloGraphSearch:
    """UCT over a DAG of kernel states, with progressive widening.

    Progressive widening rather than a fixed branching factor because the action
    space is LLM-generated and unbounded -- there is no enumerable move list to
    take an argmax over. A state earns its k-th child only after enough visits to
    suggest it is worth developing: ``len(children) < ceil(k * N**alpha)``.
    """

    def __init__(self, *, c_puct: float = 0.8, lam: float = 0.7,
                 widen_k: float = 1.0, widen_alpha: float = 0.5,
                 max_depth: int = 10, reward_scale: float = 3.0,
                 state_key_mode: str = "mechanisms",
                 merge_tolerance: float = 0.15,
                 prior: Optional[MechanismPrior] = None,
                 c_prior: float = 1.0) -> None:
        self.nodes: Dict[str, StateNode] = {}
        self.root: Optional[str] = None
        self.c_puct = float(c_puct)
        self.lam = float(lam)
        self.widen_k = float(widen_k)
        self.widen_alpha = float(widen_alpha)
        self.max_depth = int(max_depth)
        self.reward_scale = float(reward_scale)
        self.state_key_mode = str(state_key_mode)
        self.merge_tolerance = float(merge_tolerance)
        self.prior = prior
        self.c_prior = float(c_prior)
        self.splits = 0            # merges refused on the value guard
        self.total_visits = 0

    # ---------------------------------------------------------------- ingestion
    def observe(self, *, key: str, kernel_name: str, kernel_path: Optional[str],
                value: float, parent_key: Optional[str] = None,
                runnable: bool = True,
                mechanism: Optional[str] = None,
                note: str = "") -> StateNode:
        """Record a kernel, creating or merging into its state.

        Merging is the whole point, so this is where a transposition shows up: a
        kernel whose key already exists joins that node and its visit count
        rather than starting a fresh one at N=0.

        But merging is only sound when the pooled kernels are actually alike. On
        the recorded runs the feature-vector key put 47 kernels spanning a 457%
        speedup range into one state, against a 516% total range -- it pooled
        nearly everything, and a Q averaged over that is meaningless. So a merge
        whose value disagrees with the state's representative by more than
        `merge_tolerance` is REFUSED: the kernel gets a derived key of its own.
        The abstraction stays coarse where coarse is safe and splits where it is
        demonstrably wrong, instead of trusting the key to be right.
        """
        node = self.nodes.get(key)
        if (node is not None and node.rep is not None and self.merge_tolerance > 0
                and runnable and node.rep_value > 0 and value > 0):
            rel = abs(value / node.rep_value - 1.0)
            if rel > self.merge_tolerance:
                key = key + "/s" + hashlib.sha1(
                    f"{kernel_name}".encode()).hexdigest()[:6]
                self.splits += 1
                node = self.nodes.get(key)
        if node is None:
            depth = 0
            if parent_key and parent_key in self.nodes:
                depth = self.nodes[parent_key].depth + 1
            node = StateNode(key=key, depth=depth, via=mechanism)
            self.nodes[key] = node
            if self.root is None and parent_key is None:
                self.root = key
        if kernel_name not in node.members:
            node.members.append(kernel_name)
        node.runnable = node.runnable or runnable
        # `rep` is the best member: a state is branched from its best sample.
        if runnable and (node.rep is None or value > node.rep_value):
            node.rep, node.rep_path, node.rep_value = kernel_name, kernel_path, float(value)
        if parent_key and parent_key in self.nodes and parent_key != key:
            p = self.nodes[parent_key]
            if key not in p.children:
                p.children.append(key)
            if parent_key not in node.parents:
                node.parents.append(parent_key)
            # A transposition can be reached at a shallower depth than first
            # seen; keep the shortest, since max_depth is about distance from the
            # seed and the graph should not punish a node for its longest route.
            node.depth = min(node.depth, p.depth + 1) if node.parents else node.depth
        if mechanism or note:
            entry = {"mechanism": mechanism, "child": kernel_name,
                     "note": note, "value": float(value), "runnable": bool(runnable)}
            if parent_key and parent_key in self.nodes:
                self.nodes[parent_key].tried.append(entry)
        return node

    def backup(self, path: Sequence[str], reward: float, *, failed: bool = False) -> None:
        """Propagate a leaf result up every node on the selected path.

        Only the selected path, not all ancestors of the leaf. Backing up through
        every route into a merged state would count one evaluation many times and
        inflate N without new information -- the standard MCGS hazard.
        """
        self.total_visits += 1
        for k in path:
            n = self.nodes.get(k)
            if n is None:
                continue
            n.N += 1
            n.W += float(reward)
            n.M = max(n.M, float(reward))
            if failed:
                n.failures += 1

    # ---------------------------------------------------------------- selection
    def _widen_budget(self, n: StateNode) -> int:
        return max(1, math.ceil(self.widen_k * (max(n.N, 1) ** self.widen_alpha)))

    def _selectable_children(self, n: StateNode) -> List[StateNode]:
        out = []
        for k in n.children:
            c = self.nodes.get(k)
            # A state with no runnable member cannot be branched from: there is
            # no code to hand the model. It keeps its visits (so its parent is
            # penalised for producing it) but is not itself a destination.
            if c is not None and c.runnable and c.rep is not None:
                out.append(c)
        return out

    def select(self) -> Optional[Selection]:
        """Walk from the root to the state that should be expanded next."""
        if not self.root or self.root not in self.nodes:
            return None
        path: List[str] = []
        node = self.nodes[self.root]
        while True:
            path.append(node.key)
            kids = self._selectable_children(node)
            if node.depth >= self.max_depth:
                return Selection(node, path,
                                 f"depth {node.depth} reached --max_depth "
                                 f"{self.max_depth}; expanding here rather than deeper")
            if not kids:
                return Selection(node, path, "leaf state (no expandable children yet)")
            budget = self._widen_budget(node)
            if len(kids) < budget:
                return Selection(node, path,
                                 f"progressive widening: {len(kids)} child state(s) < "
                                 f"budget {budget} at N={node.N}")
            # PUCT when a mechanism prior is loaded, plain UCT otherwise.
            #
            # UCT's exploration term is blind to WHAT a child did -- it only counts
            # how often it was visited, so with ~30 evaluations it spreads budget
            # evenly over children that are not equally promising. PUCT weights
            # that term by a prior, and the measured prior here is the historical
            # advantage of the mechanism that produced each child. Mechanisms
            # already applied on the path are damped inside `weights`, so the
            # prior cannot recommend the same edit all the way down a branch.
            if self.prior is not None and self.prior.table:
                applied = self.path_mechanisms(path)
                P = self.prior.weights([c.via for c in kids], already_applied=applied)
                sqrtN = math.sqrt(max(node.N, 1))
                best, best_u = None, -float("inf")
                for c, p in zip(kids, P):
                    u = c.q(self.lam) + self.c_prior * p * sqrtN / (1 + c.N)
                    if u > best_u:
                        best, best_u = c, u
            else:
                logN = math.log(max(node.N, 1))
                best, best_u = None, -float("inf")
                for c in kids:
                    u = c.q(self.lam) + self.c_puct * math.sqrt(logN / max(c.N, 1))
                    if u > best_u:
                        best, best_u = c, u
            if best is None:
                return Selection(node, path, "no child scored; expanding here")
            node = best

    # ------------------------------------------------------------------ context
    def siblings_context(self, node: StateNode, *, limit: int = 8) -> str:
        """What has already been tried from this state, with outcomes.

        Fed to the optimization prompt in place of the mtime-ordered global
        history. Outcome-labelled on purpose: a history that lists attempts
        without saying which failed invites the model to propose them again.
        """
        if not node.tried:
            return "(nothing has been tried from this kernel state yet)"
        lines = []
        for t in node.tried[-limit:]:
            if not t.get("runnable"):
                verdict = "FAILED to compile/run"
            else:
                verdict = f"scored {t.get('value', 0.0):.4f}"
            mech = t.get("mechanism") or "(unnamed change)"
            note = f" -- {t['note']}" if t.get("note") else ""
            lines.append(f"- {mech}: {verdict}{note}")
        return "\n".join(lines)

    def path_mechanisms(self, path: Sequence[str]) -> List[str]:
        """The method_names applied along *path*, in order, repeats kept.

        This is what `mechanisms` keying consumes. It must come from the edges
        that LED to each node (`via`), not from a node's `tried` list -- `tried`
        records what was attempted FROM a node, so using it would key a child by
        its siblings' attempts and merge unrelated kernels.
        """
        out: List[str] = []
        for k in path:
            n = self.nodes.get(k)
            if n is not None and n.via:
                out.append(n.via)
        return out

    def best(self) -> Optional[StateNode]:
        """The state holding the highest chained verified value."""
        cands = [n for n in self.nodes.values() if n.rep is not None]
        return max(cands, key=lambda n: n.rep_value) if cands else None

    def stats(self) -> Dict[str, Any]:
        n_nodes = len(self.nodes)
        merged = sum(1 for n in self.nodes.values() if len(n.members) > 1)
        n1 = sum(1 for n in self.nodes.values() if n.N <= 1)
        return {
            "states": n_nodes,
            "kernels": sum(len(n.members) for n in self.nodes.values()),
            "merged_states": merged,
            "splits_refused": self.splits,
            "states_at_N<=1": n1,
            "mean_N": (sum(n.N for n in self.nodes.values()) / n_nodes) if n_nodes else 0.0,
            "total_visits": self.total_visits,
            "max_depth_seen": max((n.depth for n in self.nodes.values()), default=0),
            "prior_mechanisms": len(self.prior.table) if self.prior else 0,
        }

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "root": self.root,
            "total_visits": self.total_visits,
            "params": {
                "c_puct": self.c_puct, "lam": self.lam, "widen_k": self.widen_k,
                "widen_alpha": self.widen_alpha, "max_depth": self.max_depth,
                "reward_scale": self.reward_scale,
                "state_key_mode": self.state_key_mode,
                "merge_tolerance": self.merge_tolerance,
                "c_prior": self.c_prior,
            },
            "splits": self.splits,
            # Persisted so a resumed run selects with the SAME prior it started
            # with. Re-reading the prior file on resume would silently change the
            # search policy mid-run if the file had been refitted since.
            "prior": self.prior.to_dict() if self.prior is not None else None,
            "nodes": {k: n.to_dict() for k, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MonteCarloGraphSearch":
        p = (d or {}).get("params") or {}
        g = cls(c_puct=float(p.get("c_puct", 0.8)), lam=float(p.get("lam", 0.7)),
                widen_k=float(p.get("widen_k", 1.0)),
                widen_alpha=float(p.get("widen_alpha", 0.5)),
                max_depth=int(p.get("max_depth", 10)),
                reward_scale=float(p.get("reward_scale", 3.0)),
                state_key_mode=str(p.get("state_key_mode", "mechanisms")),
                merge_tolerance=float(p.get("merge_tolerance", 0.15)),
                c_prior=float(p.get("c_prior", 1.0)))
        if not d:
            return g
        g.prior = MechanismPrior.from_dict(d.get("prior"))
        g.root = d.get("root")
        g.total_visits = int(d.get("total_visits") or 0)
        g.splits = int(d.get("splits") or 0)
        for k, nd in (d.get("nodes") or {}).items():
            g.nodes[k] = StateNode.from_dict(nd)
        return g


def load_code_features(io_dir, round_idx: int) -> Optional[Dict[str, Any]]:
    """`code_features_used` for a round, from the machine_check result on disk.

    Read rather than recomputed: machine_check already extracts these via
    judge_gate and writes them out, so keying states off the same vector costs
    nothing and keeps the graph in the vocabulary the prompts already use.
    """
    from pathlib import Path
    f = Path(io_dir) / f"round{round_idx:03d}_machine_check_result.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    feats = d.get("code_features_used")
    return feats if isinstance(feats, dict) else None


# --------------------------------------------------------------------- self-test
if __name__ == "__main__":
    def _check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        assert cond, msg

    print("[mcgs] reward mapping")
    _check(abs(reward_from_gain(0.0) - 0.5) < 1e-9, "0% -> 0.50")
    _check(reward_from_gain(3.0) > 0.85, "+3% -> >0.85")
    _check(reward_from_gain(-3.0) < 0.15, "-3% -> <0.15")
    _check(reward_from_gain(None, failed=True) == 0.0, "failed -> 0.0")
    _check(reward_from_gain(0.4) < reward_from_gain(1.2), "monotone in the gain")

    print("[mcgs] transposition merging")
    a = state_key(mode="mechanisms", mechanisms=["vectorize", "fuse"])
    b = state_key(mode="mechanisms", mechanisms=["fuse", "vectorize"])
    _check(a == b, "commuting edits reach one state (order-independent)")
    f1 = state_key(mode="features", features={"has_reuse": True, "kernel_structure_id": 2})
    f2 = state_key(mode="features", features={"kernel_structure_id": 2, "has_reuse": True})
    _check(f1 == f2, "feature vector is order-independent")
    f3 = state_key(mode="features", features={"has_reuse": False, "kernel_structure_id": 2})
    _check(f1 != f3, "a differing feature makes a different state")
    _check(state_key(mode="features", features=None, fallback="k1")
           != state_key(mode="features", features=None, fallback="k2"),
           "missing features fall back to distinct states, never a shared bucket")
    _check(state_key(mode="code", code="a=1; // x") ==
           state_key(mode="code", code="a = 1;   // different comment"),
           "code mode ignores comments and whitespace")

    print("[mcgs] mechanisms key is a multiset, and empty is not a state")
    _check(state_key(mode="mechanisms", mechanisms=["X"]) !=
           state_key(mode="mechanisms", mechanisms=["X", "X"]),
           "applying a method twice is not the same state as applying it once")
    _check(state_key(mode="mechanisms", mechanisms=["X", "Y"]) ==
           state_key(mode="mechanisms", mechanisms=["Y", "X"]),
           "distinct methods still commute (the real transposition)")
    _check(state_key(mode="mechanisms", mechanisms=[], fallback="a") !=
           state_key(mode="mechanisms", mechanisms=[], fallback="b"),
           "unnamed changes get their own states, not a shared bucket")

    print("[mcgs] merge is refused when the pooled values disagree")
    gm = MonteCarloGraphSearch(merge_tolerance=0.15)
    gm.observe(key="S", kernel_name="s", kernel_path="/s", value=1.0)
    gm.observe(key="P", kernel_name="near", kernel_path="/n", value=1.20, parent_key="S")
    n_states = len(gm.nodes)
    gm.observe(key="P", kernel_name="close", kernel_path="/c", value=1.25, parent_key="S")
    _check(len(gm.nodes) == n_states, "a value within tolerance merges as normal")
    gm.observe(key="P", kernel_name="far", kernel_path="/f", value=0.40, parent_key="S")
    _check(len(gm.nodes) == n_states + 1,
           "a value far outside tolerance is split off instead of pooled")
    _check(gm.splits == 1, "the refusal is counted")
    _check(all(len({m for m in n.members}) <= 2 for n in gm.nodes.values()),
           "no state silently accumulates mismatched members")

    print("[mcgs] mechanism prior")
    obs = ([("good", 5.0)] * 4 + [("bad", -4.0)] * 4 + [("meh", 0.2)] * 4
           + [("rare", 9.0)])
    pr = MechanismPrior.fit(obs, min_support=2)
    _check(pr.advantage("good") > pr.advantage("meh") > pr.advantage("bad"),
           "advantage orders mechanisms by their measured gain")
    _check(pr.support("rare") == 0 and pr.advantage("rare") == 0.0,
           "a mechanism under min_support is dropped, not guessed at")
    _check(pr.advantage("never_seen") == 0.0,
           "an unknown mechanism scores 0.0 -- silence, not a negative opinion")
    _check(pr.kappa == 0.0,
           "shrinkage defaults OFF: the sweep made it monotonically worse "
           "(rho +0.324 at kappa=0 down to +0.118 at kappa=10)")
    _sh = MechanismPrior.fit(obs, kappa=10.0, min_support=2)
    _check(abs(_sh.advantage("good")) < abs(pr.advantage("good")),
           "kappa still shrinks when asked for explicitly")

    w = pr.weights(["good", "bad", "meh"])
    _check(abs(sum(w) - 1.0) < 1e-9, "weights normalise to 1")
    _check(w[0] > w[2] > w[1], "the best mechanism carries the most prior mass")
    w_rep = pr.weights(["good", "bad", "meh"], already_applied=["good"])
    _check(w_rep[0] < w[0],
           "a mechanism already on the path is damped rather than removed")
    _check(w_rep[0] > 0.0, "...but can still be re-selected if nothing else looks better")
    _check(all(x > 0 for x in pr.weights([None, None])),
           "an all-unknown candidate set degrades to uniform, never to zeros")

    _rt = MechanismPrior.from_dict(json.loads(json.dumps(pr.to_dict())))
    _check(abs(_rt.advantage("good") - pr.advantage("good")) < 1e-12,
           "the prior survives a JSON round trip")

    print("[mcgs] PUCT selection uses the prior; UCT still runs without one")
    gp = MonteCarloGraphSearch(prior=pr, c_prior=1.0, widen_k=0.0, widen_alpha=0.0)
    gp.observe(key="R", kernel_name="s", kernel_path="/s", value=1.0)
    gp.observe(key="G", kernel_name="a", kernel_path="/a", value=1.0,
               parent_key="R", mechanism="good")
    gp.observe(key="B", kernel_name="b", kernel_path="/b", value=1.0,
               parent_key="R", mechanism="bad")
    for k in ("R", "G", "B"):
        gp.backup([k], 0.5)          # identical Q, so only the prior can break the tie
    _check(gp.select().node.key == "G",
           "with equal Q, PUCT descends to the historically better mechanism")
    gu = MonteCarloGraphSearch(prior=None, widen_k=0.0, widen_alpha=0.0)
    for key, mech, parent in (("R", None, None), ("G", "good", "R"), ("B", "bad", "R")):
        gu.observe(key=key, kernel_name=key, kernel_path=f"/{key}", value=1.0,
                   parent_key=parent, mechanism=mech)
    for k in ("R", "G", "B"):
        gu.backup([k], 0.5)
    _check(gu.select() is not None, "selection still works with no prior loaded (UCT)")
    _check(gp.stats()["prior_mechanisms"] == 3 and gu.stats()["prior_mechanisms"] == 0,
           "stats report whether a prior is in play")

    print("[mcgs] graph mechanics")
    g = MonteCarloGraphSearch(max_depth=3, widen_k=1.0, widen_alpha=0.5)
    g.observe(key="S", kernel_name="seed", kernel_path="/s.py", value=1.0)
    _check(g.root == "S", "first parentless observation becomes the root")
    sel = g.select()
    _check(sel is not None and sel.node.key == "S", "root selected when alone")
    g.backup(sel.path, reward_from_gain(0.0))
    _check(g.nodes["S"].N == 1, "backup increments the visit count")

    g.observe(key="A", kernel_name="k1", kernel_path="/a.py", value=1.02,
              parent_key="S", mechanism="vectorize")
    g.backup(["S", "A"], reward_from_gain(2.0))
    g.observe(key="B", kernel_name="k2", kernel_path="/b.py", value=0.98,
              parent_key="S", mechanism="shared tile")
    g.backup(["S", "B"], reward_from_gain(-2.0))
    _check(g.nodes["S"].N == 3, "the parent accrues a visit per child evaluated")
    _check(g.nodes["A"].q() > g.nodes["B"].q(), "the winning state has the higher Q")

    # the transposition: a different edit order lands on an existing state
    n_before = len(g.nodes)
    g.observe(key="A", kernel_name="k3", kernel_path="/a2.py", value=1.05,
              parent_key="B", mechanism="vectorize")
    _check(len(g.nodes) == n_before, "a transposition creates NO new state")
    _check(len(g.nodes["A"].members) == 2, "both kernels are members of that state")
    _check(g.nodes["A"].rep == "k3", "the state's representative is its best member")
    _check("B" in g.nodes["A"].parents and "S" in g.nodes["A"].parents,
           "the merged state has both parents -- it is a DAG, not a tree")

    print("[mcgs] unrunnable states are not destinations")
    g.observe(key="Z", kernel_name="broken", kernel_path=None, value=0.0,
              parent_key="A", runnable=False, mechanism="rewrite")
    g.backup(["S", "A", "Z"], 0.0, failed=True)
    _check(g.nodes["A"].failures == 1, "the parent is charged for a failed child")
    for _ in range(12):
        s = g.select()
        _check(s.node.key != "Z", "a broken state is never selected for expansion")

    print("[mcgs] depth cap")
    g2 = MonteCarloGraphSearch(max_depth=2, widen_k=0.0, widen_alpha=0.0)
    g2.observe(key="r", kernel_name="s", kernel_path="/s", value=1.0)
    g2.observe(key="d1", kernel_name="a", kernel_path="/a", value=1.1, parent_key="r")
    g2.observe(key="d2", kernel_name="b", kernel_path="/b", value=1.2, parent_key="d1")
    for k in ("r", "d1", "d2"):
        g2.backup([k], 0.9)
    s = g2.select()
    _check(s.node.depth <= 2, f"selection stops at max_depth (got depth {s.node.depth})")

    print("[mcgs] round-trip through the checkpoint")
    blob = json.loads(json.dumps(g.to_dict()))
    g3 = MonteCarloGraphSearch.from_dict(blob)
    _check(g3.root == g.root and len(g3.nodes) == len(g.nodes), "nodes survive a round trip")
    _check(g3.nodes["A"].N == g.nodes["A"].N, "visit counts survive")
    _check(abs(g3.nodes["A"].q() - g.nodes["A"].q()) < 1e-12, "Q is reproduced exactly")
    _check(g3.best().rep == g.best().rep, "the best state is unchanged")

    print("[mcgs] sibling context is outcome-labelled")
    ctx = g.siblings_context(g.nodes["S"])
    _check("vectorize" in ctx and "shared tile" in ctx, "lists what was tried")
    ctx_a = g.siblings_context(g.nodes["A"])
    _check("FAILED" in ctx_a, "marks the failure as a failure")

    print("\n[mcgs] stats:", json.dumps(g.stats(), indent=None))
    print("[mcgs] all self-tests passed")
