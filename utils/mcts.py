"""Monte Carlo Tree Search over kernel states, replacing the one-way ratchet.

Why a TREE -- and why this was a graph until it was measured
------------------------------------------------------------
The ratchet in `main_memory_latest.py` keeps exactly one node -- `base_kernel` --
and branches from it forever. Every candidate it rejects becomes a node that is
visited once and never returned to. Measured on the 18 saved trees under `run/`:
169 nodes, 113 of them (66.9%) never used as a parent. Revisiting those nodes is
the whole reason a search sits here at all.

The first version of this module was a GRAPH (`utils/mcgs.py`). The argument for
it was a budget argument: generation is ~95% of the wall clock (16-20 min per
draw), so a 10-hour run buys ~30-40 node evaluations TOTAL, and UCT's statistics
need many visits per node to mean anything. Kernel edits ought to commute --
{vectorize, fuse} applied in either order ought to land on the same structure --
and pooling those transpositions into ONE state with two visits was the only
mechanism available that raises N per node without spending more evaluations.

It did not happen. Two independent measurements, both against the graph:

* REPLAY of all 145 scored kernels in `run/` with readable sources. Under
  ``mechanisms`` keying: 85 states, 1.71 kernels/state -- but only 2 states ever
  involved more than one ORDERING, and both were an artifact of set-based keying
  collapsing "X then X again" into {X}. Genuine commuting transpositions were
  ~absent. The pooling that did occur was re-derivation of an identical recipe,
  which is deduplication, not graph structure.
* THE GRAPHS THE LOOP ACTUALLY BUILT. Across the five `checkpoint.json` files
  under `run/` that carry a search state -- 16 nodes in total -- there are zero
  nodes with more than one parent, zero states holding more than one kernel, zero
  merges refused by the value guard, and zero back-edges. Every graph the graph
  search ever wrote to disk was already a tree.

So the merging never fired in production, and the structure it needed was paid
for anyway:

* ``observe()`` carried a merge path plus a ``merge_tolerance`` guard that split a
  state back apart when the pooled values disagreed -- machinery whose only job
  was to contain an abstraction that turned out never to abstract.
* ``select()`` carried a trajectory mask, because a merge can install a back-edge
  (a kernel produced from B whose state key is an ancestor A gives A a child B
  and B a child A). Before that guard the walk was measured as `RABABABAB...`
  with `path` growing ~20 MB/s until the process was killed -- no result, no
  error, no GPU work.
* The quick path needed a reachability check before every merge, to refuse an
  edge that would close a cycle in the host graph.
* `pathmemory.broadcast_credit` needed a three-colour DFS to tolerate cycles, and
  `principal_variation` had to descend from the root rather than walk up from the
  best node, because "the parent" of a merged state is not well defined.
* When the tolerance guard DID split, the caller reused the key it had passed in
  rather than the one the node landed under -- so `backup()` silently skipped the
  child and the next line raised KeyError.

A tree deletes all five, and gives up nothing that was ever collected. What
remains of the dedup argument is served where it belongs: `siblings_context`
tells the model what has already been tried from this node, so a recipe is not
re-derived a third time because the history block was mtime-ordered and global.

What a node is
--------------
One node, one kernel. `observe()` never merges: a kernel that arrives is a fresh
child of the node it was branched from, and node identity is a mint-on-write id
rather than an abstraction over the kernel's content. There is no `state_key`
mode to choose and nothing to tune.

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
discards losses at zero cost. Mean-backup would bury a node that produced one
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
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _norm_code(code: str) -> str:
    """Source with comments and whitespace removed.

    No longer a state key -- there are no state keys. Kept because the quick
    path content-addresses a rollout by ``sha1(_norm_code(source))`` to make
    journal replay idempotent, and a reformatting-only difference must not read
    as a different kernel there either.
    """
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"#.*?$", "", code, flags=re.M)
    return re.sub(r"\s+", "", code)


def reward_from_gain(rel_pct: Optional[float], *, scale: float = 3.0,
                     failed: bool = False) -> float:
    """Map a paired relative gain (%) onto a bounded reward in [0, 1].

    UCB's exploration constant is only interpretable against a bounded reward, so
    an unbounded speedup ratio cannot be used directly. tanh gives a smooth,
    saturating map that keeps the interesting range linear-ish: at scale=3,
    0% -> 0.50, +1% -> 0.58, +3% -> 0.88, +10% -> 1.00, and symmetrically below.

    A failed candidate scores 0 rather than 0.5. It is strictly worse than a
    measured no-op: it consumed a full evaluation and produced nothing, and a
    node that keeps emitting uncompilable code should fall out of contention on
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
    because the code-feature vector carries about two bits: over the same 145
    kernels, `is_aligned_vector_access` and `is_pointwise` never vary at all, and
    has_reuse / has_shared_memory_tile / cudagraph_eligible are ~constant. The
    simple global average is the useful object, so that is what this is.

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
        # COUNT, not set membership. A set saturates: once a mechanism is on the
        # path its prior is damped by exactly one factor no matter how many times
        # the path applies it again, so the damped weights freeze and the argmax
        # can never change however deep the walk goes. That is what lets a walk
        # ride a single mechanism into a local optimum. Compounding the penalty
        # per occurrence keeps the pressure growing with every repeat.
        # `path_mechanisms` already preserves repeats for exactly this reason.
        counts: Dict[str, int] = {}
        for m in already_applied:
            m = str(m).strip()
            if m:
                counts[m] = counts.get(m, 0) + 1
        adv = []
        for m in mechanisms:
            a = self.advantage(m)
            c = counts.get(str(m).strip(), 0) if m else 0
            if c and a > 0:
                a = a * (self.repeat_penalty ** c)
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
class TreeNode:
    """One kernel in the search tree, and the statistics of its subtree.

    One node holds exactly ONE kernel. The graph version held a `members` list
    and elected a representative from it, because a state was an abstraction that
    several kernels could satisfy; on a tree there is nothing to represent.

    Two different numbers live here and are easy to confuse:

    * ``value`` -- the kernel's CHAINED VERIFIED SPEEDUP, absolute, on the same
      basis as the number the run reports. ``parent.value * (1 + rel_pct/100)``,
      anchored at the seed. This is what selection hands the model as a parent
      and what `best()` ranks on.
    * ``q(lam)`` -- the SEARCH's value estimate for this node, in [0, 1]: a
      max-dominant mix of the rewards backed up through it. A function of gains
      RELATIVE TO A PARENT, so it says how much this line has improved, never how
      good the kernel is in absolute terms. A line climbing +50% off a bad seed
      outranks one inching +4% off a good one on `q` while being far worse on
      `value`. See `utils.pathmemory` for the `credit` field that closes the gap.
    """
    key: str
    depth: int = 0
    # The kernel this node IS. `kernel` is its name (the code file's stem), which
    # is what the checkpoint can carry; `kernel_path` resolves it back to source.
    kernel: Optional[str] = None
    kernel_path: Optional[str] = None
    value: float = 0.0                # chained verified speedup; see the docstring
    # Visit statistics.
    N: int = 0
    W: float = 0.0                    # sum of rewards, for the mean term
    M: float = 0.0                    # max reward, for the max term
    failures: int = 0                 # children that never ran
    # A tree, so exactly one parent. None iff this node is the root.
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    # What has already been tried FROM here, so expansion can be told to do
    # something else. The loop's history block was mtime-ordered and global,
    # which is how one run re-derived the same migration three times.
    tried: List[Dict[str, Any]] = field(default_factory=list)
    runnable: bool = True
    # The method_name of the edge that produced this node, i.e. the edit applied
    # to the parent to get here. `tried` cannot serve: it lists what was
    # attempted FROM here, not what led TO here.
    via: Optional[str] = None
    # Broadcast credit: the best `value` anywhere in this node's subtree, and
    # which node holds it. Written by utils.pathmemory.broadcast_credit, NOT by
    # backup(). Backup answers "how did this edit score?"; credit answers "how
    # high has this line ever reached?", which is the question a search has to
    # answer to decide whether to keep developing a line. Zero until the first
    # broadcast, so a tree that never calls it behaves exactly as before.
    credit: float = 0.0
    credit_key: Optional[str] = None

    def q(self, lam: float = 0.7) -> float:
        if self.N <= 0:
            return 0.0
        return (1.0 - lam) * (self.W / self.N) + lam * self.M

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "depth": self.depth,
            "kernel": self.kernel, "kernel_path": self.kernel_path,
            "value": self.value,
            "N": self.N, "W": self.W, "M": self.M, "failures": self.failures,
            "parent": self.parent, "children": self.children,
            "tried": self.tried, "runnable": self.runnable, "via": self.via,
            "credit": self.credit, "credit_key": self.credit_key,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeNode":
        n = cls(key=str(d["key"]))
        n.depth = int(d.get("depth") or 0)
        # `rep`/`rep_path`/`rep_value` are the graph-era spellings; a checkpoint
        # written by utils/mcgs.py is read here rather than rejected. `members`
        # is dropped on purpose -- see MonteCarloTreeSearch.from_dict.
        n.kernel = d.get("kernel", d.get("rep"))
        n.kernel_path = d.get("kernel_path", d.get("rep_path"))
        n.value = float(d.get("value", d.get("rep_value")) or 0.0)
        n.N = int(d.get("N") or 0)
        n.W = float(d.get("W") or 0.0)
        n.M = float(d.get("M") or 0.0)
        n.failures = int(d.get("failures") or 0)
        parent = d.get("parent")
        if parent is None:
            # Graph schema: a list. Provisional only -- from_dict re-derives every
            # parent by BFS from the root so the result is a tree by construction.
            ps = d.get("parents") or []
            parent = ps[0] if ps else None
        n.parent = parent
        n.children = list(d.get("children") or [])
        n.tried = list(d.get("tried") or [])
        n.runnable = bool(d.get("runnable", True))
        n.via = d.get("via")
        # Absent in checkpoints written before path memory existed; 0.0/None is
        # the same state as "never broadcast", and the next broadcast refills it.
        n.credit = float(d.get("credit") or 0.0)
        n.credit_key = d.get("credit_key")
        return n


@dataclass
class Selection:
    """What the search chose, and why -- the `why` is printed every round.

    A search whose choices are not legible cannot be debugged: the ratchet at
    least printed "[base] Keeping base_kernel ...". This carries the equivalent.
    """
    node: TreeNode
    path: List[str]
    reason: str


class MonteCarloTreeSearch:
    """UCT over a tree of kernels, with progressive widening.

    Progressive widening rather than a fixed branching factor because the action
    space is LLM-generated and unbounded -- there is no enumerable move list to
    take an argmax over. A node earns its k-th child only after enough visits to
    suggest it is worth developing: ``len(children) < ceil(k * N**alpha)``.
    """

    def __init__(self, *, c_puct: float = 0.8, lam: float = 0.7,
                 widen_k: float = 1.0, widen_alpha: float = 0.5,
                 max_depth: int = 10, reward_scale: float = 3.0,
                 prior: Optional[MechanismPrior] = None,
                 c_prior: float = 1.0,
                 epsilon: float = 0.0,
                 rng_seed: int = 0,
                 pv_bonus: float = 0.0) -> None:
        self.nodes: Dict[str, TreeNode] = {}
        self.root: Optional[str] = None
        self.c_puct = float(c_puct)
        self.lam = float(lam)
        self.widen_k = float(widen_k)
        self.widen_alpha = float(widen_alpha)
        self.max_depth = int(max_depth)
        self.reward_scale = float(reward_scale)
        self.prior = prior
        self.c_prior = float(c_prior)
        # Probability of taking a random child instead of the argmax at a given
        # descent step (Czech et al. 2021 use an epsilon-greedy trajectory for
        # the same purpose: escaping a local optimum the argmax cannot leave).
        # Default 0.0 -- OFF. It is a behavioural change to the search policy and
        # this repo's convention is that such a change is opt-in until it has won
        # its own A/B, exactly as --mcts_prior is.
        self.epsilon = float(epsilon)
        self.rng_seed = int(rng_seed)
        # Selection bonus applied to children on the principal variation -- the
        # chain of edits that actually reached the best kernel. Default 0.0 = OFF:
        # it changes the search policy, and the convention here is that such a
        # change is opt-in until it wins its own A/B, exactly as --mcts_prior and
        # --mcts_epsilon are. `pv` is the PV membership set, refilled by
        # utils.pathmemory.broadcast_credit; empty means "no bonus to apply", so
        # the term vanishes on a tree that never broadcasts.
        self.pv_bonus = float(pv_bonus)
        self.pv: set = set()
        self.total_visits = 0
        # Monotonic node-id counter. Node identity is minted, not derived from
        # the kernel's content: the graph derived it (that was the merge
        # criterion) and nothing needs that any more. Persisted so a --resume
        # keeps minting fresh ids instead of colliding with restored ones.
        self._next_id = 0
        # Set by from_dict when a graph-era checkpoint had to be flattened.
        # Printed once by the caller; carries no behaviour.
        self.migration_note: str = ""

    # ---------------------------------------------------------------- ingestion
    def _mint(self) -> str:
        while True:
            key = f"n{self._next_id:04d}"
            self._next_id += 1
            if key not in self.nodes:
                return key

    def observe(self, *, kernel_name: str, kernel_path: Optional[str],
                value: float, parent_key: Optional[str] = None,
                runnable: bool = True,
                mechanism: Optional[str] = None,
                note: str = "",
                key: Optional[str] = None) -> TreeNode:
        """Record a kernel as a NEW node under *parent_key*.

        Never merges. The graph version keyed a kernel by an abstraction over its
        content and pooled it into any existing node with the same key; that path
        is gone, along with the `merge_tolerance` guard that existed to contain
        it when the abstraction was wrong. Two kernels that happen to apply the
        same recipe are now two nodes, which is what they always were in every
        graph the loop actually built.

        *key* is accepted so a caller can pin an id (restore, or a test fixture).
        If it is already taken a fresh one is minted instead -- silently reusing
        it would be the merge this class exists to not do.
        """
        if key is None or key in self.nodes:
            key = self._mint()
        parent = self.nodes.get(parent_key) if parent_key else None
        node = TreeNode(key=key,
                        depth=(parent.depth + 1) if parent is not None else 0,
                        via=mechanism)
        node.kernel = kernel_name
        node.kernel_path = kernel_path
        node.value = float(value)
        node.runnable = bool(runnable)
        node.parent = parent.key if parent is not None else None
        self.nodes[key] = node
        if parent is not None:
            if key not in parent.children:
                parent.children.append(key)
        elif self.root is None:
            self.root = key
        if mechanism or note:
            entry = {"mechanism": mechanism, "child": kernel_name,
                     "note": note, "value": float(value), "runnable": bool(runnable)}
            if parent is not None:
                parent.tried.append(entry)
        return node

    def backup(self, path: Sequence[str], reward: float, *, failed: bool = False) -> None:
        """Propagate a leaf result up every node on the selected path."""
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
    def _widen_budget(self, n: TreeNode) -> int:
        return max(1, math.ceil(self.widen_k * (max(n.N, 1) ** self.widen_alpha)))

    def _selectable_children(self, n: TreeNode) -> List[TreeNode]:
        out = []
        for k in n.children:
            c = self.nodes.get(k)
            # A node with no runnable kernel cannot be branched from: there is no
            # code to hand the model. It keeps its visits (so its parent is
            # penalised for producing it) but is not itself a destination.
            if c is not None and c.runnable and c.kernel is not None:
                out.append(c)
        return out

    def path_to(self, key: str) -> Optional[List[str]]:
        """Root-to-*key* path, by walking `parent` up.

        Exact on a tree and O(depth). The graph version needed a BFS from the
        root for this, and could only report *a* route rather than *the* route,
        because a merged node had several parents and the search had walked only
        one of them.
        """
        if key not in self.nodes:
            return None
        out: List[str] = []
        cur: Optional[str] = key
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.append(cur)
            n = self.nodes.get(cur)
            cur = n.parent if n is not None else None
        out.reverse()
        return out

    def select(self) -> Optional[Selection]:
        """Walk from the root to the node that should be expanded next.

        On a tree the descent terminates by construction -- every step strictly
        increases depth and `observe` only ever attaches a fresh node to an
        existing one, so a cycle cannot be built. The graph version needed a
        trajectory mask here to stay acyclic; there is nothing left to mask.

        The step cap below is not that guard. It is a cheap backstop against a
        corrupted checkpoint, because the failure mode it prevents was measured
        and is unusually nasty: a walk that loops grows `path` at ~20 MB/s and
        takes the process down with no result, no error and no GPU work. It can
        never fire on a tree this class built.
        """
        if not self.root or self.root not in self.nodes:
            return None
        path: List[str] = []
        node = self.nodes[self.root]
        for _ in range(len(self.nodes) + 1):
            path.append(node.key)
            kids = self._selectable_children(node)
            if node.depth >= self.max_depth:
                return Selection(node, path,
                                 f"depth {node.depth} reached --max_depth "
                                 f"{self.max_depth}; expanding here rather than deeper")
            if not kids:
                return Selection(node, path, "leaf node (no expandable children yet)")
            budget = self._widen_budget(node)
            if len(kids) < budget:
                return Selection(node, path,
                                 f"progressive widening: {len(kids)} child node(s) < "
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
            # Principal-variation bonus (OFF unless --mcts_pv_bonus > 0). Q is an
            # average over what a child has SCORED; it says nothing about whether
            # that child is the one the run's best kernel actually descends from.
            # A line can hold the record and still lose the argmax to a sibling
            # with a better mean, at which point the search stops developing the
            # only chain known to reach the top. This term is the correction, and
            # it is a constant rather than a function of N so it biases the choice
            # without ever swamping a measured Q gap.
            def _pv(c: TreeNode) -> float:
                return self.pv_bonus if (self.pv_bonus and c.key in self.pv) else 0.0

            if self.prior is not None and self.prior.table:
                applied = self.path_mechanisms(path)
                P = self.prior.weights([c.via for c in kids], already_applied=applied)
                sqrtN = math.sqrt(max(node.N, 1))
                best, best_u = None, -float("inf")
                for c, p in zip(kids, P):
                    u = c.q(self.lam) + self.c_prior * p * sqrtN / (1 + c.N) + _pv(c)
                    if u > best_u:
                        best, best_u = c, u
            else:
                logN = math.log(max(node.N, 1))
                best, best_u = None, -float("inf")
                for c in kids:
                    u = c.q(self.lam) + self.c_puct * math.sqrt(logN / max(c.N, 1)) + _pv(c)
                    if u > best_u:
                        best, best_u = c, u
            # Epsilon-greedy escape (OFF unless --mcts_epsilon > 0). The argmax
            # above is deterministic within a call, so a node whose Q is merely
            # the best SEEN keeps being re-entered and its rivals never get the
            # visits that would correct them. Czech et al. (2021) add a random
            # exploration trajectory for this. Seeded from (rng_seed,
            # total_visits, depth) rather than global RNG state so a --resume at
            # the same total_visits draws the same value and a replayed run is
            # reproducible -- checkpoints store no RNG state.
            if self.epsilon > 0.0 and len(kids) > 1:
                # A str seed, not a tuple (unsupported) and not hash() (salted by
                # PYTHONHASHSEED, so it would differ between processes). Python
                # derives a str seed via SHA-512, which is stable across runs.
                rnd = random.Random(f"{self.rng_seed}:{self.total_visits}:{len(path)}")
                if rnd.random() < self.epsilon:
                    pick = rnd.choice(kids)
                    path.append(pick.key)
                    return Selection(pick, path,
                                     f"epsilon-greedy: took {pick.key} at random over the "
                                     f"argmax (epsilon={self.epsilon:g})")
            if best is None:
                return Selection(node, path, "no child scored; expanding here")
            node = best
        # Unreachable on a tree; see the docstring.
        return Selection(node, path, "step cap reached (corrupt tree?); expanding here")

    # ------------------------------------------------------------------ context
    def siblings_context(self, node: TreeNode, *, limit: int = 8) -> str:
        """What has already been tried from this node, with outcomes.

        Fed to the optimization prompt in place of the mtime-ordered global
        history. Outcome-labelled on purpose: a history that lists attempts
        without saying which failed invites the model to propose them again.
        This is also what remains of the graph's dedup argument -- the merging
        never fired, but "you already tried this here, and it lost" is the part
        of it that was load-bearing, and it belongs in the prompt anyway.
        """
        if not node.tried:
            return "(nothing has been tried from this kernel yet)"
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

        Consumed by the PUCT prior's repeat damping. It must come from the edges
        that LED to each node (`via`), not from a node's `tried` list -- `tried`
        records what was attempted FROM a node, so using it would report a node's
        siblings' attempts as its own lineage.
        """
        out: List[str] = []
        for k in path:
            n = self.nodes.get(k)
            if n is not None and n.via:
                out.append(n.via)
        return out

    def best(self) -> Optional[TreeNode]:
        """The node holding the highest chained verified value."""
        cands = [n for n in self.nodes.values() if n.kernel is not None]
        return max(cands, key=lambda n: n.value) if cands else None

    def stats(self) -> Dict[str, Any]:
        n_nodes = len(self.nodes)
        n1 = sum(1 for n in self.nodes.values() if n.N <= 1)
        leaves = sum(1 for n in self.nodes.values() if not n.children)
        return {
            "nodes": n_nodes,
            "leaves": leaves,
            "nodes_at_N<=1": n1,
            "mean_N": (sum(n.N for n in self.nodes.values()) / n_nodes) if n_nodes else 0.0,
            "total_visits": self.total_visits,
            "max_depth_seen": max((n.depth for n in self.nodes.values()), default=0),
            "prior_mechanisms": len(self.prior.table) if self.prior else 0,
        }

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "root": self.root,
            "total_visits": self.total_visits,
            "next_id": self._next_id,
            "params": {
                "c_puct": self.c_puct, "lam": self.lam, "widen_k": self.widen_k,
                "widen_alpha": self.widen_alpha, "max_depth": self.max_depth,
                "reward_scale": self.reward_scale,
                "c_prior": self.c_prior,
                "epsilon": self.epsilon,
                "rng_seed": self.rng_seed,
                "pv_bonus": self.pv_bonus,
            },
            # Persisted so a resumed run selects with the SAME prior it started
            # with. Re-reading the prior file on resume would silently change the
            # search policy mid-run if the file had been refitted since.
            "prior": self.prior.to_dict() if self.prior is not None else None,
            "nodes": {k: n.to_dict() for k, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MonteCarloTreeSearch":
        """Rebuild from a checkpoint, including one written by the graph search.

        A version-1 blob is `utils/mcgs.py`'s. It is read rather than rejected,
        because the visit counts are the only part of the search that cannot be
        recomputed from the artifacts on disk, and a --resume that dropped them
        would restart exploration from scratch with the budget already spent.

        Flattening rule, and what it costs
        ----------------------------------
        BFS from the root over `children`, keeping for each node the parent that
        REACHES IT FIRST. That is the shallowest route, which is exactly the
        depth the graph itself recorded (`observe` kept `min(depth, p.depth+1)`)
        and the route `max_depth` was always measured against -- so a flattened
        tree selects the same way the graph did.

        Three things are dropped, all of them measured to be empty on every graph
        this loop ever wrote (5 checkpoints, 16 nodes: zero multi-parent nodes,
        zero multi-kernel states, zero refused splits, zero back-edges):

        * extra edges: every parent but the first, and with them any back-edge.
          BFS keeps one route in and drops the edge, never the node;
        * extra `members` of a merged state. The node keeps its representative,
          which is the only member the search could ever branch from and the one
          every statistic on the node was already about;
        * nodes unreachable from the root. `select()` descends `children` from
          the root only, so these were already invisible to it -- `best()` and
          `stats()` counted them, which is how a run could announce a best state
          it could not branch from.

        Whatever it drops is reported in `migration_note` rather than in silence.
        """
        p = (d or {}).get("params") or {}
        g = cls(c_puct=float(p.get("c_puct", 0.8)), lam=float(p.get("lam", 0.7)),
                widen_k=float(p.get("widen_k", 1.0)),
                widen_alpha=float(p.get("widen_alpha", 0.5)),
                max_depth=int(p.get("max_depth", 10)),
                reward_scale=float(p.get("reward_scale", 3.0)),
                c_prior=float(p.get("c_prior", 1.0)),
                # Defaulted, so a checkpoint written before these existed resumes
                # with epsilon off rather than KeyError-ing.
                epsilon=float(p.get("epsilon", 0.0)),
                rng_seed=int(p.get("rng_seed", 0)),
                pv_bonus=float(p.get("pv_bonus", 0.0)))
        if not d:
            return g
        g.prior = MechanismPrior.from_dict(d.get("prior"))
        g.root = d.get("root")
        g.total_visits = int(d.get("total_visits") or 0)
        raw = {k: TreeNode.from_dict(nd) for k, nd in (d.get("nodes") or {}).items()}

        legacy = int(d.get("version") or 1) < 2
        dropped_members = dropped_edges = 0
        if legacy:
            for k, nd in (d.get("nodes") or {}).items():
                dropped_members += max(0, len(nd.get("members") or []) - 1)

        # BFS from the root: first reach wins, which is the shallowest route.
        order: List[str] = []
        if g.root and g.root in raw:
            raw[g.root].parent = None
            raw[g.root].depth = 0
            seen = {g.root}
            queue = [g.root]
            while queue:
                k = queue.pop(0)
                order.append(k)
                n = raw[k]
                kept: List[str] = []
                for c in n.children:
                    if c in raw and c in seen:
                        # A second route into an already-reached node, or a
                        # back-edge. Counted so the note can report it: dropping
                        # an edge in silence is how a resumed run quietly stops
                        # being the tree the operator thinks it is.
                        dropped_edges += 1
                    if c in raw and c not in seen:
                        seen.add(c)
                        raw[c].parent = k
                        raw[c].depth = n.depth + 1
                        kept.append(c)
                        queue.append(c)
                # Edges to already-seen nodes are back-edges or second routes into
                # a merged node. Dropping the EDGE (not the node) is what makes the
                # result a tree; the node keeps whichever route reached it first.
                n.children = kept
            g.nodes = {k: raw[k] for k in order}
        else:
            # No usable root: keep everything as-is rather than throwing the
            # checkpoint away. `select()` returns None until a root is observed.
            g.nodes = raw

        unreachable = len(raw) - len(g.nodes)
        # Fresh ids must not collide with restored ones, whatever their spelling
        # (graph-era keys are content hashes like "m:1f3a...", not "nNNNN").
        g._next_id = int(d.get("next_id") or 0)
        for k in g.nodes:
            if k.startswith("n") and k[1:].isdigit():
                g._next_id = max(g._next_id, int(k[1:]) + 1)

        if legacy:
            bits = [f"restored a graph-era checkpoint ({len(raw)} nodes)"]
            if dropped_edges:
                bits.append(f"{dropped_edges} extra parent edge(s)/back-edge(s) dropped")
            if dropped_members:
                bits.append(f"{dropped_members} merged kernel(s) dropped "
                            f"(representatives kept)")
            if unreachable:
                bits.append(f"{unreachable} node(s) unreachable from the root dropped")
            if not (dropped_edges or dropped_members or unreachable):
                bits.append("it was already a tree; nothing dropped")
            g.migration_note = "; ".join(bits)
        return g


# --------------------------------------------------------------------- self-test
if __name__ == "__main__":
    def _check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        assert cond, msg

    print("[mcts] reward mapping")
    _check(abs(reward_from_gain(0.0) - 0.5) < 1e-9, "0% -> 0.50")
    _check(reward_from_gain(3.0) > 0.85, "+3% -> >0.85")
    _check(reward_from_gain(-3.0) < 0.15, "-3% -> <0.15")
    _check(reward_from_gain(None, failed=True) == 0.0, "failed -> 0.0")
    _check(reward_from_gain(0.4) < reward_from_gain(1.2), "monotone in the gain")

    print("[mcts] observe() never merges")
    gm = MonteCarloTreeSearch()
    r = gm.observe(kernel_name="seed", kernel_path="/s", value=1.0)
    a = gm.observe(kernel_name="k1", kernel_path="/a", value=1.05,
                   parent_key=r.key, mechanism="vectorize")
    b = gm.observe(kernel_name="k2", kernel_path="/b", value=0.98,
                   parent_key=r.key, mechanism="shared_tile")
    # The graph's transposition: same recipe, different order. Two nodes now.
    c1 = gm.observe(kernel_name="k3", kernel_path="/c", value=1.06,
                    parent_key=a.key, mechanism="shared_tile")
    c2 = gm.observe(kernel_name="k4", kernel_path="/d", value=1.04,
                    parent_key=b.key, mechanism="vectorize")
    _check(c1.key != c2.key, "commuting edits produce TWO nodes, not one merged state")
    _check(len(gm.nodes) == 5, "every observation is its own node")
    _check(gm.observe(kernel_name="x", kernel_path="/x", value=1.0,
                      parent_key=r.key, key=a.key).key != a.key,
           "a key that is already taken is not reused -- a fresh one is minted")
    _check(all(len(n.children) == len(set(n.children)) for n in gm.nodes.values()),
           "no duplicate child edges")

    print("[mcts] every node has exactly one parent, and the root has none")
    _check(gm.nodes[gm.root].parent is None, "the root has no parent")
    _check(all(n.parent in gm.nodes for k, n in gm.nodes.items() if k != gm.root),
           "every non-root parent resolves to a real node")
    _check(gm.nodes[c1.key].parent == a.key and gm.nodes[c2.key].parent == b.key,
           "each transposition keeps its own lineage")

    print("[mcts] path_to walks up the parents exactly")
    _check(gm.path_to(c1.key) == [r.key, a.key, c1.key],
           f"root-to-node path is exact (got {gm.path_to(c1.key)})")
    _check(gm.path_to("nope") is None, "an unknown key has no path")
    _check(gm.path_mechanisms(gm.path_to(c1.key)) == ["vectorize", "shared_tile"],
           "path_mechanisms reads the `via` edges root-to-leaf, in order")

    print("[mcts] the descent always terminates and is a simple path")
    for depth_cap in (0, 1, 3, 10):
        g_t = MonteCarloTreeSearch(max_depth=depth_cap, widen_k=0.0, widen_alpha=0.0)
        prev = g_t.observe(kernel_name="s", kernel_path="/s", value=1.0)
        for i in range(6):
            prev = g_t.observe(kernel_name=f"k{i}", kernel_path=f"/{i}", value=1.0 + i,
                               parent_key=prev.key, mechanism=f"m{i}")
        for n in g_t.nodes.values():
            n.N, n.W, n.M = 1, 0.5, 0.6
        s = g_t.select()
        _check(s is not None, f"select() returns at max_depth={depth_cap}")
        _check(len(s.path) == len(set(s.path)), "the trajectory is a simple path")
        _check(s.node.depth <= depth_cap, f"selection stops at max_depth={depth_cap}")

    print("[mcts] tree mechanics")
    g = MonteCarloTreeSearch(max_depth=3, widen_k=1.0, widen_alpha=0.5)
    seed = g.observe(kernel_name="seed", kernel_path="/s.py", value=1.0)
    _check(g.root == seed.key, "the first parentless observation becomes the root")
    sel = g.select()
    _check(sel is not None and sel.node.key == seed.key, "root selected when alone")
    g.backup(sel.path, reward_from_gain(0.0))
    _check(g.nodes[seed.key].N == 1, "backup increments the visit count")

    na = g.observe(kernel_name="k1", kernel_path="/a.py", value=1.02,
                   parent_key=seed.key, mechanism="vectorize")
    g.backup([seed.key, na.key], reward_from_gain(2.0))
    nb = g.observe(kernel_name="k2", kernel_path="/b.py", value=0.98,
                   parent_key=seed.key, mechanism="shared tile")
    g.backup([seed.key, nb.key], reward_from_gain(-2.0))
    _check(g.nodes[seed.key].N == 3, "the parent accrues a visit per child evaluated")
    _check(na.q() > nb.q(), "the winning node has the higher Q")
    _check(g.best().kernel == "k1", "best() ranks on the chained value")

    print("[mcts] unrunnable nodes are not destinations")
    nz = g.observe(kernel_name="broken", kernel_path=None, value=0.0,
                   parent_key=na.key, runnable=False, mechanism="rewrite")
    g.backup([seed.key, na.key, nz.key], 0.0, failed=True)
    _check(g.nodes[na.key].failures == 1, "the parent is charged for a failed child")
    for _ in range(12):
        _check(g.select().node.key != nz.key, "a broken node is never selected")

    print("[mcts] repeat damping compounds instead of saturating")
    pr = MechanismPrior.fit([("m1", 1.0)] * 3 + [("m2", 15.0)] * 3 + [("m9", 10.0)] * 3)
    w0 = pr.weights(["m2", "m9"], already_applied=["m1"])[0]
    w1 = pr.weights(["m2", "m9"], already_applied=["m1", "m2"])[0]
    w2 = pr.weights(["m2", "m9"], already_applied=["m1", "m2", "m1", "m2"])[0]
    _check(w1 < w0, "one application of a mechanism damps its prior")
    _check(w2 < w1, "a SECOND application damps it further (set membership would freeze)")

    print("[mcts] epsilon-greedy is off by default, reproducible when on")
    _check(MonteCarloTreeSearch().epsilon == 0.0, "epsilon defaults to off")

    def _fan(**kw) -> MonteCarloTreeSearch:
        c = MonteCarloTreeSearch(widen_k=0.0, widen_alpha=0.0, **kw)
        rt = c.observe(kernel_name="r", kernel_path="/r", value=1.00, key="R")
        for nm, val in (("a", 1.05), ("b", 1.06), ("c", 1.04)):
            c.observe(kernel_name=nm, kernel_path=f"/{nm}", value=val,
                      parent_key=rt.key, mechanism=f"m_{nm}")
        for nn in c.nodes.values():
            nn.N, nn.W, nn.M = 1, 0.5, 0.6
        return c
    ge1, ge2 = _fan(epsilon=1.0, rng_seed=7), _fan(epsilon=1.0, rng_seed=7)
    _check(ge1.select().path == ge2.select().path, "same seed -> same trajectory")
    _check(len(ge1.select().path) == len(set(ge1.select().path)),
           "epsilon-greedy still yields a simple path")

    print("[mcts] mechanism prior")
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

    print("[mcts] PUCT selection uses the prior; UCT still runs without one")
    gp = MonteCarloTreeSearch(prior=pr, c_prior=1.0, widen_k=0.0, widen_alpha=0.0)
    rp = gp.observe(kernel_name="s", kernel_path="/s", value=1.0)
    gk = gp.observe(kernel_name="a", kernel_path="/a", value=1.0,
                    parent_key=rp.key, mechanism="good")
    bk = gp.observe(kernel_name="b", kernel_path="/b", value=1.0,
                    parent_key=rp.key, mechanism="bad")
    for k in (rp.key, gk.key, bk.key):
        gp.backup([k], 0.5)          # identical Q, so only the prior can break the tie
    _check(gp.select().node.key == gk.key,
           "with equal Q, PUCT descends to the historically better mechanism")
    gu = MonteCarloTreeSearch(prior=None, widen_k=0.0, widen_alpha=0.0)
    ru = gu.observe(kernel_name="R", kernel_path="/R", value=1.0)
    for nm, mech in (("G", "good"), ("B", "bad")):
        gu.observe(kernel_name=nm, kernel_path=f"/{nm}", value=1.0,
                   parent_key=ru.key, mechanism=mech)
    for k in list(gu.nodes):
        gu.backup([k], 0.5)
    _check(gu.select() is not None, "selection still works with no prior loaded (UCT)")
    _check(gp.stats()["prior_mechanisms"] == 3 and gu.stats()["prior_mechanisms"] == 0,
           "stats report whether a prior is in play")

    print("[mcts] round-trip through the checkpoint")
    blob = json.loads(json.dumps(g.to_dict()))
    g3 = MonteCarloTreeSearch.from_dict(blob)
    _check(g3.root == g.root and len(g3.nodes) == len(g.nodes), "nodes survive a round trip")
    _check(g3.nodes[na.key].N == g.nodes[na.key].N, "visit counts survive")
    _check(abs(g3.nodes[na.key].q() - g.nodes[na.key].q()) < 1e-12, "Q is reproduced exactly")
    _check(g3.best().kernel == g.best().kernel, "the best node is unchanged")
    _check(g3.migration_note == "", "a version-2 blob needs no migration note")
    _check(g3._next_id >= g._next_id, "the id counter does not rewind (no id reuse)")
    _check(g3.observe(kernel_name="new", kernel_path="/n", value=1.0,
                      parent_key=g3.root).key not in blob["nodes"],
           "a node minted after a restore cannot collide with a restored id")

    print("[mcts] a graph-era checkpoint is flattened, not rejected")
    # A version-1 blob with everything the graph could produce and a tree cannot:
    # a merged state with two members, a second parent, a back-edge (D -> A), and
    # a node unreachable from the root.
    legacy_blob = {
        "version": 1, "root": "R", "total_visits": 9, "splits": 2,
        "params": {"c_puct": 0.8, "lam": 0.7, "max_depth": 10,
                   "state_key_mode": "features", "merge_tolerance": 0.15},
        "prior": None,
        "nodes": {
            "R": {"key": "R", "depth": 0, "members": ["seed"], "rep": "seed",
                  "rep_path": "/seed.py", "rep_value": 1.0, "N": 4, "W": 2.0, "M": 0.6,
                  "parents": [], "children": ["A", "B"]},
            "A": {"key": "A", "depth": 1, "members": ["k1", "k1b"], "rep": "k1",
                  "rep_path": "/k1.py", "rep_value": 1.10, "N": 3, "W": 1.8, "M": 0.7,
                  "parents": ["R", "B"], "children": ["D"], "via": "vectorize"},
            "B": {"key": "B", "depth": 1, "members": ["k2"], "rep": "k2",
                  "rep_path": "/k2.py", "rep_value": 1.05, "N": 2, "W": 1.0, "M": 0.5,
                  "parents": ["R"], "children": ["A"], "via": "fuse"},
            "D": {"key": "D", "depth": 2, "members": ["k3"], "rep": "k3",
                  "rep_path": "/k3.py", "rep_value": 1.20, "N": 1, "W": 0.8, "M": 0.8,
                  "parents": ["A"], "children": ["A"], "via": "tile"},
            "ORPH": {"key": "ORPH", "depth": 0, "members": ["lost"], "rep": "lost",
                     "rep_path": "/lost.py", "rep_value": 9.99, "N": 1, "W": 0.1,
                     "M": 0.1, "parents": [], "children": []},
        },
    }
    gl = MonteCarloTreeSearch.from_dict(json.loads(json.dumps(legacy_blob)))
    _check(gl.root == "R", "the root survives")
    _check("ORPH" not in gl.nodes,
           "a node unreachable from the root is dropped, not left to win best()")
    _check(gl.best().kernel == "k3",
           "best() reports a node the search can actually branch from (not ORPH's 9.99)")
    _check(set(gl.nodes) == {"R", "A", "B", "D"}, f"the reachable set is kept (got {set(gl.nodes)})")
    _check(all((n.parent is None) == (k == "R") for k, n in gl.nodes.items()),
           "exactly one node -- the root -- has no parent")
    _check(gl.nodes["A"].parent == "R",
           "a merged node keeps the SHALLOWEST route in (R, not B)")
    _check(gl.nodes["A"].depth == 1 and gl.nodes["D"].depth == 2,
           "depths are re-derived from the kept routes")
    _check("A" not in gl.nodes["D"].children, "the back-edge D -> A is dropped")
    _check(gl.nodes["A"].kernel == "k1" and abs(gl.nodes["A"].value - 1.10) < 1e-12,
           "rep/rep_value are read under their new names")
    _q_expect = (1 - 0.7) * (1.8 / 3) + 0.7 * 0.7
    _check(gl.nodes["A"].N == 3 and abs(gl.nodes["A"].q() - _q_expect) < 1e-12,
           "visit counts and Q survive the flatten unchanged")
    _check(gl.total_visits == 9, "total_visits survives")
    for tok in ("extra parent", "merged kernel", "unreachable"):
        _check(tok in gl.migration_note,
               f"the migration note reports what was dropped ({tok!r})")
    print(f"       note: {gl.migration_note}")
    # The flattened tree must still be selectable and must terminate.
    sl = gl.select()
    _check(sl is not None and len(sl.path) == len(set(sl.path)),
           "a flattened graph still yields a simple path")
    _check(gl.path_to("D") == ["R", "A", "D"], "path_to is exact after flattening")
    # And it must round-trip forward into the new schema.
    gl2 = MonteCarloTreeSearch.from_dict(json.loads(json.dumps(gl.to_dict())))
    _check(set(gl2.nodes) == set(gl.nodes) and gl2.migration_note == "",
           "a flattened tree re-saves as version 2 and needs no second migration")

    print("[mcts] sibling context is outcome-labelled")
    ctx = g.siblings_context(g.nodes[seed.key])
    _check("vectorize" in ctx and "shared tile" in ctx, "lists what was tried")
    ctx_a = g.siblings_context(g.nodes[na.key])
    _check("FAILED" in ctx_a, "marks the failure as a failure")

    print("\n[mcts] stats:", json.dumps(g.stats(), indent=None))
    print("[mcts] all self-tests passed")
