# Adversarial methodology review — 002_vae_conv3x3_groupnorm_silu_residual_fused (RTX 5090, WSL2)

Reviewed: `t_sol.json`, `baselines.json`, all 4 candidate kernels, run traces (`out_*/traces.jsonl`,
`sanity_out/`), vendored harness (`third_party/SOL-ExecBench`, sourceless .pyc, verified byte-for-byte
against upstream git), upstream HEAD `a9fa080` ("Add v1.1 timing methodology (#18)", 2026-07-15).
All verification was pure-CPU (marshal-level .pyc comparison, Python/numpy arithmetic); no GPU touched.

## 1. T_SOL arithmetic — numbers correct, semantics flawed

**Recomputation (all 20 workloads, pure Python):** every value in `t_sol.json` matches
`max(4·256·256·9·B·H·W / 104.8e12, 2·B·256·H·W·4 / 1792e9)` to < 2e-6 relative. FLOP count for
conv3x3 pad=1 stride=1 (output H×W = input H×W) is right: 2·Cin·K² = 4608 FLOP per output element
per conv. Constants check out: 104.8 TFLOPS = 21760 CUDA cores × 2 × 2.407 GHz boost (NVIDIA's
official FP32 rate, and the dense TF32 tensor rate is the same figure); 1792 GB/s = 512-bit GDDR7 ×
28 Gbps. Ignored weight traffic (9.4 MB) and GN/SiLU FLOPs are legitimate for a lower bound.

**Compute-bound assumption:** holds trivially for ALL workloads — both terms scale with B·H·W, so
the compute/memory ratio is a shape-independent 19.7×. Smallest workload is **2×64×64** (there is no
batch-1 64×64 workload, contrary to the review tasking note): t_comp 0.184 ms vs t_mem 0.009 ms.

**CONFIRMED FLAW (F1): T_SOL is a direct-convolution bound, not a speed-of-light bound.**
- Winograd F(4×4,3×3) / FFT convolution reduces required multiplies 2.25–4× for exactly this conv
  shape; cuDNN uses Winograd routinely. A Winograd kernel can therefore beat "T_SOL".
- The tolerance regime (atol ~3e-3 vs a reference that is itself only TF32-accurate) plausibly admits
  fp16-tensor-core convs at 209.5 dense TFLOPS — 2× the assumed peak.
- `sol_score()` (upstream `sol_score.py`) does **not clamp at 1.0** when `t_k < t_sol` and
  `t_b > t_sol`: the formula returns >1. Any report should state the dtype/algorithm policy under
  which T_SOL is claimed, and treat SOL > 1 as a model violation, not a result.
  (No candidate exceeded it here; best t_b/t_sol ratios 1.36–2.02 are internally consistent.)

## 2. Input generation — two coverage holes (CONFIRMED from harness code)

Vendored `core/bench/io.py` is byte-identical to upstream `2d852a3`. Its heuristics
(`_is_norm_weight`, `_is_norm_bias` — "norm1"/"norm2" strip digits → "norm") mean:

**F2: `norm1_weight`/`norm2_weight` are always ONES, `norm*_bias` always ZEROS**, in every workload
and every correctness round. The GroupNorm affine transform is never exercised: a kernel that
ignores γ and β entirely passes 20/20. The Triton kernel's `*gamma + beta` path is untested.

**F3: eps is untestable.** Conv weights are generated as randn/√(shape[-1]) = randn/√3 (the heuristic
takes fan_in from the LAST dim, i.e. kernel_size=3, not Cin·K²=2304), so conv outputs have variance
≈ 768 ≫ eps=1e-6. Hardcoding eps=0 changes outputs by ~6.5e-10; even eps=1.0 changes them by
~6.5e-4 < atol. "Wrong eps handling" is invisible to this test by construction.

`num_groups=32` hardcoding is **legal** — it is `"type": "const", "value": 32` in `definition.json`
(the reference itself hardcodes 32); not a flaw.

## 3. Tolerance laxity (CONFIRMED in the actual run traces)

**F4: `required_matched_ratio` defaults to 0.99 and `max_error_cap` is null** (both visible in every
trace of this run). Up to 1% of output elements may be *arbitrarily* wrong. On 1×1024×1024 that is
2.68M elements. A wrong conv halo on the entire outermost pixel ring is ~0.4% of elements → passes.
Masked-tail boundary bugs — precisely the failure class of block/mask-based Triton kernels — are
undetectable below the 1% threshold on large workloads.

## 4. Triton fused GroupNorm precision — concern REFUTED for this benchmark, with caveats

- The claimed max reduction (256·256·8 = 524 288) is **wrong**: workloads include 1×768×768 and
  1×1024×1024, so the largest per-(batch,group) reduction is 1024·1024·8 = **8 388 608** elements.
- Even so, a bit-accurate CPU simulation of the actual scheme (fp32 tree-sums of 1024-element blocks,
  then 8192 sequential fp32 atomic adds) under the harness's input distribution (zero-mean, σ≈27.7)
  gives var relative error ~6e-7 → normalized-output error ~2e-6. Three orders of magnitude below
  atol. Observed worst `max_absolute_error` = 7.2e-4 (vs atol 3.2e-3, 4.4× margin) is dominated by
  error amplification through conv2, not by the reduction.
- **Caveat (F5):** the margin exists only because all inputs are zero-mean `randn`. The same
  simulation with group mean/std = 10 gives output error 1.2e-3; at mean/std = 30 it is 7.6e-3 —
  exceeding tolerance. Real Sana-VAE activations (post-SiLU, non-zero-mean, realistic weight scales)
  plausibly reach such regimes. The benchmark's synthetic inputs sit at the benign extreme of the
  E[x²]−E[x]² cancellation problem and cannot certify the kernel for the model it claims to
  represent. Atomics also make outputs run-to-run nondeterministic (allowed by the harness, worth a
  footnote).

## 5. Reward-hack surface

Verified: the four candidates contain no caching, no global-flag mutation, no monkey-patching; the
bridge wrapper only caches the `ModelNew` instance (legal). The harness's defenses (fresh randn per
correctness round ×10, pointer-shifting pool, elapsed_time identity check, thread/lazy checks) are
byte-identical to upstream `2d852a3`, except:

**F6: the vendored allocator predates the v1.1 fix.** Timed iterations reuse byte-identical input
VALUES 60×, and the vendored `ShiftingMemoryPoolAllocator` shifts pointers by a FIXED 256-byte
stride — the deterministic signal upstream explicitly randomized one day after vendoring
(#18, 2026-07-15). A value-hashing or loop-detecting candidate could skip compute in the timed loop
(~20× apparent speedup on large workloads: hash+copy is ~memory-bound, conv is 19.7× above the
memory bound). Not exploited by the current candidates, but the experiment should not claim
hack-robustness for third-party kernels until re-vendored to v1.1.

Also noted: outputs are checked for shape+dtype but not strides — channels_last candidates return
NHWC-strided tensors while T_b delivers NCHW-contiguous, silently externalizing any layout
conversion a downstream consumer would need (minor, ~5% of T_SOL on the largest shape).

## 6. Measurement validity on WSL2 (unlocked clocks) — quantified from this run's own traces

- Sanity control (reference mirror) scored SOL **0.500** — but ran only **3 of 20 workloads**.
- **F7: cross-session anchor drift.** In the candidate sessions, the co-measured in-process reference
  ran median **7% faster** (per-workload ratio 0.85–1.01) than the stored `baselines.json` t_b from
  the earlier baseline session. Re-anchoring SOL on the same-session reference latency:
  triton_fused 0.605→0.561, compile_default 0.476→0.432, cudagraph 0.471→~0.43, nhwc_eager
  0.382→0.337. All reported SOL scores are inflated ~0.04–0.05 by the stale anchor. Baseline was
  also measured with iterations=100 vs candidates' 50 (asymmetric sampling).
- Only medians are stored (`return_mode="median"`); raw per-iteration samples are discarded, so 3–4×
  transient outliers can be neither detected nor trimmed post-hoc.

**Guards the final report must apply:**
1. Anchor SOL on the same-session `reference_latency_ms` already present in each trace (or report
   both anchors); flag any workload where |in-proc ref / stored t_b − 1| > 5%.
2. Re-run timing with `return_mode="all"` (or log min/max) so transients are visible; report
   median + IQR, not bare medians.
3. Run the sanity mirror over all 20 workloads in every session; gate on control score 0.50 ± 0.02.
4. Treat latency differences < 10% as inconclusive under unlocked WSL2 clocks; require replication
   across ≥2 sessions with interleaved candidate/baseline order.

## 7. Incidental observations (candidate quality, not harness flaws)

- `nhwc_eager` is **bit-exact** (max_abs_err = 0.0) vs the NCHW reference on 19/20 workloads —
  strong evidence the "channels_last" path executed the same internal cuDNN kernels (Blackwell cuDNN
  transposes NCHW internally). The candidate therefore measures added layout-conversion overhead,
  not a layout win — consistent with its 0.76× median "speedup". Its docstring claim ("isolates the
  layout win") is not supported.
- All three non-fused channels_last candidates hit ~98–104 ms on 1×1024×1024 vs 44 ms in-process
  reference — consistent across mechanisms, so a real slow path (likely group_norm layout thrash),
  not a WSL2 transient.
- Baseline sanity: stored t_b median across 20 workloads = 2.4107 ms ≈ reported 2.41 ms; 20/20
  passes consistent with traces.

## Verdict

t_sol.json arithmetic and hardware constants are correct and compute-boundness holds for all 20
workloads; the Triton variance-precision worry is refuted under this harness's inputs. The setup is
refuted on: T_SOL semantics (Winograd/fp16 make it beatable, score unclamped) [F1]; GroupNorm
affine and eps never tested [F2, F3]; 1%-unbounded-error tolerance [F4]; benign-only input
distribution masking a real E[x²]−E[x]² failure regime [F5]; stale pre-v1.1 anti-caching allocator
[F6]; and ~7% stale-anchor inflation of every reported SOL score on unlocked WSL2 clocks with a
3-workload-only control [F7].
