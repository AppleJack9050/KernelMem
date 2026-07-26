# SOL-ExecBench problem 002 on RTX 5090 — results

Problem: `002_vae_conv3x3_groupnorm_silu_residual_fused` (L1), Sana VAE residual block:
`Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> +residual`,
fp32, 256 channels, 20 workloads. Device: RTX 5090 (170 SM, 32 GB GDDR7), driver 610.74,
torch 2.11.0+cu128, WSL2 (clocks NOT lockable). Timing: `cuda_events` (CUPTI needs
libcupti.so.13; this host has .12).

## Headline

All six candidates are numerically correct on 20/20 workloads. Speedups are geometric
means over the 20 workloads, anchored on the **same-session** reference latency carried in
each trace (the stored `baselines.json` t_b drifted ~10% across sessions and inflates
results by ~7-9%).

| candidate | pass | speedup (same-session anchor) | speedup (stale anchor) |
|---|---|---|---|
| `compile_unlimited` — torch.compile, dynamo cache limit raised | 20/20 | **1.096x** | 1.194x |
| `nhwc_triton_fused` — channels_last convs + Triton GN/SiLU/residual | 20/20 | **1.089x** | 1.163x |
| `agent_best` — KernelMem-generated fused CUDA C++ GroupNorm+SiLU | 20/20 | **0.935x** | 1.005x |
| `compile_cudagraph` — torch.compile reduce-overhead | 20/20 | 0.861x | 0.915x |
| `compile_default` — torch.compile, default cache limit | 20/20 | 0.858x | 0.919x |
| `nhwc_eager` — channels_last only, native GroupNorm | 20/20 | 0.741x | 0.795x |

## The agent run

`main_memory_latest.py` with the rewritten Claude Agent SDK backend, 6 rounds,
claude-opus-4-8, 34,794 tokens total. Round 0's kernel died with
`cudaErrorNoKernelImageForDevice`; the repair round fixed it and scored **1.0804x**.
Rounds 2-5 produced no kernel that beat the incumbent, so only 2 kernels were generated.

**The 1.0804x is real but does not generalize.** KernelMem binds a task to a *single*
workload (here index 18, `batch=8, H=64, W=128`). On that exact workload the SOL harness
independently measures 1.05x — consistent. Across all 20 workloads the same kernel is
0.935x. The failure mode is visible by shape:

| shape | speedup |
|---|---|
| batch 32-64, H=W=64..128 | 1.14 - 1.15x |
| batch 8, 64x128 (the bound workload) | 1.05x |
| batch 1, 293x293 | 0.70x |
| batch 1, 1024x1024 | 0.74x |

The generated kernel assigns one CTA per `(batch, group)` pair. At batch=1 that is 32 CTAs
for 170 SMs — ~19% occupancy — so it collapses precisely where the reference's
grid-stride GroupNorm still saturates the device. Optimizing against one workload selected
a schedule that is parallelism-starved at small batch.

## Caveats (from an adversarial review of this setup; see methodology_review.md)

1. **T_SOL here is a direct-convolution bound, not a true speed-of-light bound.** Winograd
   F(4x4,3x3) cuts multiplies 2.25-4x for this exact conv shape and cuDNN uses it routinely,
   so a kernel can legitimately beat "T_SOL". Upstream `sol_score()` does not clamp at 1.0.
   Treat the SOL column as indicative only; it is not leaderboard-comparable (the real
   scoring needs NVIDIA's private T_b/T_SOL on locked-clock B200).
2. **The GroupNorm affine transform is never tested.** The harness input heuristic keys on
   the name "norm" and always generates `norm*_weight`=ones, `norm*_bias`=zeros. A kernel
   that ignores gamma/beta entirely passes 20/20.
3. **`eps` is untestable.** Conv weights are drawn as randn/sqrt(3), so group variance is
   ~768 >> eps=1e-6; even hardcoding eps=1.0 stays inside tolerance.
4. **1% of output elements may be arbitrarily wrong.** `required_matched_ratio` is 0.99 and
   `max_error_cap` is null. On 1024x1024 that is 2.68M unconstrained elements — a wrong
   conv halo ring (~0.4%) would pass.
5. **The vendored anti-caching allocator predates upstream v1.1** (commit `a9fa080`,
   2026-07-15, one day after vendoring), which randomized a previously fixed 256-byte
   pointer stride. The candidates here are clean, but do not claim hack-robustness for
   third-party kernels until the tree is re-vendored.
6. **WSL2 clocks cannot be locked.** Treat differences under 10% as inconclusive and
   replicate across sessions before believing a regression.

## Reproduce

```bash
# score any candidate across all 20 workloads
python3 -m solbench_bridge evaluate \
  solbench_problems/L1/002_vae_conv3x3_groupnorm_silu_residual_fused \
  --kernel run/vae_block_002/kernels/<name>.py --task tasks/vae_block_002.py \
  --language pytorch --timing cuda_events \
  --baselines run/vae_block_002/baselines.json -o run/vae_block_002/out_<name>

python3 run/vae_block_002/summarize_v2.py    # same-session-anchored comparison
```

Note: SOL-ExecBench blocks `cpp_extension.load_inline()` on the GPU server, so the
agent's CUDA kernel had to be pre-built and loaded from a `.so`
(`run/vae_block_002/prebuilt/agent_best_prebuilt.py`). The compute is unchanged.
