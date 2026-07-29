# ==========================================================================
# ModelNew — fused VAE residual block
#   Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> + x
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels"
#
# OPTIMISATION (this revision): gridsync_fused_groupnorm
#   Each GroupNorm's  stats -> finalize -> apply  triple is now ONE persistent
#   cooperative kernel that walks the batch image-by-image with grid-wide
#   barriers.  The phase-A read and the phase-B re-read of image n are separated
#   by only C*H*W*4 bytes (L2-resident) instead of a whole kernel launch over the
#   full N*C*H*W tensor, so the apply phase hits L2 instead of DRAM and one full
#   tensor DRAM read per GroupNorm disappears.  The two gn_finalize launches are
#   folded in as well.
#
# Kernels:
#   K0  nchw_to_nhwc_kernel        : x (NCHW) -> xn (NHWC) tiled transpose
#   [vendor] at::conv2d channels_last
#   KF  gn_fused_kernel<false>     : stats + finalize + (affine+SiLU) NHWC->NHWC
#   [vendor] at::conv2d channels_last
#   KF  gn_fused_kernel<true>      : stats + finalize + (affine+SiLU+residual+
#                                    NHWC->NCHW transpose)
#   legacy fallback (kept, unmodified): gn_stats / gn_finalize / gn_apply_nhwc /
#                                       gn_apply_transpose
#
# Precision: fp32 storage/arithmetic throughout; GN reductions use fp32 partials
# combined in fp64; TF32 tensor cores are used by the convs exactly as the
# reference does.
# ==========================
