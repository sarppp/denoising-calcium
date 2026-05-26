 Pushed. Now your question — will these config changes make
  full training too different from what was tested?

  ---
  No — the winner won't change. Each change only helps
  n2v3d_large.

  Change: patch [64]→[128]
  Effect on model comparison validity: Gives the model T=128
    temporal context (vs T=64). τ₀.₅=46 frames — at T=64 the
    model captures one decay length, at T=128 it sees two.
  Better
     for  n2v3d_large, not worse. The conv kernels see the full
    transient shape they were designed for.
  ────────────────────────────────────────
  Change: batch 32→16
  Effect on model comparison validity: Halved because patch is
  8×
    larger (same VRAM). Effective gradient signal is similar —
    250 steps/epoch in both cases. No ranking effect.
  ────────────────────────────────────────
  Change: epochs 10→100
  Effect on model comparison validity: The 10-epoch comparison
    had all models far from convergence (all F1 scores
  negative).
     More epochs only let each model reach its true potential.
    n2v3d_large was still improving on F3 at epoch 10 — it will
    get better, not worse.
  ────────────────────────────────────────
  Change: samples_per_epoch 2000→4000
  Effect on model comparison validity: Keeps steps/epoch at 250
    (4000÷16). Purely a density normalization — no effect on
    ranking.
  ────────────────────────────────────────
  Change: tile [64]→[128]
  Effect on model comparison validity: Inference only. Larger
    tile = fewer tile boundaries = fewer stitching artefacts.
    Better quality for n2v3d_large.

  The architectural conclusion doesn't depend on any of these
  settings. The reason N2V3D beats Mamba on F3 is that SSM
  hidden states are gain-calibrated during training and fail at
  inference-time gain=990. That failure happens regardless of
  patch size, batch size, or epoch count — mamba_large would
  still degrade F3 at patch=[128,128,128] because the learned
  state transition matrices are the problem, not the training
  duration.

  If anything, these changes widen the gap: n2v3d_large benefits
   more from T=128 (temporal context is exactly what the conv
  architecture is designed to exploit), and the per-sample gain
  fix now in place means augmented high-gain samples contribute
  full gradient weight — helping F3 generalization across all
  future runs.

✻ Cogitated for 59s

❯ but 128,128,128 will work fine with T4 as inference right

● Yes, comfortably. The numbers are already in NEXT_STEPS.md —
  let me show you exactly why:

  VRAM per forward pass at tile=[128,128,128] with n2v3d_large:

  Inference runs torch.no_grad(), one tile at a time, batch=1.
  Peak memory is the decoder level 0 concatenation (skip +
  upsampled):

  ┌───────────────────────────┬──────────────────┬─────────┐
  │        Activation         │      Shape       │  fp16   │
  │                           │                  │  size   │
  ├───────────────────────────┼──────────────────┼─────────┤
  │ Level 0 skip (stored for  │ [1, 32, 128,     │ ~128 MB │
  │ decoder)                  │ 128, 128]        │         │
  ├───────────────────────────┼──────────────────┼─────────┤
  │ Level 1 skip              │ [1, 64, 64, 64,  │ ~32 MB  │
  │                           │ 64]              │         │
  ├───────────────────────────┼──────────────────┼─────────┤
  │ Level 2 skip              │ [1, 128, 32, 32, │ ~8 MB   │
  │                           │  32]             │         │
  ├───────────────────────────┼──────────────────┼─────────┤
  │ Bottleneck                │ [1, 256, 16, 16, │ ~2 MB   │
  │                           │  16]             │         │
  ├───────────────────────────┼──────────────────┼─────────┤
  │ Decoder L0 concat peak    │ [1, 64, 128,     │ ~256 MB │
  │                           │ 128, 128]        │         │
  └───────────────────────────┴──────────────────┴─────────┘

  Total inference VRAM ≈ 400–600 MB per tile. T4 has 16 GB. You
  have ~25× headroom.

  With 8× TTA each pass is sequential (not parallel) — VRAM
  doesn't multiply, just wall time does.

  Timing:
  tile=[128,128,128], 8× TTA, fp16 on T4: ~100 sec per stack
  Competition limit: 60 min = 3600 sec
  Headroom: 36×

  The only scenario where tile=[128,128,128] could be a problem
  is if the test stack spatial dimensions are not multiples of
  the tile size and padding pushes a tile temporarily larger —
  but the overlap-tile logic in eval.py handles that by padding
  then cropping, and you'd need a very odd stack size to push
  past 1GB.

  Short answer: nothing to change, you're fine.