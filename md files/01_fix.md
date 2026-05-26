 What changed and what's different now

  The problem (what was broken)

  With gain augmentation active (prob=0.5), each sample in a
  batch gets its own gain — some un-augmented at g_true ≈ 28,
  some augmented anywhere in [20, 2000]. But the model's
  Anscombe inverse used one shared scalar for the whole batch:

  # Before — same g for all 16 samples in the batch
  g = torch.as_tensor(params.gain, ...)   # always the batch
  median = g_true
  tgt_raw = (vol/2)² × g_true - ...      # wrong for augmented
  samples
  pred_adu = model(x, params)             # wrong inverse for
  augmented samples

  Both pred and tgt got scaled by k = g_true / g_aug. The loss
  was therefore also scaled by k. A sample with g_aug = 991 on
  an A1 stack (g_true = 28) contributed 1/35th of the gradient
  weight of an un-augmented sample. The augmented samples
  existed in the input data but were nearly invisible to the
  optimizer.

  The fix

  _make_params() now returns (NoiseParams, gains_tensor). Every
  step_* function:
  1. Reshapes gains to (B,1,1,1,1) and passes it to
  model.forward(gain_tensor=g)
  2. Computes tgt_raw with per-sample g — correct for every
  sample independently

  All 5 model forward() signatures got gain_tensor: Tensor |
  None = None. When None (inference, single tile, no
  augmentation) they fall back to the scalar — no change to the
  inference path.

  What's different during training now

  ┌─────────────────────┬─────────────────┬─────────────────┐
  │     Sample type     │     Before      │      After      │
  ├─────────────────────┼─────────────────┼─────────────────┤
  │ Un-augmented (g_aug │ ✅ correct      │ ✅ correct      │
  │  = g_true)          │                 │ (unchanged)     │
  ├─────────────────────┼─────────────────┼─────────────────┤
  │ Aug, g_aug = 200    │ loss ×0.14 (7×  │ ✅ full weight  │
  │                     │ too small)      │                 │
  ├─────────────────────┼─────────────────┼─────────────────┤
  │ Aug, g_aug = 991    │ loss ×0.028     │ ✅ full weight  │
  │                     │ (35× too small) │                 │
  ├─────────────────────┼─────────────────┼─────────────────┤
  │ Aug, g_aug = 2000   │ loss ×0.014     │ ✅ full weight  │
  │                     │ (71× too small) │                 │
  └─────────────────────┴─────────────────┴─────────────────┘

  Why this matters for F3 specifically

  F3 has gain ≈ 991 — exactly the range that was nearly muted
  before. The gain augmentation was designed specifically to
  teach the model about F3-level noise. Now those augmented
  samples compete equally with in-distribution samples in every
  batch. The model will actually learn what high-gain noise
  looks like, which should translate to a real improvement in F3
   OOD stSNR.

✻ Worked for 7m 19s

❯ one question. I ran the model test comparsion from
  @NEXT_STEPS.md and it was before we did this changes would it
   impact he winner too much or its just okay to continue
  ⎿  Read NEXT_STEPS.md (unchanged)

● Good question. Let me reason through this carefully.

  Short answer: continue with the current winner — re-running is
   not necessary

  Here's why:

  The bug affected all 4 models equally

  The broken Anscombe inverse was in the output head, not the
  backbone:

  # This identical line was wrong in ALL 5 models
  g = torch.as_tensor(params.gain, ...)   # same scalar for all
  return (z_pred / 2.0).pow(2) * g - 0.375 * g - sr2 / g

  N2V3D base, N2V3D large, Mamba base, Mamba large — all used
  the exact same formula with the exact same wrong g. The bias
  was symmetric across all 4. The architectural differences
  (conv vs SSM bottleneck, base vs large channel count) live
  entirely in the encoder/decoder, before this head. So whatever
   advantage one model had over another, it had it despite the
  same gain error applied to all.

  The comparison is about capacity and inductive bias, not
  gradient precision

  The model test answers: "does larger channel count help? does
  the Mamba SSM bottleneck help?" Those signals dominate at 10
  epochs far more than the gradient weighting of augmented
  samples. A model that's a better fit for this data will show
  it regardless of whether augmented samples contributed 1/35th
  or full gradient weight — especially at 10 epochs where you're
   still in the early learning regime.

  The one real risk: F3 OOD scores were compressed for everyone

  During the test, all 4 models saw weak gradients from
  high-gain augmented samples. This means the F3 stSNR numbers
  from that run are lower than they would be with the fix — but
  again, lower for all 4 equally. If model A beat model B on F3
  by e.g. 0.5 dB in that run, it would likely still beat it with
   the fix (maybe by a slightly different margin).

  The only scenario where the verdict could flip

  If two models were within the 1 dB decision threshold on F3,
  and one architecture happens to benefit more from correct
  high-gain gradients than the other. In practice this is
  unlikely because Mamba and N2V3D have the same output head —
  neither has a structural advantage in how it handles the gain
  tensor.

  What will actually be different in full training

  The full training run (which hasn't started yet) will have the
   fix applied. That's what matters. The model comparison just
  tells you which architecture to use — the full training is
  where the fix's benefit shows up in the final scores.

  Bottom line: trust the winner from the model comparison. Run
  full training with the fix in place. If F3 margins were very
  tight between the top two models, you could note that and keep
   an eye on F3 during full training — but don't re-run the
  10-epoch comparison.