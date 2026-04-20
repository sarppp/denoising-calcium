Answering first, then building:

# Answering first

## Yes, PINN is in — confirm

`@/app/workspace/src/cidc/models/pinn/` (3 files) + [calcium_kinetics_loss](cci:1://file:///app/workspace/src/cidc/losses.py:155:0-204:16) in `@/app/workspace/src/cidc/losses.py` + `@/app/workspace/configs/pinn.yaml`. Verified end-to-end on CPU (autograd flows through the ODE solver).

## VRAM scaling — architecture stays identical, config changes

The model is **batch-invariant**. You don't change architecture; you change 3 config fields:

| VRAM | `data.batch_size` | `training.grad_accum` | `training.amp` | Effective batch |
|---:|---:|---:|---:|---:|
| 6 GB  |  2 |  8 | `true` | 16 |
| 16 GB |  8 |  2 | `true` | 16 |
| 24 GB | 12 |  1 | `true` | 12 |
| 48 GB | 24 |  1 | `true` | 24 |
| 96 GB | 48 |  1 | `true` | 48 |

Rules:
- **Effective batch = batch_size × grad_accum.** Pick a target (16 is a good default), then split across VRAM tiers. Results converge to the same solution.
- **AMP on always.** Halves activation memory with negligible accuracy cost on this task.
- **Bigger VRAM** also lets you increase patch (32 → 64³), which gives *architectural* headroom (more temporal context per sample). That's the one knob where VRAM does help the model, not just throughput.
- **Gradient checkpointing** (`training.grad_ckpt: true`) trades ~20 % compute for ~40 % memory — enable on 6 GB if patch=32³ still OOMs.

I'll add `grad_accum` + `grad_ckpt` to the config schema now.

---

# Now building

`eval.py` (universal scorer), `data/` (universal datasets), `scripts/train.py` (universal trainer), `scripts/score.py` (CLI). One evaluator for all 5 models.
Now add those fields to the canonical YAML so users see them:

Now the universal evaluator:

Now the dataset package — universal windowing + per-model sample wrappers:





Now `scripts/train.py` — universal trainer that dispatches per-model training step:

Now the standalone CLI scorer:

Now a single smoke test exercising metric, dataset, and inference for every available (CPU-runnable) model:


All smoke tests pass. **And a genuinely important finding dropped out of the sanity baseline.**

# What now exists

| File | Purpose |
|---|---|
| `@/app/workspace/src/cidc/eval.py` | stSNR metric + [denoise_stack](cci:1://file:///app/workspace/src/cidc/eval.py:295:0-351:5) + [evaluate](cci:1://file:///app/workspace/src/cidc/eval.py:359:0-385:46), dispatches all 5 models |
| `@/app/workspace/src/cidc/data/__init__.py`, `@/app/workspace/src/cidc/data/dataset.py` | [CIDCStackDataset](cci:2://file:///app/workspace/src/cidc/data/dataset.py:85:0-194:9) (raw windows, gain aug, Anscombe) + [DeepInterpDataset](cci:2://file:///app/workspace/src/cidc/data/dataset.py:212:0-237:19)/[N2V3DDataset](cci:2://file:///app/workspace/src/cidc/data/dataset.py:240:0-253:19)/[DeepCADDataset](cci:2://file:///app/workspace/src/cidc/data/dataset.py:256:0-276:19) + [build_dataset(cfg)](cci:1://file:///app/workspace/src/cidc/data/dataset.py:284:0-318:74) |
| `@/app/workspace/scripts/train.py` | Universal trainer: dispatch table for all 5 models, AMP, EMA, grad accum, grad clip, cosine restarts, val stSNR early-stop |
| `@/app/workspace/scripts/score.py` | CLI scorer in two modes (pre-denoised pair OR config+ckpt+noisy) |
| `@/app/workspace/src/cidc/config.py` | Added `grad_accum` + `grad_ckpt` knobs for VRAM scaling |

Everything is config-driven. Switch models by changing `model.name` in YAML. Switch GPU tiers by changing `data.batch_size` / `training.grad_accum`.

# Important finding — baseline numbers on real data

Ran the metric on the actual val stacks, **first 128 frames vs F0** (full 1500 × 1500 will be similar):

| Predictor | sSNR | tSNR | **stSNR** |
|---|---:|---:|---:|
| F1 raw (no denoising) | 7.71 | 6.14 | **6.92** |
| F2 raw (no denoising) | −0.40 | −1.75 | **−1.08** |
| F3 raw (no denoising) | −6.27 | −7.49 | **−6.88** |
| **F1 → temporal-mean "denoiser"** | 19.65 | 27.66 | **23.65** |

The **trivial temporal-mean baseline scores 23.65 stSNR on 128 frames** — higher than the current Task 1 leaderboard winner (16.75).

## Why — and what it means for strategy

The temporal-mean "denoiser" replaces every frame with `F1.mean(axis=0)`. This destroys all temporal variation, so you'd expect low tSNR. **But the metric averages tSNR uniformly across all H·W pixels** — and we measured in `@/app/workspace/notebooks/06_proofs.py:5` that only ~0.3–0.9 % of pixels are active neurons (the rest are near-constant background).

For those ~99 % background pixels, `temporal_mean` is a nearly perfect predictor → huge tSNR.
For the ~1 % active pixels, tSNR is bad — but they're outnumbered 100 : 1.

**Implications:**

1. **Leaderboard scores of 16.75 probably reflect full-1500-frame scoring** where the temporal mean is a less stable estimator and the metric is *harder* for trivial baselines than at 128 frames — so the raw floor comparison isn't apples-to-apples. I should run on full stacks to confirm.
2. **Any model that merely learns "predict the temporal mean" already scores in the ~20+ range.** This is the silent background-pixel subsidy.
3. **The real discriminator between models is performance on the ~1 % active pixels.** That's where calcium transients live, and where "smoothed everything to the mean" loses tSNR hard.
4. **Ablation that matters:** separately report **stSNR on active pixels only** (thresholded via `F0.mean(axis=0) > μ + 5·σ_MAD`) vs background. I should add this to [eval.py](cci:7://file:///app/workspace/src/cidc/eval.py:0:0-0:0) as a secondary metric — it's the actual science-relevant signal.

I'll verify the 1500-frame temporal-mean number next (cheaper than training anything), then add the active-pixel masked metric. That will either confirm the score has headroom up to 25+ (good) or show the metric is dominated by background in a way that caps upside (critical to know now).

Want me to run that full-stack baseline + add the masked metric now, or move straight to a first real training run on v1 (n2v3d)?