"""07 — Architecture validation on real stacks.

QUESTION: Do U-Net, ConvNeXt, and N2V3D learn to denoise real calcium imaging data?
          How do they generalize across noise levels?
          Which one is ready to scale?

Decision gate: picks the architecture + masking strategy before full training.
Measures sSNR, tSNR, stSNR on real validation stacks (F1, F2, F3) against F0.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07 — Architecture validation on real stacks

    **QUESTION:** Do the three candidate architectures learn to denoise?
    How do they generalize across noise levels F1 → F2 → F3?

    **Dataset scope:**
    - Training: A1 stack only (level 1, g=28.4)
    - Validation: F0 (clean ground truth), F1/F2/F3 (noisy stacks at three levels)
    - Scoring: sSNR, tSNR, stSNR (50/50 blend) by comparing denoised outputs to F0

    **Key constraints from prior notebooks:**
    - τ₀.₅ = 46 frames → **T=64 is locked** (full transient visibility)
    - tSNR is harder than sSNR (1–2 dB gap on raw data)
    - Only 4 training stacks total → every frame matters
    - Spatial-only denoisers plateau on tSNR — need temporal understanding

    **This notebook validates:** can each architecture learn real temporal dynamics?
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset, DataLoader

    from cidc import load_stack, stsnr, NOISE_LEVELS
    from cidc.losses import poisson_gaussian_nll
    from cidc.noise import sample_poisson_gaussian, NoiseParams

    DATA = Path(__file__).parent.parent.parent / "data"
    return (
        DATA,
        DataLoader,
        Dataset,
        load_stack,
        mo,
        nn,
        np,
        plt,
        poisson_gaussian_nll,
        stsnr,
        torch,
    )


@app.cell
def _load_data(DATA, load_stack, np):
    # Load training stack (A1, level 1)
    a1_full = np.asarray(load_stack(DATA / "train" / "A1.tif"), dtype=np.float32)

    # Load validation stacks
    f0_full = np.asarray(load_stack(DATA / "val" / "F0.tif"), dtype=np.float32)  # clean reference
    f1_full = np.asarray(load_stack(DATA / "val" / "F1.tif"), dtype=np.float32)  # level 1, same noise as training
    f2_full = np.asarray(load_stack(DATA / "val" / "F2.tif"), dtype=np.float32)  # level 2, 2x harder
    f3_full = np.asarray(load_stack(DATA / "val" / "F3.tif"), dtype=np.float32)  # level 3, OOD

    print(f"A1 (train):      shape={a1_full.shape}  dtype={a1_full.dtype}")
    print(f"F0 (reference):  shape={f0_full.shape}  dtype={f0_full.dtype}")
    print(f"F1 (level 1):    shape={f1_full.shape}  dtype={f1_full.dtype}")
    print(f"F2 (level 2):    shape={f2_full.shape}  dtype={f2_full.dtype}")
    print(f"F3 (level 3):    shape={f3_full.shape}  dtype={f3_full.dtype}")
    return a1_full, f0_full, f1_full, f2_full, f3_full


@app.cell
def _md_patch_info(mo):
    mo.md("""
    ## Patch sampling and 3D masking

    From NB01: τ₀.₅=46 frames → **T=64 is locked**.
    From NB06: ±10 frame temporal context for blind-spot masking.

    We extract random 3D patches (T=64, H=64, W=64) from A1 for training.
    During training, center voxels are masked and predicted from context.
    """)
    return


@app.cell
def _patch_dataset(Dataset, a1_full, np, torch):
    class PatchDataset(Dataset):
        """Sample random 3D patches from stack for training with blind-spot masking."""

        def __init__(self, stack, patch_shape=(64, 64, 64), n_patches=200, seed=42):
            self.stack = stack  # (T, H, W)
            self.T, self.H, self.W = stack.shape
            self.patch_t, self.patch_h, self.patch_w = patch_shape
            self.n_patches = n_patches
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return self.n_patches

        def __getitem__(self, idx):
            # Random patch location
            t0 = self.rng.integers(0, self.T - self.patch_t)
            h0 = self.rng.integers(0, self.H - self.patch_h)
            w0 = self.rng.integers(0, self.W - self.patch_w)

            patch = self.stack[t0:t0+self.patch_t, h0:h0+self.patch_h, w0:w0+self.patch_w].copy()
            return torch.as_tensor(patch, dtype=torch.float32)

    train_ds = PatchDataset(a1_full, patch_shape=(64, 64, 64), n_patches=200, seed=42)
    print(f"Training dataset: {len(train_ds)} patches of shape {train_ds.patch_shape if hasattr(train_ds, 'patch_shape') else '(64,64,64)'}")
    return (train_ds,)


@app.cell
def _md_architectures(mo):
    mo.md("""
    ## Three candidate architectures

    All lightweight, designed to fit in 6GB VRAM at patch scale (64×64×64).

    1. **U-Net 3D:** Standard skip-connection encoder-decoder. Proven, interpretable.
    2. **ConvNeXt 3D:** Modern residual blocks with depthwise convolutions. More parameter-efficient.
    3. **N2V3D:** Sparse, receptive-field-aware design for blind-spot masking (published for this task).
    """)
    return


@app.cell
def _unet_3d(nn):
    class UNet3D(nn.Module):
        """Lightweight 3D U-Net for 64×64×64 patches."""

        def __init__(self, in_channels=1, out_channels=1, base_channels=32):
            super().__init__()
            self.in_conv = nn.Sequential(
                nn.Conv3d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

            # Encoder
            self.enc1 = self._enc_block(base_channels, base_channels * 2)
            self.enc2 = self._enc_block(base_channels * 2, base_channels * 4)

            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.ReLU(inplace=True),
            )

            # Decoder
            self.dec2 = self._dec_block(base_channels * 8, base_channels * 4)
            self.dec1 = self._dec_block(base_channels * 4, base_channels * 2)

            self.out_conv = nn.Sequential(
                nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_channels, out_channels, 3, padding=1),
            )

        def _enc_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.MaxPool3d(2),
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def _dec_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            # x: (B, 1, T, H, W)
            x = self.in_conv(x)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            b = self.bottleneck(e2)
            d2 = self.dec2(b)
            d1 = self.dec1(d2)
            out = self.out_conv(d1)
            return out

    return (UNet3D,)


@app.cell
def _convnext_3d(nn):
    class ConvNeXt3D(nn.Module):
        """Lightweight 3D ConvNeXt for 64×64×64 patches."""

        def __init__(self, in_channels=1, out_channels=1, base_channels=32, depth=4):
            super().__init__()
            self.in_conv = nn.Conv3d(in_channels, base_channels, 7, padding=3)

            # Residual blocks
            self.blocks = nn.ModuleList([
                self._residual_block(base_channels) for _ in range(depth)
            ])

            self.out_conv = nn.Sequential(
                nn.GroupNorm(1, base_channels),
                nn.Conv3d(base_channels, out_channels, 3, padding=1),
            )

        def _residual_block(self, channels):
            return nn.Sequential(
                nn.Conv3d(channels, channels * 4, 1),  # expand
                nn.GELU(),
                nn.Conv3d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4),  # depthwise
                nn.GELU(),
                nn.Conv3d(channels * 4, channels, 1),  # contract
            )

        def forward(self, x):
            x = self.in_conv(x)
            identity = x
            for block in self.blocks:
                x = x + block(x)
            x = x + identity
            out = self.out_conv(x)
            return out

    return (ConvNeXt3D,)


@app.cell
def _n2v3d_arch(nn, torch):
    class N2V3D(nn.Module):
        """N2V3D: sparse receptive field for blind-spot masking."""

        def __init__(self, in_channels=1, out_channels=1, base_channels=32):
            super().__init__()
            # Sparse input sampling: ±1 pixel + ±2 frames
            self.input_projection = nn.Conv3d(in_channels, base_channels, 1)

            # Multiple scales
            self.scale1 = self._make_scale(base_channels, base_channels * 2, kernel=3)
            self.scale2 = self._make_scale(base_channels * 2, base_channels * 4, kernel=3)

            # Merge and output
            self.merge = nn.Conv3d(base_channels * 6, base_channels * 4, 1)
            self.out_conv = nn.Sequential(
                nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_channels * 2, out_channels, 3, padding=1),
            )

        def _make_scale(self, in_ch, out_ch, kernel=3):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel, padding=kernel//2),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel, padding=kernel//2),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.input_projection(x)
            s1 = self.scale1(x)
            s2 = self.scale2(s1)
            merged = torch.cat([x, s1, s2], dim=1)
            merged = self.merge(merged)
            out = self.out_conv(merged)
            return out

    return (N2V3D,)


@app.function
def apply_voxel_mask(patch_batch, context_frames=10):
    """Apply 3D voxel blind-spot mask for self-supervised training.

    Args:
        patch_batch: (B, 1, T, H, W) tensor
        context_frames: temporal context on each side (±N frames)

    Returns:
        masked_patch: center voxel masked to 0
        t_c, h_c, w_c: center indices
    """
    B, C, T, H, W = patch_batch.shape
    t_c = T // 2
    h_c = H // 2
    w_c = W // 2

    masked = patch_batch.clone()
    masked[:, 0, t_c, h_c, w_c] = 0

    return masked, t_c, h_c, w_c


@app.cell
def _train_fn(poisson_gaussian_nll, torch):
    def train_architecture(model, train_loader, device, num_epochs=1, lr=1e-3, g=28.4, sr2=2490, apply_voxel_mask=None):
        """Train one architecture on A1 patches."""
        from torch.optim import Adam

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        history = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for patches in train_loader:
                B, T, H, W = patches.shape
                patches = patches.unsqueeze(1).to(device)  # (B, 1, T, H, W)

                # Apply voxel masking
                masked_patches, t_c, h_c, w_c = apply_voxel_mask(patches)

                # Forward
                pred = model(masked_patches)

                # Loss on center voxel prediction
                pred_center = pred[:, 0, t_c, h_c, w_c]
                obs_center = patches[:, 0, t_c, h_c, w_c]

                loss = poisson_gaussian_nll(pred_center, obs_center, gain=g, read_var=sr2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.5f}")

        return history

    return (train_architecture,)


@app.cell
def _eval_fn(np, torch):
    def evaluate_on_stack(model, stack, device, patch_size=64, stride=32):
        """Denoise a full stack by sliding 3D patches.

        Returns denoised stack and evaluation metrics vs F0.
        """
        model.eval()
        T, H, W = stack.shape
        denoised = np.zeros_like(stack)
        counts = np.ones_like(stack) * 1e-6  # avoid division by zero

        with torch.no_grad():
            for t0 in range(0, T - patch_size + 1, stride):
                for h0 in range(0, H - patch_size + 1, stride):
                    for w0 in range(0, W - patch_size + 1, stride):
                        patch = stack[t0:t0+patch_size, h0:h0+patch_size, w0:w0+patch_size]

                        if patch.shape[0] < patch_size or patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                            continue

                        patch_t = torch.as_tensor(patch, dtype=torch.float32)
                        patch_t = patch_t.unsqueeze(0).unsqueeze(0).to(device)
                        pred = model(patch_t).squeeze(0).squeeze(0).cpu().numpy()

                        denoised[t0:t0+patch_size, h0:h0+patch_size, w0:w0+patch_size] += pred
                        counts[t0:t0+patch_size, h0:h0+patch_size, w0:w0+patch_size] += 1

        denoised = denoised / counts
        return denoised

    return (evaluate_on_stack,)


@app.cell
def _run_experiment(
    ConvNeXt3D,
    DataLoader,
    N2V3D,
    UNet3D,
    evaluate_on_stack,
    f0_full,
    f1_full,
    f2_full,
    f3_full,
    mo,
    np,
    stsnr,
    torch,
    train_architecture,
    train_ds,
):
    mo.md("""
    ## Experiment: Train all three architectures on A1 and evaluate on F1/F2/F3

    **Settings:**
    - Train on A1 (level 1, g=28.4, σ_r²=2490)
    - Validate metrics on F1/F2/F3 against F0 (clean reference)
    - 1 epoch per architecture (quick validation)
    - Patch size: 64×64×64 (T=64 locked from NB01)
    - Batch size: 4
    """)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training setup
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    g_level1 = 28.4
    sr2_level1 = 2490

    # Architecture configs
    architectures = {
        "U-Net 3D": UNet3D(in_channels=1, out_channels=1, base_channels=32),
        "ConvNeXt 3D": ConvNeXt3D(in_channels=1, out_channels=1, base_channels=32, depth=4),
        "N2V3D": N2V3D(in_channels=1, out_channels=1, base_channels=32),
    }

    results = {}

    for arch_name, model in architectures.items():
        print(f"\n{'='*60}")
        print(f"Training {arch_name}")
        print(f"{'='*60}")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Train
        history = train_architecture(model, train_loader, device, num_epochs=1,
                                     lr=1e-3, g=g_level1, sr2=sr2_level1,
                                     apply_voxel_mask=apply_voxel_mask)

        # Evaluate on F1, F2, F3
        print(f"\nEvaluating on validation stacks...")
        # Baseline: input SNR (noisy vs F0)
        T_eval = min(200, len(f0_full))
        ref = np.asarray(f0_full[:T_eval], dtype=np.float32)
        baseline_metrics = {}
        for stack_name, stack in [("F1", f1_full), ("F2", f2_full), ("F3", f3_full)]:
            noisy = np.asarray(stack[:T_eval], dtype=np.float32)
            baseline = stsnr(noisy, ref)
            baseline_metrics[stack_name] = baseline.st_snr
            print(f"  {stack_name} input (baseline):  stSNR={baseline.st_snr:+6.2f} dB")

        metrics = {}

        for stack_name, stack in [("F1", f1_full), ("F2", f2_full), ("F3", f3_full)]:
            denoised = evaluate_on_stack(model, stack, device, patch_size=64, stride=32)

            # Score against F0 (first 200 frames to match NB01)
            T_eval = min(200, len(f0_full))
            ref = np.asarray(f0_full[:T_eval], dtype=np.float32)
            pred = np.asarray(denoised[:T_eval], dtype=np.float32)

            snr_result = stsnr(pred, ref)
            metrics[stack_name] = {
                "sSNR": snr_result.s_snr,
                "tSNR": snr_result.t_snr,
                "stSNR": snr_result.st_snr,
            }

            print(f"  {stack_name:10s}  sSNR={snr_result.s_snr:+6.2f} dB  "
                  f"tSNR={snr_result.t_snr:+6.2f} dB  stSNR={snr_result.st_snr:+6.2f} dB")

        results[arch_name] = metrics
    return (results,)


@app.cell
def _plot_results(mo, np, plt, results):
    mo.md("## Results Summary")

    # Create comparison table
    arch_names = list(results.keys())
    metrics_list = ["sSNR", "tSNR", "stSNR"]
    stacks_list = ["F1", "F2", "F3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, stack_name in zip(axes, stacks_list):
        x = np.arange(len(arch_names))
        width = 0.25

        for i, metric in enumerate(metrics_list):
            values = [results[arch][stack_name][metric] for arch in arch_names]
            ax.bar(x + i*width, values, width, label=metric)

        ax.set_xlabel("Architecture")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f"Validation on {stack_name} (vs F0)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(arch_names, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/arch_comparison.png", dpi=100, bbox_inches="tight")
    plt.show()

    # Print summary table
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("="*80)

    for arch in arch_names:
        print(f"\n{arch}:")
        for stack in stacks_list:
            m = results[arch][stack]
            print(f"  {stack:5s}  sSNR={m['sSNR']:+6.2f}  tSNR={m['tSNR']:+6.2f}  "
                  f"stSNR={m['stSNR']:+6.2f}  (gap sSNR→tSNR: {m['sSNR']-m['tSNR']:+.2f} dB)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    Based on results above:
    1. Which architecture generalizes best across F1→F2→F3?
    2. Does tSNR hold up, or does the model only recover sSNR?
    3. Is one architecture significantly faster or more parameter-efficient?

    Decision: Pick the winner and move to NB08 (full training with gain augmentation).
    """)
    return


if __name__ == "__main__":
    app.run()
