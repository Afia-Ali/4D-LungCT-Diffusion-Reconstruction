"""
Diffusion training loop for the phase-conditioned 3D U-Net.

This is Section 9 of the architecture notes, turned into runnable code:

  1. Pick a real intermediate phase (e.g. T30) from a training patient.
  2. Pick a random noise level t (0-999).
  3. Add exactly that much Gaussian noise to T30.
  4. Feed [noisy_T30, T00, T50] into the U-Net, with phase=0.3, timestep=t.
  5. U-Net predicts the noise that was added.
  6. Loss = MSE(predicted noise, actual noise).
  7. Backprop, update weights. Repeat across all patients/phases/noise levels.

Expects data laid out as:
  <data_root>/case<N>/T00.npy, T10.npy, ..., T90.npy         (real scans)
  <data_root>/synthetic/case<N>/T10_synth.npy, ...           (optional, augmentation)

Run modes:
  --selftest   builds tiny dummy data and runs 2 mini-epochs, to confirm the
               whole loop works end-to-end before touching real data or GPUs.
  (default)    real training run against --data-root.
"""

import argparse
import os
import random
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet3d_film import UNet3DFiLM

INTERMEDIATE_PHASES = ["T10", "T20", "T30", "T40", "T60", "T70", "T80", "T90"]


# ---------------------------------------------------------------------------
# Noise schedule (standard DDPM linear beta schedule)
# ---------------------------------------------------------------------------

class NoiseScheduler:
    """Precomputes the noise schedule and handles the forward (noising)
    process. The reverse (denoising/sampling) process is a separate script --
    this one only needs forward noising to create training examples."""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)

    def add_noise(self, x0, t):
        """x0: (B, 1, D, H, W) clean patch. t: (B,) long timestep indices.
        Returns (noisy_x, noise) -- both same shape as x0."""
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1, 1)  # (B,1,1,1,1)
        noise = torch.randn_like(x0)
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise


# ---------------------------------------------------------------------------
# Dataset: real (+ optional synthetic) intermediate phases, patch-cropped
# ---------------------------------------------------------------------------

class PhasePatchDataset(Dataset):
    """Each item = one (patient, phase) pair, patch-cropped to a fixed size.
    Every item also carries that patient's T00 and T50 anchors, cropped from
    the exact same spatial location."""

    def __init__(self, data_root, case_ids, patch_size=64,
                 synthetic_root=None, synthetic_case_ids=None):
        self.data_root = data_root
        self.patch_size = patch_size
        self.samples = []  # list of (case_id, phase_name, is_synthetic)

        for case_id in case_ids:
            for phase in INTERMEDIATE_PHASES:
                path = self._real_path(case_id, phase)
                if os.path.exists(path):
                    self.samples.append((case_id, phase, False))

        if synthetic_root is not None:
            self.synthetic_root = synthetic_root
            for case_id in (synthetic_case_ids or case_ids):
                for phase in INTERMEDIATE_PHASES:
                    path = self._synth_path(case_id, phase)
                    if os.path.exists(path):
                        self.samples.append((case_id, phase, True))

        if not self.samples:
            raise RuntimeError(
                f"No training samples found under {data_root}. "
                "Check that case folders and .npy files exist."
            )

    def _real_path(self, case_id, phase):
        return os.path.join(self.data_root, f"case{case_id}", f"{phase}.npy")

    def _synth_path(self, case_id, phase):
        return os.path.join(self.synthetic_root, f"case{case_id}", f"{phase}_synth.npy")

    def _anchor_path(self, case_id, name):
        return os.path.join(self.data_root, f"case{case_id}", f"{name}.npy")

    @staticmethod
    def _phase_to_float(phase_name):
        # "T30" -> 0.3
        return int(phase_name[1:]) / 100.0

    def _random_crop_coords(self, shape):
        p = self.patch_size
        coords = []
        for dim in shape:
            if dim <= p:
                coords.append(0)  # volume smaller than patch: start at 0, pad later
            else:
                coords.append(random.randint(0, dim - p))
        return coords

    def _crop(self, vol, coords):
        p = self.patch_size
        d0, h0, w0 = coords
        patch = vol[d0:d0 + p, h0:h0 + p, w0:w0 + p]
        # pad with zeros if volume was smaller than patch in any dimension
        pad = [(0, max(0, p - patch.shape[i])) for i in range(3)]
        if any(a or b for a, b in pad):
            patch = np.pad(patch, pad, mode="constant", constant_values=0)
        return patch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, phase, is_synthetic = self.samples[idx]

        target_path = self._synth_path(case_id, phase) if is_synthetic else self._real_path(case_id, phase)
        target = np.load(target_path).astype(np.float32)
        t00 = np.load(self._anchor_path(case_id, "T00")).astype(np.float32)
        t50 = np.load(self._anchor_path(case_id, "T50")).astype(np.float32)

        coords = self._random_crop_coords(target.shape)
        target_patch = self._crop(target, coords)
        t00_patch = self._crop(t00, coords)
        t50_patch = self._crop(t50, coords)

        # NOTE: adjust this normalization to match whatever range your existing
        # preprocessing already used. This assumes raw-ish HU values in
        # roughly [-1000, 1000] and maps them to [-1, 1]. If your .npy files
        # are already normalized, skip this and pass them through unchanged.
        def norm(x):
            return np.clip(x, -1000, 1000) / 1000.0

        target_patch = norm(target_patch)
        t00_patch = norm(t00_patch)
        t50_patch = norm(t50_patch)

        return {
            "target": torch.from_numpy(target_patch).unsqueeze(0),  # (1,D,H,W)
            "t00": torch.from_numpy(t00_patch).unsqueeze(0),
            "t50": torch.from_numpy(t50_patch).unsqueeze(0),
            "phase": torch.tensor(self._phase_to_float(phase), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, dataloader, scheduler, device, epochs, lr, checkpoint_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    history = []
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            target = batch["target"].to(device)
            t00 = batch["t00"].to(device)
            t50 = batch["t50"].to(device)
            phase = batch["phase"].to(device)

            B = target.shape[0]
            t = torch.randint(0, scheduler.T, (B,), device=device).long()

            noisy_target, noise = scheduler.add_noise(target, t)
            model_input = torch.cat([noisy_target, t00, t50], dim=1)  # (B,3,D,H,W)

            predicted_noise = model(model_input, phase, t.float())
            loss = nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} -- avg MSE loss: {avg_loss:.5f}")

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

    return history


# ---------------------------------------------------------------------------
# Self-test: builds tiny dummy data, confirms the whole loop runs end-to-end
# ---------------------------------------------------------------------------

def run_selftest():
    print("Running self-test with dummy data (no real CT data needed)...\n")
    tmp_dir = tempfile.mkdtemp()
    try:
        case_ids = [1, 2]
        vol_shape = (48, 48, 48)  # small on purpose, just to test plumbing

        for case_id in case_ids:
            case_dir = os.path.join(tmp_dir, f"case{case_id}")
            os.makedirs(case_dir, exist_ok=True)
            for name in ["T00", "T50"] + INTERMEDIATE_PHASES:
                arr = np.random.uniform(-1000, 1000, size=vol_shape).astype(np.float32)
                np.save(os.path.join(case_dir, f"{name}.npy"), arr)

        dataset = PhasePatchDataset(data_root=tmp_dir, case_ids=case_ids, patch_size=32)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = UNet3DFiLM(in_channels=3, out_channels=1, base_ch=8)  # tiny, just for speed
        scheduler = NoiseScheduler(num_timesteps=1000)

        history = train(model, dataloader, scheduler, device="cpu", epochs=2, lr=1e-3)

        assert len(history) == 2, "Expected 2 epochs of loss history"
        assert all(h == h for h in history), "Loss became NaN"
        print("\nSelf-test passed: data loads, patches crop correctly, "
              "model trains, loss is a real number.")
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true",
                         help="Run with dummy data to sanity-check the pipeline")
    parser.add_argument("--data-root", type=str, default="dirlab_preprocessed")
    parser.add_argument("--synthetic-root", type=str, default=None)
    parser.add_argument("--case-ids", type=int, nargs="+", default=[1, 2, 5, 6, 7, 8, 10])
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    else:
        dataset = PhasePatchDataset(
            data_root=args.data_root,
            case_ids=args.case_ids,
            patch_size=args.patch_size,
            synthetic_root=args.synthetic_root,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = UNet3DFiLM(in_channels=3, out_channels=1, base_ch=args.base_ch)
        scheduler = NoiseScheduler(num_timesteps=1000, device=args.device)

        print(f"Training samples: {len(dataset)}")
        train(model, dataloader, scheduler, device=args.device,
              epochs=args.epochs, lr=args.lr, checkpoint_dir=args.checkpoint_dir)
