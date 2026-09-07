"""
DDIM sampling -- turns a TRAINED model into an actual generated lung phase.

Training (diffusion_training.py) only teaches the model to guess noise.
This script is the other half: start from pure random noise, and repeatedly
ask the trained model "what noise is this?", subtract a bit, and repeat --
until a clean, generated phase comes out the other end.

This is Section 8 of the architecture notes ("The Noise Subtractor"),
using DDIM instead of full DDPM so it only takes ~50 steps instead of 1000.

Usage:
  --selftest    builds a tiny untrained model and dummy anchors, runs the
                sampling loop, and confirms the output shape/values make
                sense -- no real checkpoint or CT data needed.
  (default)     loads a real checkpoint from diffusion_training.py and
                generates a real phase from real T00/T50 anchors.
"""

import argparse
import os

import numpy as np
import torch

from unet3d_film import UNet3DFiLM
from diffusion_training import NoiseScheduler


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

class DDIMSampler:
    """Wraps a trained model + noise schedule and runs the reverse
    (denoising) process using DDIM's larger, calculated steps instead of
    DDPM's 1000 tiny ones."""

    def __init__(self, model, scheduler: NoiseScheduler, num_steps=50, device="cpu"):
        self.model = model.to(device).eval()
        self.scheduler = scheduler
        self.device = device

        # Pick num_steps evenly spaced timesteps out of the full schedule,
        # e.g. 50 steps out of 1000 -> [980, 960, ..., 20, 0]
        full_T = scheduler.T
        step_indices = np.linspace(0, full_T - 1, num_steps).round().astype(int)
        self.timesteps = list(step_indices[::-1])  # descending: noisy -> clean

    @torch.no_grad()
    def generate(self, t00, t50, phase, volume_shape):
        """
        t00, t50: (1, 1, D, H, W) tensors -- the clean boundary anchors
        phase:    float, e.g. 0.3 for T30
        volume_shape: (D, H, W) -- shape of the phase to generate

        Returns: (1, 1, D, H, W) tensor -- the generated phase
        """
        device = self.device
        x = torch.randn((1, 1, *volume_shape), device=device)  # start: pure noise
        phase_t = torch.tensor([phase], device=device, dtype=torch.float32)

        for i, t_cur in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            t_batch = torch.tensor([t_cur], device=device, dtype=torch.float32)
            model_input = torch.cat([x, t00, t50], dim=1)  # (1,3,D,H,W)
            predicted_noise = self.model(model_input, phase_t, t_batch)

            alpha_bar_t = self.scheduler.alpha_bars[t_cur]
            alpha_bar_prev = self.scheduler.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Predict the clean image from the current noisy one + predicted noise
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # keep values in a sane range

            if t_prev >= 0:
                # Deterministic DDIM step (eta=0): move to the less-noisy image
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
            else:
                x = x0_pred  # final step: this is the fully generated phase

        return x


# ---------------------------------------------------------------------------
# Self-test: confirms the sampling loop runs end-to-end on dummy data
# ---------------------------------------------------------------------------

def run_selftest():
    print("Running self-test with an untrained model and dummy anchors...\n")

    device = "cpu"
    volume_shape = (32, 32, 32)  # small on purpose, just testing plumbing

    model = UNet3DFiLM(in_channels=3, out_channels=1, base_ch=8)
    scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    sampler = DDIMSampler(model, scheduler, num_steps=10, device=device)  # few steps, fast

    t00 = torch.randn((1, 1, *volume_shape))
    t50 = torch.randn((1, 1, *volume_shape))

    output = sampler.generate(t00, t50, phase=0.3, volume_shape=volume_shape)

    assert output.shape == (1, 1, *volume_shape), "Output shape mismatch!"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf!"
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output value range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("\nSelf-test passed: sampling loop runs, output shape is correct, "
          "no NaNs. (Values are meaningless here since the model is untrained "
          "-- this only confirms the mechanics work.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a .pt file saved by diffusion_training.py")
    parser.add_argument("--data-root", type=str, default="dirlab_preprocessed")
    parser.add_argument("--case-id", type=int, default=1)
    parser.add_argument("--phase", type=str, default="T30", help="e.g. T30")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-path", type=str, default="generated_phase.npy")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required outside of --selftest mode")

        t00 = np.load(os.path.join(args.data_root, f"case{args.case_id}", "T00.npy")).astype(np.float32)
        t50 = np.load(os.path.join(args.data_root, f"case{args.case_id}", "T50.npy")).astype(np.float32)

        # Same normalization used in training -- adjust if your data differs
        def norm(x):
            return np.clip(x, -1000, 1000) / 1000.0

        t00_t = torch.from_numpy(norm(t00)).unsqueeze(0).unsqueeze(0)
        t50_t = torch.from_numpy(norm(t50)).unsqueeze(0).unsqueeze(0)

        model = UNet3DFiLM(in_channels=3, out_channels=1, base_ch=args.base_ch)
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

        scheduler = NoiseScheduler(num_timesteps=1000, device=args.device)
        sampler = DDIMSampler(model, scheduler, num_steps=args.num_steps, device=args.device)

        phase_value = int(args.phase[1:]) / 100.0
        generated = sampler.generate(t00_t.to(args.device), t50_t.to(args.device),
                                      phase=phase_value, volume_shape=t00.shape)

        result = generated.squeeze().cpu().numpy() * 1000.0  # undo normalization -> back to HU-ish scale
        np.save(args.output_path, result)
        print(f"Generated {args.phase} for case {args.case_id}, saved to {args.output_path}")
