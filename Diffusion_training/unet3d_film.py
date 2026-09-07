"""
3D U-Net with FiLM conditioning — noise predictor for the phase-conditioned
diffusion model.

What this network does at every training step:
  input  = concat([noisy_Txx, T00, T50])   -> 3 channels
  cond   = phase value (0.1-0.9) + diffusion timestep (0-1000)
  output = predicted noise, same shape as noisy_Txx (1 channel)

FiLM (Feature-wise Linear Modulation) is applied inside every residual
block: output = gamma * features + beta, where gamma/beta come from the
phase + timestep embedding. This is what makes the same network generate
T20 differently from T70 -- and know how noisy the current input is.

No attention module yet -- that's an optional add-on for later, only if
there's time left after this trains cleanly.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Embeddings: turn a single number (phase, or timestep) into a vector
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal position embedding, same idea as in Transformers
    and in DDPM's timestep embedding. Turns a scalar into a vector of sines
    and cosines at different frequencies, so the network can tell 0.31 apart
    from 0.29 easily."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,) scalar values
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device).float() / half
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class ConditionEncoder(nn.Module):
    """Combines phase embedding + timestep embedding into one conditioning
    vector that every residual block will read from."""

    def __init__(self, emb_dim=128, cond_dim=256):
        super().__init__()
        self.phase_emb = SinusoidalEmbedding(emb_dim)
        self.time_emb = SinusoidalEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, phase, timestep):
        # phase: (B,) float in [0,1]-ish, timestep: (B,) float/int
        pe = self.phase_emb(phase)
        te = self.time_emb(timestep)
        return self.mlp(torch.cat([pe, te], dim=-1))  # (B, cond_dim)


# ---------------------------------------------------------------------------
# FiLM-conditioned residual block
# ---------------------------------------------------------------------------

class FiLMResBlock3D(nn.Module):
    """Two 3D convs with a FiLM modulation in between, plus a skip connection.
    This is the basic repeated unit of the whole network."""

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        # Produces gamma and beta for this block's channel width
        self.film = nn.Linear(cond_dim, out_ch * 2)

        # If channel count changes, skip connection needs a 1x1 conv to match
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))

        gamma, beta = self.film(cond).chunk(2, dim=-1)  # each (B, out_ch)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        h = gamma * h + beta  # FiLM: output = gamma * feature + beta

        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Down/up sampling
# ---------------------------------------------------------------------------

class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ---------------------------------------------------------------------------
# The full U-Net
# ---------------------------------------------------------------------------

class UNet3DFiLM(nn.Module):
    """
    in_channels=3  -> noisy_Txx (1) + T00 (1) + T50 (1), concatenated
    out_channels=1 -> predicted noise, same shape as noisy_Txx

    base_ch controls model size -- start small (16 or 24) given the small
    dataset and GPU memory limits from patch-based training.
    """

    def __init__(self, in_channels=3, out_channels=1, base_ch=16, cond_dim=256):
        super().__init__()
        self.cond_encoder = ConditionEncoder(cond_dim=cond_dim)

        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder
        self.in_conv = nn.Conv3d(in_channels, ch1, kernel_size=3, padding=1)
        self.enc1 = FiLMResBlock3D(ch1, ch1, cond_dim)
        self.down1 = Downsample3D(ch1)

        self.enc2 = FiLMResBlock3D(ch1, ch2, cond_dim)
        self.down2 = Downsample3D(ch2)

        self.enc3 = FiLMResBlock3D(ch2, ch3, cond_dim)
        self.down3 = Downsample3D(ch3)

        # Bottleneck (this is where Phase-Aware Attention would go later)
        self.bottleneck = FiLMResBlock3D(ch3, ch4, cond_dim)

        # Decoder (mirrors encoder, with skip connections)
        self.up3 = Upsample3D(ch4)
        self.dec3 = FiLMResBlock3D(ch4 + ch3, ch3, cond_dim)

        self.up2 = Upsample3D(ch3)
        self.dec2 = FiLMResBlock3D(ch3 + ch2, ch2, cond_dim)

        self.up1 = Upsample3D(ch2)
        self.dec1 = FiLMResBlock3D(ch2 + ch1, ch1, cond_dim)

        self.out_conv = nn.Conv3d(ch1, out_channels, kernel_size=1)

    def forward(self, x, phase, timestep):
        """
        x:        (B, 3, D, H, W)  -- noisy_Txx + T00 + T50 concatenated
        phase:    (B,)             -- e.g. 0.3 for T30
        timestep: (B,)             -- e.g. 437 for a 1000-step schedule
        """
        cond = self.cond_encoder(phase, timestep)

        x0 = self.in_conv(x)
        e1 = self.enc1(x0, cond)          # skip 1
        x1 = self.down1(e1)

        e2 = self.enc2(x1, cond)          # skip 2
        x2 = self.down2(e2)

        e3 = self.enc3(x2, cond)          # skip 3
        x3 = self.down3(e3)

        b = self.bottleneck(x3, cond)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), cond)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond)

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Quick sanity check -- confirms shapes work before you touch real data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet3DFiLM(in_channels=3, out_channels=1, base_ch=16)

    # A small patch, not a full volume -- full volumes won't fit in GPU memory.
    # 64^3 is a reasonable starting patch size for a P100.
    batch_size = 2
    patch = 64
    x = torch.randn(batch_size, 3, patch, patch, patch)
    phase = torch.tensor([0.3, 0.7])
    timestep = torch.tensor([437.0, 120.0])

    out = model(x, phase, timestep)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters:   {n_params:,}")
    assert out.shape == (batch_size, 1, patch, patch, patch), "Shape mismatch!"
    print("Sanity check passed: output shape matches input spatial shape.")
