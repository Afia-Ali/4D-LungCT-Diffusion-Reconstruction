"""
TRE evaluation, sandboxed.

Original version by Diba. This version adds four things on top, all
additive -- nothing about her core landmark-matching logic, oracle/baseline
sanity checks, or selftest changed:

  1. --model-type {film, attention}: the original only knew how to load
     UNet3DFiLM. The best checkpoint right now is the attention variant
     (UNet3DFiLMAttention, verified 0.0378 mean MSE vs baseline 0.1271), so
     that needs to be loadable too.
  2. --guidance-scale / --num-samples: the original always did a single,
     unguided generate() call. The verified MSE numbers for both FiLM-only
     and attention were produced with guidance_scale=2.0 and 5-sample
     averaging (generate_averaged in ddim_sampling.py). Evaluating TRE with
     plain single-sample, no-guidance generation would not be comparing
     like with like against those numbers.
  3. --patch default changed 32 -> 48, matching PATCH_SIZE used to actually
     train the checkpoints (32 still works, divisible by 8, but would be
     evaluating the model at a patch size it never trained on).
  4. Data root default points at the current Drive account's
     "CSE499 Dataset" folder instead of a bare "dirlab_preprocessed", since
     the old path stopped resolving after the account switch (this is the
     same issue that made TRE_Evaluation.ipynb fail at the checkpoint-load
     step -- never got past that to produce any numbers).

This file is standalone on purpose. It does NOT modify unet3d_film.py,
unet3d_film_attention.py, diffusion_training.py or ddim_sampling.py. It
only imports from them, read only. Prove it works here, then merge into
the main repo.

The question this answers: how far, in millimetres, is a landmark in the
generated phase from where DIR-Lab says it actually is.

Four modes, meant to be run in this order:

  --selftest   dummy data, no CT and no checkpoint needed. Confirms the
               mechanics run and the maths is sane.

  --oracle     THE IMPORTANT ONE. Runs the whole TRE measurement against the
               REAL T30 volume instead of a generated one. If the template
               matcher works, TRE here should be near zero. If it is large,
               your EVALUATION is broken, not your model. Never trust a
               model score before this passes.

  --baseline   TRE if you assume nothing moves at all, meaning you just
               reuse the T00 landmark position. This is the number to beat.
               A model that scores worse than this has learned nothing.

  --evaluate   the real thing. Generates phases and scores them.

Usage:
  python tre_pipeline.py --selftest
  python tre_pipeline.py --oracle   --case-id 3 --phase T30
  python tre_pipeline.py --baseline --case-id 3 --phase T30
  python tre_pipeline.py --evaluate --case-id 3 --phase T30 \\
      --checkpoint "/content/drive/MyDrive/CSE499 Dataset/checkpoints_attention/epoch_80.pt" \\
      --model-type attention --guidance-scale 2.0 --num-samples 5 --patch 48
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Config you MUST check before trusting any number
# ---------------------------------------------------------------------------

# Voxel spacing in mm as (z, y, x), for the ORIGINAL DIR-Lab grid.
# WARNING: these are the commonly quoted DIR-Lab values but you must verify
# them against the dataset documentation for your copy. A wrong spacing
# scales every TRE you report. Do not skip this.
DIRLAB_SPACING_ORIGINAL = {
    1:  (2.5, 0.97, 0.97),
    2:  (2.5, 1.16, 1.16),
    3:  (2.5, 1.15, 1.15),
    4:  (2.5, 1.13, 1.13),
    5:  (2.5, 1.10, 1.10),
    6:  (2.5, 0.97, 0.97),
    7:  (2.5, 0.97, 0.97),
    8:  (2.5, 0.97, 0.97),
    9:  (2.5, 0.97, 0.97),
    10: (2.5, 0.97, 0.97),
}

# Cases 6 to 10 were captured at 512 by 512 and your preprocessing resized
# them to 256 by 256. Landmark files are in the ORIGINAL grid, so those
# cases need their x and y halved, and their in plane spacing doubled.
RESIZED_FROM_512 = {6, 7, 8, 9, 10}

# Sampled4D only ships T00 through T50, so these are the only phases with
# ground truth landmarks available.
EVALUABLE_PHASES = ["T10", "T20", "T30", "T40"]


def get_case_geometry(case_id, volume_shape):
    """Returns (scale_zyx, spacing_zyx_mm) for converting landmark coords
    onto the .npy grid and voxel distances into millimetres."""
    if case_id not in DIRLAB_SPACING_ORIGINAL:
        raise ValueError(f"No spacing on record for case {case_id}")

    sz, sy, sx = DIRLAB_SPACING_ORIGINAL[case_id]

    if case_id in RESIZED_FROM_512:
        # in plane halved in size, so coords scale by 0.5 and each remaining
        # voxel now covers twice the distance
        scale = (1.0, 0.5, 0.5)
        spacing = (sz, sy * 2.0, sx * 2.0)
    else:
        scale = (1.0, 1.0, 1.0)
        spacing = (sz, sy, sx)

    return np.array(scale), np.array(spacing)


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

def load_landmarks_zyx(path, scale_zyx):
    """DIR-Lab landmark files are whitespace separated x y z voxel coords.
    numpy arrays index as (z, y, x), so we flip, then apply the resize scale.

    Returns float array (N, 3) in (z, y, x) order on the .npy grid.
    """
    raw = np.loadtxt(path)            # (N, 3) as x, y, z
    if raw.ndim == 1:
        raw = raw[None, :]
    zyx = raw[:, [2, 1, 0]].astype(np.float64)
    return zyx * scale_zyx


def check_landmarks_in_bounds(landmarks_zyx, volume_shape, label):
    """Loud failure beats a quiet wrong answer. If landmarks fall outside the
    volume, the scaling is wrong and every TRE after this is meaningless."""
    shape = np.array(volume_shape)
    out = ((landmarks_zyx < 0) | (landmarks_zyx >= shape)).any(axis=1)
    n_out = int(out.sum())
    if n_out:
        print(f"  WARNING [{label}]: {n_out} of {len(landmarks_zyx)} landmarks "
              f"fall outside the volume {tuple(volume_shape)}. "
              "Coordinate scaling is probably wrong.")
    else:
        print(f"  [{label}] all {len(landmarks_zyx)} landmarks inside the volume.")
    return n_out == 0


# ---------------------------------------------------------------------------
# Patch helpers and template matching
# ---------------------------------------------------------------------------

def extract_patch_centered(volume, center_zyx, half_size):
    """Returns a patch centred on center_zyx. Zero padded when the window
    runs off the edge, so the shape is always predictable.

    NOTE: cell 24 in the debugging notebook returned a clipped patch here,
    which silently changed shape near edges and then got skipped by the
    shape check in the matcher. That quietly dropped edge landmarks from
    the score. Padding fixes it.
    """
    hz, hy, hx = half_size
    out_shape = (2 * hz + 1, 2 * hy + 1, 2 * hx + 1)
    patch = np.zeros(out_shape, dtype=volume.dtype)

    z, y, x = [int(round(c)) for c in center_zyx]

    z0, z1 = z - hz, z + hz + 1
    y0, y1 = y - hy, y + hy + 1
    x0, x1 = x - hx, x + hx + 1

    vz0, vy0, vx0 = max(0, z0), max(0, y0), max(0, x0)
    vz1 = min(volume.shape[0], z1)
    vy1 = min(volume.shape[1], y1)
    vx1 = min(volume.shape[2], x1)

    if vz0 >= vz1 or vy0 >= vy1 or vx0 >= vx1:
        return patch  # entirely outside, return zeros

    patch[vz0 - z0:vz1 - z0,
          vy0 - y0:vy1 - y0,
          vx0 - x0:vx1 - x0] = volume[vz0:vz1, vy0:vy1, vx0:vx1]
    return patch


def normalized_cross_correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def find_landmark(target_volume, template_patch, search_center_zyx,
                  search_radius=(4, 12, 12), patch_half_size=(2, 5, 5),
                  step=1):
    """Slides the template over a local window and returns the best matching
    centre plus its score.

    step=1 by default. Cell 24 used step 2 on y and x, which caps accuracy at
    about one voxel of error before the model is even involved. On a 1.15 mm
    grid that is roughly 1 mm of self inflicted TRE. Keep step 1 unless you
    genuinely need the speed.
    """
    best_score = -np.inf
    best_pos = tuple(int(round(c)) for c in search_center_zyx)

    sz, sy, sx = best_pos
    rz, ry, rx = search_radius

    for dz in range(-rz, rz + 1, step):
        for dy in range(-ry, ry + 1, step):
            for dx in range(-rx, rx + 1, step):
                center = (sz + dz, sy + dy, sx + dx)
                cand = extract_patch_centered(target_volume, center, patch_half_size)
                score = normalized_cross_correlation(template_patch, cand)
                if score > best_score:
                    best_score = score
                    best_pos = center

    return np.array(best_pos, dtype=np.float64), best_score


# ---------------------------------------------------------------------------
# The TRE measurement itself
# ---------------------------------------------------------------------------

def compute_tre(t00_volume, target_volume, t00_landmarks_zyx, gt_landmarks_zyx,
                spacing_zyx, search_radius=(4, 12, 12),
                patch_half_size=(2, 5, 5), step=1, verbose=False):
    """For each landmark: cut a template around it in T00, find the best match
    in target_volume, and measure how far that is from the ground truth
    position, in millimetres.

    Returns (tre_array_mm, match_scores).
    """
    tre_mm = []
    scores = []

    for i in range(len(t00_landmarks_zyx)):
        template = extract_patch_centered(t00_volume, t00_landmarks_zyx[i], patch_half_size)

        found_zyx, score = find_landmark(
            target_volume, template,
            search_center_zyx=t00_landmarks_zyx[i],
            search_radius=search_radius,
            patch_half_size=patch_half_size,
            step=step,
        )

        diff_voxels = found_zyx - gt_landmarks_zyx[i]
        diff_mm = diff_voxels * spacing_zyx
        d = float(np.linalg.norm(diff_mm))

        tre_mm.append(d)
        scores.append(score)

        if verbose and i < 5:
            print(f"    lm{i:02d}  found {found_zyx.astype(int)}  "
                  f"gt {gt_landmarks_zyx[i].astype(int)}  "
                  f"TRE {d:6.2f} mm  ncc {score:.3f}")

    return np.array(tre_mm), np.array(scores)


def compute_baseline_tre(t00_landmarks_zyx, gt_landmarks_zyx, spacing_zyx):
    """TRE if you assume the anatomy does not move at all. This is the floor
    your model has to beat to have done anything useful."""
    diff_mm = (t00_landmarks_zyx - gt_landmarks_zyx) * spacing_zyx
    return np.linalg.norm(diff_mm, axis=1)


def report(name, tre, scores=None):
    print(f"\n  {name}")
    print(f"    mean   {tre.mean():7.2f} mm")
    print(f"    median {np.median(tre):7.2f} mm")
    print(f"    std    {tre.std():7.2f} mm")
    print(f"    max    {tre.max():7.2f} mm")
    print(f"    n      {len(tre)}")
    if scores is not None:
        print(f"    mean ncc {scores.mean():.3f}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_case(data_root, case_id, phase):
    d = os.path.join(data_root, f"case{case_id}")
    t00 = np.load(os.path.join(d, "T00.npy")).astype(np.float32)
    t50 = np.load(os.path.join(d, "T50.npy")).astype(np.float32)
    tgt = np.load(os.path.join(d, f"{phase}.npy")).astype(np.float32)
    return t00, t50, tgt


def landmark_paths(landmark_root, case_id, phase):
    d = os.path.join(landmark_root, f"Case{case_id}Pack", "Sampled4D")
    return (os.path.join(d, f"case{case_id}_4D-75_T00.txt"),
            os.path.join(d, f"case{case_id}_4D-75_{phase}.txt"))


def prepare(data_root, landmark_root, case_id, phase):
    t00, t50, tgt = load_case(data_root, case_id, phase)
    scale, spacing = get_case_geometry(case_id, t00.shape)

    p_t00, p_tgt = landmark_paths(landmark_root, case_id, phase)
    lm_t00 = load_landmarks_zyx(p_t00, scale)
    lm_gt = load_landmarks_zyx(p_tgt, scale)

    print(f"\n  case{case_id}  {phase}")
    print(f"  volume shape (D,H,W): {t00.shape}")
    print(f"  coord scale (z,y,x):  {tuple(scale)}")
    print(f"  spacing mm  (z,y,x):  {tuple(np.round(spacing, 3))}")
    check_landmarks_in_bounds(lm_t00, t00.shape, "T00 landmarks")
    check_landmarks_in_bounds(lm_gt, t00.shape, f"{phase} landmarks")

    return t00, t50, tgt, lm_t00, lm_gt, spacing


# ---------------------------------------------------------------------------
# Generation, landmark centred so we never build a full volume
# ---------------------------------------------------------------------------

def load_model(model_type, checkpoint, base_ch, device):
    """model_type: 'film' loads UNet3DFiLM (the original baseline/FiLM-only
    architecture). 'attention' loads UNet3DFiLMAttention (FiLM + Phase-Aware
    Attention at the bottleneck, Section 7 -- this is the currently-best
    checkpoint, 0.0378 mean MSE vs 0.1271 baseline). Both classes share the
    same forward(x, phase, timestep) signature, so DDIMSampler doesn't care
    which one it's holding -- only the loading step needs to know."""
    import torch

    if model_type == "film":
        from unet3d_film import UNet3DFiLM as ModelClass
    elif model_type == "attention":
        from unet3d_film_attention import UNet3DFiLMAttention as ModelClass
    else:
        raise ValueError(f"model_type must be 'film' or 'attention', got {model_type!r}")

    model = ModelClass(in_channels=3, out_channels=1, base_ch=base_ch)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def generate_around_landmarks(checkpoint, t00, t50, landmarks_zyx, phase_value,
                              patch=48, base_ch=16, num_steps=50, device="cpu",
                              main_dir=".", model_type="film",
                              guidance_scale=1.0, num_samples=1):
    """Generates one patch per landmark, centred on that landmark, and pastes
    the results into an otherwise empty volume. Cheap, and enough for TRE.

    The UNet has 3 downsample stages so patch must be divisible by 8.

    guidance_scale / num_samples: pass the SAME values used to produce your
    reported MSE numbers (guidance_scale=2.0, num_samples=5 for both the
    FiLM-only and attention checkpoints) or this TRE number isn't measuring
    the same model behaviour you've already reported elsewhere. Defaults
    here are 1.0 / 1 (off) so a bare call is still cheap for a first smoke
    test -- turn them on deliberately for a real --evaluate run.
    """
    import torch
    sys.path.insert(0, main_dir)
    from diffusion_training import NoiseScheduler
    from ddim_sampling import DDIMSampler

    if patch % 8 != 0:
        raise ValueError(f"patch {patch} must be divisible by 8")

    model = load_model(model_type, checkpoint, base_ch, device)
    scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    sampler = DDIMSampler(model, scheduler, num_steps=num_steps, device=device)

    canvas = np.zeros_like(t00)
    half = patch // 2

    for i, lm in enumerate(landmarks_zyx):
        z, y, x = [int(round(c)) for c in lm]
        z0 = int(np.clip(z - half, 0, max(0, t00.shape[0] - patch)))
        y0 = int(np.clip(y - half, 0, max(0, t00.shape[1] - patch)))
        x0 = int(np.clip(x - half, 0, max(0, t00.shape[2] - patch)))

        def cut(v):
            return v[z0:z0 + patch, y0:y0 + patch, x0:x0 + patch]

        a = cut(t00)
        b = cut(t50)
        if a.shape != (patch, patch, patch):
            continue  # volume smaller than patch on some axis, skip

        # NOTE: no normalization here. The .npy files are already in [-1, 1]
        # from preprocessing. This is the same double normalization trap that
        # was patched in diffusion_training.py. ddim_sampling.py still has the
        # old clip and divide in its __main__ block, which is why this file
        # calls DDIMSampler directly instead of running that script.
        t00_t = torch.from_numpy(a).float()[None, None].to(device)
        t50_t = torch.from_numpy(b).float()[None, None].to(device)

        if num_samples > 1:
            out, _, _ = sampler.generate_averaged(
                t00_t, t50_t, phase=phase_value,
                volume_shape=(patch, patch, patch),
                num_samples=num_samples, guidance_scale=guidance_scale,
                reduction="mean",
            )
        else:
            out = sampler.generate(t00_t, t50_t, phase=phase_value,
                                   volume_shape=(patch, patch, patch),
                                   guidance_scale=guidance_scale)

        canvas[z0:z0 + patch, y0:y0 + patch, x0:x0 + patch] = \
            out.squeeze().detach().cpu().numpy()

        if (i + 1) % 10 == 0:
            print(f"    generated around {i + 1}/{len(landmarks_zyx)} landmarks")

    return canvas


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_oracle(args):
    """Score the REAL target phase. Near zero TRE means the evaluation works.
    Large TRE means the evaluation is broken and no model score is trustworthy."""
    print("\n=== ORACLE: scoring the REAL phase, not a generated one ===")
    t00, t50, tgt, lm_t00, lm_gt, spacing = prepare(
        args.data_root, args.landmark_root, args.case_id, args.phase)

    base = compute_baseline_tre(lm_t00, lm_gt, spacing)
    tre, scores = compute_tre(t00, tgt, lm_t00, lm_gt, spacing,
                              step=args.step, verbose=True)

    report("no motion baseline", base)
    report("ORACLE on real phase", tre, scores)

    print("\n  VERDICT")
    if tre.mean() < 2.0:
        print("    Evaluation pipeline looks sound. Model scores are trustworthy.")
    elif tre.mean() < base.mean():
        print("    Partly working. Better than the baseline but still high. "
              "Try a bigger patch_half_size or search_radius before "
              "blaming the model.")
    else:
        print("    BROKEN. The matcher cannot even find landmarks in the real "
              "phase, so it cannot judge a generated one. Fix this first. "
              "Usual suspects: coordinate scaling, spacing, or a search "
              "radius smaller than the actual motion.")


def run_baseline(args):
    print("\n=== BASELINE: assume nothing moves ===")
    t00, t50, tgt, lm_t00, lm_gt, spacing = prepare(
        args.data_root, args.landmark_root, args.case_id, args.phase)
    base = compute_baseline_tre(lm_t00, lm_gt, spacing)
    report("no motion baseline", base)
    print("\n  Your model must score BELOW this to have learned anything.")


def run_evaluate(args):
    print("\n=== EVALUATE: scoring generated phases ===")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --evaluate")

    t00, t50, tgt, lm_t00, lm_gt, spacing = prepare(
        args.data_root, args.landmark_root, args.case_id, args.phase)

    phase_value = int(args.phase[1:]) / 100.0
    print(f"\n  generating around {len(lm_t00)} landmarks, patch {args.patch}, "
          f"model_type={args.model_type}, guidance_scale={args.guidance_scale}, "
          f"num_samples={args.num_samples}")
    gen = generate_around_landmarks(
        args.checkpoint, t00, t50, lm_t00, phase_value,
        patch=args.patch, base_ch=args.base_ch, num_steps=args.num_steps,
        device=args.device, main_dir=args.main_dir, model_type=args.model_type,
        guidance_scale=args.guidance_scale, num_samples=args.num_samples)

    base = compute_baseline_tre(lm_t00, lm_gt, spacing)
    oracle, _ = compute_tre(t00, tgt, lm_t00, lm_gt, spacing, step=args.step)
    tre, scores = compute_tre(t00, gen, lm_t00, lm_gt, spacing,
                              step=args.step, verbose=True)

    report("no motion baseline", base)
    report("oracle on real phase", oracle)
    report("MODEL on generated phase", tre, scores)

    print("\n  VERDICT")
    if tre.mean() < base.mean():
        print(f"    Model beats the no motion baseline by "
              f"{base.mean() - tre.mean():.2f} mm. It learned something real.")
    else:
        print("    Model does NOT beat the no motion baseline. As it stands "
              "you would do better assuming the lung is static.")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["landmark", "tre_model_mm", "tre_oracle_mm",
                        "tre_baseline_mm", "ncc"])
            for i in range(len(tre)):
                w.writerow([i, f"{tre[i]:.4f}", f"{oracle[i]:.4f}",
                            f"{base[i]:.4f}", f"{scores[i]:.4f}"])
        print(f"\n  per landmark results written to {args.save_csv}")


def run_selftest():
    """No CT data, no checkpoint. Confirms the maths and the plumbing."""
    print("Running selftest with synthetic data...\n")
    rng = np.random.default_rng(0)

    shape = (40, 96, 96)
    vol = rng.normal(0, 0.2, shape).astype(np.float32)

    # plant 20 recognisable blobs, then plant them again shifted by a known
    # amount to act as the moved phase
    n = 20
    true_shift = np.array([1.0, 4.0, -3.0])
    lm_t00 = np.stack([
        rng.integers(8, shape[0] - 8, n),
        rng.integers(16, shape[1] - 16, n),
        rng.integers(16, shape[2] - 16, n),
    ], axis=1).astype(np.float64)
    lm_gt = lm_t00 + true_shift

    moved = rng.normal(0, 0.2, shape).astype(np.float32)
    for a, b in zip(lm_t00, lm_gt):
        blob = rng.normal(0, 1.0, (5, 11, 11)).astype(np.float32) * 3.0
        for target_vol, c in ((vol, a), (moved, b)):
            z, y, x = [int(v) for v in c]
            target_vol[z - 2:z + 3, y - 5:y + 6, x - 5:x + 6] = blob

    spacing = np.array([2.5, 1.15, 1.15])

    print("  geometry check")
    scale, sp = get_case_geometry(3, shape)
    assert np.allclose(scale, [1, 1, 1]), "case3 should not be rescaled"
    scale9, sp9 = get_case_geometry(9, shape)
    assert np.allclose(scale9, [1, 0.5, 0.5]), "case9 should be halved in plane"
    assert sp9[1] > sp[1] * 1.5, "case9 in plane spacing should have doubled"
    print("    case3 scale", tuple(scale), " case9 scale", tuple(scale9), " OK")

    print("\n  padding check")
    p = extract_patch_centered(vol, (0, 0, 0), (2, 5, 5))
    assert p.shape == (5, 11, 11), f"corner patch should be padded, got {p.shape}"
    print("    corner patch shape", p.shape, "OK")

    print("\n  baseline TRE (should equal the planted shift)")
    base = compute_baseline_tre(lm_t00, lm_gt, spacing)
    expected = np.linalg.norm(true_shift * spacing)
    print(f"    got {base.mean():.3f} mm, expected {expected:.3f} mm")
    assert abs(base.mean() - expected) < 1e-6, "baseline maths is wrong"

    print("\n  oracle TRE (matcher should recover the shift, so near zero)")
    tre, scores = compute_tre(vol, moved, lm_t00, lm_gt, spacing,
                              search_radius=(3, 8, 8), step=1)
    print(f"    got {tre.mean():.3f} mm, mean ncc {scores.mean():.3f}")
    assert tre.mean() < 1.0, f"matcher failed to recover a known shift ({tre.mean():.2f} mm)"

    print("\nSelftest passed. Geometry, padding, baseline maths and the "
          "template matcher all behave. Run --oracle on real data next.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--evaluate", action="store_true")

    # NOTE: defaults below assume the CURRENT Drive account layout
    # ("CSE499 Dataset" folder), not the old pre-account-switch paths that
    # made TRE_Evaluation.ipynb fail at checkpoint load. --landmark-root has
    # NO default on purpose -- verify where Sampled4D actually lives in this
    # Drive account before running --oracle/--baseline/--evaluate, same as
    # Diba's original warning about DIRLAB_SPACING_ORIGINAL: a wrong path
    # here fails loudly (FileNotFoundError), a wrong spacing fails silently.
    ap.add_argument("--data-root", type=str,
                    default="/content/drive/MyDrive/CSE499 Dataset/dirlab_preprocessed")
    ap.add_argument("--landmark-root", type=str, default=None, required=False,
                    help="Path to the folder containing CaseNPack/Sampled4D/... "
                         "No default -- must be set explicitly, verify it exists "
                         "in the current Drive account before trusting any result.")
    ap.add_argument("--main-dir", type=str, default=".",
                    help="folder holding unet3d_film.py etc, imported read only")
    ap.add_argument("--case-id", type=int, default=3)
    ap.add_argument("--phase", type=str, default="T30", choices=EVALUABLE_PHASES)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--model-type", type=str, default="attention",
                    choices=["film", "attention"],
                    help="'attention' = current best checkpoint (UNet3DFiLMAttention). "
                         "'film' = the earlier FiLM-only checkpoint (UNet3DFiLM).")
    ap.add_argument("--guidance-scale", type=float, default=2.0,
                    help="Classifier-free guidance strength. 2.0 matches the setting "
                         "used to produce your reported verified MSE numbers for both "
                         "the FiLM-only (+21.3%%) and attention (+70.3%%) checkpoints. "
                         "Only meaningful if the checkpoint was trained with "
                         "cond_dropout_prob > 0 -- both current checkpoints were.")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="5 matches the multi-sample averaging used for your reported "
                         "MSE numbers. Set to 1 for a quick, cheaper smoke test.")
    ap.add_argument("--patch", type=int, default=48,
                    help="48 matches PATCH_SIZE the checkpoints were actually trained "
                         "at. Must stay divisible by 8.")
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--step", type=int, default=1,
                    help="search stride in voxels, 1 is most accurate")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save-csv", type=str, default=None)
    args = ap.parse_args()

    needs_landmarks = args.oracle or args.baseline or args.evaluate
    if needs_landmarks and args.landmark_root is None:
        ap.error("--landmark-root is required for --oracle/--baseline/--evaluate "
                 "(see help text -- no safe default, verify the path in your "
                 "current Drive account first)")

    if args.selftest:
        run_selftest()
    elif args.oracle:
        run_oracle(args)
    elif args.baseline:
        run_baseline(args)
    elif args.evaluate:
        run_evaluate(args)
    else:
        ap.print_help()
