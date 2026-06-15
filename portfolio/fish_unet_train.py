#!/usr/bin/env python3
"""Train the 3-D U-Net on REAL FISH volumes with weak labels — end to end.

This turns the scaffold in fish_segmentation_unet3d.py into an actually-trained
model on the project's real confocal data, using *weak* labels derived from the
existing spectral-unmixing pipeline (no human voxel annotation exists yet).

Pipeline
--------
1. make-labels: read the 10 unmixed FISH volumes results/fish_3d/ic/phi_xyz_*.npy
   (shape X,Y,Z,5 = per-voxel species intensities). Weak label per voxel =
   argmax over species, with a background class where total signal is low.
   Input channels = the 5 normalized species maps + mild acquisition noise, so
   the task is genuine cross-FOV generalization, not a trivial argmax identity.
2. train: hold out whole FOVs (different patient state CH/DH and culture day),
   train on the rest, report per-class Dice on the held-out FOVs. Holding out
   ENTIRE volumes is the honest generalization test — the model must transfer
   the labeling rule across intensity scales it never saw.

CPU-runnable (downsampled) for a reproducible smoke-grade run:
    python portfolio/fish_unet_train.py --epochs 12 --downsample 2 --zpatch 16
Full-resolution training belongs on the GPU — see jobs/fish_unet_gpu.sh.

Honesty note: weak labels inherit the unmixing's errors. Dice here measures
"can a 3-D CNN learn and generalize the unmixing decision spatially", NOT
agreement with a gold-standard human mask (which doesn't exist). Stated plainly
in PORTFOLIO.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # import the architecture from the sibling scaffold
from fish_segmentation_unet3d import UNet3D, combined_loss, dice_per_class  # noqa: E402

REPO = HERE.parent
IC_DIR = REPO / "results" / "fish_3d" / "ic"
N_SPECIES = 5
N_CLASSES = N_SPECIES + 1  # +1 background
SPECIES = ["So", "An", "Vp", "Fn", "Pg"]
CLASS_NAMES = ["bg"] + SPECIES
BG_THRESH = 0.5  # total species signal below this = background


def load_volume(path: Path, downsample: int, zpatch: int, noise: float, rng):
    """Return (x: C,Z,H,W float32 input), (y: Z,H,W int64 weak label)."""
    phi = np.load(path).astype(np.float32)          # (X, Y, Z, 5)
    if downsample > 1:
        phi = phi[::downsample, ::downsample]        # decimate XY
    total = phi.sum(-1)                              # (X, Y, Z)
    lab = phi.argmax(-1) + 1                         # 1..5
    lab[total < BG_THRESH] = 0                       # background
    # normalize per-channel to [0,1]-ish, add acquisition noise for a real task
    denom = max(float(phi.max()), 1e-6)
    x = phi / denom
    x = x + rng.normal(0.0, noise, size=x.shape).astype(np.float32)
    x = np.clip(x, 0.0, None)
    # to torch layout (C, Z, H, W): phi is (X,Y,Z,C) -> (C,Z,X,Y)
    x = np.transpose(x, (3, 2, 0, 1))
    lab = np.transpose(lab, (2, 0, 1))              # (Z, X, Y)
    # fix Z to zpatch (center crop or pad) so volumes batch together
    z = x.shape[1]
    if z >= zpatch:
        s = (z - zpatch) // 2
        x, lab = x[:, s:s + zpatch], lab[s:s + zpatch]
    else:
        padz = zpatch - z
        x = np.pad(x, ((0, 0), (0, padz), (0, 0), (0, 0)))
        lab = np.pad(lab, ((0, padz), (0, 0), (0, 0)))
    return torch.from_numpy(x.copy()), torch.from_numpy(lab.astype(np.int64).copy())


class FishWeak(Dataset):
    def __init__(self, paths, downsample, zpatch, noise, seed):
        self.items = []
        rng = np.random.default_rng(seed)
        for p in paths:
            self.items.append(load_volume(p, downsample, zpatch, noise, rng))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def class_weights(ds, eps=1e-3):
    counts = torch.zeros(N_CLASSES)
    for _, y in ds:
        counts += torch.bincount(y.flatten(), minlength=N_CLASSES)
    w = 1.0 / (counts + eps)
    return (w / w.sum() * N_CLASSES)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--downsample", type=int, default=2, help="XY decimation (CPU: 2)")
    ap.add_argument("--zpatch", type=int, default=16)
    ap.add_argument("--base", type=int, default=8, help="U-Net width (CPU: 8, GPU: 16-32)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--holdout", nargs="*", default=["phi_xyz_CH_d21", "phi_xyz_DH_d21"],
                    help="volume stems held out for the generalization test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    all_paths = sorted(IC_DIR.glob("phi_xyz_*.npy"))
    if not all_paths:
        sys.exit(f"no FISH volumes in {IC_DIR}")
    hold = set(args.holdout)
    train_p = [p for p in all_paths if p.stem not in hold]
    val_p = [p for p in all_paths if p.stem in hold]
    print(f"train FOVs: {[p.stem for p in train_p]}")
    print(f"held-out  : {[p.stem for p in val_p]}")

    train_ds = FishWeak(train_p, args.downsample, args.zpatch, args.noise, args.seed)
    val_ds = FishWeak(val_p, args.downsample, args.zpatch, args.noise, args.seed + 99)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(cin=N_SPECIES, n_classes=N_CLASSES, base=args.base).to(dev)
    ce_w = class_weights(train_ds).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()):,}  "
          f"class_weights={[round(v,2) for v in ce_w.tolist()]}")

    def evaluate():
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_ds:
                logits = model(x[None].to(dev))
                dices.append(dice_per_class(logits, y[None].to(dev)))
        return np.nanmean(np.array(dices), axis=0)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = combined_loss(model(x), y, ce_w=ce_w)
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep == 0 or (ep + 1) % 3 == 0 or ep == args.epochs - 1:
            d = evaluate()
            macro = np.nanmean(d[1:])  # exclude background
            print(f"epoch {ep+1:2d}/{args.epochs}  train_loss={tot/len(train_dl):.4f}  "
                  f"held-out macro-Dice(species)={macro:.3f}  "
                  f"per-class={dict(zip(CLASS_NAMES, [round(float(v),2) for v in d]))}")

    ckpt = HERE / "fish_unet3d_weak.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"saved {ckpt}")
    print("NOTE: weak-label Dice = cross-FOV generalization of the unmixing rule, "
          "not agreement with a human gold mask (none exists yet).")


if __name__ == "__main__":
    main()
