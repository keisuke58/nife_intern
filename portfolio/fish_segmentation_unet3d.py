#!/usr/bin/env python3
"""3-D U-Net scaffold for FISH/CLSM biofilm segmentation (medical-imaging track).

STATUS: architecture + training loop + CPU smoke test are real and run today.
The model is NOT yet trained on annotated data — voxel-level ground-truth masks
for the Heine FISH stacks don't exist yet (current pipeline uses classical
spectral unmixing, see fish_decode.py). This file is the deep-learning bridge:
plug in masks, dispatch to the vancouver01 GPU, train. Treat as Outlook / WIP,
not a finished result.

Why it belongs in a Fujifilm medical-imaging portfolio: it shows I can stand up
a volumetric segmentation model (the REiLI/diagnostic-imaging workhorse) end to
end — 3-D conv encoder-decoder, skip connections, soft-Dice + cross-entropy
loss, class imbalance handling — and that I already own the upstream confocal
preprocessing (.lif -> per-voxel multichannel volumes) it consumes.

Smoke test (CPU, synthetic volume, ~seconds):
    python portfolio/fish_segmentation_unet3d.py --smoke-test

Real training (dispatch to GPU; needs data dir of .npy volumes + masks):
    python portfolio/fish_segmentation_unet3d.py --data-dir <dir> --epochs 50

Input volume convention: (C=4 FISH channels, D, H, W) float32, intensity-normed.
Output: per-voxel class logits over {background + 5 species} = 6 classes,
matching the 5-species decode in fish_decode.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 4          # FISH dye channels (405/488/561/647)
N_CLASSES = 6            # background + {So, An, Vp, Fn, Pg} per guild_replicator_dieckow
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Architecture: compact 3-D U-Net (2 down / 2 up). Small enough for a single
# GPU on confocal-sized patches; depth/width are CLI-tunable.
# --------------------------------------------------------------------------- #
class DoubleConv3d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, cin=IN_CHANNELS, n_classes=N_CLASSES, base=16):
        super().__init__()
        self.enc1 = DoubleConv3d(cin, base)
        self.enc2 = DoubleConv3d(base, base * 2)
        self.bottleneck = DoubleConv3d(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv3d(base * 2, base)
        self.head = nn.Conv3d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# --------------------------------------------------------------------------- #
# Loss: soft-Dice + cross-entropy. Dice handles the heavy background imbalance
# typical of biofilm masks; CE keeps gradients well-behaved early.
# --------------------------------------------------------------------------- #
def soft_dice_loss(logits, target, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, probs.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def combined_loss(logits, target, ce_w=None):
    return F.cross_entropy(logits, target, weight=ce_w) + soft_dice_loss(logits, target)


def dice_per_class(logits, target):
    pred = logits.argmax(1)
    out = []
    for c in range(logits.shape[1]):
        p, t = (pred == c), (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        out.append(2 * inter / denom if denom else float("nan"))
    return out


# --------------------------------------------------------------------------- #
def smoke_test(patch=(16, 48, 48), base=8):
    """End-to-end forward/backward on a synthetic volume — proves the graph runs."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(base=base).to(dev)
    x = torch.randn(1, IN_CHANNELS, *patch, device=dev)
    # synthetic target: threshold a smoothed channel into pseudo-classes
    with torch.no_grad():
        seed = F.avg_pool3d(x[:, :1], 3, 1, 1)
        y = (torch.clamp(seed * 2 + 2.5, 0, N_CLASSES - 1)).long().squeeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev}  params={n_params:,}  patch={patch}")
    for step in range(5):
        opt.zero_grad()
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        opt.step()
        print(f"  step {step}: loss={loss.item():.4f}")
    d = dice_per_class(model(x), y)
    print("per-class Dice (synthetic):", [f"{v:.2f}" for v in d])
    print("SMOKE TEST PASSED — architecture, loss, and backprop all run.")


def train(data_dir: Path, epochs: int, base: int, lr: float):
    """Real training entry point. Expects <data_dir>/{volumes,masks}/*.npy pairs.

    Intentionally minimal — wire in the actual Dataset once annotated masks exist;
    this documents the intended GPU run rather than fabricating results.
    """
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    class FishVolumes(Dataset):
        def __init__(self, root: Path):
            self.vols = sorted((root / "volumes").glob("*.npy"))
            self.masks = sorted((root / "masks").glob("*.npy"))
            assert self.vols and len(self.vols) == len(self.masks), "need paired volumes/masks"

        def __len__(self):
            return len(self.vols)

        def __getitem__(self, i):
            x = torch.from_numpy(np.load(self.vols[i]).astype("float32"))
            y = torch.from_numpy(np.load(self.masks[i]).astype("int64"))
            return x, y

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no GPU visible — dispatch to vancouver01 for real training "
              "(see jobs/run_hamilton_kegg_gpu.sh for the conda env pattern).")
    ds = FishVolumes(data_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    model = UNet3D(base=base).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = combined_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{epochs}  mean_loss={tot/len(dl):.4f}")
    ckpt = HERE / "fish_unet3d.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-test", action="store_true", help="CPU synthetic forward/backward")
    ap.add_argument("--data-dir", type=Path, help="dir with volumes/ and masks/ subdirs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--base", type=int, default=16, help="base channel width")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.smoke_test or args.data_dir is None:
        smoke_test(base=8)
    else:
        train(args.data_dir, args.epochs, args.base, args.lr)


if __name__ == "__main__":
    main()
