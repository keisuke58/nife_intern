---
description: Run the full 3-D FISH extraction over all .lif FOVs and show the findings
---
Run the batch 3-D FISH structure extraction and report the lateral findings.

```bash
python3 scripts/analysis/fish_3d_batch.py --decode-ds 2
```

This decodes every titanium FOV per-voxel (no xy-averaging), measuring per (cond,day): lateral heterogeneity
(xy-projection CV), Fn–Pg lateral-patch + 3-D Manders M1, biomass; and saves one `phi(x,y,z,5)` IC per (cond,day)
to `results/fish_3d/ic/`.

After it finishes, view + summarise the two figures:
- `results/fish_3d/fish_3d_fnpg_coloc.png` — the Fn–Pg lateral-vs-3D coloc split (the depth-separation finding)
- `results/fish_3d/fish_3d_lateral_heterogeneity.png` — DH homogenises vs CH patchy

If `$ARGUMENTS` is a single FOV spec (e.g. `--file "HOBIC FISH/241018_*.lif" --series 0`), instead run
`scripts/analysis/fish_3d_profile.py $ARGUMENTS` for a single-FOV orthoview + per-axis profiles.
Heavy job (~84 FOVs, tens of minutes) — run it in the background and poll.