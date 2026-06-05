#!/usr/bin/env python3
"""
Recomposite fig_fish_clsm_feature.png so that the species legend is placed
in the gap between the left (xy MIP) and right (xz MIP) panels, instead of
being embedded in the bottom-left of the left panel.

Run:
    python scripts/figures/composite_fish_clsm_feature.py

Output: 30_Masterarbeit/figures/fig_fish_clsm_feature.png  (overwrites in place)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ORIG = Path("/home/nishioka/IKM_Hiwi/nife/figures/lif_slides/220518_HOBIC22_5Spezies_FISH_Tag1__slide_feature.png")
THESIS_FIG = Path("/home/nishioka/LUHsummer26/30_Masterarbeit/figures")
OUT = THESIS_FIG / "fig_fish_clsm_feature.png"

# ── Load ORIGINAL (not previously modified) ───────────────────────────────────
img = Image.open(ORIG).convert("RGBA")
arr = np.array(img)
H, W = arr.shape[:2]   # 881, 1856

# ── Panel boundaries (detected: gap is cols 928-1047, all-black) ──────────────
LEFT_END    = 928
RIGHT_START = 1048
GAP_W_ORIG  = RIGHT_START - LEFT_END   # 120 px

left_arr  = arr[:, :LEFT_END].copy()
right_arr = arr[:, RIGHT_START:].copy()

# ── Erase embedded legend from bottom-left of left panel ─────────────────────
# The legend overlays the microscopy at rows 740-881, cols 0-215
left_arr[738:, :220, :3] = 0   # fill black
left_arr[738:, :220, 3]  = 255

# ── Create new composite with wider legend gap (200 px) ───────────────────────
LEG_W  = 200
NEW_W  = LEFT_END + LEG_W + (W - RIGHT_START)   # 928+200+808 = 1936
new_arr = np.zeros((H, NEW_W, 4), dtype=np.uint8)
new_arr[:, :LEFT_END]                   = left_arr
new_arr[:, LEFT_END + LEG_W:]           = right_arr
# gap stays black (zeros)

new_img = Image.fromarray(new_arr, "RGBA")

# ── Draw legend into the gap using PIL ────────────────────────────────────────
draw = ImageDraw.Draw(new_img)

SPECIES = [
    ("S. oralis",     (60,  120, 200, 255)),   # blue
    ("A. naeslundii", (50,  170,  70, 255)),   # green
    ("Vd/Vp",         (210, 190,  40, 255)),   # yellow
    ("F. nucleatum",  (140,  60, 180, 255)),   # purple
    ("P. gingivalis", (200,  40,  40, 255)),   # red
]

# Try to load a font; fall back to default if unavailable
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size=20)
except:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)
    except:
        font = ImageFont.load_default()

# Centre the legend block vertically
N = len(SPECIES)
row_h  = 34    # pixels per entry
BOX_W  = 18
BOX_H  = 18
block_h = N * row_h
y0 = (H - block_h) // 2 + 20   # vertical centre
x0 = LEFT_END + (LEG_W - BOX_W - 6 - 100) // 2  # horizontal centre in gap

# Header text
draw.text((LEFT_END + 5, y0 - 32), "Species", fill=(220,220,220,255), font=font)

for k, (name, color) in enumerate(SPECIES):
    y = y0 + k * row_h
    bx = LEFT_END + 14
    # Coloured square
    draw.rectangle([bx, y, bx + BOX_W, y + BOX_H], fill=color)
    # White text
    draw.text((bx + BOX_W + 8, y), name,
              fill=(230, 230, 230, 255), font=font)

# ── Save ─────────────────────────────────────────────────────────────────────
new_img.save(str(OUT), dpi=(200, 200))
print(f"Saved: {OUT}  ({NEW_W}×{H})")
