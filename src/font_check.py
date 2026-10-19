import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from labels import LETTERS

FONT_DIR = r"C:\Windows\Fonts"
CANDIDATES = ["arial.ttf", "tahoma.ttf", "times.ttf", "segoeui.ttf",
              "calibri.ttf", "verdana.ttf", "georgia.ttf", "cour.ttf", "trebuc.ttf"]

rows, used = [], []
for name in CANDIDATES:
    path = os.path.join(FONT_DIR, name)
    if not os.path.exists(path):
        continue
    font = ImageFont.truetype(path, 44)
    tiles = []
    for ch, _ in LETTERS:
        img = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(img).text((32, 32), ch, font=font, fill=255, anchor="mm")
        tiles.append(np.array(img))
    rows.append(np.hstack(tiles[:16]))
    rows.append(np.hstack(tiles[16:]))
    used.append(name)

Image.fromarray(np.vstack(rows)).save("data/font_check.png")
print("fonts found, in picture order (2 rows each):", used)
