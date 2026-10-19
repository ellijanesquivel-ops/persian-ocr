import os
import sys

from PIL import Image

from labels import LETTERS
from recognizer import load_model, open_image, predict

folder = sys.argv[1] if len(sys.argv) > 1 else "photos"
model = load_model()
os.makedirs("photos_out", exist_ok=True)
ok = tot = 0
for f in sorted(os.listdir(folder)):
    if not f.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    try:
        img = open_image(os.path.join(folder, f))
        top, processed = predict(model, img)
    except ValueError as e:
        print(f"{f}: ERROR - {e}")
        continue
    Image.fromarray((processed * 255).astype("uint8")).save(
        os.path.join("photos_out", "seen_" + os.path.splitext(f)[0] + ".png"))
    pred, conf = top[0]
    note = ""
    prefix = f.split("_")[0]
    if prefix.isdigit():
        tot += 1
        hit = int(prefix) == pred
        ok += hit
        note = "OK" if hit else f"WRONG (expected class {prefix})"
    print(f"{f}: class {pred} U+{ord(LETTERS[pred][0]):04X} conf {conf * 100:.0f}%  {note}")
if tot:
    print(f"\naccuracy on photos: {ok} / {tot}")
