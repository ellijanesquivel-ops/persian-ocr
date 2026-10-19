import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from recognizer import load_model, open_image, predict

fails = 0


def check(name, ok):
    global fails
    fails += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + name)


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def top_class(data, nbytes=None):
    img = open_image(io.BytesIO(data), len(data) if nbytes is None else nbytes)
    return predict(model, img)[0][0][0]


def fails_cleanly(data, nbytes=None):
    try:
        top_class(data, nbytes)
    except ValueError:
        return True
    except Exception:
        return False
    return False


d = np.load("data/persian_alphabet.npz")
s = np.load("data/split.npz")
i = [j for j in s["val"] if d["labels"][j] == 1][0]
base = 255 - d["images"][i]  # dark letter on light "paper"

print("--- software defects ---")
model = load_model()
check("model loads", True)
try:
    load_model(Path("nope.pt"))
    check("missing model file gives an error", False)
except FileNotFoundError:
    check("missing model file gives an error", True)

check("clean image recognized", top_class(png(base)) == 1)
for size in (32, 128, 1000, 3000):
    big = np.array(Image.fromarray(base).resize((size, size)))
    check(f"image size {size}px", top_class(png(big)) == 1)
check("20 repeated predictions identical", len({top_class(png(base)) for _ in range(20)}) == 1)
check("empty file rejected", fails_cleanly(b""))
check("random bytes rejected", fails_cleanly(b"this is not an image"))
check("truncated PNG rejected", fails_cleanly(png(base)[:50]))
check("blank white image rejected", fails_cleanly(png(np.full((64, 64, 3), 255, np.uint8))))
check("blank black image rejected", fails_cleanly(png(np.zeros((64, 64, 3), np.uint8))))
check("file over 10MB rejected", fails_cleanly(png(base), 11 * 1024 * 1024))

print("--- recognition robustness (not defects, just information) ---")
rng = np.random.default_rng(0)
variants = {
    "rotated 8 degrees": np.array(Image.fromarray(base).rotate(8, fillcolor=(255, 255, 255))),
    "low contrast": (base * 0.5 + 100).astype(np.uint8),
    "noisy": np.clip(base + rng.normal(0, 15, base.shape), 0, 255).astype(np.uint8),
}
for name, arr in variants.items():
    print(("OK    " if top_class(png(arr)) == 1 else "MISS  ") + name)

print(f"\n{fails} defect(s)")
sys.exit(1 if fails else 0)
