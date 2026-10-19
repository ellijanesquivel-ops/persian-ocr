from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from model import Net
from preprocess import normalize

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "cnn_v3.pt"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000             # pixels


def load_model(path=MODEL_PATH):
    net = Net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def open_image(file, nbytes=None):
    """file: a path or file-like object. Returns an RGB PIL image, or raises ValueError."""
    if nbytes is None and isinstance(file, (str, Path)):
        nbytes = Path(file).stat().st_size
    if nbytes == 0:
        raise ValueError("The file is empty.")
    if nbytes is not None and nbytes > MAX_FILE_SIZE:
        raise ValueError("The file is larger than 10MB.")
    try:
        img = Image.open(file)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba).convert("RGB")
    except Exception:
        raise ValueError("This file is not a readable image (it may be corrupted).")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


def predict(model, img):
    """Returns ([(class, probability), ...top 3], the 64x64 image the model saw)."""
    x = normalize(np.array(img))  # raises ValueError if no letter is found
    t = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    top = torch.topk(probs, 3)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)], x
