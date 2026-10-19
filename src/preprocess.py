import cv2
import numpy as np

SIZE = 64


def normalize(img):
    """img: uint8 array (H, W) or (H, W, 3) RGB, any polarity.
    Returns a SIZE x SIZE float32 array in 0..1, white letter on black."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask.mean() > 127:  # mostly white: background is light, so invert
        mask = 255 - mask

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("No letter found in the image")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = int(max(h, w) * 1.3)
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    out = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32) / 255.0
