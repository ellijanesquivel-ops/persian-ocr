import numpy as np
import cv2
from preprocess import normalize

d = np.load("data/persian_alphabet.npz")
img = d["images"][0]
a = normalize(img)
b = normalize(255 - img)
print("shape:", a.shape, "range:", a.min(), a.max())
print("max difference, normal vs inverted input:", float(np.abs(a - b).max()))
gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 64))
row = np.hstack([gray, (a * 255).astype("uint8"), (b * 255).astype("uint8")])
cv2.imwrite("data/preprocess_check.png", cv2.resize(row, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
