import numpy as np
from PIL import Image
from labels import LETTERS

d = np.load("data/persian_alphabet.npz")
images, labels, fonts = d["images"], d["labels"], d["fonts"]

print("images:", images.shape, images.dtype, "min/max:", images.min(), images.max())
print("classes:", len(set(labels.tolist())), "| images per class:", np.bincount(labels).min(), "to", np.bincount(labels).max())
print("fonts:", len(set(fonts.tolist())))
print("mean pixel value:", round(float(images.mean()), 1))
print("labels file has", len(LETTERS), "letters")

tiles = [images[np.where(labels == c)[0][0]] for c in range(32)]
rows = [np.hstack(tiles[i * 8:(i + 1) * 8]) for i in range(4)]
Image.fromarray(np.vstack(rows)).resize((1024, 512)).save("data/montage.png")
print("saved data/montage.png")
