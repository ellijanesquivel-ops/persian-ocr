import numpy as np
import torch
from collections import Counter

from model import Net
from preprocess import normalize

orig = np.load("data/persian_alphabet.npz")
split = np.load("data/split.npz")
win = np.load("data/windows_letters.npz")
ws = np.load("data/split_windows.npz")

ids = split["test"]
ids = ids[orig["labels"][ids] != 10]
wi = np.where(np.isin(win["fonts"], ws["test"]))[0]
imgs = np.concatenate([orig["images"][ids], win["images"][wi]])
y = np.concatenate([orig["labels"][ids], win["labels"][wi]])
X = torch.tensor(np.stack([normalize(i) for i in imgs])).unsqueeze(1)

net = Net()
net.load_state_dict(torch.load("models/cnn_v4.pt", map_location="cpu"))
net.eval()
with torch.no_grad():
    pred = net(X).argmax(1).numpy()

print("test images:", len(y))
print("TEST accuracy:", round(float((pred == y).mean()) * 100, 1), "%")
print("class 10 (zal) accuracy:", round(float((pred[y == 10] == 10).mean()) * 100, 1), "%")
print("class 12 (zay) accuracy:", round(float((pred[y == 12] == 12).mean()) * 100, 1), "%")
wrong = Counter((int(t), int(p)) for t, p in zip(y, pred) if t != p)
print("top mistakes (true, predicted): count ->", wrong.most_common(8))
