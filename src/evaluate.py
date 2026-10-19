import numpy as np
import torch
from collections import Counter
from model import Net
from preprocess import normalize

d = np.load("data/persian_alphabet.npz")
s = np.load("data/split.npz")
ids = s["val"]
X = torch.tensor(np.stack([normalize(d["images"][i]) for i in ids])).unsqueeze(1)
y = d["labels"][ids]

net = Net()
net.load_state_dict(torch.load("models/cnn.pt"))
net.eval()
with torch.no_grad():
    pred = net(X).argmax(1).numpy()

print("validation accuracy:", round(float((pred == y).mean()) * 100, 1), "%")
wrong = Counter((int(t), int(p)) for t, p in zip(y, pred) if t != p)
print("mistakes (true class, predicted class): count")
for pair, n in wrong.most_common(10):
    print("  ", pair, n)
per = [(float((pred[y == c] == c).mean()), c) for c in range(32)]
print("weakest classes (accuracy, class):", sorted(per)[:5])
