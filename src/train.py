import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from preprocess import normalize

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

d = np.load("data/persian_alphabet.npz")
s = np.load("data/split.npz")
images, labels = d["images"], d["labels"]


def prep(ids):
    x = np.stack([normalize(images[i]) for i in ids])
    return torch.tensor(x).unsqueeze(1), torch.tensor(labels[ids], dtype=torch.long)


Xtr, ytr = prep(s["train"])
Xva, yva = prep(s["val"])

aug = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.1)),
    v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
])


class DS(Dataset):
    def __init__(self, X, y, tf=None):
        self.X, self.y, self.tf = X, y, tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.tf:
            x = self.tf(x)
        return x, self.y[i]


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(), nn.MaxPool2d(2))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
                                  nn.Linear(128, 32))

    def forward(self, x):
        return self.head(self.features(x))


model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
train_dl = DataLoader(DS(Xtr, ytr, aug), batch_size=64, shuffle=True)
val_dl = DataLoader(DS(Xva, yva), batch_size=128)


def accuracy(dl):
    model.eval()
    ok = 0
    with torch.no_grad():
        for x, y in dl:
            ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return ok / len(dl.dataset)


EPOCHS = 60
best, t0 = 0.0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    acc = accuracy(val_dl)
    sched.step()

    mark = ""
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "models/cnn_v2.pt")
        mark = "  <- saved"
    print(f"epoch {ep:2d}  train loss {total / len(train_dl.dataset):.3f}  val acc {acc * 100:.1f}%{mark}")

print(f"best val acc: {best * 100:.1f}%  time: {time.time() - t0:.0f}s")
