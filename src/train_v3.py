import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from model import Net
from preprocess import normalize

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

orig = np.load("data/persian_alphabet.npz")
split = np.load("data/split.npz")
win = np.load("data/windows_letters.npz")
BAD_CLASS = 10  # original class 10 images are copies of class 12


def orig_ids(name):
    ids = split[name]
    return ids[orig["labels"][ids] != BAD_CLASS]


wfonts = sorted(set(win["fonts"].tolist()))
assert len(wfonts) >= 5, "need at least 5 working Windows fonts"
rng = np.random.default_rng(42)
rng.shuffle(wfonts)
k = max(1, round(0.2 * len(wfonts)))
w_test, w_val, w_train = wfonts[:k], wfonts[k:2 * k], wfonts[2 * k:]
np.savez("data/split_windows.npz", train=w_train, val=w_val, test=w_test)
print("windows fonts train/val/test:", len(w_train), len(w_val), len(w_test))


def gather(ids, wf):
    wi = np.where(np.isin(win["fonts"], wf))[0]
    imgs = np.concatenate([orig["images"][ids], win["images"][wi]])
    labs = np.concatenate([orig["labels"][ids], win["labels"][wi]])
    x = np.stack([normalize(i) for i in imgs])
    return torch.tensor(x).unsqueeze(1), torch.tensor(labs, dtype=torch.long)


Xtr, ytr = gather(orig_ids("train"), w_train)
Xva, yva = gather(orig_ids("val"), w_val)
print("train/val images:", len(ytr), len(yva))

class Thickness:
    """Randomly thicken or thin the strokes, like a photo of a bold or faint letter."""
    def __call__(self, x):
        r = np.random.rand()
        if r < 0.25:
            return torch.nn.functional.max_pool2d(x.unsqueeze(0), 3, 1, 1).squeeze(0)
        if r < 0.5:
            return -torch.nn.functional.max_pool2d(-x.unsqueeze(0), 3, 1, 1).squeeze(0)
        return x


aug = v2.Compose([
    v2.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.8, 1.1), shear=8),
    v2.RandomPerspective(distortion_scale=0.25, p=0.5),
    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.3, 1.8))], p=0.5),
    Thickness(),
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


model = Net().to(device)
counts = torch.bincount(ytr, minlength=32).float()
weights = (counts.sum() / (32 * counts)).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
EPOCHS = 60
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
train_dl = DataLoader(DS(Xtr, ytr, aug), batch_size=64, shuffle=True)
val_dl = DataLoader(DS(Xva, yva), batch_size=128)


def predict_all(dl):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(out)


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
    sched.step()
    p = predict_all(val_dl)
    acc = (p == yva).float().mean().item()
    zal = (p[yva == 10] == 10).float().mean().item()
    zay = (p[yva == 12] == 12).float().mean().item()
    mark = ""
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "models/cnn_v4.pt")
        mark = "  <- saved"
    print(f"epoch {ep:2d}  loss {total / len(ytr):.3f}  val {acc * 100:.1f}%  "
          f"zal {zal * 100:.0f}%  zay {zay * 100:.0f}%{mark}")

print(f"best val acc: {best * 100:.1f}%  time: {time.time() - t0:.0f}s")
