import numpy as np

d = np.load("data/persian_alphabet.npz")
fonts = d["fonts"]

names = sorted(set(fonts.tolist()))
rng = np.random.default_rng(42)
rng.shuffle(names)

train_f, val_f, test_f = names[:40], names[40:49], names[49:]

def idx(fs):
    return np.where(np.isin(fonts, fs))[0]

train_idx, val_idx, test_idx = idx(train_f), idx(val_f), idx(test_f)
np.savez("data/split.npz", train=train_idx, val=val_idx, test=test_idx)

print("train/val/test images:", len(train_idx), len(val_idx), len(test_idx))
print("fonts shared between sets:",
      len(set(train_f) & set(val_f)) + len(set(train_f) & set(test_f)) + len(set(val_f) & set(test_f)))
