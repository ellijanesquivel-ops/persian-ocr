import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from preprocess import normalize

d = np.load("data/persian_alphabet.npz")
s = np.load("data/split.npz")
images, labels = d["images"], d["labels"]

def prep(ids):
    return np.array([normalize(images[i]).ravel() for i in ids])

X_train, y_train = prep(s["train"]), labels[s["train"]]
X_val, y_val = prep(s["val"]), labels[s["val"]]

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_val)

print("validation accuracy:", round(float((pred == y_val).mean()) * 100, 1), "%")
wrong = Counter((int(t), int(p)) for t, p in zip(y_val, pred) if t != p)
print("most common mistakes (true class, predicted class):", wrong.most_common(6))
