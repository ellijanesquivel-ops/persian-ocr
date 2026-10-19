import torch.nn as nn


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
