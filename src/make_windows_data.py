import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from labels import LETTERS

FONT_DIR = r"C:\Windows\Fonts"


def draw(font, ch):
    img = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(img).text((32, 32), ch, font=font, fill=255, anchor="mm")
    return np.array(img)


def is_working(font):
    missing = draw(font, "\u0378")
    for ch, _ in LETTERS:
        t = draw(font, ch)
        if t.max() == 0 or (t == missing).all():
            return False
    return True


CANDIDATES = ["arial.ttf", "arialbd.ttf", "tahoma.ttf", "tahomabd.ttf", "times.ttf",
              "timesbd.ttf", "segoeui.ttf", "segoeuib.ttf", "segoeuil.ttf", "calibri.ttf",
              "calibrib.ttf", "cour.ttf", "courbd.ttf", "majalla.ttf", "majallab.ttf",
              "trado.ttf", "tradbdo.ttf", "simpo.ttf", "simpbdo.ttf", "andlso.ttf",
              "uighur.ttf", "aldhabi.ttf", "urdtype.ttf"]


def main():
    images, labels, names, kept, skipped = [], [], [], [], []
    for name in CANDIDATES:
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            continue
        font = ImageFont.truetype(path, 36)
        zal, zay = draw(font, LETTERS[10][0]), draw(font, LETTERS[12][0])
        if not is_working(font) or (zal == zay).all():
            skipped.append(name)
            continue
        kept.append(name)
        for c, (ch, _) in enumerate(LETTERS):
            t = draw(font, ch)
            images.append(np.stack([t, t, t], axis=-1))
            labels.append(c)
            names.append("win_" + name)
    images = np.array(images, dtype=np.uint8)
    np.savez("data/windows_letters.npz", images=images,
             labels=np.array(labels, dtype=np.int32), fonts=np.array(names))
    print("kept:", kept)
    print("skipped:", skipped)
    print("images:", images.shape)
    zal = [images[i] for i in range(len(labels)) if labels[i] == 10]
    zay = [images[i] for i in range(len(labels)) if labels[i] == 12]
    Image.fromarray(np.vstack([np.hstack(zal), np.hstack(zay)])).save("data/windows_zal_zay.png")


if __name__ == "__main__":
    main()
