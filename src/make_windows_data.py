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
