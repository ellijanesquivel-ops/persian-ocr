import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

from labels import LETTERS          # noqa: E402
from model import Net               # noqa: E402
from preprocess import normalize    # noqa: E402

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000             # pixels
AMBIGUOUS = {10, 12}              # classes with identical training images


@st.cache_resource
def load_model():
    net = Net()
    net.load_state_dict(torch.load(BASE / "models" / "cnn.pt", map_location="cpu"))
    net.eval()
    return net


def read_image(uploaded):
    if uploaded.size > MAX_FILE_SIZE:
        raise ValueError("The file is larger than 10MB.")
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            img = img.convert("RGBA")
            white = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, img)
        img = img.convert("RGB")
    except Exception:
        raise ValueError("This file is not a readable image (it may be empty or corrupted).")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


def predict(model, img):
    x = normalize(np.array(img))  # same preprocessing as training
    t = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    top = torch.topk(probs, 3)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)], x


st.set_page_config(layout="centered", page_title="Persian Letter Recognition")
st.header("Persian Letter Recognition")
st.write("Here is where you test out our app and see if it can recognize the Persian letters you write. "
         "You can insert an image of a persian letter in the box below and click the 'Predict' button "
         "to see the model's prediction.")

alphabet_img = BASE / "persian-letters.png"
if alphabet_img.exists():
    st.image(str(alphabet_img), caption="Persian Alphabets")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file models/cnn.pt not found. Train the model first: python src\\train.py")
    st.stop()

with st.container(border=True):
    st.title("Output:")
    uploaded = st.file_uploader("Upload an image of a Persian letter", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = None
        try:
            img = read_image(uploaded)
        except ValueError as e:
            st.error(str(e))

        if img is not None:
            st.write("Image inserted:")
            st.image(img, width=250)

            if st.button("Predict"):
                try:
                    top, processed = predict(model, img)
                except ValueError:
                    st.error("No letter was found in this image. Try a clearer picture.")
                else:
                    best, conf = top[0]
                    letter, label = LETTERS[best]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"<div style='font-size:96px;text-align:center'>{letter}</div>",
                                    unsafe_allow_html=True)
                    with c2:
                        st.write(f"**Slide label:** {label}")
                        st.write(f"**Unicode:** U+{ord(letter):04X}")
                        st.write(f"**Confidence:** {conf * 100:.1f}%")
                        st.caption("What the model saw:")
                        st.image(processed, clamp=True, width=96)
                    st.write("Top 3: " + ", ".join(f"{LETTERS[i][0]} {p * 100:.0f}%" for i, p in top))
                    st.caption("Confidence is the model's own score. It is not proof that the answer is correct.")
                    if best in AMBIGUOUS:
                        st.warning("Known limitation: the training data has identical images for "
                                   "dhal (U+0630) and zay (U+0632), so this model cannot tell these two letters apart.")

# Information about limitations
with st.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    """)