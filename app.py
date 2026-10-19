import sys
from pathlib import Path

import streamlit as st

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

from labels import LETTERS                                # noqa: E402
from recognizer import load_model, open_image, predict    # noqa: E402


@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(layout="centered", page_title="Persian Letter Recognition")
st.header("Persian Letter Recognition")
st.write("Here is where you test out our app and see if it can recognize the Persian letters you write. "
         "You can insert an image of a persian letter in the box below and click the 'Predict' button "
         "to see the model's prediction.")

alphabet_img = BASE / "persian-letters.png"
if alphabet_img.exists():
    st.image(str(alphabet_img), caption="Persian Alphabets")

try:
    model = get_model()
except FileNotFoundError:
    st.error("Model file models/cnn_v3.pt not found. Train it first: python src\\train_v3.py")
    st.stop()

with st.container(border=True):
    st.title("Output:")
    uploaded = st.file_uploader("Upload an image of a Persian letter", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = None
        try:
            img = open_image(uploaded, uploaded.size)
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

with st.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - One Persian letter per image, dark letter on a light background works best
    """)
