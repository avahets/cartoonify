import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Cartoonify an Image", layout="centered")
st.title("🎨 Cartoonify Your Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def cartoonify(image):
    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    resized1 = cv2.resize(original, (960, 540))

    gray = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
    resized2 = cv2.resize(gray, (960, 540))

    blurred = cv2.medianBlur(resized2, 5)
    resized3 = cv2.resize(blurred, (960, 540))

    edges = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    resized4 = cv2.resize(edges, (960, 540))

    color = cv2.bilateralFilter(resized1, 9, 300, 300)
    resized5 = cv2.resize(color, (960, 540))

    cartoon = cv2.bitwise_and(color, color, mask=resized4)
    resized6 = cv2.resize(cartoon, (960, 540))

    return [resized1, resized2, resized3, resized4, resized5, resized6]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    steps = cartoonify(image)

    titles = ['Original', 'Grayscale', 'Blurred', 'Edges', 'Smoothed', 'Cartoonified']
    st.subheader("🧪 Cartoonification Stages")
    for title, img in zip(titles, steps):
        st.markdown(f"**{title}**")
        if len(img.shape) == 2:
            st.image(img, channels="GRAY")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save the final cartoonified image
    final = cv2.cvtColor(steps[-1], cv2.COLOR_BGR2RGB)
    result = Image.fromarray(final)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("📥 Download Cartoon Image", data=byte_im, file_name="cartoonified_image.png", mime="image/png")
else:
    st.info("Upload an image to begin cartoonifying!")
