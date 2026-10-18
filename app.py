import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# These labels and the 224x224 image size come from the training notebook.
CLASS_NAMES = ["cordana", "sigatoka", "pestalotiopsis", "healthy"]
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Banana Leaf Disease Classifier",
    page_icon="🍃",
    layout="centered",
    title="HABIMANA Phocas Reg no:26RP00445",
)

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("Model.keras", compile=False)

st.title("🍃 Banana Leaf Disease Classifier")
st.write(
    "Upload a clear photo of a banana leaf. The trained model predicts one of "
    "four classes: Cordana, Sigatoka, Pestalotiopsis, or Healthy."
)
st.info(
    "This is an educational prediction tool, not a substitute for confirmation "
    "by an agricultural expert."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        if st.button("Predict leaf class", type="primary"):
            with st.spinner("Analyzing image..."):
                # Match notebook preprocessing: RGB, resize to 224x224, scale to 0–1.
                resized = image.resize(IMAGE_SIZE)
                array = np.asarray(resized, dtype=np.float32) / 255.0
                batch = np.expand_dims(array, axis=0)

                model = load_trained_model()
                probabilities = model.predict(batch, verbose=0)[0]
                best_index = int(np.argmax(probabilities))
                best_class = CLASS_NAMES[best_index]
                confidence = float(probabilities[best_index])

            st.subheader(f"Prediction: {best_class.replace('_', ' ').title()}")
            st.metric("Model confidence", f"{confidence:.1%}")
            st.write("Scores for each class")
            sorted_indices = np.argsort(probabilities)[::-1]
            for index in sorted_indices:
                st.write(f"**{CLASS_NAMES[int(index)].title()}** — {probabilities[index]:.1%}")
                st.progress(float(probabilities[index]))
    except Exception as exc:
        st.error("The image could not be processed or the model could not be loaded.")
        st.caption(f"Technical detail: {exc}")
else:
    st.caption("For best results, use a well-lit image with the leaf clearly visible.")
