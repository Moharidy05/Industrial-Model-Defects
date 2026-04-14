import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Industrial QC Inspector",
    page_icon="⚙️",
    layout="centered"
)

@st.cache_resource
def load_qc_model():
    return tf.keras.models.load_model('industrial_defect_model.keras', compile=False)

model = load_qc_model()

def preprocess_image(image):
    image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("⚙️ Automated Quality Control Inspector")
st.markdown("""
Upload an image of a casted mechanical part. 
The Deep Learning engine will analyze the surface for cracks, tears, or deformations.
""")

st.divider()

uploaded_file = st.file_uploader("Upload Casting Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    with col2:
        st.markdown("### Analysis Results")
        with st.spinner("Analyzing surface integrity..."):
            
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0][0]
            
            if prediction >= 0.5:
                confidence = prediction * 100
                st.success("✅ **STATUS: PASS / OK**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.info("No critical surface defects detected.")
            else:
                confidence = (1 - prediction) * 100
                st.error("🚨 **STATUS: DEFECT DETECTED**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.warning("Structural anomalies found. Route part for manual inspection.")

st.divider()
st.caption("Powered by Custom Deep Learning Architecture.")
