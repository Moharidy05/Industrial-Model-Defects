import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Industrial QC Inspector",
    page_icon="⚙️",
    layout="centered"
)

# 2. Load the Model (Cached so it doesn't reload on every interaction)
@st.cache_resource
def load_qc_model():
    return tf.keras.models.load_model('industrial_defect_model.h5', compile=False)
    
model = load_qc_model()

# 3. Image Preprocessing Function
def preprocess_image(image):
    # Resize to match MobileNetV2 input
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Ensure the image has 3 color channels (RGB)
    if len(img_array.shape) == 2:  # If grayscale, convert to RGB
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 4. UI Layout
st.title("⚙️ Automated Quality Control Inspector")
st.markdown("""
Upload an image of a casted mechanical part (e.g., pump impeller). 
The Deep Learning engine will analyze the surface for cracks, tears, or deformations.
""")

st.divider()

# 5. File Uploader
uploaded_file = st.file_uploader("Upload Casting Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    with col2:
        st.markdown("### Analysis Results")
        with st.spinner("Analyzing surface integrity..."):
            
            # Preprocess and Predict
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0][0]
            
            # Classes: 0 = Defective, 1 = OK
            # Calculate confidence based on distance from the 0.5 threshold
            if prediction >= 0.5:
                confidence = prediction * 100
                st.success("✅ **STATUS: PASS / OK**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                st.info("No critical surface defects detected.")
            else:
                confidence = (1 - prediction) * 100
                st.error("🚨 **STATUS: DEFECT DETECTED**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                st.warning("Warning: Structural anomalies found. Route part for manual inspection.")

st.divider()
st.caption("Powered by Custom Deep Learning Architecture.")
