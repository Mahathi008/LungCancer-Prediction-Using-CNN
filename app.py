import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from datetime import datetime


st.set_page_config(
    page_title="Lung Cancer Classifier",
    page_icon="🩺",
    layout="wide"
)


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
INPUT_SIZE = (256, 256)   
 
def load_safe_model():
    
    model_paths = [
        os.path.join(MODEL_DIR, "lung_cancer_densenet_final.keras")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                st.sidebar.success(f"✅ Loaded model from: {os.path.basename(path)}")
                
                 
                test_input = np.random.rand(1, *INPUT_SIZE, 3).astype(np.float32)
                _ = model.predict(test_input)
                
                return model
            except Exception as e:
                st.sidebar.warning(f"⚠️ Failed to load {os.path.basename(path)}: {str(e)}")
                continue
    
    
    st.sidebar.warning("⚠️ No valid model found - creating emergency model")
    from tensorflow.keras.applications import DenseNet121
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(*INPUT_SIZE, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


model = load_safe_model()


def preprocess_image(image):
    try:
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        
        image = image.resize(INPUT_SIZE)
        array = img_to_array(image) / 255.0
        return np.expand_dims(array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None


st.title("🫁 Lung Cancer Classification")
uploaded_file = st.file_uploader("Upload a lung CT scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        processed_img = preprocess_image(img)
        if processed_img is not None:
            pred = model.predict(processed_img)
            confidence = float(pred[0][0])
            diagnosis = "Non-Cancerous" if confidence > 0.5 else "Cancerous"
            confidence_pct = confidence*100 if diagnosis == "Non-Cancerous" else (1-confidence)*100
            
            st.success(f"**Prediction:** {diagnosis} ({confidence_pct:.2f}% confidence)")
            
            
            st.progress(int(confidence_pct))
            st.caption(f"Confidence: {confidence_pct:.2f}%")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")


with st.sidebar:
    st.subheader("Model Information")
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Input Shape: {model.input_shape}")
    st.write(f"Output Shape: {model.output_shape}")
    
    if st.checkbox("Show advanced info"):
        st.write("Available models:")
        for f in os.listdir(MODEL_DIR):
            st.write(f"- {f} ({os.path.getsize(os.path.join(MODEL_DIR, f))} bytes)")
    
    st.info("""
    **Instructions:**
    1. Upload a lung CT scan image (JPEG/PNG)
    2. The AI will analyze the image
    3. Results show confidence percentage
    """)


st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")