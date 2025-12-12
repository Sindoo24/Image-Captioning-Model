import streamlit as st
import time
from PIL import Image
import sys
import os
import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_loader import ModelLoader
from model.caption_generator import CaptionGenerator

st.set_page_config(page_title="AI Image Captioner", layout="wide", page_icon="🖼️")
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🖼️ AI Image Captioning Model")
st.markdown("Run open-source Vision-Language models locally.")

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ["Salesforce/blip-image-captioning-large", "nlpconnect/vit-gpt2-image-captioning"]
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Image Upload")

uploaded_file = st.sidebar.file_uploader("Drag and drop an image here", type=["jpg", "png", "jpeg"])

@st.cache_resource
def load_resources(model_name):
    loader = ModelLoader()
    return loader.load_model(model_name)

try:
    with st.spinner(f"Loading {model_choice}... (First run takes time)"):
        model, processor = load_resources(model_choice)
        generator = CaptionGenerator(model, processor)
    st.sidebar.success("Model Active ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

tab1, tab2 = st.tabs(["✨ Generator", "📜 Activity Logs"])

with tab1:
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Generate Caption")
            if st.button("✨ Generate ", type="primary"):
                start_time = time.time()
                with st.spinner("Analyzing image features..."):
                    try:
                    
                        caption = generator.generate_caption(image)
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        st.success(f"**Caption:** {caption}")
                        st.info(f"⏱️ Time: {duration:.2f}s | Model: {model_choice}")

                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        log_entry = {
                            "Time": timestamp,
                            "Image Name": uploaded_file.name,
                            "Model": model_choice.split("/")[1], 
                            "Caption": caption,
                            "Duration": f"{duration:.2f}s",
                            "Image Object": image 
                        }
                        st.session_state.history.append(log_entry)
                        
                        
                        st.download_button(
                            label="⬇️ Download Caption",
                            data=f"Image: {uploaded_file.name}\nCaption: {caption}",
                            file_name=f"caption_{int(time.time())}.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("👈 Please upload an image in the sidebar to get started.")


with tab2:
    st.header("Session Activity Log")
    
    if len(st.session_state.history) > 0:
        display_data = [{k: v for k, v in entry.items() if k != "Image Object"} for entry in st.session_state.history]
        df = pd.DataFrame(display_data)
        
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Detailed History")
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{len(st.session_state.history)-i}: {entry['Image Name']} ({entry['Time']})"):
                col_hist1, col_hist2 = st.columns([1, 3])
                with col_hist1:
                    st.image(entry['Image Object'], width=150)
                with col_hist2:
                    st.write(f"**Model:** {entry['Model']}")
                    st.write(f"**Caption:** {entry['Caption']}")
                    st.write(f"**Speed:** {entry['Duration']}")
        
        if st.button("Clear Logs"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No captions generated yet in this session.")

st.markdown("---")
st.caption("Image Captioning Model ")