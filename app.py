import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import gdown
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Preformatted,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InstruNet AI", 
    layout="centered",
    page_icon="🎵",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM UI ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    .main-container {
        background: #ffffff;
        border-radius: 32px;
        padding: 2rem;
        margin: 1rem auto;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        max-width: 1200px;
    }

    .main-title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #4a5568;
        margin-bottom: 40px;
        font-size: 18px;
    }

    .upload-section {
        background: #f7fafc;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }

    /* ✅ FIX: Remove unwanted white boxes */
    div[data-testid="stVerticalBlock"] > div:empty {
        display: none;
    }

    section.main > div:first-child {
        padding-top: 0rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div class="main-title">🎵 InstruNet AI</div>
    <div class="subtitle">Automatic Musical Instrument Detection from Audio</div>
</div>
""", unsafe_allow_html=True)

# ---------------- LABELS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻"),
}
labels = list(label_map.keys())

# ---------------- MODEL ----------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "instrunet_cnn.keras")
FILE_ID = "1qVlfOXIVthbxdYFQfrxsxCSo1sJTMrXb"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=True, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- AUDIO PROCESS ----------------
def audio_to_spectrogram(audio_path, img_size=224):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db)
    plt.axis("off")
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp.png").convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "🎵 Upload Audio File",
    type=["wav", "mp3"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with open("input.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("input.wav")

    with st.spinner("Analyzing..."):
        X = audio_to_spectrogram("input.wav")
        pred = model.predict(X)[0]

    result = labels[np.argmax(pred)]
    name, icon = label_map[result]
    conf = float(np.max(pred))

    st.markdown(f"""
    <div style="background:#1e3c72;color:white;padding:20px;border-radius:15px;text-align:center;">
        {icon} {name} <br>
        Confidence: {conf:.2%}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
