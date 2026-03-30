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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}

.block-container {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
    border-radius: 32px;
    padding: 2rem !important;
    margin-top: 1rem;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    max-width: 1200px;
}

div[data-testid="stVerticalBlock"] > div:empty {
    display: none;
}

.main-title {
    font-size: 56px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    margin-bottom: 40px;
}

.stFileUploader {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.12);
}

.result-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 32px;
    border-radius: 24px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    margin-top: 30px;
}

.waveform-container {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.12);
}

.footer {
    text-align: center;
    padding: 2rem;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align: center;">
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
MODEL_PATH = "model/instrunet_cnn.keras"
FILE_ID = "1qVlfOXIVthbxdYFQfrxsxCSo1sJTMrXb"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- AUDIO PROCESS ----------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4,4))
    librosa.display.specshow(mel_db)
    plt.axis("off")
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp.png").convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

# ---------------- INTENSITY TEXT (UPDATED) ----------------
def generate_intensity_text(scores):
    lines = ["🎵 Instrument Intensity:\n"]
    for inst, val in scores.items():
        bars = "█" * int(val * 20)
        lines.append(f"{inst:20} | {bars:<20} | {val:.2f}")
    return "\n".join(lines)

# ---------------- UI ----------------
uploaded_file = st.file_uploader("🎵 Choose your audio file (WAV or MP3)", type=["wav","mp3"])

if uploaded_file:
    with open("input.wav","wb") as f:
        f.write(uploaded_file.read())

    st.audio("input.wav")

    with st.spinner("Analyzing..."):
        X = audio_to_spectrogram("input.wav")
        pred = model.predict(X)[0]

    result = labels[np.argmax(pred)]
    name, icon = label_map[result]
    conf = float(np.max(pred))

    st.markdown(f"""
    <div class="result-card">
        {icon} {name}<br>
        <span style="font-size:18px;">Confidence: {conf:.2%}</span>
    </div>
    """, unsafe_allow_html=True)

    chart_data = {
        label_map[labels[i]][0]: float(pred[i]) for i in range(len(pred))
    }

    st.bar_chart(chart_data)

    # ✅ FIXED INTENSITY DISPLAY (VERTICAL)
    st.markdown("### 🎯 Instrument Intensity")
    intensity_text = generate_intensity_text(chart_data)
    st.text(intensity_text)

    st.success("✅ Analysis complete!")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
InstruNet AI - Made with 🎵
</div>
""", unsafe_allow_html=True)
