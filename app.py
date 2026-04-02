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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #121a33 45%, #0f172a 100%);
        color: #f8fafc;
    }

    .main > div {
        padding-top: 1.2rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 28px 30px;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #ffffff;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .info-chip-wrap {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .info-chip {
        background: rgba(59, 130, 246, 0.14);
        color: #dbeafe;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.9rem;
        border: 1px solid rgba(96, 165, 250, 0.2);
    }

    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(59,130,246,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .upload-box {
        background: rgba(255,255,255,0.04);
        border: 1.5px dashed rgba(148,163,184,0.4);
        border-radius: 20px;
        padding: 18px;
    }

    .result-banner {
        background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(99,102,241,0.20));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .result-main {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .result-sub {
        color: #dbeafe;
        font-size: 1rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div[data-testid="stFileUploader"] {
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }

    .stButton>button, .stDownloadButton>button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        font-weight: 700;
        padding: 0.72rem 1rem;
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
    }

    .stDownloadButton>button:hover, .stButton>button:hover {
        border-color: rgba(255,255,255,0.20);
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
        color: white;
    }

    div[data-testid="stAudio"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 10px 12px;
    }

    .stCodeBlock {
        border-radius: 16px;
    }

    h1, h2, h3, h4, h5, h6, p, label, div {
        color: inherit;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 10px;
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🎵 InstruNet AI</div>
    <p class="hero-subtitle">
        Automatic Musical Instrument Detection using deep learning-based audio analysis.
        Upload an audio file, visualize the waveform, inspect confidence scores, and download the generated report.
    </p>
    <div class="info-chip-wrap">
        <div class="info-chip">CNN-Based Prediction</div>
        <div class="info-chip">Waveform Visualization</div>
        <div class="info-chip">Confidence Analysis</div>
        <div class="info-chip">PDF + JSON Export</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ⚙️ Hyperparameter Tuning")
st.sidebar.markdown("Fine-tune preprocessing parameters for spectrogram generation.")

n_fft = st.sidebar.slider("FFT Size", 512, 4096, 2048, step=512)
hop_length = st.sidebar.slider("Hop Length", 128, 1024, 256, step=128)
n_mels = st.sidebar.slider("Mel Bands", 64, 256, 128, step=32)

colormap = st.sidebar.selectbox(
    "Spectrogram Color",
    ["magma", "viridis", "plasma"]
)

threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎼 Supported Instruments")
st.sidebar.markdown("""
- 🎹 Piano  
- 🎸 Acoustic Guitar  
- ⚡🎸 Electric Guitar  
- 🎻 Violin
""")

# ---------------- LABELS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻")
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

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db, cmap=colormap)
    plt.axis("off")
    plt.savefig("temp_spec.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp_spec.png").convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)

# ---------------- WAVEFORM ----------------
def create_waveform_image(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, color="cyan")
    plt.title("Audio Waveform")
    plt.tight_layout()
    plt.savefig("waveform.png", bbox_inches="tight")
    plt.close()

    return "waveform.png"

# ---------------- CONFIDENCE GRAPH ----------------
def create_confidence_graph(scores):
    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 3))
    plt.bar(names, values)
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("confidence.png", bbox_inches="tight")
    plt.close()

    return "confidence.png"

# ---------------- INTENSITY ----------------
def generate_intensity_text(scores):
    text = "Instrument Intensity:\n"
    for inst, val in scores.items():
        bars = "|" * int(val * 20)
        text += f"{inst}: {bars}\n"
    return text

# ---------------- PDF ----------------
def generate_pdf(result, waveform_path, confidence_path, intensity_text):
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("InstruNet AI Report", styles["Title"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(f"Audio File: {result['audio_file']}", styles["Normal"]))
    elements.append(Paragraph(f"Detected Instrument: {result['detected_instrument']}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {result['confidence']:.2f}", styles["Normal"]))

    elements.append(Spacer(1, 10))
    elements.append(Preformatted(intensity_text, styles["Code"]))

    if os.path.exists(waveform_path):
        elements.append(RLImage(waveform_path, width=400, height=150))

    if os.path.exists(confidence_path):
        elements.append(RLImage(confidence_path, width=400, height=200))

    doc.build(elements)
    return pdf_path

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Audio File</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Audio (.wav or .mp3)", type=["wav", "mp3"])

st.markdown(
    '<div class="small-note">Supported formats: WAV, MP3</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Uploaded Audio</div>', unsafe_allow_html=True)
        st.audio("input_audio.wav")
        st.markdown(f"<div class='small-note'>File name: {uploaded_file.name}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
        st.success("Model loaded successfully")
        st.markdown(f"**FFT Size:** {n_fft}")
        st.markdown(f"**Hop Length:** {hop_length}")
        st.markdown(f"**Mel Bands:** {n_mels}")
        st.markdown(f"**Threshold:** {threshold:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing audio and generating prediction..."):
        X_test = audio_to_spectrogram("input_audio.wav")
        pred = model.predict(X_test)[0]

    idx = np.argmax(pred)
    detected_name, icon = label_map[labels[idx]]
    confidence = float(np.max(pred))

    if confidence < threshold:
        detected_name = "Uncertain"
        icon = "⚠️"

    waveform_path = create_waveform_image("input_audio.wav")

    chart_data = {
        label_map[labels[i]][0]: float(pred[i])
        for i in range(len(pred))
    }

    confidence_path = create_confidence_graph(chart_data)
    intensity_text = generate_intensity_text(chart_data)

    # ---------------- RESULT SUMMARY ----------------
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Prediction Summary</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{icon}</div>
            <div class="metric-label">Detected Class</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{detected_name}</div>
            <div class="metric-label">Instrument</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{confidence:.2f}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-banner">
            <div class="result-main">{icon} {detected_name}</div>
            <div class="result-sub">Prediction confidence: {confidence:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- VISUALIZATION ----------------
    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Audio Visualization</div>', unsafe_allow_html=True)
        st.image(waveform_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- INTENSITY ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Instrument Intensity</div>', unsafe_allow_html=True)
    st.code(intensity_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- EXPORTS ----------------
    result = {
        "audio_file": uploaded_file.name,
        "detected_instrument": detected_name,
        "confidence": confidence,
        "scores": chart_data
    }

    pdf_path = generate_pdf(result, waveform_path, confidence_path, intensity_text)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            "⬇ Download JSON",
            json.dumps(result, indent=4),
            file_name="instrument_result.json",
            mime="application/json"
        )

    with dl2:
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download PDF",
                f,
                file_name="instrument_report.pdf",
                mime="application/pdf"
            )

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Welcome</div>
        <p style="color:#cbd5e1; margin-bottom:0;">
            Upload an audio file to detect the most likely musical instrument, visualize the waveform,
            review confidence scores, and export the results as JSON or PDF.
        </p>
    </div>
    """, unsafe_allow_html=True)
