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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles - Professional Dark Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Apply main white card directly to the Streamlit main block */
    .block-container {
        background: #ffffff;
        border-radius: 32px;
        padding: 2rem !important;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        max-width: 1200px;
    }

    /* Hide accidental empty blocks */
    div[data-testid="stVerticalBlock"] > div:empty {
        display: none;
    }

    /* Animated Gradient Title */
    .main-title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        animation: gradientShift 3s ease infinite;
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #4a5568;
        margin-bottom: 40px;
        font-size: 18px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        position: relative;
    }

    .subtitle::after {
        content: '';
        display: block;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        margin: 12px auto 0;
        border-radius: 2px;
    }

    /* Upload Section - Professional Style */
    .upload-section {
        background: #f7fafc;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .upload-section:hover {
        border-color: #2a5298;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }

    /* Result Card - Professional Gradient */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 32px;
        border-radius: 24px;
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px -15px rgba(30, 60, 114, 0.4);
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease-out;
        font-family: 'Inter', sans-serif;
    }

    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px -12px rgba(30, 60, 114, 0.6);
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Custom File Uploader Text */
    .stFileUploader label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    .stFileUploader {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(30, 60, 114, 0.4);
    }

    /* Audio Player */
    .stAudio {
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }

    /* Headers - Dark and Clear */
    h1, h2, h3, .stSubheader {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1a202c !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    h2 {
        font-size: 28px !important;
        border-left: 4px solid #2a5298;
        padding-left: 16px;
    }

    h3 {
        font-size: 22px !important;
        color: #2d3748 !important;
    }

    /* Regular Text */
    p, .stMarkdown, .stText {
        color: #4a5568 !important;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }

    /* Code Block - Professional Style */
    .stCodeBlock, pre {
        background: #f1f5f9 !important;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        color: #1a202c !important;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }

    code {
        color: #2a5298 !important;
        background: #eef2ff !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Waveform Container - Clean White */
    .waveform-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 12px;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .stSpinner {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Success Message */
    .stSuccess {
        background: #f0fdf4 !important;
        color: #166534 !important;
        border-radius: 12px !important;
        border-left: 4px solid #22c55e !important;
        padding: 1rem !important;
    }

    /* Info/Warning Messages */
    .stInfo {
        background: #eff6ff !important;
        color: #1e40af !important;
        border-radius: 12px !important;
        border-left: 4px solid #3b82f6 !important;
    }

    /* Dataframe/Table */
    .stDataFrame {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #4a5568 !important;
    }

    /* Slider/Labels */
    .stSlider label {
        color: #1a202c !important;
        font-weight: 500 !important;
    }

    /* Select Box */
    .stSelectbox label {
        color: #1a202c !important;
    }

    /* Radio Buttons */
    .stRadio label {
        color: #1a202c !important;
    }

    /* Checkbox */
    .stCheckbox label {
        color: #1a202c !important;
    }

    /* Number Input */
    .stNumberInput label {
        color: #1a202c !important;
    }

    /* Text Input */
    .stTextInput label {
        color: #1a202c !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #1a202c !important;
        font-weight: 600 !important;
        background: #f8fafc !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #4a5568 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
    }

    .stTabs [aria-selected="true"] {
        color: #2a5298 !important;
        background: white !important;
        border-bottom: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 36px;
        }

        .result-card {
            font-size: 24px;
            padding: 20px;
        }

        .block-container {
            padding: 1rem !important;
        }

        h2 {
            font-size: 24px !important;
        }

        h3 {
            font-size: 20px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER WITH ANIMATION ----------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div class="main-title">🎵 InstruNet AI</div>
    <div class="subtitle">Automatic Musical Instrument Detection from Audio</div>
</div>
""", unsafe_allow_html=True)

# ---------------- LABELS + ICONS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻"),
}
labels = list(label_map.keys())

# ---------------- MODEL SETUP ----------------
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
def audio_to_spectrogram(audio_path, img_size=224):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=4096,
        hop_length=256,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db, cmap="magma")
    plt.axis("off")
    plt.savefig("temp_spec.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp_spec.png").convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)


# ---------------- INTENSITY TEXT ----------------
def generate_intensity_text(scores):
    text = "🎵 Instrument Intensity:\n\n"
    for inst, val in scores.items():
        bars = "█" * int(val * 20)
        text += f"{inst:20} [{bars:<20}] {val:.2f}\n"
    return text


# ---------------- WAVEFORM IMAGE ----------------
def create_waveform_image(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    plt.figure(figsize=(8, 3))
    plt.style.use("default")
    plt.plot(y, color="#2a5298", alpha=0.8, linewidth=1)
    plt.title("Audio Waveform Analysis", color="#1a202c", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", color="#4a5568")
    plt.ylabel("Amplitude", color="#4a5568")
    plt.grid(alpha=0.3, color="#cbd5e0")
    plt.tight_layout()
    plt.savefig("waveform.png", facecolor="white", edgecolor="none", dpi=100)
    plt.close()

    return "waveform.png"


# ---------------- CONFIDENCE GRAPH ----------------
def create_confidence_graph(scores):
    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 3))
    plt.style.use("default")
    colors = ["#1e3c72", "#2a5298", "#3b6cb0", "#4c7fc9"]
    bars = plt.bar(names, values, color=colors, alpha=0.8)
    plt.ylabel("Confidence Score", color="#4a5568", fontsize=12)
    plt.ylim(0, 1)
    plt.title("Instrument Detection Confidence", color="#1a202c", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, color="#cbd5e0")
    plt.tight_layout()

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            color="#1a202c",
            fontweight="bold",
        )

    plt.savefig("confidence.png", facecolor="white", edgecolor="none", dpi=100)
    plt.close()

    return "confidence.png"


# ---------------- PDF GENERATION ----------------
def generate_pdf(result, waveform_path, confidence_path, intensity_text):
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("InstruNet AI Report", styles["Title"]))
    elements.append(Spacer(1, 15))
    elements.append(
        Paragraph(f"<b>Audio File:</b> {result['audio_file']}", styles["Normal"])
    )
    elements.append(
        Paragraph(
            f"<b>Detected Instrument:</b> {result['detected_instrument']}",
            styles["Normal"],
        )
    )
    elements.append(
        Paragraph(f"<b>Confidence:</b> {result['confidence']:.2f}", styles["Normal"])
    )

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Instrument Intensity:</b>", styles["Heading2"]))
    elements.append(Preformatted(intensity_text, styles["Code"]))
    elements.append(Spacer(1, 15))

    if os.path.exists(waveform_path):
        elements.append(Paragraph("<b>Audio Waveform:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(waveform_path, width=400, height=150))
        elements.append(Spacer(1, 15))

    if os.path.exists(confidence_path):
        elements.append(Paragraph("<b>Confidence Scores:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(confidence_path, width=400, height=200))

    doc.build(elements)
    return pdf_path


# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "🎵 Choose your audio file (WAV or MP3)",
    type=["wav", "mp3"],
    help="Upload a clear recording of a musical instrument for best results"
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎧 Audio Preview")
        st.audio("input_audio.wav")

    with st.spinner("🎵 Analyzing your audio with InstruNet AI..."):
        X_test = audio_to_spectrogram("input_audio.wav")
        pred = model.predict(X_test)[0]

        detected_code = labels[np.argmax(pred)]
        detected_name, icon = label_map[detected_code]
        confidence = float(np.max(pred))

        waveform_path = create_waveform_image("input_audio.wav")

    # Waveform Visualization
    st.markdown("### 📊 Audio Analysis")
    st.markdown('<div class="waveform-container">', unsafe_allow_html=True)
    st.image(waveform_path, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Result Card
    st.markdown(
        f'<div class="result-card">'
        f"{icon} {detected_name}<br>"
        f'<span style="font-size: 18px;">Confidence: {confidence:.1%}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Confidence Scores
    st.markdown("### 📈 Confidence Analysis")
    chart_data = {
        label_map[labels[i]][0]: float(pred[i]) for i in range(len(pred))
    }
    st.bar_chart(chart_data)

    confidence_path = create_confidence_graph(chart_data)
    st.image(confidence_path, use_container_width=True)

    # Intensity Text
    intensity_text = generate_intensity_text(chart_data)
    st.markdown("### 🎯 Instrument Intensity")
    st.code(intensity_text, language="")

    result = {
        "audio_file": uploaded_file.name,
        "detected_instrument": detected_name,
        "confidence": confidence,
        "scores": chart_data,
    }

    json_str = json.dumps(result, indent=4)

    # Download Section
    st.markdown("### 📥 Download Reports")
    st.markdown("Get detailed analysis reports in your preferred format")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 Download JSON Report",
            json_str,
            file_name="prediction.json",
            mime="application/json",
            use_container_width=True,
        )

    pdf_path = generate_pdf(
        result,
        waveform_path,
        confidence_path,
        intensity_text,
    )

    with open(pdf_path, "rb") as f:
        with col2:
            st.download_button(
                "📑 Download PDF Report",
                f,
                file_name="prediction.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.success("✅ Analysis complete! Your reports are ready for download.")

# Footer
st.markdown("""
<div class="footer">
    <p style="color: #718096;">InstruNet AI - Powered by Deep Learning | Made with 🎵 for musicians and audio enthusiasts</p>
    <p style="color: #a0aec0; font-size: 10px;">Supports Piano, Acoustic Guitar, Electric Guitar, and Violin detection</p>
</div>
""", unsafe_allow_html=True)
