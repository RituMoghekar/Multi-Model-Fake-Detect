import subprocess
import joblib
import streamlit as st
from transformers import pipeline
from explainability.audio_segments import explain_audio_segments

# ---------------- PATHS ----------------
WHISPER_EXE = r"D:\Ritu_Minor\models\Release\whisper-cli.exe"
WHISPER_MODEL = r"D:\Ritu_Minor\models\Release\models\ggml-tiny.en.bin"
MODEL_PATH = r"D:\Ritu_Minor\models\audio\audio_mfcc_model.pkl"

LABELS = ["real speech", "fake speech"]


# ---------------- CACHE MODELS (IMPORTANT FIX) ----------------
@st.cache_resource
def load_mfcc_model():
    print("🔊 Loading MFCC audio model...")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_classifier():
    print("🧠 Loading zero-shot classifier...")
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )


model = load_mfcc_model()
classifier = load_classifier()


# ---------------- WHISPER ----------------
def transcribe_audio(audio_path):
    command = [
        WHISPER_EXE,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "-nt",
        "-t", "4"   # speed improvement (threads)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


# ---------------- CONFIDENCE LABEL ----------------
def confidence_level(score):
    if score >= 0.80:
        return "high"
    elif score >= 0.60:
        return "medium"
    return "low"


# ---------------- MAIN FUNCTION ----------------
def predict_audio(audio_path):

    # 1. TRANSCRIPTION
    text = transcribe_audio(audio_path)

    if not text or len(text.strip()) < 5:
        raise ValueError("Audio transcription too short.")

    # 2. CLASSIFICATION
    prediction = classifier(text, LABELS)

    pred_label_raw = prediction["labels"][0]
    confidence = float(prediction["scores"][0])

    label = "real" if "real" in pred_label_raw else "fake"

    # 3. EXPLAINABILITY (safe fallback)
    segments = [{
        "start": 0,
        "end": 1,
        "text": text
    }]

    explanations = explain_audio_segments(segments, classifier)

    return {
        "modality": "audio",
        "label": label,
        "confidence": confidence,
        "confidence_level": confidence_level(confidence),
        "transcription": text,
        "explanation": text,
        "segments": explanations
    }


# ---------------- TEST ----------------
if __name__ == "__main__":
    test_audio = "sample.wav"
    print(predict_audio(test_audio))