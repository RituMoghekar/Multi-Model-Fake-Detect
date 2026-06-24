import librosa
import numpy as np
import joblib
import subprocess

# Absolute paths
MODEL_PATH = r"D:\Ritu_Minor\models\audio\audio_mfcc_model.pkl"
WHISPER_EXE = r"D:\Ritu_Minor\models\Release\whisper-cli.exe"
WHISPER_MODEL = r"D:\Ritu_Minor\models\Release\models\ggml-tiny.en.bin"

print("🔊 Loading MFCC audio model...")
model = joblib.load(MODEL_PATH)
print("✅ Audio MFCC model loaded\n")


def confidence_level(score):
    if score >= 0.80:
        return "high"
    elif score >= 0.60:
        return "medium"
    else:
        return "low"


def extract_mfcc(audio_path, n_mfcc=40):
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def extract_text_from_audio(audio_path):
    """Use Whisper CLI to transcribe audio into text."""
    try:
        command = [
            WHISPER_EXE,
            "-m", WHISPER_MODEL,
            "-f", audio_path,
            "-nt"  # no timestamps → cleaner output
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return "Transcription failed"

        # Clean output
        text = result.stdout.strip()
        return text if text else "No speech detected"

    except Exception as e:
        return f"Error in transcription: {str(e)}"


def predict_audio(audio_path):
    """Predict whether audio is real or fake using MFCC + Whisper transcription."""
    features = extract_mfcc(audio_path)

    if features is None:
        return {
            "modality": "audio",
            "label": "uncertain",
            "confidence": 0.0,
            "confidence_level": "low",
            "explanation": "Audio could not be processed",
            "transcription": "Audio processing failed"
        }

    features = features.reshape(1, -1)

    # Predict probabilities
    probs = model.predict_proba(features)[0]

    # Since classes are [0, 1] and 1 = fake
    real_prob = probs[0]
    fake_prob = probs[1]

    # Final label
    label = "fake" if fake_prob > real_prob else "real"

    # Confidence should match predicted label
    confidence = max(real_prob, fake_prob)

    # Whisper transcription
    transcription = extract_text_from_audio(audio_path)

    # Debug logs
    print("\n===== AUDIO DEBUG =====")
    print("Audio:", audio_path)
    print("Probabilities:", probs)
    print("Real probability:", real_prob)
    print("Fake probability:", fake_prob)
    print("Predicted label:", label)
    print("Confidence:", confidence)
    print("========================\n")

    return {
        "modality": "audio",
        "label": label,
        "confidence": float(confidence),
        "confidence_level": confidence_level(confidence),
        "explanation": "Prediction based on MFCC acoustic patterns",
        "transcription": transcription
    }
