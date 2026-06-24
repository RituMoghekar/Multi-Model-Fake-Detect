
import os
import cv2
import subprocess
import tempfile
import numpy as np

from predict_image import predict_image
from predict_audio_live import predict_audio

FFMPEG_PATH = r"D:\ffmpeg\bin\ffmpeg.exe"
FRAME_INTERVAL = 30


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)

        frame_count += 1

    cap.release()
    return saved_frames


def analyze_frames(frame_paths):
    fake_scores = []
    explanations = []

    for frame in frame_paths:
        result = predict_image(frame)
        explanations.append(result.get("explanation", ""))

        if result["label"] == "fake":
            fake_scores.append(result["confidence"])
        else:
            fake_scores.append(1 - result["confidence"])

    avg_fake_score = float(np.mean(fake_scores))
    label = "fake" if avg_fake_score >= 0.5 else "real"

    unique_explanations = list(set(explanations))
    final_expl = unique_explanations[0] if unique_explanations else "Frame-level visual analysis"

    return {
        "modality": "image",
        "label": label,
        "confidence": avg_fake_score if label == "fake" else 1 - avg_fake_score,
        "explanation": final_expl
    }


def extract_audio(video_path, audio_path):
    command = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def predict_video(video_path):
    """
    Returns a LIST of modality-wise predictions
    """
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ---------- VIDEO FRAMES ----------
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frames = extract_frames(video_path, frames_dir)
        if frames:
            video_result = analyze_frames(frames)
            results.append(video_result)

        # ---------- AUDIO ----------
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio_result = predict_audio(audio_path)
            audio_result["modality"] = "audio"  # ✅ FORCE SAFETY
            results.append(audio_result)

    return results