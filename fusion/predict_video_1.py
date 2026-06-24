import os
import cv2
import tempfile
import subprocess
import numpy as np
from PIL import Image


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


def extract_audio(video_path, audio_path):
    FFMPEG_PATH = r"D:\ffmpeg\bin\ffmpeg.exe"

    command = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())


def predict_video(video_path):

    # 👉 import INSIDE function (IMPORTANT FIX)
    from predict_image_1 import predict_image
    from predict_audio_live_1 import predict_audio

    with tempfile.TemporaryDirectory() as tmpdir:

        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frames = extract_frames(video_path, frames_dir)

        image_results = []
        if frames:
            image_results = [predict_image(f) for f in frames]

        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)

        audio_result = None
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio_result = predict_audio(audio_path)

        final_results = []

        if image_results:
            image_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            final_results.append(image_results[0])

        if audio_result:
            final_results.append(audio_result)

        return final_results