# 🚀 Multimodal Fake Content Detection System (Image + Audio + Text + Video)

An advanced **AI-driven multimodal forensic system** that detects fake or misleading content across **image, audio, text, and video modalities**, with integrated **explainability (Grad-CAM, zero-shot reasoning, segment-level audio analysis, and fusion-based decision making)**.

This project simulates a real-world **AI content authenticity engine** combining deep learning, NLP, speech processing, and multimodal fusion into a unified interactive web application.

---

## 🔍 Key Features

### 🧠 Multimodal AI Detection Engine

* Supports **Image, Text, Audio, and Video** inputs
* Independent modality-wise prediction models
* Unified decision system via **confidence-based fusion**

---

### 🖼️ Image Fake Detection (CNN + Grad-CAM Explainability)

* Fine-tuned **ResNet18 classifier**
* Outputs:

  * Real / Fake classification
  * Confidence score
* **Explainability:**

  * Grad-CAM heatmap highlighting influential regions
  * Visual interpretation of model decision

---

### 📝 Text Fake Detection (Zero-Shot NLP Model)

* Uses `facebook/bart-large-mnli` transformer model
* Classifies:

  * Real vs Fake speech/content
* **Explainability:**

  * Token-level importance scoring
  * Masking-based contribution analysis
* Supports hypothesis-driven semantic reasoning

---

### 🎧 Audio Fake Detection (MFCC + Whisper + NLP Fusion)

* Pipeline:

  * Audio → Whisper transcription → NLP classification
* MFCC-based pretrained classifier for audio authenticity
* **Explainability:**

  * Segment-level speech breakdown
  * Confidence per audio segment
  * Transcription visualization

---

### 🎬 Video Fake Detection (Frame + Audio Fusion)

* Video decomposed into:

  * Key frames (CV2 extraction)
  * Audio track (FFmpeg extraction)
* Each frame analyzed via image model
* Audio analyzed via MFCC + Whisper pipeline
* Outputs:

  * Top contributing frame (with Grad-CAM)
  * Audio authenticity analysis
  * Unified video-level decision

---

### 🔗 Fusion Intelligence Engine

* Combines outputs from all modalities
* Weighted confidence aggregation:

  * Image weight (e.g., 0.4)
  * Audio weight (e.g., 0.6)
* Final decision:

  * Real / Fake / Uncertain
* **Explainability:**

  * Modality contribution breakdown
  * Confidence impact analysis

---

### 📊 Explainable AI Dashboard (XAI Layer)

* Grad-CAM visual overlays (image/video)
* Token importance heatmap (text)
* Audio segment-level explanations
* Fusion contribution visualization

---

### ⚡ Streamlit Interactive Interface

* Upload support for:

  * Image / Audio / Text / Video
* Real-time prediction pipeline
* Interactive explanation panels
* Fusion visualization dashboard

---

## 🧠 System Architecture

```
                ┌──────────────┐
                │   Input Data  │
                └──────┬───────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
 Image Model      Text Model       Audio Model
(ResNet18)     (BART MNLI)     (MFCC + Whisper)
     │                 │                 │
     └──────────────┬────────────────────┘
                    │
         Multimodal Fusion Engine
                    │
        Final Prediction (Real/Fake)
                    │
          Explainability Layer
 (Grad-CAM + NLP + Audio Segments)
```

---

## 🛠️ Tech Stack

### 🔬 Machine Learning / AI

* PyTorch
* TorchVision (ResNet18)
* Hugging Face Transformers
* Facebook BART (MNLI)
* OpenCV
* NumPy, SciPy
* Librosa (MFCC features)

---

### 🎧 Speech & Audio

* Whisper (speech-to-text)
* FFmpeg (audio extraction)
* MFCC feature extraction

---

### 🖼️ Computer Vision

* OpenCV
* Grad-CAM visualization
* PIL (image processing)

---

### 🌐 Web Application

* Streamlit
* HTML/CSS custom UI components
* Real-time inference pipeline

---

## 📂 Project Structure

```
Ritu_Minor/
│
├── fusion/                          # 🔥 MAIN STREAMLIT APP (DEPLOY THIS)
│   ├── app1.py                      # Streamlit frontend
│   │
│   ├── predict_image_1.py         # Image inference + Grad-CAM
│   ├── predict_text_1.py          # NLP inference (zero-shot / classifier)
│   ├── predict_audio_live_1.py    # Audio inference (MFCC + Whisper)
│   ├── predict_video_1.py         # Video multimodal inference
│   │
│   ├── explainability/            # 🔍 Interpretability modules (APP-USED)
│   │   ├── gradcam.py
│   │   ├── text_highlight.py
│   │   └── audio_segments.py
│   │
│   ├── assets/                    # UI images, icons, sample files
│   └── utils/ (optional)          # helper functions (recommended)
│
│
├── scripts/                        # 🧠 TRAINING + DATA PIPELINE (NOT DEPLOYED)
│   ├── dataset_loader.py
│   ├── train_image_model.py
│   ├── train_text_model.py
│   ├── train_audio_model.py
│   ├── train_video_model.py
│   ├── data_split.py
│   └── preprocessing/
│
│
├── models/                         # 🤖 MODEL ARTIFACTS (OPTIONAL IN GIT)
│   ├── resnet18_finetuned.pth
│   ├── audio_mfcc_model.pkl
│   ├── text_model.pkl (if any)
│   └── video_model.pth
│
│
├── data/                           # 📦 RAW / PROCESSED DATA (DO NOT PUSH)
│   ├── image/
│   ├── video/
│   └── audio/
│
│
├── README.md                       # 📘 Project documentation
├── requirements.txt                # dependencies
├── .gitignore                      # ignore data, venv, checkpoints

```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd fusion
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv env
env\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
streamlit run app1.py
```

---

## 📊 Key Outputs

### 🖼️ Image

* Fake/Real prediction
* Grad-CAM heatmap explanation

---

### 📝 Text

* Zero-shot classification
* Token-level importance scoring

---

### 🎧 Audio

* Transcription (Whisper)
* Segment-level fake/real detection

---

### 🎬 Video

* Key frame analysis
* Audio + visual fusion
* Final multimodal verdict

---

## 🔗 Fusion Output Example

```
Final Prediction: FAKE
Confidence: 91%

Image: Fake (0.99)
Audio: Real (0.53)

Contribution:
- Image: +0.40
- Audio: -0.32
```

---

## 🚀 Research Highlights

* Built a **true multimodal AI system**
* Integrated **deep learning + NLP + speech + CV**
* Designed **explainable AI pipeline (XAI)**
* Implemented **fusion-based decision architecture**
* Simulated real-world **AI content verification system**

---

## 📈 Future Improvements

* Transformer-based video understanding model
* Attention-based multimodal fusion (instead of weighted sum)
* Faster inference optimization (ONNX / TorchScript)
* Real-time streaming fake detection
* Deployment on cloud (AWS / HuggingFace Spaces)

---

## 👨‍💻 Author

Developed as a **Multimodal AI Research Project** for academic + placement portfolio.

---


* Real multimodal AI system (not single-model project)
* Explainable AI (XAI integrated, not black-box)
* Video + Audio + Text + Image fusion pipeline
* Industry-style architecture design
* Research + product hybrid system

---

