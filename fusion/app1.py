# app.py
import streamlit as st
import tempfile
import os

from predict_image_1 import predict_image
from predict_video_1 import predict_video
from predict_audio_live_1 import predict_audio
from predict_text_1 import predict_text  # zero-shot text predictor
from fusion import fuse_predictions

if "input_type" not in st.session_state:
    st.session_state["input_type"] = "Image"


def format_label(label):
    if label == "fake":
        return "⚠️ Misleading"
    elif label == "real":
        return "✅ Factual"
    elif label == "uncertain":
        return "❓ Uncertain"
    return label


st.set_page_config(page_title="Multi-Modal Fake Content Detection", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────
#  GLOBAL CSS + ANIMATIONS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Root variables ── */
:root {
  --bg-deep:       #020408;
  --bg-card:       #070d14;
  --bg-card2:      #0b1420;
  --border-dim:    rgba(0,200,255,0.12);
  --border-bright: rgba(0,200,255,0.45);
  --cyan:          #00c8ff;
  --cyan-dim:      rgba(0,200,255,0.15);
  --red:           #ff2d55;
  --red-dim:       rgba(255,45,85,0.18);
  --green:         #00ff88;
  --green-dim:     rgba(0,255,136,0.18);
  --yellow:        #ffc400;
  --yellow-dim:    rgba(255,196,0,0.18);
  --text-bright:   #e8f4ff;
  --text-mid:      #7ea8c4;
  --font-display:  'Orbitron', monospace;
  --font-body:     'Rajdhani', sans-serif;
  --font-mono:     'Space Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  color: var(--text-bright) !important;
}

/* Deep space background */
.stApp {
  background: var(--bg-deep) !important;
  background-image:
    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,120,200,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 80% 80%, rgba(0,60,120,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 30% 20% at 10% 90%, rgba(0,200,255,0.06) 0%, transparent 50%) !important;
}

/* Hide Streamlit chrome */
footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 960px !important; }

/* ── Scanline overlay ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,200,255,0.015) 2px,
    rgba(0,200,255,0.015) 4px
  );
  pointer-events: none;
  z-index: 999;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.3); border-radius: 3px; }

/* ──────────────────────────────────────────────
   HEADER
────────────────────────────────────────────── */
.hdr-wrap {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  position: relative;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(0,0,0,0));
  border-bottom: 1px solid rgba(0,200,255,0.15);
  backdrop-filter: blur(6px);
}
.hdr-wrap::before {
  content: '';
  position: absolute;
  top: 0; left: 50%; transform: translateX(-50%);
  width: 500px; height: 3px;
  box-shadow: 0 0 15px var(--cyan), 0 0 40px var(--cyan);
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: scanH 3s ease-in-out infinite;
}
@keyframes scanH {
  0%,100% { opacity: 0.4; width: 200px; }
  50%      { opacity: 1;   width: 420px; }
}

.hdr-badge {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.25em;
  color: var(--cyan);
  background: rgba(0,200,255,0.08);
  border: 1px solid rgba(0,200,255,0.3);
  border-radius: 2px;
  padding: 3px 12px;
  margin-bottom: 1rem;
  animation: pulse-badge 2s ease-in-out infinite;
}
@keyframes pulse-badge {
  0%,100% { box-shadow: 0 0 6px rgba(0,200,255,0.2); }
  50%      { box-shadow: 0 0 18px rgba(0,200,255,0.6); }
}

.hdr-title {
  font-family: var(--font-display) !important;
  font-size: clamp(1.4rem, 3.5vw, 2.3rem) !important;
  font-weight: 900 !important;
  letter-spacing: 0.04em;
  line-height: 1.2;
  color: #fff !important;
  text-shadow:
    0 0 20px rgba(0,200,255,0.8),
    0 0 60px rgba(0,200,255,0.35),
    0 0 100px rgba(0,200,255,0.15);
  animation: title-glow 3s ease-in-out infinite alternate;
  margin-bottom: 0.6rem;
}
@keyframes title-glow {
  from { text-shadow: 0 0 15px rgba(0,200,255,0.6), 0 0 50px rgba(0,200,255,0.25); }
  to   { text-shadow: 0 0 30px rgba(0,200,255,1.0), 0 0 80px rgba(0,200,255,0.55), 0 0 120px rgba(0,200,255,0.25); }
}

.hdr-sub {
  font-family: var(--font-body);
  font-size: 1.05rem;
  font-weight: 300;
  color: var(--text-mid) !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ──────────────────────────────────────────────
   SECTION LABELS
────────────────────────────────────────────── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.3em;
  color: #00e0ff;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  opacity: 0.9;
}

/* ──────────────────────────────────────────────
   MODALITY CARDS  (pure HTML – rendered once)
────────────────────────────────────────────── */
.modality-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 1.5rem;
}
.mod-card {
  background: var(--bg-card2);
  border: 1px solid var(--border-dim);
  border-radius: 8px;
  padding: 18px 10px;
  text-align: center;
  cursor: pointer;
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.mod-card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(0,200,255,0.06) 0%, transparent 60%);
  opacity: 0;
  transition: opacity 0.25s;
}
.mod-card:hover { border-color: var(--border-bright); transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,200,255,0.15); }
.mod-card:hover::after { opacity: 1; }
.mod-card.active { border-color: var(--cyan); box-shadow: 0 0 20px rgba(0,200,255,0.35), inset 0 0 20px rgba(0,200,255,0.05); }
.mod-icon { font-size: 2rem; display: block; margin-bottom: 8px; }
.mod-label { font-family: var(--font-display); font-size: 0.72rem; letter-spacing: 0.1em; color: var(--text-mid); text-transform: uppercase; }

/* ──────────────────────────────────────────────
   STREAMLIT SELECT BOX  (invisible but functional)
────────────────────────────────────────────── */
div[data-testid="stSelectbox"] {
  display: none !important;
}

/* ──────────────────────────────────────────────
   CARD CONTAINERS
────────────────────────────────────────────── */
.panel {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(0,200,255,0.1);
  border-radius: 6px;
  padding: 1.2rem;
  
}
.panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.4;
}

/* ──────────────────────────────────────────────
   FILE UPLOADER
────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg-card2) !important;
  border: 1px dashed rgba(0,200,255,0.25) !important;
  border-radius: 8px !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0,200,255,0.55) !important;
  box-shadow: 0 0 20px rgba(0,200,255,0.1) !important;
}
[data-testid="stFileUploader"] label {
  color: var(--text-mid) !important;
  font-family: var(--font-body) !important;
}

/* Text area */
textarea {
  background: var(--bg-card2) !important;
  border: 1px solid rgba(0,200,255,0.2) !important;
  border-radius: 6px !important;
  color: var(--text-bright) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}
textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 12px rgba(0,200,255,0.2) !important;
}

/* ──────────────────────────────────────────────
   ANALYZE BUTTON
────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
  font-family: var(--font-display) !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  color: #000 !important;
  background: linear-gradient(135deg, #00c8ff 0%, #0080cc 100%) !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.75rem 2.5rem !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: hidden !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 0 20px rgba(0,200,255,0.4), 0 4px 15px rgba(0,0,0,0.5) !important;
}

div[data-testid="stButton"] > button {
  width: 100% !important;
}

div[data-testid="stButton"] > button::before {
  content: '';
  position: absolute;
  top: -50%; left: -60%;
  width: 40%; height: 200%;
  background: rgba(255,255,255,0.25);
  transform: skewX(-20deg);
  transition: left 0.4s ease;
}
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 35px rgba(0,200,255,0.7), 0 6px 20px rgba(0,0,0,0.6) !important;
}
div[data-testid="stButton"] > button:hover::before { left: 130%; }
div[data-testid="stButton"] > button:active {
  transform: translateY(0px) scale(0.98) !important;
  box-shadow: 0 0 15px rgba(0,200,255,0.5) !important;
}

/* ──────────────────────────────────────────────
   RESULT CARDS
────────────────────────────────────────────── */
.result-card {
  border-radius: 10px;
  padding: 1.5rem 1.8rem;
  margin-bottom: 1rem;
  animation: fadeSlide 0.5s ease forwards;
  position: relative;
  overflow: hidden;
}
@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

.result-fake {
  background: linear-gradient(135deg, rgba(255,45,85,0.12) 0%, rgba(10,10,18,0.95) 60%);
  border: 1px solid rgba(255,45,85,0.5);
  box-shadow: 0 0 30px rgba(255,45,85,0.2), inset 0 0 40px rgba(255,45,85,0.05);
}
.result-real {
  background: linear-gradient(135deg, rgba(0,255,136,0.12) 0%, rgba(10,10,18,0.95) 60%);
  border: 1px solid rgba(0,255,136,0.5);
  box-shadow: 0 0 30px rgba(0,255,136,0.2), inset 0 0 40px rgba(0,255,136,0.05);
}
.result-uncertain {
  background: linear-gradient(135deg, rgba(255,196,0,0.12) 0%, rgba(10,10,18,0.95) 60%);
  border: 1px solid rgba(255,196,0,0.5);
  box-shadow: 0 0 30px rgba(255,196,0,0.2), inset 0 0 40px rgba(255,196,0,0.05);
}

.result-modality {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  opacity: 0.6;
}
.result-verdict {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 900;
  letter-spacing: 0.05em;
  margin-bottom: 0.4rem;
}
.verdict-fake     { color: var(--red);    text-shadow: 0 0 20px rgba(255,45,85,0.8); }
.verdict-real     { color: var(--green);  text-shadow: 0 0 20px rgba(0,255,136,0.8); }
.verdict-uncertain{ color: var(--yellow); text-shadow: 0 0 20px rgba(255,196,0,0.8); }

.conf-bar-wrap {
  margin: 10px 0 12px;
  height: 6px;
  background: rgba(255,255,255,0.07);
  border-radius: 3px;
  overflow: hidden;
}
.conf-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 1s ease;
  animation: barGrow 1s ease forwards;
}
@keyframes barGrow { from { width: 0 !important; } }

.conf-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-mid);
  margin-bottom: 10px;
}

.explain-box {
  background: rgba(0,0,0,0.3);
  border-left: 2px solid rgba(0,200,255,0.3);
  border-radius: 0 4px 4px 0;
  padding: 8px 12px;
  font-family: var(--font-body);
  font-size: 0.88rem;
  color: var(--text-mid) !important;
  margin-top: 8px;
}

/* ──────────────────────────────────────────────
   FUSION PANEL
────────────────────────────────────────────── */
.fusion-panel {
  background: linear-gradient(135deg, rgba(0,200,255,0.06) 0%, rgba(7,13,20,0.98) 50%);
  border: 1px solid var(--border-bright);
  border-radius: 10px;
  padding: 1.6rem 1.8rem;
  margin-top: 1.2rem;
  position: relative;
  animation: fadeSlide 0.6s ease 0.2s both;
  box-shadow: 0 0 40px rgba(0,200,255,0.1);
}
.fusion-panel::before {
  content: 'FUSION ANALYSIS';
  position: absolute;
  top: -10px; left: 20px;
  font-family: var(--font-mono);
  font-size: 0.55rem;
  letter-spacing: 0.3em;
  color: var(--cyan);
  background: var(--bg-deep);
  padding: 0 10px;
}

.fusion-score-ring {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 1rem;
}
.ring-num {
  font-family: var(--font-display);
  font-size: 3.2rem;
  font-weight: 900;
  line-height: 1;
}
.ring-meta { flex: 1; }
.ring-meta-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.25em;
  color: var(--text-mid);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.ring-meta-value {
  font-family: var(--font-display);
  font-size: 1.3rem;
  font-weight: 700;
}

/* JSON / detail boxes */
.detail-json {
  background: rgba(0,0,0,0.4);
  border: 1px solid rgba(0,200,255,0.1);
  border-radius: 6px;
  padding: 10px 14px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-mid) !important;
  margin-bottom: 8px;
  white-space: pre-wrap;
  word-break: break-all;
}
.detail-json span.key   { color: var(--cyan); }
.detail-json span.str   { color: #a8ff80; }
.detail-json span.num   { color: #ffd580; }
.detail-json span.bool  { color: #ff8080; }

/* ──────────────────────────────────────────────
   STATUS MESSAGES
────────────────────────────────────────────── */
.stInfo, [data-testid="stNotification"] {
  background: rgba(0,120,200,0.12) !important;
  border: 1px solid rgba(0,200,255,0.25) !important;
  border-radius: 6px !important;
  color: var(--cyan) !important;
  font-family: var(--font-body) !important;
}
.stSuccess {
  background: rgba(0,200,100,0.1) !important;
  border: 1px solid rgba(0,255,136,0.3) !important;
  border-radius: 6px !important;
  color: var(--green) !important;
  font-family: var(--font-body) !important;
}
.stError {
  background: rgba(255,45,85,0.1) !important;
  border: 1px solid rgba(255,45,85,0.4) !important;
  border-radius: 6px !important;
  color: var(--red) !important;
  font-family: var(--font-body) !important;
}
.stWarning {
  background: rgba(255,196,0,0.1) !important;
  border: 1px solid rgba(255,196,0,0.3) !important;
  border-radius: 6px !important;
  color: var(--yellow) !important;
  font-family: var(--font-body) !important;
}

/* ──────────────────────────────────────────────
   FOOTER
────────────────────────────────────────────── */
.footer {
  text-align: center;
  padding: 2rem 1rem 1rem;
  border-top: 1px solid var(--border-dim);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(0,0,0,0));
  border-bottom: 1px solid rgba(0,200,255,0.15);
  backdrop-filter: blur(6px);
  margin-top: 2rem;
}
.footer-text {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  color: rgba(126,168,196,0.45);
  text-transform: uppercase;
}
.footer-line {
  width: 60px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  margin: 8px auto;
  opacity: 0.4;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SOUND EFFECTS  (tiny embedded JS)
# ──────────────────────────────────────────────
st.markdown("""
<script>
function playTone(type) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain); gain.connect(ctx.destination);
  if (type === 'click') {
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
    osc.start(); osc.stop(ctx.currentTime + 0.1);
  } else if (type === 'success') {
    osc.frequency.setValueAtTime(523, ctx.currentTime);
    osc.frequency.setValueAtTime(659, ctx.currentTime + 0.1);
    osc.frequency.setValueAtTime(784, ctx.currentTime + 0.2);
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
    osc.start(); osc.stop(ctx.currentTime + 0.45);
  } else if (type === 'warning') {
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(220, ctx.currentTime);
    osc.frequency.setValueAtTime(180, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.start(); osc.stop(ctx.currentTime + 0.35);
  }
}
// expose globally so inline onclick can call it
window.playTone = playTone;
</script>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hdr-wrap">
  <div class="hdr-badge">◈ THREAT INTELLIGENCE SYSTEM ◈</div>
  <div class="hdr-title">Multimodal Fake Content<br>Detection System</div>
  <div class="hdr-sub">Detecting deception across text • image • audio • video</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
#  MODALITY SELECTION (FINAL CLEAN VERSION)
# ──────────────────────────────────────────────

st.markdown('<div class="section-label"> SELECT MODALITY</div>', unsafe_allow_html=True)

# Initialize state (important)
if "input_type" not in st.session_state:
    st.session_state["input_type"] = "Image"

cols = st.columns(4)

modalities = ["Image", "Audio", "Video", "Text"]
icons = ["🖼️", "🎙️", "🎥", "📝"]

for i, mod in enumerate(modalities):
    with cols[i]:

        is_active = (st.session_state["input_type"] == mod)

        btn = st.button(f"{icons[i]}\n{mod}", key=f"btn_{mod}")

        if btn:
            st.session_state["input_type"] = mod

        if is_active:
            st.markdown(f"""
            <div style="
                text-align:center;
                margin-top:-10px;
                color:#00c8ff;
                font-size:12px;
                letter-spacing:2px;">
                ▲ ACTIVE
            </div>
            """, unsafe_allow_html=True)

# Final selected modality
input_type = st.session_state["input_type"]

# ──────────────────────────────────────────────
#  INPUT AREA
# ──────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(f'<div class="section-label"> UPLOAD {input_type.upper()} FILE</div>', unsafe_allow_html=True)

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        show_preview = st.checkbox("👁️ Show Preview", key="img_preview")

        if show_preview:
            st.markdown('<div class="section-label"> PREVIEW</div>', unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            uploaded_file.seek(0)

elif input_type == "Audio":
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        show_preview = st.checkbox("👁️ Show Preview", key="audio_preview")

        if show_preview:
            st.markdown('<div class="section-label"> PREVIEW</div>', unsafe_allow_html=True)
            st.audio(uploaded_file)
            uploaded_file.seek(0)


elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file is not None:
        show_preview = st.checkbox("👁️ Show Preview", key="video_preview")

        if show_preview:
            st.markdown('<div class="section-label"> PREVIEW</div>', unsafe_allow_html=True)
            video_bytes = uploaded_file.read()
            st.video(video_bytes)
            uploaded_file.seek(0)
            #st.video(uploaded_file)
else:
    uploaded_file = None

uploaded_text = None

if input_type == "Text":
    st.markdown('<div class="section-label" style="margin-top:1rem">ENTER TEXT</div>', unsafe_allow_html=True)

    uploaded_text = st.text_area(
        "Paste or type content to analyze:",
        height=160,
        placeholder="Enter text content to analyze for authenticity...",
        label_visibility="collapsed"
    )


st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  ANALYZE BUTTON
# ──────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 3, 2])

with col2:
    analyze = st.button("🚀 ANALYZE CONTENT", use_container_width=True)

# ──────────────────────────────────────────────
#  HELPER: render a result card
# ──────────────────────────────────────────────
def result_card(modality: str, label: str, confidence: float, explanation: str):
    raw = str(label).strip().lower()

    if raw == "fake":
        css_class = "result-fake"
        verdict_class = "verdict-fake"

    elif raw == "real":
        css_class = "result-real"
        verdict_class = "verdict-real"

    else:
        css_class = "result-uncertain"
        verdict_class = "verdict-uncertain"

    color_map = {"fake": "#ff2d55", "real": "#00ff88", "uncertain": "#ffc400"}
    bar_color = color_map.get(raw, "#00c8ff")
    pct = int(confidence * 100)
    if isinstance(explanation, str) and explanation != "N/A":
        explain_html = f'<div class="explain-box">ℹ️ {explanation}</div>'
    else:
        explain_html = ""
    st.markdown(f"""
    <div class="result-card {css_class}">
      <div class="result-modality">◈ {modality.upper()} ANALYSIS</div>
      <div class="result-verdict {verdict_class}">{format_label(label)}</div>
      <div class="conf-label">CONFIDENCE · {pct}%</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:{pct}%; background: linear-gradient(90deg, {bar_color}88, {bar_color});"></div>
      </div>
      {explain_html}
    </div>
    """, unsafe_allow_html=True)


def render_json_pretty(obj):
    import json
    raw = json.dumps(obj, indent=2)
    # simple colorisation
    import re
    raw = re.sub(r'"([^"]+)":', r'<span class="key">"\1"</span>:', raw)
    raw = re.sub(r': "([^"]*)"', r': <span class="str">"\1"</span>', raw)
    raw = re.sub(r': (\d+\.?\d*)', r': <span class="num">\1</span>', raw)
    raw = re.sub(r': (true|false|null)', r': <span class="bool">\1</span>', raw)
    st.markdown(f'<div class="detail-json">{raw}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  PREDICTION LOGIC
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
if analyze:

    if input_type == "Text":
        if not uploaded_text or not uploaded_text.strip():
            st.warning("⚠️ Please enter valid text.")
            st.stop()
    else:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a file.")
            st.stop()

    predictions = []

    try:
        # ── loading indicator ──
        with st.spinner("⚡ Analyzing content... Please wait"):

            # ---------- IMAGE ----------
            if input_type == "Image" and uploaded_file:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                st.info("🔍 Predicting image...")

                try:
                    img_result = predict_image(tmp_path)
                    predictions.append(img_result)

                    result_card(
                        "Image",
                        img_result['label'],
                        img_result['confidence'],
                        img_result.get("explanation", "N/A")
                    )

                    # ───────── IMAGE EXPLAINABILITY ─────────
                    st.markdown(
                        '<div class="section-label">IMAGE EXPLAINABILITY</div>',
                        unsafe_allow_html=True
                    )

                    explanation = img_result.get("explanation")

                    if explanation is not None:
                        st.image(
                            explanation,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True
                        )
                    else:
                        st.info("No visual explanation available.")

                except Exception as e:
                    st.error(f"Image model failed: {str(e)}")
                    st.stop()

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # ---------- AUDIO ----------
            if input_type == "Audio" and uploaded_file:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name

                st.info("🔍 Predicting audio using MFCC model...")

                try:
                    audio_result = predict_audio(audio_path)
                    predictions.append(audio_result)

                    result_card(
                        "Audio",
                        audio_result['label'],
                        audio_result['confidence'],
                        audio_result.get("explanation", "N/A")
                    )

                    # ───────── AUDIO EXPLAINABILITY ─────────
                    st.markdown(
                        '<div class="section-label">AUDIO EXPLAINABILITY</div>',
                        unsafe_allow_html=True
                    )

                    segments = audio_result.get("segments")

                    if segments:
                        for seg in segments:
                            st.markdown(
                                f"**{seg['label']}** "
                                f"[{seg['confidence']:.2f}] "
                                f"({seg['start']:.2f}s - {seg['end']:.2f}s): "
                                f"{seg['text']}"
                            )
                    else:
                        st.info("No segment-level explanation available.")

                    # Whisper transcription
                    st.markdown(
                        '<div class="section-label">EXTRACTED SPEECH</div>',
                        unsafe_allow_html=True
                    )

                    transcription = audio_result.get("transcription")
                    if not transcription:
                        transcription = "No transcription available"

                    st.markdown(f"""
                    <div class="detail-json" style="
                        border-left:3px solid #00c8ff;
                        background: rgba(0,200,255,0.05);
                        padding:10px;
                        border-radius:6px;">
                        {transcription}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Audio model failed: {str(e)}")

                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

            # ---------- VIDEO ----------
            if input_type == "Video" and uploaded_file:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name

                st.info("🔍 Predicting video (frames + audio MFCC)...")

                try:
                    video_results = predict_video(video_path)
                    predictions.extend(video_results)

                    image_results = [
                        r for r in video_results if r["modality"] == "image"
                    ]
                    audio_results = [
                        r for r in video_results if r["modality"] == "audio"
                    ]

                    for r in video_results:
                        result_card(
                            r['modality'],
                            r['label'],
                            r['confidence'],
                            r.get("explanation", "N/A")
                        )

                    # ---------- VIDEO IMAGE EXPLAINABILITY ----------
                    image_results = sorted(image_results, key=lambda x: x.get("confidence", 0), reverse=True)
                    if image_results:
                        top_image = image_results[0]

                        st.markdown(
                            '<div class="section-label">VIDEO FRAME EXPLAINABILITY</div>',
                            unsafe_allow_html=True
                        )

                        explanation = top_image.get("explanation")

                        if explanation is not None:
                            try:
                                st.image(
                                explanation,
                                caption="Most Important Frame (Grad-CAM)",
                                use_container_width=True
                                )
                            except Exception as e:
                                st.warning(f"Could not render explanation: {e}")
                        else:
                            st.info("No visual explanation detected in video frames.")

                    # ---------- VIDEO AUDIO EXPLAINABILITY ----------
                    if audio_results:
                        audio_result = audio_results[0]

                        st.markdown(
                            '<div class="section-label">VIDEO AUDIO EXPLAINABILITY</div>',
                            unsafe_allow_html=True
                        )

                        segments = audio_result.get("segments")

                        if segments and isinstance(segments, list):
                            for seg in segments:
                                st.markdown(
                                    f"**{seg['label']}** "
                                    f"[{seg['confidence']:.2f}] "
                                    f"({seg['start']:.2f}s - {seg['end']:.2f}s): "
                                    f"{seg['text']}"
                                )

                except Exception as e:
                    st.error(f"Video model failed: {str(e)}")

                finally:
                    if os.path.exists(video_path):
                        os.remove(video_path)

            # ---------- TEXT ----------
            if input_type == "Text" and uploaded_text and uploaded_text.strip():
                st.info("🔍 Predicting text...")

                try:
                    text_result = predict_text(uploaded_text, modality="text")
                    predictions.append(text_result)

                    result_card(
                        "Text",
                        text_result['label'],
                        text_result['confidence'],
                        text_result.get("explanation", "N/A")
                    )

                    # ───────── TEXT EXPLAINABILITY ─────────
                    st.markdown(
                        '<div class="section-label">TEXT EXPLAINABILITY</div>',
                        unsafe_allow_html=True
                    )

                    explanation = text_result.get("explanation", [])

                    if explanation:
                        for word, score in explanation:
                            st.markdown(f"- **{word}** : {score:.3f}")
                    else:
                        st.info("No explainability available.")

                except Exception as e:
                    st.error(f"Text model failed: {str(e)}")

            # ---------- FUSION ----------
            if len(predictions) >= 2:
                st.info("🔗 Performing fusion across modalities...")
                fusion_result = fuse_predictions(predictions)

                fl = fusion_result['final_label']

                if fl == "fake":
                    fcolor = "#ff2d55"
                    fclass = "verdict-fake"
                    st.markdown("<script>playTone('warning');</script>", unsafe_allow_html=True)

                elif fl == "real":
                    fcolor = "#00ff88"
                    fclass = "verdict-real"
                    st.markdown("<script>playTone('success');</script>", unsafe_allow_html=True)
                    st.success("✅ Analysis complete")

                else:
                    fcolor = "#ffc400"
                    fclass = "verdict-uncertain"

                fs_pct = int(fusion_result['final_confidence'] * 100)

                import json

                st.markdown(f"""
                <div class="fusion-panel">
                    <div class="fusion-score-ring">
                        <div class="ring-num" style="color:{fcolor}; text-shadow:0 0 25px {fcolor}88;">
                            {fs_pct}%
                        </div>
                        <div class="ring-meta">
                            <div class="ring-meta-label">FUSION CONFIDENCE</div>
                            <div class="ring-meta-value {fclass}">
                                {format_label(fl)}
                            </div>
                        </div>
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-label">MODALITY BREAKDOWN</div>', unsafe_allow_html=True)

                for detail in fusion_result["details"]:
                    st.json(detail)

                st.success("✅ Analysis complete")

            else:
                st.warning("No valid input provided for prediction.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-line"></div>
  <div class="footer-text">Developed for Multimodal Fake Content Detection Research</div>
  <div class="footer-line"></div>
</div>
""", unsafe_allow_html=True)