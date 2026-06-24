# predict_image.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from explainability.gradcam import generate_gradcam, overlay_cam

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\Ritu_Minor\models\resnet18_finetuned_balanced.pth"
IMG_SIZE = 224

# ---------------- MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION ----------------
def predict_image(image_path):
    """
    Predict image (real/fake) and generate Grad-CAM overlay.
    Returns dictionary with:
        - label
        - confidence
        - explanation (path to overlay)
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    label = "fake" if pred_class == 0 else "real"

    # ✅ Grad-CAM
    model.zero_grad()

    with torch.enable_grad():
       target_layer = model.layer4[-1]
       cam = generate_gradcam(
           model,
           img_tensor,
           target_layer
       )
    cam_overlay = overlay_cam(image_path, cam)

    return {
        "modality": "image",
        "label": label,
        "confidence": confidence,
        "explanation": cam_overlay
    }