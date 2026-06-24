# predict_image.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "D:/Ritu_Minor/models/resnet18_finetuned_balanced.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

def predict_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    fake_prob = probs[0].item()
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else probs[1].item()
    print(probs)
    return {
        "modality": "image",
        "label": label,
        "confidence": float(confidence),
        "explanation": "Visual artifacts and facial inconsistencies analyzed"
    }
