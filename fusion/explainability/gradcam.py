import cv2
import numpy as np
import torch
from torchvision import transforms

# ---------------- GRADCAM CORE ----------------
def generate_gradcam(model, image_tensor, target_layer):
    """
    Generates Grad-CAM heatmap
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].detach().cpu().numpy()
    acts = activations[0].detach().cpu().numpy()

    weights = np.mean(grads, axis=(2, 3))
    if weights.ndim == 2:
        weights = weights[0]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[0, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    return cam


# ---------------- OVERLAY FUNCTION ----------------
def overlay_cam(image_path, cam, alpha=0.5):
    """
    Overlays Grad-CAM heatmap on original image
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay