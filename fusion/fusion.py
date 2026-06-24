# fusion.py
"""
Fusion module for combining predictions from multiple modalities:
- image
- video (frame+audio)
- text
- audio (MFCC based)

Supports:
- Dynamic modality-aware fusion
- Weight normalization only for active modalities
- Explainability (per modality contribution)
"""

from typing import List, Dict


# ---------------- HELPERS ----------------
def signed_score(label: str, confidence: float) -> float:
    """
    Convert label/confidence to signed score:
    - fake -> positive
    - real -> negative
    """
    return confidence if label == "fake" else -confidence


# ---------------- FUSION FUNCTION ----------------
def fuse_predictions(predictions: List[Dict], weights: Dict[str, float] = None) -> Dict:
    """
    Dynamic fusion:
    - If single modality → full weight (1.0)
    - If multiple modalities → normalize weights of only present modalities
    """

    # Default research-driven weights
    if weights is None:
        weights = {
            "image": 0.20,
            "video": 0.35,
            "text": 0.15,
            "audio": 0.30
        }

    # Get active modalities
    active_modalities = [p.get("modality") for p in predictions if p.get("modality") in weights]

    # If only one modality → direct decision (100% weight)
    if len(active_modalities) == 1:
        p = predictions[0]
        label = p.get("label", "real")
        confidence = p.get("confidence", 0.0)

        return {
            "final_label": label,
            "fusion_score": round(confidence if label == "fake" else -confidence, 4),
            "details": [{
                "modality": p["modality"],
                "label": label,
                "confidence": round(confidence, 4),
                "weight": 1.0,
                "contribution": round(confidence if label == "fake" else -confidence, 4)
            }]
        }

    # ---------------- MULTI-MODAL FUSION ----------------
    # Normalize weights only among active modalities
    total_active_weight = sum(weights[m] for m in active_modalities)

    fusion_score = 0.0
    explanation = []

    for p in predictions:
        modality = p.get("modality")
        label = p.get("label", "real")
        confidence = p.get("confidence", 0.0)

        if modality not in weights:
            continue  # skip unknown modality

        normalized_weight = weights[modality] / total_active_weight
        contrib = normalized_weight * signed_score(label, confidence)
        fusion_score += contrib

        explanation.append({
            "modality": modality,
            "label": label,
            "confidence": round(confidence, 4),
            "weight": round(normalized_weight, 4),
            "contribution": round(contrib, 4)
        })

    final_label = "fake" if fusion_score > 0 else "real"
    final_confidence = abs(fusion_score)

    return {
        "final_label": final_label,
        "fusion_score": round(fusion_score, 4),
        "final_confidence": round(final_confidence, 4),
        "details": explanation
    }


# ---------------- TEST ----------------
if __name__ == "__main__":
    # Example: multi-modal (image + audio + text)
    preds = [
        {"modality": "image", "label": "fake", "confidence": 0.92},
        {"modality": "audio", "label": "fake", "confidence": 0.81},
        {"modality": "text",  "label": "real", "confidence": 0.65},
    ]

    result = fuse_predictions(preds)
    print("\n===== FUSION RESULT =====")
    print(result)

    # Example: single modality (audio only)
    preds2 = [
        {"modality": "audio", "label": "fake", "confidence": 0.96},
    ]

    result2 = fuse_predictions(preds2)
    print("\n===== SINGLE MODALITY RESULT =====")
    print(result2)
