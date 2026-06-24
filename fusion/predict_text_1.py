# predict_text_1.py

from transformers import pipeline
from explainability.text_highlight import highlight_text_importance


# ---------------- CONFIG ----------------
MODEL_NAME = "facebook/bart-large-mnli"

LABELS = [
    "factual statement",
    "misleading statement"
]


# ---------------- LOAD MODEL ----------------
print("🧠 Loading text model...")

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=-1
)

print("✅ Text model loaded\n")


# ---------------- PREDICTION ----------------
def predict_text(text, modality="text", top_n=5):
    """
    Predict real/fake text.
    Also provides important word explainability.
    """

    # Original prediction logic preserved
    result = classifier(
        text,
        LABELS,
        hypothesis_template="This statement is {}."
    )

    scores = dict(
        zip(
            result["labels"],
            result["scores"]
        )
    )


    factual_score = scores.get(
        "factual statement",
        0.0
    )

    misleading_score = scores.get(
        "misleading statement",
        0.0
    )


    if misleading_score > factual_score:
        label = "fake"
        confidence = misleading_score
        target_label = "misleading statement"

    else:
        label = "real"
        confidence = factual_score
        target_label = "factual statement"


    # ---------------- EXPLAINABILITY ----------------
    explanation = highlight_text_importance(
        text,
        classifier,
        target_label
    )[:top_n]


    return {
        "modality": modality,
        "label": label,
        "confidence": float(confidence),
        "explanation": explanation
    }


# ---------------- TEST ----------------
if __name__ == "__main__":

    sample = """
    The earth revolves around the sun.
    """

    print(
        predict_text(sample)
    )