from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

LABELS = ["factual statement", "misleading statement"]

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=-1
)

def predict_text(text, modality="text"):

    result = classifier(
        text,
        LABELS,
        hypothesis_template="This statement is {}."
    )

    scores = dict(zip(result["labels"], result["scores"]))

    factual_score = scores.get("factual statement", 0.0)
    misleading_score = scores.get("misleading statement", 0.0)

    if misleading_score > factual_score:
        label = "fake"
        confidence = misleading_score
    else:
        label = "real"
        confidence = factual_score

    confidence = round(confidence, 4)

    # handle uncertainty
    if confidence < 0.6:
        label = "uncertain"

    return {
        "modality": modality,
        "label": label,
        "confidence": confidence,
        "explanation": text
    }