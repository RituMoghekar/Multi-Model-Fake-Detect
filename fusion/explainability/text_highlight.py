import numpy as np

def highlight_text_importance(text, classifier, target_label):
    """
    FAST explainability:
    - Only 1 model call (base prediction)
    - No repeated inference per word
    """

    words = text.split()

    # ---------------- BASE SCORE ----------------
    base = classifier(
        text,
        ["factual statement", "misleading statement"],
        hypothesis_template="This statement is {}."
    )

    base_scores = dict(zip(base["labels"], base["scores"]))
    base_score = base_scores[target_label]

    # ---------------- WORD IMPORTANCE (FAST HEURISTIC) ----------------
    # Instead of recomputing model, assign contribution weights
    word_scores = []

    # Normalize word influence using simple heuristic
    # (longer words = slightly higher impact weight)
    total_len = sum(len(w) for w in words) + 1e-6

    for w in words:
        weight = len(w) / total_len

        # importance approximation
        score = base_score * weight

        word_scores.append((w, float(score)))

    # sort by importance
    top_k = min(10, max(3, len(words) // 3))
    return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_k]