def explain_audio_segments(segments, classifier, threshold=0.0):
    explanations = []

    for seg in segments:
        result = classifier(seg["text"], ["real speech", "fake speech"])

        label = result["labels"][0]
        score = result["scores"][0]

        if score >= threshold:
            explanations.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "label": label,
                "confidence": round(score, 4)
            })

    return explanations