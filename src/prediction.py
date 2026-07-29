"""
prediction.py

Single-image prediction logic. Used by the Flask/FastAPI prediction
endpoint to turn an uploaded image into a class label + confidence score.
"""
import numpy as np

from src.preprocessing import load_and_preprocess_image
from src.model import load_trained_model, get_class_indices

_model_cache = {"model": None}

# Below this confidence, the top class is more likely a forced guess on an
# out-of-distribution image (e.g. not a buffalo/elephant/rhino/zebra at all)
# than a real match — chosen from observed confidence on known classes
# (94-99%) vs. off-domain images (~55-65%).
LOW_CONFIDENCE_THRESHOLD = 0.75


def get_model():
    """Lazily loads and caches the trained model (avoids reloading on every
    request)."""
    if _model_cache["model"] is None:
        _model_cache["model"] = load_trained_model()
    return _model_cache["model"]


def reload_model():
    """Forces the next get_model() call to reload from disk — call this
    right after a retraining run completes so the API serves the new
    weights without needing a restart."""
    _model_cache["model"] = None


def predict_image(image_path_or_file):
    """
    Runs inference on a single image.

    Returns:
        {
            "predicted_class": "elephant",
            "confidence": 0.94,
            "probabilities": {"buffalo": 0.01, "elephant": 0.94, ...}
        }
    """
    model = get_model()
    class_indices = get_class_indices()
    idx_to_class = {v: k for k, v in class_indices.items()}

    x = load_and_preprocess_image(image_path_or_file)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    predicted_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
    }
