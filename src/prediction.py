"""
Single-image prediction for PhytoGuard AI.

The Predictor keeps the feature extractor and the classification head in memory,
so each request only preprocesses the image and runs one forward pass. It can
reload the head after retraining without restarting the API.
"""

import threading

import numpy as np

from src.config import CLASS_NAMES_PATH, FEATURE_EXTRACTOR_PATH, HEAD_PATH, IMG_SIZE, NUM_THREADS, TOP_K
from src.model import FeatureExtractor, load_class_names, load_head, softmax
from src.preprocessing import preprocess_image_bytes, preprocess_image_file


def format_class_name(class_name):
    """Turn 'Tomato___Early_blight' into crop, condition and a readable label."""
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip()
    return {
        "crop": crop,
        "condition": condition,
        "is_healthy": condition.lower() == "healthy",
        "label": f"{crop}: {condition}",
    }


class Predictor:
    def __init__(self, extractor_path=FEATURE_EXTRACTOR_PATH, head_path=HEAD_PATH,
                 class_names_path=CLASS_NAMES_PATH):
        self.head_path = head_path
        self.class_names_path = class_names_path
        self._lock = threading.Lock()
        self.extractor = FeatureExtractor(extractor_path, num_threads=NUM_THREADS)
        self.reload()

    @property
    def model(self):
        return self.weights

    def reload(self):
        """Load (or reload) the classification head and class names from disk."""
        weights, bias = load_head(self.head_path)
        class_names = load_class_names(self.class_names_path)
        with self._lock:
            self.weights, self.bias, self.class_names = weights, bias, class_names
            # Warm-up so the first real request is not slow
            self.extractor(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))

    def extract_features(self, image_batch):
        with self._lock:
            return self.extractor(image_batch)

    def _predict_batch(self, image_batch, top_k):
        with self._lock:
            features = self.extractor(image_batch)
            probabilities = softmax(features @ self.weights + self.bias)[0]
            class_names = self.class_names

        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        for index in top_indices:
            name = class_names[index]
            top_predictions.append({
                "class": name,
                **format_class_name(name),
                "confidence": round(float(probabilities[index]) * 100, 2),
            })

        best = top_predictions[0]
        return {
            "predicted_class": best["class"],
            "label": best["label"],
            "crop": best["crop"],
            "condition": best["condition"],
            "is_healthy": best["is_healthy"],
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
        }

    def predict_bytes(self, image_bytes, top_k=TOP_K):
        """Predict from raw uploaded image bytes (used by the API)."""
        return self._predict_batch(preprocess_image_bytes(image_bytes), top_k)

    def predict_file(self, image_path, top_k=TOP_K):
        """Predict from an image file on disk."""
        return self._predict_batch(preprocess_image_file(image_path), top_k)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.prediction path/to/leaf_image.jpg")
        sys.exit(1)

    result = Predictor().predict_file(sys.argv[1])
    print(f"Prediction: {result['label']} ({result['confidence']:.2f}%)")
    print("Top predictions:")
    for rank, item in enumerate(result["top_predictions"], start=1):
        print(f"  {rank}. {item['label']} - {item['confidence']:.2f}%")
