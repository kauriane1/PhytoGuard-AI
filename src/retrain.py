"""
Retraining pipeline for PhytoGuard AI (production).

Flow (same logic as Section 6 of the notebook):
1. Check the trigger: enough new images, or a manual request.
2. Retrain on the new images plus replay images from data/train.
3. Evaluate the current and the retrained model on the same held-out data/test images.
4. Keep the retrained model only if it is at least as accurate. Archive the old one.
5. Record every run in models/retrain_log.json.

What is retrained in production: the final classification layer, on top of the
frozen MobileNetV2 feature extractor. Features for data/train and data/test are
cached, so only new uploads need to pass through the network. This keeps
retraining fast and light enough for a small cloud instance. Full fine-tuning of
the top MobileNetV2 layers is done offline with TensorFlow (notebook, Section 6),
after which `python -m src.model --export` refreshes the serving files.
"""

import json
import os
import shutil
from datetime import datetime

import numpy as np

from src.config import (
    ARCHIVE_DIR, BASE_DIR, CLASS_NAMES_PATH, FEATURE_CACHE_PATH, HEAD_LEARNING_RATE,
    HEAD_PATH, HEAD_RETRAIN_STEPS, METRICS_PATH, RETRAIN_LOG_PATH, RETRAIN_THRESHOLD,
    TEST_DIR, TRAIN_DIR, UPLOAD_DIR, VALID_EXTENSIONS,
)
from src.model import load_class_names, load_head, save_head, softmax
from src.preprocessing import load_labelled_folder, preprocess_image_file


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def count_images(folder, class_names):
    """Number of valid labelled images waiting in a folder."""
    paths, _, _ = load_labelled_folder(folder, class_names)
    return len(paths)


def should_retrain(new_image_count, manual_trigger=False, threshold=RETRAIN_THRESHOLD):
    """Decide whether retraining should start, and explain why."""
    if manual_trigger:
        return True, "Manual retraining requested by the user."
    if new_image_count >= threshold:
        return True, f"{new_image_count} new images reached the threshold of {threshold}."
    return False, f"Only {new_image_count} new images. Retraining starts at {threshold}."


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def _key(path):
    return os.path.relpath(path, BASE_DIR).replace("\\", "/")


def load_feature_cache():
    if not os.path.exists(FEATURE_CACHE_PATH):
        return {}
    data = np.load(FEATURE_CACHE_PATH)
    return dict(zip(data["paths"].tolist(), data["features"]))


def save_feature_cache(cache):
    if not cache:
        return
    keys = sorted(cache)
    np.savez(FEATURE_CACHE_PATH, paths=np.array(keys), features=np.stack([cache[k] for k in keys]))


def get_features(paths, extractor, cache):
    """Return features for each image, using the cache and extracting only what is missing."""
    features = []
    for path in paths:
        key = _key(path)
        if key not in cache:
            cache[key] = extractor(preprocess_image_file(path))[0]
        features.append(cache[key])
    return np.stack(features).astype(np.float32)


# ---------------------------------------------------------------------------
# Training the classification head
# ---------------------------------------------------------------------------

def _loss_and_accuracy(features, labels, weights, bias):
    probs = softmax(features @ weights + bias)
    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-9))
    accuracy = float(np.mean(probs.argmax(axis=1) == labels))
    return float(loss), accuracy


def train_head(features, labels, weights, bias, val_features, val_labels,
               steps=HEAD_RETRAIN_STEPS, learning_rate=HEAD_LEARNING_RATE, l2=1e-2):
    """
    Continue training the softmax layer with Adam, starting from the current weights.

    The L2 term keeps the new weights close to the current ones, which protects
    what the model already knows (similar to replay). The best weights on the
    validation split are kept (early stopping).
    """
    w0, b0 = weights.copy(), bias.copy()
    w, b = weights.copy(), bias.copy()
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    one_hot = np.eye(w.shape[1], dtype=np.float32)[labels]
    best = (np.inf, w.copy(), b.copy(), 0)
    patience, bad_steps = 30, 0

    for step in range(1, steps + 1):
        probs = softmax(features @ w + b)
        grad_logits = (probs - one_hot) / len(labels)
        grad_w = features.T @ grad_logits + l2 * (w - w0)
        grad_b = grad_logits.sum(axis=0) + l2 * (b - b0)

        for param, grad, m, v in ((w, grad_w, m_w, v_w), (b, grad_b, m_b, v_b)):
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        val_loss, _ = _loss_and_accuracy(val_features, val_labels, w, b)
        if val_loss < best[0] - 1e-5:
            best = (val_loss, w.copy(), b.copy(), step)
            bad_steps = 0
        else:
            bad_steps += 1
            if bad_steps >= patience:
                break

    return best[1], best[2], best[3]


def _split(n, val_fraction=0.15, seed=42):
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_fraction))
    return order[n_val:], order[:n_val]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def retrain_model(extractor, new_data_dir=UPLOAD_DIR, replay_data_dir=TRAIN_DIR,
                  test_data_dir=TEST_DIR, steps=HEAD_RETRAIN_STEPS):
    """Retrain on new plus replay data and compare the result with the current model."""
    class_names = load_class_names(CLASS_NAMES_PATH)

    new_paths, new_labels, skipped = load_labelled_folder(new_data_dir, class_names)
    if not new_paths:
        raise ValueError("No new labelled images found, so there is nothing to retrain on.")
    replay_paths, replay_labels, _ = load_labelled_folder(replay_data_dir, class_names)
    test_paths, test_labels, _ = load_labelled_folder(test_data_dir, class_names)

    cache = load_feature_cache()
    x_new = get_features(new_paths, extractor, cache)
    x_replay = get_features(replay_paths, extractor, cache)
    x_test = get_features(test_paths, extractor, cache)
    save_feature_cache(cache)

    x_all = np.concatenate([x_new, x_replay])
    y_all = np.array(new_labels + replay_labels)
    y_test = np.array(test_labels)
    train_idx, val_idx = _split(len(y_all))

    weights, bias = load_head(HEAD_PATH)
    old_loss, old_accuracy = _loss_and_accuracy(x_test, y_test, weights, bias)

    new_weights, new_bias, best_step = train_head(
        x_all[train_idx], y_all[train_idx], weights, bias,
        x_all[val_idx], y_all[val_idx], steps=steps,
    )
    new_loss, new_accuracy = _loss_and_accuracy(x_test, y_test, new_weights, new_bias)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": "classification layer retrained on frozen MobileNetV2 features",
        "new_images": len(new_paths),
        "replay_images": len(replay_paths),
        "skipped_folders": skipped,
        "epochs_run": best_step,
        "evaluation_set": f"data/test ({len(test_paths)} images)",
        "old_test_loss": round(old_loss, 4),
        "old_test_accuracy": round(old_accuracy, 4),
        "new_test_loss": round(new_loss, 4),
        "new_test_accuracy": round(new_accuracy, 4),
    }
    return (new_weights, new_bias), results


def _read_log():
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH) as f:
            return json.load(f)
    return []


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def promote_if_better(candidate, results):
    """Replace the current model only if the retrained model is at least as accurate."""
    promoted = results["new_test_accuracy"] >= results["old_test_accuracy"]

    if promoted:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(HEAD_PATH, os.path.join(ARCHIVE_DIR, f"phytoguard_head_{stamp}.npz"))
        save_head(*candidate)

    results["promoted"] = promoted
    log = _read_log()
    log.append(results)
    _write_json(RETRAIN_LOG_PATH, log)

    if promoted:
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        metrics.update({
            "model_version": f"v{len(log) + 1}",
            "trained_on": results["timestamp"],
            "last_retrain_test_accuracy": results["new_test_accuracy"],
            "last_retrain_evaluation_set": results["evaluation_set"],
        })
        _write_json(METRICS_PATH, metrics)

    return promoted


def move_uploads_to_train(class_names, upload_dir=UPLOAD_DIR, train_dir=TRAIN_DIR):
    """After retraining, move used uploads into data/train so they become replay data.

    Only folders with a known class name are moved. Unknown folders stay in uploads.
    Cached features follow the files, so they are not extracted again.
    """
    moved = 0
    if not os.path.isdir(upload_dir):
        return moved

    cache = load_feature_cache()
    for class_name in os.listdir(upload_dir):
        source_dir = os.path.join(upload_dir, class_name)
        if not os.path.isdir(source_dir) or class_name not in class_names:
            continue
        target_dir = os.path.join(train_dir, class_name)
        os.makedirs(target_dir, exist_ok=True)
        for file_name in os.listdir(source_dir):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue
            source = os.path.join(source_dir, file_name)
            target = os.path.join(target_dir, file_name)
            if os.path.exists(target):
                stem, ext = os.path.splitext(file_name)
                target = os.path.join(target_dir, f"{stem}_{datetime.now().strftime('%H%M%S%f')}{ext}")
            shutil.move(source, target)
            if _key(source) in cache:
                cache[_key(target)] = cache.pop(_key(source))
            moved += 1
        if not os.listdir(source_dir):
            os.rmdir(source_dir)
    save_feature_cache(cache)
    return moved


def run_retraining(extractor=None, new_data_dir=UPLOAD_DIR, manual_trigger=False,
                   steps=HEAD_RETRAIN_STEPS):
    """Full pipeline: trigger check, retrain, compare, promote, and clean up uploads."""
    class_names = load_class_names(CLASS_NAMES_PATH)
    new_count = count_images(new_data_dir, class_names)

    start, reason = should_retrain(new_count, manual_trigger)
    if not start:
        return {"started": False, "reason": reason, "new_images": new_count}

    if extractor is None:
        from src.model import FeatureExtractor
        extractor = FeatureExtractor()

    candidate, results = retrain_model(extractor, new_data_dir=new_data_dir, steps=steps)
    promote_if_better(candidate, results)

    # Only real user uploads are consumed. The sample folder data/retrain is kept.
    if os.path.abspath(new_data_dir) == os.path.abspath(UPLOAD_DIR):
        results["uploads_moved_to_train"] = move_uploads_to_train(class_names)

    return {"started": True, "reason": reason, **results}


def build_feature_cache():
    """Pre-compute features for data/train and data/test (run once before deployment)."""
    from src.model import FeatureExtractor

    class_names = load_class_names(CLASS_NAMES_PATH)
    extractor = FeatureExtractor()
    cache = load_feature_cache()
    for folder in (TRAIN_DIR, TEST_DIR):
        paths, _, _ = load_labelled_folder(folder, class_names)
        get_features(paths, extractor, cache)
    save_feature_cache(cache)
    return len(cache)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain the PhytoGuard AI model.")
    parser.add_argument("--data", default=UPLOAD_DIR, help="Folder of new labelled images (class subfolders).")
    parser.add_argument("--manual", action="store_true", help="Retrain even if below the threshold.")
    parser.add_argument("--build-cache", action="store_true", help="Pre-compute features for data/train and data/test.")
    args = parser.parse_args()

    if args.build_cache:
        print("Cached features:", build_feature_cache())
    else:
        print(json.dumps(run_retraining(new_data_dir=args.data, manual_trigger=args.manual), indent=2))
