"""
Retraining pipeline for PhytoGuard AI.

Flow (same as Section 6 of the notebook):
1. Check the trigger: enough new images, or a manual request.
2. Fine-tune the current model on the new images plus replay images from data/train.
3. Evaluate the current and the retrained model on the same held-out data/test images.
4. Keep the retrained model only if it is at least as accurate. Archive the old one.
5. Record every run in models/retrain_log.json.

In production the comparison uses data/test (380 images, 10 per class), because
the full TFDS test split used in the notebook is not shipped with the app.
"""

import json
import os
import shutil
from datetime import datetime

from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.config import (
    ARCHIVE_DIR, CLASS_NAMES_PATH, FINE_TUNE_LAYERS, METRICS_PATH, MODEL_PATH,
    RETRAIN_EPOCHS, RETRAIN_LEARNING_RATE, RETRAIN_LOG_PATH, RETRAIN_THRESHOLD,
    TEST_DIR, TRAIN_DIR, UPLOAD_DIR, VALID_EXTENSIONS,
)
from src.model import load_class_names, load_model, prepare_for_fine_tuning
from src.preprocessing import build_dataset, load_labelled_folder


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


def _read_log():
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH) as f:
            return json.load(f)
    return []


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def retrain_model(new_data_dir=UPLOAD_DIR, replay_data_dir=TRAIN_DIR, test_data_dir=TEST_DIR,
                  model_path=MODEL_PATH, epochs=RETRAIN_EPOCHS,
                  learning_rate=RETRAIN_LEARNING_RATE, fine_tune_layers=FINE_TUNE_LAYERS):
    """Fine-tune the saved model on new plus replay data and compare it with the current model."""
    class_names = load_class_names(CLASS_NAMES_PATH)

    new_paths, new_labels, skipped = load_labelled_folder(new_data_dir, class_names)
    if not new_paths:
        raise ValueError("No new labelled images found, so there is nothing to retrain on.")

    replay_paths, replay_labels, _ = load_labelled_folder(replay_data_dir, class_names)
    test_paths, test_labels, _ = load_labelled_folder(test_data_dir, class_names)
    test_ds = build_dataset(test_paths, test_labels)

    # 1. Baseline: the current model on the held-out test images
    current_model = load_model(model_path)
    old_loss, old_accuracy = current_model.evaluate(test_ds, verbose=0)
    del current_model

    # 2. Combine new and replay images, keep 15% aside for validation
    all_paths = new_paths + replay_paths
    all_labels = new_labels + replay_labels
    train_p, val_p, train_l, val_l = train_test_split(
        all_paths, all_labels, test_size=0.15, random_state=42
    )

    # 3. Fine-tune a fresh copy of the current model
    candidate = load_model(model_path)
    prepare_for_fine_tuning(candidate, fine_tune_layers, learning_rate)

    history = candidate.fit(
        build_dataset(train_p, train_l, training=True),
        validation_data=build_dataset(val_p, val_l),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )],
        verbose=2,
    )

    # 4. Evaluate the retrained model on the same test images
    new_loss, new_accuracy = candidate.evaluate(test_ds, verbose=0)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "new_images": len(new_paths),
        "replay_images": len(replay_paths),
        "skipped_folders": skipped,
        "epochs_run": len(history.history["loss"]),
        "evaluation_set": f"data/test ({len(test_paths)} images)",
        "old_test_loss": round(float(old_loss), 4),
        "old_test_accuracy": round(float(old_accuracy), 4),
        "new_test_loss": round(float(new_loss), 4),
        "new_test_accuracy": round(float(new_accuracy), 4),
    }
    return candidate, results


def promote_if_better(candidate, results, model_path=MODEL_PATH):
    """Replace the current model only if the retrained model is at least as accurate."""
    promoted = results["new_test_accuracy"] >= results["old_test_accuracy"]

    if promoted:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_name = f"phytoguard_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        shutil.copy(model_path, os.path.join(ARCHIVE_DIR, archive_name))
        candidate.save(model_path)

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
    """
    moved = 0
    if not os.path.isdir(upload_dir):
        return moved

    for class_name in os.listdir(upload_dir):
        source_dir = os.path.join(upload_dir, class_name)
        if not os.path.isdir(source_dir) or class_name not in class_names:
            continue
        target_dir = os.path.join(train_dir, class_name)
        os.makedirs(target_dir, exist_ok=True)
        for file_name in os.listdir(source_dir):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue
            target = os.path.join(target_dir, file_name)
            if os.path.exists(target):
                stem, ext = os.path.splitext(file_name)
                target = os.path.join(target_dir, f"{stem}_{datetime.now().strftime('%H%M%S%f')}{ext}")
            shutil.move(os.path.join(source_dir, file_name), target)
            moved += 1
        if not os.listdir(source_dir):
            os.rmdir(source_dir)
    return moved


def run_retraining(new_data_dir=UPLOAD_DIR, manual_trigger=False, epochs=RETRAIN_EPOCHS):
    """Full pipeline: trigger check, retrain, compare, promote, and clean up uploads."""
    class_names = load_class_names(CLASS_NAMES_PATH)
    new_count = count_images(new_data_dir, class_names)

    start, reason = should_retrain(new_count, manual_trigger)
    if not start:
        return {"started": False, "reason": reason, "new_images": new_count}

    candidate, results = retrain_model(new_data_dir=new_data_dir, epochs=epochs)
    promote_if_better(candidate, results)

    # Only real user uploads are consumed. The sample folder data/retrain is kept.
    if os.path.abspath(new_data_dir) == os.path.abspath(UPLOAD_DIR):
        results["uploads_moved_to_train"] = move_uploads_to_train(class_names)

    return {"started": True, "reason": reason, **results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain the PhytoGuard AI model.")
    parser.add_argument("--data", default=UPLOAD_DIR, help="Folder of new labelled images (class subfolders).")
    parser.add_argument("--manual", action="store_true", help="Retrain even if below the threshold.")
    parser.add_argument("--epochs", type=int, default=RETRAIN_EPOCHS)
    args = parser.parse_args()

    print(json.dumps(run_retraining(args.data, args.manual, args.epochs), indent=2))
