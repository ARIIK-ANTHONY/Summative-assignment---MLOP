"""
preprocessing.py

Shared image preprocessing utilities for the African Wildlife classification
pipeline. Used by the training notebook, the prediction API, and the
retraining pipeline, so preprocessing is guaranteed to be identical at train
time and inference time.
"""
import os
import random
import shutil
from collections import Counter

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
# Kept small (rather than the notebook's 32) so retraining fits in memory-
# constrained deployments (e.g. Render's free 512MB tier) without changing
# the preprocessing itself - training throughput isn't a concern here since
# retraining runs a handful of epochs on a small uploaded batch, not full
# from-scratch training.
BATCH_SIZE = 8
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def load_and_preprocess_image(image_path_or_file):
    """
    Load a single image (path or file-like object) and preprocess it into
    the (1, 224, 224, 3) tensor MobileNetV2 expects. Used by the prediction
    API for single-image inference.
    """
    img = Image.open(image_path_or_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def build_data_generators(train_dir, test_dir=None, validation_split=0.2, seed=42):
    """
    Builds train/validation/(test) generators with augmentation applied only
    to the training subset. Returns (train_generator, val_generator,
    test_generator_or_None).
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        seed=seed,
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        seed=seed,
    )

    test_generator = None
    if test_dir is not None and os.path.isdir(test_dir):
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=CLASS_NAMES,
            shuffle=False,
        )

    return train_generator, val_generator, test_generator


def _majority_class(label_path):
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return None
    counts = Counter()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts[int(line.split()[0])] += 1
    return counts.most_common(1)[0][0] if counts else None


def convert_yolo_to_classification(yolo_root, out_train_dir, out_test_dir,
                                    class_names=CLASS_NAMES, test_split=0.15, seed=42):
    """
    Converts a YOLO-format detection dataset into a classification folder
    layout:
        out_train_dir/<class>/*.jpg
        out_test_dir/<class>/*.jpg

    Walks the entire yolo_root recursively looking for jpg files with a
    matching txt label file next to them, rather than assuming a fixed
    images/labels/train/val/test structure. This matters because the raw
    Kaggle download (kaggle.com/datasets/biancaferreira/african-wildlife)
    is NOT pre-split into train/test/val the way some mirrors are, so a
    train/test split is done here instead. Used both for the initial
    dataset setup and whenever bulk data is uploaded in YOLO format for
    retraining.
    """
    class_ids = {name: i for i, name in enumerate(class_names)}
    pairs_by_class = {name: [] for name in class_names}

    def find_label_path(img_path):
        # label file might sit right next to the image (co-located), or in
        # a parallel "labels/" tree mirroring "images/" - handle both
        same_dir_label = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(same_dir_label):
            return same_dir_label
        if (os.sep + "images" + os.sep) in img_path:
            alt = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            alt = os.path.splitext(alt)[0] + ".txt"
            if os.path.exists(alt):
                return alt
        return same_dir_label

    for root, _dirs, files in os.walk(yolo_root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, fname)
            label_path = find_label_path(img_path)
            cls_id = _majority_class(label_path)
            if cls_id is None or cls_id >= len(class_names):
                continue
            pairs_by_class[class_names[cls_id]].append(img_path)

    random.seed(seed)
    stats = Counter()
    for cls_name, paths in pairs_by_class.items():
        random.shuffle(paths)
        split_point = int(len(paths) * (1 - test_split))
        train_paths, test_paths = paths[:split_point], paths[split_point:]

        for out_dir, split_paths in [(out_train_dir, train_paths), (out_test_dir, test_paths)]:
            dest_dir = os.path.join(out_dir, cls_name)
            os.makedirs(dest_dir, exist_ok=True)
            for src_path in split_paths:
                shutil.copy2(src_path, dest_dir)
                stats[(out_dir, cls_name)] += 1

    return stats


def save_uploaded_images_for_retraining(files, class_label, upload_dir):
    """
    Saves a batch of newly uploaded images (e.g. from the retraining UI) into
    upload_dir/<class_label>/, so they can be folded into the next retraining
    run's training folder.

    `files` is a list of (filename, file_bytes_or_fileobj) tuples.
    """
    dest_dir = os.path.join(upload_dir, class_label)
    os.makedirs(dest_dir, exist_ok=True)
    saved_paths = []
    for fname, file_obj in files:
        dest_path = os.path.join(dest_dir, fname)
        if hasattr(file_obj, "read"):
            with open(dest_path, "wb") as out_f:
                out_f.write(file_obj.read())
        else:
            with open(dest_path, "wb") as out_f:
                out_f.write(file_obj)
        saved_paths.append(dest_path)
    return saved_paths
