"""
model.py

Model architecture, training, and retraining logic for the African Wildlife
classifier (MobileNetV2 transfer learning). Shared by the training notebook
and the /retrain API endpoint so both use an identical model definition.
"""
import os
import json

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocessing import CLASS_NAMES, build_data_generators

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.keras")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")


def build_model(num_classes=len(CLASS_NAMES), fine_tune_layers=0):
    """
    Builds the MobileNetV2 transfer-learning classifier.
    fine_tune_layers=0 -> base fully frozen (feature extraction only).
    fine_tune_layers=N -> unfreezes the last N layers of the base for
    fine-tuning (used during retraining continuation).
    """
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    if fine_tune_layers <= 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    lr = 1e-5 if fine_tune_layers > 0 else 1e-3
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(train_dir, test_dir=None, epochs=20, fine_tune_layers=0,
          checkpoint_path=BEST_MODEL_PATH, initial_model_path=None):
    """
    Trains (or retrains, if initial_model_path is given) the classifier.

    - train_dir: folder containing <class_name>/ subfolders of images
      (this is what gets updated when new bulk data is uploaded for
      retraining — the retraining trigger simply calls this function again
      pointing at the updated train_dir).
    - initial_model_path: if provided, loads these weights as the starting
      point instead of ImageNet weights — used for incremental retraining on
      newly uploaded data rather than training completely from scratch.

    Returns (model, history, class_indices).
    """
    train_gen, val_gen, test_gen = build_data_generators(train_dir, test_dir)

    if initial_model_path and os.path.exists(initial_model_path):
        model = load_model(initial_model_path)
    else:
        model = build_model(fine_tune_layers=fine_tune_layers)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy",
        save_best_only=True, verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, checkpoint],
    )

    model.save(FINAL_MODEL_PATH)
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    return model, history, train_gen.class_indices


def load_trained_model(path=FINAL_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run training first."
        )
    return load_model(path)


def get_class_indices(path=CLASS_INDEX_PATH):
    if not os.path.exists(path):
        return {name: i for i, name in enumerate(CLASS_NAMES)}
    with open(path) as f:
        return json.load(f)
