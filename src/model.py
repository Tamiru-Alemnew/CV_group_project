"""
Stage 2 - Module 1: CNN Architecture

Defines the MathSymbolCNN that classifies 32×32 character images into
one of 16 symbol classes (digits 0–9 + operators +, -, x, =, *, /).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import config

# Lazy TF import so the module can be imported without TF installed
try:
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
    )
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class MathSymbolCNN:
    """
    Three-block convolutional network for handwritten math symbol classification.

    Architecture rationale:
      • Filters grow 32 → 64 → 128 across blocks because early layers detect
        simple patterns (edges, curves) while deeper layers need more filters
        to encode the increasing number of abstract feature combinations.
      • padding='same' preserves spatial dimensions through each conv layer
        so MaxPooling controls the resolution reduction schedule.
      • MaxPooling halves spatial size at each block, progressively building
        translation invariance while reducing computation.
      • Dropout rates increase from 0.25 (conv) to 0.5 (dense) because dense
        layers have far more parameters and are much more prone to overfitting.
      • BatchNormalization after conv and before activation stabilises the
        distribution of layer inputs, allowing higher learning rates and
        reducing sensitivity to weight initialisation.
    """

    def __init__(self):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required for MathSymbolCNN. "
                              "Install with: pip install tensorflow")
        self.model: tf.keras.Model | None = None

    # ─────────────────────────────────────────────────────────────────────────

    def build_model(self) -> tf.keras.Model:
        """
        Construct and return the Sequential CNN model.

        Input shape:  (32, 32, 1)   — single-channel greyscale
        Output shape: (16,)         — softmax probabilities per class

        Returns:
            Compiled tf.keras.Sequential model.
        """
        w, h = config.IMAGE_SIZE

        model = tf.keras.Sequential(name='MathSymbolCNN')

        # ── Input ────────────────────────────────────────────────────────────
        model.add(layers.Input(shape=(h, w, 1)))

        # ── Block 1: Low-level features (edges, curves, stroke endpoints) ────
        # L2 regularisation penalises large weights, reducing overfitting
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        # BatchNorm before the second conv: normalises activations so the
        # following layer always receives a well-scaled distribution
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        # MaxPool 2×2: halves spatial size (32→16), building local invariance
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # ── Block 2: Mid-level features (character parts, stroke junctions) ──
        # 64 filters: double the first block to capture more complex patterns
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))   # 16→8
        model.add(layers.Dropout(0.25))

        # ── Block 3: High-level features (complete symbol shapes) ────────────
        # 128 filters: each filter can now respond to a distinct symbol-level
        # feature combination assembled from the lower block representations
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))   # 8→4
        model.add(layers.Dropout(0.25))

        # ── Classification head ───────────────────────────────────────────────
        # Flatten converts the 4×4×128 feature volume to a 2048-d vector
        model.add(layers.Flatten())

        # Dense 256: combines all spatial features into a global representation
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        # Higher dropout (0.5) here because dense layers have many more
        # parameters than conv layers and overfit more aggressively
        model.add(layers.Dropout(0.50))

        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.30))

        # Softmax output: 16 probabilities summing to 1.0
        model.add(layers.Dense(config.NUM_CLASSES, activation='softmax'))

        self.model = model
        return model

    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        """
        Compile the model with Adam + categorical cross-entropy.

        Why Adam: adaptive learning rate per-parameter, robust to sparse
        gradients and well-suited for classification tasks.
        Why categorical_crossentropy: our labels are one-hot vectors (hard
        targets), making cross-entropy the correct probabilistic loss.

        Args:
            learning_rate: Initial learning rate (reduced during training by
                           ReduceLROnPlateau callback).
        """
        if self.model is None:
            self.build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def get_callbacks(self, save_path: str = None) -> list:
        """
        Return the three training callbacks.

        1. EarlyStopping   – stops training when val_loss stops improving.
                             restore_best_weights=True keeps the best checkpoint.
        2. ReduceLROnPlateau – halves learning rate when val_loss plateaus,
                               preventing the optimizer from oscillating.
        3. ModelCheckpoint – saves the model file only when val_accuracy improves.
        4. CSVLogger       – appends one row per epoch to training_log.csv.

        Args:
            save_path: Path for the .h5 model checkpoint.
                       Defaults to config.MODEL_PATH.
        """
        if save_path is None:
            save_path = str(config.MODEL_PATH)

        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            CSVLogger(
                filename=str(config.TRAINING_LOG_PATH),
                append=True
            ),
        ]

    def summary(self):
        """Print the full model summary with layer shapes and parameter counts."""
        if self.model is None:
            self.build_model()
        self.model.summary()

    def plot_architecture(self, save_path: str = None):
        """
        Save a diagram of the model architecture.

        Requires pydot and graphviz to be installed. Gracefully skips if absent.

        Args:
            save_path: Path for the output PNG. Defaults to config.ARCH_DIAGRAM_IMG.
        """
        if save_path is None:
            save_path = str(config.ARCH_DIAGRAM_IMG)
        if self.model is None:
            self.build_model()
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                dpi=96
            )
            print(f"[Model] Architecture diagram saved → {save_path}")
        except Exception as e:
            print(f"[Model] plot_architecture skipped ({e}). "
                  "Install 'pydot' and 'graphviz' to enable.")
