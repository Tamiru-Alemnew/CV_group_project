import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import config

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

    def __init__(self):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required. pip install tensorflow")
        self.model: tf.keras.Model | None = None

    def build_model(self) -> tf.keras.Model:
        w, h = config.IMAGE_SIZE

        model = tf.keras.Sequential(name='MathSymbolCNN_v3')
        model.add(layers.Input(shape=(h, w, 1)))

        # Block 1
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Block 2
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Block 3
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Block 4
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.25))

        # Head
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.50))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.30))
        model.add(layers.Dense(config.NUM_CLASSES, activation='softmax'))

        self.model = model
        return model

    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        if self.model is None:
            self.build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def get_callbacks(self, save_path: str = None) -> list:
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
        if self.model is None:
            self.build_model()
        self.model.summary()

    def plot_architecture(self, save_path: str = None):
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
            print(f"[Model] plot_architecture skipped ({e}).")
