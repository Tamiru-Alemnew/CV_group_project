"""
Stage 2 - Module 2: Model Training Pipeline

Loads prepared dataset arrays, trains the MathSymbolCNN, evaluates on
the held-out test set, and saves all training artefacts.
"""

import sys, os, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import config

# Set all seeds before importing TF for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(config.RANDOM_SEED)
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _IMPORT_ERR = str(_e)

from src.model import MathSymbolCNN


class ModelTrainer:
    """
    End-to-end training manager: data loading → training → evaluation →
    confusion matrix → prediction visualisation.
    """

    def __init__(self):
        if not _DEPS_OK:
            raise ImportError(f"Missing dependency: {_IMPORT_ERR}")
        self.cnn      = MathSymbolCNN()
        self.model    = None
        self.history  = None
        # Arrays populated by load_data()
        self.x_train = self.y_train = None
        self.x_val   = self.y_val   = None
        self.x_test  = self.y_test  = None
        self.y_train_raw = self.y_test_raw = None

    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """
        Load all numpy arrays from data/processed/ and one-hot encode labels.

        Why one-hot encoding: categorical_crossentropy expects integer targets
        as one-hot vectors (e.g., class 3 → [0,0,0,1,0,…]) rather than raw
        integer scalars. to_categorical performs this conversion.
        """
        pd_ = config.PROCESSED_DIR
        required = ['train_images', 'train_labels', 'val_images', 'val_labels',
                    'test_images', 'test_labels']
        for name in required:
            if not (pd_ / f'{name}.npy').exists():
                raise FileNotFoundError(
                    f"'{pd_}/{name}.npy' not found. "
                    "Run DataPreparator.prepare() first."
                )

        self.x_train     = np.load(pd_ / 'train_images.npy')
        self.y_train_raw = np.load(pd_ / 'train_labels.npy').astype(int)
        self.x_val       = np.load(pd_ / 'val_images.npy')
        y_val_raw        = np.load(pd_ / 'val_labels.npy').astype(int)
        self.x_test      = np.load(pd_ / 'test_images.npy')
        self.y_test_raw  = np.load(pd_ / 'test_labels.npy').astype(int)

        # One-hot encode
        self.y_train = to_categorical(self.y_train_raw, config.NUM_CLASSES)
        self.y_val   = to_categorical(y_val_raw,        config.NUM_CLASSES)
        self.y_test  = to_categorical(self.y_test_raw,  config.NUM_CLASSES)

        # Add channel dimension for conv layers: (N, H, W) → (N, H, W, 1)
        if self.x_train.ndim == 3:
            self.x_train = self.x_train[..., np.newaxis]
            self.x_val   = self.x_val[...,   np.newaxis]
            self.x_test  = self.x_test[...,  np.newaxis]

        print("[Trainer] Data loaded:")
        for name, arr in [('train', self.x_train), ('val', self.x_val),
                           ('test', self.x_test)]:
            print(f"  {name:5s}: images {arr.shape}  "
                  f"labels {getattr(self, f'y_{name}').shape}")

    def compute_class_weights(self) -> dict:
        """
        Compute per-class weights to handle class imbalance.

        Why: MNIST has ~6 000 samples per digit but operator classes may have
        far fewer images. Unweighted training would bias the model towards
        digits. Inverse-frequency weights upscale the loss contribution from
        under-represented classes, forcing the model to pay equal attention.
        """
        present = np.unique(self.y_train_raw)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=present,
            y=self.y_train_raw
        )
        cw = {int(c): float(w) for c, w in zip(present, weights)}
        print("[Trainer] Class weights (top 5 by weight):",
              sorted(cw.items(), key=lambda kv: -kv[1])[:5])
        return cw

    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> tf.keras.callbacks.History:
        """
        Build, compile, and train the CNN.

        Training augmentation is applied only to the training generator —
        validation and test data are evaluated on clean unmodified images to
        get an unbiased estimate of real-world performance.
        """
        if self.x_train is None:
            self.load_data()

        self.model = self.cnn.build_model()
        self.cnn.compile_model(config.LEARNING_RATE)
        self.cnn.summary()
        self.cnn.plot_architecture()

        class_weights = self.compute_class_weights()

        # Augmentation only for training — keeps val/test evaluation clean
        aug_cfg = {k: v for k, v in config.AUGMENTATION.items()
                   if k not in ('fill_mode', 'cval')}
        aug_cfg['fill_mode'] = config.AUGMENTATION['fill_mode']
        aug_cfg['cval']      = config.AUGMENTATION['cval']

        train_gen = ImageDataGenerator(**aug_cfg)
        val_gen   = ImageDataGenerator()   # No augmentation for validation

        train_flow = train_gen.flow(
            self.x_train, self.y_train,
            batch_size=config.BATCH_SIZE,
            seed=config.RANDOM_SEED
        )
        val_flow = val_gen.flow(
            self.x_val, self.y_val,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

        callbacks = self.cnn.get_callbacks(str(config.MODEL_PATH))

        print(f"\n[Trainer] Training for up to {config.EPOCHS} epochs …")
        t0 = time.time()
        self.history = self.cnn.model.fit(
            train_flow,
            epochs=config.EPOCHS,
            validation_data=val_flow,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        elapsed = time.time() - t0
        print(f"[Trainer] Training completed in {elapsed:.1f}s")

        # Also export in TF SavedModel format for production deployment
        try:
            self.cnn.model.export(str(config.SAVEDMODEL_PATH))
            print(f"[Trainer] SavedModel exported → {config.SAVEDMODEL_PATH}")
        except Exception as e:
            print(f"[Trainer] SavedModel export skipped ({e})")

        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisations
    # ─────────────────────────────────────────────────────────────────────────

    def plot_training_history(self, history=None,
                               save_path: str = None) -> plt.Figure:
        """
        4-panel figure: accuracy, loss, learning rate, overfit gap.

        Args:
            history:   keras History object. Defaults to self.history.
            save_path: Output PNG path. Defaults to config.TRAINING_HIST_IMG.
        """
        if history is None:
            history = self.history
        if history is None:
            raise RuntimeError("No training history. Run train() first.")
        if save_path is None:
            save_path = str(config.TRAINING_HIST_IMG)

        h   = history.history
        eps = range(1, len(h['accuracy']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History  |  MathSymbolCNN',
                     fontsize=14, fontweight='bold')

        # Panel 1 — Accuracy
        ax = axes[0, 0]
        ax.plot(eps, h['accuracy'],     label='Train',      color='steelblue')
        ax.plot(eps, h['val_accuracy'], label='Validation', color='darkorange')
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2 — Loss
        ax = axes[0, 1]
        ax.plot(eps, h['loss'],     label='Train',      color='steelblue')
        ax.plot(eps, h['val_loss'], label='Validation', color='darkorange')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Categorical Cross-Entropy')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 3 — Learning rate
        ax = axes[1, 0]
        if 'lr' in h:
            ax.semilogy(eps, h['lr'], color='green')
            ax.set_title('Learning Rate (log scale)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('LR')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'LR not logged', ha='center', va='center',
                    transform=ax.transAxes)

        # Panel 4 — Overfit gap
        ax = axes[1, 1]
        gap = [tr - vl for tr, vl in zip(h['accuracy'], h['val_accuracy'])]
        ax.plot(eps, gap, color='red')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.fill_between(eps, gap, 0, where=[g > 0 for g in gap],
                         alpha=0.2, color='red', label='Overfitting region')
        ax.set_title('Overfit Gap (Train − Val Accuracy)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gap')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Trainer] Training history plot → {save_path}")
        return fig

    def evaluate_model(self, model_path: str = None) -> dict:
        """
        Load best saved model, evaluate on test set, plot confusion matrix.

        Returns:
            Dict with 'loss', 'accuracy', 'report', 'confusion_matrix'.
        """
        if model_path is None:
            model_path = str(config.MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

        if self.x_test is None:
            self.load_data()

        best = tf.keras.models.load_model(model_path)
        loss, acc = best.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\n[Trainer] Test loss    : {loss:.4f}")
        print(f"[Trainer] Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")

        y_pred_probs = best.predict(self.x_test, verbose=0)
        y_pred       = np.argmax(y_pred_probs, axis=1)

        class_names = [config.CLASS_MAP[i] for i in range(config.NUM_CLASSES)]
        report = classification_report(self.y_test_raw, y_pred,
                                        target_names=class_names)
        print("\nClassification Report:\n", report)

        # Confusion matrix
        cm  = confusion_matrix(self.y_test_raw, y_pred)
        fig = plt.figure(figsize=(14, 11))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.3)
        plt.title('Confusion Matrix  |  Test Set', fontsize=14, fontweight='bold')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        fig.savefig(str(config.CONFUSION_MAT_IMG), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Trainer] Confusion matrix → {config.CONFUSION_MAT_IMG}")

        return {'loss': loss, 'accuracy': acc, 'report': report, 'cm': cm}

    def visualize_predictions(self, n: int = 25, model_path: str = None,
                               save_path: str = None) -> plt.Figure:
        """
        Show 25 random test samples in a 5×5 grid.

        Correct predictions are framed in green, incorrect in red.
        Each cell shows: image, true label, predicted label, confidence %.
        """
        if model_path is None:
            model_path = str(config.MODEL_PATH)
        if self.x_test is None:
            self.load_data()
        if save_path is None:
            save_path = str(config.PREDICTIONS_IMG)

        best         = tf.keras.models.load_model(model_path)
        y_pred_probs = best.predict(self.x_test, verbose=0)
        y_pred       = np.argmax(y_pred_probs, axis=1)
        confs        = np.max(y_pred_probs, axis=1)

        rng     = np.random.default_rng(config.RANDOM_SEED)
        indices = rng.choice(len(self.x_test), size=n, replace=False)

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        fig.suptitle('Prediction Samples  (Green=Correct  Red=Incorrect)',
                     fontsize=13, fontweight='bold')
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            ax      = axes[i]
            true_l  = int(self.y_test_raw[idx])
            pred_l  = int(y_pred[idx])
            conf    = float(confs[idx])
            correct = true_l == pred_l

            ax.imshow(self.x_test[idx, :, :, 0], cmap='gray')
            ax.set_title(
                f"T:{config.CLASS_MAP[true_l]}  "
                f"P:{config.CLASS_MAP[pred_l]}  "
                f"{conf*100:.0f}%",
                fontsize=7,
                color='green' if correct else 'red'
            )
            for spine in ax.spines.values():
                spine.set_edgecolor('green' if correct else 'red')
                spine.set_linewidth(2)
            ax.axis('off')

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Trainer] Prediction samples → {save_path}")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience runner
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self) -> None:
        """Train, plot history, evaluate, and visualise predictions."""
        self.train()
        self.plot_training_history()
        results = self.evaluate_model()
        self.visualize_predictions()
        print(f"\n[Trainer] Final test accuracy: {results['accuracy']*100:.2f}%")


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run_full_pipeline()
