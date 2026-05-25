"""
Stage 2 - Module 3: Grad-CAM Visualisation

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to reveal
WHICH pixels of a character image the CNN attends to when making its
classification decision. This proves the model learned meaningful visual
features (strokes, loops, endpoints) rather than spurious correlations.

Works with both Sequential CNN models and Functional MobileNetV2 models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import config

try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:
    _TF_OK = False


class GradCAM:
    """
    Grad-CAM: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization."

    Supports both:
      • Sequential CNN   — layer-by-layer forward pass interception.
      • Functional models (MobileNetV2) — dual-output gradient model.

    Interpretation:
      • Red regions  → high attention (decisive pixels for this class).
      • Blue regions → low attention (model ignores these pixels).
    """

    def __init__(self, model):
        """
        Args:
            model: Loaded tf.keras.Model.

        Does NOT raise on failure — sets self.enabled = False instead so the
        pipeline continues without Grad-CAM rather than refusing to load.
        """
        if not _TF_OK:
            raise ImportError("TensorFlow required for GradCAM.")

        self.model          = model
        self.last_conv_name = None
        self._grad_model    = None   # dual-output model (preferred path)
        self.enabled        = False

        conv_layer = self._find_last_conv(model)
        if conv_layer is None:
            print("[GradCAM] WARNING: No Conv2D found — heatmaps disabled.")
            return

        self.last_conv_name = conv_layer.name
        print(f"[GradCAM] Target layer: '{self.last_conv_name}'")

        # Try to build a dual-output gradient model.
        # This works for both Sequential CNN (conv_layer is top-level) and
        # Functional MobileNetV2 (conv_layer is inside backbone sub-model).
        try:
            self._grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output],
                name='gradcam_model'
            )
            self.enabled = True
            print(f"[GradCAM] Dual-output model ready.")
        except Exception as e:
            # Fallback: layer-by-layer mode (Sequential only)
            print(f"[GradCAM] Dual-output model failed ({e}). Using layer-by-layer mode.")
            self.enabled = True   # still enabled, uses fallback path

    # ─────────────────────────────────────────────────────────────────────────
    # Layer discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _find_last_conv(self, model) -> object:
        """
        Recursively search for the last Conv2D layer, including inside nested
        sub-models (e.g. the MobileNetV2 backbone).

        Returns the layer object, or None if not found.
        """
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
            if hasattr(layer, 'layers'):           # sub-model
                result = self._find_last_conv(layer)
                if result is not None:
                    return result
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Core computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_gradcam(self, image: np.ndarray,
                        class_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Grad-CAM heatmap and overlay for one image.

        Algorithm (Selvaraju et al.):
          1. Forward pass through a dual-output model that returns both the
             last-conv feature maps AND the final class scores simultaneously.
          2. Compute gradients of the target class score w.r.t. those feature maps
             using tf.GradientTape (both tensors live inside the tape context so
             no explicit tape.watch() is needed).
          3. Global-average-pool the gradients → one importance weight per channel.
          4. Linearly combine channels with their weights → raw heatmap.
          5. ReLU → keep only regions that increase this class's score.
          6. Resize to 32×32, normalise, apply COLORMAP_JET.
          7. Blend 60 % original + 40 % heatmap → interpretable overlay.

        Args:
            image:     Float32 (H,W), (H,W,1), or (1,H,W,1).
            class_idx: Target class index to explain.

        Returns:
            heatmap_colored: (H,W,3) uint8 BGR heatmap.
            overlay:         (H,W,3) uint8 BGR blended image.

        Falls back to a blank overlay if Grad-CAM is disabled or fails.
        """
        # ── Prepare input tensor (1, H, W, 1) ────────────────────────────────
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, :, :, np.newaxis]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img[np.newaxis]
        elif img.ndim == 3:
            img = img[np.newaxis, :, :, :1]

        h_img, w_img = img.shape[1], img.shape[2]
        img_tensor   = tf.cast(img, tf.float32)

        # ── Fallback: blank overlay when disabled ─────────────────────────────
        if not self.enabled or self.last_conv_name is None:
            blank = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            return blank, blank

        # ── Path A: dual-output gradient model (CNN + MobileNetV2) ───────────
        if self._grad_model is not None:
            try:
                return self._gradcam_via_model(img_tensor, class_idx, img)
            except Exception as e:
                print(f"[GradCAM] Dual-model path failed ({e}), trying fallback.")

        # ── Path B: layer-by-layer (Sequential CNN only) ──────────────────────
        try:
            return self._gradcam_layerwise(img_tensor, class_idx, img)
        except Exception as e:
            print(f"[GradCAM] Layer-wise path failed ({e}). Returning blank.")
            blank = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            return blank, blank

    def _gradcam_via_model(self, img_tensor, class_idx, img_np):
        """Grad-CAM using the pre-built dual-output gradient model."""
        with tf.GradientTape() as tape:
            # Both tensors are computed inside the tape — no explicit watch needed.
            conv_outputs, predictions = self._grad_model(img_tensor, training=False)
            class_score = predictions[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)
        return self._build_overlay(conv_outputs, grads, img_np)

    def _gradcam_layerwise(self, img_tensor, class_idx, img_np):
        """Grad-CAM by running layers one-by-one (Sequential CNN only)."""
        with tf.GradientTape() as tape:
            x = img_tensor
            conv_outputs = None
            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.last_conv_name:
                    conv_outputs = x
                    tape.watch(conv_outputs)
            predictions = x
            class_score = predictions[:, class_idx]

        if conv_outputs is None:
            raise RuntimeError("Target conv layer not reached during forward pass.")
        grads = tape.gradient(class_score, conv_outputs)
        return self._build_overlay(conv_outputs, grads, img_np)

    def _build_overlay(self, conv_outputs, grads, img_np):
        """Convert (conv_outputs, grads) → (heatmap_colored BGR, overlay BGR)."""
        # Global-average-pool gradients: (1, H_f, W_f, C) → (C,)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted sum of feature maps → scalar heatmap (H_f, W_f)
        conv_out = conv_outputs[0]                          # (H_f, W_f, C)
        heatmap  = conv_out @ pooled[..., tf.newaxis]       # (H_f, W_f, 1)
        heatmap  = tf.squeeze(heatmap)                      # (H_f, W_f)

        # ReLU — keep only positive class influence
        heatmap_np = tf.maximum(heatmap, 0).numpy()
        if heatmap_np.ndim == 0:
            heatmap_np = np.zeros((4, 4), dtype=np.float32)

        mx = heatmap_np.max()
        if mx > 0:
            heatmap_np = heatmap_np / mx
        heatmap_uint8 = np.uint8(255 * heatmap_np)

        h_img, w_img = img_np.shape[1], img_np.shape[2]
        heatmap_resized = cv2.resize(heatmap_uint8, (w_img, h_img))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        orig_grey = np.uint8(img_np[0, :, :, 0] * 255)
        orig_bgr  = cv2.cvtColor(orig_grey, cv2.COLOR_GRAY2BGR)
        overlay   = cv2.addWeighted(orig_bgr, 0.60, heatmap_colored, 0.40, 0)

        return heatmap_colored, overlay

    # ─────────────────────────────────────────────────────────────────────────
    # Batch visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_gradcam(self,
                          images: list | np.ndarray,
                          labels: list | np.ndarray,
                          save_path: str = None,
                          n_samples: int = 16,
                          model_name: str = '') -> plt.Figure | None:
        """
        Create a grid: original image | Grad-CAM overlay, one row per sample.

        Picks one representative sample per class where possible so every
        symbol appears in the visualisation.

        Args:
            images:     Float32 character images (H,W) or (H,W,1).
            labels:     Integer class labels.
            save_path:  Output PNG path. Defaults to config.GRADCAM_IMG.
            n_samples:  Maximum rows to show.
            model_name: Shown in the figure title.

        Returns:
            matplotlib Figure, or None if Grad-CAM is disabled.
        """
        if not self.enabled:
            print("[GradCAM] Disabled — skipping visualisation.")
            return None

        if save_path is None:
            save_path = str(config.GRADCAM_IMG)

        images = np.array(images)
        labels = np.array(labels)

        # Try to pick one sample per class for a balanced view
        selected = []
        for cls in range(config.NUM_CLASSES):
            idx = np.where(labels == cls)[0]
            if len(idx) > 0:
                selected.append(idx[0])
            if len(selected) >= n_samples:
                break
        # Pad with random samples if fewer than n_samples classes found
        while len(selected) < min(n_samples, len(images)):
            selected.append(np.random.randint(0, len(images)))
        selected = selected[:n_samples]

        n   = len(selected)
        fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.2))
        title = f'Grad-CAM  —  {model_name + "  " if model_name else ""}Red=High Attention  Blue=Low'
        fig.suptitle(title, fontsize=11, fontweight='bold')

        if n == 1:
            axes = [axes]

        for row, idx in enumerate(selected):
            img      = np.array(images[idx], dtype=np.float32)
            true_cls = int(labels[idx])

            img_4d = img[np.newaxis, :, :, np.newaxis] if img.ndim == 2 else (
                     img[np.newaxis] if img.ndim == 3 else img)
            preds    = self.model.predict(img_4d, verbose=0)[0]
            pred_cls = int(np.argmax(preds))
            conf     = float(preds[pred_cls])

            _, overlay = self.compute_gradcam(img, pred_cls)

            orig_show = img[:, :, 0] if img.ndim == 3 else img
            axes[row][0].imshow(orig_show, cmap='gray', vmin=0, vmax=1)
            axes[row][0].set_title(f"True: {config.CLASS_MAP.get(true_cls,'?')}", fontsize=8)
            axes[row][0].axis('off')

            axes[row][1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            correct = '✓' if pred_cls == true_cls else '✗'
            axes[row][1].set_title(
                f"Pred: {config.CLASS_MAP.get(pred_cls,'?')}  {conf*100:.0f}%  {correct}",
                fontsize=8
            )
            axes[row][1].axis('off')

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[GradCAM] Saved → {save_path}")
        return fig
