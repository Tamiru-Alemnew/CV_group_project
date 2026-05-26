import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import config

try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:
    _TF_OK = False


class SymbolRecognizer:

    def __init__(self, model_path: str = None,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        if not _TF_OK:
            raise ImportError("TensorFlow required for SymbolRecognizer.")

        self.threshold  = confidence_threshold
        self.model_path = model_path or str(config.MODEL_PATH)
        self.model: tf.keras.Model | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Train the model first with:  python main.py --train"
            )
        print(f"[Recognizer] Loading model from {self.model_path} …", end=' ')
        try:
            self.model = tf.keras.models.load_model(self.model_path, safe_mode=False)
        except Exception:
            # Keras 3.x can't infer Lambda output shape on deserialisation —
            # rebuild architecture from the h5 config and load weights manually
            self.model = self._rebuild_mobilenet(self.model_path)
        print("done.")

        w, h = config.IMAGE_SIZE
        dummy = np.zeros((1, h, w, 1), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("[Recognizer] Warmup complete. Ready for inference.")

    def _rebuild_mobilenet(self, model_path: str) -> 'tf.keras.Model':
        """Load a Keras 2.x MobileNetV2 h5 that fails normal load_model in Keras 3.x."""
        import h5py, json

        with h5py.File(model_path, 'r') as f:
            cfg_raw = f.attrs.get('model_config', None)
        if cfg_raw is None:
            raise ValueError("No model_config in h5 file.")
        cfg_str = cfg_raw.decode('utf-8') if isinstance(cfg_raw, bytes) else str(cfg_raw)
        cfg = json.loads(cfg_str)

        def _patch_lambda(node):
            if isinstance(node, dict):
                if node.get('class_name') == 'Lambda':
                    node.get('config', {}).setdefault('output_shape', [None, 32, 32, 3])
                for v in node.values():
                    _patch_lambda(v)
            elif isinstance(node, list):
                for v in node:
                    _patch_lambda(v)

        _patch_lambda(cfg)
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.model_from_json(json.dumps(cfg))

        def _find_layer(m, name):
            for lyr in m.layers:
                if lyr.name == name:
                    return lyr
                if hasattr(lyr, 'layers'):
                    found = _find_layer(lyr, name)
                    if found is not None:
                        return found
            return None

        # Keras 2.x h5 stores backbone weights several group levels deep;
        # load_weights(by_name=True) in Keras 3.x only looks one level deep.
        def _load_group(group):
            layer_names = [n.decode() if isinstance(n, bytes) else n
                           for n in group.attrs.get('layer_names', [])]
            for name in layer_names:
                if name not in group:
                    continue
                sub = group[name]
                w_names = [n.decode() if isinstance(n, bytes) else n
                           for n in sub.attrs.get('weight_names', [])]
                if w_names:
                    lyr = _find_layer(model, name)
                    if lyr is not None and lyr.weights:
                        try:
                            lyr.set_weights([np.array(sub[wn]) for wn in w_names
                                             if wn in sub])
                        except Exception:
                            pass
                else:
                    _load_group(sub)

        with h5py.File(model_path, 'r') as f:
            if 'model_weights' in f:
                _load_group(f['model_weights'])

        return model

    def prepare_image(self, char_image: np.ndarray) -> np.ndarray:
        w, h = config.IMAGE_SIZE
        img  = np.array(char_image, dtype=np.float32)

        if img.ndim == 2:
            pass
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.ndim == 4:
            img = img[0, :, :, 0]

        import cv2
        if img.max() > 1.0:
            img = img / 255.0

        if img.shape != (h, w):
            img = cv2.resize(img.astype(np.float32), (w, h))

        return img[np.newaxis, :, :, np.newaxis]

    def recognize_single(self, char_image: np.ndarray) -> dict:
        inp         = self.prepare_image(char_image)
        probs       = self.model.predict(inp, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1]

        pred_idx   = int(top_indices[0])
        confidence = float(probs[pred_idx])

        top3 = [
            {'symbol': config.CLASS_MAP[int(top_indices[j])],
             'confidence': float(probs[top_indices[j]])}
            for j in range(3)
        ]

        return {
            'symbol':      config.CLASS_MAP[pred_idx],
            'confidence':  confidence,
            'is_reliable': confidence >= self.threshold,
            'top3':        top3,
            'raw_probs':   probs.tolist(),
        }

    def recognize_sequence(self, characters: list[dict]) -> list[dict]:
        results = []
        for ch in characters:
            rec = self.recognize_single(ch['image'])
            results.append({**ch, **rec})
        return results

    def batch_recognize(self, image_list: list[np.ndarray]) -> list[dict]:
        if not image_list:
            return []

        w, h = config.IMAGE_SIZE
        batch = np.stack([
            self.prepare_image(img)[0] for img in image_list
        ], axis=0)

        t0         = time.time()
        all_probs  = self.model.predict(batch, verbose=0)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[Recognizer] Batch {len(image_list)} images in {elapsed_ms:.1f} ms")

        results = []
        for probs in all_probs:
            top_indices = np.argsort(probs)[::-1]
            pred_idx    = int(top_indices[0])
            conf        = float(probs[pred_idx])
            results.append({
                'symbol':      config.CLASS_MAP[pred_idx],
                'confidence':  conf,
                'is_reliable': conf >= self.threshold,
                'top3':        [{'symbol': config.CLASS_MAP[int(top_indices[j])],
                                  'confidence': float(probs[top_indices[j]])}
                                 for j in range(3)],
                'raw_probs':   probs.tolist(),
            })
        return results

    def get_confidence_report(self, results: list[dict]) -> float:
        print("\n" + "─" * 52)
        print(f"{'#':>3}  {'Symbol':>8}  {'Confidence':>12}  {'Reliable':>8}")
        print("─" * 52)

        confs = []
        for r in results:
            idx   = r.get('index', '?')
            sym   = r.get('symbol', '?')
            conf  = r.get('confidence', 0.0)
            rel   = r.get('is_reliable', False)
            flag  = '' if rel else '  ⚠ LOW'
            print(f"  {str(idx):>3}  {sym:>8}  {conf*100:>10.1f}%  "
                  f"{'Yes' if rel else 'No':>8}{flag}")
            confs.append(conf)

        print("─" * 52)
        overall = float(np.prod(confs) ** (1.0 / len(confs))) if confs else 0.0
        print(f"  Overall equation confidence: {overall*100:.1f}%\n")

        if overall < 0.60:
            print("  ⚠  Low confidence — retake with better lighting.")

        return overall

    def context_correction(self, results: list[dict]) -> list[dict]:
        corrected = list(results)
        n = len(corrected)
        MATH_OPS = {'+', '-', '*', '/'}

        def _sym(idx):
            if 0 <= idx < n:
                return corrected[idx].get('symbol', '')
            return ''

        def _fix(idx, new_sym, reason):
            corrected[idx] = {**corrected[idx], 'symbol': new_sym,
                               'corrected': True, 'correction_reason': reason}

        # Keep only the most-confident '=' if there are multiple
        eq_indices = [i for i, r in enumerate(corrected) if r.get('symbol') == '=']
        if len(eq_indices) > 1:
            best_eq = max(eq_indices, key=lambda i: corrected[i].get('confidence', 0.0))
            for i in eq_indices:
                if i == best_eq:
                    continue
                alt = '+'
                for pred in corrected[i].get('top3', [])[1:]:
                    candidate = pred.get('symbol', '')
                    if candidate != '=' and candidate:
                        alt = candidate
                        break
                _fix(i, alt, f'duplicate = → {alt}')

        for i, r in enumerate(corrected):
            sym  = r.get('symbol', '')
            prev = _sym(i - 1)
            nxt  = _sym(i + 1)

            if sym == 'x' and prev in config.DIGIT_SYMBOLS and nxt in config.DIGIT_SYMBOLS:
                _fix(i, '*', 'x between two digits → multiply')

            elif sym == '*' and nxt == '=' and prev in config.DIGIT_SYMBOLS:
                _fix(i, '+', '* before = → likely +')

            elif sym == '*' and i == n - 1:
                _fix(i, '+', '* at end → likely +')

            elif sym == '/':
                digit_or_var = config.DIGIT_SYMBOLS | {'x'}
                if not (prev in digit_or_var and nxt in digit_or_var):
                    _fix(i, '1', '/ not flanked by digits → likely 1')

            elif (sym in MATH_OPS and prev in MATH_OPS
                    and not (sym == '*' and prev == '*')):
                if prev != '=':
                    _fix(i, '+', f'consecutive operators → likely +')

            elif sym == '=' and i == 0:
                _fix(i, '+', '= as first symbol → likely +')

            elif (i == n - 1
                    and r.get('confidence', 1.0) < 0.70
                    and sym in config.DIGIT_SYMBOLS
                    and any(_sym(j) == '=' for j in range(n))):
                eq_pos = next(j for j in range(n) if _sym(j) == '=')
                if i > eq_pos:
                    _fix(i, 'x', 'low-conf digit after = → likely unknown variable')

        # Drop operators sitting immediately before '='
        cleaned: list = []
        for i, r in enumerate(corrected):
            nxt_sym = corrected[i + 1].get('symbol', '') if i + 1 < len(corrected) else ''
            sym_now = r.get('symbol', '')
            if sym_now in ('+', '-', '*', '/') and nxt_sym == '=':
                continue
            cleaned.append(r)
        corrected = cleaned
        n = len(corrected)

        # Long digit runs (≥3) probably hide a misread operator
        i = 0
        while i < len(corrected):
            sym = corrected[i].get('symbol', '')
            if sym not in config.DIGIT_SYMBOLS:
                i += 1
                continue
            run_start = i
            while i < len(corrected) and corrected[i].get('symbol', '') in config.DIGIT_SYMBOLS:
                i += 1
            run_end = i
            run_len = run_end - run_start
            if run_len >= 3:
                interior = list(range(run_start + 1, run_end - 1))
                if interior:
                    lowest = min(interior, key=lambda k: corrected[k].get('confidence', 1.0))
                    _fix(lowest, '+', f'{run_len} consecutive digits → middle is likely +')

        return corrected

    def visualize_recognition(self,
                               original_image: np.ndarray,
                               characters: list[dict],
                               results: list[dict],
                               save_path: str = None) -> plt.Figure:
        n = len(results)
        if n == 0:
            return None

        fig = plt.figure(figsize=(max(12, n * 1.5), 8))
        fig.suptitle('Symbol Recognition Results', fontsize=13, fontweight='bold')

        ax_orig = fig.add_subplot(3, 1, 1)
        if len(original_image.shape) == 3:
            import cv2 as _cv2
            ax_orig.imshow(_cv2.cvtColor(original_image, _cv2.COLOR_BGR2RGB))
        else:
            ax_orig.imshow(original_image, cmap='gray')
        ax_orig.set_title('Original Preprocessed Image', fontsize=9)
        ax_orig.axis('off')

        for col, res in enumerate(results):
            ax_crop = fig.add_subplot(3, n, n + col + 1)
            ax_crop.imshow(characters[col]['image'], cmap='gray', vmin=0, vmax=255)
            ax_crop.set_title(res.get('symbol', '?'), fontsize=10, fontweight='bold')
            ax_crop.axis('off')

            ax_bar = fig.add_subplot(3, n, 2 * n + col + 1)
            conf   = res.get('confidence', 0.0)
            color  = 'green' if conf >= 0.90 else ('orange' if conf >= 0.70 else 'red')
            ax_bar.barh([''], [conf], color=color, height=0.4)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_title(f"{conf*100:.0f}%", fontsize=7)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Recognizer] Saved → {save_path}")

        return fig
