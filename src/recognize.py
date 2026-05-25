"""
Stage 3 - Module 1: Symbol Recognizer

Loads the trained CNN once at startup, runs each 32×32 character crop
through it, and returns rich prediction metadata including top-3 classes,
raw probabilities, and a context-aware correction pass.
"""

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
    """
    Wraps the trained CNN for inference on segmented character images.

    Key design decisions:
    • Model is loaded ONCE at __init__ and cached — avoids re-loading the
      400 MB+ file on every prediction call.
    • A warmup dummy prediction is run at startup because TF JIT-compiles
      the computation graph on the first real call, which can take 1–2 s.
    • batch_recognize stacks all images into one numpy array and calls
      model.predict once — far more efficient than N individual calls.
    """

    def __init__(self, model_path: str = None,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Args:
            model_path:           Path to .h5 model. Defaults to config.MODEL_PATH.
            confidence_threshold: Predictions below this are flagged unreliable.
        """
        if not _TF_OK:
            raise ImportError("TensorFlow required for SymbolRecognizer.")

        self.threshold = confidence_threshold
        self.model_path = model_path or str(config.MODEL_PATH)
        self.model: tf.keras.Model | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Keras model and run a warmup prediction."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Train the model first with:  python main.py --train"
            )
        print(f"[Recognizer] Loading model from {self.model_path} …", end=' ')
        self.model = tf.keras.models.load_model(self.model_path)
        print("done.")

        # Warmup: first TF call triggers JIT compilation — do it now, not
        # when the user is waiting for a real result
        w, h = config.IMAGE_SIZE
        dummy = np.zeros((1, h, w, 1), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("[Recognizer] Warmup complete. Ready for inference.")

    # ─────────────────────────────────────────────────────────────────────────

    def prepare_image(self, char_image: np.ndarray) -> np.ndarray:
        """
        Normalise and reshape a character crop for model input.

        Ensures: float32, shape (1, H, W, 1), values in [0, 1].

        Args:
            char_image: uint8 or float32 array, any shape as long as it
                        contains a single character image.

        Returns:
            Float32 array of shape (1, H, W, 1) ready for model.predict().
        """
        w, h = config.IMAGE_SIZE
        img  = np.array(char_image, dtype=np.float32)

        if img.ndim == 2:
            pass  # (H, W)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]  # (H, W, 1) → (H, W)
        elif img.ndim == 4:
            img = img[0, :, :, 0]  # Already batched

        # Normalise if still in [0, 255] range
        import cv2
        if img.max() > 1.0:
            img = img / 255.0

        # Resize if shape doesn't match
        if img.shape != (h, w):
            img = cv2.resize(img.astype(np.float32), (w, h))

        return img[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)

    def recognize_single(self, char_image: np.ndarray) -> dict:
        """
        Run the CNN on one character image and return full prediction metadata.

        Args:
            char_image: Character crop (uint8 or float32).

        Returns:
            Dict with keys:
                symbol          – predicted symbol string (e.g., '3', '+')
                confidence      – float in [0, 1]
                is_reliable     – bool, True if confidence ≥ threshold
                top3            – list of 3 dicts {symbol, confidence}
                raw_probs       – full 16-element softmax output
        """
        inp         = self.prepare_image(char_image)
        probs       = self.model.predict(inp, verbose=0)[0]      # (16,)
        top_indices = np.argsort(probs)[::-1]                    # Descending

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
        """
        Recognise all characters from the segmenter in one pass.

        Merges recognition results with position-type metadata from
        the segmenter so downstream modules have everything they need.

        Args:
            characters: Character dicts from Segmenter.segment().

        Returns:
            List of enriched dicts — all segmenter fields plus all
            recognizer fields (symbol, confidence, top3, …).
        """
        results = []
        for ch in characters:
            rec = self.recognize_single(ch['image'])
            results.append({**ch, **rec})
        return results

    def batch_recognize(self, image_list: list[np.ndarray]) -> list[dict]:
        """
        Run a single batched model.predict() over all images.

        Stacking images into one array and calling predict once is
        significantly more efficient than N individual forward passes
        because it fully utilises GPU/CPU parallelism per batch.

        Args:
            image_list: List of character image arrays.

        Returns:
            List of recognition result dicts.
        """
        if not image_list:
            return []

        w, h = config.IMAGE_SIZE
        batch = np.stack([
            self.prepare_image(img)[0] for img in image_list
        ], axis=0)  # (N, H, W, 1)

        t0         = time.time()
        all_probs  = self.model.predict(batch, verbose=0)   # (N, 16)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[Recognizer] Batch inference: {len(image_list)} images "
              f"in {elapsed_ms:.1f} ms  "
              f"({elapsed_ms/len(image_list):.1f} ms/image)")

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

    # ─────────────────────────────────────────────────────────────────────────

    def get_confidence_report(self, results: list[dict]) -> float:
        """
        Print a formatted confidence table and return overall equation score.

        Warns if any individual prediction is below threshold, or if the
        overall equation confidence is below 0.6 (suggesting photo retake).

        Args:
            results: List of recognition result dicts.

        Returns:
            Overall equation confidence (geometric mean of per-symbol confs).
        """
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
            print("  ⚠  Overall confidence is low.")
            print("     Retake the photo with better lighting and clearer writing.")

        return overall

    def context_correction(self, results: list[dict]) -> list[dict]:
        """
        Apply rule-based corrections for common recognition mistakes.

        Rules:
        1. 'x' between two digit symbols → likely multiply '*'.
           Reason: in arithmetic "3x5" the letter-x is visually similar to
           the multiply sign × which the CNN may confuse with the variable x.
        2. '0' / 'O' (digit zero) disambiguation — handled implicitly since
           CLASS_MAP has no 'O'; any confusion resolves to digit '0'.
        3. '1' / 'l' disambiguation — same rationale; no 'l' class.

        Args:
            results: Recognition result list from recognize_sequence().

        Returns:
            Corrected list (new dicts where changes were made).
        """
        corrected = list(results)  # Shallow copy
        n = len(corrected)

        # x is in config.OPERATOR_SYMBOLS (class 12) but is a variable, not a math op
        MATH_OPS = {'+', '-', '*', '/'}

        def _sym(idx):
            if 0 <= idx < n:
                return corrected[idx].get('symbol', '')
            return ''

        def _fix(idx, new_sym, reason):
            corrected[idx] = {**corrected[idx], 'symbol': new_sym,
                               'corrected': True, 'correction_reason': reason}

        # ── Pre-pass: deduplicate multiple '=' signs ──────────────────────────
        # A valid equation has exactly one '='. When the model is uncertain it
        # tends to map many symbols to '='. Keep the most-confident one; replace
        # the rest with their next-best non-'=' prediction from top3, or '+'.
        eq_indices = [i for i, r in enumerate(corrected) if r.get('symbol') == '=']
        if len(eq_indices) > 1:
            best_eq = max(eq_indices, key=lambda i: corrected[i].get('confidence', 0.0))
            for i in eq_indices:
                if i == best_eq:
                    continue
                alt = '+'
                for pred in corrected[i].get('top3', [])[1:]:   # skip rank-1 which is '='
                    candidate = pred.get('symbol', '')
                    if candidate != '=' and candidate:
                        alt = candidate
                        break
                _fix(i, alt, f'duplicate = (kept most-confident at #{best_eq}) → {alt}')

        for i, r in enumerate(corrected):
            sym  = r.get('symbol', '')
            prev = _sym(i - 1)
            nxt  = _sym(i + 1)

            # Rule 1: 'x' between two digits → multiply
            if sym == 'x' and prev in config.DIGIT_SYMBOLS and nxt in config.DIGIT_SYMBOLS:
                _fix(i, '*', 'x between two digits → multiply')

            # Rule 2: '*' directly before '=' with a digit/var on its left
            # e.g. ['3','*','=','2'] — the '*' is almost certainly '+' or '-'
            # because multiply requires a right operand before the equals sign.
            elif sym == '*' and nxt == '=' and prev in config.DIGIT_SYMBOLS:
                _fix(i, '+', '* before = with digit on left → likely +')

            # Rule 3: '*' at the very end of the sequence (trailing operator)
            elif sym == '*' and i == n - 1:
                _fix(i, '+', '* at end of sequence → likely +')

            # Rule 3.5: '/' in a non-division context → likely handwritten '1'.
            # Canvas users draw '1' as a thin vertical stroke; the CNN maps it
            # to '/' because EMNIST '1' has a serif the canvas version lacks.
            # Division '/' is only valid when flanked by a digit/var on BOTH sides.
            elif sym == '/':
                digit_or_var = config.DIGIT_SYMBOLS | {'x'}
                if not (prev in digit_or_var and nxt in digit_or_var):
                    _fix(i, '1', '/ not flanked by digits/vars → likely handwritten 1')

            # Rule 4: consecutive math operators → second is likely a misread.
            # Exception: '* *' is the Python power operator '**' — keep both.
            # Uses MATH_OPS (excludes 'x' which is a variable, not an operator).
            elif (sym in MATH_OPS and prev in MATH_OPS
                    and not (sym == '*' and prev == '*')):
                # Don't touch unary minus: leading '-' or '-' right after '='
                if prev != '=':
                    _fix(i, '+', f'consecutive operators ({prev},{sym}) → second is likely +')

            # Rule 5: '=' appearing as the very first symbol → not an equals sign
            elif sym == '=' and i == 0:
                _fix(i, '+', '= as first symbol → likely +')

            # Rule 6a: low-confidence symbol at the end after '=' → likely '?' (unknown)
            # '?' means "solve for this" — treat as variable x
            # The CNN has no '?' class so it maps it to a random digit
            elif (i == n - 1
                    and r.get('confidence', 1.0) < 0.70
                    and sym in config.DIGIT_SYMBOLS
                    and any(_sym(j) == '=' for j in range(n))):
                eq_pos = next(j for j in range(n) if _sym(j) == '=')
                if i > eq_pos:
                    _fix(i, 'x', 'low-conf digit after = → likely ? (unknown variable)')

        # Cleanup pass A: remove any operator sitting immediately before '='
        # This arises when Rule 2 and Rule 6 both fire and leave a trailing
        # operator on the LHS (e.g. '3 + 8 + = 2' → '3 + 8 = 2').
        cleaned: list = []
        for i, r in enumerate(corrected):
            nxt_sym = corrected[i + 1].get('symbol', '') if i + 1 < len(corrected) else ''
            sym_now = r.get('symbol', '')
            if sym_now in ('+', '-', '*', '/') and nxt_sym == '=':
                continue  # drop this trailing operator
            cleaned.append(r)
        corrected = cleaned
        n = len(corrected)

        # Rule 6: 3+ consecutive digit tokens with no operator between them
        # → the one with lowest confidence is probably a misread '+' or '-'.
        # Simple equations like "1+1=2" have exactly one operator on each side
        # of '='; long pure-digit runs (length ≥ 3) indicate a missing operator.
        i = 0
        while i < len(corrected):
            sym = corrected[i].get('symbol', '')
            if sym not in config.DIGIT_SYMBOLS:
                i += 1
                continue
            # Start of a digit run
            run_start = i
            while i < len(corrected) and corrected[i].get('symbol', '') in config.DIGIT_SYMBOLS:
                i += 1
            run_end = i  # exclusive
            run_len = run_end - run_start
            if run_len >= 3:
                # Find lowest-confidence digit in the interior of the run
                interior = list(range(run_start + 1, run_end - 1))
                if interior:
                    lowest = min(interior, key=lambda k: corrected[k].get('confidence', 1.0))
                    _fix(lowest, '+',
                         f'{run_len} consecutive digits → middle digit (lowest conf) is likely +')

        return corrected

    # ─────────────────────────────────────────────────────────────────────────

    def visualize_recognition(self,
                               original_image: np.ndarray,
                               characters: list[dict],
                               results: list[dict],
                               save_path: str = None) -> plt.Figure:
        """
        Three-row figure: original image | character crops | confidence bars.

        Row 1: full equation image.
        Row 2: individual 32×32 character crops.
        Row 3: predicted symbol + horizontal confidence bar per character.

        Confidence colours:
            Green  ≥ 0.90 — high confidence
            Orange 0.70–0.90 — acceptable
            Red    < 0.70 — unreliable

        Args:
            original_image: Source image (greyscale or BGR).
            characters:     Character dicts from segmenter.
            results:        Recognition result dicts.
            save_path:      Optional output PNG path.

        Returns:
            matplotlib Figure.
        """
        n = len(results)
        if n == 0:
            return None

        fig = plt.figure(figsize=(max(12, n * 1.5), 8))
        fig.suptitle('Symbol Recognition Results', fontsize=13, fontweight='bold')

        # Row 1: original image (full width)
        ax_orig = fig.add_subplot(3, 1, 1)
        if len(original_image.shape) == 3:
            import cv2 as _cv2
            ax_orig.imshow(_cv2.cvtColor(original_image, _cv2.COLOR_BGR2RGB))
        else:
            ax_orig.imshow(original_image, cmap='gray')
        ax_orig.set_title('Original Preprocessed Image', fontsize=9)
        ax_orig.axis('off')

        # Rows 2 & 3: per-character crops and bars
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
            print(f"[Recognizer] Recognition figure → {save_path}")

        return fig
