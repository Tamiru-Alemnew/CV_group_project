"""
Stage 5 - Master Pipeline Integrator

MathSolverPipeline connects every module into a single solve_from_image()
call. External code (the web app, CLI, tests) only needs to interact with
this class — all module wiring is handled internally.
"""

import sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import config

from src.preprocessing    import ImagePreprocessor
from src.segmentation     import Segmenter
from src.equation_parser  import EquationParser
from src.math_solver      import MathSolver
from src.step_formatter   import StepFormatter
from src.history_manager  import HistoryManager


class MathSolverPipeline:
    """
    End-to-end pipeline: raw image → preprocessing → segmentation →
    recognition → parsing → solving → formatted output.

    The CNN model (SymbolRecognizer + GradCAM) is loaded lazily to allow
    the pipeline to be used for solving manually-typed equations even when
    the model has not been trained yet.
    """

    VERSION = '1.0'

    def __init__(self, model_path: str = None,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Initialise all modules.  CNN model is loaded if the file exists.

        Args:
            model_path:           Path to .h5 model. Defaults to config.MODEL_PATH.
            confidence_threshold: Forwarded to SymbolRecognizer.
        """
        self.model_path  = model_path or str(config.MODEL_PATH)
        self.threshold   = confidence_threshold
        self.model_ready = False

        # Always-available modules
        self.preprocessor = ImagePreprocessor()
        self.segmenter    = Segmenter()
        self.parser       = EquationParser()
        self.solver       = MathSolver()
        self.formatter    = StepFormatter()
        self.history      = HistoryManager()

        # CNN-dependent modules (loaded only when model exists)
        self.recognizer = None
        self.gradcam    = None
        self._try_load_model()

        print(f"\n[Pipeline v{self.VERSION}] Initialised.")
        print(f"  Preprocessor  : ready")
        print(f"  Segmenter     : ready")
        print(f"  Parser        : ready")
        print(f"  Solver        : ready")
        print(f"  CNN model     : {'ready' if self.model_ready else 'NOT LOADED (train first)'}")

    def _try_load_model(self) -> None:
        """Attempt to load the CNN. Sets self.model_ready on success."""
        if not Path(self.model_path).exists():
            return
        try:
            from src.recognize import SymbolRecognizer
            from src.gradcam   import GradCAM
            self.recognizer  = SymbolRecognizer(self.model_path, self.threshold)
            self.gradcam     = GradCAM(self.recognizer.model)
            self.model_ready = True
        except Exception as e:
            print(f"[Pipeline] WARNING: could not load CNN model — {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Primary solve entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def solve_from_image(self, image_input) -> dict:
        """
        Run the complete pipeline on an image and return a comprehensive result.

        Args:
            image_input: str/Path to an image file  OR  numpy BGR/grey array.

        Returns:
            Comprehensive result dict containing:
                preprocessing  – success flag + list of intermediate images
                segmentation   – character_count, annotated_image, characters
                recognition    – symbol_results, confidence, gradcam_overlays
                parsing        – transformation_steps, equation_str, validation
                solution       – full MathSolver output + LaTeX-formatted steps
                metadata       – total_time_ms, timestamp, pipeline_version
        """
        t_total = time.time()
        result  = {
            'preprocessing': {}, 'segmentation': {}, 'recognition': {},
            'parsing':       {}, 'solution':     {}, 'metadata':    {},
        }

        # ── Stage 1: Preprocessing ────────────────────────────────────────
        t0 = time.time()
        try:
            if isinstance(image_input, (str, Path)):
                normalized = self.preprocessor.preprocess(str(image_input))
            else:
                # numpy array: write to temp file so preprocess() can load it
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                cv2.imwrite(tmp_path, image_input)
                normalized = self.preprocessor.preprocess(tmp_path)
                os.unlink(tmp_path)

            binary = self.preprocessor.get_binary_image()
            result['preprocessing'] = {
                'success':          True,
                'time_ms':          round((time.time() - t0) * 1000, 1),
                'pipeline_steps':   self.preprocessor.pipeline_steps,
            }
        except Exception as e:
            result['preprocessing'] = {'success': False, 'error': str(e)}
            result['metadata']      = self._meta(t_total)
            return result

        # ── Stage 2: Segmentation ─────────────────────────────────────────
        t0 = time.time()
        try:
            characters  = self.segmenter.segment(binary)
            annotated   = self.segmenter.visualize_segmentation(binary, characters)
            result['segmentation'] = {
                'success':         True,
                'character_count': len(characters),
                'characters':      characters,
                'annotated_image': annotated,
                'time_ms':         round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            result['segmentation'] = {'success': False, 'error': str(e)}
            result['metadata']     = self._meta(t_total)
            return result

        if not characters:
            result['segmentation']['error'] = 'No characters detected'
            result['metadata'] = self._meta(t_total)
            return result

        # ── Stage 3: Recognition (CNN required) ───────────────────────────
        t0 = time.time()
        if self.model_ready and self.recognizer:
            try:
                rec_results  = self.recognizer.recognize_sequence(characters)
                rec_results  = self.recognizer.context_correction(rec_results)
                confidence   = self.recognizer.get_confidence_report(rec_results)

                # Grad-CAM overlays
                cam_overlays = []
                if self.gradcam:
                    for ch, rec in zip(characters, rec_results):
                        cls_idx  = config.INVERSE_CLASS_MAP.get(rec.get('symbol', ''), 0)
                        _, overlay = self.gradcam.compute_gradcam(
                            ch['image'].astype(np.float32) / 255.0, cls_idx
                        )
                        cam_overlays.append(overlay)

                result['recognition'] = {
                    'success':        True,
                    'symbol_results': rec_results,
                    'confidence':     confidence,
                    'gradcam_overlays': cam_overlays,
                    'time_ms':        round((time.time() - t0) * 1000, 1),
                }
                symbols_for_parser = rec_results
            except Exception as e:
                result['recognition'] = {'success': False, 'error': str(e)}
                symbols_for_parser    = []
        else:
            result['recognition'] = {
                'success': False,
                'error':   'CNN model not loaded. Train the model first.',
            }
            symbols_for_parser = []

        # ── Stage 4: Parsing ──────────────────────────────────────────────
        t0 = time.time()
        if symbols_for_parser:
            try:
                equation_str, steps = self.parser.parse(symbols_for_parser)
                valid, err_msg, eq_type = self.parser.validate_equation(equation_str)
                self.parser.display_parse_steps(symbols_for_parser, equation_str)

                result['parsing'] = {
                    'success':              True,
                    'equation_str':         equation_str,
                    'transformation_steps': steps,
                    'is_valid':             valid,
                    'validation_error':     err_msg,
                    'equation_type':        eq_type,
                    'time_ms':              round((time.time() - t0) * 1000, 1),
                }
            except Exception as e:
                result['parsing'] = {'success': False, 'error': str(e)}
        else:
            result['parsing'] = {'success': False, 'error': 'No recognised symbols to parse.'}

        # ── Stage 5: Solving ──────────────────────────────────────────────
        t0 = time.time()
        if result['parsing'].get('success') and result['parsing'].get('is_valid'):
            try:
                equation_str  = result['parsing']['equation_str']
                solution      = self.solver.solve_equation(equation_str)
                latex_solution = self.formatter.format_for_latex(solution)
                display_text   = self.formatter.format_for_display(solution)

                result['solution'] = {
                    'success':       True,
                    'raw':           solution,
                    'latex':         latex_solution,
                    'display_text':  display_text,
                    'time_ms':       round((time.time() - t0) * 1000, 1),
                }

                # Save to history
                self.history.add_record({
                    'image_path':    str(image_input) if isinstance(image_input, (str, Path)) else 'array',
                    'equation':      equation_str,
                    'solution':      solution,
                    'confidence':    result['recognition'].get('confidence', 0.0),
                    'equation_type': solution.get('equation_type', ''),
                })

                print("\n" + display_text)
            except Exception as e:
                result['solution'] = {'success': False, 'error': str(e)}
        else:
            result['solution'] = {
                'success': False,
                'error': result['parsing'].get('error', 'Parsing failed'),
            }

        # ── Metadata ──────────────────────────────────────────────────────
        result['metadata'] = self._meta(t_total)
        return result

    def _meta(self, t_start: float) -> dict:
        return {
            'total_time_ms':    round((time.time() - t_start) * 1000, 1),
            'timestamp':        datetime.now().isoformat(),
            'pipeline_version': self.VERSION,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Direct equation solving (bypass image pipeline)
    # ─────────────────────────────────────────────────────────────────────────

    def solve_equation_direct(self, equation_str: str) -> dict:
        """
        Solve a manually typed equation string without any image processing.

        Useful for testing the solver and for the web app's type-in mode.

        Args:
            equation_str: Clean equation string, e.g. '2*x + 5 = 15'.

        Returns:
            Full solution dict from MathSolver.
        """
        solution      = self.solver.solve_equation(equation_str)
        display_text  = self.formatter.format_for_display(solution)
        print(display_text)
        return solution

    # ─────────────────────────────────────────────────────────────────────────
    # Camera capture
    # ─────────────────────────────────────────────────────────────────────────

    def solve_from_camera(self) -> dict:
        """
        Open webcam, display live feed, capture on SPACEBAR, solve.

        Returns:
            Same comprehensive result dict as solve_from_image().
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (cv2.VideoCapture(0) failed).")

        print("[Pipeline] Camera opened. Press SPACEBAR to capture, ESC to cancel.")
        captured = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Math Equation Solver — Press SPACE to capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                captured = frame.copy()
                break
            if key == 27:   # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured is None:
            raise RuntimeError("No frame captured.")

        print("[Pipeline] Frame captured. Processing …")
        return self.solve_from_image(captured)

    # ─────────────────────────────────────────────────────────────────────────
    # HTML report
    # ─────────────────────────────────────────────────────────────────────────

    def generate_report(self, pipeline_result: dict) -> Path:
        """
        Create a self-contained HTML report and save it to output/reports/.

        Includes:
          • All preprocessing step images (base64-encoded).
          • Segmentation annotated image.
          • Full solution steps.
          • Confidence scores.
          • Timestamp in filename.

        Args:
            pipeline_result: Dict returned by solve_from_image().

        Returns:
            Path to the saved HTML file.
        """
        import base64, io
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path  = config.REPORTS_DIR / f'report_{timestamp}.html'
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        def _img_to_b64(img: np.ndarray) -> str:
            """Convert numpy image to base64 PNG string for embedding."""
            import cv2 as _cv2
            _, buf = _cv2.imencode('.png', img)
            return base64.b64encode(buf).decode('utf-8')

        # Build preprocessing image strips
        prep_html = ''
        steps = pipeline_result.get('preprocessing', {}).get('pipeline_steps', {})
        for step_name, step_img in steps.items():
            if isinstance(step_img, np.ndarray):
                b64 = _img_to_b64(step_img if step_img.dtype == np.uint8
                                   else (step_img * 255).astype(np.uint8))
                prep_html += (
                    f'<div style="display:inline-block;margin:4px;text-align:center">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="height:120px;border:1px solid #ccc" />'
                    f'<br><small>{step_name}</small></div>'
                )

        # Solution steps
        sol = pipeline_result.get('solution', {}).get('raw', {})
        steps_html = ''
        for step in sol.get('steps', []):
            steps_html += (
                f'<div style="margin:8px 0;padding:8px;background:#f9f9f9;'
                f'border-left:3px solid #2196F3">'
                f'<strong>Step {step["step_number"]}: {step["description"]}</strong><br>'
                f'<code>{step["expression"]}</code><br>'
                f'<em style="color:#555">{step.get("explanation","")}</em></div>'
            )

        equation    = pipeline_result.get('parsing', {}).get('equation_str', 'N/A')
        answer      = sol.get('answer_str', 'N/A')
        eq_type     = sol.get('equation_type', 'N/A')
        conf        = pipeline_result.get('recognition', {}).get('confidence', 0.0)
        total_ms    = pipeline_result.get('metadata', {}).get('total_time_ms', 0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Math Equation Solver Report — {timestamp}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; }}
    h1   {{ color: #1976D2; }} h2 {{ color: #333; border-bottom: 1px solid #ddd; }}
    .answer {{ font-size: 1.4em; font-weight: bold; color: #2e7d32;
               padding: 12px; background: #e8f5e9; border-radius: 6px; }}
    .meta   {{ color: #888; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>AI Handwritten Math Equation Solver</h1>
  <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
     Total processing time: {total_ms:.0f} ms</p>

  <h2>Recognised Equation</h2>
  <p style="font-size:1.2em"><code>{equation}</code></p>
  <p>Type: <strong>{eq_type}</strong> &nbsp;&nbsp;
     CNN confidence: <strong>{conf*100:.1f}%</strong></p>

  <h2>Answer</h2>
  <div class="answer">{answer}</div>

  <h2>Step-by-Step Solution</h2>
  {steps_html if steps_html else '<p>No steps available.</p>'}

  <h2>Preprocessing Pipeline</h2>
  <div>{prep_html if prep_html else '<p>No preprocessing images available.</p>'}</div>

  <hr>
  <p class="meta">Pipeline version {pipeline_result.get('metadata',{}).get('pipeline_version','?')}</p>
</body>
</html>"""

        out_path.write_text(html, encoding='utf-8')
        print(f"[Pipeline] HTML report saved → {out_path}")
        return out_path
