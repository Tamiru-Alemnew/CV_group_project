"""
demo/evaluation.py — Benchmark and Evaluation Script

Measures end-to-end accuracy (requires a trained model + sample images)
and individual stage processing times.

Run with:
    python demo/evaluation.py
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from src.math_solver     import MathSolver
from src.equation_parser import EquationParser
from src.step_formatter  import StepFormatter


# ─── Sample test set (used when no real images are available) ────────────────
SAMPLE_EQUATIONS = [
    # (true_equation,  expected_solutions_as_floats, expected_type)
    ('2*x + 5 = 15',          [5.0],        'linear'),
    ('3*x - 7 = 14',          [7.0],        'linear'),
    ('x + 10 = 25',           [15.0],       'linear'),
    ('4*x = 20',              [5.0],        'linear'),
    ('x/2 + 3 = 8',           [10.0],       'linear'),
    ('x - 3 = 12',            [15.0],       'linear'),
    ('5*x + 2 = 27',          [5.0],        'linear'),
    ('2*x - 8 = 0',           [4.0],        'linear'),
    ('x**2 + 5*x + 6 = 0',   [-2.0, -3.0], 'quadratic'),
    ('x**2 - 4 = 0',          [2.0, -2.0],  'quadratic'),
    ('x**2 - 5*x + 6 = 0',   [2.0, 3.0],   'quadratic'),
    ('x**2 - 9 = 0',          [3.0, -3.0],  'quadratic'),
    ('x**2 + 2*x + 1 = 0',   [-1.0],       'quadratic'),
    ('x**2 - 6*x + 9 = 0',   [3.0],        'quadratic'),
    ('25 + 37',               [62.0],       'expression'),
    ('144/12',                [12.0],       'expression'),
    ('2**8',                  [256.0],      'expression'),
    ('100 - 37',              [63.0],       'expression'),
    ('15*4',                  [60.0],       'expression'),
    ('81/9',                  [9.0],        'expression'),
]


def _check(result: dict, expected: list, tol: float = 1e-4) -> bool:
    if not result.get('success'):
        return False
    actual = [float(s) for s in result.get('solution', [])]
    return all(any(abs(a - e) < tol for a in actual) for e in expected)


# ─── Accuracy benchmark ──────────────────────────────────────────────────────

def test_on_sample_set() -> float:
    """
    Run the solver on the 20 sample equations and print a results table.

    Returns:
        End-to-end accuracy as a float in [0, 1].
    """
    solver  = MathSolver()
    correct = 0
    total   = len(SAMPLE_EQUATIONS)

    print('\n' + '═' * 90)
    print('SAMPLE SET EVALUATION')
    print('═' * 90)
    header = f"{'#':>3}  {'True Equation':<32}  {'Type':<12}  {'Correct?':>9}  {'Conf':>6}"
    print(header)
    print('─' * 90)

    for i, (eq, exp_sols, eq_type) in enumerate(SAMPLE_EQUATIONS, 1):
        result  = solver.solve_equation(eq)
        passed  = _check(result, exp_sols)
        if passed:
            correct += 1
        mark    = '✓' if passed else '✗'
        got_str = result.get('answer_str', result.get('error', '?'))[:28]
        print(f"  {i:>2}  {eq:<32}  {eq_type:<12}  {mark:>9}  {got_str}")

    accuracy = correct / total
    print('─' * 90)
    print(f"\n  Correct     : {correct} / {total}")
    print(f"  Accuracy    : {accuracy*100:.1f}%")
    print('═' * 90)
    return accuracy


# ─── Speed benchmark ─────────────────────────────────────────────────────────

def benchmark_speed(n_runs: int = 50) -> dict:
    """
    Measure average processing time per stage over n_runs repetitions.

    Stages timed:
        preprocessing, segmentation, recognition*, parsing, solving, total

    (* recognition requires a trained model; skipped if unavailable.)

    Returns:
        Dict mapping stage name → average milliseconds.
    """
    import cv2

    print(f'\n' + '═' * 50)
    print(f'SPEED BENCHMARK  ({n_runs} runs per stage)')
    print('═' * 50)

    timings: dict[str, list] = {
        'preprocessing':  [],
        'segmentation':   [],
        'recognition':    [],
        'parsing':        [],
        'solving':        [],
    }

    # ── Preprocessing & Segmentation ─────────────────────────────────────
    from src.preprocessing import ImagePreprocessor
    from src.segmentation  import Segmenter

    H, W  = 200, 620
    img   = np.full((H, W), 240, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for glyph, cx in [('3',40),('+',155),('5',270),('=',380),('8',490)]:
        cv2.putText(img, glyph, (cx, 140), font, 3.0, 25, 7, cv2.LINE_AA)

    tmp_path = str(config.TESTS_OUTPUT_DIR / '_bench_input.png')
    cv2.imwrite(tmp_path, img)

    preprocessor = ImagePreprocessor()
    segmenter    = Segmenter()

    for _ in range(n_runs):
        t0 = time.perf_counter()
        preprocessor.preprocess(tmp_path)
        timings['preprocessing'].append((time.perf_counter() - t0) * 1000)

        binary = preprocessor.get_binary_image()
        t0 = time.perf_counter()
        chars = segmenter.segment(binary)
        timings['segmentation'].append((time.perf_counter() - t0) * 1000)

    # ── Parsing ───────────────────────────────────────────────────────────
    parser   = EquationParser()
    test_seq = ['2', 'x', '+', '5', '=', '1', '5']
    for _ in range(n_runs):
        t0 = time.perf_counter()
        parser.parse(test_seq)
        timings['parsing'].append((time.perf_counter() - t0) * 1000)

    # ── Solving ───────────────────────────────────────────────────────────
    solver = MathSolver()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        solver.solve_equation('2*x + 5 = 15')
        timings['solving'].append((time.perf_counter() - t0) * 1000)

    # ── Recognition (optional) ────────────────────────────────────────────
    if config.MODEL_PATH.exists():
        try:
            from src.recognize import SymbolRecognizer
            rec    = SymbolRecognizer()
            dummy  = np.zeros((32, 32), dtype=np.uint8)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                rec.recognize_single(dummy)
                timings['recognition'].append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  (Recognition skipped: {e})")

    # ── Report ────────────────────────────────────────────────────────────
    averages: dict[str, float] = {}
    total_avg = 0.0

    print(f"\n  {'Stage':<20} {'Avg (ms)':>10}  {'Min (ms)':>10}  {'Max (ms)':>10}")
    print('  ' + '─' * 55)

    for stage, times in timings.items():
        if not times:
            continue
        avg = sum(times) / len(times)
        averages[stage] = round(avg, 2)
        total_avg      += avg
        print(f"  {stage:<20} {avg:>10.2f}  "
              f"{min(times):>10.2f}  {max(times):>10.2f}")

    print('  ' + '─' * 55)
    print(f"  {'TOTAL (no CNN)':<20} {total_avg - averages.get('recognition', 0):>10.2f}")
    if averages.get('recognition'):
        full = total_avg
        print(f"  {'TOTAL (with CNN)':<20} {full:>10.2f}")

    print('═' * 50)
    return averages


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    accuracy  = test_on_sample_set()
    timings   = benchmark_speed(n_runs=30)

    print('\n' + '═' * 50)
    print('EVALUATION COMPLETE')
    print(f"  Solver accuracy : {accuracy*100:.1f}%")
    print(f"  Full pipeline   : "
          f"{sum(timings.values()):.1f} ms avg (no CNN)")
    print('═' * 50)
