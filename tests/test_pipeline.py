"""
Integration Tests — Stage 1 + Math Solver

Tests the equation parser and math solver directly (no trained model
required). Full CNN-dependent pipeline tests are noted below but require
python main.py --train to be run first.

Run with:
    python tests/test_pipeline.py
"""

import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.equation_parser import EquationParser
from src.math_solver     import MathSolver
from src.step_formatter  import StepFormatter

OUTPUT_DIR = config.TESTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'


# ─── Test helpers ────────────────────────────────────────────────────────────

def _check_solution(result: dict, expected_solutions: list,
                    tolerance: float = 1e-4) -> bool:
    """
    Check that every expected solution is present in result['solution'].
    Compares numerically with given tolerance.
    """
    if not result.get('success'):
        return False
    actual = [float(s) for s in result.get('solution', [])]
    for exp in expected_solutions:
        if not any(abs(a - exp) < tolerance for a in actual):
            return False
    return True


def _run_test(name: str, equation: str, expected_solutions: list,
              solver: MathSolver, formatter: StepFormatter,
              report_lines: list) -> bool:
    """Run one solver test case and print the result."""
    t0     = time.time()
    result = solver.solve_equation(equation)
    ms     = (time.time() - t0) * 1000
    passed = _check_solution(result, expected_solutions)

    status = PASS if passed else FAIL
    print(f"\n{status} {name}")
    print(f"       Equation : {equation}")
    print(f"       Expected : {expected_solutions}")
    print(f"       Got      : {result.get('solution', [])}  "
          f"({result.get('answer_str','?')})")
    print(f"       Time     : {ms:.1f} ms")

    if not passed and result.get('error'):
        print(f"       Error    : {result['error']}")

    # Show steps
    display = formatter.format_for_display(result)
    for line in display.split('\n'):
        print(f"  {line}")

    report_lines.append({
        'test':     name,
        'equation': equation,
        'expected': str(expected_solutions),
        'got':      str(result.get('solution', [])),
        'passed':   passed,
        'time_ms':  round(ms, 1),
    })
    return passed


# ─── Parser tests ────────────────────────────────────────────────────────────

def _parser_impl() -> bool:
    parser = EquationParser()
    print('\n' + '═' * 60)
    print('EQUATION PARSER TESTS')
    print('═' * 60)

    cases = [
        (['2', 'x', '+', '5', '=', '1', '5'],           '2*x + 5 = 15'),
        (['1', '0', '0', '+', '5', '=', '1', '0', '5'], '100 + 5 = 105'),
        (['3', 'x', '-', '7', '=', '1', '4'],           '3*x - 7 = 14'),
        (['-', '2', 'x', '+', '4', '=', '0'],           '-2*x + 4 = 0'),
    ]

    all_pass = True
    for syms, expected in cases:
        eq_str, _ = parser.parse(syms)
        passed = eq_str.replace(' ', '') == expected.replace(' ', '')
        status = PASS if passed else FAIL
        print(f"{status} parse({syms})  →  '{eq_str}'  (expected '{expected}')")
        all_pass = all_pass and passed
    return all_pass


def test_parser():
    """Test EquationParser transformations."""
    assert _parser_impl(), "One or more parser transformation tests failed"


# ─── Solver tests ────────────────────────────────────────────────────────────

def _solver_impl() -> tuple:
    solver    = MathSolver()
    formatter = StepFormatter()
    report    = []

    test_cases = [
        ('Linear 1',              '2*x + 5 = 15',         [5.0]),
        ('Linear 2',              '3*x - 7 = 14',         [7.0]),
        ('Linear 3',              'x + 10 = 25',          [15.0]),
        ('Linear 4',              '4*x = 20',             [5.0]),
        ('Linear 5',              'x/2 + 3 = 8',          [10.0]),
        ('Quadratic 1',           'x**2 + 5*x + 6 = 0',  [-2.0, -3.0]),
        ('Quadratic 2 (diff sq)', 'x**2 - 4 = 0',        [2.0, -2.0]),
        ('Quadratic 3',           'x**2 - 5*x + 6 = 0',  [2.0, 3.0]),
        ('Arithmetic 1',          '25 + 37',              [62.0]),
        ('Arithmetic 2',          '144/12',               [12.0]),
    ]

    print('\n' + '═' * 60)
    print('MATH SOLVER TESTS')
    print('═' * 60)

    passed_count = 0
    for name, eq, expected in test_cases:
        ok = _run_test(name, eq, expected, solver, formatter, report)
        if ok:
            passed_count += 1

    print('\n─── Special cases ───')
    r_no_sol = solver.solve_equation('2*x + 3 = 2*x + 5')
    print(f"  No-solution case:        success={r_no_sol['success']}  "
          f"error='{r_no_sol.get('error','')}'")

    r_inf = solver.solve_equation('2*x + 4 = 2*x + 4')
    print(f"  Infinite-solutions case: success={r_inf['success']}  "
          f"error='{r_inf.get('error','')}'")

    return passed_count, len(test_cases), report


def test_solver():
    """Test MathSolver on all specified equations."""
    passed, total, _ = _solver_impl()
    assert passed == total, f"Solver: {passed}/{total} test cases passed"


# ─── Stage 1 test ────────────────────────────────────────────────────────────

def _stage1_impl() -> bool:
    from src.preprocessing import ImagePreprocessor
    from src.segmentation  import Segmenter
    import numpy as np

    print('\n' + '═' * 60)
    print('STAGE 1 PIPELINE SMOKE TEST')
    print('═' * 60)

    H, W  = 200, 620
    img   = np.full((H, W), 240, dtype=np.uint8)
    import cv2
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for glyph, cx in [('3', 40), ('+', 155), ('5', 270), ('=', 380), ('8', 490)]:
        cv2.putText(img, glyph, (cx, 140), font, 3.0, 25, 7, cv2.LINE_AA)

    tmp_path = str(OUTPUT_DIR / 'stage1_test_input.png')
    cv2.imwrite(tmp_path, img)

    preprocessor = ImagePreprocessor()
    normalized   = preprocessor.preprocess(tmp_path)
    binary       = preprocessor.get_binary_image()

    assert binary is not None, "Preprocessing returned no binary image"
    assert normalized.dtype == np.float32, "Normalised image not float32"
    assert 0.0 <= normalized.min() and normalized.max() <= 1.0

    segmenter  = Segmenter()
    characters = segmenter.segment(binary)

    print(f"  Preprocessing : {PASS} — output shape {normalized.shape}")
    print(f"  Segmentation  : "
          f"{'PASS' if len(characters) >= 3 else 'WARN'} — "
          f"{len(characters)} characters detected")

    fig = preprocessor.visualize_steps(save_path=str(OUTPUT_DIR / 'pp_pipeline.png'))
    plt_close_if_fig(fig)
    segmenter.visualize_segmentation(binary, characters,
                                     save_path=str(OUTPUT_DIR / 'segmentation.png'))

    return len(characters) >= 3


def test_stage1_pipeline():
    """Quick Stage 1 smoke test using synthetic equation image."""
    assert _stage1_impl(), "Stage 1: fewer than 3 characters detected"


def plt_close_if_fig(fig):
    try:
        import matplotlib.pyplot as plt
        if fig:
            plt.close(fig)
    except Exception:
        pass


# ─── Report writer ───────────────────────────────────────────────────────────

def write_report(parser_ok: bool, passed: int, total: int,
                 stage1_ok: bool, report_rows: list) -> None:
    """Save a plain-text test report to tests/output/test_report.txt."""
    lines = [
        '=' * 60,
        'AI HANDWRITTEN MATH EQUATION SOLVER — TEST REPORT',
        '=' * 60,
        f'Parser tests   : {"PASS" if parser_ok else "FAIL"}',
        f'Solver tests   : {passed}/{total} passed',
        f'Stage 1 smoke  : {"PASS" if stage1_ok else "FAIL"}',
        '',
        'Detailed solver results:',
        f"{'Test':<26} {'Equation':<30} {'Expected':<18} {'Result':<20} {'ms':>6}",
        '-' * 110,
    ]
    for r in report_rows:
        lines.append(
            f"{'PASS' if r['passed'] else 'FAIL'} {r['test']:<23} "
            f"{r['equation']:<30} {r['expected']:<18} {r['got']:<20} "
            f"{r['time_ms']:>5.1f}"
        )
    lines += ['', '=' * 60]

    report_path = OUTPUT_DIR / 'test_report.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n[Tests] Report saved → {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser_ok             = _parser_impl()
    passed, total, report = _solver_impl()
    stage1_ok             = _stage1_impl()

    write_report(parser_ok, passed, total, stage1_ok, report)

    print('\n' + '═' * 60)
    print('SUMMARY')
    print('═' * 60)
    print(f"  Parser tests   : {'PASS' if parser_ok else 'FAIL'}")
    print(f"  Solver tests   : {passed}/{total} passed  "
          f"({'PASS' if passed == total else 'FAIL'})")
    print(f"  Stage 1 smoke  : {'PASS' if stage1_ok else 'WARN'}")

    all_pass = parser_ok and (passed == total) and stage1_ok
    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
