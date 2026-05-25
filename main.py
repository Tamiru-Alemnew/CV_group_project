"""
main.py — Command-line entry point for the AI Handwritten Math Equation Solver.

Usage
-----
  python main.py --mode app                        # Launch Streamlit web app
  python main.py --mode image --image photo.jpg   # Process a single image
  python main.py --mode camera                    # Capture from webcam
  python main.py --mode test                      # Run integration tests
  python main.py --train                          # Prepare data + train CNN
"""

import argparse, sys, subprocess
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))
import config


def run_app():
    """Launch the Streamlit web application."""
    app_path = Path(__file__).parent / 'app' / 'app.py'
    print(f"[Main] Launching Streamlit app: {app_path}")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(app_path)],
                   check=True)


def run_image(image_path: str):
    """Process a single image through the full pipeline."""
    if not Path(image_path).exists():
        print(f"[Main] ERROR: Image not found: {image_path}")
        sys.exit(1)

    from src.pipeline import MathSolverPipeline
    pipeline = MathSolverPipeline()
    result   = pipeline.solve_from_image(image_path)

    # Print summary
    parsing  = result.get('parsing',  {})
    solution = result.get('solution', {})
    meta     = result.get('metadata', {})

    print('\n' + '=' * 60)
    print('PIPELINE RESULT SUMMARY')
    print('=' * 60)
    print(f"Equation : {parsing.get('equation_str',  'N/A')}")
    print(f"Answer   : {solution.get('raw', {}).get('answer_str', 'N/A')}")
    print(f"Type     : {solution.get('raw', {}).get('equation_type', 'N/A')}")
    print(f"Time     : {meta.get('total_time_ms', 0):.0f} ms")

    # Save HTML report
    if result.get('solution', {}).get('success'):
        report_path = pipeline.generate_report(result)
        print(f"Report   : {report_path}")


def run_camera():
    """Open webcam, capture, solve."""
    from src.pipeline import MathSolverPipeline
    pipeline = MathSolverPipeline()
    result   = pipeline.solve_from_camera()
    meta     = result.get('metadata', {})
    print(f"\n[Main] Done in {meta.get('total_time_ms', 0):.0f} ms")


def run_tests():
    """Run the Stage 1 and full pipeline integration tests."""
    test_scripts = [
        Path(__file__).parent / 'tests' / 'test_pipeline.py',
    ]
    for script in test_scripts:
        if script.exists():
            print(f"[Main] Running {script.name} …")
            result = subprocess.run([sys.executable, str(script)], capture_output=False)
            if result.returncode != 0:
                print(f"[Main] {script.name} FAILED (exit {result.returncode})")
            else:
                print(f"[Main] {script.name} PASSED")
        else:
            print(f"[Main] Test script not found: {script}")


def run_training():
    """Prepare dataset and train the CNN model."""
    print('[Main] Step 1/2: Preparing dataset …')
    from src.data_preparation import DataPreparator
    prep   = DataPreparator()
    splits = prep.prepare()

    print('\n[Main] Step 2/2: Training CNN …')
    from src.train import ModelTrainer
    trainer = ModelTrainer()
    trainer.run_full_pipeline()

    print('\n[Main] Training complete.')
    print(f"[Main] Model saved → {config.MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description='AI Handwritten Math Equation Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--mode',  choices=['app', 'image', 'camera', 'test'],
                        default='app', help='Run mode')
    parser.add_argument('--image', type=str, default=None,
                        help='Image path (required when --mode image)')
    parser.add_argument('--train', action='store_true',
                        help='Prepare data and train the CNN model')

    args = parser.parse_args()

    if args.train:
        run_training()
        return

    if args.mode == 'app':
        run_app()
    elif args.mode == 'image':
        if not args.image:
            print('ERROR: --image <path> is required for --mode image')
            sys.exit(1)
        run_image(args.image)
    elif args.mode == 'camera':
        run_camera()
    elif args.mode == 'test':
        run_tests()


if __name__ == '__main__':
    main()
