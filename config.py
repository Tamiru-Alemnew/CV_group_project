"""
config.py — Central configuration for AI Handwritten Math Equation Solver.

Every constant, path, and hyper-parameter lives here.
Change a value once and it propagates to every module automatically.
"""

from pathlib import Path

# ── Project root (this file's directory) ────────────────────────────────────
ROOT = Path(__file__).parent

# ── Image / model constants ──────────────────────────────────────────────────
IMAGE_SIZE           = (32, 32)   # (width, height) CNN input dimensions
NUM_CLASSES          = 18         # Total symbol classes (0-9, +, -, x, =, *, /, (, ))
CONFIDENCE_THRESHOLD = 0.70       # Below this → flag prediction as unreliable
RANDOM_SEED          = 42

# ── Symbol class map: integer → symbol string ────────────────────────────────
CLASS_MAP: dict[int, str] = {
    0:  '0',  1:  '1',  2:  '2',  3:  '3',  4:  '4',
    5:  '5',  6:  '6',  7:  '7',  8:  '8',  9:  '9',
    10: '+',            # plus sign
    11: '-',            # minus sign
    12: 'x',            # algebraic variable
    13: '=',            # equals sign
    14: '*',            # multiply  (asterisk → valid Python / SymPy)
    15: '/',            # divide    (forward slash)
    16: '(',            # open parenthesis
    17: ')',            # close parenthesis
}
INVERSE_CLASS_MAP: dict[str, int] = {v: k for k, v in CLASS_MAP.items()}

# Digits as a set for quick membership tests
DIGIT_SYMBOLS = {CLASS_MAP[i] for i in range(10)}
OPERATOR_SYMBOLS = {CLASS_MAP[i] for i in range(10, 18)}

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SYMBOLS_DIR   = DATA_DIR / 'symbols'

# Operator sub-folder name → class index
SYMBOL_FOLDER_MAP: dict[str, int] = {
    'plus':        10,
    'minus':       11,
    'variable':    12,
    'equals':      13,
    'multiply':    14,
    'divide':      15,
    'open_paren':  16,
    'close_paren': 17,
}

# ── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR       = ROOT / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'
CHECKPOINTS_DIR  = MODELS_DIR / 'checkpoints'

MODEL_PATH            = SAVED_MODELS_DIR / 'best_model.h5'
CNN_MODEL_PATH        = SAVED_MODELS_DIR / 'best_cnn_model.h5'
MOBILENET_MODEL_PATH  = SAVED_MODELS_DIR / 'best_mobilenet_model.h5'
SAVEDMODEL_PATH       = SAVED_MODELS_DIR / 'best_model_tf'

TRAINING_LOG_PATH  = MODELS_DIR / 'training_log.csv'

# ── Single-model artefacts (legacy / CNN-only runs) ──────────────────────────
TRAINING_HIST_IMG  = MODELS_DIR / 'training_history.png'
CONFUSION_MAT_IMG  = MODELS_DIR / 'confusion_matrix.png'
PREDICTIONS_IMG    = MODELS_DIR / 'prediction_samples.png'
GRADCAM_IMG        = MODELS_DIR / 'gradcam_results.png'
ARCH_DIAGRAM_IMG   = MODELS_DIR / 'model_architecture.png'

# ── Dual-model comparison artefacts (produced by Colab dual-train run) ───────
CNN_TRAINING_HIST_IMG  = MODELS_DIR / 'cnn_training_history.png'
MN_TRAINING_HIST_IMG   = MODELS_DIR / 'mobilenet_training_history.png'
MODEL_COMPARISON_IMG   = MODELS_DIR / 'model_comparison.png'
CONFUSION_MATRICES_IMG = MODELS_DIR / 'confusion_matrices.png'
PER_CLASS_ACC_IMG      = MODELS_DIR / 'per_class_accuracy.png'
PREDICTION_SAMPLES_IMG = MODELS_DIR / 'prediction_samples_grid.png'
GRADCAM_CNN_IMG        = MODELS_DIR / 'gradcam_cnn.png'
GRADCAM_MN_IMG         = MODELS_DIR / 'gradcam_mobilenet.png'
GRADCAM_CMP_IMG        = MODELS_DIR / 'gradcam_comparison.png'

# ── Output / history paths ───────────────────────────────────────────────────
OUTPUT_DIR       = ROOT / 'output'
REPORTS_DIR      = OUTPUT_DIR / 'reports'
HISTORY_JSON     = OUTPUT_DIR / 'equation_history.json'
TESTS_OUTPUT_DIR = ROOT / 'tests' / 'output'

# ── Training hyper-parameters ────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# ── Data augmentation parameters (fed to ImageDataGenerator) ─────────────────
AUGMENTATION: dict = {
    'rotation_range':     15,   # ±15° rotation
    'width_shift_range':  0.1,  # ±10% horizontal shift
    'height_shift_range': 0.1,  # ±10% vertical shift
    'zoom_range':         0.1,  # ±10% zoom
    'shear_range':        0.1,  # ±10% shear
    'fill_mode':          'constant',
    'cval':               0,    # fill new pixels with black
}

# ── Auto-create required directories on import ───────────────────────────────
_required_dirs = [
    RAW_DIR, PROCESSED_DIR, SYMBOLS_DIR,
    SAVED_MODELS_DIR, CHECKPOINTS_DIR,
    REPORTS_DIR, TESTS_OUTPUT_DIR,
]
for _folder_name in SYMBOL_FOLDER_MAP:
    _required_dirs.append(SYMBOLS_DIR / _folder_name)

for _d in _required_dirs:
    _d.mkdir(parents=True, exist_ok=True)
