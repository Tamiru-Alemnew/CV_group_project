from pathlib import Path

ROOT = Path(__file__).parent

IMAGE_SIZE           = (32, 32)
NUM_CLASSES          = 18
CONFIDENCE_THRESHOLD = 0.70
RANDOM_SEED          = 42

CLASS_MAP: dict[int, str] = {
    0:  '0',  1:  '1',  2:  '2',  3:  '3',  4:  '4',
    5:  '5',  6:  '6',  7:  '7',  8:  '8',  9:  '9',
    10: '+',
    11: '-',
    12: 'x',
    13: '=',
    14: '*',
    15: '/',
    16: '(',
    17: ')',
}
INVERSE_CLASS_MAP: dict[str, int] = {v: k for k, v in CLASS_MAP.items()}

DIGIT_SYMBOLS    = {CLASS_MAP[i] for i in range(10)}
OPERATOR_SYMBOLS = {CLASS_MAP[i] for i in range(10, 18)}

DATA_DIR      = ROOT / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SYMBOLS_DIR   = DATA_DIR / 'symbols'

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

MODELS_DIR       = ROOT / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'
CHECKPOINTS_DIR  = MODELS_DIR / 'checkpoints'

MODEL_PATH           = SAVED_MODELS_DIR / 'best_model.h5'
CNN_MODEL_PATH       = SAVED_MODELS_DIR / 'best_cnn_model.h5'
MOBILENET_MODEL_PATH = SAVED_MODELS_DIR / 'best_mobilenet_model.h5'
SAVEDMODEL_PATH      = SAVED_MODELS_DIR / 'best_model_tf'

TRAINING_LOG_PATH  = MODELS_DIR / 'training_log.csv'

TRAINING_HIST_IMG  = MODELS_DIR / 'training_history.png'
CONFUSION_MAT_IMG  = MODELS_DIR / 'confusion_matrix.png'
PREDICTIONS_IMG    = MODELS_DIR / 'prediction_samples.png'
GRADCAM_IMG        = MODELS_DIR / 'gradcam_results.png'
ARCH_DIAGRAM_IMG   = MODELS_DIR / 'model_architecture.png'

CNN_TRAINING_HIST_IMG  = MODELS_DIR / 'cnn_training_history.png'
MN_TRAINING_HIST_IMG   = MODELS_DIR / 'mobilenet_training_history.png'
MODEL_COMPARISON_IMG   = MODELS_DIR / 'model_comparison.png'
CONFUSION_MATRICES_IMG = MODELS_DIR / 'confusion_matrices.png'
PER_CLASS_ACC_IMG      = MODELS_DIR / 'per_class_accuracy.png'
PREDICTION_SAMPLES_IMG = MODELS_DIR / 'prediction_samples_grid.png'
GRADCAM_CNN_IMG        = MODELS_DIR / 'gradcam_cnn.png'
GRADCAM_MN_IMG         = MODELS_DIR / 'gradcam_mobilenet.png'
GRADCAM_CMP_IMG        = MODELS_DIR / 'gradcam_comparison.png'

OUTPUT_DIR       = ROOT / 'output'
REPORTS_DIR      = OUTPUT_DIR / 'reports'
HISTORY_JSON     = OUTPUT_DIR / 'equation_history.json'
TESTS_OUTPUT_DIR = ROOT / 'tests' / 'output'

EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

AUGMENTATION: dict = {
    'rotation_range':     15,
    'width_shift_range':  0.1,
    'height_shift_range': 0.1,
    'zoom_range':         0.1,
    'shear_range':        0.1,
    'fill_mode':          'constant',
    'cval':               0,
}

_required_dirs = [
    RAW_DIR, PROCESSED_DIR, SYMBOLS_DIR,
    SAVED_MODELS_DIR, CHECKPOINTS_DIR,
    REPORTS_DIR, TESTS_OUTPUT_DIR,
]
for _folder_name in SYMBOL_FOLDER_MAP:
    _required_dirs.append(SYMBOLS_DIR / _folder_name)

for _d in _required_dirs:
    _d.mkdir(parents=True, exist_ok=True)
