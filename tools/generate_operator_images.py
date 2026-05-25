"""
Generate synthetic operator images for classes 10-15.

Produces ~600 images per class using a mix of:
  • PIL text rendering across multiple fonts and sizes
  • OpenCV stroke drawing (for +, -, =, /)
  • Augmentation: rotation, translation, scale, Gaussian noise, blur, dilation

Run from the project root:
    python tools/generate_operator_images.py
"""

import sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import config

# ── Constants ────────────────────────────────────────────────────────────────

RNG    = np.random.default_rng(42)
random.seed(42)

IMG_H, IMG_W  = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]  # 32 x 32
N_PER_CLASS   = 600   # images to generate per operator class
CANVAS        = 64    # render at higher resolution then downsample → cleaner edges

FONTS = [
    p for p in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
    ] if Path(p).exists()
]
print(f"[Gen] Using {len(FONTS)} fonts: {[Path(f).name for f in FONTS]}")


# ── Augmentation helpers ─────────────────────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """Apply random geometric + photometric augmentation to a CANVAS×CANVAS image."""
    # Random rotation ±18°
    angle = random.uniform(-18, 18)
    M = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=0)

    # Random translation ±6 px
    tx, ty = random.uniform(-6, 6), random.uniform(-6, 6)
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, T, (CANVAS, CANVAS), borderValue=0)

    # Random scale 0.75–1.10  (downscale allowed — keeps symbol within canvas)
    scale = random.uniform(0.75, 1.10)
    new_s = int(CANVAS * scale)
    if new_s < 8:
        new_s = 8
    img_s = cv2.resize(img, (new_s, new_s))
    out   = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    o     = (CANVAS - new_s) // 2
    if new_s <= CANVAS:
        out[max(o, 0):max(o, 0) + new_s, max(o, 0):max(o, 0) + new_s] = (
            img_s[:CANVAS - max(o, 0), :CANVAS - max(o, 0)]
        )
    else:
        off = (new_s - CANVAS) // 2
        out = img_s[off:off + CANVAS, off:off + CANVAS]
    img = out

    # Random dilation (thicken strokes)
    if random.random() < 0.5:
        k = random.choice([2, 3, 4])
        img = cv2.dilate(img, np.ones((k, k), np.uint8), iterations=1)

    # Gaussian blur (simulate slight focus variation)
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Gaussian noise
    if random.random() < 0.5:
        noise = RNG.integers(0, 40, img.shape, dtype=np.uint8)
        img   = cv2.add(img, noise)

    return img


def _finalise(img: np.ndarray) -> np.ndarray:
    """Resize augmented CANVAS image to 32×32, threshold, ensure white-on-black."""
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    return img


# ── Renderers ────────────────────────────────────────────────────────────────

def _pil_render(char: str) -> np.ndarray:
    """Render character using a random font at a random size on a CANVAS canvas."""
    font_path = random.choice(FONTS) if FONTS else None
    size      = random.randint(28, 50)
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pil_img = Image.new('L', (CANVAS, CANVAS), 0)
    draw    = ImageDraw.Draw(pil_img)

    # Centre the glyph
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - tw) // 2 - bbox[0]
    y = (CANVAS - th) // 2 - bbox[1]
    # Random small offset so the symbol is not always perfectly centred
    x += random.randint(-4, 4)
    y += random.randint(-4, 4)
    draw.text((x, y), char, fill=255, font=font)

    return np.array(pil_img, dtype=np.uint8)


def _cv_plus() -> np.ndarray:
    """Draw a plus sign with OpenCV lines."""
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    c, hl = CANVAS // 2, random.randint(12, 24)
    thick = random.choice([2, 3, 4, 5])
    cv2.line(img, (c, c - hl), (c, c + hl), 255, thick)
    cv2.line(img, (c - hl, c), (c + hl, c), 255, thick)
    return img


def _cv_minus() -> np.ndarray:
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    c, hl = CANVAS // 2, random.randint(12, 22)
    thick = random.choice([2, 3, 4, 5])
    cv2.line(img, (c - hl, c), (c + hl, c), 255, thick)
    return img


def _cv_equals() -> np.ndarray:
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    c, hl = CANVAS // 2, random.randint(12, 22)
    gap   = random.randint(4, 8)
    thick = random.choice([2, 3, 4])
    cv2.line(img, (c - hl, c - gap // 2), (c + hl, c - gap // 2), 255, thick)
    cv2.line(img, (c - hl, c + gap // 2), (c + hl, c + gap // 2), 255, thick)
    return img


def _cv_multiply() -> np.ndarray:
    """Draw × as two diagonal lines (×-style multiply)."""
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    c, hl = CANVAS // 2, random.randint(10, 20)
    thick = random.choice([2, 3, 4])
    cv2.line(img, (c - hl, c - hl), (c + hl, c + hl), 255, thick)
    cv2.line(img, (c + hl, c - hl), (c - hl, c + hl), 255, thick)
    return img


def _cv_divide() -> np.ndarray:
    """Draw / as a diagonal line."""
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    c, hl = CANVAS // 2, random.randint(14, 22)
    thick = random.choice([2, 3, 4])
    cv2.line(img, (c - hl, c + hl), (c + hl, c - hl), 255, thick)
    return img


def _cv_variable() -> np.ndarray:
    """Draw 'x' as two crossing diagonals — matches the algebraic variable style."""
    return _cv_multiply()   # same stroke pattern; fonts will supply variety


# ── Per-class generators ─────────────────────────────────────────────────────

def _generators_for(symbol: str):
    """Return a list of callables that produce raw CANVAS images for the symbol."""
    # Chars that map well to text rendering
    text_chars = {
        '+': ['+'],
        '-': ['-', '—'],
        'x': ['x', 'X'],
        '=': ['='],
        '*': ['×', '*', '✕'],
        '/': ['/', '÷'],
    }
    # Stroke-based fallbacks
    stroke_fns = {
        '+': _cv_plus,
        '-': _cv_minus,
        '=': _cv_equals,
        '*': _cv_multiply,
        '/': _cv_divide,
        'x': _cv_variable,
    }
    gens = []
    for ch in text_chars.get(symbol, [symbol]):
        gens.append(lambda c=ch: _pil_render(c))
    fn = stroke_fns.get(symbol)
    if fn:
        gens.append(fn)
    return gens


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_all():
    out_dir = config.SYMBOLS_DIR
    total   = 0

    for folder_name, class_idx in config.SYMBOL_FOLDER_MAP.items():
        symbol   = config.CLASS_MAP[class_idx]
        dest     = out_dir / folder_name
        dest.mkdir(parents=True, exist_ok=True)

        # Remove stale generated files so re-runs stay clean
        for old in dest.glob('gen_*.png'):
            old.unlink()

        gens  = _generators_for(symbol)
        saved = 0
        for i in range(N_PER_CLASS):
            gen = gens[i % len(gens)]
            try:
                raw = gen()
                aug = _augment(raw)
                img = _finalise(aug)
                cv2.imwrite(str(dest / f'gen_{i:04d}.png'), img)
                saved += 1
            except Exception as e:
                print(f"  [warn] {folder_name} sample {i}: {e}")

        print(f"[Gen] {folder_name:10s} (class {class_idx}, '{symbol}'): {saved} images saved")
        total += saved

    print(f"\n[Gen] Done — {total} operator images generated in {out_dir}")


if __name__ == '__main__':
    generate_all()
