"""
Stage 1 Integration Test
========================
Runs a full end-to-end test of both Stage 1 modules:
  Module 1 – ImagePreprocessor
  Module 2 – CharacterSegmenter

Usage
-----
# Uses the auto-generated synthetic test equation:
    python tests/test_stage1.py

# Uses your own photo of a handwritten equation:
    python tests/test_stage1.py path/to/your_equation.jpg

All output images are saved to tests/output/.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for headless / CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so we can import from src/ without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from preprocessing import ImagePreprocessor
from segmentation  import Segmenter as CharacterSegmenter

OUTPUT_DIR = Path(__file__).parent / 'output'


# ===========================================================================
# Synthetic test-image generator
# ===========================================================================

def create_test_equation_image(save_path: Path) -> Path:
    """
    Generate a synthetic handwritten-style equation image for testing.

    Produces an image that approximates "3 + 5 = 8" with:
    - Off-white paper background with subtle noise (paper grain)
    - Dark-gray thick strokes (pen writing)
    - A slight 3-degree tilt (to exercise the deskew step)
    - A handful of random noise dots (to exercise morphological opening)

    Args:
        save_path: Where to write the PNG file.

    Returns:
        The path where the image was saved.
    """
    print("[TestGen] Creating synthetic test equation image …")

    H, W = 200, 620
    # Start with off-white paper
    img = np.full((H, W), fill_value=240, dtype=np.uint8)

    # Add subtle random paper texture
    rng   = np.random.default_rng(seed=7)
    noise = rng.integers(0, 18, size=(H, W), dtype=np.uint8)
    img   = cv2.subtract(img, noise)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness  = 7
    ink_color  = 25   # Dark gray — not pure black, like real pen ink

    # Character positions (cx=left edge of glyph, cy=baseline)
    equation_chars = [
        ('3', (40,  140)),
        ('+', (155, 140)),
        ('5', (270, 140)),
        ('=', (380, 140)),
        ('8', (490, 140)),
    ]
    for glyph, (cx, cy) in equation_chars:
        cv2.putText(img, glyph, (cx, cy), font, font_scale,
                    ink_color, thickness, cv2.LINE_AA)

    # Apply a slight tilt to test the deskew pipeline step
    center = (W // 2, H // 2)
    M      = cv2.getRotationMatrix2D(center, angle=3.0, scale=1.0)
    img    = cv2.warpAffine(img, M, (W, H),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=240)

    # Scatter a few noise dots (small specks that opening should remove)
    for _ in range(25):
        nx = int(rng.integers(0, W))
        ny = int(rng.integers(0, H))
        r  = int(rng.integers(1, 3))
        cv2.circle(img, (nx, ny), r, int(rng.integers(30, 80)), -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    print(f"[TestGen] Test image saved → {save_path}")
    return save_path


# ===========================================================================
# Module 1 test
# ===========================================================================

def test_preprocessor(image_path: Path) -> tuple:
    """
    Test ImagePreprocessor: run all 8 pipeline steps and save the figure.

    Args:
        image_path: Path to the input equation image.

    Returns:
        (preprocessor, normalized_image, binary_image)
    """
    print("\n" + "─" * 60)
    print("MODULE 1 — IMAGE PREPROCESSOR")
    print("─" * 60)

    preprocessor = ImagePreprocessor(
        blur_kernel_size=(5, 5),
        morph_kernel_size=(3, 3)
    )

    print(f"Input image : {image_path}")
    normalized = preprocessor.preprocess(image_path)
    binary     = preprocessor.get_binary_image()

    print(f"Output shape : {normalized.shape}")
    print(f"Output dtype : {normalized.dtype}")
    print(f"Value range  : [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Binary shape : {binary.shape}  dtype={binary.dtype}  "
          f"unique_values={np.unique(binary).tolist()}")

    # Save pipeline visualization
    pipeline_path = OUTPUT_DIR / 'preprocessing_pipeline.png'
    fig = preprocessor.visualize_steps(save_path=str(pipeline_path))
    plt.close(fig)

    print(f"\n[Module 1] PASSED ✓  —  pipeline figure saved.")
    return preprocessor, normalized, binary


# ===========================================================================
# Module 2 test
# ===========================================================================

def test_segmenter(binary_image: np.ndarray) -> list:
    """
    Test CharacterSegmenter: detect characters and save visualizations.

    Args:
        binary_image: uint8 binary image from Module 1.

    Returns:
        List of character dicts from segment().
    """
    print("\n" + "─" * 60)
    print("MODULE 2 — CHARACTER SEGMENTER")
    print("─" * 60)

    segmenter = CharacterSegmenter(
        min_contour_area=50,
        padding=4,
        target_size=(32, 32)
    )

    characters = segmenter.segment(binary_image)

    print(f"\nCharacters detected : {len(characters)}")
    if not characters:
        print("[Module 2] WARNING: Zero characters found. "
              "Adjust min_contour_area or check preprocessing output.")
        return characters

    print("\nPer-character details:")
    print(f"  {'#':>3}  {'bbox (x,y,w,h)':>22}  {'position':>12}  {'center':>14}")
    print("  " + "-" * 60)
    for c in characters:
        print(f"  {c['index']:>3}  {str(c['bbox']):>22}  "
              f"{c['position_type']:>12}  {str(c['center']):>14}")

    # Save bounding-box annotated image
    bbox_path = OUTPUT_DIR / 'segmentation_bboxes.png'
    annotated = segmenter.visualize_segmentation(
        binary_image, characters, save_path=str(bbox_path)
    )

    # Save individual character-crop grid
    crops_path = OUTPUT_DIR / 'segmented_characters.png'
    fig = segmenter.visualize_crops(characters, save_path=str(crops_path))
    if fig:
        plt.close(fig)

    print(f"\n[Module 2] PASSED ✓  —  {len(characters)} characters segmented.")
    return characters


# ===========================================================================
# Combined summary visualization
# ===========================================================================

def save_summary_figure(preprocessor: ImagePreprocessor,
                        binary_image: np.ndarray,
                        characters: list) -> None:
    """
    Produce a single-page summary figure combining preprocessing and segmentation.

    Layout (3 rows):
        Row 1 — Five key preprocessing steps side-by-side.
        Row 2 — Binary image annotated with colored bounding boxes.
        Row 3 — Individual 32×32 character crops (up to 10).

    Args:
        preprocessor: ImagePreprocessor instance (has pipeline_steps populated).
        binary_image: uint8 binary output from Module 1.
        characters:   Character dict list from Module 2.
    """
    if not characters:
        print("[Summary] Skipping summary figure (no characters found).")
        return

    print("\n[Summary] Building stage1_summary.png …")

    fig = plt.figure(figsize=(22, 13))
    fig.suptitle('Stage 1 Complete Pipeline Results  |  AI Handwritten Math Equation Solver',
                 fontsize=15, fontweight='bold', y=0.98)

    steps      = list(preprocessor.pipeline_steps.items())
    step_indices = [0, 1, 3, 5, 6]     # original, gray, binary, opened, closed
    row1_labels  = ['1. Original', '2. Grayscale', '4. Otsu Binary',
                    '6. Opened', '7. Closed']

    # ── Row 1: key preprocessing steps ──────────────────────────────────
    for col, (step_idx, label) in enumerate(zip(step_indices, row1_labels)):
        if step_idx >= len(steps):
            continue
        _, img = steps[step_idx]
        ax = fig.add_subplot(3, 5, col + 1)

        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            vmax = 1.0 if img.max() <= 1.0 else 255
            ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)

        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.axis('off')

    # ── Row 2: bounding-box annotation ───────────────────────────────────
    segmenter = CharacterSegmenter()
    annotated = segmenter.visualize_segmentation(binary_image, characters)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax2.set_title(
        f'Detected Characters: {len(characters)} found — '
        'Green=normal  Red=superscript  Blue=subscript',
        fontsize=10
    )
    ax2.axis('off')

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=(0, 200/255, 0),     label='Normal'),
        mpatches.Patch(color=(220/255, 0, 0),     label='Superscript'),
        mpatches.Patch(color=(0, 0, 220/255),     label='Subscript'),
    ]
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=8)

    # ── Row 3: individual 32×32 crops ────────────────────────────────────
    n_show = min(len(characters), 10)
    for i in range(n_show):
        ax = fig.add_subplot(3, 10, 21 + i)
        ax.imshow(characters[i]['image'], cmap='gray', vmin=0, vmax=255)
        pos_short = characters[i]['position_type'][0].upper()  # N / S / B
        ax.set_title(f"#{i} {pos_short}", fontsize=7)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    summary_path = OUTPUT_DIR / 'stage1_summary.png'
    fig.savefig(str(summary_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Summary] Saved → {summary_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> list:
    """
    Run the complete Stage 1 test suite.

    Returns:
        The list of character dicts found by the segmenter.
    """
    print("=" * 60)
    print("STAGE 1 PIPELINE TEST")
    print("AI Handwritten Math Equation Solver")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {OUTPUT_DIR.resolve()}")

    # ── Determine input image ─────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"Error: '{image_path}' not found.")
            sys.exit(1)
        print(f"Using provided image: {image_path}")
    else:
        synthetic_path = PROJECT_ROOT / 'data' / 'raw' / 'test_equation.png'
        image_path     = create_test_equation_image(synthetic_path)

    # ── Module 1: Preprocessing ────────────────────────────────────────
    preprocessor, normalized, binary = test_preprocessor(image_path)

    if binary is None:
        print("Fatal: Preprocessing produced no binary image.")
        sys.exit(1)

    # ── Module 2: Segmentation ─────────────────────────────────────────
    characters = test_segmenter(binary)

    # ── Combined summary ───────────────────────────────────────────────
    save_summary_figure(preprocessor, binary, characters)

    # ── Final report ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 TEST RESULTS")
    print("=" * 60)
    print(f"Characters detected : {len(characters)}")
    print(f"\nSaved outputs:")
    for fname in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  {fname.relative_to(PROJECT_ROOT)}")

    if len(characters) > 0:
        print("\nStatus : SUCCESS — Stage 1 is ready for Stage 2 (CNN training).")
    else:
        print("\nStatus : WARNING — No characters detected.")
        print("  Suggestions:")
        print("  • Lower min_contour_area in CharacterSegmenter (currently 50)")
        print("  • Check preprocessing_pipeline.png for unexpected binary output")
        print("  • Try a higher-contrast input image")

    return characters


if __name__ == '__main__':
    main()
