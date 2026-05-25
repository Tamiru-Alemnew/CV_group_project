"""
Stage 1 - Module 1: Image Preprocessor

Takes a raw photo of a handwritten equation and applies a pipeline of
classical computer vision techniques to produce a clean binary image
ready for character segmentation and CNN recognition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config


class ImagePreprocessor:
    """
    Cleans a raw handwritten equation photo through an ordered pipeline:
    grayscale → blur → Otsu threshold → deskew →
    morphological opening → closing → normalization.
    """

    def __init__(self, blur_kernel_size=(5, 5), morph_kernel_size=(2, 2)):
        """
        Args:
            blur_kernel_size: Odd-integer (w, h) for Gaussian blur.
            morph_kernel_size: (w, h) for morphological structuring element.
        """
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        # Stores every intermediate result so visualize_steps() can show all stages
        self.pipeline_steps: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Individual pipeline steps
    # ──────────────────────────────────────────────────────────────────────────

    def load_image(self, image_path) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            image_path: str or Path to the image file.

        Returns:
            BGR numpy array.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError: OpenCV cannot decode the file.
        """
        image_path = str(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot decode '{image_path}'. Check it is a valid image.")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to single-channel grayscale.

        Why: Colour information is irrelevant for pen-on-paper equations.
        Collapsing 3 channels to 1 reduces compute and removes colour-based noise
        (pen tint, paper yellowing) without losing any structural information.
        """
        if len(image.shape) == 2:
            return image
        # Luminance-weighted combination preserves perceptual brightness
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to suppress high-frequency noise.

        Why: Camera sensors and paper grain produce rapid pixel-intensity
        fluctuations (high-frequency noise). Gaussian blur is a low-pass
        filter — it averages each pixel with its neighbours weighted by a
        bell-curve, smoothing out noise while preserving slower changes (ink
        strokes). Without blur, Otsu thresholding would classify texture pixels
        as ink, badly corrupting the binary result.
        """
        # sigma=0 → OpenCV auto-computes optimal sigma from kernel size
        return cv2.GaussianBlur(image, self.blur_kernel_size, sigmaX=0)

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Binarise the image using Otsu's automatic global thresholding.

        Why: We need a pure black-and-white image with ink=white, paper=black.
        A fixed threshold fails when lighting varies across photos. Otsu's method
        minimises the weighted intra-class variance of the foreground / background
        pixel distributions, automatically finding the optimal threshold for each
        image regardless of lighting conditions.

        THRESH_BINARY_INV inverts so ink→255 (white), paper→0 (black), matching
        the MNIST training data convention and the requirement of findContours.
        """
        _, binary = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate the image to correct for handwriting written at a slight angle.

        Why: A 3-5° tilt causes characters to appear at different vertical heights,
        confusing left-to-right segmentation order and the superscript detector.
        Deskewing aligns the dominant text direction with the horizontal axis.

        Algorithm:
          1. Collect all ink pixel coordinates.
          2. cv2.minAreaRect finds the smallest enclosing rotated rectangle,
             whose long axis aligns with the dominant text direction.
          3. Extract and normalise the angle.
          4. Apply inverse rotation to bring text to horizontal.
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 5:
            return image  # Too few pixels to detect angle reliably

        angle = cv2.minAreaRect(coords)[-1]

        # Normalise minAreaRect's [-90, 0) convention to a meaningful rotation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Skip corrections smaller than 0.5° — likely measurement noise
        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0   # New border pixels = black (background)
        )

    def apply_morphological_opening(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small isolated noise specks via morphological opening.

        Why: Opening = erosion → dilation. Erosion shrinks all white regions;
        tiny specks disappear entirely while character strokes survive (they are
        thick enough). Dilation restores the surviving strokes to their original
        size. Net effect: small noise dots gone, characters preserved.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def apply_morphological_closing(self, image: np.ndarray) -> np.ndarray:
        """
        Fill small gaps within character strokes via morphological closing.

        Why: Closing = dilation → erosion. Dilation expands white regions,
        bridging micro-gaps caused by uneven ink coverage. Erosion shrinks them
        back. Net effect: holes inside characters (e.g., inside '0', '8')
        are filled; overall shape is unchanged.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Scale pixel values from [0, 255] to [0.0, 1.0].

        Why: CNNs train significantly faster and more stably with normalised
        inputs. Values in [0, 1] prevent activation saturation and match the
        expected input distribution of models trained on standard datasets.
        """
        return image.astype(np.float32) / 255.0

    # ──────────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def preprocess(self, image_path) -> np.ndarray:
        """
        Execute the complete preprocessing pipeline.

        Runs all 8 operations in order and stores each intermediate result
        in self.pipeline_steps for later visualisation.

        Args:
            image_path: Path to raw input image.

        Returns:
            float32 numpy array (H × W) with values in [0.0, 1.0].
        """
        self.pipeline_steps = {}

        original = self.load_image(image_path)
        self.pipeline_steps['1_original'] = original.copy()

        gray = self.to_grayscale(original)
        self.pipeline_steps['2_grayscale'] = gray.copy()

        blurred = self.apply_gaussian_blur(gray)
        self.pipeline_steps['3_blurred'] = blurred.copy()

        binary = self.apply_otsu_threshold(blurred)
        self.pipeline_steps['4_binary_otsu'] = binary.copy()

        deskewed = self.deskew(binary)
        self.pipeline_steps['5_deskewed'] = deskewed.copy()

        opened = self.apply_morphological_opening(deskewed)
        self.pipeline_steps['6_opened'] = opened.copy()

        closed = self.apply_morphological_closing(opened)
        self.pipeline_steps['7_closed'] = closed.copy()

        normalized = self.normalize(closed)
        self.pipeline_steps['8_normalized'] = normalized.copy()

        return normalized

    def get_binary_image(self) -> np.ndarray | None:
        """
        Return the cleaned binary uint8 image (values 0 or 255).

        The segmenter needs integer pixel values for contour detection —
        it cannot use the float normalised image.
        """
        return self.pipeline_steps.get('7_closed', None)

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def visualize_steps(self, save_path: str = None, figsize: tuple = (20, 10)):
        """
        Display every preprocessing step side-by-side in one figure.

        This is the primary Classical CV demonstration visualisation. Each
        panel shows one intermediate image with a title explaining what that
        operation does and WHY it matters for reading handwritten characters.

        Args:
            save_path: If provided, save the figure to this path.
            figsize:   (width, height) of the figure in inches.

        Returns:
            matplotlib Figure object.

        Raises:
            RuntimeError: preprocess() has not been called yet.
        """
        if not self.pipeline_steps:
            raise RuntimeError("Call preprocess() before visualize_steps().")

        titles = [
            '1. Original\nRaw camera photo',
            '2. Grayscale\nColour removed — only intensity needed',
            '3. Gaussian Blur\nCamera noise & paper texture suppressed',
            '4. Otsu Threshold\nAdaptive binarisation — ink=white, paper=black',
            '5. Deskewed\nText angle detected & corrected to horizontal',
            '6. Morph. Opening\nErosion+Dilation: isolated noise dots removed',
            '7. Morph. Closing\nDilation+Erosion: gaps inside characters filled',
            '8. Normalised [0–1]\nPixel values scaled for CNN input layer',
        ]

        steps = list(self.pipeline_steps.items())
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle('Image Preprocessing Pipeline  |  Stage 1 · Classical CV',
                     fontsize=14, fontweight='bold', y=0.99)
        axes = axes.flatten()

        for i, ((_, img), title) in enumerate(zip(steps, titles)):
            ax = axes[i]
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                vmax = 1.0 if img.max() <= 1.0 else 255
                ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8.5, pad=5)
            ax.axis('off')
            if len(img.shape) == 2:
                ax.set_xlabel(
                    f'range [{img.min():.0f},{img.max():.0f}]  '
                    f'{img.shape[1]}×{img.shape[0]}',
                    fontsize=7, color='#666'
                )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Preprocessor] Saved → {save_path}")

        return fig
