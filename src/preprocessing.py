import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config


class ImagePreprocessor:

    def __init__(self, blur_kernel_size=(5, 5), morph_kernel_size=(2, 2)):
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.pipeline_steps: dict = {}

    def load_image(self, image_path) -> np.ndarray:
        image_path = str(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot decode '{image_path}'.")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.blur_kernel_size, sigmaX=0)

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        # THRESH_BINARY_INV: ink→255, paper→0 to match MNIST convention
        _, binary = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def deskew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 5:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    def apply_morphological_opening(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def apply_morphological_closing(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    def preprocess(self, image_path) -> np.ndarray:
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
        return self.pipeline_steps.get('7_closed', None)

    def visualize_steps(self, save_path: str = None, figsize: tuple = (20, 10)):
        if not self.pipeline_steps:
            raise RuntimeError("Call preprocess() before visualize_steps().")

        titles = [
            '1. Original\nRaw camera photo',
            '2. Grayscale\nColour removed',
            '3. Gaussian Blur\nNoise suppressed',
            '4. Otsu Threshold\nAdaptive binarisation',
            '5. Deskewed\nAngle corrected',
            '6. Morph. Opening\nNoise dots removed',
            '7. Morph. Closing\nCharacter gaps filled',
            '8. Normalised [0–1]\nScaled for CNN input',
        ]

        steps = list(self.pipeline_steps.items())
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle('Image Preprocessing Pipeline',
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
