"""
Stage 1 - Module 3: Dataset Preparation

Combines MNIST digits with custom operator symbol images into a 16-class
training dataset, applies augmentation configuration, splits 70/15/15,
and saves prepared arrays to data/processed/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import config


class DataPreparator:
    """
    Orchestrates the full dataset preparation pipeline:
    MNIST load → resize → operator images → combine →
    normalize → stratified split → save → visualize.
    """

    def __init__(self,
                 symbols_dir: str | Path = None,
                 processed_dir: str | Path = None):
        self.symbols_dir   = Path(symbols_dir)   if symbols_dir   else config.SYMBOLS_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – Load MNIST
    # ─────────────────────────────────────────────────────────────────────────

    def load_mnist(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the full MNIST dataset (train + test combined, 70 000 images).

        Why MNIST: provides 70 000 diverse handwriting samples for digits 0–9
        from hundreds of writers, giving excellent class coverage without manual
        collection. We merge the original splits and re-split ourselves to ensure
        consistent class balance across our train/val/test sets.

        Uses TensorFlow/Keras exclusively for this data-loading step.
        """
        try:
            from tensorflow.keras.datasets import mnist
        except ImportError as e:
            raise ImportError("TensorFlow required for MNIST. "
                              "pip install tensorflow") from e

        print("[DataPrep] Loading MNIST …")
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        images = np.concatenate([x_tr, x_te], axis=0)
        labels = np.concatenate([y_tr, y_te], axis=0)
        print(f"[DataPrep] MNIST: {len(images):,} images  {images.shape[1]}×{images.shape[2]}")
        return images, labels

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – Resize MNIST 28×28 → 32×32
    # ─────────────────────────────────────────────────────────────────────────

    def resize_mnist(self, images: np.ndarray) -> np.ndarray:
        """
        Upscale MNIST images to config.IMAGE_SIZE (32×32).

        Why: Our segmenter outputs 32×32 crops; all CNN inputs must match.
        INTER_CUBIC bicubic interpolation gives smoother upscaling than bilinear
        by considering a 4×4 neighbourhood, preserving stroke continuity.
        """
        w, h = config.IMAGE_SIZE
        print(f"[DataPrep] Resizing MNIST: 28×28 → {w}×{h} …")
        resized = np.empty((len(images), h, w), dtype=np.uint8)
        for i, img in enumerate(images):
            resized[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        return resized

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – Load custom operator symbol images
    # ─────────────────────────────────────────────────────────────────────────

    def load_symbol_images(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load math operator images from the symbols directory.

        Folder → class mapping (from config.SYMBOL_FOLDER_MAP):
            plus/     → 10 (+)
            minus/    → 11 (-)
            variable/ → 12 (x)
            equals/   → 13 (=)
            multiply/ → 14 (*)
            divide/   → 15 (/)

        All images are converted to greyscale, resized to 32×32, and normalised
        to the white-ink-on-black convention that matches MNIST and our segmenter.
        """
        w, h = config.IMAGE_SIZE
        all_images: list[np.ndarray] = []
        all_labels: list[int]        = []

        for folder_name, class_idx in config.SYMBOL_FOLDER_MAP.items():
            folder = self.symbols_dir / folder_name
            if not folder.exists():
                print(f"[DataPrep] WARNING: '{folder}' missing — "
                      f"skipping class {class_idx} ('{config.CLASS_MAP[class_idx]}')")
                continue

            files = (list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
                     + list(folder.glob('*.jpeg')) + list(folder.glob('*.bmp')))
            if not files:
                print(f"[DataPrep] WARNING: no images in '{folder}'")
                continue

            loaded = 0
            for fp in files:
                img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                # Enforce white-ink-on-black: invert if background is light
                if np.mean(img) > 127:
                    img = cv2.bitwise_not(img)
                all_images.append(img)
                all_labels.append(class_idx)
                loaded += 1

            print(f"[DataPrep] Class {class_idx} ('{config.CLASS_MAP[class_idx]}'): "
                  f"{loaded} images")

        if not all_images:
            print("[DataPrep] WARNING: no operator images — dataset will be digits only.")
            w, h = config.IMAGE_SIZE
            return (np.empty((0, h, w), dtype=np.uint8),
                    np.empty((0,), dtype=np.int64))

        return np.array(all_images, dtype=np.uint8), np.array(all_labels, dtype=np.int64)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – Combine
    # ─────────────────────────────────────────────────────────────────────────

    def combine_datasets(self,
                         mnist_images:   np.ndarray, mnist_labels:   np.ndarray,
                         symbol_images:  np.ndarray, symbol_labels:  np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """
        Concatenate MNIST digit data with custom operator images.
        """
        if len(symbol_images) == 0:
            return mnist_images, mnist_labels
        combined_images = np.concatenate([mnist_images, symbol_images], axis=0)
        combined_labels = np.concatenate([mnist_labels, symbol_labels], axis=0)
        print(f"[DataPrep] Combined: {len(combined_images):,} images, "
              f"{len(np.unique(combined_labels))} classes")
        return combined_images, combined_labels

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – Normalize
    # ─────────────────────────────────────────────────────────────────────────

    def normalize(self, images: np.ndarray) -> np.ndarray:
        """Scale pixel values to [0.0, 1.0] for CNN compatibility."""
        return images.astype(np.float32) / 255.0

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 – Stratified split 70 / 15 / 15
    # ─────────────────────────────────────────────────────────────────────────

    def split_dataset(self, images: np.ndarray, labels: np.ndarray) -> dict:
        """
        Stratified 70 / 15 / 15 train / val / test split.

        Why stratified: our classes are heavily imbalanced (~6 000 samples per
        digit vs potentially far fewer per operator). Stratification guarantees
        each split has the same class proportion as the full dataset, ensuring
        all classes are represented in validation and test sets.
        """
        print("[DataPrep] Splitting 70 / 15 / 15 (stratified) …")
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(
            images, labels, test_size=0.30, random_state=config.RANDOM_SEED,
            stratify=labels
        )
        x_val, x_te, y_val, y_te = train_test_split(
            x_tmp, y_tmp, test_size=0.50, random_state=config.RANDOM_SEED,
            stratify=y_tmp
        )
        splits = {
            'train': {'images': x_tr,  'labels': y_tr},
            'val':   {'images': x_val, 'labels': y_val},
            'test':  {'images': x_te,  'labels': y_te},
        }
        for name, d in splits.items():
            print(f"  {name:5s}: {len(d['labels']):7,} samples")
        return splits

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 – Save
    # ─────────────────────────────────────────────────────────────────────────

    def save_to_disk(self, splits: dict) -> None:
        """
        Persist split arrays as .npy files plus a JSON class map.

        Why .npy: numpy's native format loads in a single fast memory-mapped
        read — much faster than re-running the entire preparation pipeline.
        """
        print(f"[DataPrep] Saving to '{self.processed_dir}' …")
        for name, d in splits.items():
            np.save(self.processed_dir / f'{name}_images.npy', d['images'])
            np.save(self.processed_dir / f'{name}_labels.npy', d['labels'])
            print(f"  {name}: images {d['images'].shape}, labels {d['labels'].shape}")

        with open(self.processed_dir / 'class_labels.json', 'w') as f:
            json.dump({str(k): v for k, v in config.CLASS_MAP.items()}, f,
                      indent=2, ensure_ascii=False)
        print("[DataPrep] Done.")

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics & visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def print_statistics(self, splits: dict) -> None:
        """Print per-class sample counts for every split."""
        print("\n" + "=" * 62)
        print("DATASET STATISTICS")
        print("=" * 62)
        for name, d in splits.items():
            labels = d['labels']
            print(f"\n{name.upper()} — {len(labels):,} samples  "
                  f"shape {d['images'].shape}")
            unique, counts = np.unique(labels, return_counts=True)
            for cls, cnt in zip(unique, counts):
                sym = config.CLASS_MAP.get(int(cls), f'cls{cls}')
                bar = '█' * min(40, cnt // 150)
                print(f"  Class {int(cls):2d} '{sym}': {cnt:6,}  {bar}")
        print("=" * 62)

    def visualize_statistics(self, splits: dict,
                              save_path: str = None) -> plt.Figure:
        """Bar-chart of per-class sample counts across all splits."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Dataset Class Distribution  |  Stage 1 · Module 3',
                     fontsize=13, fontweight='bold')
        palette = ['steelblue', 'darkorange', 'mediumseagreen']
        syms    = [config.CLASS_MAP.get(i, str(i)) for i in range(config.NUM_CLASSES)]

        for ax, (name, d), color in zip(axes, splits.items(), palette):
            counts = np.zeros(config.NUM_CLASSES, dtype=int)
            for cls, cnt in zip(*np.unique(d['labels'], return_counts=True)):
                if int(cls) < config.NUM_CLASSES:
                    counts[int(cls)] = cnt
            bars = ax.bar(range(config.NUM_CLASSES), counts, color=color, alpha=0.75)
            ax.set_xticks(range(config.NUM_CLASSES))
            ax.set_xticklabels(syms, fontsize=8)
            ax.set_title(f"{name.capitalize()}  ({len(d['labels']):,})", fontweight='bold')
            ax.set_xlabel('Symbol Class')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(counts) * 0.01,
                            f'{cnt:,}', ha='center', va='bottom', fontsize=6)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[DataPrep] Stats chart → {save_path}")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Master pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def prepare(self, normalize: bool = True) -> dict:
        """
        Run the full dataset preparation pipeline end-to-end.

        Returns:
            Dict with 'train', 'val', 'test' splits ready for CNN training.
        """
        print("\n" + "=" * 62)
        print("STAGE 1 · MODULE 3: DATASET PREPARATION")
        print("=" * 62 + "\n")

        mnist_imgs, mnist_lbls = self.load_mnist()
        mnist_imgs             = self.resize_mnist(mnist_imgs)
        sym_imgs,   sym_lbls   = self.load_symbol_images()
        all_imgs, all_lbls     = self.combine_datasets(mnist_imgs, mnist_lbls,
                                                        sym_imgs,   sym_lbls)
        if normalize:
            all_imgs = self.normalize(all_imgs)

        splits = self.split_dataset(all_imgs, all_lbls)
        self.print_statistics(splits)
        self.save_to_disk(splits)
        return splits


if __name__ == '__main__':
    prep   = DataPreparator()
    splits = prep.prepare()
    fig    = prep.visualize_statistics(splits,
                 save_path=str(config.PROCESSED_DIR / 'stats.png'))
    plt.show()
