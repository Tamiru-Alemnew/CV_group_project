import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


class Segmenter:

    _POSITION_COLORS = {
        'normal':      (0, 200, 0),
        'superscript': (0, 0, 220),
        'subscript':   (220, 0, 0),
    }

    def __init__(self,
                 min_contour_area: int = 30,
                 padding: int = 5,
                 target_size: tuple = None):
        self.min_contour_area = min_contour_area
        self.padding          = padding
        self.target_size      = target_size or config.IMAGE_SIZE

    def find_contours(self, binary_image: np.ndarray) -> list:
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def filter_contours(self, contours: list) -> list:
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

    def get_bounding_boxes(self, contours: list) -> list:
        return [cv2.boundingRect(c) for c in contours]

    def sort_left_to_right(self, boxes: list) -> list:
        return sorted(boxes, key=lambda b: b[0])

    def merge_compound_boxes(self, boxes: list) -> list:
        """Merge two-bar symbols (e.g. '=') whose contours are detected separately."""
        if len(boxes) < 2:
            return boxes

        avg_h = float(np.mean([h for (_, _, _, h) in boxes])) if boxes else 1.0
        merged = []
        used   = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            xi, yi, wi, hi = boxes[i]
            best_j   = -1
            best_gap = float('inf')

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                xj, yj, wj, hj = boxes[j]

                overlap_x = min(xi + wi, xj + wj) - max(xi, xj)
                min_w     = min(wi, wj)
                if min_w == 0 or overlap_x / min_w < 0.50:
                    continue

                gap_y = max(yi, yj) - min(yi + hi, yj + hj)
                if gap_y > avg_h * 0.40:
                    continue

                thin_i = hi < wi * 0.35
                thin_j = hj < wj * 0.35
                if not (thin_i or thin_j):
                    continue

                if gap_y < best_gap:
                    best_gap = gap_y
                    best_j   = j

            if best_j >= 0:
                xj, yj, wj, hj = boxes[best_j]
                x_new = min(xi, xj)
                y_new = min(yi, yj)
                x2    = max(xi + wi, xj + wj)
                y2    = max(yi + hi, yj + hj)
                merged.append((x_new, y_new, x2 - x_new, y2 - y_new))
                used[i] = used[best_j] = True
            else:
                merged.append(boxes[i])
                used[i] = True

        return merged

    def detect_position_type(self, boxes: list) -> list:
        """Classify each character as normal, superscript, or subscript by vertical position and size."""
        if not boxes:
            return []

        centers_y  = [y + h / 2.0 for (x, y, w, h) in boxes]
        heights    = [h             for (x, y, w, h) in boxes]
        avg_cy     = float(np.mean(centers_y))
        avg_h      = float(np.mean(heights))
        pos_thresh  = avg_h * 0.60
        size_thresh = avg_h * 0.65

        types = []
        for (cy, h) in zip(centers_y, heights):
            if cy < avg_cy - pos_thresh and h < size_thresh:
                types.append('superscript')
            elif cy > avg_cy + pos_thresh and h < size_thresh:
                types.append('subscript')
            else:
                types.append('normal')
        return types

    def crop_and_resize(self, image: np.ndarray, box: tuple) -> np.ndarray:
        x, y, w, h = box
        ih, iw = image.shape[:2]

        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(iw, x + w + self.padding)
        y2 = min(ih, y + h + self.padding)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((1, 1), dtype=np.uint8)

        # Preserve aspect ratio, centred on black canvas (matches EMNIST normalisation)
        tw, th = self.target_size
        ch_h, ch_w = crop.shape[:2]
        scale  = min(tw / ch_w, th / ch_h)
        new_w  = max(1, int(ch_w * scale))
        new_h  = max(1, int(ch_h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((th, tw), dtype=np.uint8)
        off_x  = (tw - new_w) // 2
        off_y  = (th - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def segment(self, binary_image: np.ndarray) -> list[dict]:
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)

        contours       = self.find_contours(binary_image)
        valid_contours = self.filter_contours(contours)

        if not valid_contours:
            print("[Segmenter] Warning: no valid character contours found.")
            return []

        boxes        = self.get_bounding_boxes(valid_contours)
        sorted_boxes = self.sort_left_to_right(boxes)
        sorted_boxes = self.merge_compound_boxes(sorted_boxes)
        pos_types    = self.detect_position_type(sorted_boxes)

        characters = []
        for idx, (box, pt) in enumerate(zip(sorted_boxes, pos_types)):
            x, y, w, h = box
            characters.append({
                'image':         self.crop_and_resize(binary_image, box),
                'bbox':          {'x': x, 'y': y, 'width': w, 'height': h},
                'index':         idx,
                'position_type': pt,
                'center':        {'x_center': x + w // 2, 'y_center': y + h // 2},
            })
        return characters

    def visualize_segmentation(self,
                                source_image: np.ndarray,
                                characters: list,
                                save_path: str = None) -> np.ndarray:
        if len(source_image.shape) == 2:
            annotated = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = source_image.copy()

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.45
        thick  = 1

        for ch in characters:
            b  = ch['bbox']
            x, y, w, h = b['x'], b['y'], b['width'], b['height']
            color = self._POSITION_COLORS.get(ch['position_type'], (0, 200, 0))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            label = str(ch['index'])
            (lw, lh), _ = cv2.getTextSize(label, font, fscale, thick)
            cv2.rectangle(annotated, (x, max(0, y - lh - 5)), (x + lw + 2, y), color, -1)
            cv2.putText(annotated, label, (x + 1, max(lh, y - 2)),
                        font, fscale, (255, 255, 255), thick, cv2.LINE_AA)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated)
            print(f"[Segmenter] Segmentation image saved → {save_path}")

        return annotated

    def visualize_crops(self,
                        characters: list,
                        save_path: str = None,
                        max_per_row: int = 10) -> plt.Figure | None:
        if not characters:
            return None

        n      = len(characters)
        n_cols = min(n, max_per_row)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 1.6, n_rows * 2.0))
        fig.suptitle(f'Segmented Character Crops  ({n} found)',
                     fontsize=11, fontweight='bold')

        if n_rows == 1 and n_cols == 1:
            axes_grid = [[axes]]
        elif n_rows == 1:
            axes_grid = [list(axes)]
        elif n_cols == 1:
            axes_grid = [[ax] for ax in axes]
        else:
            axes_grid = [list(row) for row in axes]

        abbr = {'normal': 'N', 'superscript': 'SUP', 'subscript': 'SUB'}
        for i, ch in enumerate(characters):
            r, c = divmod(i, n_cols)
            ax = axes_grid[r][c]
            ax.imshow(ch['image'], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"#{ch['index']} {abbr.get(ch['position_type'], '?')}",
                         fontsize=7)
            ax.axis('off')

        for i in range(n, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes_grid[r][c].axis('off')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Segmenter] Crop grid saved → {save_path}")

        return fig
