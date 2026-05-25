"""
Stage 1 - Module 2: Character Segmenter

Takes a clean preprocessed binary image (white ink on black background)
and splits it into individual 32×32 character crops with full metadata.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


class Segmenter:
    """
    Detects, extracts, and classifies every symbol in a preprocessed binary
    equation image.

    Pipeline:
        find_contours → filter → bounding_boxes → sort_L2R →
        position_type → crop_and_resize
    """

    _POSITION_COLORS = {
        'normal':      (0, 200, 0),   # Green
        'superscript': (0, 0, 220),   # Red   (raised → exponent)
        'subscript':   (220, 0, 0),   # Blue  (lowered)
    }

    def __init__(self,
                 min_contour_area: int = 30,
                 padding: int = 5,
                 target_size: tuple = None):
        """
        Args:
            min_contour_area: Minimum pixel area to be treated as a real character.
            padding:          Extra pixels added around each character crop.
            target_size:      Output (w, h); defaults to config.IMAGE_SIZE.
        """
        self.min_contour_area = min_contour_area
        self.padding          = padding
        self.target_size      = target_size or config.IMAGE_SIZE

    # ─────────────────────────────────────────────────────────────────────────

    def find_contours(self, binary_image: np.ndarray) -> list:
        """
        Trace boundaries of all connected white regions.

        Why RETR_EXTERNAL: retrieves only outermost contours, ignoring inner
        holes (e.g., inside '0'). This ensures each symbol produces exactly
        one contour rather than an outer ring plus an inner hole.

        Why CHAIN_APPROX_SIMPLE: compresses straight-line runs to two endpoints,
        saving memory without losing bounding-box accuracy.
        """
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def filter_contours(self, contours: list) -> list:
        """
        Discard contours smaller than min_contour_area.

        Why: Real characters occupy substantially more pixels than noise specks.
        Any region below the threshold is almost certainly not a symbol.
        """
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

    def get_bounding_boxes(self, contours: list) -> list:
        """
        Extract axis-aligned bounding rectangles from contours.

        Returns list of (x, y, w, h) tuples defining crop regions.
        """
        return [cv2.boundingRect(c) for c in contours]

    def sort_left_to_right(self, boxes: list) -> list:
        """
        Sort bounding boxes in reading order (left edge ascending).

        Why: findContours returns contours in raster-scan order, not reading
        order. Sorting by x-coordinate gives correct left-to-right sequence
        required for equation parsing.
        """
        return sorted(boxes, key=lambda b: b[0])

    def merge_compound_boxes(self, boxes: list) -> list:
        """
        Merge pairs of bounding boxes that form a single symbol split across
        two contours — the primary case being the '=' sign, whose two horizontal
        bars are detected as separate contours by findContours.

        Merge criteria (both must hold):
          1. Horizontal overlap ≥ 50 % of the narrower box's width.
          2. Vertical gap between the boxes < avg character height × 0.4.
          3. At least one of the two boxes is 'thin' (height < 0.35 × its width).

        The merged box is the axis-aligned union of the two source boxes.
        """
        if len(boxes) < 2:
            return boxes

        avg_h = float(np.mean([h for (_, _, _, h) in boxes])) if boxes else 1.0

        merged   = []
        used     = [False] * len(boxes)

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

                # 1. Horizontal overlap
                overlap_x = min(xi + wi, xj + wj) - max(xi, xj)
                min_w     = min(wi, wj)
                if min_w == 0 or overlap_x / min_w < 0.50:
                    continue

                # 2. Vertical gap
                gap_y = max(yi, yj) - min(yi + hi, yj + hj)
                if gap_y > avg_h * 0.40:
                    continue

                # 3. At least one thin box
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
        """
        Classify each character as 'normal', 'superscript', or 'subscript'.

        Why: Exponents (x²) are written raised above the baseline. Detecting
        vertical position is essential for correct equation parsing.

        A character is a superscript only when BOTH conditions hold:
          1. Its y-center is more than 60% of avg-height ABOVE the mean   (position)
          2. Its height is less than 65% of the average character height   (size)

        Requiring both conditions avoids false positives: in real handwriting
        a normal digit may sit slightly high, but it is never also much smaller
        than the surrounding characters. True exponents (x²) satisfy both —
        they are written raised AND are visibly smaller.
        """
        if not boxes:
            return []

        centers_y  = [y + h / 2.0 for (x, y, w, h) in boxes]
        heights    = [h             for (x, y, w, h) in boxes]
        avg_cy     = float(np.mean(centers_y))
        avg_h      = float(np.mean(heights))
        pos_thresh = avg_h * 0.60   # must be this far above mean
        size_thresh = avg_h * 0.65  # must also be this much smaller than avg

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
        """
        Crop a character region with padding and resize to target_size.

        Why padding: Without padding a stroke may touch the crop boundary,
        differing from MNIST images which always have whitespace around them.
        5 pixels of padding recreates that natural margin.

        Why INTER_AREA: pixel-area resampling minimises aliasing when
        downscaling by averaging nearby pixels, preserving stroke detail.
        """
        x, y, w, h = box
        ih, iw = image.shape[:2]

        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(iw, x + w + self.padding)
        y2 = min(ih, y + h + self.padding)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((1, 1), dtype=np.uint8)

        # Preserve aspect ratio so narrow characters (e.g. '1', '/', '-') keep
        # their natural proportions.  EMNIST normalises digits the same way:
        # the digit is scaled to fit inside the frame, then centred on black.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def segment(self, binary_image: np.ndarray) -> list[dict]:
        """
        Run the complete segmentation pipeline.

        Args:
            binary_image: uint8 or float32 binary image, white ink on black.

        Returns:
            List of character dicts (sorted left-to-right), each containing:
              'image'         – 32×32 uint8 crop
              'bbox'          – dict {x, y, width, height}
              'index'         – 0-based sequence position
              'position_type' – 'normal' | 'superscript' | 'subscript'
              'center'        – dict {x_center, y_center}
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_segmentation(self,
                                source_image: np.ndarray,
                                characters: list,
                                save_path: str = None) -> np.ndarray:
        """
        Draw green bounding boxes with index labels on the source image.

        Box colours:
            Green  → normal
            Red    → superscript
            Blue   → subscript

        Args:
            source_image: Grayscale or BGR image to annotate (not modified in place).
            characters:   Character dicts from segment().
            save_path:    Optional save path.

        Returns:
            BGR annotated image.
        """
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
        """
        Show all segmented character crops in a grid.

        Args:
            characters:  Character dicts from segment().
            save_path:   Optional save path.
            max_per_row: Maximum columns per row.

        Returns:
            matplotlib Figure or None if no characters.
        """
        if not characters:
            return None

        n      = len(characters)
        n_cols = min(n, max_per_row)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 1.6, n_rows * 2.0))
        fig.suptitle(f'Segmented Character Crops  ({n} found)  32×32 px each',
                     fontsize=11, fontweight='bold')

        # Normalise axes to 2-D list
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
