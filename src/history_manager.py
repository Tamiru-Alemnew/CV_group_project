"""
Stage 4 - Module 3: History Manager

Persists every solved equation to disk as a JSON file and provides
statistics, CSV export, and in-memory querying over the history.
"""

import sys, json, csv
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _make_serializable(obj):
    """
    Recursively convert obj to a JSON-serialisable Python primitive.

    Handles numpy scalars, SymPy types, and nested structures.
    """
    # Numpy scalar types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    # SymPy types → string
    try:
        import sympy
        if isinstance(obj, sympy.Basic):
            return str(obj)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


class HistoryManager:
    """
    Tracks all equations solved by the system in a JSON file on disk.

    Schema of each history record:
        image_path      – str (source image or 'manual')
        equation        – str (recognised equation string)
        solution        – dict (full solver result, JSON-safe)
        confidence      – float (overall recognition confidence)
        timestamp       – str ISO 8601
        equation_type   – str ('arithmetic' | 'linear' | 'quadratic' | 'expression')
    """

    def __init__(self, history_path: str | Path = None):
        """
        Args:
            history_path: Path to the JSON history file.
                          Defaults to config.HISTORY_JSON.
        """
        self.history_path = Path(history_path) if history_path else config.HISTORY_JSON
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self.load_history()

    # ─────────────────────────────────────────────────────────────────────────

    def load_history(self) -> list[dict]:
        """
        Load existing history from the JSON file into memory.

        Returns:
            List of history record dicts (empty list if file absent).
        """
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"[History] Could not parse {self.history_path}; starting fresh.")
                self._records = []
        else:
            self._records = []
        return self._records

    def save_history(self) -> None:
        """Persist the in-memory records list back to the JSON file."""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(_make_serializable(self._records), f,
                          indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[History] WARNING: could not save history — {e}")

    def add_record(self, record: dict) -> None:
        """
        Append a new solved-equation record and immediately save to disk.

        Args:
            record: Dict with at minimum these keys:
                image_path, equation, solution, confidence, equation_type.
                A timestamp is added automatically if absent.
        """
        record.setdefault('timestamp', datetime.now().isoformat())
        self._records.append(_make_serializable(record))
        self.save_history()
        print(f"[History] Saved record #{len(self._records)}: "
              f"'{record.get('equation', '?')}' "
              f"({record.get('equation_type', '?')})")

    # ─────────────────────────────────────────────────────────────────────────

    def get_stats_report(self) -> str:
        """
        Return a formatted multi-line statistics string.

        Shows:
          • Total equations solved.
          • Count breakdown by equation type.
          • Average confidence score.
          • Most frequent equation types.
        """
        if not self._records:
            return "No equations solved yet."

        n = len(self._records)
        types: dict[str, int] = {}
        confs = []

        for r in self._records:
            t = r.get('equation_type', 'unknown')
            types[t] = types.get(t, 0) + 1
            c = r.get('confidence')
            if c is not None:
                try:
                    confs.append(float(c))
                except (TypeError, ValueError):
                    pass

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        ranked   = sorted(types.items(), key=lambda kv: -kv[1])

        lines = [
            "╔══════════════════════════════════════╗",
            "║         HISTORY STATISTICS           ║",
            "╠══════════════════════════════════════╣",
            f"║  Total equations solved : {n:<11}║",
            f"║  Average confidence     : {avg_conf*100:<9.1f}% ║",
            "╠══════════════════════════════════════╣",
            "║  By equation type:                   ║",
        ]
        for t, cnt in ranked:
            pct = cnt / n * 100
            lines.append(f"║    {t:<14}: {cnt:>4}  ({pct:>5.1f}%)       ║")
        lines.append("╚══════════════════════════════════════╝")
        return '\n'.join(lines)

    def export_csv(self, filepath: str | Path = None) -> Path:
        """
        Export complete history as a CSV file for download.

        Columns: timestamp, equation, equation_type, confidence, answer, success.

        Args:
            filepath: Output path. Defaults to config.OUTPUT_DIR/history_export.csv.

        Returns:
            Path to the written CSV file.
        """
        if filepath is None:
            filepath = config.OUTPUT_DIR / 'history_export.csv'
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ['timestamp', 'equation', 'equation_type',
                      'confidence', 'answer', 'success']

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in self._records:
                sol = r.get('solution', {})
                writer.writerow({
                    'timestamp':     r.get('timestamp', ''),
                    'equation':      r.get('equation', ''),
                    'equation_type': r.get('equation_type', ''),
                    'confidence':    r.get('confidence', ''),
                    'answer':        sol.get('answer_str', '') if isinstance(sol, dict) else '',
                    'success':       sol.get('success', '') if isinstance(sol, dict) else '',
                })

        print(f"[History] CSV exported → {filepath}")
        return filepath

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def records(self) -> list[dict]:
        """Read-only access to the in-memory records list."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
