# AI Handwritten Math Equation Solver

An end-to-end Computer Vision system that reads a photo or hand-drawn image of a handwritten math equation, recognises each symbol with a trained CNN, and returns a step-by-step symbolic solution via an interactive web interface.

---

## Architecture

```
Input Image / Canvas Drawing
         │
         ▼
┌─────────────────────┐
│    Preprocessing    │  Grayscale · Gaussian Blur · Otsu Threshold
│    (Classical CV)   │  Deskew · Morphological Clean · Normalise
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Segmentation     │  findContours · Filter noise · Sort L→R
│    (Classical CV)   │  Pad · Resize 32×32 · Detect superscripts
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Symbol Recognition │  CNN v3  or  MobileNetV2 (selectable)
│   (Deep Learning)   │  18-class softmax · Confidence threshold
│                     │  Grad-CAM attention heatmaps
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Equation Parser   │  Digit merge · Implicit × · Exponents
│                     │  Negative handling · SymPy validation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    SymPy Solver     │  Arithmetic · Linear · Quadratic
│   (Symbolic Math)   │  Step-by-step pedagogical output
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Streamlit App     │  Draw · Upload · Compare models
│     (Web UI)        │  History · Grad-CAM · Model metrics
└─────────────────────┘
```

---

## Models

| Model | Val Accuracy | Test Accuracy | Classes | Input |
|-------|-------------|--------------|---------|-------|
| **CNN v3** | **98.82%** | **98.83%** | 18 | 32×32 grayscale |
| MobileNetV2 | 95.46% | 95.59% | 18 | 32×32 → RGB |

18 symbol classes: digits `0–9`, operators `+ − × ÷ = x ( )`

Training: 30 epochs, batch size 128, class-weighted loss, ReduceLROnPlateau.

### Training Data

Total dataset: **314,445 samples** — train 220,111 / val 47,167 / test 47,167

| Source | Samples |
|--------|---------|
| EMNIST Digits — handwritten digits 0–9 | 280,000 |
| EMNIST Letters — handwritten x/X | 3,437 |
| HASYv2 — handwritten operators + − × ÷ / | 2,172 |
| Kaggle Math Symbols — digits and operators | 8,836 |
| Synthetic (PIL + augmentation) — operators and parentheses | 20,000 |

---

## Project Structure

```
math_equation_solver/
├── app/
│   └── app.py                  Streamlit web app (4 tabs)
├── src/
│   ├── preprocessing.py        Classical CV preprocessing
│   ├── segmentation.py         Contour detection & character extraction
│   ├── model.py                CNN v3 architecture
│   ├── gradcam.py              Grad-CAM (CNN + MobileNetV2)
│   ├── recognize.py            Inference wrapper
│   ├── equation_parser.py      Symbol sequence → equation string
│   ├── math_solver.py          SymPy symbolic solver
│   ├── step_formatter.py       Solution formatter
│   ├── history_manager.py      Session history
│   ├── pipeline.py             End-to-end integrator
│   └── data_preparation.py     Dataset builder
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── demo/
│   └── evaluation.py           Accuracy & speed benchmark
├── tests/
│   ├── test_pipeline.py
│   └── test_stage1.py
├── config.py                   Central configuration
└── main.py                     CLI entry point
```

---

## Installation

```bash
git clone https://github.com/Tamiru-Alemnew/CV_group_project.git
cd CV_group_project

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place trained model files in `models/saved_models/`:
- `best_cnn_model.h5`
- `best_mobilenet_model.h5` *(optional — enables side-by-side comparison)*

---

## Usage

```bash
streamlit run app/app.py                        # web app
python main.py --mode image --image photo.jpg   # single image via CLI
python demo/evaluation.py                       # accuracy & speed benchmark
```

---

## References

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML.
- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE TSM.
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [HASYv2 Dataset](https://zenodo.org/record/259444)
- [Handwritten Math Symbols (Kaggle)](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols)
