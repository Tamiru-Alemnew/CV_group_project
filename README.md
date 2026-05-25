# AI Handwritten Math Equation Solver

A full end-to-end Computer Vision system that reads a **photo or hand-drawn image** of a handwritten math equation, recognises each symbol using a trained CNN, and returns a **step-by-step symbolic solution** through an interactive web interface.

Built as a university Computer Vision course project.

---

## Live Demo

> Deploy with `streamlit run app/app.py` — see [Installation](#installation) below.

---

## Pipeline

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

## Features

- **Dual model support** — switch between CNN v3 and MobileNetV2, or run both side-by-side for comparison
- **18 symbol classes** — digits 0–9, operators `+ - * / = x ( )`
- **Canvas drawing** — draw directly in the browser with instant recognition
- **Image upload** — photograph a handwritten equation and upload it
- **Grad-CAM visualisation** — see which pixels the model attends to for each prediction
- **Step-by-step solver** — arithmetic, linear equations, and quadratic equations via SymPy
- **History tab** — browse all previously solved equations in the session
- **Model metrics tab** — training curves, confusion matrices, per-class accuracy, Grad-CAM grids

---

## Model Performance

| Model | Val Accuracy | Classes | Input |
|-------|-------------|---------|-------|
| **CNN v3** | **98.95%** | 18 | 32×32 grayscale |
| MobileNetV2 | ~93% | 18 | 32×32 → RGB |

### Training Data

| Source | Type | Samples |
|--------|------|---------|
| EMNIST Digits | Real handwritten digits 0–9 | 280 000 |
| EMNIST Letters | Real handwritten x/X | ~5 600 |
| HASYv2 | Real handwritten operators +−×÷= | ~8 000 |
| Kaggle Math Symbols | Real handwritten 0–9, operators | ~2 400 |
| Synthetic (PIL + augmentation) | `( )` parentheses | 5 000 |

---

## Installation

```bash
git clone https://github.com/Tamiru-Alemnew/CV_group_project.git
cd CV_group_project

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Place trained model files in `models/saved_models/`:
- `best_cnn_model.h5`
- `best_mobilenet_model.h5` *(optional — enables comparison mode)*

---

## Usage

```bash
# Launch the web app
streamlit run app/app.py

# Process a single image from CLI
python main.py --mode image --image photo.jpg

# Run tests
python main.py --mode test
```

| Command | Description |
|---------|-------------|
| `streamlit run app/app.py` | Launch web application |
| `python main.py --mode app` | Same via CLI |
| `python main.py --mode image --image photo.jpg` | Single image |
| `python main.py --mode camera` | Webcam capture |
| `python demo/evaluation.py` | Accuracy & speed benchmark |

---

## Project Structure

```
math_equation_solver/
├── app/
│   └── app.py                  Streamlit web app (4 tabs)
├── src/
│   ├── preprocessing.py        Classical CV preprocessing
│   ├── segmentation.py         Contour detection & character extraction
│   ├── data_preparation.py     Dataset builder
│   ├── model.py                CNN v3 architecture
│   ├── gradcam.py              Grad-CAM (CNN + MobileNetV2)
│   ├── recognize.py            Inference wrapper
│   ├── equation_parser.py      Symbol sequence → equation string
│   ├── math_solver.py          SymPy symbolic solver
│   ├── step_formatter.py       Solution formatter
│   ├── history_manager.py      Session history
│   └── pipeline.py             End-to-end integrator
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── demo/
│   └── evaluation.py           Benchmark script
├── tests/
│   ├── test_pipeline.py
│   └── test_stage1.py
├── tools/
│   └── generate_operator_images.py
├── config.py                   Central configuration
├── main.py                     CLI entry point
└── requirements.txt
```

---

## Datasets & References

| Dataset | Description |
|---------|-------------|
| [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) | 280 000 handwritten digit images |
| [HASYv2](https://zenodo.org/record/259444) | 168 000 handwritten math symbols |
| [Handwritten Math Symbols](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols) | 82-class operator dataset |

### Key Papers

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML.
- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE TSM.

---

## Team

| Name | Role |
|------|------|
| Tamiru Alemnew | |
| [Member 2] | |
| [Member 3] | |
