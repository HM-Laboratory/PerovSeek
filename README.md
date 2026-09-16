# PerovSeek

Bayesian formulation optimization and spectral prediction of power conversion efficiency (PCE) for perovskite solar cells.

[![Tests](https://github.com/HM-Laboratory/PerovSeek/actions/workflows/tests.yml/badge.svg)](https://github.com/HM-Laboratory/PerovSeek/actions/workflows/tests.yml)

[中文说明](README_zh-CN.md) · [Demo notebook](notebooks/PerovSeek_demo.ipynb) · [Workflow guide](docs/workflow.md) · [Data and models](docs/data_sources.md)

## Demonstration

https://github.com/user-attachments/assets/94b40619-ab18-4f79-aa78-81a9c72ec9ac

**69 seconds · 1080p · Chinese and English subtitles.** The video follows formulation optimization, high-throughput experiments, spectral prediction, and device feedback. Each software view is shown for 4 seconds.

[Download the full video](https://github.com/HM-Laboratory/PerovSeek/raw/refs/heads/master/assets/videos/PerovSeek_demo.mp4) · [Subtitles](assets/videos/PerovSeek_demo.srt) · [Video description](docs/video.md)

## Quick start

Use **Python 3.12**. A CPU is sufficient; the trained PCE checkpoint is included.

```bash
git clone https://github.com/HM-Laboratory/PerovSeek.git
cd PerovSeek
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
jupyter notebook notebooks/PerovSeek_demo.ipynb
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`.

Run the notebook cells in order. Data loading, the PCE objective, optimization parameters, prediction calls and result tables are shown in the notebook. The optimization and spectral model implementations are in the `perovseek` package. Results are saved to `outputs/`.

## Workflow

| Stage | Input and result |
| --- | --- |
| **1. Bayesian optimization** | Read 126 formulations and editable bounds from Excel; recommend six candidates using measured PCE as the objective. |
| **2. High-throughput experiments and characterization** | Prepare films and acquire absorption and two PL spectra, retaining the link between formulation and sample. |
| **3. Pretrained-model prediction** | Predict PCE for 2,268 spectral records and rank them by predicted efficiency. |
| **4. Device validation and feedback** | Display the corresponding device PCE for the eight selected spectral records. |

The nine formulation components are **DMF, NFM, EA, Me-4, Py3, 4PADCB, 4-FBSA, F3EABr and SPFBS**. Column headers are already provided in `data/formulations.xlsx`. Edit the `Formulations` and `Bounds` sheets to update measurements and search bounds; no column-renaming code is needed.

The supplied formulation and spectral examples are independent datasets. The displayed device values belong to existing spectral records. A new experimental cycle requires each recommended formulation to be paired with its own spectra and measured device PCE. The demonstration leaves the formulation workbook unchanged.

## Repository structure

```text
PerovSeek/
├── notebooks/
│   └── PerovSeek_demo.ipynb     # Single demonstration entry point
├── perovseek/
│   ├── bayesian.py             # Bayesian formulation optimization
│   ├── spectra.py              # Spectral preprocessing and prediction
│   ├── training.py             # Optional PCE model fine-tuning
│   └── models/                 # Spectral model architecture
├── data/
│   ├── formulations.xlsx       # Formulations and optimization bounds
│   └── spectra/                # Spectral workbooks
├── checkpoints/                # Encoder and trained PCE weights
├── assets/videos/              # Full demonstration video and subtitles
├── docs/                       # Workflow, data provenance and validation
├── scripts/                    # Training command
├── tests/                      # Workflow checks
└── .github/workflows/          # Continuous integration
```

Generated outputs and local environments are excluded from version control.

## Optional model training

`checkpoints/spectral_pce_state.pt` contains the encoder and supervised PCE head used by the demonstration. `checkpoints/spectral_encoder.pt` provides the pretraining weights for fine-tuning:

```bash
python scripts/train_spectral_model.py \
  --data data/spectra/1.59eV_additive_data.xlsx \
  --encoder checkpoints/spectral_encoder.pt \
  --output outputs/training/spectral_pce_state.pt \
  --epochs 100
```

Preprocessing, checkpoint provenance and the evaluation split are described in [Data and models](docs/data_sources.md).

## Validation and development

```bash
python -m unittest discover -s tests -v
```

See [Validation](docs/validation.md) for notebook acceptance checks and model reference results, and [Contributing](CONTRIBUTING.md) for development instructions.
