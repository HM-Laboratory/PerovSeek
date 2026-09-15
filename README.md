# PerovSeek

PerovSeek combines Bayesian formulation optimization with spectral prediction of perovskite solar-cell PCE. The demo calls two Python modules, `perovseek.bayesian` and `perovseek.spectra`, with editable inputs and optimization parameters.

PerovSeek 将贝叶斯配方优化与钙钛矿太阳能电池 PCE 光谱预测结合。演示通过 `perovseek.bayesian` 和 `perovseek.spectra` 两个 Python 模块运行，输入数据和优化参数均可修改。

## Install and run / 安装与运行

Use Python 3.12. Run these commands from the repository root:

使用 Python 3.12，在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m pip install -e .
jupyter notebook notebooks/PerovSeek_demo.ipynb
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Run the notebook cells in order. The supplied PCE checkpoint supports immediate prediction.

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。按顺序运行 notebook 单元格，附带的 PCE 检查点可直接用于预测。

## Four stages / 四个阶段

1. **Bayesian optimization / 贝叶斯优化** — Read measured formulations and bounds from Excel, set PCE as the target, and generate candidate formulations.
   从 Excel 读取实测配方与边界，设置 PCE 目标和优化参数，生成候选配方。
2. **High-throughput experimentation and characterization / 高通量实验与表征** — Prepare the recommended films and collect absorption and PL spectra with sample identifiers.
   制备候选薄膜，采集吸收和 PL 光谱，并记录样本编号。
3. **Pretrained model prediction / 预训练模型预测** — Load the PCE checkpoint, predict PCE from the spectra, and rank samples for device testing.
   加载 PCE 检查点，通过光谱预测 PCE，对样本排序并筛选待测器件。
4. **Device validation and feedback / 器件验证与反馈** — Measure device PCE and add complete measured records to the formulation data for the next round.
   实测器件 PCE，将完整的实测记录加入配方数据，进入下一轮优化。

See [workflow.md](workflow.md) for parameters and outputs, and [data sources](docs/data_sources.md) for preparation and model details.

参数与输出见 [workflow.md](workflow.md)，数据整理与模型说明见[数据来源](docs/data_sources.md)。

## Repository layout / 目录结构

```text
notebooks/PerovSeek_demo.ipynb    Demo / 演示
perovseek/bayesian.py            Bayesian optimization / 贝叶斯优化
perovseek/spectra.py             Spectral inference / 光谱推理
perovseek/models/                Model architecture / 模型结构
data/formulations.xlsx          Formulations and bounds / 配方与边界
data/formulations_sources.csv   Source-row mapping / 来源行映射
data/spectra/                   Spectral workbooks / 光谱数据
data/source/                    Original formulation workbooks / 原始配方数据
checkpoints/                    Encoder and PCE weights / 编码器与 PCE 权重
scripts/train_spectral_model.py  Optional fine-tuning / 可选微调
outputs/                        Generated results / 运行结果
legacy/                         Earlier notebooks and utilities / 原有程序
```

## Train a PCE model / 训练 PCE 模型

`checkpoints/spectral_pce_state.pt` contains the encoder and a supervised PCE head. `checkpoints/spectral_encoder.pt` contains the spectral pre-training weights. To fine-tune a new PCE model:

`checkpoints/spectral_pce_state.pt` 包含编码器与监督训练的 PCE 预测头；`checkpoints/spectral_encoder.pt` 为光谱预训练权重。需要重新微调时执行：

```bash
python scripts/train_spectral_model.py \
  --data data/spectra/1.59eV_additive_data.xlsx \
  --encoder checkpoints/spectral_encoder.pt \
  --output outputs/training/spectral_pce_state.pt \
  --epochs 100
```

The original files are retained under the paths listed in [reorganization.md](docs/reorganization.md).

原有文件保留在[目录调整记录](docs/reorganization.md)所列路径。

## Verification / 验证

Run `python -m unittest discover -s tests -v`. See [validation.md](docs/validation.md) for the executed workflow and model checks.

运行 `python -m unittest discover -s tests -v`；完整运行与模型核验见[验证记录](docs/validation.md)。
