# PerovSeek

PerovSeek combines Bayesian formulation optimization with spectral prediction of perovskite solar-cell PCE. Its single demonstration notebook, `notebooks/PerovSeek_demo.ipynb`, calls the `perovseek.bayesian` and `perovseek.spectra` libraries in eight code cells.

PerovSeek 将贝叶斯配方优化与钙钛矿太阳能电池 PCE 光谱预测结合。唯一演示入口为 `notebooks/PerovSeek_demo.ipynb`，通过八个代码单元调用 `perovseek.bayesian` 和 `perovseek.spectra` 库。

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

1. **Bayesian optimization / 贝叶斯优化** — Read 126 measured formulations and editable bounds from Excel, then generate six candidates.
   从已写好表头的 Excel 读取 126 条实测配方与可调边界，生成六条候选配方。
2. **High-throughput experimentation and characterization / 高通量实验与表征** — Prepare films and collect absorption and PL spectra, retaining the formulation-to-sample mapping.
   制备薄膜，采集吸收和 PL 光谱，保存配方与样本的对应关系。
3. **Pretrained model prediction / 预训练模型预测** — Predict all 2,268 spectral records with the supplied checkpoint and select the top eight.
   使用附带检查点预测全部 2,268 条光谱记录，按预测 PCE 排序并选出前八条。
4. **Device validation and feedback / 器件验证与反馈** — Retrieve the selected records' device PCE from the same spectral workbook and display a `PCE (%)` table.
   从同一光谱工作簿读取所选记录对应的器件 PCE，展示 `PCE (%)` 表。

The formulation and spectral examples are independent datasets. The displayed device values belong to existing spectral records; a new optimization round requires measured PCE paired with each new formulation. The demo does not append records to the formulation workbook.

配方与光谱示例使用独立数据集。展示的器件数值对应已有光谱记录；开展下一轮优化时，须将新配方与其器件实测 PCE 配对。演示不向配方工作簿追加记录。

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
legacy/                         Original reference material / 原始参考程序
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

Run `python -m unittest discover -s tests -v`. See [validation.md](docs/validation.md) for notebook verification steps and model reference checks.

运行 `python -m unittest discover -s tests -v`；演示运行的验证方法与模型参考核验见[验证说明](docs/validation.md)。
