# PerovSeek

## English

PerovSeek combines Bayesian formulation optimization with spectral learning for perovskite experiments. **[PerovSeek_workflow.ipynb](PerovSeek_workflow.ipynb)** is the single, step-by-step entry point for the public software demonstration. It exposes data loading, parameters, Gaussian-process fitting, acquisition optimization, candidate tables, spectral preprocessing, model construction, training and inference in notebook cells. The original analysis files remain available.

### Run the notebook

Use Python 3.12; this workflow was validated with Python 3.12.14. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
jupyter notebook PerovSeek_workflow.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Select this environment's Python kernel and run the cells in order. The spectral example includes **100 epochs of supervised fine-tuning**; its duration depends on the hardware. [workflow_helpers.py](workflow_helpers.py) supports the training/checkpoint handoff while the main workflow remains visible in the notebook.

The existing inputs are:

```text
Bayesian optimization/data/BO_SAM_Additive.xlsx
data/1.59eV_additive_data.xlsx
Pre-trained model/Pre-trained_model.pth
Pre-trained model/model.py
Pre-trained model/utility.py
```

### What this example demonstrates

- **Bayesian optimization:** 57 existing formulation records; the demo explicitly uses PCE, 6 new candidates, seed 42, 128 Monte Carlo samples and 4 restarts. The original BO example uses `Voc*FF` and a batch of 62. The notebook displays the actual configuration and exports its recommendations.
- **Spectral learning:** the full public `1.59eV` workbook contains 2,268 examples. The supplied pre-trained checkpoint provides the spectral encoder/reconstruction model. PCE prediction requires a newly trained supervised head and fine-tuning; the checkpoint alone does not predict PCE.
- **Separate examples:** the formulation records and spectral workbook have no verified sample-level pairing in this workflow. Newly recommended formulations are unmeasured. Existing spectral labels are not measurements of those new recommendations.

This is an executable demonstration using public repository data, **not a reproduction of the manuscript's reported all-in-one (AIO) optimization results**. Scores describe the local run. Random train/validation/test separation applies to supervised fine-tuning; overlap with encoder pre-training is unknown. Raw predictions, including any values outside the physical PCE range, should be retained. See [the workflow guide](workflow.md) for the sequence, provenance and limits.

## 中文

PerovSeek 将贝叶斯配方优化与光谱学习结合，用于钙钛矿实验。[PerovSeek_workflow.ipynb](PerovSeek_workflow.ipynb) 是公开软件演示的**单一入口**：在 notebook 单元格中逐步展开数据导入、参数配置、高斯过程拟合、采集函数优化、候选表、光谱预处理、模型构建、训练与推理。原有分析文件继续保留。

### 安装与运行

使用 Python 3.12（本流程已在 3.12.14 验证），在仓库根目录执行上方命令，创建虚拟环境、安装 `requirements-demo.txt` 并启动 notebook。Windows 使用 `.venv\Scripts\activate` 激活环境。选择该环境的 Python 内核，按顺序运行单元格。光谱示例需要完成 **100 个 epoch 的监督微调**，耗时取决于硬件。`workflow_helpers.py` 辅助训练及检查点保存，不替代 notebook 中展开的主要步骤。

输入均为本仓库已有的公开数据与代码，路径见上方列表：57 条 BO 配方记录、包含 2,268 个样本的 `1.59eV` 光谱工作簿、预训练检查点及其模型代码。

### 演示范围

- 默认 BO 演示明确改用 **PCE 目标、6 个候选、seed 42、128 个蒙特卡洛样本及 4 次重启**；原 BO 示例使用 `Voc*FF` 和每批 62 个候选。实际配置与输出均在 notebook 中展示。
- 原预训练检查点用于光谱编码及重建，不能直接输出 PCE；需新建监督预测头并完成微调。
- 配方记录与光谱工作簿之间没有经过验证的样本配对关系。新推荐配方尚未实测，不能将已有光谱标签视为这些新候选的实验反馈。

本流程展示公开代码与数据的实际运行，**不等同于论文所报告的 AIO 优化结果复现**。评价指标仅对应本次运行；随机划分仅确保监督微调阶段的训练、验证和测试分离，编码器预训练的数据重叠情况未知。应保留包括负值在内的原始预测。流程、来源和适用边界详见 [workflow guide](workflow.md)。
