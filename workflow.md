# Workflow guide / 工作流程

## English

### 1. Open the single notebook

Install the Python 3.12 environment described in [README.md](../README.md), then open [PerovSeek_workflow.ipynb](PerovSeek_workflow.ipynb) from the repository root. This workflow was validated with Python 3.12.14. Keep the existing source-directory layout. Run the cells from top to bottom so that imports, configuration and data checks are visible before any fit or prediction.

The notebook is the main executable narrative. [workflow_helpers.py](../workflow_helpers.py) provides the small training/checkpoint support needed to reuse the original training implementation; it is not a hidden replacement for the BO or inference workflow.

### 2. Read and optimize measured formulations

The `init` sheet of `Bayesian optimization/data/BO_SAM_Additive.xlsx` supplies 57 measured records. Inspect the selected columns and objective before fitting. Preserve the workbook's component names and units, including labels that differ from manuscript terminology.

The visible cells prepare the tensors, fit a Gaussian-process surrogate, construct `qLogNoisyExpectedImprovement`, optimize the acquisition function under the displayed bounds and sum constraints, and export the candidate table. The solvent ratio is fixed at DMF:NFM:EA = 81:17:2 for this example.

| Setting | New notebook demonstration | Original BO example |
|---|---|---|
| Objective | PCE | `Voc*FF` |
| Candidate batch size | 6 | 62 |
| Random seed | 42 | Consult the original notebook |
| Monte Carlo samples | 128 | Consult the original notebook |
| Acquisition restarts | 4 | Consult the original notebook |

The smaller computational budget supports the demonstration. It does not establish that the resulting candidates are equivalent to the original campaign's recommendations. Inspect the notebook for all other actual model and optimizer settings; the paper and example code should not be assumed to use identical configurations. GP means and uncertainties are model outputs, not device measurements or demonstrated performance gains.

### 3. Load the full spectral example

`data/1.59eV_additive_data.xlsx` supplies 2,268 examples. The notebook reads absorption, top-excited PL, bottom-excited PL and the `PCE#rs1` labels. It retains the repository's preprocessing: absorption divided by 4, top PL by 120,000 and bottom PL by 150,000, with wavelength-offset channels passed to the model. Inspect the displayed dimensions and sample/label checks before continuing.

The `1.59eV` filename is a source label; the workflow does not silently rename it to a manuscript bandgap label. These spectral examples are not established as paired with the BO formulation records.

### 4. Fine-tune, save and evaluate

The distributed `Pre-trained_model.pth` supplies the spectral pre-training model. Build the supervised PCE model with a prediction head and fine-tune the last 10 encoder blocks and that head for 100 epochs using the visible configuration. This training step is required before PCE inference.

For the 2,268-sample example, seed 42 defines 500 training, 268 validation and 1,500 test examples. Select the model using validation results, then evaluate the test set. Save the trained weights together with the architecture, preprocessing, target scale and split/configuration information needed to reload them. Do not select or tune the model using the test scores.

Report PCE in percent and MAE in **percentage points**. Plot and export all raw predictions, including negative or other physically invalid outputs; those values reveal limitations of the fitted model. Do not clip values before calculating errors or narrow plot limits to hide them.

### 5. Interpret the workflow and its limits

The source materials are the existing data, notebooks, `model.py`, `utility.py` and checkpoint distributed in [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek). The new notebook exposes one route through these materials while retaining the original files.

The BO and spectral sections are executable but separate public examples. They do not constitute a newly completed experimental closed loop: new BO candidates have not been fabricated or measured, and existing spectral labels must not be presented as their validation. Actual device measurements would be required before adding new feedback records.

Random splitting supports assessment of this supervised run; it does not establish generalization to a new campaign, composition family or fabrication process. The public materials do not establish whether test examples overlap with encoder pre-training. This notebook therefore makes no independent claim to reproduce the paper's AIO accuracy, efficiency gains or optimized formulation.

## 中文

### 1. 从单一 notebook 开始

按 [README.md](../README.md) 安装 Python 3.12 环境（本流程已在 3.12.14 验证），在仓库根目录打开 [PerovSeek_workflow.ipynb](PerovSeek_workflow.ipynb)，保持原有输入目录结构并顺序执行单元格。notebook 展开数据、配置、拟合与推理过程；[workflow_helpers.py](../workflow_helpers.py) 仅辅助复用原训练流程与保存检查点。

### 2. 配方导入与贝叶斯优化

`Bayesian optimization/data/BO_SAM_Additive.xlsx` 的 `init` 表含 57 条实测记录。先检查选用的列、目标和单位，再构建张量、拟合高斯过程、创建 `qLogNoisyExpectedImprovement` 并按所示边界与求和约束优化候选，最后导出候选表。保留源工作簿的组分名称，不自动替换成论文中的其他拼写。该示例固定 DMF:NFM:EA = 81:17:2。

新 notebook 默认采用 **PCE、6 个候选、seed 42、128 个蒙特卡洛样本及 4 次重启**。原 BO 示例的目标为 `Voc*FF`，每批 62 个候选。缩减计算预算用于演示，不意味着生成的候选与原实验过程等价；其他模型与优化参数以单元格实际配置为准。GP 均值和不确定度是预测，不是实测效率，也不能据此宣称已经获得性能提升。

### 3. 光谱导入与微调

完整的 `data/1.59eV_additive_data.xlsx` 包含 2,268 个样本。读取吸收、上/下激发 PL 和 `PCE#rs1` 标签，并保留原预处理：吸收除以 4，上激发 PL 除以 120,000，下激发 PL 除以 150,000，另加入波长偏移通道。执行前核对维度及样本/标签检查结果。`1.59eV` 保留为源文件标签；这些样本与 BO 配方记录没有经过验证的配对关系。

公开的 `Pre-trained_model.pth` 不是现成的 PCE 预测器。需构建监督预测头，并按展示配置对最后 10 个编码器块及预测头进行 100 个 epoch 的微调。seed 42 将数据分为 500 个训练、268 个验证和 1,500 个测试样本。使用验证集选择检查点，再评价测试集；保存权重时同时记录重新加载所需的架构、预处理、目标缩放和划分/配置。

### 4. 评价与真实实验反馈

PCE 使用百分数，MAE 使用**百分数百分点**。绘图和导出均应保留全部原始预测，包括负值及其他非物理输出；不能先截断预测再计算误差，也不能缩窄坐标范围隐藏这些点。

材料来自 [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek) 中已有的公开数据、notebook、模型代码和检查点。新流程保留原文件，不包含私有文稿或演示视频。

BO 与光谱部分是可运行的不同示例，并未在本次运行中完成新的实验闭环。新推荐配方没有被制备或实测，现有光谱标签不能作为其器件验证；只有取得真实器件测量后才能添加实验反馈。

随机划分仅用于本次监督微调评价，不能证明对新实验过程、新组分体系或新制备工艺的泛化。公开材料尚不能确认编码器预训练是否覆盖这些测试样本，因此本流程不宣称独立复现论文中的 AIO 准确率、效率提升或最优配方。
