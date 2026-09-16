# Workflow / 工作流程

Run the eight code cells in `notebooks/PerovSeek_demo.ipynb` in order. The notebook handles inputs, parameters and presentation; the optimization and spectral model implementations reside in the `perovseek` package. Formulation optimization and spectral prediction use independent example datasets.

按顺序运行 `notebooks/PerovSeek_demo.ipynb` 的八个代码单元。演示负责输入、参数和结果展示，优化与光谱模型实现封装在 `perovseek` 包中。配方优化与光谱预测使用独立的示例数据集。

## 1. Bayesian optimization / 贝叶斯优化

Open `data/formulations.xlsx`. `Formulations` contains 126 records of nine components and measured `PCE`, with headers in the first row. `Bounds` contains `Component`, `Lower` and `Upper`; edit these cells to change the search region. Read either sheet directly with `pandas.read_excel`.

打开 `data/formulations.xlsx`。`Formulations` 包含 126 条九组分与实测 `PCE` 记录，第一行为表头。`Bounds` 包含 `Component`、`Lower` 和 `Upper`，可修改上下界控制搜索范围。两个表均可通过 `pandas.read_excel` 直接读取。

The component order is `DMF`, `NFM`, `EA`, `Me-4`, `Py3`, `4PADCB`, `4-FBSA`, `F3EABr`, `SPFBS`. Solvent fractions must satisfy **DMF + NFM + EA = 1**. Each search upper bound must exceed its lower bound, and the three solvent ranges must permit a sum of one.

组分顺序为 `DMF`、`NFM`、`EA`、`Me-4`、`Py3`、`4PADCB`、`4-FBSA`、`F3EABr`、`SPFBS`。溶剂比例须满足 **DMF + NFM + EA = 1**。各搜索上界须大于下界，且三个溶剂的范围须允许其总和等于 1。

`perovseek.bayesian.recommend` fits a Gaussian process to measured PCE and selects a batch with qLogNEI. The notebook exposes these defaults:

`perovseek.bayesian.recommend` 根据实测 PCE 拟合高斯过程，通过 qLogNEI 选择下一批候选。演示使用以下可调默认参数：

| Parameter / 参数 | Default / 默认值 |
| --- | ---: |
| Target / 目标 | `PCE` |
| Batch size / 候选数量 | 6 |
| Seed / 随机种子 | 42 |
| MC samples / 蒙特卡洛样本数 | 128 |
| Acquisition restarts / 采集函数重启次数 | 4 |
| Initial samples / 初始采样数 | 128 |
| Assumed observation noise SD / 假定观测噪声标准差 | 2.0 PCE percentage points / 百分数百分点 |
| GP fitting iterations / 高斯过程拟合迭代上限 | 40 |
| Acquisition optimization iterations / 采集函数优化迭代上限 | 80 |

The returned table contains candidate IDs `D01`–`D06`, nine compositions, `PCE_pred` and `PCE_std`. These are model estimates for the recommended formulations. The notebook displays six candidates and exports the full table to `outputs/recommended_formulations.csv`. Use the full-precision values when preparing experiments.

返回表包含候选编号 `D01`–`D06`、九组分比例、`PCE_pred` 和 `PCE_std`，后两列为模型估计。演示展示六条候选，将完整表导出至 `outputs/recommended_formulations.csv`。准备实验时使用导出文件中的完整精度数值。

## 2. High-throughput experimentation and characterization / 高通量实验与表征

Use the candidate formulations for film preparation and optical characterization. Retain the candidate ID with each film's absorption, top-excited PL and bottom-excited PL measurements. Store new measurements in the same layout and wavelength grid as the example workbook under `data/spectra/`.

根据候选配方制备薄膜并进行光学表征。将候选编号与各薄膜的吸收、上激发 PL、下激发 PL 测量记录对应保存。新增光谱采用 `data/spectra/` 中示例工作簿的表格布局和波长网格。

This stage is a laboratory step. The supplied spectral workbook contains existing measurements, independent of the six candidates generated in stage 1.

本阶段需要在实验室完成。附带光谱工作簿提供已有测量数据，与第一阶段生成的六条候选配方相互独立。

## 3. Pretrained model prediction / 预训练模型预测

The notebook reads the optical workbook with `load_spectra(SPECTRA_FILE, label=None)`, so device PCE labels are excluded from inference results. `SpectralPredictor` loads `checkpoints/spectral_pce_state.pt` and predicts all 2,268 records with `subset="all"`; no training runs in the notebook.

演示通过 `load_spectra(SPECTRA_FILE, label=None)` 读取光谱工作簿，推理结果不包含器件 PCE 标签。`SpectralPredictor` 加载 `checkpoints/spectral_pce_state.pt`，使用 `subset="all"` 预测全部 2,268 条记录；演示中不运行训练。

The notebook displays absorption and PL spectra, then ranks records by predicted PCE and selects the top eight. Predictions are expressed in percent and exported without clipping. `outputs/spectral_predictions.csv` contains all predictions; `outputs/ranked_spectra.csv` contains the selected eight, preserving `sample_id` and `source_row`.

演示展示吸收和 PL 光谱，再按预测 PCE 排序，选出前八条记录。预测值以百分数表示，完整导出，不作截断。`outputs/spectral_predictions.csv` 保存全部预测，`outputs/ranked_spectra.csv` 保存所选八条，均保留 `sample_id` 和 `source_row`。

## 4. Device validation and feedback / 器件验证与反馈

The notebook reads `Param` from the same spectral workbook and selects `Param.iloc[ranked.source_row]`. `source_row` is the zero-based data-row position, not an Excel row number. It checks that the selected `Unnamed: 0` identifiers match `ranked.sample_id` in order, then retrieves `PCE#rs1` (Excel column R).

演示从同一光谱工作簿读取 `Param`，通过 `Param.iloc[ranked.source_row]` 选取记录。`source_row` 是从零开始的数据行位置，不是 Excel 行号。程序先核对所选 `Unnamed: 0` 编号与 `ranked.sample_id` 顺序一致，再读取 `PCE#rs1`（Excel R 列）。

Sample identifiers repeat in the supplied workbook, so retain both `source_row` and `sample_id` when matching records. The final table shows `Sample` and `PCE (%)`; `outputs/device_results.csv` preserves `sample_id`, `source_row` and `PCE` for all eight selected records.

附带工作簿的样本编号存在重复，因此匹配记录时须同时保留 `source_row` 和 `sample_id`。最终表格展示 `Sample` 与 `PCE (%)`，`outputs/device_results.csv` 保存所选八条记录的 `sample_id`、`source_row` 和 `PCE`。

These device values belong to the existing spectral records. To close a new experimental cycle, associate each new device measurement with its actual nine-component formulation, then add only complete paired measurements to `Formulations` before running optimization again. The demo leaves the 126-record formulation workbook unchanged.

这些器件数值对应已有光谱记录。开展新一轮实验闭环时，应将新器件测量结果与其实际九组分配方关联，只将完整且已配对的实测记录加入 `Formulations`，再运行优化。演示保持原有 126 条配方记录不变。

Data preparation, units and checkpoint training are described in [data_sources.md](data_sources.md).

数据整理、单位和检查点训练说明见 [data_sources.md](data_sources.md)。
