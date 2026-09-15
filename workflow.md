# Workflow / 工作流程

## 1. Bayesian optimization / 贝叶斯优化

Open `data/formulations.xlsx`. `Formulations` contains nine components and measured `PCE`, with headers in the first row. `Bounds` contains `Component`, `Lower` and `Upper`; edit these cells to change the search region. Read either sheet directly with `pandas.read_excel`.

打开 `data/formulations.xlsx`。`Formulations` 包含九个组分与实测 `PCE`，第一行为表头。`Bounds` 包含 `Component`、`Lower` 和 `Upper`，可修改上下界控制搜索范围。两个表均可通过 `pandas.read_excel` 直接读取。

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

The returned table contains candidate IDs, nine compositions, `PCE_pred` and `PCE_std`. These are model estimates for the recommended formulations. Save the full-precision values when preparing experiments.

返回表包含候选编号、九组分比例、`PCE_pred` 和 `PCE_std`，后两列为模型估计。准备实验时使用导出文件中的完整精度数值。

## 2. High-throughput experimentation and characterization / 高通量实验与表征

Use the candidate formulations for film preparation and optical characterization. Retain the candidate ID with each film's absorption, top-excited PL and bottom-excited PL measurements. Store new measurements in the same layout and wavelength grid as the example workbook under `data/spectra/`.

根据候选配方制备薄膜并进行光学表征。将候选编号与各薄膜的吸收、上激发 PL、下激发 PL 测量记录对应保存。新增光谱采用 `data/spectra/` 中示例工作簿的表格布局和波长网格。

## 3. Pretrained model prediction / 预训练模型预测

`load_spectra` reads the optical workbook. `SpectralPredictor` loads `checkpoints/spectral_pce_state.pt`. Use `subset="test"` to evaluate the saved 1,500-example test split of the supplied workbook, or `subset="all"` for all rows in a compatible workbook.

`load_spectra` 读取光谱工作簿，`SpectralPredictor` 加载 `checkpoints/spectral_pce_state.pt`。`subset="test"` 使用附带数据中已保存的 1,500 个测试样本；`subset="all"` 可预测兼容格式工作簿中的全部样本。

The notebook plots spectra and predicted versus measured PCE, then sorts samples by predicted PCE. Predictions are exported without clipping. PCE is expressed in percent, while MAE and RMSE use percentage points. Keep sample identifiers with the prediction table when selecting devices for measurement.

演示绘制光谱与预测、实测 PCE 对照图，并按预测 PCE 对样本排序。预测值完整导出，不作截断。PCE 以百分数表示，MAE、RMSE 以百分数百分点表示。选择待测器件时保留预测表中的样本编号。

## 4. Device validation and feedback / 器件验证与反馈

The feedback template leaves measured PCE empty for the new candidates. After device testing, associate each measurement with its formulation and add complete measured rows to `Formulations`. Review the bounds and run optimization again. Missing measurements remain blank until an experiment supplies them.

反馈模板中，新候选的实测 PCE 保持空白。器件测试后，将测量结果与对应配方关联，并把完整的实测记录加入 `Formulations`；检查搜索边界后再次运行优化。缺少的测量结果在实验完成前保持空白。

Data preparation, units and checkpoint training are described in [data_sources.md](docs/data_sources.md).

数据整理、单位和检查点训练说明见 [data_sources.md](docs/data_sources.md)。
