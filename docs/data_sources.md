# Data and models / 数据与模型

The demo uses the public datasets, spectral architecture and training implementation from [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek). Formulation optimization and spectral prediction use separate example datasets.

演示采用 [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek) 公开的数据、光谱模型结构与训练实现。配方优化和光谱预测分别使用独立的示例数据集。

## Formulations / 配方

`data/formulations.xlsx` contains the 126 records with complete nine-component inputs and measured `PCE` from sheets `1`–`5` of `data/source/Bayesian optimization data.xlsx` (62, 16, 16, 16 and 16 records). Rows missing measured PCE are excluded; `PCE_pred` is not substituted for a measurement. The source label `4F-BASZ` is mapped to `4-FBSA` in the prepared table.

`data/formulations.xlsx` 取自 `data/source/Bayesian optimization data.xlsx` 的第 `1`–`5` 表，包含九组分与实测 `PCE` 均完整的 126 条记录，各表分别为 62、16、16、16、16 条。缺少实测 PCE 的行不纳入，也不使用 `PCE_pred` 代替实测值。整理表中将源列名 `4F-BASZ` 映射为 `4-FBSA`。

DMF, NFM and EA are volume fractions. Each row is divided by its solvent sum; three source records have sums of 0.99 or 1.01. The six SAM/passivator values retain the numerical scale of the source workbook. The initial `Bounds` values are the observed minimum and maximum of the prepared inputs.

DMF、NFM 和 EA 为体积分数，整理时将各行的三种溶剂分别除以其总和，其中三条原始记录的总和为 0.99 或 1.01。其余六个 SAM、钝化剂组分保留源工作簿的数值刻度。`Bounds` 初始值取整理后各输入列的最小值和最大值。

`data/formulations_sources.csv` maps each prepared Excel row to its source sheet, Excel row and sample identifier, and retains the raw solvent fractions and their sum. The original workbook is unchanged.

`data/formulations_sources.csv` 将整理表中的每个 Excel 行对应到来源工作表、Excel 行号和样本编号，并保留原始溶剂比例及其总和。原始工作簿保持不变。

## Spectra and checkpoint / 光谱与检查点

`data/spectra/1.59eV_additive_data.xlsx` contains 2,268 examples. The model joins 300 absorption points and two 66-point PL traces. Signal divisors are 4, 120,000 and 150,000 for absorption, top-excited PL and bottom-excited PL respectively. A second channel contains wavelength minus 400 nm; training divides the label `PCE#rs1` by 25, and inference multiplies predictions by 25.

`data/spectra/1.59eV_additive_data.xlsx` 包含 2,268 个样本。模型拼接 300 个吸收数据点与两组各 66 点的 PL 光谱。吸收、上激发 PL、下激发 PL 分别除以 4、120,000、150,000；第二通道为波长减去 400 nm。训练时将标签 `PCE#rs1` 除以 25，推理时将预测值乘以 25。

The supplied PCE checkpoint was fine-tuned for 100 epochs with seed 42, using 500 training, 268 validation and 1,500 test examples. Training updates the PCE head and the final ten encoder blocks; validation loss selects the saved weights. The checkpoint stores architecture, preprocessing, source hashes and split indices alongside the model weights.

附带的 PCE 检查点以随机种子 42 微调 100 个 epoch，使用 500 个训练、268 个验证和 1,500 个测试样本。训练更新 PCE 预测头及最后十个编码器块，根据验证损失选择权重。检查点同时保存模型结构、预处理、来源文件哈希及数据划分索引。

Evaluation describes this supervised example split. The public files do not establish its overlap with encoder pre-training. Model outputs can fall outside the physical PCE range, so exported values and plots retain the complete predictions.

评价结果对应本示例的监督学习划分；公开文件未明确其与编码器预训练数据的重叠情况。模型输出可能超出 PCE 的物理范围，导出数据与图表保留完整预测值。

The upstream workbook with updated headers is preserved as `data/source/Bayesian optimization data (updated headers).xlsx` (commit `9cbe58a`). Its values across all 15 sheets match the original workbook; the `4-FBSA` header matches the prepared demo table.

上游新上传的表头修订版保存在 `data/source/Bayesian optimization data (updated headers).xlsx`（提交 `9cbe58a`）。15 个工作表的数值均与原表一致，`4-FBSA` 表头与演示数据表一致。
