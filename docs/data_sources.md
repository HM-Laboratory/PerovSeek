# Data and models / 数据与模型

The demo uses the public datasets, spectral architecture and training implementation from [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek). Formulation optimization and spectral prediction use separate example datasets.

演示采用 [HM-Laboratory/PerovSeek](https://github.com/HM-Laboratory/PerovSeek) 公开的数据、光谱模型结构与训练实现。配方优化和光谱预测分别使用独立的示例数据集。

## Formulations / 配方

`data/formulations.xlsx` contains the 126 records with complete nine-component inputs and measured `PCE` from sheets `1`–`5` of the original workbook `Bayesian optimization data.xlsx` (62, 16, 16, 16 and 16 records). Rows missing measured PCE are excluded; `PCE_pred` is not substituted for a measurement. The source label `4F-BASZ` is mapped to `4-FBSA` in the prepared table.

`data/formulations.xlsx` 取自原始工作簿 `Bayesian optimization data.xlsx` 的第 `1`–`5` 表，包含九组分与实测 `PCE` 均完整的 126 条记录，各表分别为 62、16、16、16、16 条。缺少实测 PCE 的行不纳入，也不使用 `PCE_pred` 代替实测值。整理表中将源列名 `4F-BASZ` 映射为 `4-FBSA`。

DMF, NFM and EA are volume fractions. Each row is divided by its solvent sum; three source records have sums of 0.99 or 1.01. The six SAM/passivator values retain the numerical scale of the source workbook. The initial `Bounds` values are the observed minimum and maximum of the prepared inputs.

DMF、NFM 和 EA 为体积分数，整理时将各行的三种溶剂分别除以其总和，其中三条原始记录的总和为 0.99 或 1.01。其余六个 SAM、钝化剂组分保留源工作簿的数值刻度。`Bounds` 初始值取整理后各输入列的最小值和最大值。

[formulation_provenance.csv](formulation_provenance.csv) maps each prepared Excel row to its original workbook name, sheet, Excel row and sample identifier, and retains the raw solvent fractions and their sum. Only the prepared formulation table and spectral workbooks are distributed in `data/`; the original formulation workbook is not required to run the demonstration. Its SHA-256 is `8076d7164e3e6d22a26f45fe55bb6dc5af67ee3c7cdfa4b1efb397087d861fd8`.

[formulation_provenance.csv](formulation_provenance.csv) 记录整理表各行对应的原始工作簿名称、工作表、Excel 行号和样本编号，并保留原始溶剂比例及其总和。`data/` 仅提供整理后的配方表与光谱工作簿；运行演示不需要原始配方工作簿。

## Spectra and checkpoint / 光谱与检查点

`data/spectra/1.59eV_additive_data.xlsx` contains 2,268 spectral records. The model joins 300 absorption points and two 66-point PL traces. Signal divisors are 4, 120,000 and 150,000 for absorption, top-excited PL and bottom-excited PL respectively. A second channel contains wavelength minus 400 nm; training divides the label `PCE#rs1` by 25, and inference multiplies predictions by 25.

`data/spectra/1.59eV_additive_data.xlsx` 包含 2,268 条光谱记录。模型拼接 300 个吸收数据点与两组各 66 点的 PL 光谱。吸收、上激发 PL、下激发 PL 分别除以 4、120,000、150,000；第二通道为波长减去 400 nm。训练时将标签 `PCE#rs1` 除以 25，推理时将预测值乘以 25。

The demonstration calls `load_spectra(..., label=None)` and predicts all records, then ranks them by predicted PCE. In its final stage, it reads the original device PCE from `Param.PCE#rs1` (Excel column R) for the eight selected records. `Param`'s `Unnamed: 0` column supplies the displayed sample identifiers; identifiers repeat six or seven times in this workbook. The prediction's `source_row` is the zero-based position in `Param`, so matching uses the row position and verifies the sample identifier. It is not an Excel row number or a unique sample identifier.

演示通过 `load_spectra(..., label=None)` 读取数据，对全部记录预测并按预测 PCE 排序。最后阶段从 `Param.PCE#rs1`（Excel R 列）读取所选八条记录的原始器件 PCE。展示的样本编号来自 `Param` 的 `Unnamed: 0` 列，同一编号在该工作簿中出现六或七次。预测结果中的 `source_row` 是 `Param` 中从零开始的行位置，匹配时据此定位并核对样本编号；它不是 Excel 行号，也不是唯一的样本编号。

The spectral records and their device PCE are independent of the 126-record formulation dataset. The demonstration does not establish a pairing between these records and the newly recommended `D01`–`D06` formulations, and does not add them to `data/formulations.xlsx`.

光谱记录及对应器件 PCE 与 126 条配方数据相互独立。演示没有建立这些记录与新推荐的 `D01`–`D06` 配方之间的对应关系，也不将它们加入 `data/formulations.xlsx`。

The supplied PCE checkpoint was fine-tuned for 100 epochs with seed 42, using 500 training, 268 validation and 1,500 test examples. Training updates the PCE head and the final ten encoder blocks; validation loss selects the saved weights. The checkpoint stores architecture, preprocessing, source hashes and split indices alongside the model weights.

附带的 PCE 检查点以随机种子 42 微调 100 个 epoch，使用 500 个训练、268 个验证和 1,500 个测试样本。训练更新 PCE 预测头及最后十个编码器块，根据验证损失选择权重。检查点同时保存模型结构、预处理、来源文件哈希及数据划分索引。

Optional test evaluation describes this supervised example split. The all-record demonstration also includes training and validation examples, so its ranking is not a held-out performance evaluation. The public files do not establish overlap with encoder pre-training. Model outputs can fall outside the physical PCE range; exports retain the complete predictions.

可选测试集评价对应本示例的监督学习划分。演示的全量预测还包含训练集和验证集记录，因此排序结果不代表独立测试性能。公开文件未明确其与编码器预训练数据的重叠情况。模型输出可能超出 PCE 的物理范围，导出数据保留完整预测值。

## File integrity / 文件校验

SHA-256 checksums for the distributed data, model weights and demonstration video are listed in [checksums.sha256](checksums.sha256). Run the following from the repository root:

随仓库提供的数据、权重和视频的 SHA-256 见 [checksums.sha256](checksums.sha256)。在仓库根目录运行：

```bash
# macOS
shasum -a 256 -c docs/checksums.sha256
# Linux
sha256sum -c docs/checksums.sha256
```
