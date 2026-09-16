# Demonstration video / 演示视频

[Play in the repository homepage](../README.md#demonstration) · [Download MP4](https://github.com/HM-Laboratory/PerovSeek/raw/refs/heads/master/assets/videos/PerovSeek_demo.mp4) · [Subtitles](../assets/videos/PerovSeek_demo.srt)

The video presents the PerovSeek workflow: nine-component Bayesian formulation optimization, high-throughput experimentation and characterization, pretrained-model prediction, and device validation and feedback.

视频展示 PerovSeek 的九组分贝叶斯配方优化、高通量实验与表征、预训练模型预测，以及器件验证与反馈流程。

| Property | Value |
| --- | --- |
| Duration | 69.3 seconds |
| Resolution | 1920 × 1080 |
| Frame rate | 30 fps |
| Encoding | H.264 MP4 |
| Subtitles | Chinese above English, burned into the video; SRT also provided |
| Audio | None |
| Software views | Six notebook code/output captures, 4 seconds each |

## Sequence

| Time | Stage |
| --- | --- |
| 00:00.0–00:03.0 | Title |
| 00:03.0–00:16.5 | Bayesian optimization: data, parameters and recommendations |
| 00:16.5–00:41.3 | High-throughput experiments and optical characterization |
| 00:41.3–00:50.8 | Spectra, prediction code and the ranking by predicted PCE |
| 00:50.8–01:07.3 | Device preparation, characterization and PCE feedback |
| 01:07.3–01:09.3 | Closing workflow diagram |

The laboratory clips retain their original speed and sequence. The vacuum-deposition label is shown in both Chinese and English. The notebook captures show actual code and generated results; prediction code and the sorted table appear together.

实验片段保留原始速度与顺序，真空沉积画面配有中英文标签。软件画面展示 Notebook 的代码和运行结果，模型预测代码与效率排序表同屏显示。

## Data context

Formulation optimization and spectral prediction use independent example datasets. The device-PCE table contains the existing `Param.PCE#rs1` values for the selected spectral records, matched by `source_row` and verified with `sample_id`. The video does not demonstrate completed experiments on the newly recommended formulations. Data preparation and model details are documented in [Data and models](data_sources.md).

配方优化与光谱预测使用独立示例数据。器件 PCE 表来自所选光谱记录已有的 `Param.PCE#rs1`，通过 `source_row` 定位并核对 `sample_id`。视频未展示新推荐配方的完整实验验证；数据整理和模型细节见[数据与模型](data_sources.md)。
