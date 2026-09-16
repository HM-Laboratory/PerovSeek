# PerovSeek

用于钙钛矿太阳能电池的贝叶斯配方优化与光谱 PCE 预测。

[English](README.md) · [演示 Notebook](notebooks/PerovSeek_demo.ipynb) · [工作流程](docs/workflow.md) · [数据与模型](docs/data_sources.md)

## 演示视频

https://github.com/user-attachments/assets/94b40619-ab18-4f79-aa78-81a9c72ec9ac

全长约 69 秒，1080p，中英双语字幕。视频依次展示贝叶斯优化、高通量实验与表征、预训练模型预测、器件验证与反馈；软件画面每张显示 4 秒。

[下载完整版](https://github.com/HM-Laboratory/PerovSeek/raw/refs/heads/master/assets/videos/PerovSeek_demo.mp4) · [字幕文件](assets/videos/PerovSeek_demo.srt) · [视频说明](docs/video.md)

## 快速开始

使用 **Python 3.12**。CPU 即可运行，仓库已提供用于预测的模型权重。

```bash
git clone https://github.com/HM-Laboratory/PerovSeek.git
cd PerovSeek
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
jupyter notebook notebooks/PerovSeek_demo.ipynb
```

Windows 下使用 `py -3.12 -m venv .venv` 创建环境，通过 `.venv\Scripts\Activate.ps1` 激活。

按顺序运行 Notebook 中的代码单元。Notebook 展示数据导入、PCE 目标、优化参数、模型预测和结果；算法实现放在 `perovseek` 包中。运行结果保存在 `outputs/`。

## 四个阶段

| 阶段 | 内容 |
| --- | --- |
| **贝叶斯优化** | 从 Excel 读取 126 条配方及搜索边界，以 PCE 为目标推荐六条候选配方。 |
| **高通量实验与表征** | 制备薄膜，采集吸收和双面激发 PL 光谱，保留配方与样本的对应关系。 |
| **预训练模型预测** | 对 2,268 条光谱记录预测 PCE，并按预测效率排序。 |
| **器件验证与反馈** | 展示所选八条光谱记录对应的器件 PCE。 |

九组分为 **DMF、NFM、EA、Me-4、Py3、4PADCB、4-FBSA、F3EABr、SPFBS**。`data/formulations.xlsx` 已填写表头，修改 `Formulations` 和 `Bounds` 工作表即可更新数据与搜索范围。

配方与光谱示例使用独立数据集，器件 PCE 对应已有光谱记录。开展新的实验闭环时，需将每条新配方与其光谱及器件 PCE 配对。演示不会修改配方工作簿。

## 目录入口

| 路径 | 用途 |
| --- | --- |
| `notebooks/PerovSeek_demo.ipynb` | 唯一演示入口 |
| `perovseek/` | 贝叶斯优化、光谱模型与训练实现 |
| `data/formulations.xlsx` | 配方、PCE 与优化边界 |
| `data/spectra/` | 光谱工作簿 |
| `checkpoints/` | 编码器与 PCE 模型权重 |
| `assets/videos/` | 完整视频与字幕 |
| `docs/` | 流程、数据来源、模型和验证说明 |
| `scripts/` | 可选训练命令 |
| `tests/` | 工作流程检查 |

训练命令见 [英文首页](README.md#optional-model-training)。运行检查：

```bash
python -m unittest discover -s tests -v
```

详细步骤见[验证说明](docs/validation.md)与[开发说明](CONTRIBUTING.md)。
