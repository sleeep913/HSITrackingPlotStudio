# 内置 Ground Truth 数据说明

HSI Tracking Plot Studio V1.4.0 内置了 HOTC2020、HOTC2023 和 HOTC2024 的公开 Ground Truth，方便复现实验和统一评测。CSV 是对公开标注的格式转换，不改变边界框数值。

## 官方来源

- HOTC2020：https://www.hsitracking.com/hot2020/
- HOTC2023：https://www.hsitracking.com/hot2023/
- HOTC2024：https://www.hsitracking.com/hot2024/

使用这些数据开展研究时，请引用对应挑战、数据集论文及官方页面要求的文献。

## CSV 结构

所有内置文件使用统一字段：

```text
dataset,modality,sequence,frame,x,y,width,height
```

- `dataset`：数据集标识。
- `modality`：`vis`、`rednir`、`nir` 或 `unspecified`。
- `sequence`：序列名称。
- `frame`：从 1 开始的帧号。
- `x,y,width,height`：目标框坐标与尺寸。

## 数据清单

| 数据集 | 序列数 | 帧数 | SHA-256 |
| --- | ---: | ---: | --- |
| HOTC2020 | 35 | 16,574 | `d4a3d42a219ea1a4c950a941a838a35bf628082da6978042416f6f98365c2333` |
| HOTC2023 | 87 | 28,430 | `ee118ff81332360b4c8bfb64cf7d6735cd0e3957dc6b25d833e2a07b9be17f2e` |
| HOTC2024 | 117 | 51,730 | `32f32bd9f36822d2937917c160ce133c8e2c2916d5171ba870debada117ee17b` |

机器可读清单位于 `resources/Groundtruth/datasets.json`。

原始标注中的离屏 `[-1,-1,-1,-1]` 和退化框保持原值，没有在格式转换时修补或删除；软件会在预检报告中明确显示这些记录，并按项目既有的 MATLAB 兼容评测口径处理。

## 权利说明

仓库的 MIT License 适用于本项目软件代码，不会改变上游数据集及标注的归属和使用条件。内置 CSV 的来源、引用和使用要求以对应 HOTC 官方页面为准。
