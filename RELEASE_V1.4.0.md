# HSI Tracking Plot Studio V1.4.0

本版本与 V1.3.0 并行发布，不覆盖旧版目录。

## 下载与运行

下载 `HSITrackingPlotStudio-v1.4.0-windows-x64.zip`，解压后运行顶层目录中的 `HSITrackingPlotStudio.exe`。发布附件中的 `SHA256SUMS.txt` 可用于验证下载完整性。

## 新增功能

- Ground Truth 支持逐序列 TXT 目录和合并 CSV 文件。
- 跟踪结果支持原有 TXT 目录和 `ID,x,y,width,height` 提交 CSV。
- 内置 HOTC2020、HOTC2023、HOTC2024 Ground Truth，可在数据源配置页一键选择。
- 内置 CSV 使用 `dataset,modality,sequence,frame,x,y,width,height` 明确字段。
- CSV 导入检查表头、ID、重复帧、帧号连续性、NaN 完整性和无穷值。

## 兼容性

- 旧版 TXT 导入、项目文件、指标计算和绘图流程保持兼容。
- CSV 在应用缓存中生成按内容哈希隔离的逐序列只读视图，不修改源 CSV。

## 验证

- 完整自动化测试：303 项通过。
- Windows 目录版启动冒烟测试通过。
- 内置数据清单包含逐文件 SHA-256 校验值。

## 系统要求

- Windows 10/11 x64。
- 目录版无需另行安装 Python。
