# 快速开始：量化你的 MobileNetV2 模型

## 🚀 三步完成量化

### 步骤 1: 安装 ESP-PPQ 工具包

```bash
pip install esp-ppq
```

如果上面的命令失败，尝试：
```bash
pip install git+https://github.com/espressif/esp-ppq.git
```

### 步骤 2: 配置量化参数

打开 `quantize_my_model.py` 文件，修改 `QuantConfig` 类中的参数：

```python
class QuantConfig:
    # 1. 设置你的 ONNX 模型路径
    ONNX_PATH = "./output/face_classifier_96x96.onnx"  # 👈 修改这里

    # 2. 设置输出路径
    ESPDL_MODEL_PATH = "./output/face_classifier_320x320_int8.espdl"

    # 3. 设置输入尺寸（必须与训练时一致）
    INPUT_SHAPE = [3, 320, 320]  # 👈 你的是 320x320

    # 4. 设置校准数据集路径
    CALIB_DIR = "./data/train"  # 👈 使用训练集即可

    # 5. 选择目标平台
    TARGET = "esp32p4"  # 或 "esp32s3"
```

### 步骤 3: 运行量化

```bash
python quantize_my_model.py
```

量化完成后会生成 `.espdl` 文件，可以直接部署到 ESP32！

## ⚙️ 配置详解

### 必须修改的参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `ONNX_PATH` | 训练好的 ONNX 模型路径 | `"./output/face_classifier_96x96.onnx"` |
| `INPUT_SHAPE` | 输入图像尺寸 [C,H,W] | `[3, 320, 320]` |
| `CALIB_DIR` | 校准数据集目录 | `"./data/train"` |
| `TARGET` | 目标 ESP32 芯片 | `"esp32p4"` 或 `"esp32s3"` |

### 可选配置参数

| 参数 | 说明 | 默认值 | 建议 |
|------|------|--------|------|
| `CALIB_SIZE` | 校准样本数量 | `512` | 500-1000 |
| `NUM_OF_BITS` | 量化位数 | `8` | 8 或 16 |
| `BATCH_SIZE` | 批次大小 | `32` | 根据内存调整 |
| `CALIB_STEPS` | 校准步数 | `16` | 10-32 |
| `OPTIM_QUANT_METHOD` | 优化方法 | `None` | 见下文 |

## 🎯 量化优化方法

### 1. 默认量化（推荐先尝试）
```python
OPTIM_QUANT_METHOD = None
```
- 最快，适合大多数情况
- 如果精度满足要求，就用这个

### 2. 层级均衡量化（精度优化）
```python
OPTIM_QUANT_METHOD = ["LayerwiseEqualization_quantization"]
```
- 如果默认量化精度损失>5%，使用此方法
- 量化时间稍长，但精度更好

### 3. 混合精度量化（高级）
```python
OPTIM_QUANT_METHOD = ["MixedPrecision_quantization"]
```
- 对误差大的层使用 16-bit，其他层 8-bit
- 需要手动指定哪些层用 16-bit

## 📊 校准数据集准备

### 方法 1: 使用训练集（推荐）
```python
CALIB_DIR = "./data/train"
CALIB_SIZE = 512  # 从训练集中随机选取 512 张
```

### 方法 2: 准备单独的校准集
创建一个包含有代表性样本的文件夹：
```
calib_data/
  ├── pos/  # 500 张有图像的样本
  └── nag/  # 500 张无图像的样本
```

然后设置：
```python
CALIB_DIR = "./calib_data"
```

### 方法 3: 使用随机数据（仅测试）
```python
USE_RANDOM_CALIB = True  # ⚠️ 实际使用时必须改为 False
```

## 🔍 量化结果评估

量化完成后，会显示：
- 量化模型保存路径
- 模型文件大小
- 各层的量化误差报告

### 精度损失参考
- **< 1%**: 优秀，可直接使用
- **1-3%**: 良好，可接受
- **3-5%**: 尚可，考虑优化
- **> 5%**: 需要优化，尝试其他量化方法

## 🛠️ 常见问题

### Q: 量化报错 "FileNotFoundError: ONNX 模型不存在"
**A:** 确保已经先运行训练脚本并导出了 ONNX 模型：
```bash
python train_mobilenetv2.py
```

### Q: 量化报错 "校准数据集目录不存在"
**A:** 检查 `CALIB_DIR` 路径是否正确，确保目录存在且包含数据

### Q: 量化后精度损失太大（>5%）
**A:** 尝试以下方法：
1. 增加 `CALIB_SIZE` 到 1000
2. 使用 `LayerwiseEqualization_quantization`
3. 使用混合精度量化（部分层 16-bit）

### Q: 量化需要多长时间？
**A:**
- 默认量化: 3-5 分钟
- 层级均衡: 5-10 分钟
- 混合精度: 5-15 分钟

### Q: .espdl 文件如何部署到 ESP32？
**A:**
1. 使用 ESP-DL 库（需要在 ESP32 项目中集成）
2. 将 .espdl 文件复制到 ESP32 的文件系统
3. 使用 ESP-DL API 加载模型并进行推理

## 📝 完整示例

```bash
# 1. 安装工具
pip install esp-ppq

# 2. 修改 quantize_my_model.py 中的配置
# - ONNX_PATH = "./output/face_classifier_96x96.onnx"
# - INPUT_SHAPE = [3, 320, 320]
# - CALIB_DIR = "./data/train"
# - TARGET = "esp32p4"

# 3. 运行量化
python quantize_my_model.py

# 4. 检查输出
# 成功后会生成: ./output/face_classifier_320x320_int8.espdl
```

## 🎉 下一步

量化成功后：
1. ✅ 查看量化报告，确认精度损失可接受
2. ✅ 在验证集上测试量化模型
3. ✅ 将 .espdl 文件部署到 ESP32-S3/P4
4. ✅ 在设备上进行实际推理测试

Good luck! 🚀
