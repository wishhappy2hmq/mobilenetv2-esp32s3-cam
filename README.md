# ESP32-S3 MobileNetV2 实时图像分类系统

基于 ESP32-S3 和 MobileNetV2 的实时图像分类系统，支持摄像头采集、HTTP 视频流和 Web 界面显示。

## 项目特性

- ✅ **实时视频流**：通过 HTTP 服务器提供 MJPEG 视频流
- ✅ **高效推理**：每 10 帧执行一次图像分类推理
- ✅ **Web 界面**：实时显示分类结果和置信度
- ✅ **WiFi 连接**：支持 WiFi 连接和远程访问
- ✅ **二分类模型**：基于 MobileNetV2 训练的二分类模型（none/exist）

## 硬件要求

- **开发板**：ESP32-S3-EYE 或兼容板
- **内存**：至少 8MB PSRAM
- **摄像头**：OV2640 或兼容摄像头
- **WiFi**：支持 2.4GHz WiFi

## 模型信息

### 模型架构
- **基础架构**：MobileNetV2
- **输入尺寸**：320x320x3 (RGB)
- **输出类别**：2 类（none, exist）
- **量化方式**：INT8 量化
- **模型格式**：ESP-DL (.espdl)

### 模型训练与转换

模型是基于 MobileNetV2 架构训练的二分类模型，完整的训练和转换流程位于 [`mobilenetv2_train`](mobilenetv2_train/) 目录：

1. **训练**：使用 PyTorch 训练 MobileNetV2 模型
2. **导出**：转换为 ONNX 格式
3. **量化**：使用 ESP-DL 工具进行 INT8 量化
4. **部署**：生成 `.espdl` 模型文件用于 ESP32-S3

详细的训练和量化步骤请参考：
- [训练代码](mobilenetv2_train/train_mobilenetv2.py)
- [量化脚本](mobilenetv2_train/quantize_my_model.py)
- [量化指南](mobilenetv2_train/QUANTIZATION_GUIDE.md)

### 模型性能
- **推理时间**：~300-500ms (ESP32-S3 @ 240MHz)
- **模型大小**：~3.5MB (量化后)
- **精度**：INT8 量化，接近浮点精度

## 快速开始

### 1. 环境准备

```bash
# 安装 ESP-IDF v5.5.1
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.5.1
./install.sh esp32s3
. ./export.sh
```

### 2. 配置项目

编辑 [main/app_main.cpp](main/app_main.cpp:36-38) 修改 WiFi 配置：

```cpp
#define WIFI_SSID      "Your_WiFi_SSID"
#define WIFI_PASSWORD  "Your_WiFi_Password"
```

### 3. 编译和烧录

```bash
# 设置目标芯片
idf.py set-target esp32s3

# 编译项目
idf.py build

# 烧录到设备
idf.py -p COM3 flash monitor
```

### 4. 访问 Web 界面

1. 设备启动后，从串口监视器获取 IP 地址
2. 在浏览器中访问：`http://<ESP32_IP_ADDRESS>`
3. 查看实时视频流和分类结果

## 项目结构

```
mobilenetv2_esp32/
├── main/                          # 主程序代码
│   ├── app_main.cpp              # 主应用程序入口
│   └── CMakeLists.txt
├── imagenet_cls/                  # 分类模型组件
│   ├── models/s3/                # ESP32-S3 模型文件
│   │   └── image_classifier_320x320_int8.espdl
│   ├── imagenet_cls.hpp          # 分类器头文件
│   └── README.md
├── mobilenetv2_train/             # 模型训练和转换代码
│   ├── train_mobilenetv2.py      # 训练脚本
│   ├── quantize_my_model.py      # 量化脚本
│   ├── test.py                   # 测试脚本
│   ├── QUANTIZATION_GUIDE.md     # 量化详细指南
│   └── QUICK_START_QUANTIZATION.md
├── esp-dl/                        # ESP-DL 深度学习库
├── managed_components/            # ESP 组件依赖
├── partitions.csv                 # 分区表
├── sdkconfig                      # ESP-IDF 配置
├── CMakeLists.txt                 # CMake 配置
└── README.md                      # 本文件

```

## 配置说明

### 推理配置

在 [app_main.cpp](main/app_main.cpp:41) 中修改推理间隔：

```cpp
#define INFERENCE_INTERVAL 10  // 每 N 帧推理一次
```

### 摄像头配置

摄像头参数在 [app_main.cpp](main/app_main.cpp:160-190) 中配置：
- 分辨率：QVGA (320x240)
- 格式：JPEG
- 质量：12
- 帧缓冲：2 个（位于 PSRAM）

### 分区配置

分区表定义在 [partitions.csv](partitions.csv)，包含：
- Factory app 分区：用于主程序
- Storage 分区：用于配置存储

## Web 界面功能

- 📹 实时 MJPEG 视频流
- 📊 分类结果和置信度显示
- ⏱️ 推理耗时显示
- 🔄 自动更新（500ms 刷新率）

## 性能指标

| 指标 | 数值 |
|------|------|
| 视频帧率 | ~30 FPS |
| 推理间隔 | 每 10 帧 |
| 图像输入 | 320x240 JPEG |
| 模型输入 | 320x320 RGB |

## 故障排除

### 编译错误

如果遇到 GCC 内部编译器错误（Segmentation fault），请：
1. 完全清理构建目录：`idf.py fullclean`
2. 重新编译：`idf.py build`
3. 如果仍然失败，尝试限制并行编译：`idf.py build -j2`

### WiFi 连接失败

1. 检查 WiFi 凭据是否正确
2. 确认 WiFi 是 2.4GHz 频段
3. 检查串口监视器的错误日志

### 摄像头初始化失败

1. 检查摄像头连接
2. 确认使用的是 ESP32-S3-EYE 或兼容板
3. 确保 PSRAM 已启用（在 sdkconfig 中）

### 推理速度慢

1. 确认 CPU 频率设置为 240MHz
2. 检查 PSRAM 是否正确启用
3. 考虑增加 `INFERENCE_INTERVAL` 值

## 开发指南

### 修改模型

1. 训练新模型（参考 [mobilenetv2_train](mobilenetv2_train/)）
2. 转换为 ONNX 并量化为 `.espdl`
3. 替换 `imagenet_cls/models/s3/image_classifier_320x320_int8.espdl`
4. 更新类别标签（如有必要）

### 自定义 Web 界面

Web 界面 HTML 代码位于 [app_main.cpp](main/app_main.cpp:323-404)，可以直接修改。

### 添加新功能

- HTTP 处理器：在 `app_main.cpp` 中添加新的 `httpd_uri_t`
- 推理逻辑：修改 `inference_task` 函数

## 参考资料

- [ESP-IDF 编程指南](https://docs.espressif.com/projects/esp-idf/zh_CN/latest/)
- [ESP-DL 文档](https://github.com/espressif/esp-dl)
- [ESP32-Camera 组件](https://github.com/espressif/esp32-camera)
- [MobileNetV2 论文](https://arxiv.org/abs/1801.04381)

## 许可证

本项目使用 Apache License 2.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请提交 Issue。
