"""
量化 MobileNetV2 二分类模型为 ESP32 可用的 ESPDL 格式

使用方法:
1. 确保已安装 esp-ppq: pip install esp-ppq
2. 准备校准数据集（建议从训练集中选取500-1000张有代表性的图片）
3. 修改下面的配置参数
4. 运行: python quantize_my_model.py
"""

import os
import torch
from typing import Iterable, Tuple
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

# 检查 esp_ppq 是否安装
try:
    from esp_ppq import QuantizationSettingFactory, QuantizationSetting
    from esp_ppq.api import espdl_quantize_onnx, get_target_platform
except ImportError:
    print("错误: 未安装 esp-ppq 工具包")
    print("请运行: pip install esp-ppq")
    print("或从源码安装: pip install git+https://github.com/espressif/esp-ppq.git")
    exit(1)

# ==================== 配置参数 ====================

class QuantConfig:
    """量化配置参数 - 请根据你的需求修改这些参数"""

    # ========== 模型配置 ==========
    # 训练好的 ONNX 模型路径
    ONNX_PATH = "./output/image_classifier_96x96.onnx"

    # 输出的 ESPDL 模型路径
    ESPDL_MODEL_PATH = "./output/image_classifier_320x320_int8.espdl"

    # 输入图像尺寸 (与训练时一致)
    INPUT_SHAPE = [3, 320, 320]  # [C, H, W]

    # ========== 校准数据集配置 ==========
    # 校准数据集路径（建议使用训练集的子集）
    # 目录结构应该是:
    #   CALIB_DIR/
    #     ├── pos/  (有图像的样本)
    #     └── nag/  (无图像的样本)
    CALIB_DIR = "./data/train"  # 如果没有单独的校准集，可以使用训练集

    # 是否使用随机数据进行校准（仅用于测试，实际使用请设为 False）
    USE_RANDOM_CALIB = False

    # 校准数据集大小（从数据集中选取的样本数量）
    CALIB_SIZE = 50  

    # ========== 量化配置 ==========
    # 目标平台: 'esp32s3' 或 'esp32p4'
    TARGET = "esp32s3"

    # 量化位数: 8 或 16
    NUM_OF_BITS = 8

    # 量化优化方法（可选）:
    # None: 默认量化（推荐先尝试）
    # ["LayerwiseEqualization_quantization"]: 层级均衡量化（如果默认量化精度损失大）
    # ["MixedPrecision_quantization"]: 混合精度量化（某些层用16-bit）
    OPTIM_QUANT_METHOD = None  # 或 ["LayerwiseEqualization_quantization"]

    # ========== 其他配置 ==========
    BATCH_SIZE = 32
    DEVICE = "cpu"  # 'cuda' 或 'cpu'
    CALIB_STEPS = 16  # 校准步数（使用多少个 batch 进行校准）

# ==================== 辅助函数 ====================

def collate_fn1(x: Tuple) -> torch.Tensor:
    """合并 batch 数据"""
    return torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0)


def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
    """将数据移动到指定设备"""
    return batch.to(QuantConfig.DEVICE)


def quant_setting_mobilenet_v2(
    onnx_path: str,
    optim_quant_method: list = None,
) -> Tuple[QuantizationSetting, str]:
    """
    创建 MobileNetV2 的量化设置

    Args:
        onnx_path: ONNX 模型路径
        optim_quant_method: 量化优化方法列表

    Returns:
        量化设置和 ONNX 路径的元组
    """
    # 创建基础量化设置
    quant_setting = QuantizationSettingFactory.espdl_setting()

    if optim_quant_method is not None:
        if "MixedPrecision_quantization" in optim_quant_method:
            print("使用混合精度量化 (Mixed Precision)")
            # 注意: 这些层名称是 MobileNetV2 特定的，可能需要根据你的模型结构调整
            # 你可以通过量化报告查看哪些层的误差较大，然后将它们分配到 16-bit
            # quant_setting.dispatching_table.append(
            #     "layer_name",  # 替换为实际的层名称
            #     get_target_platform(QuantConfig.TARGET, 16),
            # )
            pass

        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            print("使用层级均衡量化 (Layerwise Equalization)")
            # 启用层级均衡
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = 4
            quant_setting.equalization_setting.value_threshold = 0.4
            quant_setting.equalization_setting.opt_level = 2
            quant_setting.equalization_setting.interested_layers = None
        else:
            raise ValueError(
                "不支持的量化方法。请使用 'MixedPrecision_quantization' 或 'LayerwiseEqualization_quantization'"
            )

    return quant_setting, onnx_path


def load_calibration_dataset() -> DataLoader:
    """
    加载校准数据集

    Returns:
        DataLoader 对象
    """
    if QuantConfig.USE_RANDOM_CALIB:
        print("警告: 使用随机数据进行校准（仅用于测试）")

        def load_random_data() -> Iterable:
            return [
                torch.rand(size=QuantConfig.INPUT_SHAPE)
                for _ in range(QuantConfig.BATCH_SIZE * QuantConfig.CALIB_STEPS)
            ]

        return DataLoader(
            dataset=load_random_data(),
            batch_size=QuantConfig.BATCH_SIZE,
            shuffle=False,
        )

    # 使用真实数据集
    if not os.path.exists(QuantConfig.CALIB_DIR):
        raise FileNotFoundError(
            f"校准数据集目录不存在: {QuantConfig.CALIB_DIR}\n"
            "请设置正确的 CALIB_DIR 路径，或将 USE_RANDOM_CALIB 设为 True"
        )

    print(f"从目录加载校准数据集: {QuantConfig.CALIB_DIR}")

    # 数据预处理（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((QuantConfig.INPUT_SHAPE[1], QuantConfig.INPUT_SHAPE[2])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(QuantConfig.CALIB_DIR, transform=transform)

    # 限制数据集大小
    if len(dataset) > QuantConfig.CALIB_SIZE:
        print(f"数据集大小: {len(dataset)}, 使用前 {QuantConfig.CALIB_SIZE} 个样本进行校准")
        dataset = Subset(dataset, indices=list(range(QuantConfig.CALIB_SIZE)))
    else:
        print(f"数据集大小: {len(dataset)}")

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=QuantConfig.BATCH_SIZE,
        shuffle=True,  # 随机选择更有代表性
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn1,
    )

    return dataloader


def quantize_model():
    """
    量化模型主函数
    """
    print("=" * 60)
    print("MobileNetV2 模型量化")
    print("=" * 60)

    # 检查输入模型是否存在
    if not os.path.exists(QuantConfig.ONNX_PATH):
        raise FileNotFoundError(
            f"ONNX 模型不存在: {QuantConfig.ONNX_PATH}\n"
            "请先运行训练脚本生成 ONNX 模型"
        )

    print(f"\n配置信息:")
    print(f"  输入模型: {QuantConfig.ONNX_PATH}")
    print(f"  输出模型: {QuantConfig.ESPDL_MODEL_PATH}")
    print(f"  输入尺寸: {QuantConfig.INPUT_SHAPE}")
    print(f"  目标平台: {QuantConfig.TARGET}")
    print(f"  量化位数: {QuantConfig.NUM_OF_BITS}-bit")
    print(f"  校准数据: {QuantConfig.CALIB_DIR}")
    print(f"  优化方法: {QuantConfig.OPTIM_QUANT_METHOD or '默认'}")

    # 创建输出目录
    os.makedirs(os.path.dirname(QuantConfig.ESPDL_MODEL_PATH), exist_ok=True)

    # 加载校准数据集
    print("\n" + "=" * 60)
    print("步骤 1: 加载校准数据集")
    print("=" * 60)
    dataloader = load_calibration_dataset()

    # 创建量化设置
    print("\n" + "=" * 60)
    print("步骤 2: 配置量化参数")
    print("=" * 60)
    quant_setting, onnx_path = quant_setting_mobilenet_v2(
        QuantConfig.ONNX_PATH,
        QuantConfig.OPTIM_QUANT_METHOD
    )

    # 执行量化
    print("\n" + "=" * 60)
    print("步骤 3: 执行模型量化（这可能需要几分钟）")
    print("=" * 60)

    try:
        quant_ppq_graph = espdl_quantize_onnx(
            onnx_import_file=onnx_path,
            espdl_export_file=QuantConfig.ESPDL_MODEL_PATH,
            calib_dataloader=dataloader,
            calib_steps=QuantConfig.CALIB_STEPS,
            input_shape=[1] + QuantConfig.INPUT_SHAPE,
            target=QuantConfig.TARGET,
            num_of_bits=QuantConfig.NUM_OF_BITS,
            collate_fn=collate_fn2,
            setting=quant_setting,
            device=QuantConfig.DEVICE,
            error_report=True,  # 生成误差报告
            skip_export=False,
            export_test_values=False,
            verbose=1,  # 显示详细信息
        )

        print("\n" + "=" * 60)
        print("量化成功！")
        print("=" * 60)
        print(f"\n量化模型已保存到: {QuantConfig.ESPDL_MODEL_PATH}")

        # 显示文件大小
        if os.path.exists(QuantConfig.ESPDL_MODEL_PATH):
            size_mb = os.path.getsize(QuantConfig.ESPDL_MODEL_PATH) / (1024 * 1024)
            print(f"模型大小: {size_mb:.2f} MB")

        print("\n下一步:")
        print("1. 检查量化误差报告，确保精度损失在可接受范围内")
        print("2. 如果精度损失较大，可以尝试:")
        print("   - 增加校准数据集大小")
        print("   - 使用 LayerwiseEqualization_quantization")
        print("   - 使用 MixedPrecision_quantization")
        print("3. 将 .espdl 模型部署到 ESP32 设备上进行测试")

    except Exception as e:
        print(f"\n量化失败: {e}")
        print("\n常见问题排查:")
        print("1. 确保 ONNX 模型路径正确")
        print("2. 确保校准数据集路径正确且包含数据")
        print("3. 确保安装了 esp-ppq 及其依赖")
        print("4. 检查输入尺寸是否与训练时一致")
        raise


if __name__ == "__main__":
    quantize_model()
