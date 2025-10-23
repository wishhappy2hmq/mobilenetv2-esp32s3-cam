"""
MobileNetV2 模型测试脚本
测试训练好的模型性能
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import onnxruntime as ort
import os

# ==================== 测试配置参数 ====================

class TestConfig:
    """测试配置类 - 在这里修改所有测试参数"""

    # 模型路径
    pytorch_model = './output/image_classifier_mobilenetv2.pth'
    onnx_model = './output/image_classifier_320x320.onnx'

    # 测试图片路径（单张测试）
    test_image = './testimages/image_20251023_163810_093.jpg'  # 修改为你的测试图片路径

    # 测试图片文件夹路径（批量测试）
    test_image_dir = './testimages'  # 批量测试时使用

    # 图像尺寸
    img_size = 320

    # 测试模式
    batch_test = True  # True: 批量测试, False: 单张测试

    # 是否保存结果图像
    save_results = True

    # 是否测试 PyTorch 模型
    test_pytorch = False

    # 是否测试 ONNX 模型
    test_onnx = True

    # 是否显示模型大小分析
    show_model_analysis = True

# ==================== 模型定义 ====================


class imageClassifierMobileNetV2(nn.Module):
    """基于 MobileNetV2 的轻量化图像分类器"""

    def __init__(self, num_classes=2):
        super(imageClassifierMobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# ==================== 图像预处理 ====================


def preprocess_image(image_path, img_size=320):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

    return image_tensor


def save_result_image(image_path, predicted_class, confidence, save_path):
    """
    在原图上绘制分类结果并保存

    Args:
        image_path: 原始图像路径
        predicted_class: 预测类别 (0 或 1)
        confidence: 置信度
        save_path: 保存路径
    """
    # 读取原始图像
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # 准备文本
    # 注意：ImageFolder 按字母顺序分配标签
    # nag (无图像) -> 0, pos (有图像) -> 1
    class_name = '无图像' if predicted_class == 0 else '有图像'
    text = f'{class_name}\n置信度: {confidence*100:.2f}%'

    # 获取图像尺寸
    width, height = image.size

    # 尝试加载中文字体，如果失败则使用默认字体
    font_size = max(20, int(min(width, height) * 0.05))  # 根据图像大小调整字体
    try:
        # Windows 系统中文字体
        font = ImageFont.truetype("msyh.ttc", font_size)  # 微软雅黑
    except:
        try:
            font = ImageFont.truetype("simsun.ttc", font_size)  # 宋体
        except:
            # 如果没有中文字体，使用默认字体
            font = ImageFont.load_default()

    # 绘制半透明背景
    # 使用文本边界框来确定背景大小
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 背景位置（左上角）
    margin = 10
    bg_x0 = margin
    bg_y0 = margin
    bg_x1 = margin + text_width + 20
    bg_y1 = margin + text_height + 20

    # 创建一个临时图层用于半透明效果
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # 根据预测结果选择背景颜色
    # nag (无图像) -> 0, pos (有图像) -> 1
    if predicted_class == 1:  # pos - 有图像
        bg_color = (0, 255, 0, 200)  # 绿色，半透明
        text_color = (0, 100, 0, 255)  # 深绿色文字
    else:  # nag - 无图像
        bg_color = (255, 0, 0, 200)  # 红色，半透明
        text_color = (139, 0, 0, 255)  # 深红色文字

    # 绘制背景矩形
    overlay_draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=bg_color)

    # 合并图层
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')

    # 在合并后的图像上绘制文字
    draw = ImageDraw.Draw(image)
    draw.text((margin + 10, margin + 10), text, fill=text_color[:3], font=font)

    # 保存图像
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    image.save(save_path)
    print(f"✓ 结果图像已保存到: {save_path}")

# ==================== PyTorch 模型测试 ====================


def test_pytorch_model(model_path, image_path, img_size=320, save_result=True):
    """测试 PyTorch 模型"""
    print("="*50)
    print("测试 PyTorch 模型")
    print("="*50)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = imageClassifierMobileNetV2(num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  设备: {device}")
    print(f"  最佳准确率: {checkpoint.get('best_acc', 0):.2f}%")

    # 预处理图像
    image_tensor = preprocess_image(image_path, img_size).to(device)

    # 推理
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(image_tensor)

        # 计时
        start_time = time.time()
        num_runs = 100

        for _ in range(num_runs):
            output = model(image_tensor)

        end_time = time.time()

        # 结果
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms

    print(f"\n推理结果:")
    print(f"  预测类别: {'无图像' if predicted_class == 0 else '有图像'}")
    print(f"  置信度: {confidence*100:.2f}%")
    print(f"  平均推理时间: {avg_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.1f}")

    # 显示两类概率
    print(f"\n类别概率:")
    print(f"  无图像 (nag): {probabilities[0][0].item()*100:.2f}%")
    print(f"  有图像 (pos): {probabilities[0][1].item()*100:.2f}%")

    # 保存结果图像
    if save_result:
        # 生成保存路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        save_dir = './test_results'
        save_path = os.path.join(save_dir, f'{base_name}_result{ext}')

        save_result_image(image_path, predicted_class, confidence, save_path)

    return predicted_class, confidence, avg_time

# ==================== ONNX 模型测试 ====================


def test_onnx_model(onnx_path, image_path, img_size=320, save_result=True):
    """测试 ONNX 模型"""
    print("\n" + "="*50)
    print("测试 ONNX 模型")
    print("="*50)

    # 创建 ONNX Runtime 会话
    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"✓ ONNX 模型加载成功")
    print(f"  输入: {input_name}")
    print(f"  输出: {output_name}")

    # 预处理图像
    image_tensor = preprocess_image(image_path, img_size)
    image_numpy = image_tensor.numpy()

    # 推理
    # 预热
    for _ in range(10):
        _ = session.run([output_name], {input_name: image_numpy})

    # 计时
    start_time = time.time()
    num_runs = 100

    for _ in range(num_runs):
        output = session.run([output_name], {input_name: image_numpy})

    end_time = time.time()

    # 结果
    logits = output[0][0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    avg_time = (end_time - start_time) / num_runs * 1000  # ms

    print(f"\n推理结果:")
    print(f"  预测类别: {'无图像' if predicted_class == 0 else '有图像'}")
    print(f"  置信度: {confidence*100:.2f}%")
    print(f"  平均推理时间: {avg_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.1f}")

    # 显示两类概率
    print(f"\n类别概率:")
    print(f"  无图像 (nag): {probabilities[0]*100:.2f}%")
    print(f"  有图像 (pos): {probabilities[1]*100:.2f}%")

    # 保存结果图像
    if save_result:
        # 生成保存路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        save_dir = './test_results'
        save_path = os.path.join(save_dir, f'{base_name}_onnx_result{ext}')

        save_result_image(image_path, int(predicted_class), float(confidence), save_path)

    return predicted_class, confidence, avg_time

# ==================== 模型大小分析 ====================


def analyze_model_size(pytorch_path, onnx_path):
    """分析模型大小"""
    print("\n" + "="*50)
    print("模型大小分析")
    print("="*50)

    import os

    if os.path.exists(pytorch_path):
        pytorch_size = os.path.getsize(pytorch_path) / (1024 * 1024)  # MB
        print(f"PyTorch 模型: {pytorch_size:.2f} MB")

    if os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"ONNX 模型: {onnx_size:.2f} MB")

    # 参数量
    model = imageClassifierMobileNetV2(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"\n参数量:")
    print(f"  总参数: {total_params/1e6:.2f}M")
    print(f"  可训练参数: {trainable_params/1e6:.2f}M")

# ==================== 批量测试 ====================


def batch_test(model_path, test_image_dir, img_size=320, save_results=True):
    """批量测试多张图片"""
    print("\n" + "="*50)
    print("批量测试")
    print("="*50)

    from glob import glob

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = imageClassifierMobileNetV2(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 获取所有图片
    image_files = glob(os.path.join(test_image_dir, '*.jpg')) + \
        glob(os.path.join(test_image_dir, '*.png'))

    print(f"找到 {len(image_files)} 张图片\n")

    correct = 0
    total = 0
    save_dir = './test_results/batch'

    for img_path in image_files:
        image_tensor = preprocess_image(img_path, img_size).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted].item()

        # 假设文件名包含标签信息 (例如: pos_001.jpg, nag_002.jpg)
        # nag (无图像) -> 0, pos (有图像) -> 1
        true_label = 1 if 'pos' in os.path.basename(img_path).lower() else 0

        is_correct = (predicted == true_label)
        correct += is_correct
        total += 1

        status = "✓" if is_correct else "✗"
        pred_label = "无图像" if predicted == 0 else "有图像"
        true_label_str = "无图像" if true_label == 0 else "有图像"

        print(
            f"{status} {os.path.basename(img_path):30s} | 预测: {pred_label:6s} | 真实: {true_label_str:6s} | 置信度: {confidence*100:.2f}%")

        # 保存结果图像
        if save_results:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            save_path = os.path.join(save_dir, f'{base_name}_result{ext}')
            save_result_image(img_path, predicted, confidence, save_path)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")

    if save_results:
        print(f"\n✓ 所有结果图像已保存到: {save_dir}")

# ==================== 主函数 ====================


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试 MobileNetV2 图像分类模型')
    parser.add_argument('--pytorch-model', type=str, default='./output/image_classifier_mobilenetv2.pth',
                        help='PyTorch 模型路径')
    parser.add_argument('--onnx-model', type=str, default='./output/image_classifier_320x320.onnx',
                        help='ONNX 模型路径')
    parser.add_argument('--image', type=str,
                        help='测试图片路径')
    parser.add_argument('--img-size', type=int, default=320,
                        help='图像大小')
    parser.add_argument('--batch-test', action='store_true',
                        help='批量测试模式')

    args = parser.parse_args()

    # 检查是否提供了图片路径
    if not args.image:
        parser.error("请提供测试图片路径 (--image)")

    # 单张图片测试
    if not args.batch_test:
        # 测试 PyTorch 模型
        if os.path.exists(args.pytorch_model):
            test_pytorch_model(args.pytorch_model, args.image, args.img_size)

        # 测试 ONNX 模型
        if os.path.exists(args.onnx_model):
            test_onnx_model(args.onnx_model, args.image, args.img_size)

        # 分析模型大小
        analyze_model_size(args.pytorch_model, args.onnx_model)

    # 批量测试
    else:
        batch_test(args.pytorch_model, args.image, args.img_size)


if __name__ == '__main__':
    import sys

    # 如果没有命令行参数，使用配置文件中的参数
    if len(sys.argv) == 1:
        print("="*50)
        print("MobileNetV2 模型测试")
        print("="*50)
        print("\n使用配置参数:")
        print(f"  模式: {'批量测试' if TestConfig.batch_test else '单张测试'}")

        if TestConfig.batch_test:
            print(f"  测试文件夹: {TestConfig.test_image_dir}")
            print(f"  图像尺寸: {TestConfig.img_size}x{TestConfig.img_size}")
            print(f"  保存结果: {'是' if TestConfig.save_results else '否'}")
            print("="*50)

            if os.path.exists(TestConfig.pytorch_model):
                batch_test(
                    TestConfig.pytorch_model,
                    TestConfig.test_image_dir,
                    TestConfig.img_size,
                    TestConfig.save_results
                )
            else:
                print(f"\n错误: 找不到模型文件 {TestConfig.pytorch_model}")
        else:
            print(f"  测试图片: {TestConfig.test_image}")
            print(f"  图像尺寸: {TestConfig.img_size}x{TestConfig.img_size}")
            print(f"  保存结果: {'是' if TestConfig.save_results else '否'}")
            print(f"  测试 PyTorch: {'是' if TestConfig.test_pytorch else '否'}")
            print(f"  测试 ONNX: {'是' if TestConfig.test_onnx else '否'}")
            print("="*50)

            if not os.path.exists(TestConfig.test_image):
                print(f"\n错误: 找不到测试图片 {TestConfig.test_image}")
                sys.exit(1)

            # 测试 PyTorch 模型
            if TestConfig.test_pytorch and os.path.exists(TestConfig.pytorch_model):
                test_pytorch_model(
                    TestConfig.pytorch_model,
                    TestConfig.test_image,
                    TestConfig.img_size,
                    TestConfig.save_results
                )
            elif TestConfig.test_pytorch:
                print(f"\n警告: 找不到 PyTorch 模型文件 {TestConfig.pytorch_model}")

            # 测试 ONNX 模型
            if TestConfig.test_onnx and os.path.exists(TestConfig.onnx_model):
                test_onnx_model(
                    TestConfig.onnx_model,
                    TestConfig.test_image,
                    TestConfig.img_size,
                    TestConfig.save_results
                )
            elif TestConfig.test_onnx:
                print(f"\n警告: 找不到 ONNX 模型文件 {TestConfig.onnx_model}")

            # 模型大小分析
            if TestConfig.show_model_analysis:
                analyze_model_size(TestConfig.pytorch_model, TestConfig.onnx_model)

    # 如果有命令行参数，使用原来的命令行参数模式
    else:
        main()
