"""
MobileNetV2 2分类模型训练脚本
用于 ESP32-S3 部署的轻量化模型


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==================== 配置参数 ====================


class Config:
    # 数据集路径
    data_root = './data'  # 数据集根目录

    # 训练参数
    img_size = 320  # 图像大小（ESP32-S3 推荐小尺寸）
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    num_classes = 2  # 有 / 无

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 保存路径
    save_dir = './output'
    model_save_path = os.path.join(save_dir, 'image_classifier_mobilenetv2.pth')
    onnx_save_path = os.path.join(save_dir, 'image_classifier_320x320.onnx')

# ==================== 数据准备 ====================


def prepare_data_transforms():
    """数据增强和预处理"""
    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_dataloaders():
    """创建数据加载器"""
    train_transform, val_transform = prepare_data_transforms()

    # 使用 ImageFolder 加载数据
    # 期望的目录结构：
    # data/
    #   ├── train/
    #   │   ├── nag/    # 无图像 (标签 0，按字母顺序)
    #   │   └── pos/    # 有图像 (标签 1，按字母顺序)
    #   └── val/
    #       ├── nag/    # 无图像 (标签 0)
    #       └── pos/    # 有图像 (标签 1)
    #
    # 注意：ImageFolder 会按照文件夹名称的字母顺序自动分配标签
    #       nag < pos (字母顺序)，所以 nag=0, pos=1

    train_dataset = datasets.ImageFolder(
        os.path.join(Config.data_root, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(Config.data_root, 'val'),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")

    return train_loader, val_loader

# ==================== 模型定义 ====================


class imageClassifierMobileNetV2(nn.Module):
    """基于 MobileNetV2 的轻量化图像分类器"""

    def __init__(self, num_classes=2, pretrained=True):
        super(imageClassifierMobileNetV2, self).__init__()

        # 加载预训练的 MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # 替换分类头
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# ==================== 训练函数 ====================


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

# ==================== 导出 ONNX ====================


def export_to_onnx(model, save_path, img_size=320):
    """导出模型到 ONNX 格式"""
    model.eval()

    # 创建 dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(Config.device)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"模型已导出到: {save_path}")

    # 验证 ONNX 模型
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过！")

# ==================== 主训练流程 ====================


def main():
    """主训练流程"""

    # 创建输出目录
    os.makedirs(Config.save_dir, exist_ok=True)

    print("="*50)
    print("MobileNetV2 二分类训练")
    print(f"设备: {Config.device}")
    print(f"图像大小: {Config.img_size}x{Config.img_size}")
    print(f"批次大小: {Config.batch_size}")
    print(f"训练轮数: {Config.num_epochs}")
    print("="*50)

    # 准备数据
    train_loader, val_loader = create_dataloaders()

    # 创建模型
    model = imageClassifierMobileNetV2(
        num_classes=Config.num_classes,
        pretrained=True
    ).to(Config.device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 训练循环
    best_acc = 0.0

    for epoch in range(Config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{Config.num_epochs}")
        print(f"{'='*50}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.device
        )

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 学习率调整
        scheduler.step(val_acc)
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, Config.model_save_path)
            print(f"✓ 保存最佳模型，验证准确率: {best_acc:.2f}%")

    print("\n" + "="*50)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("="*50)

    # 加载最佳模型并导出 ONNX
    checkpoint = torch.load(Config.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n导出 ONNX 模型...")
    export_to_onnx(model, Config.onnx_save_path, Config.img_size)

    print("\n完成！")
    print(f"PyTorch 模型: {Config.model_save_path}")
    print(f"ONNX 模型: {Config.onnx_save_path}")


if __name__ == '__main__':
    main()
