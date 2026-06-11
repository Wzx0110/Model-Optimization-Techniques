from optimize_pipeline import (
    apply_ptq, apply_structured_pruning, benchmark,
    distillation_loss, evaluate_acc
)
from cnn_pytorch import BasicCNN, get_dataloaders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def visualize_soft_labels(teacher, dataloader, device):
    print("\n" + "="*55)
    print("Visualizing Soft Labels (T=1 vs T=4)")
    print("="*55)

    teacher.eval()
    images, labels = next(iter(dataloader))
    images = images[:4].to(device)
    labels = labels[:4]

    with torch.no_grad():
        teacher_inputs = F.interpolate(images, size=(
            224, 224), mode='bilinear', align_corners=False)
        logits = teacher(teacher_inputs)

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))

    # 反正規化顯示原始圖片
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(device)

    for i in range(4):
        # 原圖
        img = images[i] * std + mean
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"True Label: {classes[labels[i]]}", fontweight='bold')
        axes[i, 0].axis('off')

        # Softmax T=1 vs T=4
        for j, T in enumerate([1, 4]):
            probs = F.softmax(logits[i] / T, dim=0).cpu().numpy()
            color = 'skyblue' if T == 1 else 'lightgreen'
            axes[i, j+1].bar(classes, probs, color=color, edgecolor='black')
            axes[i, j+1].set_title(f"Teacher Softmax (T={T})")
            axes[i, j+1].tick_params(axis='x', rotation=45)
            axes[i, j+1].set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs("image", exist_ok=True)
    plt.savefig('image/soft_labels_comparison.png')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    trainloader, testloader = get_dataloaders(batch_size=128)
    baseline_path = os.path.join("model", "basic_cnn_best.pth")

    if not os.path.exists(baseline_path):
        print("Error: Please run cnn_pytorch.py first to generate basic_cnn_best.pth")
        sys.exit(1)

    # 載入 Baseline
    baseline_model = BasicCNN().to(device)
    baseline_model.load_state_dict(torch.load(baseline_path))

    # Combine Pruning + Quantization
    print("="*55)
    print("Combine Pruning + Quantization")
    print("="*55)

    # 先剪枝
    pruned_model = apply_structured_pruning(baseline_model, device)
    print("\nFine-tuning pruned model for 2 epochs before quantization...")
    optimizer_prune = optim.Adam(pruned_model.parameters(), lr=1e-4)
    for epoch in range(2):
        pruned_model.train()
        for inputs, labels in trainloader:
            optimizer_prune.zero_grad()
            loss = F.cross_entropy(pruned_model(
                inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer_prune.step()

    # 將剪枝後的模型進行量化
    pq_model = apply_ptq(pruned_model, testloader)
    metrics_pq = benchmark(pq_model, testloader, torch.device(
        'cpu'), is_quantized=True, model_name="Pruned + PTQ int8")

    # KD Grid Search & Soft Labels
    print("\n" + "="*55)
    print("KD Grid Search (T \u2208 {2,4,8}, \u03B1 \u2208 {0.7,0.9})")
    print("="*55)

    teacher_model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)

    provided_teacher_path = os.path.join("model", "provided_resnet50.pth")
    teacher_model.load_state_dict(torch.load(
        provided_teacher_path, map_location=device))

    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # 執行視覺化
    visualize_soft_labels(teacher_model, testloader, device)

    # 執行 Grid Search
    T_list = [2, 4, 8]
    alpha_list = [0.7, 0.9]
    best_acc = 0
    best_combo = None

    print("\nGrid Search (training 5 epochs per combination to evaluate trend)...")

    scaler = torch.amp.GradScaler(
        'cuda' if torch.cuda.is_available() else 'cpu')

    for T in T_list:
        for alpha in alpha_list:
            student = BasicCNN().to(device)
            optimizer_kd = optim.Adam(student.parameters(), lr=0.001)

            for epoch in range(5):
                student.train()
                start_time = time.time()
                running_loss = 0.0

                for i, (inputs, labels) in enumerate(trainloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer_kd.zero_grad()

                    teacher_inputs = F.interpolate(inputs, size=(
                        224, 224), mode='bilinear', align_corners=False)

                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        with torch.no_grad():
                            t_logits = teacher_model(teacher_inputs)
                        s_logits = student(inputs)
                        loss = distillation_loss(
                            s_logits, t_logits, labels, T=T, alpha=alpha)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer_kd)
                    scaler.update()
                    running_loss += loss.item()

            acc = evaluate_acc(student, testloader, device)
            print(f"[{'T='+str(T):<4} | {'\u03B1='+str(alpha):<8}] -> Val Acc: {acc:.2f}%")

            # 最佳結果
            if acc > best_acc:
                best_acc = acc
                best_combo = (T, alpha)

    print("\n" + "="*55)
    print(
        f"Best Combination: T={best_combo[0]}, \u03B1={best_combo[1]} (Acc: {best_acc:.2f}%)")
    print("="*55)
