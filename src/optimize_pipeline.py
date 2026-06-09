import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import time
import os
import copy
import sys
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 導入資料集與模型
from cnn_pytorch import BasicCNN, get_dataloaders

def evaluate_acc(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return 100 * correct / total

class PrunedBasicCNN(BasicCNN):
    def __init__(self, ch1, ch2, ch3, num_classes=10):
        super(BasicCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, ch1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch3)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(ch3 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.dequant = torch.quantization.DeQuantStub()

# 評估指標 Benchmark
def benchmark(model, dataloader, device, is_quantized=False, model_name="Model"):
    model.eval()
    val_acc = evaluate_acc(model, dataloader, device if not is_quantized else 'cpu')

    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")

    dummy_input = torch.randn(1, 3, 32, 32)
    flops_g = 0
    params = 0

    if not is_quantized:
        dummy_input = dummy_input.to(device)
        model.to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        stats = summary(model, input_size=(1, 3, 32, 32), verbose=0)
        flops_g = stats.total_mult_adds / 1e9
    
    # Warmup
    for _ in range(10): model(dummy_input)
    start_time = time.time()
    for _ in range(100): model(dummy_input)
    latency_ms = ((time.time() - start_time) / 100) * 1000

    print(f"\n=======================================================")
    print(f"Performance Analysis: {model_name}")
    print(f"=======================================================")
    if not is_quantized:
        print(f"1. Parameters: {params:.4f} M")
    print(f"2. Model Size: {size_mb:.2f} MB")
    if not is_quantized:
        print(f"3. FLOPs:      {flops_g:.6f} GFLOPs")
    else:
        print(f"3. FLOPs:      N/A (Quantized)")
    print(f"4. Latency:    {latency_ms:.2f} ms (avg over 100 runs)")
    print(f"5. Accuracy:   {val_acc:.2f}%")
    print(f"=======================================================\n")
    
    return {"acc": val_acc, "size": size_mb, "latency": latency_ms, "flops": flops_g}

# 優化技術實作 (PTQ, Pruning, KD)

def apply_ptq(model, dataloader):
    print("--- PTQ int8 Quantization ---")
    q_model = copy.deepcopy(model).cpu()
    q_model.eval()
    torch.quantization.fuse_modules(q_model, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3', 'relu3']], inplace=True)
    q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(q_model, inplace=True)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            q_model(inputs)
            if i > 10: break
    torch.quantization.convert(q_model, inplace=True)
    return q_model

def apply_structured_pruning(model, device):
    print("--- Structured Pruning (Layer-wise Bottom 30%) ---")
    convs = [model.conv1, model.conv2, model.conv3]
    keep_idx = []
    for c in convs:
        norm = torch.norm(c.weight.data, p=1, dim=(1,2,3))
        threshold = torch.quantile(norm, 0.3)
        keep = torch.where(norm > threshold)[0]
        if len(keep) == 0: keep = torch.tensor([0]).to(device)
        keep_idx.append(keep)

    ch1, ch2, ch3 = len(keep_idx[0]), len(keep_idx[1]), len(keep_idx[2])
    print(f"Original Channels: 32 -> 64 -> 128")
    print(f"Pruned Channels  : {ch1} -> {ch2} -> {ch3}")
    
    p_model = PrunedBasicCNN(ch1, ch2, ch3).to(device)
    
    p_model.conv1.weight.data = model.conv1.weight.data[keep_idx[0]]
    p_model.bn1.weight.data = model.bn1.weight.data[keep_idx[0]]
    p_model.bn1.bias.data = model.bn1.bias.data[keep_idx[0]]
    p_model.bn1.running_mean.data = model.bn1.running_mean.data[keep_idx[0]]
    p_model.bn1.running_var.data = model.bn1.running_var.data[keep_idx[0]]

    p_model.conv2.weight.data = model.conv2.weight.data[keep_idx[1]][:, keep_idx[0], :, :]
    p_model.bn2.weight.data = model.bn2.weight.data[keep_idx[1]]
    p_model.bn2.bias.data = model.bn2.bias.data[keep_idx[1]]
    p_model.bn2.running_mean.data = model.bn2.running_mean.data[keep_idx[1]]
    p_model.bn2.running_var.data = model.bn2.running_var.data[keep_idx[1]]

    p_model.conv3.weight.data = model.conv3.weight.data[keep_idx[2]][:, keep_idx[1], :, :]
    p_model.bn3.weight.data = model.bn3.weight.data[keep_idx[2]]
    p_model.bn3.bias.data = model.bn3.bias.data[keep_idx[2]]
    p_model.bn3.running_mean.data = model.bn3.running_mean.data[keep_idx[2]]
    p_model.bn3.running_var.data = model.bn3.running_var.data[keep_idx[2]]

    fc1_w = model.fc1.weight.data.view(512, 128, 8, 8)
    p_model.fc1.weight.data = fc1_w[:, keep_idx[2], :, :].reshape(512, ch3 * 8 * 8)
    p_model.fc1.bias.data = model.fc1.bias.data
    p_model.fc2.weight.data = model.fc2.weight.data
    p_model.fc2.bias.data = model.fc2.bias.data
    
    return p_model

def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.9):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    trainloader, testloader = get_dataloaders(batch_size=128)
    os.makedirs("model", exist_ok=True)
    os.makedirs("image", exist_ok=True)

    # 載入 Baseline
    print("--- Loading Baseline BasicCNN ---")
    baseline_model = BasicCNN().to(device)
    baseline_path = os.path.join("model", "basic_cnn_best.pth")
    
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path))
    else:
        print(f"Error: {baseline_path} not found. Please run cnn_pytorch.py first.")
        sys.exit(1)

    metrics_baseline = benchmark(baseline_model, testloader, device, model_name="Float32 Baseline")

    # PTQ int8
    q_model = apply_ptq(baseline_model, testloader)
    metrics_ptq = benchmark(q_model, testloader, torch.device('cpu'), is_quantized=True, model_name="PTQ int8")

    # Structured Pruning & Fine-Tuning
    pruned_model = apply_structured_pruning(baseline_model, device)
    print("\nFine-tuning pruned model for 5 epochs (lr=1e-4)...")
    optimizer_prune = optim.Adam(pruned_model.parameters(), lr=1e-4)
    
    for epoch in range(5):
        pruned_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_prune.zero_grad()
            loss = F.cross_entropy(pruned_model(inputs), labels)
            loss.backward()
            optimizer_prune.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(trainloader)
        acc = evaluate_acc(pruned_model, testloader, device)
        print(f"Epoch [{epoch+1}/5] - Train Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")
        
    metrics_prune = benchmark(pruned_model, testloader, device, model_name="Pruned Model")


    # Knowledge Distillation
    print("--- Knowledge Distillation ---")
    print("Loading ResNet-56 Teacher (~94% on CIFAR-10)...")
    teacher_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).to(device)
    teacher_model.eval()
    
    student_model = BasicCNN().to(device)
    optimizer_kd = optim.Adam(student_model.parameters(), lr=0.001)
    
    print("\nTraining Student Model with KD (T=4, alpha=0.9) for 30 epochs...")
    for epoch in range(30):
        student_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_kd.zero_grad()
            student_logits = student_model(inputs)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            loss = distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.9)
            loss.backward()
            optimizer_kd.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(trainloader)
        acc = evaluate_acc(student_model, testloader, device)
        print(f"Epoch [{epoch+1}/30] - Train Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")
            
    metrics_kd = benchmark(student_model, testloader, device, model_name="Distilled Student")

    # 產出圖表
    print("--- Comparative Charts ---")
    labels_plt = ['Baseline', 'PTQ int8', 'Pruned', 'Distillation']
    accs = [metrics_baseline['acc'], metrics_ptq['acc'], metrics_prune['acc'], metrics_kd['acc']]
    sizes = [metrics_baseline['size'], metrics_ptq['size'], metrics_prune['size'], metrics_kd['size']]
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    ax1.bar(labels_plt, sizes, color='lightgreen', edgecolor='black', alpha=0.8, width=0.4, label='Model Size (MB)')
    ax2.plot(labels_plt, accs, color='darkblue', marker='o', linewidth=2, markersize=8, label='Accuracy (%)')
    
    ax1.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(sizes) * 1.3)
    ax2.set_ylim(min(accs) - 5, max(accs) + 5)
    plt.title('Optimization Techniques Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('image/optimization_comparison.png')

    # 結果輸出
    print("\n" + "="*60)
    print("Optimization Summary".center(60))
    print("="*60)
    print(f"{'Model Variant':<22} | {'Accuracy (%)':<15} | {'Model Size (MB)':<15}")
    print("-" * 60)
    print(f"{'1. Float32 Baseline':<22} | {metrics_baseline['acc']:<15.2f} | {metrics_baseline['size']:<15.2f}")
    print(f"{'2. PTQ int8':<22} | {metrics_ptq['acc']:<15.2f} | {metrics_ptq['size']:<15.2f}")
    print(f"{'3. Pruned Model':<22} | {metrics_prune['acc']:<15.2f} | {metrics_prune['size']:<15.2f}")
    print(f"{'4. Distilled Student':<22} | {metrics_kd['acc']:<15.2f} | {metrics_kd['size']:<15.2f}")
    print("="*60 + "\n")