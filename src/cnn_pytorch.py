import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from torchinfo import summary
import os

# Dataset Preparation


def get_dataloaders(batch_size=128):
    # 訓練集使用 Data Augmentation 防止 overfitting
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # 測試集只 Normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# Network Architecture


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # 為了 PTQ 量化加入的節點
        self.quant = torch.quantization.QuantStub() 

        # Input shape: (B, 3, 32, 32)
        # Output: (B, 32, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        # Output: (B, 32, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: (B, 64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        # Output: (B, 64, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: (B, 128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Output: (B, 128*8*8) = (B, 8192)
        self.flatten = nn.Flatten()
        # Output: (B, 512)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        # Output: (B, 10)
        self.fc2 = nn.Linear(512, 10)

        # 反量化節點
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# Residual Block Architecture


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = torch.relu(out)
        return out


class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Training Loop


def train_and_evaluate(model, trainloader, testloader, epochs=30, lr=0.001, device='cpu', save_path='model/cnn_best.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        history['train_loss'].append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Val Loss
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Val Accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(testloader)
        val_acc = 100 * correct / total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # 儲存 Validation 最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return history

# Plotting


def plot_metrics(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss',
             color='blue', marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='Val Loss',
             color='red', marker='s', markersize=4)
    plt.title(f'{title} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy',
             color='green', marker='^', markersize=4)
    plt.title(f'{title} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_dir = 'image'
    file_path = os.path.join(
        save_dir, f"{title.replace(' ', '_')}_metrics.png")
    plt.savefig(file_path)

# Performance Analysis


def analyze_model_performance(model, device, model_name="Model"):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    print(f"\n{'='*55}")
    print(f"Performance Analysis: {model_name}")
    print(f"{'='*55}")

    # Parameters & Model Size
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 ** 2)
    print(f"1. Parameters: {params/1e6:.4f} M")
    print(f"2. Model Size: {size_mb:.2f} MB")

    # FLOPs
    stats = summary(model, input_size=(1, 3, 32, 32), verbose=0)
    flops_g = stats.total_mult_adds / 1e9  # 轉換為 GFLOPs
    print(f"3. FLOPs:      {flops_g:.6f} GFLOPs")

    # Latency
    with torch.no_grad():
        [model(dummy_input) for _ in range(10)]  # Warm up

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    N = 100
    with torch.no_grad():
        for _ in range(N):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    avg_latency = ((time.time() - start_time) / N) * 1000  # ms
    print(f"4. Latency:    {avg_latency:.2f} ms (avg over {N} runs)")

    # Peak Memory (CUDA only)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"5. Peak Mem:   {peak_mem:.2f} MB (batch_size=1 inference)")
    else:
        print(f"5. Peak Mem:   (Requires CUDA to measure)")

    print(f"{'='*55}\n")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_dataloaders(batch_size=128)

    print("\n--- Training Basic CNN ---")
    basic_model = BasicCNN().to(device)
    analyze_model_performance(basic_model, device, "Basic CNN")
    basic_save_path = os.path.join('model', 'basic_cnn_best.pth')
    basic_history = train_and_evaluate(
        basic_model, trainloader, testloader, epochs=30, device=device, save_path=basic_save_path)
    plot_metrics(basic_history, "Basic CNN")

    print("\n--- Training ResNet CNN ---")
    resnet_model = ResNetCNN().to(device)
    analyze_model_performance(resnet_model, device, "ResNet CNN")
    resnet_save_path = os.path.join('model', 'resnet_cnn_best.pth')
    resnet_history = train_and_evaluate(
        resnet_model, trainloader, testloader, epochs=30, device=device, save_path=resnet_save_path)
    plot_metrics(resnet_history, "ResNet CNN")

    # 比較結果
    print(f"\nBasic CNN Accuracy: {basic_history['val_acc'][-1]:.2f}%")
    print(f"ResNet CNN Accuracy: {resnet_history['val_acc'][-1]:.2f}%")
