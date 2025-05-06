import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import os

# --- Configuration ---
PRELOAD_DATA_TO_GPU = True
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 128
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

# --- Check PyTorch Version ---
pt_version = torch.__version__
pt_version_major = int(pt_version.split('.')[0])
use_compile = pt_version_major >= 2

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

use_amp = False
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    use_amp = True
    print("cuDNN benchmark mode enabled.")
print(f"AMP enabled: {use_amp}")

# --- Data Setup ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
len_train_dataset = len(train_dataset)

train_loader = None
all_images_gpu = all_labels_gpu = None
num_batches_per_epoch = 0

if PRELOAD_DATA_TO_GPU and device.type == 'cuda':
    print("Preloading entire dataset to GPU...")
    t0 = time.time()
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len_train_dataset, shuffle=False)
    all_images_cpu, all_labels_cpu = next(iter(temp_loader))
    all_images_gpu = all_images_cpu.to(device)
    all_labels_gpu = all_labels_cpu.to(device)
    del temp_loader, all_images_cpu, all_labels_cpu
    torch.cuda.synchronize()
    num_batches_per_epoch = (len_train_dataset + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE
    print(f"Preload time: {time.time() - t0:.2f}s, memory: {all_images_gpu.nelement() * all_images_gpu.element_size() / 1024**2:.2f} MB")
else:
    num_workers = min(os.cpu_count() or 1, 8)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=MINI_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=(device.type == 'cuda'),
                                               persistent_workers=False)
    num_batches_per_epoch = len(train_loader)

# --- Model ---
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)

if use_compile:
    try:
        model = torch.compile(model, mode='max-autotune')
        print("Model compiled with torch.compile().")
    except Exception as e:
        print(f"torch.compile failed: {e}")
        use_compile = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# --- Warmup ---
if use_compile or (PRELOAD_DATA_TO_GPU and device.type == 'cuda'):
    print("Running warmup...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(min(num_batches_per_epoch, 5)):
            if PRELOAD_DATA_TO_GPU:
                indices = torch.randperm(len_train_dataset, device='cpu')[:MINI_BATCH_SIZE].to(device)
                images, labels = all_images_gpu[indices], all_labels_gpu[indices]
            else:
                warmup_iter = iter(train_loader)
                images_cpu, labels_cpu = next(warmup_iter)
                images = images_cpu.to(device, non_blocking=True)
                labels = labels_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(images), labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    torch.cuda.current_stream().wait_stream(s)
    print("Warmup complete.")

# --- Training Loop ---
print("Starting training loop...\n")
total_start_time = time.time()
epoch_times = []

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss_gpu = torch.tensor(0.0, device=device)

    if PRELOAD_DATA_TO_GPU:
        indices = torch.randperm(len_train_dataset, device='cpu').to(device)
        batch_ranges = range(0, len_train_dataset, MINI_BATCH_SIZE)
    else:
        data_iter = iter(train_loader)

    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

    for i in range(num_batches_per_epoch):
        if PRELOAD_DATA_TO_GPU:
            start_idx = i * MINI_BATCH_SIZE
            end_idx = min(start_idx + MINI_BATCH_SIZE, len_train_dataset)
            batch_indices = indices[start_idx:end_idx]
            images, labels = all_images_gpu[batch_indices], all_labels_gpu[batch_indices]
        else:
            try:
                images_cpu, labels_cpu = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images_cpu, labels_cpu = next(data_iter)
            images = images_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss_gpu += loss

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        ender.record()
        torch.cuda.synchronize()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {starter.elapsed_time(ender)/1000:.2f}s", end='; ')

    epoch_duration = time.time() - epoch_start_time
    epoch_times.append(epoch_duration)
    avg_loss = (epoch_loss_gpu / num_batches_per_epoch).item()
    print(f"Loss: {avg_loss:.4f}")

# --- Summary ---
total_training_time = time.time() - total_start_time
avg_epoch_time = np.mean(epoch_times)

print("\n--- Training Summary ---")
print(f"Device: {device}")
print(f"torch.compile: {use_compile}")
print(f"AMP: {use_amp}")
print(f"Preload to GPU: {PRELOAD_DATA_TO_GPU and device.type == 'cuda'}")
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Average Time Per Epoch: {avg_epoch_time:.2f} seconds")
