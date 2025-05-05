# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import os

# --- Configuration ---
PRELOAD_DATA_TO_GPU = True # Set to True to load entire dataset onto GPU (only for small datasets that can fit in GPU memory!)
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 128
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

# --- Check PyTorch Version for torch.compile ---
pt_version = torch.__version__
pt_version_major = int(pt_version.split('.')[0])
use_compile = pt_version_major >= 2
if not use_compile:
    print(f"Warning: PyTorch version {pt_version} detected. Need PyTorch 2.0+ for torch.compile().")
    print("Proceeding without torch.compile().")

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Enable cuDNN benchmark mode & AMP ---
use_amp = False
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    use_amp = True # Enable AMP only on CUDA
    print("cuDNN benchmark mode enabled.")
print(f"Automatic Mixed Precision (AMP) enabled: {use_amp}")

# --- Data Loading Strategy ---
train_loader = None
all_images_gpu = None
all_labels_gpu = None
len_train_dataset = 0
num_batches_per_epoch = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST dataset...")
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)
len_train_dataset = len(train_dataset)

if PRELOAD_DATA_TO_GPU and device.type == 'cuda':
    print("Preloading entire dataset to GPU...")
    start_preload_time = time.time()
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len_train_dataset, shuffle=False)
    all_images_cpu, all_labels_cpu = next(iter(temp_loader))
    all_images_gpu = all_images_cpu.to(device)
    all_labels_gpu = all_labels_cpu.to(device)
    num_batches_per_epoch = (len_train_dataset + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE
    del temp_loader, all_images_cpu, all_labels_cpu
    torch.cuda.synchronize()
    end_preload_time = time.time()
    print(f"Dataset preloaded to GPU in {end_preload_time - start_preload_time:.2f} seconds.")
    print(f"WARNING: Using preloaded data. This consumes significant GPU memory ({all_images_gpu.nelement() * all_images_gpu.element_size() / 1024**2:.2f} MB for images).")
else:
    print("Using standard DataLoader.")
    num_workers = min(os.cpu_count() if os.cpu_count() else 1, 8)
    pin_memory_enabled = (device.type == 'cuda')
    persistent_workers_enabled = num_workers > 0
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=MINI_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory_enabled,
                                               persistent_workers=persistent_workers_enabled)
    num_batches_per_epoch = len(train_loader)
    print(f"Using {num_workers} DataLoader workers.")
    print(f"Pin memory enabled: {pin_memory_enabled}")
    print(f"Persistent workers enabled: {persistent_workers_enabled}")


print(f"Dataset loaded. Number of training samples: {len_train_dataset}")
print(f"Mini-batch size: {MINI_BATCH_SIZE}")
print(f"Number of mini-batches per epoch: {num_batches_per_epoch}")

# --- Model Definition ---
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        if x.ndim > 2:
             x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)

if use_compile:
    print("Applying torch.compile() to the model...")
    try:
        compile_mode = 'max-autotune'
        model = torch.compile(model, mode=compile_mode)
        print(f"torch.compile(mode='{compile_mode}') applied successfully.")
    except Exception as e:
        print(f"Warning: torch.compile() failed with error: {e}")
        print("Proceeding without torch.compile().")
        use_compile = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- Automatic Mixed Precision (AMP) Setup ---
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"\nInitializing training for {NUM_EPOCHS} epochs...")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Optimizer: SGD")
print(f"Loss Function: CrossEntropyLoss")
print("-" * 30)

# --- Training Loop ---
total_start_time = time.time()
epoch_times = []

model.train()

if use_compile or (PRELOAD_DATA_TO_GPU and device.type == 'cuda'):
    print("Performing warmup step(s)...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(min(num_batches_per_epoch, 5)):
             
             # Prepare dummy data or first batch
            if PRELOAD_DATA_TO_GPU and device.type == 'cuda':
                indices = torch.randperm(len_train_dataset, device=device)[:MINI_BATCH_SIZE]
                images, labels = all_images_gpu[indices], all_labels_gpu[indices]
            elif train_loader:
                 images, labels = next(iter(train_loader))
                 images = images.to(device, non_blocking=True)
                 labels = labels.to(device, non_blocking=True)
            else: # Fallback: Create dummy data
                 images = torch.randn(MINI_BATCH_SIZE, 1, 28, 28, device=device)
                 labels = torch.randint(0, NUM_CLASSES, (MINI_BATCH_SIZE,), device=device)

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

    torch.cuda.current_stream().wait_stream(s)
    print("Warmup complete.")

print("Starting main training loop...")
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss_gpu = torch.tensor(0.0, device=device)

    if PRELOAD_DATA_TO_GPU and device.type == 'cuda':
        indices = torch.randperm(len_train_dataset, device=device)
        iterable_data = range(0, len_train_dataset, MINI_BATCH_SIZE)
    else:
        iterable_data = enumerate(train_loader)

    for loop_item in iterable_data:
        # 1. Prepare Data based on the iteration strategy
        if PRELOAD_DATA_TO_GPU and device.type == 'cuda':
            start_idx = loop_item
            end_idx = min(start_idx + MINI_BATCH_SIZE, len_train_dataset)
            batch_indices = indices[start_idx:end_idx]
            images = all_images_gpu[batch_indices]
            labels = all_labels_gpu[batch_indices]
        else:
            i, (images_cpu, labels_cpu) = loop_item
            images = images_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)

        # 2. Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # 3. Forward Pass with Automatic Mixed Precision context
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 4. Backward Pass (Compute Gradients with Scaling if AMP is enabled)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 5. Update Parameters (Optimizer Step with Unscaling if AMP is enabled)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Accumulate loss on GPU. Use detach() to free graph memory for the loss tensor.
        epoch_loss_gpu += loss

    if device.type == 'cuda':
        torch.cuda.synchronize()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)

    average_epoch_loss = (epoch_loss_gpu / num_batches_per_epoch).item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {average_epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")

# --- End of Training ---
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
average_epoch_time = np.mean(epoch_times) if epoch_times else 0

print("\n--- Training Summary ---")
print(f"Device Used: {device}")
print(f"torch.compile() used: {use_compile}")
print(f"AMP used: {use_amp}")
print(f"Preloaded Data to GPU: {PRELOAD_DATA_TO_GPU and device.type == 'cuda'}")
if not (PRELOAD_DATA_TO_GPU and device.type == 'cuda'):
    print(f"DataLoader Workers: {num_workers}")
print(f"Total Training Time (including potential warmup): {total_training_time:.2f} seconds")
print(f"Average Time Per Epoch: {average_epoch_time:.2f} seconds")