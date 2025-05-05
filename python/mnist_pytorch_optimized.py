# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np # Still useful for average time calculation at the end
import os # For setting num_workers based on CPU count

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

# --- Enable cuDNN benchmark mode (if using CUDA and input sizes are constant) ---
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark mode enabled.")

# --- Hyperparameters ---
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 128
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

# --- Data Loading and Preprocessing ---
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

num_workers = min(os.cpu_count() if os.cpu_count() else 1, 4)
print(f"Using {num_workers} DataLoader workers.")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=MINI_BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=(device.type == 'cuda'),
                                           persistent_workers=num_workers > 0)

print(f"Dataset loaded. Number of training samples: {len(train_dataset)}")
print(f"Mini-batch size: {MINI_BATCH_SIZE}")
print(f"Number of mini-batches per epoch: {len(train_loader)}")

# --- Model Definition ---
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        # Using Float32 for the layer itself, AMP will handle casting
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Flatten the image first if it's not already flat
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

# --- Instantiate Model, Loss, and Optimizer ---
model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)

# --- Apply torch.compile() if available ---
if use_compile:
    print("Applying torch.compile() to the model...")
    try:
        model = torch.compile(model, mode='max-autotune') # Or try 'max-autotune'
        print("torch.compile() applied successfully.")
    except Exception as e:
        print(f"Warning: torch.compile() failed with error: {e}")
        print("Proceeding without torch.compile().")
        use_compile = False # Disable compile flag if it failed

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- Automatic Mixed Precision (AMP) Setup ---
use_amp = (device.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
print(f"Automatic Mixed Precision (AMP) enabled: {use_amp}")

print(f"\nInitializing training for {NUM_EPOCHS} epochs...")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Optimizer: SGD")
print(f"Loss Function: CrossEntropyLoss")

# --- Training Loop ---
total_start_time = time.time()
epoch_times = []

model.train()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # 1. Prepare Data and move to device (GPU or CPU)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 2. Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # 3. Forward Pass with Automatic Mixed Precision context
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 4. Backward Pass (Compute Gradients with Scaling)
        scaler.scale(loss).backward()

        # 5. Update Parameters (Optimizer Step with Unscaling)
        scaler.step(optimizer)

        # 6. Update the scale factor for the next iteration.
        scaler.update()

        # Accumulate loss for the epoch
        epoch_loss += loss.item()
        num_batches += 1

    # --- End of Epoch ---
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    average_epoch_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {average_epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")

# --- End of Training ---
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
average_epoch_time = np.mean(epoch_times) if epoch_times else 0

print("\n--- Training Summary ---")
print(f"Device Used: {device}")
print(f"torch.compile() used: {use_compile}")
print(f"AMP used: {use_amp}")
print(f"DataLoader Workers: {num_workers}")
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Average Time Per Epoch: {average_epoch_time:.2f} seconds")