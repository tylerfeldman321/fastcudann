import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std deviation for MNIST
])

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=MINI_BATCH_SIZE,
                                           shuffle=True,
                                           pin_memory=torch.cuda.is_available())

print(f"Dataset loaded. Number of training samples: {len(train_dataset)}")
print(f"Mini-batch size: {MINI_BATCH_SIZE}")
print(f"Number of mini-batches per epoch: {len(train_loader)}")

# --- Model Definition ---
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

# --- Instantiate Model, Loss, and Optimizer ---
model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

print(f"\nInitializing training for {NUM_EPOCHS} epochs...")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Optimizer: SGD")
print(f"Loss Function: CrossEntropyLoss")

# --- Training Loop ---
total_start_time = time.time()
epoch_times = []

# Set model to training mode
model.train()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # 1. Prepare Data and move to device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # 2. Zero the gradients
        optimizer.zero_grad()

        # 3. Forward Pass
        outputs = model(images)

        # 4. Calculate Loss
        loss = criterion(outputs, labels)

        # 5. Backward Pass
        loss.backward()

        # 6. Update Parameters
        optimizer.step()

        # Accumulate loss for the epoch
        epoch_loss += loss.item()
        num_batches += 1

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
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Average Time Per Epoch: {average_epoch_time:.2f} seconds")
