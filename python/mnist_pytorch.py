import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.1
EPOCHS = 10
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load MNIST Dataset using PyTorch ---
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x)),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, # Use larger batch for testing
                              shuffle=False, num_workers=2)
print("Dataset loaded.")

# --- 2. Define the Neural Network using PyTorch ---
print("Defining the PyTorch model...")
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = SingleLayerNN(INPUT_SIZE, NUM_CLASSES).to(device)
print("Model defined and moved to device.")
print(model)

# --- 3. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
print("Loss function and optimizer defined.")

# --- 4. Training Loop ---
print(f"Starting training for {EPOCHS} epochs...")
start_time_total = time.time()
epoch_times = []

for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    running_loss = 0.0
    num_batches = 0
    model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # --- Forward Pass ---
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # --- Backward Pass and Optimize ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    # --- Epoch End ---
    end_time_epoch = time.time()
    epoch_duration = end_time_epoch - start_time_epoch
    epoch_times.append(epoch_duration)
    avg_loss = running_loss / num_batches
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f}s")

end_time_total = time.time()
total_training_time = end_time_total - start_time_total
average_epoch_time = np.mean(epoch_times)

print("Training finished.")
print(f"Total Training Time: {total_training_time:.2f}s")
print(f"Average Epoch Time: {average_epoch_time:.2f}s")

# --- 5. Evaluate on Test Set ---
print("Evaluating on test set...")
model.eval()
test_loss = 0.0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        _, predicted_labels = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_labels == labels).sum().item()

average_test_loss = test_loss / total_samples
test_accuracy = correct_predictions / total_samples

print(f"Test Set Loss: {average_test_loss:.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_samples})")

# --- 6. Show Sample Predictions ---
print("Showing sample predictions...")

# Get a batch of test images (need original, non-flattened images for display)
# Re-create loader without flattening for visualization
vis_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
vis_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=vis_transform)
vis_testloader = DataLoader(vis_testset, batch_size=5, shuffle=True) # Small batch for visualization

dataiter = iter(vis_testloader)
images_vis, labels_vis = next(dataiter)

# Prepare images for the model (flatten and move to device)
images_model = images_vis.view(images_vis.shape[0], -1).to(device)
labels_vis = labels_vis.to(device)

# Make predictions
model.eval()
with torch.no_grad():
    outputs_sample = model(images_model)
    _, predicted_labels_sample = torch.max(outputs_sample.data, 1)

# Move data back to CPU for plotting
images_vis_np = images_vis.cpu().numpy()
labels_vis_np = labels_vis.cpu().numpy()
predicted_labels_sample_np = predicted_labels_sample.cpu().numpy()

# Plot the first few images and their predictions
num_images_to_show = 5
fig, axes = plt.subplots(1, num_images_to_show, figsize=(10, 3))
for i in range(num_images_to_show):
    image_to_show = images_vis_np[i].squeeze() # Remove channel dim
    ax = axes[i]
    ax.imshow(image_to_show, cmap='gray')
    ax.set_title(f"Pred: {predicted_labels_sample_np[i]}\nTrue: {labels_vis_np[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Sample predictions displayed.")
