# Import necessary libraries
import numpy as np
import torch
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

# --- 1. Load MNIST Dataset using PyTorch ---
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000,
                              shuffle=False, num_workers=2)
print("Dataset loaded.")

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred_softmax, y_true_one_hot):
    """Compute cross-entropy loss."""
    y_pred_clipped = np.clip(y_pred_softmax, 1e-12, 1. - 1e-12)
    loss = -np.sum(y_true_one_hot * np.log(y_pred_clipped)) / y_pred_softmax.shape[0]
    return loss

def one_hot(y, num_classes):
    """Convert integer labels to one-hot vectors."""
    return np.eye(num_classes)[y]

# --- 2. Initialize Network Parameters (Weights and Biases) ---
print("Initializing network parameters...")
limit = np.sqrt(6 / (INPUT_SIZE + NUM_CLASSES))
W = np.random.uniform(-limit, limit, (INPUT_SIZE, NUM_CLASSES))
b = np.zeros((1, NUM_CLASSES))
print("Parameters initialized.")

# --- 3. Training Loop ---
print(f"Starting training for {EPOCHS} epochs...")
start_time_total = time.time()
epoch_times = []

for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    running_loss = 0.0
    num_batches = 0

    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels from PyTorch DataLoader
        inputs_torch, labels_torch = data

        # Convert to NumPy, flatten images, and ensure correct types
        inputs_np = inputs_torch.numpy().reshape(inputs_torch.shape[0], -1)
        labels_np = labels_torch.numpy()

        # Convert labels to one-hot encoding
        labels_one_hot = one_hot(labels_np, NUM_CLASSES)

        # --- Forward Pass ---
        z = inputs_np @ W + b
        a = softmax(z)
        loss = cross_entropy_loss(a, labels_one_hot)
        running_loss += loss
        num_batches += 1

        # --- Backward Pass (Backpropagation) ---
        # Gradient of loss w.r.t. softmax input (z)
        # For cross-entropy loss with softmax, this simplifies nicely
        dz = a - labels_one_hot # Shape: [Batch, Num_Classes]

        # Gradient of loss w.r.t. weights (W)
        # dL/dW = dL/dz * dz/dW = dz * X^T (averaged over batch)
        dW = inputs_np.T @ dz / inputs_np.shape[0] # Shape: [Input_Size, Num_Classes]

        # Gradient of loss w.r.t. bias (b)
        # dL/db = dL/dz * dz/db = dz * 1 (averaged over batch)
        db = np.sum(dz, axis=0, keepdims=True) / inputs_np.shape[0] # Shape: [1, Num_Classes]

        # --- Update Parameters ---
        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db

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

# --- 4. Evaluate on Test Set ---
print("Evaluating on test set...")
test_loss = 0.0
correct_predictions = 0
total_samples = 0

# No gradient calculation needed for testing
with torch.no_grad(): # Use torch context just to iterate easily
    for data in testloader:
        images_torch, labels_torch = data
        images_np = images_torch.numpy().reshape(images_torch.shape[0], -1)
        labels_np = labels_torch.numpy()
        labels_one_hot = one_hot(labels_np, NUM_CLASSES)

        # Forward pass
        z_test = images_np @ W + b
        a_test = softmax(z_test)

        # Calculate loss
        loss = cross_entropy_loss(a_test, labels_one_hot)
        test_loss += loss * images_np.shape[0] # Accumulate total loss

        # Calculate accuracy
        predicted_labels = np.argmax(a_test, axis=1)
        correct_predictions += np.sum(predicted_labels == labels_np)
        total_samples += images_np.shape[0]

average_test_loss = test_loss / total_samples
test_accuracy = correct_predictions / total_samples

print(f"Test Set Loss: {average_test_loss:.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_samples})")

# --- 5. Show Sample Predictions ---
print("Showing sample predictions...")

# Get a batch of test images
dataiter = iter(testloader)
images_torch, labels_torch = next(dataiter)
images_np = images_torch.numpy().reshape(images_torch.shape[0], -1)
labels_np = labels_torch.numpy()

# Make predictions on this batch
z_sample = images_np @ W + b
a_sample = softmax(z_sample)
predicted_labels_sample = np.argmax(a_sample, axis=1)

# Plot the first few images and their predictions
num_images_to_show = 5
fig, axes = plt.subplots(1, num_images_to_show, figsize=(10, 3))
for i in range(num_images_to_show):
    image_to_show = images_torch[i].squeeze().numpy()
    ax = axes[i]
    ax.imshow(image_to_show, cmap='gray')
    ax.set_title(f"Pred: {predicted_labels_sample[i]}\nTrue: {labels_np[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Sample predictions displayed.")
