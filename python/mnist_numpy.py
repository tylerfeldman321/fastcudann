import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time

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
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=MINI_BATCH_SIZE,
                                           shuffle=True)
print(f"Dataset loaded. Number of training samples: {len(train_dataset)}")
print(f"Mini-batch size: {MINI_BATCH_SIZE}")
print(f"Number of mini-batches per epoch: {len(train_loader)}")


def softmax(z):
    """Compute softmax values for each set of scores in z."""
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true_one_hot):
    """Compute cross-entropy loss."""
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true_one_hot * np.log(y_pred), axis=1)
    return np.mean(loss)

def to_one_hot(labels, num_classes):
    """Convert integer labels to one-hot vectors."""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

# --- Network Initialization ---
W = np.random.randn(INPUT_SIZE, NUM_CLASSES) * 0.01
b = np.zeros((1, NUM_CLASSES))

print(f"\nInitializing training for {NUM_EPOCHS} epochs...")
print(f"Learning Rate: {LEARNING_RATE}")

# --- Training Loop ---
total_start_time = time.time()
epoch_times = []

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # 1. Prepare Data
        # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
        images_np = images.reshape(images.shape[0], -1).numpy()
        # Convert labels to one-hot encoding
        labels_np = labels.numpy()
        labels_one_hot = to_one_hot(labels_np, NUM_CLASSES)

        # 2. Forward Pass
        z = np.dot(images_np, W) + b
        y_pred = softmax(z)

        # 3. Calculate Loss
        loss = cross_entropy_loss(y_pred, labels_one_hot)
        epoch_loss += loss
        num_batches += 1

        # 4. Backward Pass (Gradient Calculation)
        # Gradient of loss w.r.t. logits (z)
        # dL/dz = y_pred - y_true_one_hot
        dz = y_pred - labels_one_hot

        # Gradient of loss w.r.t. weights (W)
        # dL/dW = dL/dz * dz/dW = (y_pred - y_true).T * x
        dW = (1 / images_np.shape[0]) * np.dot(images_np.T, dz)

        # Gradient of loss w.r.t. biases (b)
        # dL/db = dL/dz * dz/db = (y_pred - y_true)
        db = (1 / images_np.shape[0]) * np.sum(dz, axis=0, keepdims=True)

        # 5. Update Parameters (Vanilla Gradient Descent)
        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db

    # --- End of Epoch ---
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    average_epoch_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {average_epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")

# --- End of Training ---
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
average_epoch_time = np.mean(epoch_times)

print("\n--- Training Summary ---")
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Average Time Per Epoch: {average_epoch_time:.2f} seconds")