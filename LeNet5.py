# =============================================================================
# LeNet-5 on MNIST using PyTorch
# Implements the classic LeNet-5 architecture, trains on MNIST, and plots
# loss and accuracy vs epochs.
# =============================================================================

# --- Import libraries ---
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Set hyperparameters ---
batch_size = 64          # Number of samples per training step
num_classes = 10         # MNIST has 10 digits (0-9)
learning_rate = 0.001    # Step size for optimizer
num_epochs = 20          # Number of full passes over the training set (until convergence)

# --- Select device (GPU if available, else CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# --- Load and preprocess MNIST dataset ---
# Training set: 60,000 images
# LeNet-5 expects 32x32 input, we resize to 28x28 to match the input size of the model
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),         # Convert PIL image to tensor [0, 1]
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # MNIST train statistics
    ]),
    download=True,
)

# Test set: 10,000 images
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1325,), std=(0.3105,)),  # MNIST test statistics
    ]),
    download=True,
)

# Create data loaders to feed batches during training and evaluation
# Shuffle=True to randomize order each epoch
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# --- Define the LeNet-5 architecture ---
# Input: 1x32x32 (grayscale). Output: num_classes logits.
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Block 1: Conv 1x32x32 -> 6x28x28 -> pool -> 6x14x14
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),  # 1 input channel, 6 filters
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: Conv 6x14x14 -> 16x10x10 -> pool -> 16x5x5
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Fully connected layers: 16*5*5=400 -> 120 -> 84 -> num_classes
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)   # Flatten for FC layers
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Instantiate model and move it to the selected device (GPU/CPU)
model = LeNet5(num_classes).to(device)
print(model)

# --- Define loss function and optimizer ---
cost = nn.CrossEntropyLoss()   # Standard for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Prepare lists to record metrics per epoch (for plotting) ---
train_losses = []
train_accuracies = []
test_accuracies = []
total_step = len(train_loader)   # Number of batches per epoch


def evaluate_accuracy(model, data_loader, device):
    """
    Compute classification accuracy (percentage) on a dataset.
    Sets model to eval mode, runs without gradients, then restores train mode.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   # Predicted class per sample
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total


# --- Training loop ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Loop over all batches in the training set
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: compute predictions and loss
        outputs = model(images)
        loss = cost(outputs, labels)

        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()        # Compute gradients via backpropagation
        optimizer.step()       # Update model parameters

        # Accumulate loss and train accuracy for this epoch
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Print progress every 400 batches to monitor training progress
        if (i + 1) % 400 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

    # Compute average loss and accuracies for this epoch
    avg_loss = running_loss / total_step
    train_acc = 100 * correct_train / total_train
    test_acc = evaluate_accuracy(model, test_loader, device)

    # Store metrics for later plotting
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(
        "Epoch [{}/{}] Summary - Avg Loss: {:.4f}, Train Acc: {:.2f}%, Test Acc: {:.2f}%".format(
            epoch + 1, num_epochs, avg_loss, train_acc, test_acc
        )
    )

# --- Final evaluation on test set ---
model.eval()
final_test_acc = evaluate_accuracy(model, test_loader, device)
print(f"\nFinal accuracy on test set: {final_test_acc:.2f}%")

# --- Plot loss and accuracy vs epochs ---
epochs_range = range(1, num_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left plot: training loss per epoch
ax1.plot(epochs_range, train_losses, "b-o", markersize=4)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Loss vs Epochs")
ax1.grid(True, alpha=0.3)

# Right plot: train and test accuracy per epoch
ax2.plot(epochs_range, train_accuracies, "b-o", markersize=4, label="Train Accuracy")
ax2.plot(epochs_range, test_accuracies, "r-s", markersize=4, label="Test Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy vs Epochs")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lenet5_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlots saved as 'lenet5_training_curves.png'")
