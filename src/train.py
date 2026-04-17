import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix


from src.model import CNN
from src.data import get_data


def train_model(epochs=5, lr=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data(batch_size=batch_size)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        acc = evaluate(model, test_loader, device)
        test_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")


    torch.save(model.state_dict(), "model.pth")


    save_plots(train_losses, test_accuracies)
    save_confusion_matrix(model, test_loader, device)

    return test_accuracies[-1]


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def save_plots(train_losses, accuracies):
    os.makedirs("outputs/plots", exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs, train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    for x, y in zip(epochs, train_losses):
        plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=8)

    plt.grid()
    plt.savefig("outputs/plots/loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, accuracies, marker='o')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")


    for x, y in zip(epochs, accuracies):
        plt.text(x, y, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.grid()
    plt.savefig("outputs/plots/accuracy.png")
    plt.close()

def save_confusion_matrix(model, test_loader, device):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.close()

def experiment_epochs(epoch_list):
    results = []

    for num_epochs in epoch_list:
        print(f"\nTraining with {num_epochs} epochs")
        acc = train_model(epochs=num_epochs)
        results.append(acc)

    return results


def plot_epochs_vs_accuracy(epoch_list, accuracies):
    plt.figure()
    plt.plot(epoch_list, accuracies, marker='o')

    for x, y in zip(epoch_list, accuracies):
        plt.text(x, y, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.grid()

    plt.savefig("outputs/plots/epochs_accuracy.png")
    plt.close()


if __name__ == "__main__":
    epoch_list = [1, 2, 3, 5]
    accuracies = experiment_epochs(epoch_list)
    plot_epochs_vs_accuracy(epoch_list, accuracies)