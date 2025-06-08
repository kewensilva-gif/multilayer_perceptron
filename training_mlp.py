import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mlp import MLP

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Dataset
train_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes

def training(model, criterion, optimizer, train_loader_fold, min_error=0.03):
    epoch = 0
    max_epochs = 300
    while True:
        model.train()
        cumulative_batch_error = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader_fold:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            error = criterion(outputs, labels)
            error.backward()
            optimizer.step()

            cumulative_batch_error += error.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_error = cumulative_batch_error / len(train_loader_fold)
        epoch_accuracy = 100 * correct / total

        print(f"Época {epoch+1} | Loss: {epoch_error:.4f} | Acurácia: {epoch_accuracy:.2f}%")

        if (epoch_error <= min_error and epoch_accuracy == 100) or epoch >= max_epochs -1:
            if epoch_error <= min_error:
                print(f"Critério de erro baixo ({epoch_error:.4f} <= {min_error}) atingido.")
            if epoch >= max_epochs -1:
                print(f"Máximo de épocas ({max_epochs}) atingido.")
            break
        epoch += 1

def plot_confusion_matrix(cm, class_names_list):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Valor Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

def evaluate_model(model, loader, device, class_names_list):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de Confusão:")
    print(cm)
    plot_confusion_matrix(cm, class_names_list)

    return cm

def training_kfolds(batch_size, k, learning_rate=0.1):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    for fold, (train_id, val_id) in enumerate(kf.split(train_dataset)):
        print(f"\n===== FOLD {fold + 1}/{k} =====")

        train_subset = Subset(train_dataset, train_id)
        val_subset = Subset(train_dataset, val_id)

        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # A cada fold é criado um novo modelo
        model = MLP(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), learning_rate)

        training(model, criterion, optimizer, train_loader_fold)

        print(f"\nAvaliação do modelo no Fold {fold + 1}")
        evaluate_model(model, val_loader_fold, device, class_names)

if __name__ == '__main__':
    training_kfolds(10, 5)
    print("\nValidação finalizada")