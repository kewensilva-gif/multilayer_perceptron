import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mlp import MLP

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Dataset e DataLoader
train_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes

# Modelo, perda e otimizador
model = MLP(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def training(min_error=0.3):
    epoch = 0
    max_epochs = 300
    while True:
        model.train()
        cumulative_batch_error = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        epoch_error = cumulative_batch_error / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Época {epoch+1} | Loss: {epoch_error:.4f} | Acurácia: {epoch_accuracy:.2f}%")

        if epoch_error <= min_error or epoch >= max_epochs -1:
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
    print("\nAvaliação do modelo")
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

    # all_labels_clean = [int(label) for label in all_labels]
    # all_preds_clean = [int(pred) for pred in all_preds]
    # print("labels",all_labels_clean)
    # print("preds",all_preds_clean)

    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de Confusão:")
    print(cm)
    plot_confusion_matrix(cm, class_names_list)

    return cm

if __name__ == '__main__':
    training()
    print("Treinamento da MLP concluído!")

    # Salvar modelo
    torch.save(model.state_dict(), "modelo_mlp.pth")
    print("Modelo MLP salvo com sucesso!")

    # Testar modelo
    test_dataset = datasets.ImageFolder(root="dataset-test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    evaluate_model(model, test_loader, device, class_names)