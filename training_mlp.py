import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mlp import MLP

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

# Modelo, perda e otimizador
num_classes = len(train_dataset.classes)
model = MLP(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def training():
    epoch = 1
    while True:
        model.train()
        cumulative_batch_error = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print(f"Labels: {labels}")
            optimizer.zero_grad()
            outputs = model(images)
            # print(f"\n\nOutputs: {outputs}")
            error = criterion(outputs, labels)
            # print(f"\n\nLoss outputs: {loss}")
            error.backward()
            optimizer.step()

            cumulative_batch_error += error.item()
            [0.9, 0.0, 0.1]
            # Cálculo da acurácia
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_error = cumulative_batch_error / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Época {epoch+1} | Loss: {epoch_error:.4f} | Acurácia: {epoch_accuracy:.2f}%")

        if(epoch_error <= 0.3): break
        epoch+=1

if __name__ == '__main__':
    training()
    print("Treinamento da MLP concluído!")

    torch.save(model.state_dict(), "modelo_mlp.pth")
    print("Modelo MLP salvo com sucesso!")
