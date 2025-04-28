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

# Treinamento
def training(num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            print(f"Labels: {labels}")
            optimizer.zero_grad()
            outputs = model(images)
            print(f"\n\nOutputs: {outputs}")
            loss = criterion(outputs, labels)
            print(f"\n\nLoss outputs: {loss}")
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Cálculo da acurácia
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Época {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acurácia: {epoch_acc:.2f}%")

if __name__ == '__main__':
    training()
    print("Treinamento da MLP concluído!")

    torch.save(model.state_dict(), "modelo_mlp.pth")
    print("Modelo MLP salvo com sucesso!")
