import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision import datasets, transforms
from mlp import MLP # Importa sua classe MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# --- 1. Definir as Transformações para as Imagens
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),       
    # transforms.Normalize((0.5,), (0.5,)) # Apenas para imagens colordas
])

# --- 2. Carregar o Dataset Completo Usando ImageFolder ---
dataset_path = 'dataset-train'
full_dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=data_transforms
)

# Verifica se o dataset foi carregado corretamente printando pastas
class_to_idx = full_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"Mapeamento de classes: {class_to_idx}")
print(f"Total de imagens carregadas: {len(full_dataset)}")

# --- 3. Defina os parâmetros da validação cruzada ---
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# --- 4. Salva informações ---
all_fold_accuracies = []
all_fold_losses = []

# Parâmetros do modelo e treinamento
input_size_grayscale = 64 * 64 * 1 
output_size = len(class_to_idx)
learning_rate = 0.001
num_epochs = 20 # Número de épocas por fold
batch_size = 8

print(f"Iniciando validação cruzada com {num_folds} folds...")

# --- 5. Loop de Treinamento e Avaliação para cada Fold ---
for fold, (train_index, val_index) in enumerate(kf.split(full_dataset)):
    print(f"index:{val_index}, train{train_index}")
    print(f"\n--- Fold {fold+1}/{num_folds} ---")

    # Cria subconjuntos (Subsets) do dataset completo para cada fold
    train_subset = Subset(full_dataset, train_index)
    val_subset = Subset(full_dataset, val_index)
    print(f"  Tamanho do conjunto de treinamento: {len(train_subset)}")
    print(f"  Tamanho do conjunto de validação: {len(val_subset)}")

    # Cria DataLoaders para o fold atual a partir dos Subsets
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

    # Inicializar o modelo, função de perda e otimizador para cada fold
    # Isso garante que o modelo comece "do zero" para cada fold
    model = MLP(output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Loop de Treinamento para o Fold Atual
    model.train()
    for epoch in range(num_epochs):
        current_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item() * images.size(0) # Acumula perda ponderada pelo batch_size

        avg_epoch_loss = current_loss / len(train_subset)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.5f}')

    # --- Avaliar o Modelo no Conjunto de Validação do Fold Atual ---
    model.eval() # Modo de avaliação
    correct_predictions = 0
    total_samples = 0
    fold_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            fold_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_fold_loss = fold_loss / total_samples
    accuracy = 100 * correct_predictions / total_samples
    print(f'  Accuracy for Fold {fold+1}: {accuracy:.2f}%')
    print(f'  Loss for Fold {fold+1}: {avg_fold_loss:.4f}')

    all_fold_accuracies.append(accuracy)
    all_fold_losses.append(avg_fold_loss)

# --- 6. Resultados Finais da Validação Cruzada ---
print("\n--- Resultados Finais da Validação Cruzada ---")
print(f"Accuracies por fold: {all_fold_accuracies}")
print(f"Losses por fold: {all_fold_losses}")
print(f"Média da Accuracy: {np.mean(all_fold_accuracies):.2f}% +/- {np.std(all_fold_accuracies):.2f}")
print(f"Média da Loss: {np.mean(all_fold_losses):.4f} +/- {np.std(all_fold_losses):.4f}")