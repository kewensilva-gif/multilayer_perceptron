import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from torch.utils.data import DataLoader
from mlp import MLP
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

def training(model, criterion, optimizer, train_loader, min_error=0.03, max_epochs=300):
    epoch = 0
    
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

        if (epoch_error <= min_error and epoch_accuracy == 100) or epoch >= max_epochs -1:
            if epoch_error <= min_error:
                print(f"Critério de erro baixo ({epoch_error:.4f} <= {min_error}) atingido.")
            if epoch >= max_epochs -1:
                print(f"Máximo de épocas ({max_epochs}) atingido.")
            break
        epoch += 1

    return epoch

def save_confusion_matrix(cm, class_names_list, save_path):

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Valor Verdadeiro')
    plt.title('Matriz de Confusão')
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de Confusão:")
    print(cm)
    # plot_confusion_matrix(cm, class_names_list)

    accuracy = np.diag(cm).sum() / cm.sum()
    print(f"Acurácia deste Fold: {accuracy * 100:.2f}%\n")
    return accuracy, cm

def training_kfolds(train_dataset, num_classes, device, class_names, batch_size, k, save_path_matrix, learning_rate=0.1):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    fold_accuracies = []
    epochs = []

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

        epoch = training(model, criterion, optimizer, train_loader_fold)

        print(f"\nAvaliação do modelo no Fold {fold + 1}")
        accuracy, cm = evaluate_model(model, val_loader_fold, device)
        save_confusion_matrix(cm, class_names, f"{save_path_matrix}/{fold + 1}.png")
        fold_accuracies.append(accuracy)
        epochs.append(epoch)
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_epochs = np.mean(epochs)

    print("\n===== AVALIAÇÃO FINAL KFOLDS =====")
    print(f"Acurácia Média: {mean_accuracy * 100:.2f}%")
    print(f"Desvio Padrão da Acurácia: {std_accuracy * 100:.2f}%")

    return mean_accuracy, std_accuracy, mean_epochs

def main():
    train_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    while True:
        print("\n=== Menu de Treinamento ===")
        print("1. Treinamento com K-Folds")
        print("2. Treinamento Final com Todos os Dados")
        print("3. Teste de taxa de aprendizado")
        print("0. Sair")

        opcao = int(input("Selecione uma opção: "))

        if opcao == 1:
            batch_size = 10
            k = 5

            training_kfolds(
                train_dataset=train_dataset, 
                num_classes=num_classes, 
                device=device, 
                class_names=class_names,
                batch_size=batch_size, 
                k=k,
                save_path_matrix="kfold-results"
            )
            print("\nK-Fold finalizado")

        elif opcao == 2:
            batch_size = 10
            lr = 0.1
            model = MLP(num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            training(model, criterion, optimizer, train_loader)
            torch.save(model.state_dict(), "modelo_mlp.pth")

        elif opcao == 3:
            learning_rate_list = [0.1, 0.01, 0.001]
            mean_std_list = []
            batch_size = 10
            k = 5
            index = 0
            for lr in learning_rate_list:
                mean_accuracy, std_accuracy, mean_epochs = training_kfolds(
                    train_dataset=train_dataset, 
                    num_classes=num_classes, 
                    device=device, 
                    class_names=class_names,
                    batch_size=batch_size, 
                    k=k,
                    learning_rate=lr,
                    save_path_matrix=f"kfold-results/lr-{index}"
                )

                mean_std_list.append(
                    {
                        "lr": lr,
                        "mean": mean_accuracy,
                        "std": std_accuracy,
                        "mean_epochs": mean_epochs
                    }
                )
                index+=1

            print("\n===== RESULTADO =====")
            print(f"{'Taxa de Aprend.':<20} | {'Acurácia Média':<20} | {'Desvio Padrão':<20} | {'Média de Épocas':<20}")
            print("-" * (22) * 4)

            for result in mean_std_list:
                lr = result['lr']
                mean_acc = f"{result['mean'] * 100:.2f}%"
                std_acc = f"{result['std'] * 100:.2f}%"
                mean_epochs = result['mean_epochs']
                print(f"{lr:<20} | {mean_acc:<20} | {std_acc:<20} | {mean_epochs:<20}")


        elif opcao == 0:
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == '__main__':
    main()