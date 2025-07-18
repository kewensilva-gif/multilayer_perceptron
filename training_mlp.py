import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

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

def predict_image(model_path, image_path, class_names, device):
    """
    Função para fazer predição em uma imagem individual
    
    Args:
        model_path (str): Caminho para o modelo salvo (.pth)
        image_path (str): Caminho para a imagem a ser classificada
        class_names (list): Lista com os nomes das classes
        device (str): Dispositivo (cuda/cpu)
    
    Returns:
        tuple: (classe_predita, confiança, probabilidades_todas_classes)
    """
    
    # Verificar se o arquivo do modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    
    # Verificar se a imagem existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada em: {image_path}")
    
    try:
        # Carregar e preprocessar a imagem
        image = Image.open(image_path)
        
        # Aplicar as mesmas transformações usadas no treinamento
        image_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar dimensão batch
        
        # Carregar o modelo
        num_classes = len(class_names)
        model = MLP(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Fazer predição
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
            all_probabilities = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, all_probabilities
        
    except Exception as e:
        raise RuntimeError(f"Erro ao processar a imagem: {str(e)}")

def show_prediction_results(image_path, predicted_class, confidence, all_probabilities, class_names):
    """
    Função para exibir os resultados da predição de forma visual
    
    Args:
        image_path (str): Caminho da imagem
        predicted_class (str): Classe predita
        confidence (float): Confiança da predição
        all_probabilities (numpy.array): Probabilidades de todas as classes
        class_names (list): Lista com nomes das classes
    """
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mostrar a imagem original
    image = Image.open(image_path)
    ax1.imshow(image, cmap='gray' if image.mode == 'L' else None)
    ax1.set_title(f'Imagem Original\nPredição: {predicted_class}\nConfiança: {confidence:.2%}')
    ax1.axis('off')
    
    # Mostrar probabilidades de todas as classes
    colors = ['red' if class_name == predicted_class else 'blue' for class_name in class_names]
    bars = ax2.bar(class_names, all_probabilities, color=colors, alpha=0.7)
    ax2.set_title('Probabilidades por Classe')
    ax2.set_ylabel('Probabilidade')
    ax2.set_xlabel('Classes')
    ax2.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar, prob in zip(bars, all_probabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

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
        print("4. Predição em imagem individual")
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
            print("\nModelo salvo como 'modelo_mlp.pth'")

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

        elif opcao == 4:
            model_path = "modelo_interacao.pth"
            
            # image_path = "recortes_binarios_image7/objeto_1_01.png"
            image_path = "dataset-test/5_interacao/inter_ventilador_69.png"
            # image_path = "dataset-test/4_tv/tv_CHoPRV.png"
            # image_path = "dataset-test/3_helice/helice_dIDvlT.png"
            # image_path = "dataset-test/1_colcheias/colcheias_1mo8g1.png"
            # image_path = "dataset-test/0_lampada/balao_G9SSUz.png"
            
            try:
                predicted_class, confidence, all_probabilities = predict_image(
                    model_path, image_path, class_names, device
                )
                
                print(f"\n===== RESULTADO DA PREDIÇÃO =====")
                print(f"Classe predita: {predicted_class}")
                print(f"Classe verdadeira: {image_path.split('/')[-2]}")
                print(f"Confiança: {confidence:.2%}")
                print(f"\nProbabilidades de todas as classes:")
                for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
                    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
                
                # Perguntar se deseja mostrar visualização
                show_viz = input("\nDeseja visualizar os resultados? (s/n): ").strip().lower()
                if show_viz == 's':
                    show_prediction_results(image_path, predicted_class, confidence, all_probabilities, class_names)
                    
            except Exception as e:
                print(f"Erro na predição: {str(e)}")

        elif opcao == 0:
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == '__main__':
    main()