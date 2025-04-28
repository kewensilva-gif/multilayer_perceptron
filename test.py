import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import string
import torch.nn.functional as F
from mlp import MLP

# Configurações
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_dir = "dataset-test"
output_dir = "dataset-binario"
os.makedirs(output_dir, exist_ok=True)

class_names = ["lampada", "colcheias", "floco", "helice", "tv"]  # adapte conforme suas classes

# Transformação para entrada do modelo
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Modelo
num_classes = len(class_names)
model = MLP(num_classes).to(device)
model.load_state_dict(torch.load("modelo_mlp.pth", map_location=device))
model.eval()

def generate_key(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def transform_image(img_path):
    image = Image.open(img_path).convert("L")  # Grayscale
    return transform(image).unsqueeze(0).to(device)

def binarizar_e_salvar(img_path, nome_saida):
    image = Image.open(img_path).convert("L")  # Grayscale
    image = image.resize((64, 64))

    # Binariza: tudo maior que 0 vira 255
    binary = image.point(lambda p: 255 if p > 0 else 0)

    binary.save(os.path.join(output_dir, nome_saida))

def rename_img(predicted, original_path):
    predicted_class = class_names[predicted.item()] if predicted.item() < len(class_names) else str(predicted.item())
    new_name = f"{predicted_class}_{generate_key()}.png"
    return predicted_class, new_name

def prediction(image):
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted, confidence.item() * 100

def test_images():
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        
        image_tensor = transform_image(img_path)
        predicted, confidence = prediction(image_tensor)
        
        predicted_class, new_name = rename_img(predicted, img_path)

        # Salva imagem binária na nova pasta
        binarizar_e_salvar(img_path, new_name)

        print(f"Imagem: {new_name} -> Predição: {predicted_class} ({confidence:.2f}% de confiança)")

if __name__ == '__main__':
    test_images()
