import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

variacao = 25

def mostrarImagem(image, titulo):
    if len(image.shape) == 3 and image.shape[2] == 3:
        test_pixel = image[0, 0]
        test_hsv = cv2.cvtColor(np.uint8([[test_pixel]]), cv2.COLOR_RGB2HSV)[0][0]
        if np.array_equal(test_pixel, test_hsv):
            img_plot = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            formato = "hsv"
        else:
            img_plot = image
            formato = "rgb"
    else:
        img_plot = image
        formato = "cinza"
    plt.figure(figsize=(3,3))
    plt.imshow(img_plot, cmap="gray" if formato == "cinza" else None)
    plt.title(f"{titulo} ({formato})")
    plt.axis("off")
    plt.show()

def salvarImagem(caminho, image):
    if len(image.shape) == 2:
        image = (image * 255).astype(np.uint8) if image.max() == 1 else image
        cv2.imwrite(caminho, image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        test_pixel = image[0, 0]
        test_hsv = cv2.cvtColor(np.uint8([[test_pixel]]), cv2.COLOR_RGB2HSV)[0][0]
        if np.array_equal(test_pixel, test_hsv):
            img_plot = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            img_plot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(caminho, img_plot)
    else:
        cv2.imwrite(caminho, image)

def definirMascaraOBJ(pasta_saida, imagem, imagem_hsv, c, n):
    selected_color = rgb_to_hsv(*c)
    h_value = int(selected_color[0])
    h_min = max(0, h_value - variacao)
    h_max = min(179, h_value + variacao)
    lower_red = np.array([h_min, 70, 20], dtype=np.uint8)
    upper_red = np.array([h_max, 255, 150], dtype=np.uint8)
    mask = cv2.inRange(imagem_hsv, lower_red, upper_red)
    return mask

def rgb_to_hsv(r, g, b):
    rgb_color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def carregaImagem(imagem_caminho):
    imagem = cv2.imread(imagem_caminho)
    if imagem is None:
        return None
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

def removeRuido(mask, deltaRuido):
    return cv2.medianBlur(mask, deltaRuido)

def segObjetos(imagem_caminho, pasta_saida, nomeImagem, deltaRuido=3, tamSeg=200):
    imagem = carregaImagem(imagem_caminho)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {imagem_caminho}")
        return []

    for pasta in ["target", "binary", "binary_view", "colored", "masks"]:
        os.makedirs(f"{pasta}_{pasta_saida}", exist_ok=True)

    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)

    cores = [[196, 125, 137], [198, 131, 127], [173, 105, 129], [154, 87, 105]]
    masks = [definirMascaraOBJ(pasta_saida, imagem, imagem_hsv, cor, i+1) for i, cor in enumerate(cores)]
    mask_red = sum(masks)

    mask_red = removeRuido(mask_red, deltaRuido)
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    multCoordenadas = []

    for i, contorno in enumerate(contornos):
        mask_objeto = np.zeros_like(mask_red)
        cv2.drawContours(mask_objeto, [contorno], -1, 255, thickness=cv2.FILLED)
        mask_objeto_binario = (mask_objeto > 0).astype(np.uint8)

        x, y, w, h = cv2.boundingRect(contorno)
        objeto = imagem[y:y+h, x:x+w]
        altura, largura, _ = objeto.shape
        total_pixels = altura * largura

        if total_pixels > 1000 and altura > 20 and largura > 20:
            segmento_mascara = mask_red[y:y+h, x:x+w]
            mascara_para_salvar = (segmento_mascara > 0).astype(np.uint8) * 255

            objeto_rgba = cv2.cvtColor(objeto, cv2.COLOR_RGB2RGBA)
            objeto_rgba[:, :, 3] = segmento_mascara
            objeto_bgra = cv2.cvtColor(objeto_rgba, cv2.COLOR_RGBA2BGRA)

            segmento_recortado = mask_objeto[y:y+h, x:x+w]
            segmento_binario = (segmento_recortado > 0).astype(np.uint8)

            # cv2.imwrite(f"masks_{pasta_saida}/{nomeImagem}_{i+1}.png", mascara_para_salvar)
            # cv2.imwrite(f"target_{pasta_saida}/{nomeImagem}_{i+1}_seg.png", objeto_bgra)
            # salvarImagem(f"colored_{pasta_saida}/{nomeImagem}_{i+1}_seg.png", objeto)
            # salvarImagem(f"binary_view_{pasta_saida}/{nomeImagem}_{i+1}_seg.png", segmento_recortado)
            # salvarImagem(f"binary_{pasta_saida}/{nomeImagem}_{i+1}_seg.png", segmento_binario)

            multCoordenadas.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'target_rectangle': f"colored_{pasta_saida}/{nomeImagem}_{i+1}_seg.png",
                'binary': f"binary_{pasta_saida}/{nomeImagem}_{i+1}_seg.png",
                'target': f"target_{pasta_saida}/{nomeImagem}_{i+1}_seg.png",
                'mask': f"masks_{pasta_saida}/{nomeImagem}_{i+1}.png",
                'mask_red' : mask_red,
                'contour': contorno,
            })
            print(f"Coordenadas do objeto {i+1}: x={x}, y={y}, w={w}, h={h}")
        else:
            print(f"Objeto {i+1} ignorado (ruído, {total_pixels}px)")
    return multCoordenadas

def converterParaBinario(imagem_rgb):
    """
    Converte uma imagem RGB para binário usando as mesmas cores de detecção
    """
    imagem_hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    
    # Usar as mesmas cores do algoritmo original
    cores = [[196, 125, 137], [198, 131, 127], [173, 105, 129], [154, 87, 105]]
    masks = []
    
    for cor in cores:
        selected_color = rgb_to_hsv(*cor)
        h_value = int(selected_color[0])
        h_min = max(0, h_value - variacao)
        h_max = min(179, h_value + variacao)
        lower_red = np.array([h_min, 70, 20], dtype=np.uint8)
        upper_red = np.array([h_max, 255, 150], dtype=np.uint8)
        mask = cv2.inRange(imagem_hsv, lower_red, upper_red)
        masks.append(mask)
    
    # Combinar todas as máscaras
    mask_combined = sum(masks)
    
    # Remover ruído
    mask_combined = cv2.medianBlur(mask_combined, 3)
    kernel = np.ones((5,5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    
    # Converter para binário (0 ou 255)
    mask_binario = (mask_combined > 0).astype(np.uint8) * 255
    
    return mask_binario

# Processamento de uma imagem para obter as coordenadas
print("Processando uma imagem para obter coordenadas dos alvos...")
coordenada_image1 = segObjetos("dataset_interacao/imagens2/base.jpg", "base", "base")

# Carregar image7 para fazer os recortes
image7 = cv2.imread("dataset_interacao/imagens2/image8.jpg")
image7_rgb = cv2.cvtColor(image7, cv2.COLOR_BGR2RGB)

# Criar pasta para salvar os recortes binários da image7
os.makedirs("recortes_binarios_image7", exist_ok=True)

print(f"\nResultados da image1:")
print(f"Objetos detectados: {len(coordenada_image1)}")

if coordenada_image1:
    print(f"\nProcessando recortes da image7 usando coordenadas da image1...")
    
    for i, coord in enumerate(coordenada_image1):
        # Fazer recorte na image7 usando coordenadas da image1
        x, y, w, h = coord['x'], coord['y'], coord['w'], coord['h']
        
        # Verificar se as coordenadas estão dentro dos limites da image7
        altura_img, largura_img = image7.shape[:2]
        
        if x + w <= largura_img and y + h <= altura_img:
            # Recortar da image7
            recorte_image7 = image7_rgb[y:y+h, x:x+w]
            
            # Converter para binário
            recorte_binario = converterParaBinario(recorte_image7)
            
            # Converter para formato 0 e 1 (pixel 0 permanece 0, pixel > 0 vira 1)
            recorte_01 = (recorte_binario > 0).astype(np.uint8)
            
            # Salvar recorte binário (0-255)
            cv2.imwrite(f"recortes_binarios_image7/objeto_{i+1}_binario.png", recorte_binario)
            
            # Salvar recorte formato 0-1 como PNG
            cv2.imwrite(f"recortes_binarios_image7/objeto_{i+1}_01.png", recorte_01)
            
            # Salvar recorte colorido para comparação (opcional)
            recorte_bgr = cv2.cvtColor(recorte_image7, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"recortes_binarios_image7/objeto_{i+1}_colorido.png", recorte_bgr)
            
            print(f"Objeto {i+1}: Recortes salvos (coordenadas: x={x}, y={y}, w={w}, h={h})")
            print(f"  - Colorido: objeto_{i+1}_colorido.png")
            print(f"  - Binário (0-255): objeto_{i+1}_binario.png") 
            print(f"  - Formato 0-1: objeto_{i+1}_01.png")
        else:
            print(f"Objeto {i+1}: Coordenadas fora dos limites da image7")
    
    print(f"\nRecortes binários da image7 salvos na pasta 'recortes_binarios_image7/'")
else:
    print("Nenhum objeto detectado na image1!")

cv2.waitKey(0)