# ShapeNet MLP — Classificação de Imagens com Perceptron Multicamadas

> Rede neural MLP em PyTorch para classificação de imagens 64x64 em 5 classes de formas/objetos, com validação cruzada K-Fold e busca de learning rate.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Classes do Dataset](#classes-do-dataset)
- [Arquitetura da Rede](#arquitetura-da-rede)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Resultados](#resultados)

---

## Sobre o Projeto

**ShapeNet MLP** (nome sugerido para substituir "multilayer_perceptron") implementa uma rede neural do tipo **Perceptron Multicamadas (MLP)** para classificar imagens em escala de cinza (64×64 pixels) em **5 categorias** de formas simples. O projeto inclui treinamento com validação cruzada K-Fold, comparação de taxas de aprendizado e pipeline de inferência com binarização.

---

## Classes do Dataset

| Classe | Label | Exemplos |
|---|---|---|
| 0 | Lâmpada | 21 treino / 10 teste |
| 1 | Colcheias (notas musicais) | 21 treino / 10 teste |
| 2 | Floco de neve | 21 treino / 10 teste |
| 3 | Hélice | 21 treino / 10 teste |
| 4 | TV | 21 treino / 10 teste |

**Total:** 105 imagens de treino + 50 imagens de teste

---

## Arquitetura da Rede

```
Input (64×64 = 4096) → Flatten → Linear(4096, 256) → ReLU → Linear(256, 5) → Output
```

| Camada | Entrada | Saída |
|---|---|---|
| Flatten | 64×64 | 4096 |
| Linear + ReLU | 4096 | 256 |
| Linear (output) | 256 | 5 |

> O arquivo `mlp.py` contém 5 variações de arquitetura comentadas (64 a 256 neurônios, 1 a 2 camadas ocultas).

---

## Funcionalidades

| Script | Funcionalidade |
|---|---|
| `mlp.py` | Definição da arquitetura MLP (`nn.Module`) |
| `training_mlp.py` | Treinamento com menu interativo (K-Fold, treino final, comparação de LR) |
| `test.py` | Inferência em imagens de teste + binarização + exportação |

### Menu de Treinamento (`training_mlp.py`)

1. **K-Fold Cross-Validation** (k=5) — salva matriz de confusão por fold
2. **Treinamento final** — treina em todos os dados, salva pesos em `modelo_mlp.pth`
3. **Comparação de Learning Rate** — testa 0.1, 0.01, 0.001 com K-Fold

### Parâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Loss | CrossEntropyLoss |
| Critério de parada | Erro ≤ 0.03 com 100% accuracy **ou** 300 épocas |
| K-Fold | k = 5 |
| Learning rates testadas | 0.1, 0.01, 0.001 |

---

## Instalação

### 1. Criar ambiente virtual

```sh
python -m venv venv
```

### 2. Ativar o ambiente virtual

- **Windows (CMD)**:
  ```sh
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```sh
  venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```sh
  source venv/bin/activate
  ```

### 3. Instalar dependências

```sh
pip install -r requirements.txt
```

---

## Uso

### Treinar o modelo

```sh
python training_mlp.py
```

O menu interativo será exibido com as opções de treinamento.

### Executar inferência

```sh
python test.py
```

Classifica as imagens em `dataset-test/` e exporta versões binarizadas para `dataset-binario/`.


## Estrutura de Pastas

```
ShapeNet-MLP/
├── mlp.py                # Arquitetura da rede MLP (nn.Module)
├── training_mlp.py       # Treinamento com K-Fold, LR search, menu interativo
├── test.py               # Inferência + binarização de imagens
├── modelo_mlp.pth        # Pesos salvos da MLP treinada
├── requirements.txt      # Dependências Python
├── dataset-train/        # 105 imagens de treino (5 classes × ~21)
│   ├── 0_lampada/
│   ├── 1_colcheias/
│   ├── 2_floco/
│   ├── 3_helice/
│   └── 4_tv/
├── dataset-test/         # 50 imagens de teste
├── dataset-binario/      # Saída binarizada do test.py
└── kfold-results/        # Matrizes de confusão por fold e por LR
    ├── 1.png ... 5.png
    ├── lr-0/
    ├── lr-1/
    └── lr-2/
```

---

## Resultados

Os resultados do K-Fold e comparação de learning rates são salvos como imagens de matrizes de confusão na pasta `kfold-results/`.

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| **PyTorch** | Framework de deep learning |
| **torchvision** | Transforms e carregamento de datasets |
| **NumPy / SciPy** | Computação numérica |
| **scikit-learn** | Métricas (K-Fold, confusion matrix) |
| **Matplotlib / Seaborn** | Visualização de resultados |
| **Pillow** | Processamento de imagens |

---

## Licença

Este projeto está sob a licença MIT.