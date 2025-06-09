# Multilayer Perceptron (MLP)

Este projeto implementa uma rede neural do tipo **Perceptron Multicamadas (MLP)** utilizando **PyTorch**, com foco em classificação de imagens e detecção de alvos nelas.



## 🚀 Configurando o Ambiente Virtual

Siga os passos abaixo para configurar e rodar o projeto corretamente:

### 1️⃣ Criar um ambiente virtual

No terminal ou prompt de comando, execute:

```sh
python -m venv venv
```

### 2️⃣ Ativar o ambiente virtual

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

### 3️⃣ Instalar dependências

Com o ambiente virtual ativado, instale as dependências do projeto executando:

```sh
pip install -r requirements.txt
```

### 4️⃣ Executar o projeto

Se o projeto possui um script principal (por exemplo, `training_mlp.py`, `train.py` ou `main.py`), execute:

```sh
python training_mlp.py
```


## 📁 Estrutura

```
multilayer_perceptron/
├── mlp.py              # Arquitetura da rede MLP
├── training_mlp.py     # Treinamento do modelo
├── test.py             # Avaliação do modelo treinado
├── modelo_mlp.pth      # Pesos salvos da MLP treinada
```


## 📝 Licença

Este projeto está sob a licença MIT.