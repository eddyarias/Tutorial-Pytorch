
from google.colab import files
files.upload()  # Se abrirá un diálogo para que subas el archivo kaggle.json

!pip install tqdm


!mkdir -p ~/.kaggle  # Crea el directorio ~/.kaggle si no existe
!mv /content/kaggle\ \(6\).json ~/.kaggle/kaggle.json  # Mueve el archivo y lo renombra a kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!ls -l ~/.kaggle/
!kaggle datasets download -d mczielinski/bitcoin-historical-data

!unzip bitcoin-historical-data.zip -d ./dataset-folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('/content/dataset-folder/btcusd_1-min_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
# Agrupamos los datos en intervalos de 6 minutos
df.set_index('Timestamp', inplace=True)
df = df.resample('6T').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna().reset_index()

# Mostrar las filas que contienen valores nulos antes de rellenarlas
valores_nulos = df[df.isnull().any(axis=1)]
print(valores_nulos)

# Rellenar valores nulos
df.fillna(method='ffill', inplace=True)

# Seleccionamos las columnas de interés
data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
# Verificamos el uso de memoria
print(df.info(memory_usage='deep'))

from sklearn.preprocessing import MinMaxScaler
# Escalamos los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creamos secuencias de entrada y salida
def create_sequences(data, n_steps):
    X = []
    y = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 3])  # Predicción del precio de cierre
    return np.array(X), np.array(y)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

n_steps = 5  # Usamos los últimos 5 intervalos (30 minutos) para predecir el siguiente (6 minutos)
X, y = create_sequences(scaled_data, n_steps)

# Convertimos los datos a tensores de PyTorch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Detectar si la GPU está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando {device}")

# Dividimos en entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Mover los datos a la GPU
X_train = X_train.to(device).float()
y_train = y_train.to(device).float()
X_test = X_test.to(device).float()
y_test = y_test.to(device).float()

class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, hidden_dim, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Capa de embebido para las secuencias de entrada
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Definimos una capa de Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Capa final para producir la salida
        self.fc_out = nn.Linear(hidden_dim * n_steps, output_dim)

    def forward(self, src):
        # Aplicamos embebido
        src = self.input_layer(src)

        # Paso por el Transformer
        output = self.transformer_encoder(src)

        # Aplanamos la salida y la pasamos por la capa final
        output = output.view(output.size(0), -1)
        output = self.fc_out(output)

        return output

# Parámetros del modelo
input_dim = X_train.shape[2]  # Número de características (5: Open, High, Low, Close, Volume)
hidden_dim = 64  # Dimensión de la capa oculta
n_heads = 4  # Número de cabezas en la atención
num_layers = 2  # Número de capas del Transformer
output_dim = 1  # Una salida (predicción del precio de cierre)

# Creamos el modelo
model = TransformerModel(input_dim, n_heads, num_layers, hidden_dim, output_dim)

# Mover el modelo a la GPU
model = model.to(device).float()

# Definimos el optimizador y la función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

from tqdm import tqdm  # Para la barra de progreso

# Función para entrenar el modelo
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        
        # Usar tqdm para mostrar el progreso
        pbar = tqdm(range(0, X_train.size(0), batch_size), desc=f'Epoch {epoch+1}/{epochs}', leave=False)

        for i in pbar:
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # Asegurarnos de que el batch esté en float32
            batch_x = batch_x.float()
            batch_y = batch_y.float()

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Actualizar la barra de progreso con el valor de pérdida
            pbar.set_postfix({'Loss': loss.item()})

        # Evaluamos en el conjunto de prueba
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.float())
            test_loss = criterion(test_outputs, y_test.float())
        model.train()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Test Loss: {test_loss.item()}')

# Entrenar el modelo asegurándonos que esté en float32
train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64)


# Predicción
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Desescalamos los datos para obtener los valores originales
predictions_rescaled = scaler.inverse_transform(
    np.concatenate((X_test[:, -1, :-1].cpu().numpy(), predictions.cpu().numpy()), axis=1))[:, -1]

# Desescalamos también los valores de prueba reales
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((X_test[:, -1, :-1].cpu().numpy(), y_test.cpu().unsqueeze(1).numpy()), axis=1))[:, -1]
