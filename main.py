import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from neural_network import NeuralNetwork

path = 'data'

# informacoes gerais
epocas = 50
lr = 0.001
arquitetura = [64, 32]
regularizacao = 0.2

# encontrar arquivo csv
csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break
if csv_file is None:
    raise FileNotFoundError("Arquivo CSV não encontrado")

# carregar dataset
df = pd.read_csv(csv_file)

# transformar classes em números
encoder = LabelEncoder()
df["class"] = encoder.fit_transform(df["class"])
print("Classes encontradas:")
for nome, valor in zip(encoder.classes_, encoder.transform(encoder.classes_)):
    print(nome, "->", valor)

# separar atributos e classe
X = df.drop(columns=[
    "class",
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID"
])

y = df["class"]

# normalizar dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# converter para tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)

# criar rede neural
model = NeuralNetwork(input_size=X_train.shape[1], hidden_layers=arquitetura, dropout=regularizacao)

print("\nModelo:")
print(model)

# função de perda
criterion = nn.CrossEntropyLoss()

# otimizador
otimizador = torch.optim.Adam(model.parameters(), lr=lr)

# treinamento
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Test Loss: {test_loss.item():.4f}"
        )

# avaliação
model.eval()
with torch.no_grad():
    train_predictions = model(X_train).argmax(dim=1)
    test_predictions = model(X_test).argmax(dim=1)
    train_accuracy = (train_predictions == y_train).float().mean()
    test_accuracy = (test_predictions == y_test).float().mean()

print("\nResultados")

print("treino:", train_accuracy.item())
print("teste:", test_accuracy.item())

# configuração utilizada

print("\nConfiguração")'

print("Arquitetura:", arquitetura)
print("Épocas:", epochs)
print("Learning Rate:", lr)
print("Regularização:", regularizacao)
