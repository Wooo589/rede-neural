import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from neural_network import NeuralNetwork

path = "data"

# informações para realizar os testes do modelo
print("Configuração do modelo")
epocas = int(input("Quantidade de épocas: "))
lr = float(input("Taxa de aprendizado: "))
largura = int(input("Largura da rede: "))
profundidade = int(input("Profundidade da rede: "))
regularizacao = float(input("Regularização: "))

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

model = NeuralNetwork(
    input_size=X_train.shape[1], largura=largura, profundidade=profundidade,dropout=regularizacao
)

print("\nModelo:")
print(model)

# função de perda
criterion = nn.CrossEntropyLoss()

# Métodos de otimização a serem utilizados
print("\nMétodos de otimização:")
print("1 - Adam")
print("2 - RMSprop")
print("3 - SGD")
print("4 - ASGD")

opcao = int(input("Escolha o otimizador: "))

if opcao == 1:
    otimizador = torch.optim.Adam(model.parameters(), lr=lr)
    nome_otimizador = "Adam"

elif opcao == 2:
    otimizador = torch.optim.RMSprop(model.parameters(), lr=lr)
    nome_otimizador = "RMSprop"

elif opcao == 3:
    otimizador = torch.optim.SGD(model.parameters(), lr=lr)
    nome_otimizador = "SGD"

elif opcao == 4:
    otimizador = torch.optim.ASGD(model.parameters(), lr=lr)
    nome_otimizador = "ASGD"

else:
    raise ValueError("Opção inválida")

# treinamento
for epoca in range(epocas):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    otimizador.zero_grad()
    loss.backward()
    otimizador.step()
    if (epoca + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        print(
            f"Epoch {epoca + 1}/{epocas} | "
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

# resultados e configuração
print("\nResultados")
print("Treino:", train_accuracy.item())
print("Teste :", test_accuracy.item())
print("\nConfiguração escolhida")
print("Largura:", largura)
print("Profundidade:", profundidade)
print("Épocas:", epocas)
print("Taxa de aprendizado:", lr)
print("Regularização:", regularizacao)