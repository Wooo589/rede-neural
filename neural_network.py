import torch.nn as nn

# Rede neural de classificação de galáxias, estrelas e quasares

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout):
        super().__init__()
        
        # camadas
        layers = []

        # adicionar os dados nas camadas
        for neurons in hidden_layers:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = neurons
        layers.append(nn.Linear(input_size, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
