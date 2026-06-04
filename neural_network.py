import torch.nn as nn

# rede neural para classificar galáxias, estrelas e quasares
class NeuralNetwork(nn.Module):

    def __init__(self, input_size, largura, profundidade, dropout):

        super().__init__()

        # lista de camadas da rede
        layers = []

        # criar camadas ocultas
        for _ in range(profundidade):
            layers.append(nn.Linear(input_size, largura))
            layers.append(nn.ReLU())

            # regularização
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = largura
        layers.append(nn.Linear(input_size, 3))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)