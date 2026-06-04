# Atividade obrigatória

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/Wooo589/rede-neural.git
cd rede-neural
```
### 2. Crie um ambiente virtual

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale o arquivo requirements.txt
```bash
pip install -r requirements.txt
```
## Execução
Ambas as alternativas são válidas, irá depender da sua máquina:
```bash
python3 main.py
```
```bash
python main.py
```
## Testar o efeito de variar a largura e a profundidade da rede neural

| Tentativa | Largura | Profundidade | Efeito |
|-----------|----------|--------------|---------|
| 1 | 16 | 2 | Treinamento rápido, com os valores de perda próximos entre si e baixa capacidade de representação. |
| 2 | 128 | 6 | Maior capacidade de aprendizado, por conta do alcance de 96% de acurácia e da apresentação de um desempenho bom. |
| 3 | 96 | 4 | Equilíbrio entre complexidade e desempenho, por conta de 97% advindo da acurácia, com perdas de treino e teste próximas, o que indica ausência de sobreajuste. |

---

## Treinar o modelo testando diferentes valores de épocas e learning rate, identificando ocorrências de overfitting e underfitting

| Tentativa | Épocas | Learning Rate | Overfitting? | Underfitting? |
|------------|---------|---------------|--------------|---------------|
| 1 | 10 | 0.01 | Não | Sim. Os valores de acurácia indicam um leve underfitting devido ao pequeno número de épocas e à baixa capacidade da rede. |
| 2 | 200 | 0.0005 | Não | Não. |
| 3 | 800 | 0.001 | Não | Não. |

---

## Testar o efeito da regularização e de outros métodos de otimização

| Tentativa | Regularização | Otimizador |
|------------|-------------------------------|------------|
| 1 | 0.001 | Adam |
| 2 | 0.0005 | Adam |
| 3 | 0.0 | RMSprop |

### Explicações adicionais
    - O otimizador Adam apresentou convergência rápida e resultados consistentes durante os experimentos.
    - O RMSprop também apresentou desempenho satisfatório, porém sem ganhos significativos em relação ao Adam para este conjunto de dados.
    - Diante da rotina da faculdade, não foi possível trabalhar em um cenário para verificar overfitting. 
