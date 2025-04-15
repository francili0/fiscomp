import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.tanh(x)

plt.plot(x, y)
plt.title("Função tanh")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.show()

import numpy as np

# Função de ativação sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

# Dados de entrada e saída esperada
entrada = np.array([[0,0], [0,1], [1,0], [1,1]])
saida_esperada = np.array([[0], [1], [1], [0]])

# Inicialização de pesos aleatórios
np.random.seed(42)
pesos0 = 2 * np.random.rand(2, 4) - 1
pesos1 = 2 * np.random.rand(4, 1) - 1

# Treinamento
for i in range(10000):
    camada0 = entrada
    camada1 = sigmoid(np.dot(camada0, pesos0))
    camada2 = sigmoid(np.dot(camada1, pesos1))

    erro = saida_esperada - camada2
    if i % 1000 == 0:
        print(f"Erro na iteração {i}: {np.mean(np.abs(erro))}")

    delta2 = erro * sigmoid_derivada(camada2)
    delta1 = delta2.dot(pesos1.T) * sigmoid_derivada(camada1)

    pesos1 += camada1.T.dot(delta2)
    pesos0 += camada0.T.dot(delta1)

# Teste
print("\nSaída final após o treinamento:")
print(camada2)

import numpy as np

# Funções de ativação e suas derivadas
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - x**2  # x já é tanh(x), não precisa recalcular

# Classe da Rede Neural
class RedeNeural:
    def __init__(self):
        np.random.seed(42)
        self.pesos0 = 2 * np.random.rand(2, 4) - 1  # Entrada (2) → Oculta (4)
        self.pesos1 = 2 * np.random.rand(4, 1) - 1  # Oculta (4) → Saída (1)

    def feedforward(self, entrada):
        self.camada0 = entrada
        self.camada1 = tanh(np.dot(self.camada0, self.pesos0))
        self.camada2 = tanh(np.dot(self.camada1, self.pesos1))
        return self.camada2

    def backpropagation(self, saida_esperada):
        erro = saida_esperada - self.camada2
        delta2 = erro * tanh_derivada(self.camada2)
        delta1 = delta2.dot(self.pesos1.T) * tanh_derivada(self.camada1)

        # Atualização dos pesos
        self.pesos1 += self.camada1.T.dot(delta2)
        self.pesos0 += self.camada0.T.dot(delta1)

        return np.mean(np.abs(erro))

    def treinar(self, entrada, saida_esperada, epocas=10000):
        for i in range(epocas):
            self.feedforward(entrada)
            erro = self.backpropagation(saida_esperada)
            if i % 1000 == 0:
                print(f"Erro na iteração {i}: {erro:.4f}")

# Dados de entrada e saída
entrada = np.array([[0,0], [0,1], [1,0], [1,1]])
saida_esperada = np.array([[0], [1], [1], [0]])

# Criando e treinando a rede
rede = RedeNeural()
rede.treinar(entrada, saida_esperada)

# Resultado final
saida_final = rede.feedforward(entrada)
print("\nSaída final após o treinamento:")
print(saida_final)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Funções de ativação tanh e derivada
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - x ** 2

# Dados de entrada e saída
entrada = np.array([[0,0], [0,1], [1,0], [1,1]])
saida_esperada = np.array([[0], [1], [1], [0]])

# Rede Neural simples com tanh
class RedeNeural:
    def __init__(self):
        np.random.seed(42)
        self.pesos0 = 2 * np.random.rand(2, 4) - 1
        self.pesos1 = 2 * np.random.rand(4, 1) - 1

    def feedforward(self, entrada):
        self.camada0 = entrada
        self.camada1 = tanh(np.dot(self.camada0, self.pesos0))
        self.camada2 = tanh(np.dot(self.camada1, self.pesos1))
        return self.camada2

    def backpropagation(self, saida_esperada):
        erro = saida_esperada - self.camada2
        delta2 = erro * tanh_derivada(self.camada2)
        delta1 = delta2.dot(self.pesos1.T) * tanh_derivada(self.camada1)

        self.pesos1 += self.camada1.T.dot(delta2)
        self.pesos0 += self.camada0.T.dot(delta1)

        return np.mean(np.abs(erro))

# Inicializa a rede e armazena dados para visualização
rede = RedeNeural()
epocas = 2000
erros = []
saidas = []

for epoca in range(epocas):
    saida = rede.feedforward(entrada)
    erro = rede.backpropagation(saida_esperada)
    erros.append(erro)
    saidas.append(saida.copy())

# ---------- VISUALIZAÇÃO COM ANIMAÇÃO ---------- #

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

# Gráfico 1: erro por época
ax1.set_title("Erro médio por época")
ax1.set_xlim(0, epocas)
ax1.set_ylim(0, 1)
erro_line, = ax1.plot([], [], color='blue')

# Gráfico 2: saída da rede para as 4 entradas
ax2.set_title("Saída da rede para o problema XOR")
ax2.set_ylim(-0.5, 1.5)
ax2.set_xticks(range(4))
ax2.set_xticklabels(['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
barras = ax2.bar(range(4), [0, 0, 0, 0], color='green')

def animar(i):
    erro_line.set_data(range(i+1), erros[:i+1])
    for j, bar in enumerate(barras):
        bar.set_height(saidas[i][j][0])
    return erro_line, *barras

ani = FuncAnimation(fig, animar, frames=epocas, interval=1, blit=True, repeat=False)
plt.tight_layout()
plt.show()

import numpy as np

# Funções de ativação
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - x**2  # Derivada já recebe tanh(x)

# Dados do problema XOR
entrada = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

saida_esperada = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Inicializar pesos aleatórios
np.random.seed(42)
pesos0 = 2 * np.random.rand(2, 4) - 1  # 2 entradas → 4 neurônios na camada oculta
pesos1 = 2 * np.random.rand(4, 1) - 1  # 4 neurônios → 1 saída

# Treinamento
epocas = 10000
for i in range(epocas):
    # Feedforward
    camada1 = tanh(np.dot(entrada, pesos0))
    camada2 = tanh(np.dot(camada1, pesos1))

    # Backpropagation
    erro = saida_esperada - camada2
    delta2 = erro * tanh_derivada(camada2)
    delta1 = delta2.dot(pesos1.T) * tanh_derivada(camada1)

    # Ajuste dos pesos
    pesos1 += camada1.T.dot(delta2)
    pesos0 += entrada.T.dot(delta1)

    # Exibir erro ocasionalmente
    if i % 1000 == 0:
        print(f"Época {i} - Erro médio: {np.mean(np.abs(erro)):.4f}")

# Resultado final
print("\nSaída final da rede após o treinamento:")
print(camada2.round(3))
       