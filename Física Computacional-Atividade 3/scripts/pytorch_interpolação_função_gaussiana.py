import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Configurações
np.random.seed(42)
torch.manual_seed(42)

# Definição da Rede Neural (a mesma classe MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        camadas = []
        tamanhos = [input_size] + hidden_sizes + [output_size]
        for i in range(len(tamanhos) - 1):
            camadas.append(nn.Linear(tamanhos[i], tamanhos[i+1]))
            if i < len(tamanhos) - 2: # Não adiciona ativação na última camada
                camadas.append(nn.Tanh())
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

# Define a função Gaussiana
def gaussian_function(x_np):
    if isinstance(x_np, torch.Tensor):
        x_np = x_np.numpy()
    return np.exp(-x_np**2).astype(np.float32)

# Parâmetros de treinamento e dados
num_samples_train = 100
num_samples_test = 200
learning_rate = 0.001
epochs = 10000 # A Gaussiana é suave, pode convergir mais rápido que a sinc
noise_level = 0.01

# Intervalo para x. A Gaussiana é mais pronunciada perto de 0.
x_min, x_max = -5, 5 # Similar ao script scikit-learn ajustado

print(f"Treinando para a função: Gaussiana e^(-x^2)")

# 1. Geração de Dados de Treinamento
x_train_np = np.random.uniform(x_min, x_max, num_samples_train).reshape(-1, 1).astype(np.float32)
y_train_np = gaussian_function(x_train_np)

# Adiciona ruído
noise = np.random.normal(0, noise_level, y_train_np.shape).astype(np.float32)
y_train_np_noisy = y_train_np + noise

# Converte dados de treinamento para tensores PyTorch
x_train_torch = torch.from_numpy(x_train_np)
y_train_torch = torch.from_numpy(y_train_np_noisy)

# 2. Definição do Modelo, Função de Perda e Otimizador
input_size = 1
hidden_sizes = [10, 10, 10] # Mesma arquitetura
output_size = 1

model_gauss = MLP(input_size, hidden_sizes, output_size)
criterion_gauss = nn.MSELoss()
optimizer_gauss = optim.Adam(model_gauss.parameters(), lr=learning_rate)

# 3. Loop de Treinamento
for epoch in range(epochs):
    model_gauss.train()
    outputs = model_gauss(x_train_torch)
    loss = criterion_gauss(outputs, y_train_torch)
    optimizer_gauss.zero_grad()
    loss.backward()
    optimizer_gauss.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Função: Gaussiana, Época [{epoch+1}/{epochs}], Perda: {loss.item():.6f}')

# 4. Geração de Dados de Teste e Avaliação
x_test_np = np.linspace(x_min, x_max, num_samples_test).reshape(-1, 1).astype(np.float32)
x_test_torch = torch.from_numpy(x_test_np)

model_gauss.eval()
with torch.no_grad():
    y_test_pred_torch = model_gauss(x_test_torch)

y_test_pred_np = y_test_pred_torch.numpy()
y_test_true_np = gaussian_function(x_test_np) # Valores verdadeiros sem ruído

mse = mean_squared_error(y_test_true_np, y_test_pred_np)
print(f"Função: Gaussiana - PyTorch - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}\n")

# 5. Plotagem dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(x_train_np, y_train_np_noisy, alpha=0.5, label='Dados de Treinamento (com ruído)', color='orange')
plt.plot(x_test_np, y_test_true_np, label='Função $e^{-x^2}$ Verdadeira', color='blue', linewidth=2) # Usando LaTeX para a fórmula
plt.plot(x_test_np, y_test_pred_np, label='Interpolação da Rede Neural (PyTorch)', color='red', linestyle='--', linewidth=2)
plt.title('Interpolação PyTorch da Função Gaussiana $e^{-x^2}$')
plt.xlabel('x')
plt.ylabel('$e^{-x^2}$')
plt.legend()
plt.grid(True)
plt.show()