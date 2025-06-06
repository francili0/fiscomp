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

# Define a função sinc
def sinc_function(x_np):
    # Converte para array numpy se for tensor, para usar np.where
    if isinstance(x_np, torch.Tensor):
        x_np = x_np.numpy()
    return np.where(x_np == 0, 1.0, np.sin(x_np) / x_np).astype(np.float32)

# Parâmetros de treinamento e dados
num_samples_train = 100
num_samples_test = 200
learning_rate = 0.001
epochs = 15000  # Aumentado um pouco, sinc pode ser mais complexa
noise_level = 0.01

# Intervalo para x
x_min, x_max = -10, 10

print(f"Treinando para a função: sinc(x)")

# 1. Geração de Dados de Treinamento
x_train_np = np.random.uniform(x_min, x_max, num_samples_train).reshape(-1, 1).astype(np.float32)
y_train_np = sinc_function(x_train_np)

# Adiciona ruído
noise = np.random.normal(0, noise_level, y_train_np.shape).astype(np.float32)
y_train_np_noisy = y_train_np + noise

# Converte dados de treinamento para tensores PyTorch
x_train_torch = torch.from_numpy(x_train_np) # x_train_np já está filtrado (sempre foi)
y_train_torch = torch.from_numpy(y_train_np_noisy)

# 2. Definição do Modelo, Função de Perda e Otimizador
input_size = 1
hidden_sizes = [10, 10, 10] # Mesma arquitetura
output_size = 1

model_sinc = MLP(input_size, hidden_sizes, output_size)
criterion_sinc = nn.MSELoss()
optimizer_sinc = optim.Adam(model_sinc.parameters(), lr=learning_rate)

# 3. Loop de Treinamento
for epoch in range(epochs):
    model_sinc.train()
    outputs = model_sinc(x_train_torch)
    loss = criterion_sinc(outputs, y_train_torch)
    optimizer_sinc.zero_grad()
    loss.backward()
    optimizer_sinc.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Função: sinc(x), Época [{epoch+1}/{epochs}], Perda: {loss.item():.6f}')

# 4. Geração de Dados de Teste e Avaliação
x_test_np = np.linspace(x_min, x_max, num_samples_test).reshape(-1, 1).astype(np.float32)
x_test_torch = torch.from_numpy(x_test_np)

model_sinc.eval()
with torch.no_grad():
    y_test_pred_torch = model_sinc(x_test_torch)

y_test_pred_np = y_test_pred_torch.numpy()
y_test_true_np = sinc_function(x_test_np) # Recalcula os valores verdadeiros sem ruído

mse = mean_squared_error(y_test_true_np, y_test_pred_np)
print(f"Função: sinc(x) - PyTorch - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}\n")

# 5. Plotagem dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(x_train_np, y_train_np_noisy, alpha=0.5, label='Dados de Treinamento (com ruído)', color='orange')
plt.plot(x_test_np, y_test_true_np, label='Função sinc(x) Verdadeira', color='blue', linewidth=2)
plt.plot(x_test_np, y_test_pred_np, label='Interpolação da Rede Neural (PyTorch)', color='red', linestyle='--', linewidth=2)
plt.title('Interpolação PyTorch da Função sinc(x) = sin(x)/x')
plt.xlabel('x')
plt.ylabel('sinc(x)')
plt.legend()
plt.grid(True)
plt.show()