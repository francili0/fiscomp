import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Configurações
np.random.seed(42)
torch.manual_seed(42)

# Definição da Rede Neural
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        camadas = []
        tamanhos = [input_size] + hidden_sizes + [output_size]
        for i in range(len(tamanhos) - 1):
            camadas.append(nn.Linear(tamanhos[i], tamanhos[i+1]))
            if i < len(tamanhos) - 2: # Não adiciona ativação na última camada (antes da saída)
                camadas.append(nn.Tanh()) # Função de ativação tanh
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

# Funções a serem aprendidas
functions = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan
}

# Parâmetros de treinamento e dados
num_samples_train = 100
num_samples_test = 200
learning_rate = 0.001 # Mantido igual ao MLPRegressor
epochs = 10000 # Número de épocas de treinamento. Pode precisar de ajuste.

# Intervalo para x
x_min, x_max = 0, 2 * np.pi

# Criação da figura para os subplots
plt.figure(figsize=(12, 15))

for i, (name, func) in enumerate(functions.items(), 1):
    print(f"Treinando para a função: {name}(x)")

    # 1. Geração de Dados de Treinamento
    x_train_np = np.random.uniform(x_min, x_max, num_samples_train).reshape(-1, 1).astype(np.float32)
    y_train_np = func(x_train_np).astype(np.float32)

    # Tratamento especial para a função tangente
    if name == 'tan':
        y_train_np[np.abs(y_train_np) > 10] = np.nan
        mask_train_finite = ~np.isnan(y_train_np).flatten()
        x_train_np_filtered = x_train_np[mask_train_finite]
        y_train_np_filtered = y_train_np[mask_train_finite].reshape(-1,1) # Manter como coluna
    else:
        # Adiciona ruído moderado para seno e cosseno
        noise = np.random.normal(0, 0.1, y_train_np.shape).astype(np.float32)
        y_train_np_filtered = y_train_np + noise
        x_train_np_filtered = x_train_np

    # Converte dados de treinamento para tensores PyTorch
    x_train_torch = torch.from_numpy(x_train_np_filtered)
    y_train_torch = torch.from_numpy(y_train_np_filtered)

    # 2. Definição do Modelo, Função de Perda e Otimizador
    input_size = 1
    hidden_sizes = [10, 10, 10] # 3 camadas ocultas com 10 neurônios cada
    output_size = 1
    
    model = MLP(input_size, hidden_sizes, output_size)
    criterion = nn.MSELoss() # Erro Quadrático Médio (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Otimizador Adam

    # 3. Loop de Treinamento
    for epoch in range(epochs):
        model.train() # Coloca o modelo em modo de treinamento

        # Forward pass: calcula as predições
        outputs = model(x_train_torch)
        # Calcula a perda
        loss = criterion(outputs, y_train_torch)

        # Backward pass e otimização
        optimizer.zero_grad() # Zera os gradientes acumulados
        loss.backward()       # Calcula os gradientes da perda em relação aos parâmetros
        optimizer.step()      # Atualiza os pesos do modelo

        if (epoch + 1) % 1000 == 0:
            print(f'Função: {name}, Época [{epoch+1}/{epochs}], Perda: {loss.item():.6f}')

    # 4. Geração de Dados de Teste e Avaliação
    x_test_np = np.linspace(x_min, x_max, num_samples_test).reshape(-1, 1).astype(np.float32)
    x_test_torch = torch.from_numpy(x_test_np)

    model.eval() # Coloca o modelo em modo de avaliação (desativa dropout, batchnorm, etc.)
    with torch.no_grad(): # Desabilita o cálculo de gradientes para a avaliação
        y_test_pred_torch = model(x_test_torch)
    
    y_test_pred_np = y_test_pred_torch.numpy() # Converte predições para numpy para plotagem e MSE
    y_test_true_np = func(x_test_np).astype(np.float32)

    if name == 'tan':
        y_test_true_np[np.abs(y_test_true_np) > 10] = np.nan
        # Para o MSE, precisamos de uma máscara para os NaNs também nas predições, se houver,
        # mas é mais seguro comparar com y_test_true_np que já tem NaNs onde deveria.
        mask_eval_finite = ~np.isnan(y_test_true_np.flatten())
        mse = mean_squared_error(y_test_true_np[mask_eval_finite], y_test_pred_np[mask_eval_finite])
    else:
        mse = mean_squared_error(y_test_true_np, y_test_pred_np)
    print(f"Função: {name} - PyTorch - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}\n")

    # 5. Plotagem dos resultados
    plt.subplot(3, 1, i)
    plt.scatter(x_train_np_filtered, y_train_np_filtered, alpha=0.5, label='Dados de Treinamento')
    plt.plot(x_test_np, y_test_true_np, label=f'{name}(x) Verdadeira', color='blue', linewidth=2)
    plt.plot(x_test_np, y_test_pred_np, label=f'{name}(x) Prevista (PyTorch)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Aproximação PyTorch da Função {name}(x)')
    plt.xlabel('x (radianos)')
    plt.ylabel(f'{name}(x)')
    plt.legend()
    plt.grid(True)
    if name == 'tan':
        plt.ylim(-10, 10)

plt.tight_layout()
plt.suptitle("Interpolação com PyTorch: Seno, Cosseno, Tangente", fontsize=16, y=1.02)
plt.show()