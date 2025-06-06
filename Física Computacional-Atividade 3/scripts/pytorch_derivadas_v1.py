import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configurações
np.random.seed(42)
torch.manual_seed(42)

# Definição da Rede Neural
class MLP_Derivadas(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_Derivadas, self).__init__()
        camadas = []
        # Camada de entrada para a primeira camada oculta
        current_size = input_size
        for hidden_size in hidden_sizes:
            camadas.append(nn.Linear(current_size, hidden_size))
            camadas.append(nn.Tanh())
            current_size = hidden_size
        # Camada da última camada oculta para a saída
        camadas.append(nn.Linear(current_size, output_size))
        # Não há ativação na camada de saída para problemas de regressão
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

# Função para gerar dados de polinômios e suas derivadas (adaptada do script scikit-learn)
def generate_polynomial_data_pytorch(nx, qtde_funcoes, p_max):
    x_base_np = np.linspace(-1, 1, nx, dtype=np.float32) # Usar float32
    
    lista_y_np = []
    lista_dy_np = []
    
    for _ in range(qtde_funcoes):
        p = np.random.randint(0, p_max + 1)
        coeffs = np.random.randn(p + 1).astype(np.float32)

        polinomio_vals_np = np.polyval(coeffs, x_base_np)
        derivada_coeffs = np.polyder(coeffs)
        if derivada_coeffs.size == 0: # Derivada de constante é 0
            derivada_vals_np = np.zeros_like(polinomio_vals_np)
        else:
            derivada_vals_np = np.polyval(derivada_coeffs, x_base_np)

        max_abs_polinomio = np.max(np.abs(polinomio_vals_np))
        if max_abs_polinomio < 1e-6:
            max_abs_polinomio = 1.0

        polinomio_vals_norm_np = polinomio_vals_np / max_abs_polinomio
        derivada_vals_norm_np = derivada_vals_np / max_abs_polinomio # Normalizado pelo mesmo fator

        noise_y = 0.01 * np.random.randn(nx).astype(np.float32)
        noise_dy = 0.01 * np.random.randn(nx).astype(np.float32)

        lista_y_np.append(polinomio_vals_norm_np + noise_y)
        lista_dy_np.append(derivada_vals_norm_np + noise_dy)

    # Converte listas para arrays NumPy antes de empilhar
    y_out_np = np.array(lista_y_np, dtype=np.float32)
    dy_out_np = np.array(lista_dy_np, dtype=np.float32)
    
    return y_out_np, dy_out_np, x_base_np


# Parâmetros para geração de dados e treinamento
num_pontos_por_funcao = 50   # nx: Número de pontos para discretizar cada função
num_funcoes_treino = 10000 # qtde: Quantidade de pares (função, derivada)
grau_max_polinomio = 10    # p_max
learning_rate = 0.001
epochs = 2000 # Ajustar conforme necessário. O treinamento pode ser longo.
                # No script scikit-learn, max_iter era 1000, com n_iter_no_change=50.

# 1. Geração de Dados
print("Gerando dados de treinamento (polinômios)...")
Y_data_np, dY_data_np, x_grid_np = generate_polynomial_data_pytorch(
    nx=num_pontos_por_funcao,
    qtde_funcoes=num_funcoes_treino,
    p_max=grau_max_polinomio
)

# Divide os dados em conjuntos de treino e teste
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    Y_data_np, dY_data_np, test_size=0.2, random_state=42
)

# Converte dados para tensores PyTorch
X_train_torch = torch.from_numpy(X_train_np)
y_train_torch = torch.from_numpy(y_train_np)
X_test_torch = torch.from_numpy(X_test_np)
y_test_torch = torch.from_numpy(y_test_np)


# 2. Definição do Modelo, Função de Perda e Otimizador
input_size = num_pontos_por_funcao  # A entrada é o vetor da função discretizada
hidden_sizes = [64, 128, 64]      # Arquitetura diferente, mais neurônios.
                                     # A original era (10,)*10 = 10 camadas de 10 neurônios.
                                     # (10,)*10 é bem profunda, [64, 128, 64] é mais "larga". Testar.
                                     # Vamos usar a arquitetura original para consistência:
hidden_sizes_original = [10] * 10 # 10 camadas ocultas com 10 neurônios cada
output_size = num_pontos_por_funcao # A saída é o vetor da derivada discretizada

model_deriv = MLP_Derivadas(input_size, hidden_sizes_original, output_size)
criterion_deriv = nn.MSELoss()
optimizer_deriv = optim.Adam(model_deriv.parameters(), lr=learning_rate)

# 3. Loop de Treinamento
print("Iniciando o treinamento da rede para aprender derivadas (PyTorch)...")
for epoch in range(epochs):
    model_deriv.train()
    
    # Forward pass
    outputs = model_deriv(X_train_torch)
    loss = criterion_deriv(outputs, y_train_torch)
    
    # Backward pass e otimização
    optimizer_deriv.zero_grad()
    loss.backward()
    optimizer_deriv.step()
    
    if (epoch + 1) % 100 == 0: # Imprime a cada 100 épocas
        # Avaliação no conjunto de teste (polinômios) para acompanhar
        model_deriv.eval()
        with torch.no_grad():
            test_outputs = model_deriv(X_test_torch)
            test_loss = criterion_deriv(test_outputs, y_test_torch)
        print(f'Época [{epoch+1}/{epochs}], Perda Treino: {loss.item():.6f}, Perda Teste (Polinômios): {test_loss.item():.6f}')
        model_deriv.train() # Volta para o modo de treino

print("Treinamento concluído.")

# Avalia o desempenho final no conjunto de teste (polinômios)
model_deriv.eval()
with torch.no_grad():
    y_pred_test_polinomios_torch = model_deriv(X_test_torch)
y_pred_test_polinomios_np = y_pred_test_polinomios_torch.numpy()
mse_polinomios = mean_squared_error(y_test_np, y_pred_test_polinomios_np)
print(f"\nPyTorch - Mean Squared Error (Erro Quadrático Médio) nos polinômios de teste: {mse_polinomios:.6f}")


# --- Teste com funções conhecidas (generalização) ---
plt.figure(figsize=(15, 5))
x_plot = x_grid_np # x_grid_np já é o array de pontos de -1 a 1

def normalizar_funcao_para_teste_pytorch(func_vals_np):
    """Normaliza os valores da função da mesma forma que no treinamento."""
    max_abs_func = np.max(np.abs(func_vals_np))
    if max_abs_func < 1e-6: max_abs_func = 1.0
    return func_vals_np / max_abs_func, max_abs_func

# Teste 1: seno
plt.subplot(1, 3, 1)
func_y_sin_np = np.sin(np.pi * x_plot).astype(np.float32)
func_dy_sin_real_np = np.pi * np.cos(np.pi * x_plot).astype(np.float32)

func_y_sin_norm_np, max_abs_sin = normalizar_funcao_para_teste_pytorch(func_y_sin_np)
func_y_sin_norm_torch = torch.from_numpy(func_y_sin_norm_np.reshape(1, -1)) # (1, num_pontos)

model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_sin_norm_torch)
predicted_derivative_norm_np = predicted_derivative_norm_torch.numpy().flatten()
predicted_derivative_sin_np = predicted_derivative_norm_np * max_abs_sin


plt.plot(x_plot, func_y_sin_np, label='Entrada: sin(πx)', color='black')
plt.plot(x_plot, func_dy_sin_real_np, label='Derivada Real (πcos(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_sin_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: sin(πx)")
plt.ylim(-np.pi - 0.5, np.pi + 0.5)
plt.legend()
plt.grid(True)

# Teste 2: cosseno
plt.subplot(1, 3, 2)
func_y_cos_np = np.cos(np.pi * x_plot).astype(np.float32)
func_dy_cos_real_np = -np.pi * np.sin(np.pi * x_plot).astype(np.float32)

func_y_cos_norm_np, max_abs_cos = normalizar_funcao_para_teste_pytorch(func_y_cos_np)
func_y_cos_norm_torch = torch.from_numpy(func_y_cos_norm_np.reshape(1, -1))

model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_cos_norm_torch)
predicted_derivative_norm_np = predicted_derivative_norm_torch.numpy().flatten()
predicted_derivative_cos_np = predicted_derivative_norm_np * max_abs_cos


plt.plot(x_plot, func_y_cos_np, label='Entrada: cos(πx)', color='black')
plt.plot(x_plot, func_dy_cos_real_np, label='Derivada Real (-πsin(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_cos_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: cos(πx)")
plt.ylim(-np.pi - 0.5, np.pi + 0.5)
plt.legend()
plt.grid(True)

# Teste 3: x²
plt.subplot(1, 3, 3)
func_y_x2_np = (x_plot**2).astype(np.float32)
func_dy_x2_real_np = (2 * x_plot).astype(np.float32)

func_y_x2_norm_np, max_abs_x2 = normalizar_funcao_para_teste_pytorch(func_y_x2_np)
func_y_x2_norm_torch = torch.from_numpy(func_y_x2_norm_np.reshape(1, -1))

model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_x2_norm_torch)
predicted_derivative_norm_np = predicted_derivative_norm_torch.numpy().flatten()
predicted_derivative_x2_np = predicted_derivative_norm_np * max_abs_x2

plt.plot(x_plot, func_y_x2_np, label='Entrada: x²', color='black')
plt.plot(x_plot, func_dy_x2_real_np, label='Derivada Real (2x)', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_x2_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: x²")
plt.ylim(np.min(func_dy_x2_real_np) - 0.5, np.max(func_dy_x2_real_np) + 0.5)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle("Teste da Rede Neural PyTorch para Calcular Derivadas", fontsize=16, y=1.02)
plt.show()