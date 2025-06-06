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

# --- GPU CONFIGURATION ---
# 1. Definir o dispositivo (GPU se disponível, senão CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
# --- END GPU CONFIGURATION ---

# Definição da Rede Neural (MLP_Derivadas)
class MLP_Derivadas(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_Derivadas, self).__init__()
        camadas = []
        current_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            camadas.append(nn.Linear(current_size, hidden_size))
            # camadas.append(nn.BatchNorm1d(hidden_size)) # Descomente para testar BatchNorm
            camadas.append(nn.Tanh())
            current_size = hidden_size
        camadas.append(nn.Linear(current_size, output_size))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

# Função para gerar dados de polinômios e suas derivadas
def generate_polynomial_data_pytorch(nx, qtde_funcoes, p_max):
    x_base_np = np.linspace(-1, 1, nx, dtype=np.float32)
    lista_y_np = []
    lista_dy_np = []
    for _ in range(qtde_funcoes):
        p = np.random.randint(0, p_max + 1)
        coeffs = np.random.randn(p + 1).astype(np.float32)
        polinomio_vals_np = np.polyval(coeffs, x_base_np)
        derivada_coeffs = np.polyder(coeffs)
        if derivada_coeffs.size == 0:
            derivada_vals_np = np.zeros_like(polinomio_vals_np)
        else:
            derivada_vals_np = np.polyval(derivada_coeffs, x_base_np)

        max_abs_polinomio = np.max(np.abs(polinomio_vals_np))
        if max_abs_polinomio < 1e-6: max_abs_polinomio = 1.0
        polinomio_vals_norm_np = polinomio_vals_np / max_abs_polinomio
        derivada_vals_norm_np = derivada_vals_np / max_abs_polinomio

        noise_level = 0.01 # Nível de ruído
        noise_y = noise_level * np.random.randn(nx).astype(np.float32)
        noise_dy = noise_level * np.random.randn(nx).astype(np.float32)
        lista_y_np.append(polinomio_vals_norm_np + noise_y)
        lista_dy_np.append(derivada_vals_norm_np + noise_dy)
    y_out_np = np.array(lista_y_np, dtype=np.float32)
    dy_out_np = np.array(lista_dy_np, dtype=np.float32)
    return y_out_np, dy_out_np, x_base_np

# Parâmetros
num_pontos_por_funcao = 50
num_funcoes_treino = 30000
grau_max_polinomio = 10
learning_rate = 0.001
epochs = 20000
weight_decay_lambda = 1e-5
max_grad_norm = 1.0

# 1. Geração de Dados
print(f"Gerando {num_funcoes_treino} dados de treinamento (polinômios)...")
Y_data_np, dY_data_np, x_grid_np = generate_polynomial_data_pytorch(
    nx=num_pontos_por_funcao,
    qtde_funcoes=num_funcoes_treino,
    p_max=grau_max_polinomio
)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    Y_data_np, dY_data_np, test_size=0.2, random_state=42
)

# Converte para tensores PyTorch (inicialmente na CPU)
X_train_torch_cpu = torch.from_numpy(X_train_np)
y_train_torch_cpu = torch.from_numpy(y_train_np)
X_test_torch_cpu = torch.from_numpy(X_test_np)
y_test_torch_cpu = torch.from_numpy(y_test_np)

# 2. Definição do Modelo, Função de Perda e Otimizador
input_size = num_pontos_por_funcao
hidden_sizes_wider = [128, 256, 128]
output_size = num_pontos_por_funcao

model_deriv = MLP_Derivadas(input_size, hidden_sizes_wider, output_size)
# --- GPU ACTION: Mover o modelo para o dispositivo ---
model_deriv.to(device)
# --- END GPU ACTION ---

criterion_deriv = nn.MSELoss() # A função de perda não precisa ser movida para a GPU
optimizer_deriv = optim.Adam(model_deriv.parameters(), lr=learning_rate, weight_decay=weight_decay_lambda)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_deriv, mode='min', patience=10, factor=0.5, min_lr=1e-7)

train_losses_epoch = []
test_losses_epoch = []
learning_rates_epoch = []
evaluation_epochs = []

# 3. Loop de Treinamento
print("Iniciando o treinamento da rede para aprender derivadas (PyTorch)...")
for epoch in range(epochs):
    model_deriv.train()
    optimizer_deriv.zero_grad()

    # --- GPU ACTION: Mover dados de treinamento para o dispositivo ---
    inputs = X_train_torch_cpu.to(device)
    labels = y_train_torch_cpu.to(device)
    # --- END GPU ACTION ---

    outputs = model_deriv(inputs)
    loss = criterion_deriv(outputs, labels) # 'outputs' e 'labels' estão no mesmo dispositivo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_deriv.parameters(), max_norm=max_grad_norm)
    optimizer_deriv.step()

    if (epoch + 1) % 100 == 0:
        model_deriv.eval()
        with torch.no_grad():
            # --- GPU ACTION: Mover dados de teste para o dispositivo ---
            test_inputs = X_test_torch_cpu.to(device)
            test_labels = y_test_torch_cpu.to(device)
            # --- END GPU ACTION ---
            test_outputs = model_deriv(test_inputs)
            test_loss = criterion_deriv(test_outputs, test_labels)
        
        current_lr = optimizer_deriv.param_groups[0]['lr']
        # test_loss é um tensor na GPU, .item() o traz para CPU como um número Python
        print(f'Época [{epoch+1}/{epochs}], Perda Treino: {loss.item():.6f}, Perda Teste (Polinômios): {test_loss.item():.6f}, LR: {current_lr:.7f}')
        
        train_losses_epoch.append(loss.item())
        test_losses_epoch.append(test_loss.item())
        learning_rates_epoch.append(current_lr)
        evaluation_epochs.append(epoch + 1)

        scheduler.step(test_loss.item()) # O scheduler espera um escalar Python
        model_deriv.train()

print("Treinamento concluído.")

# Avaliação final no conjunto de teste (polinômios)
model_deriv.eval()
with torch.no_grad():
    # --- GPU ACTION: Mover dados de teste para o dispositivo ---
    test_inputs_final = X_test_torch_cpu.to(device)
    # --- END GPU ACTION ---
    y_pred_test_polinomios_torch = model_deriv(test_inputs_final)

# --- GPU ACTION: Mover resultado para CPU antes de converter para NumPy ---
y_pred_test_polinomios_np = y_pred_test_polinomios_torch.cpu().numpy()
# --- END GPU ACTION ---
mse_polinomios = mean_squared_error(y_test_np, y_pred_test_polinomios_np) # y_test_np já está na CPU
print(f"\nPyTorch - Mean Squared Error (Erro Quadrático Médio) FINAL nos polinômios de teste: {mse_polinomios:.6f}")

# Plotagem das Curvas de Perda
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(evaluation_epochs, train_losses_epoch, label='Perda de Treinamento')
plt.plot(evaluation_epochs, test_losses_epoch, label='Perda de Teste (Polinômios)')
plt.xlabel('Épocas')
plt.ylabel('MSE Loss')
plt.title('Curvas de Perda Durante o Treinamento')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(evaluation_epochs, learning_rates_epoch, label='Taxa de Aprendizado (LR)')
plt.xlabel('Épocas')
plt.ylabel('Learning Rate')
plt.title('Taxa de Aprendizado Durante o Treinamento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Teste com funções conhecidas (generalização) ---
plt.figure(figsize=(15, 5))
x_plot = x_grid_np

def normalizar_funcao_para_teste_pytorch(func_vals_np):
    max_abs_func = np.max(np.abs(func_vals_np))
    if max_abs_func < 1e-6: max_abs_func = 1.0
    return func_vals_np / max_abs_func, max_abs_func

# Teste 1: seno
plt.subplot(1, 3, 1)
func_y_sin_np = np.sin(np.pi * x_plot).astype(np.float32)
func_dy_sin_real_np = np.pi * np.cos(np.pi * x_plot).astype(np.float32)
func_y_sin_norm_np, max_abs_sin = normalizar_funcao_para_teste_pytorch(func_y_sin_np)
# --- GPU ACTION: Mover tensor de entrada para o dispositivo ---
func_y_sin_norm_torch = torch.from_numpy(func_y_sin_norm_np.reshape(1, -1)).to(device)
# --- END GPU ACTION ---
model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_sin_norm_torch)
# --- GPU ACTION: Mover resultado para CPU antes de converter para NumPy ---
predicted_derivative_norm_np = predicted_derivative_norm_torch.cpu().numpy().flatten()
# --- END GPU ACTION ---
predicted_derivative_sin_np = predicted_derivative_norm_np * max_abs_sin
plt.plot(x_plot, func_y_sin_np, label='Entrada: sin(πx)', color='black')
plt.plot(x_plot, func_dy_sin_real_np, label='Derivada Real (πcos(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_sin_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: sin(πx)")
plt.ylim(-np.pi * 1.2, np.pi * 1.2)
plt.legend()
plt.grid(True)

# Teste 2: cosseno
plt.subplot(1, 3, 2)
func_y_cos_np = np.cos(np.pi * x_plot).astype(np.float32)
func_dy_cos_real_np = -np.pi * np.sin(np.pi * x_plot).astype(np.float32)
func_y_cos_norm_np, max_abs_cos = normalizar_funcao_para_teste_pytorch(func_y_cos_np)
# --- GPU ACTION: Mover tensor de entrada para o dispositivo ---
func_y_cos_norm_torch = torch.from_numpy(func_y_cos_norm_np.reshape(1, -1)).to(device)
# --- END GPU ACTION ---
model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_cos_norm_torch)
# --- GPU ACTION: Mover resultado para CPU antes de converter para NumPy ---
predicted_derivative_norm_np = predicted_derivative_norm_torch.cpu().numpy().flatten()
# --- END GPU ACTION ---
predicted_derivative_cos_np = predicted_derivative_norm_np * max_abs_cos
plt.plot(x_plot, func_y_cos_np, label='Entrada: cos(πx)', color='black')
plt.plot(x_plot, func_dy_cos_real_np, label='Derivada Real (-πsin(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_cos_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: cos(πx)")
plt.ylim(-np.pi * 1.2, np.pi * 1.2)
plt.legend()
plt.grid(True)

# Teste 3: x²
plt.subplot(1, 3, 3)
func_y_x2_np = (x_plot**2).astype(np.float32)
func_dy_x2_real_np = (2 * x_plot).astype(np.float32)
func_y_x2_norm_np, max_abs_x2 = normalizar_funcao_para_teste_pytorch(func_y_x2_np)
# --- GPU ACTION: Mover tensor de entrada para o dispositivo ---
func_y_x2_norm_torch = torch.from_numpy(func_y_x2_norm_np.reshape(1, -1)).to(device)
# --- END GPU ACTION ---
model_deriv.eval()
with torch.no_grad():
    predicted_derivative_norm_torch = model_deriv(func_y_x2_norm_torch)
# --- GPU ACTION: Mover resultado para CPU antes de converter para NumPy ---
predicted_derivative_norm_np = predicted_derivative_norm_torch.cpu().numpy().flatten()
# --- END GPU ACTION ---
predicted_derivative_x2_np = predicted_derivative_norm_np * max_abs_x2
plt.plot(x_plot, func_y_x2_np, label='Entrada: x²', color='black')
plt.plot(x_plot, func_dy_x2_real_np, label='Derivada Real (2x)', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_x2_np, label='Derivada Prevista (PyTorch)', color='red', linestyle='dashed', linewidth=2)
plt.title("PyTorch - Teste Derivada: x²")
plt.ylim(np.min(func_dy_x2_real_np) * 1.2 - 0.1, np.max(func_dy_x2_real_np) * 1.2 + 0.1)
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Teste da Rede Neural PyTorch para Calcular Derivadas (Com Melhorias e Uso de GPU)", fontsize=16)
plt.show()