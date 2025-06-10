import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- Configurações e Parâmetros ---
torch.manual_seed(42)
np.random.seed(42)

T_amb = 25
T0 = 95.0 # Usar float para consistência
r_true = 0.005 # O valor que esperamos descobrir
t_min, t_max = 0.0, 1000.0

# --- Carregar e Preparar Dados ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dados_treinamento.npz')
data = np.load(file_path)
t_data_np = data['t_dados']
T_data_np = data['T_ruidoso']

# Normalizar o tempo para a escala [0, 1]
t_data_norm = torch.tensor((t_data_np - t_min) / (t_max - t_min), dtype=torch.float32).view(-1, 1)
T_data = torch.tensor(T_data_np, dtype=torch.float32).view(-1, 1)

# --- Definição da Rede Neural que aprende N(t) e r ---
class PINNLearnR(nn.Module):
    def __init__(self, initial_r=0.01):
        super().__init__()
        # A parte da rede que aprende a forma da curva
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        # ### PONTO-CHAVE 1: 'r' é um parâmetro treinável ###
        # O otimizador irá ajustar este valor para minimizar a perda.
        self.r = nn.Parameter(torch.tensor([initial_r], dtype=torch.float32))

    def forward(self, t_normalized):
        # A rede retorna a parte N(t) da solução
        return self.net(t_normalized)

# --- Inicialização ---
model = PINNLearnR(initial_r=0.02) # Damos um chute inicial para 'r'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 15000

# Pontos de física, normalizados e com gradiente
t_phys_norm = torch.tensor(((np.linspace(t_min, t_max, 200).reshape(-1, 1)) - t_min) / (t_max - t_min), dtype=torch.float32).requires_grad_(True)

print("Iniciando o treinamento da PINN para descobrir 'r'...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- PONTO-CHAVE 2: Usamos a mesma "Trial Solution" de antes ---
    # T(t) = T0 + t * N(t), o que garante T(0)=T0
    
    # 1. Perda nos Dados
    N_data = model(t_data_norm)
    t_data_unnorm = torch.tensor(t_data_np, dtype=torch.float32).view(-1, 1)
    T_pred_data = T0 + t_data_unnorm * N_data
    loss_data = torch.mean((T_pred_data - T_data)**2)

    # 2. Perda na Física
    N_phys = model(t_phys_norm)
    dN_dt_norm = torch.autograd.grad(N_phys, t_phys_norm, grad_outputs=torch.ones_like(N_phys), create_graph=True)[0]
    dN_dt = dN_dt_norm * (1.0 / (t_max - t_min))
    
    t_phys_unnorm = torch.tensor(np.linspace(t_min, t_max, 200).reshape(-1, 1), dtype=torch.float32)
    dT_dt = N_phys + t_phys_unnorm * dN_dt

    T_pred_phys = T0 + t_phys_unnorm * N_phys
    
    # ### PONTO-CHAVE 3: Usamos o 'r' treinável no resíduo ###
    residual = dT_dt - model.r * (T_amb - T_pred_phys)
    loss_phys = torch.mean(residual**2)
    
    # Perda Total Ponderada
    loss = loss_data + 1.0 * loss_phys

    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_phys.item():.6f} | r: {model.r.item():.5f}")

r_estimado_final = model.r.item()
print("\n--- Resultados Finais ---")
print(f"Valor verdadeiro de r: {r_true}")
print(f"Valor de r descoberto pela PINN: {r_estimado_final:.5f}")

# --- Avaliação e Visualização ---
model.eval()
t_test_np = np.linspace(t_min, t_max, 500).reshape(-1, 1)
t_test_norm = torch.tensor((t_test_np - t_min) / (t_max - t_min), dtype=torch.float32)

with torch.no_grad():
    N_test = model(t_test_norm)
    T_pinn = T0 + t_test_np * N_test.numpy()

T_analitica = T_amb + (T0 - T_amb) * np.exp(-r_true * t_test_np)

plt.figure(figsize=(12, 7))
plt.plot(t_test_np, T_analitica, 'k--', linewidth=2, label=f'Solução Analítica (r={r_true})')
plt.plot(t_test_np, T_pinn, 'purple', linestyle='-', linewidth=2, label=f'PINN (r estimado ≈ {r_estimado_final:.4f})')
plt.scatter(t_data_np, T_data_np, color='blue', zorder=5, s=50, label='Dados de Treinamento')
plt.title('PINN Descobrindo Parâmetro Físico (Método Trial Solution)', fontsize=16)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(T_amb - 5, T0 + 5)
plt.show()