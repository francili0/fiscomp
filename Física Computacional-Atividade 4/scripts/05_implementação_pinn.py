import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- Configurações e Parâmetros ---
torch.manual_seed(42)
np.random.seed(42)

r = 0.005
T_amb = 25
T0 = 95.0 # Usar float para consistência
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

# --- Definição da Rede Neural N(t) ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t_normalized):
        return self.net(t_normalized)

# --- Inicialização ---
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 15000 # AUMENTAMOS AS ÉPOCAS

# Pontos de física, normalizados e com gradiente
t_phys_norm = torch.tensor(((np.linspace(t_min, t_max, 200).reshape(-1, 1)) - t_min) / (t_max - t_min), dtype=torch.float32).requires_grad_(True)

print("Iniciando o treinamento final (com ajustes finos)...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Perda nos Dados (loss_data)
    N_data = model(t_data_norm)
    t_data_unnorm = torch.tensor(t_data_np, dtype=torch.float32).view(-1, 1)
    T_pred_data = T0 + t_data_unnorm * N_data
    loss_data = torch.mean((T_pred_data - T_data)**2)

    # 2. Perda na Física (loss_phys)
    N_phys = model(t_phys_norm)
    dN_dt_norm = torch.autograd.grad(N_phys, t_phys_norm, grad_outputs=torch.ones_like(N_phys), create_graph=True)[0]
    dN_dt = dN_dt_norm * (1.0 / (t_max - t_min))
    
    t_phys_unnorm = torch.tensor(np.linspace(t_min, t_max, 200).reshape(-1, 1), dtype=torch.float32)
    dT_dt = N_phys + t_phys_unnorm * dN_dt

    T_pred_phys = T0 + t_phys_unnorm * N_phys
    residual = dT_dt - r * (T_amb - T_pred_phys)
    loss_phys = torch.mean(residual**2)
    
    # Perda Total Ponderada - PESO DA FÍSICA AUMENTADO
    loss = loss_data + 1.0 * loss_phys

    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_phys.item():.6f}")

print("Treinamento concluído.")

# --- Avaliação e Visualização ---
model.eval()
t_test_np = np.linspace(t_min, t_max, 500).reshape(-1, 1)
t_test_norm = torch.tensor((t_test_np - t_min) / (t_max - t_min), dtype=torch.float32)

with torch.no_grad():
    N_test = model(t_test_norm)
    T_pinn = T0 + t_test_np * N_test.numpy()

T_analitica = T_amb + (T0 - T_amb) * np.exp(-r * t_test_np)

plt.figure(figsize=(12, 7))
plt.plot(t_test_np, T_analitica, 'k--', linewidth=2, label='Solução Analítica')
plt.plot(t_test_np, T_pinn, 'g-', linewidth=2, label='Previsão da PINN (Final)')
plt.scatter(t_data_np, T_data_np, color='blue', zorder=5, s=50, label='Dados de Treinamento')
plt.title('PINN com "Trial Solution" Extrapola Corretamente', fontsize=16)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(T_amb - 5, T0 + 5)
plt.show()