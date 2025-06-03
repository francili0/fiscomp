import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reprodutibilidade
np.random.seed(0)
torch.manual_seed(0)

# ----------------------------
# GERAR DADOS SINTÉTICOS
# ----------------------------
T_amb = 25
T0 = 90
r_true = 0.005  # Valor real usado para gerar os dados

# Dados com ruído em t ∈ [0, 200]
t_data = np.linspace(0, 200, 10).reshape(-1, 1)
T_exact = (T0 - T_amb) * np.exp(-r_true * t_data) + T_amb
noise = np.random.normal(0, 0.5, size=T_exact.shape)
T_obs = T_exact + noise

# Normalização do tempo
t_mean, t_std = t_data.mean(), t_data.std()
t_data_norm = (t_data - t_mean) / t_std

# Tensors
t_data_tensor = torch.tensor(t_data_norm, dtype=torch.float32, requires_grad=True)
T_obs_tensor = torch.tensor(T_obs, dtype=torch.float32)

# ----------------------------
# DEFINIÇÃO DA PINN COM r TREINÁVEL
# ----------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Parâmetro r como valor treinável (inicializa com chute)
        self.r = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

# Inicializa modelo e otimizador
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# TREINAMENTO
# ----------------------------
lambda_data = 1.0
lambda_phys = 1.0

for epoch in range(5000):
    optimizer.zero_grad()

    # --- Perda de dados ---
    T_pred = model(t_data_tensor)
    loss_data = torch.mean((T_pred - T_obs_tensor)**2)

    # --- Perda física ---
    t_phys = torch.linspace(0, 1000, 100).reshape(-1, 1)
    t_phys_norm = (t_phys - t_mean) / t_std
    t_phys_tensor = torch.tensor(t_phys_norm, dtype=torch.float32, requires_grad=True)

    T_phys = model(t_phys_tensor)
    dT_dt = torch.autograd.grad(T_phys, t_phys_tensor, torch.ones_like(T_phys),
                                create_graph=True)[0]
    eq_residual = dT_dt - model.r * (T_amb - T_phys)
    loss_phys = torch.mean(eq_residual**2)

    # --- Perda total ---
    loss = lambda_data * loss_data + lambda_phys * loss_phys
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"[{epoch}] Loss: {loss.item():.4f} | r: {model.r.item():.5f}")

# ----------------------------
# AVALIAÇÃO
# ----------------------------
# Tempo normal e normalizado
t_test = np.linspace(0, 1000, 500).reshape(-1, 1)
t_test_norm = (t_test - t_mean) / t_std
t_test_tensor = torch.tensor(t_test_norm, dtype=torch.float32)

with torch.no_grad():
    T_pinn = model(t_test_tensor).numpy()

# Solução analítica com r verdadeiro
T_true = (T0 - T_amb) * np.exp(-r_true * t_test) + T_amb

# ----------------------------
# VISUALIZAÇÃO
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, 'k--', label='Solução Analítica (r verdadeiro)')
plt.plot(t_test, T_pinn, 'r-', label='PINN (r treinado)')
plt.scatter(t_data, T_obs, color='blue', label='Dados sintéticos', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title(f'PINN aprendendo r ≈ {model.r.item():.5f}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
