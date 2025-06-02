import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------
# Dados sintéticos
# -----------------------
T_amb = 25
T0 = 90
r = 0.005
sigma = 0.5
n_points = 10

def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

t_data = np.linspace(0, 200, n_points)
T_clean = T_analitica(t_data)
T_noisy = T_clean + np.random.normal(0, sigma, size=n_points)

t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
T_data_tensor = torch.tensor(T_noisy, dtype=torch.float32).view(-1, 1)

# -----------------------
# PINN
# -----------------------
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

    def forward(self, t):
        return self.net(t)

def physics_residual(model, t_phys):
    t_phys.requires_grad_(True)
    T = model(t_phys)
    dTdt = torch.autograd.grad(T, t_phys, grad_outputs=torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    return dTdt + r * (T - T_amb)

# Instanciar PINN
pinn = PINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Pontos físicos (mais densos para ajudar o aprendizado)
t_phys = torch.linspace(0, 1000, 100).view(-1, 1)

# Treinamento
for epoch in range(5000):
    pinn.train()

    # Previsão e perda de dados
    T_pred = pinn(t_data_tensor)
    loss_data = loss_fn(T_pred, T_data_tensor)

    # Resíduo da equação diferencial
    res = physics_residual(pinn, t_phys)
    loss_phys = torch.mean(res**2)

    # Perda total: dados + física
    loss = loss_data + loss_phys

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f}  Data Loss = {loss_data.item():.5f}  Phys Loss = {loss_phys.item():.5f}")

# -----------------------
# Avaliação
# -----------------------
t_test = np.linspace(0, 1000, 500)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)

pinn.eval()
with torch.no_grad():
    T_pinn = pinn(t_test_tensor).numpy()

T_true = T_analitica(t_test)

# -----------------------
# Comparar com rede neural simples
# (Repetir aqui se ainda não estiver em sessão ou importar resultados anteriores)
# -----------------------
# Exemplo: comparar T_pred do modelo anterior se salvo

# -----------------------
# Gráfico comparativo
# -----------------------
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, label='Solução Analítica', lw=2)
plt.plot(t_test, T_pinn, label='PINN', lw=2)
plt.scatter(t_data, T_noisy, color='red', label='Dados com Ruído', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('PINN vs Solução Analítica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
