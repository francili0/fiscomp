import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------
# Dados sintéticos
# -----------------------
T_amb = 25
T0 = 90
r_true = 0.005
sigma = 0.5
n_points = 10

def T_analitica(t, r=r_true):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

t_data = np.linspace(0, 200, n_points)
T_clean = T_analitica(t_data)
T_noisy = T_clean + np.random.normal(0, sigma, size=n_points)

t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
T_data_tensor = torch.tensor(T_noisy, dtype=torch.float32).view(-1, 1)

# -----------------------
# PINN com r como parâmetro treinável
# -----------------------
class PINN_learn_r(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.r = nn.Parameter(torch.tensor(0.01))  # chute inicial para r

    def forward(self, t):
        return self.net(t)

    def physics_residual(self, t):
        t.requires_grad_(True)
        T = self.forward(t)
        dTdt = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T),
                                   create_graph=True, retain_graph=True)[0]
        return dTdt + self.r * (T - T_amb)

# -----------------------
# Treinamento
# -----------------------
model = PINN_learn_r()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

t_phys = torch.linspace(0, 1000, 100).view(-1, 1)

for epoch in range(7000):
    model.train()
    pred_T = model(t_data_tensor)
    loss_data = loss_fn(pred_T, T_data_tensor)
    
    res = model.physics_residual(t_phys)
    loss_phys = torch.mean(res**2)
    
    loss = loss_data + loss_phys

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | r estimado: {model.r.item():.6f}")

# -----------------------
# Avaliação
# -----------------------
t_test = np.linspace(0, 1000, 500)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)

model.eval()
with torch.no_grad():
    T_pinn = model(t_test_tensor).numpy()
    r_est = model.r.item()

T_true = T_analitica(t_test, r=r_true)

# -----------------------
# Gráfico comparativo
# -----------------------
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, label=f'Solução Analítica (r={r_true})', lw=2)
plt.plot(t_test, T_pinn, label=f'PINN aprendendo r (r ≈ {r_est:.5f})', lw=2)
plt.scatter(t_data, T_noisy, color='red', label='Dados com Ruído', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('PINN aprendendo a taxa de resfriamento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
