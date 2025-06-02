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
n_points = 100

def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

t_data = np.linspace(0, 200, n_points)
T_clean = T_analitica(t_data)
T_noisy = T_clean + np.random.normal(0, sigma, size=n_points)

# Normalização simples para melhorar treinamento
t_min, t_max = 0, 1000
t_data_norm = t_data / t_max

# Convertendo para tensores PyTorch
t_train = torch.tensor(t_data_norm, dtype=torch.float32).view(-1, 1)
T_train = torch.tensor(T_noisy, dtype=torch.float32).view(-1, 1)

# -----------------------
# Rede Neural
# -----------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Treinamento
n_epochs = 3000
for epoch in range(n_epochs):
    model.train()
    pred = model(t_train)
    loss = loss_fn(pred, T_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------
# Avaliação
# -----------------------
t_test = np.linspace(0, 1000, 500)
t_test_norm = torch.tensor(t_test / t_max, dtype=torch.float32).view(-1, 1)

model.eval()
with torch.no_grad():
    T_pred = model(t_test_norm).numpy()

T_true = T_analitica(t_test)

# -----------------------
# Gráfico
# -----------------------
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, label='Solução Analítica', lw=2)
plt.plot(t_test, T_pred, label='Rede Neural', lw=2)
plt.scatter(t_data, T_noisy, color='red', label='Dados com ruído', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Extrapolação com Rede Neural vs Solução Analítica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
