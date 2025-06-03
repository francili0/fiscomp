import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reprodutibilidade
np.random.seed(42)
torch.manual_seed(42)

# --- Dados sintéticos ---
T_amb = 25
T0 = 90
r = 0.005

# 10 pontos entre 0 e 200s
t_train = np.linspace(0, 200, 10).reshape(-1, 1)
T_exact = (T0 - T_amb) * np.exp(-r * t_train) + T_amb
noise = np.random.normal(0, 0.5, size=T_exact.shape)
T_train = T_exact + noise

# Normalizar tempo (opcional mas ajuda)
t_mean, t_std = t_train.mean(), t_train.std()
t_train_norm = (t_train - t_mean) / t_std

# --- Conversão para Tensor ---
t_tensor = torch.tensor(t_train_norm, dtype=torch.float32)
T_tensor = torch.tensor(T_train, dtype=torch.float32)

# --- Rede neural simples ---
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# --- Treinamento ---
for epoch in range(2000):
    model.train()
    pred = model(t_tensor)
    loss = loss_fn(pred, T_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Previsão em t = 0 a 1000 ---
t_test = np.linspace(0, 1000, 500).reshape(-1, 1)
t_test_norm = (t_test - t_mean) / t_std
t_test_tensor = torch.tensor(t_test_norm, dtype=torch.float32)
with torch.no_grad():
    T_pred = model(t_test_tensor).numpy()

# Solução analítica para comparação
T_analytical = (T0 - T_amb) * np.exp(-r * t_test) + T_amb

# --- Plotagem ---
plt.figure(figsize=(10, 5))
plt.plot(t_test, T_analytical, 'k--', label='Solução analítica')
plt.plot(t_test, T_pred, 'r-', label='Rede Neural')
plt.scatter(t_train, T_train, color='blue', label='Dados sintéticos (ruído)', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Regressão com NN vs. Solução Analítica (Extrapolação até 1000s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
