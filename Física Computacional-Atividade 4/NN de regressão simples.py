import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ======== Gerar dados sintéticos ========
T0 = 90
T_amb = 25
r = 0.005

def T_analytical(t):
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

# Dados de entrada (com ruído)
np.random.seed(42)
t_data = np.linspace(0, 200, 10)
T_clean = T_analytical(t_data)
noise = np.random.normal(0.0, 0.5, size=t_data.shape)
T_noisy = T_clean + noise

# Converter para tensores PyTorch
t_train = torch.tensor(t_data, dtype=torch.float32).unsqueeze(1) / 1000  # normalizar entrada [0,1]
T_train = torch.tensor(T_noisy, dtype=torch.float32).unsqueeze(1)

# ======== Definir Rede Neural Simples ========
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# ======== Treinamento ========
for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    pred = model(t_train)
    loss = loss_fn(pred, T_train)
    loss.backward()
    optimizer.step()

# ======== Avaliação ========
model.eval()
t_test = np.linspace(0, 1000, 300)
t_test_norm = torch.tensor(t_test / 1000, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    T_pred = model(t_test_norm).squeeze().numpy()

T_true = T_analytical(t_test)

# ======== Visualização ========
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, 'k--', label='Solução Analítica')
plt.plot(t_test, T_pred, 'b-', label='NN (regressão)')
plt.scatter(t_data, T_noisy, color='red', label='Dados sintéticos', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Ajuste de rede neural vs solução analítica')
plt.grid(True)
plt.legend()
plt.show()
