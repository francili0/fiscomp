import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ======== 1. Parâmetros físicos ========
T0 = 90            # Temperatura inicial do café (°C)
T_amb = 25         # Temperatura ambiente (°C)
r = 0.005          # Taxa de resfriamento (1/s)

# ======== 2. Função analítica da solução da EDO ========
def T_analytical(t):
    """Solução analítica da EDO de resfriamento de Newton"""
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

# ======== 3. Geração de dados sintéticos com ruído ========
np.random.seed(42)                         # Para reprodutibilidade
t_data = np.linspace(0, 200, 10)           # 10 pontos entre 0 e 200 s
T_clean = T_analytical(t_data)             # Temperatura exata (sem ruído)
noise = np.random.normal(0.0, 0.5, t_data.shape)  # Ruído gaussiano
T_noisy = T_clean + noise                  # Dados com ruído

# ======== 4. Conversão para tensores PyTorch ========
t_train = torch.tensor(t_data, dtype=torch.float32).unsqueeze(1) / 1000  # Normalizar tempo [0, 1]
T_train = torch.tensor(T_noisy, dtype=torch.float32).unsqueeze(1)        # Temperaturas com ruído

# ======== 5. Definição da rede neural (MLP simples) ========
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),   # Camada oculta 1 com 32 neurônios
            nn.Tanh(),
            nn.Linear(32, 32),  # Camada oculta 2 com 32 neurônios
            nn.Tanh(),
            nn.Linear(32, 1)    # Saída: 1 neurônio (temperatura)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()                                     # Instanciar modelo
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Otimizador
loss_fn = nn.MSELoss()                                 # Função de perda MSE

# ======== 6. Treinamento da rede neural ========
for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    pred = model(t_train)              # Previsão da rede
    loss = loss_fn(pred, T_train)     # Comparação com dados reais
    loss.backward()
    optimizer.step()

# ======== 7. Avaliação do modelo treinado ========
model.eval()
t_test = np.linspace(0, 1000, 300)                         # Tempo para teste (0–1000 s)
t_test_norm = torch.tensor(t_test / 1000, dtype=torch.float32).unsqueeze(1)  # Normalizar tempo
with torch.no_grad():
    T_pred = model(t_test_norm).squeeze().numpy()          # Previsão da rede
T_true = T_analytical(t_test)                              # Solução analítica

# ======== 8. Visualização dos resultados ========
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_true, 'k--', label='Solução Analítica')
plt.plot(t_test, T_pred, 'b-', label='Rede Neural (regressão)')
plt.scatter(t_data, T_noisy, color='red', label='Dados sintéticos', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Ajuste de rede neural vs solução analítica')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
