import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dados_treinamento.npz')

# --- Configurações e Reprodutibilidade ---
torch.manual_seed(42)
np.random.seed(42)

# --- Carregar Dados ---
try:
    data = np.load(file_path)
    t_data = torch.tensor(data['t_dados'], dtype=torch.float32).view(-1, 1)
    T_data = torch.tensor(data['T_ruidoso'], dtype=torch.float32).view(-1, 1)
except FileNotFoundError:
    # Mensagem de erro aprimorada
    print(f"Erro: Arquivo '{file_path}' não encontrado.")
    print("Por favor, execute o script '02_gerar_dados_sinteticos.py' primeiro.")
    exit()


# --- Definir a Rede Neural de Regressão ---
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t):
        return self.net(t)

model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# --- Treinamento ---
epochs = 2000
for epoch in range(epochs):
    model.train()
    T_pred = model(t_data)
    loss = loss_fn(T_pred, T_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Avaliação e Extrapolação ---
t_test = torch.linspace(0, 1000, 500).view(-1, 1)
model.eval()
with torch.no_grad():
    T_pred_test = model(t_test)

# --- Solução Analítica para Comparação ---
r, T_amb, T0 = 0.005, 25, 95
T_analitica = T_amb + (T0 - T_amb) * np.exp(-r * t_test.numpy())

# --- Plot ---
plt.figure(figsize=(12, 7))
plt.plot(t_test.numpy(), T_analitica, 'k--', label='Solução Analítica')
plt.plot(t_test.numpy(), T_pred_test.numpy(), 'r-', label='Previsão da NN Simples')
plt.scatter(t_data.numpy(), T_data.numpy(), color='blue', zorder=5, label='Dados de Treinamento')
plt.title('Falha da NN Simples em Extrapolar', fontsize=16)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()