import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- Construir o caminho do arquivo de forma robusta ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dados_treinamento.npz')

# --- Configurações e Reprodutibilidade ---
torch.manual_seed(42)
np.random.seed(42)

# --- Parâmetros Físicos e de Domínio ---
r = 0.005
T_amb = 25
T0 = 95

### PONTO CRÍTICO 1: Definir o domínio de tempo completo ###
# O problema inteiro acontece no intervalo de 0 a 1000 segundos.
t_min, t_max = 0.0, 1000.0

# --- Carregar Dados de Treinamento ---
try:
    data = np.load(file_path)
    # Carregamos os dados em formato numpy primeiro
    t_data_np = data['t_dados']
    T_data_np = data['T_ruidoso']
except FileNotFoundError:
    print(f"Erro: Arquivo '{file_path}' não encontrado.")
    print("Por favor, execute o script '02_gerar_dados_sinteticos.py' primeiro.")
    exit()

# --- Normalização e Conversão para Tensores ---
### PONTO CRÍTICO 2: Normalizar TODOS os tempos para a escala [0, 1] ###
# Normalizamos o tempo dos dados de treinamento
t_data = torch.tensor((t_data_np - t_min) / (t_max - t_min), dtype=torch.float32).view(-1, 1)
T_data = torch.tensor(T_data_np, dtype=torch.float32).view(-1, 1)

# --- Definição da PINN (a arquitetura não muda) ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # A rede recebe uma entrada normalizada [0, 1] e retorna a Temperatura
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t_normalized):
        return self.net(t_normalized)

# --- Inicialização do Modelo e Otimizador ---
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Treinamento da PINN ---
epochs = 8000 # Aumentei um pouco as épocas para garantir a convergência

# Criamos os pontos de física no domínio original
t_phys_np = np.linspace(t_min, t_max, 100).reshape(-1, 1)
# E normalizamos para alimentar a rede
t_phys = torch.tensor((t_phys_np - t_min) / (t_max - t_min), dtype=torch.float32).requires_grad_(True)


for epoch in range(epochs):
    optimizer.zero_grad()

    # Perda nos dados (a rede recebe t_data normalizado)
    T_pred_data = model(t_data)
    loss_data = torch.mean((T_pred_data - T_data)**2)

    # Perda na física (a rede recebe t_phys normalizado)
    T_pred_phys = model(t_phys)

    # ### PONTO CRÍTICO 3: A derivada precisa considerar a normalização ###
    # dT/dt = (dT/dt_norm) * (dt_norm/dt)
    # dt_norm/dt = 1 / (t_max - t_min)
    dT_dt_norm = torch.autograd.grad(
        T_pred_phys, t_phys,
        grad_outputs=torch.ones_like(T_pred_phys),
        create_graph=True
    )[0]
    # Aplicamos a Regra da Cadeia
    dT_dt = dT_dt_norm * (1.0 / (t_max - t_min))

    residual = dT_dt - r * (T_amb - T_pred_phys)
    loss_phys = torch.mean(residual**2)
    
    loss = loss_data + 0.1 * loss_phys # Ajustei o peso da física para 0.1, que ajuda a estabilizar

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f} | Loss Data: {loss_data.item():.4f} | Loss Phys: {loss_phys.item():.4f}")


# --- Avaliação e Visualização ---
model.eval()
# Criamos os tempos de teste no domínio original
t_test_np = np.linspace(t_min, t_max, 500).reshape(-1, 1)
# Normalizamos para a previsão
t_test = torch.tensor((t_test_np - t_min) / (t_max - t_min), dtype=torch.float32)

with torch.no_grad():
    T_pinn = model(t_test).numpy()

# Solução analítica para comparação (usa o tempo original)
T_analitica = T_amb + (T0 - T_amb) * np.exp(-r * t_test_np)

# Plot do resultado final (o eixo X usa o tempo original)
plt.figure(figsize=(12, 7))
plt.plot(t_test_np, T_analitica, 'k--', linewidth=2, label='Solução Analítica')
plt.plot(t_test_np, T_pinn, 'g-', linewidth=2, label='Previsão da PINN (Corrigido)')
plt.scatter(t_data_np, T_data_np, color='blue', zorder=5, s=50, label='Dados de Treinamento')
plt.title('PINN com Física Conhecida Extrapola Corretamente', fontsize=16)
plt.xlabel('Tempo (s)', fontsize=14)
plt.ylabel('Temperatura (°C)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()