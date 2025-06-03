import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
T_amb = 25
T0 = 90
r = 0.005

# Gerar 10 pontos no intervalo [0, 2000]
t_data = np.linspace(0, 2000, 10)
T_true = (T0 - T_amb) * np.exp(-r * t_data) + T_amb

# Adicionar ruído gaussiano (média 0, desvio padrão 0.5)
noise = np.random.normal(loc=0.0, scale=0.5, size=t_data.shape)
T_noisy = T_true + noise

# Visualização
plt.figure(figsize=(8, 5))
plt.plot(t_data, T_true, 'k--', label='Solução analítica (sem ruído)')
plt.scatter(t_data, T_noisy, color='red', label='Dados sintéticos (com ruído)', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Dados sintéticos para treinamento de PINN')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
