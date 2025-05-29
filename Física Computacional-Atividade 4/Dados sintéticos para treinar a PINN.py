import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da equação
T0 = 90
T_amb = 25
r = 0.005

# Solução analítica
def T_analytical(t):
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

# Gerar 10 pontos entre 0 e 200 s
np.random.seed(42)  # reprodutibilidade
t_data = np.linspace(0, 400, 10)
T_clean = T_analytical(t_data)

# Adicionar ruído gaussiano (média 0, desvio padrão 0.5)
noise = np.random.normal(loc=0.0, scale=0.5, size=t_data.shape)
T_noisy = T_clean + noise

# Plot para visualização
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, 1000, 100), T_analytical(np.linspace(0, 1000, 100)), 'k--', label='Solução analítica')
plt.scatter(t_data, T_noisy, color='red', label='Dados sintéticos com ruído', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Dados sintéticos para treinamento da PINN')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar os dados gerados
for i in range(len(t_data)):
    print(f"t = {t_data[i]:6.2f} s | T = {T_noisy[i]:6.2f} °C")
