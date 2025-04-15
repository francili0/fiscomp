import numpy as np
import matplotlib.pyplot as plt

# Constantes
g = 9.8  # aceleração da gravidade (m/s²)

# Dados iniciais
v0 = 10.0           # velocidade inicial (m/s)
angulo = 90         # ângulo de lançamento (graus)
angulo_rad = np.radians(angulo)

# Componentes da velocidade
v0x = v0 * np.cos(angulo_rad)
v0y = v0 * np.sin(angulo_rad)

# Tempo total de voo
t_total = 2 * v0y / g

# Intervalo de tempo
t = np.linspace(0, t_total, num=100)

# Trajetória
x = v0x * t
y = v0y * t - 0.5 * g * t**2

# Plotando o gráfico
plt.figure(figsize=(10, 5))
plt.plot(x, y, label=f'v0 = {v0} m/s, ângulo = {angulo}°')
plt.title('Lançamento Oblíquo')
plt.xlabel('Distância (m)')
plt.ylabel('Altura (m)')
plt.grid(True)
plt.legend()
plt.show()
