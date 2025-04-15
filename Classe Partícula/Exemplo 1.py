input("Digite a idade: ")

print ("A informação digitada foi:,input(Digite a idade: )")

dado=input("Digite a idade: ")
print("A informação digitada foi: , dado")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Gradiente de U
def grad_U(x, y):
    dU_dx = np.cos(x) * np.cos(y) + (4 * x * y**2) / 1000
    dU_dy = -np.sin(x) * np.sin(y) + (4 * x**2 * y) / 1000
    return dU_dx, dU_dy

# Hiperparâmetros
alpha = 0.1
max_iter = 200
epsilon = 1e-5
x, y = 2.5, 1.5  # Posição inicial

# Armazena trajetória
xs, ys, us = [x], [y], [U(x, y)]

# Gradiente descendente
for _ in range(max_iter):
    grad_x, grad_y = grad_U(x, y)
    if np.sqrt(grad_x**2 + grad_y**2) < epsilon:
        break
    x -= alpha * grad_x
    y -= alpha * grad_y
    xs.append(x)
    ys.append(y)
    us.append(U(x, y))

# Malha de pontos
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Preparar gráfico
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(contour, ax=ax)
particle, = ax.plot([], [], 'ro', markersize=6)
path, = ax.plot([], [], 'r--', linewidth=1)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Animação do Gradiente Descendente')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Função de atualização
def update(frame):
    particle.set_data(xs[frame], ys[frame])
    path.set_data(xs[:frame+1], ys[:frame+1])
    return particle, path

ani = FuncAnimation(fig, update, frames=len(xs), interval=200, repeat=False)

plt.show()
