import numpy as np
import matplotlib.pyplot as plt

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Gradiente de U(x, y)
def grad_U(x, y):
    dU_dx = np.cos(x) * np.cos(y) + (4 * x * y**2) / 1000
    dU_dy = -np.sin(x) * np.sin(y) + (4 * x**2 * y) / 1000
    return dU_dx, dU_dy

# Hiperparâmetros
alpha = 0.1           # Taxa de aprendizado (experimente diferentes valores!)
max_iter = 200        # Número máximo de iterações
epsilon = 1e-5        # Critério de parada

# Ponto inicial (varie!)
x, y = 2.5, 1.5

# Guardar trajetória
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

# Malha para visualização
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Gráficos
plt.figure(figsize=(14, 6))

# (a) Gráfico de contorno + trajetória
plt.subplot(1, 2, 1)
contour = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.plot(xs, ys, 'ro--', label='Trajetória da partícula')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Caminho da partícula no plano (x, y)')
plt.legend()

# (b) Evolução de U ao longo das iterações
plt.subplot(1, 2, 2)
plt.plot(us, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('U(x, y)')
plt.title('Evolução do valor de U(x, y)')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Gradiente da função
def grad_U(x, y):
    dU_dx = np.cos(x) * np.cos(y) + (4 * x * y**2) / 1000
    dU_dy = -np.sin(x) * np.sin(y) + (4 * x**2 * y) / 1000
    return dU_dx, dU_dy

# Hiperparâmetros
alpha = 0.1
epsilon = 1e-5
max_iter = 200
x, y = 2.5, 1.5  # Posição inicial

# Trajetória
xs, ys = [x], [y]

for _ in range(max_iter):
    grad_x, grad_y = grad_U(x, y)
    if np.sqrt(grad_x**2 + grad_y**2) < epsilon:
        break
    x -= alpha * grad_x
    y -= alpha * grad_y
    xs.append(x)
    ys.append(y)

# Malha para contorno
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Gráfico base
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(contour, ax=ax)
particle, = ax.plot([], [], 'ro', markersize=6)
path, = ax.plot([], [], 'r--', linewidth=1)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradiente Descendente: Trajetória da Partícula')

# Atualização da animação
def update(i):
    particle.set_data(xs[i], ys[i])
    path.set_data(xs[:i+1], ys[:i+1])
    return particle, path

ani = FuncAnimation(fig, update, frames=len(xs), interval=200, repeat=False)
plt.show()
