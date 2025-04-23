import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
epsilon = 1e-6
x, y = 2.5, 1.5  # Ponto inicial

# Guardar trajetória
xs, ys, us = [x], [y], [U(x, y)]

# Algoritmo de gradiente descendente
for _ in range(max_iter):
    grad_x, grad_y = grad_U(x, y)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    if grad_norm < epsilon:
        print("Convergência atingida!")
        break
    x -= alpha * grad_x
    y -= alpha * grad_y
    xs.append(x)
    ys.append(y)
    us.append(U(x, y))

# Malha para gráfico 3D
x_vals = np.linspace(-5, 5, 200)
y_vals = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.plot(xs, ys, us, 'r.-', label='Trajetória do gradiente', linewidth=2)
ax.set_title('Gradiente Descendente em 3D')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x, y)')
ax.legend()
plt.tight_layout()
plt.show()
