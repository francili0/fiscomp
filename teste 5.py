import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Criar grade para x e y
x_vals = np.linspace(-4, 4, 300)
y_vals = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Criar figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superfície 3D
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Título e rótulos
ax.set_title('Gráfico 3D de U(x, y) = sin(x)cos(y) + 2(xy)^2/1000')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x, y)')

# Barra de cores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Criar grade de pontos
x_vals = np.linspace(-4, 4, 300)
y_vals = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Criar a figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superfície 3D da função
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Curvas de nível projetadas no plano XY (em z=-1.5 só para aparecer abaixo da superfície)
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 0.5, cmap='viridis')

# Rótulos e título
ax.set_title('Gráfico 3D de U(x, y) com Curvas de Nível na Base')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x, y)')

# Ajustes do eixo Z
ax.set_zlim(np.min(Z) - 0.5, np.max(Z))

# Barra de cores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função U(x, y)
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Gradiente de U
def grad_U(x, y):
    dU_dx = np.cos(x) * np.cos(y) + 4 * x * y**2 / 1000
    dU_dy = -np.sin(x) * np.sin(y) + 4 * x**2 * y / 1000
    return np.array([dU_dx, dU_dy])

# Parâmetros do gradiente descendente
learning_rate = 0.1
max_iter = 100
x, y = 2.0, 2.0

trajectory = [(x, y, U(x, y))]

# Gradiente descendente
for _ in range(max_iter):
    grad = grad_U(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    trajectory.append((x, y, U(x, y)))

trajectory = np.array(trajectory)

# Grade para superfície
x_vals = np.linspace(-4, 4, 300)
y_vals = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superfície da função
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Curvas de nível na base
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 0.5, cmap='viridis')

# Trajetória do gradiente descendente
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color='red', marker='o', markersize=3, linewidth=2, label='Trajetória')

# Eixos e rótulos
ax.set_title('Trajetória do Gradiente Descendente em 3D')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x, y)')
ax.set_zlim(np.min(Z) - 0.5, np.max(Z))
ax.legend()

# Barra de cores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()
