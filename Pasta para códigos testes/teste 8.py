import numpy as np
import matplotlib.pyplot as plt

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
max_iter = 200
epsilon = 1e-6
x, y = 2.5, 1.5  # Posição inicial

# Armazenar trajetória
xs, ys, us = [x], [y], [U(x, y)]

# Gradiente descendente
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
else:
    print("Máximo de iterações atingido.")

# Gráfico de contorno com trajetória e setas
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

plt.figure(figsize=(8, 6))
contour = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(contour)

# Trajetória
plt.plot(xs, ys, 'r--o', label='Trajetória')

# Adiciona setas (vetores) na trajetória
for i in range(len(xs) - 1):
    dx = xs[i+1] - xs[i]
    dy = ys[i+1] - ys[i]
    plt.quiver(xs[i], ys[i], dx, dy, angles='xy', scale_units='xy', scale=1, color='white', width=0.003)

plt.title('Gradiente Descendente - Trajetória com Vetores')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico do valor de U por iteração
plt.figure(figsize=(6, 3))
plt.plot(us, 'b-o')
plt.title('Valor de U(x, y) por Iteração (Epoch)')
plt.xlabel('Iteração')
plt.ylabel('U(x, y)')
plt.grid(True)
plt.show()
