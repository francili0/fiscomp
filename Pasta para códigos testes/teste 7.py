import numpy as np
import matplotlib.pyplot as plt

# Função U(x) = x^2(x^2 - 1)
def U(x):
    return x**2 * (x**2 - 1)

# Derivada: dU/dx = 4x^3 - 2x
def dU(x):
    return 4 * x**3 - 2 * x

# Parâmetros
x = 2.0                   # Ponto inicial
alpha = 0.08             # Taxa de aprendizado (ajustável)
epsilon = 1e-4            # Critério de parada
max_iter = 1000           # Máximo de iterações

# Trajetória
x_vals = [x]
y_vals = [U(x)]

# Gradiente descendente
for i in range(max_iter):
    grad = dU(x)
    if abs(grad) < epsilon:
        print(f'Convergência atingida em {i} iterações.')
        break
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(U(x))
else:
    print('Máximo de iterações atingido sem convergência.')

# Curva da função
x_curve = np.linspace(-2.5, 2.5, 400)
y_curve = U(x_curve)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_curve, y_curve, label=r'$U(x) = x^2(x^2 - 1)$', color='blue')
plt.plot(x_vals, y_vals, 'ro--', label='Trajetória do gradiente')

# Setinhas
for i in range(len(x_vals) - 1):
    dx = x_vals[i+1] - x_vals[i]
    dy = y_vals[i+1] - y_vals[i]
    plt.quiver(x_vals[i], y_vals[i], dx, dy, angles='xy', scale_units='xy', scale=1, color='red', width=0.005)

plt.xlabel('x')
plt.ylabel('U(x)')
plt.title(f'Gradiente Descendente com α = {alpha}')
plt.grid(True)
plt.legend()
plt.show()
