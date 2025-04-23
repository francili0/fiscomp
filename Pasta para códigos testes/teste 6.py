import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def U(x):
    return x**4 - x**2

def dU(x):
    return 4 * x**3 - 2 * x

# Parâmetros
x = 2            # Tente mudar para -2, -0.5, 0.5, etc.
alpha = 0.01       # Taxa de aprendizado menor, pois a curva cresce rápido
epochs = 50

# Guarda os passos
x_vals = [x]
y_vals = [U(x)]

# Gradiente descendente
for _ in range(epochs):
    grad = dU(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(U(x))

# Curva da função
x_curve = np.linspace(-5.0, 5.0, 500)
y_curve = U(x_curve)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='U(x) = x²(x - 1)(x + 1)', color='blue')
point, = ax.plot([], [], 'ro')
line, = ax.plot([], [], 'r--', alpha=0.1)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(min(y_curve) - 1, max(y_vals)*1.1)
ax.set_xlabel('x')
ax.set_ylabel('U(x)')
ax.set_title('Gradiente Descendente em U(x) = x²(x - 1)(x + 1)')
ax.grid(True)
ax.legend()

# Função de animação
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, line

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()
