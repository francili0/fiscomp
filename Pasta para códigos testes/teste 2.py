import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def U(x):
    return x**2 - 1

def dU_dx(x):
    return 2 * x

# Parâmetros do algoritmo
alpha = 0.9      # taxa de aprendizado
epsilon = 0.01   # tolerância
max_iter = 1000  # número máximo de iterações
x = 5.0          # posição inicial

# Armazena a trajetória
trajectory = [x]

# Gradiente descendente
for i in range(max_iter):
    grad = dU_dx(x)
    if abs(grad) < epsilon:
        break
    x = x - alpha * grad
    trajectory.append(x)

# Plotando a função e a trajetória
x_vals = np.linspace(-6, 6, 400)
y_vals = U(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="U(x) = x² - 1", color="blue")
plt.plot(trajectory, [U(x) for x in trajectory], 'ro-', label="Trajetória")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.title("Gradiente Descendente na função U(x) = x² - 1")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Mínimo encontrado: x = {x:.4f} após {len(trajectory)-1} iterações")
