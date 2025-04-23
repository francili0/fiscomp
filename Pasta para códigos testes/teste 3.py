import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def U(x):
    return x**2 - 1

def dU_dx(x):
    return 2 * x

# Parâmetros fixos
epsilon = 0.01
max_iter = 1000
x0 = 5.0
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Criar figura
plt.figure(figsize=(12, 6))

# Plot da função
x_vals = np.linspace(-6, 6, 400)
y_vals = U(x_vals)
plt.plot(x_vals, y_vals, 'k-', label='U(x) = x² - 1')

# Trajetórias para diferentes taxas de aprendizado
for alpha, color in zip(learning_rates, colors):
    x = x0
    trajectory = [x]
    
    for _ in range(max_iter):
        grad = dU_dx(x)
        if abs(grad) < epsilon:
            break
        x = x - alpha * grad
        trajectory.append(x)
    
    y_traj = [U(x) for x in trajectory]
    plt.plot(trajectory, y_traj, 'o-', label=f'α = {alpha}', color=color)

# Plot final
plt.title("Comparação de trajetórias para diferentes taxas de aprendizado")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.show()
