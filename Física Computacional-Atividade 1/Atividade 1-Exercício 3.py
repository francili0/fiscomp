import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def U(x):
    return x**2 * (x - 1) * (x + 1) + x / 4

def dU(x):
    # Derivada da função: use regra do produto + derivada da constante
    return (4 * x**3 - 2 * x) + 1/4

# Parâmetros
x = 2.0                 # Valor inicial
alpha = 0.09           # Taxa de aprendizado (experimente outros valores!)
epsilon = 0.001         # Critério de parada
max_iter = 1000

# Guardar trajetória
x_vals = [x]
y_vals = [U(x)]

# Gradiente descendente
for _ in range(max_iter):
    grad = dU(x)
    if abs(grad) < epsilon:
        print("Convergência atingida!")
        break
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(U(x))
else:
    print("Máximo de iterações atingido.")

# Curva da função
x_plot = np.linspace(-2.5, 2.5, 1000)
y_plot = U(x_plot)

# Gráfico
plt.figure(figsize=(5, 3))
plt.plot(x_plot, y_plot, label='U(x)', color='blue')
plt.plot(x_vals, y_vals, 'ro--', label='Trajetória do gradiente')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Gradiente Descendente em U(x) = x²(x-1)(x+1) + x/4')
plt.grid(True)
plt.legend()
plt.show()
