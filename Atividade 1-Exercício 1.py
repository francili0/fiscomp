import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def U(x):
    return x**2 - 1

def dU(x):
    return 2 * x

# Parâmetros
x = 5.0                   # Posição inicial
alpha = 0.1               # Taxa de aprendizado
epsilon = 0.01            # Tolerância
max_iter = 1000           # Número máximo de iterações

# Guardar trajetória
x_vals = [x]
y_vals = [U(x)]

# Algoritmo de gradiente descendente
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
x_curve = np.linspace(-5, 5, 400)
y_curve = U(x_curve)

# Gráfico
plt.figure(figsize=(5, 3))
plt.plot(x_curve, y_curve, label='U(x) = x² - 1', color='blue')
plt.plot(x_vals, y_vals, 'ro--', label='Trajetória do gradiente')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Gradiente Descendente em U(x) = x² - 1')
plt.grid(True)
plt.legend()
plt.show()
