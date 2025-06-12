#Importa bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def U(x):
    return x**2 - 1

def dU(x):
    return 2 * x

# Parâmetros
x = 5.0
alpha = 0.9
epsilon = 0.01
max_iter = 1000

# Guardar trajetória
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
x_curve = np.linspace(-5, 5, 400)
y_curve = U(x_curve)

# Gráfico com setas
plt.figure(figsize=(6, 4))
plt.plot(x_curve, y_curve, label='U(x) = x² - 1', color='blue')
plt.plot(x_vals, y_vals, 'ro--', label='Trajetória do gradiente')

# Adicionar setas
for i in range(len(x_vals) - 1):
    dx = x_vals[i+1] - x_vals[i]
    dy = y_vals[i+1] - y_vals[i]
    plt.quiver(
        x_vals[i], y_vals[i],   # ponto de partida
        dx, dy,                 # vetor (delta x, delta y)
        angles='xy', scale_units='xy', scale=1,
        color='red', width=0.005
    )
#Configura rótulos, título, grade e exibe a legenda e o gráfico final.
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Gradiente Descendente com Setas')
plt.grid(True)
plt.legend()
plt.show()
