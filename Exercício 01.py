def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros
x = 10            # Valor inicial
alpha = 0.1       # Taxa de aprendizado
epochs = 100      # Número de iterações

for i in range(epochs):
    grad = df(x)          # Calcula o gradiente
    x = x - alpha * grad  # Atualiza x
    print(f"Iteração {i+1}: x = {x:.5f}, f(x) = {f(x):.5f}")
def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros
x = 10            # Valor inicial
alpha = 0.1       # Taxa de aprendizado
epochs = 100      # Número de iterações

for i in range(epochs):
    grad = df(x)          # Calcula o gradiente
    x = x - alpha * grad  # Atualiza x
    print(f"Iteração {i+1}: x = {x:.5f}, f(x) = {f(x):.5f}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros do gradiente descendente
x = 5.0
alpha = 0.1
epochs = 1000

# Lista para armazenar os passos
x_vals = [x]
y_vals = [f(x)]

# Executa o gradiente descendente e armazena os valores
for _ in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Curva da função para desenhar
x_curve = np.linspace(-11, 11, 400)
y_curve = f(x_curve)

# Configuração do plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='f(x) = x²', color='blue')
point, = ax.plot([], [], 'ro')  # ponto atual
line, = ax.plot([], [], 'r--', alpha=0.6)  # linha dos passos
ax.set_xlim(-11, 11)
ax.set_ylim(0, max(y_vals)*1.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradiente Descendente em f(x) = x²')
ax.grid(True)
ax.legend()

# Função de animação
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, line

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros
x = 10.0         # Valor inicial
alpha = 0.1      # Taxa de aprendizado
epochs = 25      # Número de iterações

# Para plotar os passos
x_vals = [x]
y_vals = [f(x)]

# Gradiente descendente
for _ in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Plot da função
x_plot = np.linspace(-11, 11, 400)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x²', color='blue')
plt.scatter(x_vals, y_vals, color='red', label='Passos do gradiente')
plt.plot(x_vals, y_vals, color='red', linestyle='--', alpha=0.5)
plt.title('Gradiente Descendente em f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros
x = 10.0         # Chute inicial
alpha = 0.1      # Taxa de aprendizado
epochs = 50      # Número de iterações

for i in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    print(f"Iteração {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros do gradiente descendente
x = 5.0
alpha = 0.001
epochs = 1000

# Lista para armazenar os passos
x_vals = [x]
y_vals = [f(x)]

# Executa o gradiente descendente e armazena os valores
for _ in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Curva da função para desenhar
x_curve = np.linspace(-11, 11, 400)
y_curve = f(x_curve)

# Configuração do plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='f(x) = x²', color='blue')
point, = ax.plot([], [], 'ro')  # ponto atual
line, = ax.plot([], [], 'r--', alpha=0.1)  # linha dos passos
ax.set_xlim(-11, 11)
ax.set_ylim(0, max(y_vals)*1.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradiente Descendente em f(x) = x²')
ax.grid(True)
ax.legend()

# Função de animação
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, line

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def f(x):
    return x**2 - 1

def df(x):
    return 2 * x

# Parâmetros do gradiente descendente
x = 100.0
alpha = 0.1
epochs = 1000

# Lista para armazenar os passos
x_vals = [x]
y_vals = [f(x)]

# Executa o gradiente descendente
for _ in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Curva da função para desenhar
x_curve = np.linspace(-10, 10, 200)
y_curve = f(x_curve)

# Configuração do plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='f(x) = x² - 1', color='blue')
point, = ax.plot([], [], 'ro')  # ponto atual
line, = ax.plot([], [], 'r--', alpha=0.6)  # linha dos passos
ax.set_xlim(-11, 11)
ax.set_ylim(min(y_curve) - 1, max(y_vals)*1.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradiente Descendente em f(x) = x² - 1')
ax.grid(True)
ax.legend()

# Função de animação
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, line

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def U(x):
    return x**4 - x**2

def dU(x):
    return 4 * x**3 - 2 * x

# Parâmetros
x = 2.0            # Tente mudar para -2, -0.5, 0.5, etc.
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
x_curve = np.linspace(-2.5, 2.5, 500)
y_curve = U(x_curve)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='U(x) = x²(x - 1)(x + 1)', color='blue')
point, = ax.plot([], [], 'ro')
line, = ax.plot([], [], 'r--', alpha=0.6)
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


import numpy as np
import matplotlib.pyplot as plt

# Função e gradiente
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

def grad_U(x, y):
    dUx = np.cos(x) * np.cos(y) + (4 * x * y**2) / 1000
    dUy = -np.sin(x) * np.sin(y) + (4 * x**2 * y) / 1000
    return dUx, dUy

# Parâmetros
x, y = 2.0, 2.0      # Ponto inicial
alpha = 0.1          # Taxa de aprendizado
epochs = 100

# Guarda os valores
xs, ys, zs = [x], [y], [U(x, y)]

# Gradiente descendente
for _ in range(epochs):
    dx, dy = grad_U(x, y)
    x = x - alpha * dx
    y = y - alpha * dy
    xs.append(x)
    ys.append(y)
    zs.append(U(x, y))

# Gera malha para gráfico de contorno
X, Y = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-4, 4, 400))
Z = U(X, Y)

# Gráfico 1: Contorno com os passos
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.plot(xs, ys, 'ro--', label='Caminho')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contorno de U(x, y) com Caminho do Gradiente')
plt.legend()

# Gráfico 2: Evolução de U(x, y)
plt.subplot(1, 2, 2)
plt.plot(zs, 'r-')
plt.xlabel('Iteração')
plt.ylabel('U(x, y)')
plt.title('Evolução do Custo')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Função e gradiente
def f(x, y):
    return np.sin(x) * np.cos(y) + (x * y)**2 / 100

def grad_f(x, y):
    df_dx = np.cos(x) * np.cos(y) + (2 * x * y**2) / 100
    df_dy = -np.sin(x) * np.sin(y) + (2 * x**2 * y) / 100
    return df_dx, df_dy

# Inicialização
x, y = 2.0, 2.0
alpha = 0.1
epochs = 100

xs, ys, zs = [x], [y], [f(x, y)]

# Gradiente descendente
for _ in range(epochs):
    dx, dy = grad_f(x, y)
    x -= alpha * dx
    y -= alpha * dy
    xs.append(x)
    ys.append(y)
    zs.append(f(x, y))

# Malha para contorno
X, Y = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-4, 4, 400))
Z = f(X, Y)

# Gráficos
plt.figure(figsize=(12, 5))

# Gráfico 1: contorno com trajetória
plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.plot(xs, ys, 'ro--', label='Caminho do gradiente')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Caminho do gradiente em f(x, y)')
plt.legend()

# Gráfico 2: evolução do custo
plt.subplot(1, 2, 2)
plt.plot(zs, 'r-')
plt.xlabel('Iteração')
plt.ylabel('f(x, y)')
plt.title('Evolução de f(x, y)')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Função e gradiente
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return 2 * x, 2 * y

# Inicialização
x, y = 3.0, 2.0   # Ponto inicial (pode testar outros)
alpha = 0.1
epochs = 50

xs, ys, zs = [x], [y], [f(x, y)]

# Gradiente descendente
for _ in range(epochs):
    dx, dy = grad_f(x, y)
    x -= alpha * dx
    y -= alpha * dy
    xs.append(x)
    ys.append(y)
    zs.append(f(x, y))

# Malha para contorno
X, Y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-4, 4, 400))
Z = f(X, Y)

# Gráficos
plt.figure(figsize=(12, 5))

# Gráfico 1: contorno com caminho
plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')
plt.colorbar(contour)
plt.plot(xs, ys, 'ro--', label='Caminho do gradiente')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Caminho do gradiente em f(x, y) = x² + y²')
plt.legend()

# Gráfico 2: evolução da função
plt.subplot(1, 2, 2)
plt.plot(zs, 'g-')
plt.xlabel('Iteração')
plt.ylabel('f(x, y)')
plt.title('Evolução de f(x, y) ao longo das iterações')

plt.tight_layout()
plt.show()

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
x_curve = np.linspace(-6, 6, 400)
y_curve = U(x_curve)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_curve, label='U(x) = x² - 1', color='blue')
plt.plot(x_vals, y_vals, 'ro--', label='Trajetória do gradiente')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def f(x):
    return x**2

def df(x):
    return 2 * x

# Parâmetros
alpha = 0.1
x0 = 5
epsilon = 0.01
max_iter = 100

# Executa o algoritmo e guarda os passos
x_vals = [x0]
y_vals = [f(x0)]
x = x0

for _ in range(max_iter):
    grad = df(x)
    if abs(grad) < epsilon:
        break
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Prepara o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
x_plot = np.linspace(-6, 6, 400)
y_plot = f(x_plot)
ax.plot(x_plot, y_plot, label='f(x) = x²', color='blue')
point, = ax.plot([], [], 'ro', label='Partícula')
path, = ax.plot([], [], 'r--', alpha=0.5)

ax.set_xlim(-6, 6)
ax.set_ylim(-1, max(y_plot) + 5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradiente Descendente em f(x) = x²')
ax.grid(True)
ax.legend()

# Função de atualização da animação
def update(i):
    point.set_data(x_vals[i], y_vals[i])
    path.set_data(x_vals[:i+1], y_vals[:i+1])
    return point, path

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Função e derivada
def f(x):
    return x**2

def df(x):
    return 2*x

# Parâmetros do gradiente descendente
x_init = 8              # ponto inicial
learning_rate = 0.1
n_iter = 25             # número de iterações

# Armazenar os passos para visualização
x_vals = [x_init]
f_vals = [f(x_init)]

x = x_init
for _ in range(n_iter):
    x = x - learning_rate * df(x)
    x_vals.append(x)
    f_vals.append(f(x))

# Criar o gráfico
x_curve = np.linspace(-10, 10, 400)
y_curve = f(x_curve)

plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_curve, label="f(x) = x²", color="blue")
plt.scatter(x_vals, f_vals, color="red")
plt.plot(x_vals, f_vals, color="red", linestyle="--", label="Trajetória")

for i, (x_pt, y_pt) in enumerate(zip(x_vals, f_vals)):
    plt.annotate(f'{i}', (x_pt, y_pt), textcoords="offset points", xytext=(0,5), ha='center')

plt.title("Gradiente Descendente em f(x) = x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função e derivada
def f(x):
    return x**2 - 1

def df(x):
    return 2 * x

# Parâmetros do gradiente descendente
x = 10.0
alpha = 0.1
epochs = 30

# Lista para armazenar os passos
x_vals = [x]
y_vals = [f(x)]

# Executa o gradiente descendente
for _ in range(epochs):
    grad = df(x)
    x = x - alpha * grad
    x_vals.append(x)
    y_vals.append(f(x))

# Curva da função para desenhar
x_curve = np.linspace(-11, 11, 400)
y_curve = f(x_curve)

# Configuração do plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_curve, y_curve, label='f(x) = x² - 1', color='blue')
point, = ax.plot([], [], 'ro')  # ponto atual
line, = ax.plot([], [], 'r--', alpha=0.6)  # linha dos passos
ax.set_xlim(-11, 11)
ax.set_ylim(min(y_curve) - 1, max(y_vals)*1.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradiente Descendente em f(x) = x² - 1')
ax.grid(True)
ax.legend()

# Função de animação
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, line

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=300, repeat=False)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Função e sua derivada
def U(x):
    return x**2 - 1

def dU(x):
    return 2*x

# Parâmetros do gradiente descendente
x_init = 5               # ponto inicial
learning_rate = 0.2
n_iter = 15

# Armazenar os passos
x_vals = [x_init]
f_vals = [U(x_init)]

x = x_init
for _ in range(n_iter):
    x = x - learning_rate * dU(x)
    x_vals.append(x)
    f_vals.append(U(x))

# Curva da função
x_curve = np.linspace(-6, 6, 400)
y_curve = U(x_curve)

# Criar gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_curve, label="U(x) = x² - 1", color="blue")

# Trajetória com setas
for i in range(len(x_vals) - 1):
    x0, y0 = x_vals[i], f_vals[i]
    x1, y1 = x_vals[i + 1], f_vals[i + 1]
    plt.arrow(x0, y0, x1 - x0, y1 - y0,
              head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True)

# Marcar os pontos
plt.scatter(x_vals, f_vals, color='red')
plt.title("Gradiente Descendente em U(x) = x² - 1")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.grid(True)
plt.legend()
plt.show()

