import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Gera funções polinomiais aleatórias de grau até 10
def generate_data(nx, qtde, pmax):
    x = np.linspace(-1, 1, nx).reshape(-1, 1)
    y = []
    dy = []
    
    for _ in range(qtde):
        # Gera um polinômio aleatório de grau p ≤ pmax
        p = np.random.randint(0, pmax + 1)
        coeffs = np.random.randn(p + 1)

        # Avalia o polinômio e sua derivada
        polinomio = np.polyval(coeffs, x)
        derivada = np.polyval(np.polyder(coeffs), x)

        # Normaliza os valores (para facilitar o aprendizado)
        polinomio /= np.max(np.abs(polinomio))
        derivada /= np.max(np.abs(polinomio))

        # Adiciona ruído (simula dados reais)
        noise_y = 0.05 * np.random.randn(len(x), 1)
        noise_dy = 0.05 * np.random.randn(len(x), 1)

        y.append(polinomio + noise_y)
        dy.append(derivada + noise_dy)

    y = np.hstack(y).T
    dy = np.hstack(dy).T
    return y, dy

# Gera os dados com polinômios até grau 10
y, dy = generate_data(nx=50, qtde=10000, pmax=10)

# Divide entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(y, dy, test_size=0.2, random_state=42)

# Define e treina a rede neural
model = MLPRegressor(
    hidden_layer_sizes=(10,) * 10,
    activation='tanh',
    solver='adam',
    max_iter=100000,
    random_state=42,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    n_iter_no_change=50,
    tol=1e-8,
    verbose=True
)

model.fit(X_train, y_train)

# Avalia o desempenho
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (Erro Quadrático Médio): {mse:.6f}")

# Testa com funções conhecidas
plt.figure(figsize=(10, 4))
new_x = np.linspace(-1, 1, y.shape[1]).reshape(1, -1)

# Teste 1: seno
plt.subplot(131)
new_y = np.sin(2 * np.pi * new_x)
new_dy = np.cos(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Entrada: sin(x)', color='black')
plt.plot(new_x[0], new_dy[0], label='Derivada real', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Derivada prevista', color='red', linestyle='dashed')
plt.title("Teste 1: sin(x)")
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)

# Teste 2: cosseno
plt.subplot(132)
new_y = np.cos(2 * np.pi * new_x)
new_dy = -np.sin(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Entrada: cos(x)', color='black')
plt.plot(new_x[0], new_dy[0], label='Derivada real', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Derivada prevista', color='red', linestyle='dashed')
plt.title("Teste 2: cos(x)")
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)

# Teste 3: x²
plt.subplot(133)
new_y = new_x ** 2
new_dy = 2 * new_x
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Entrada: x²', color='black')
plt.plot(new_x[0], new_dy[0], label='Derivada real', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Derivada prevista', color='red', linestyle='dashed')
plt.title("Teste 3: x²")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

