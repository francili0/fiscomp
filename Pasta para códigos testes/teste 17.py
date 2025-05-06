import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------
# GERADOR DE DADOS
# ----------------------------
np.random.seed(42)

def generate_data(nx, qtde, pmax):
    x = np.linspace(-1, 1, nx).reshape(-1, 1)
    y = []
    dy = []

    for _ in range(qtde):
        p = np.random.randint(0, pmax + 1)
        coeffs = np.random.randn(p + 1)

        polinomio = np.polyval(coeffs, x)
        noise = 0.1 * np.random.randn(len(x)).reshape(-1, 1)
        y.append(polinomio / np.max(np.abs(polinomio)) + noise)

        noise = 0.1 * np.random.randn(len(x)).reshape(-1, 1)
        dy.append(np.polyval(np.polyder(coeffs), x) / np.max(np.abs(polinomio)) + noise)

    y = np.hstack(y).T
    dy = np.hstack(dy).T
    return y, dy

# ----------------------------
# GERAÇÃO E SEPARAÇÃO DOS DADOS
# ----------------------------
y, dy = generate_data(50, 10000, 10)
X_train, X_test, y_train, y_test = train_test_split(y, dy, test_size=0.2, random_state=42)

# ----------------------------
# DEFINIÇÃO E TREINAMENTO DO MODELO
# ----------------------------
neurons = 10
layers = 10

model = MLPRegressor(
    hidden_layer_sizes=tuple([neurons] * layers),
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

# ----------------------------
# AVALIAÇÃO DO MODELO
# ----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# ----------------------------
# TESTES VISUAIS COM FUNÇÕES CONHECIDAS
# ----------------------------
plt.figure(figsize=(10, 4))
new_x = np.linspace(0, 1, y.shape[1]).reshape(1, -1)

# Teste 1: seno
plt.subplot(131)
new_y = np.sin(2 * np.pi * new_x)
new_dy = 2 * np.pi * np.cos(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Input: sin(2πx)', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-2*np.pi, 2*np.pi)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 2: cosseno
plt.subplot(132)
new_y = np.cos(2 * np.pi * new_x)
new_dy = -2 * np.pi * np.sin(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Input: cos(2πx)', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-2*np.pi, 2*np.pi)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 3: parabólica
plt.subplot(133)
new_y = new_x ** 2
new_dy = 2 * new_x
predicted_derivative = model.predict(new_y)

plt.plot(new_x[0], new_y[0], label='Input: x²', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
