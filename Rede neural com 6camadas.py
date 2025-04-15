import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saídas esperadas
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Função sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Função para treinar a rede
def train_network(lr, epochs=10000):
    np.random.seed(42)

    input_size = 2
    h1, h2, h3, h4, h5, h6 = 4, 4, 4, 4, 4, 4
    output_size = 1

    # Inicializar pesos e bias para 6 camadas ocultas
    W1 = np.random.randn(input_size, h1)
    b1 = np.zeros((1, h1))

    W2 = np.random.randn(h1, h2)
    b2 = np.zeros((1, h2))

    W3 = np.random.randn(h2, h3)
    b3 = np.zeros((1, h3))

    W4 = np.random.randn(h3, h4)
    b4 = np.zeros((1, h4))

    W5 = np.random.randn(h4, h5)
    b5 = np.zeros((1, h5))

    W6 = np.random.randn(h5, h6)
    b6 = np.zeros((1, h6))

    W7 = np.random.randn(h6, output_size)
    b7 = np.zeros((1, output_size))

    loss_history = []

    for _ in range(epochs):
        # Forward pass
        a1 = sigmoid(X @ W1 + b1)
        a2 = sigmoid(a1 @ W2 + b2)
        a3 = sigmoid(a2 @ W3 + b3)
        a4 = sigmoid(a3 @ W4 + b4)
        a5 = sigmoid(a4 @ W5 + b5)
        a6 = sigmoid(a5 @ W6 + b6)
        y_pred = sigmoid(a6 @ W7 + b7)

        # Erro
        error = y - y_pred
        loss = np.mean(np.square(error))
        loss_history.append(loss)

        # Backpropagation
        d7 = error * sigmoid_deriv(y_pred)
        d6 = d7 @ W7.T * sigmoid_deriv(a6)
        d5 = d6 @ W6.T * sigmoid_deriv(a5)
        d4 = d5 @ W5.T * sigmoid_deriv(a4)
        d3 = d4 @ W4.T * sigmoid_deriv(a3)
        d2 = d3 @ W3.T * sigmoid_deriv(a2)
        d1 = d2 @ W2.T * sigmoid_deriv(a1)

        # Atualização dos pesos e bias
        W7 += a6.T @ d7 * lr
        b7 += np.sum(d7, axis=0, keepdims=True) * lr

        W6 += a5.T @ d6 * lr
        b6 += np.sum(d6, axis=0, keepdims=True) * lr

        W5 += a4.T @ d5 * lr
        b5 += np.sum(d5, axis=0, keepdims=True) * lr

        W4 += a3.T @ d4 * lr
        b4 += np.sum(d4, axis=0, keepdims=True) * lr

        W3 += a2.T @ d3 * lr
        b3 += np.sum(d3, axis=0, keepdims=True) * lr

        W2 += a1.T @ d2 * lr
        b2 += np.sum(d2, axis=0, keepdims=True) * lr

        W1 += X.T @ d1 * lr
        b1 += np.sum(d1, axis=0, keepdims=True) * lr

    return loss_history

# Testar com uma taxa de aprendizado (lr)
lr = 0.005
loss_history = train_network(lr)

# Plotar a perda
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label=f"lr = {lr}")
plt.title("Curva de perda para rede com 6 camadas ocultas")
plt.xlabel("Época")
plt.ylabel("Erro quadrático médio")
plt.legend()
plt.grid(True)
plt.show()

# Previsões finais
print("\nPrevisões finais:")
print(y_pred.round())
