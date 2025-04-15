import numpy as np
import matplotlib.pyplot as plt  # <-- Biblioteca para plotar

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

# Inicialização
np.random.seed(42)

input_size = 2
hidden1_size = 4
hidden2_size = 4
hidden3_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden1_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, hidden3_size)
b3 = np.zeros((1, hidden3_size))

W4 = np.random.randn(hidden3_size, output_size)
b4 = np.zeros((1, output_size))

# Treinamento
epochs = 10000
lr = 0.1
loss_history = []  # <-- Armazena a perda

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    z4 = np.dot(a3, W4) + b4
    y_pred = sigmoid(z4)

    # Erro
    error = y - y_pred
    loss = np.mean(np.square(error))
    loss_history.append(loss)

    # Backpropagation
    d4 = error * sigmoid_deriv(y_pred)
    d3 = d4.dot(W4.T) * sigmoid_deriv(a3)
    d2 = d3.dot(W3.T) * sigmoid_deriv(a2)
    d1 = d2.dot(W2.T) * sigmoid_deriv(a1)

    # Atualização
    W4 += a3.T.dot(d4) * lr
    b4 += np.sum(d4, axis=0, keepdims=True) * lr

    W3 += a2.T.dot(d3) * lr
    b3 += np.sum(d3, axis=0, keepdims=True) * lr

    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f"Época {epoch} - Erro: {loss:.4f}")

# Previsões
print("\nPrevisões finais:")
print(y_pred.round())

# Plotar a perda
plt.plot(loss_history)
plt.title("Perda durante o treinamento")
plt.xlabel("Época")
plt.ylabel("Erro quadrático médio")
plt.grid(True)
plt.show()
