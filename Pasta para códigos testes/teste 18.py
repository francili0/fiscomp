import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Gerar dados: função seno
# -------------------------------
X = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)

y = np.cos(X)


# -------------------------------
# Funções de ativação
# -------------------------------
def sine_activation(x):
    return np.sin(x)

def dsine_activation(x):
    return np.cos(x)

# -------------------------------
# Inicialização dos pesos
# -------------------------------
np.random.seed(0)
W1 = np.random.randn(1, 10) * 0.5   # 1 entrada, 10 neurônios ocultos
b1 = np.zeros((1, 10))
W2 = np.random.randn(10, 1) * 0.5   # 10 entradas, 1 saída
b2 = np.zeros((1, 1))

# -------------------------------
# Forward pass
# -------------------------------
def forward(X):
    z1 = X @ W1 + b1
    a1 = sine_activation(z1)
    z2 = a1 @ W2 + b2
    y_hat = z2  # Saída linear (sem ativação final)
    cache = (X, z1, a1, z2, y_hat)
    return y_hat, cache

# -------------------------------
# Função de treinamento
# -------------------------------
def train(epochs=5000, lr=0.01):
    global W1, b1, W2, b2
    losses = []
    for epoch in range(epochs):
        y_hat, (X_cache, z1, a1, z2, _) = forward(X)

        # Erro quadrático médio (MSE)
        loss = np.mean((y_hat - y) ** 2)
        losses.append(loss)

        # Gradiente da saída
        dL_dyhat = 2 * (y_hat - y)
        dL_dW2 = a1.T @ dL_dyhat
        dL_db2 = np.sum(dL_dyhat, axis=0, keepdims=True)

        # Retropropagação para camada oculta
        dL_da1 = dL_dyhat @ W2.T
        da1_dz1 = dsine_activation(z1)
        dL_dz1 = dL_da1 * da1_dz1

        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Atualização dos pesos
        W1 -= lr * dL_dW1
        b1 -= lr * dL_db1
        W2 -= lr * dL_dW2
        b2 -= lr * dL_db2

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    return losses

# -------------------------------
# Treinamento
# -------------------------------
losses = train()

# -------------------------------
# Predição final
# -------------------------------
y_hat, _ = forward(X)

# -------------------------------
# Visualização
# -------------------------------
plt.figure(figsize=(10, 4))

# Gráfico 1: Seno real vs previsão da rede
plt.subplot(1, 2, 1)
plt.plot(X, y, label="sin(x)", color='blue')
plt.plot(X, y_hat, label="Predição da rede", color='red')
plt.title("Função seno vs saída da rede")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Gráfico 2: Erro ao longo do tempo
plt.subplot(1, 2, 2)
plt.plot(losses, color='green')
plt.title("Erro (MSE) durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("Erro")

plt.tight_layout()
plt.show()

# ---------------------------------------
# Predição final da rede (depois de treinar)
# ---------------------------------------
y_hat, _ = forward(X)

# ---------------------------------------
# Gráfico: função real vs predição da rede
# ---------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(X, y, label='Função real: sin(x)', color='blue')
plt.plot(X, y_hat, label='Predição da rede', color='red')
plt.title('Predição da Rede Neural vs sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
