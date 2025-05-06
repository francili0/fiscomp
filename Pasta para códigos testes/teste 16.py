import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gerar dados: y = sin(x)
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo MLP
mlp = MLPRegressor(hidden_layer_sizes=(15, 15), activation='tanh', solver='adam',
                   max_iter=1000, random_state=1)

# Treinar
mlp.fit(X_train, y_train)

# Prever
y_pred = mlp.predict(X_test)

# Avaliar
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio: {mse:.4f}")

# Visualizar
X_plot = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y_plot = mlp.predict(X_plot)

plt.figure(figsize=(8, 4))
plt.plot(X, y, 'b.', label='Dados reais')
plt.plot(X_plot, y_plot, 'r-', label='Previsão MLP')
plt.legend()
plt.title("Aproximação da função seno com MLPRegressor")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
