import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Gerar dados de exemplo com função linear
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 3 + 0.1 * np.random.randn(100)  # y = 2x + 3 com ruído

# Criar e treinar o modelo MLP com (2, 2) neurônios
mlp = MLPRegressor(hidden_layer_sizes=(2, 2), activation='relu', solver='adam',
                   max_iter=5000, random_state=1)
mlp.fit(X, y)

# Fazer previsões
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = mlp.predict(X_test)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Dados reais (linha com ruído)')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Previsão do MLP')
plt.title('Regressão Linear com MLPRegressor (2 neurônios por camada)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

