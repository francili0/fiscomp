import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Gerar dados de exemplo
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Criar e treinar o modelo MLP com camadas menores
mlp = MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', max_iter=5000, random_state=1)
mlp.fit(X, y)

# Fazer previsões
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = mlp.predict(X_test)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Dados reais')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Previsão do MLP (2, 2)')
plt.title('Regressão com MLPRegressor (2 neurônios por camada)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
