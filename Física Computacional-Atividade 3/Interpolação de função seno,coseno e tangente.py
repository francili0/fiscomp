import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Funções a serem aprendidas
functions = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan
}

# Configurações comuns
np.random.seed(42)
num_samples = 100
num_test_samples = 200
x_train = np.random.uniform(0, 2 * np.pi, num_samples).reshape(-1, 1)
x_test = np.linspace(0, 2 * np.pi, num_test_samples).reshape(-1, 1)

# Criação da figura para os subplots
plt.figure(figsize=(12, 10))

for i, (name, func) in enumerate(functions.items(), 1):
    # Gera y com ruído (exceto para tan onde aplicamos limites)
    y_train = func(x_train)
    if name == 'tan':
        # Remove valores extremos que causariam problemas para a rede e para visualização
        y_train = np.tan(x_train)
        y_train[np.abs(y_train) > 10] = np.nan  # remove outliers
        mask = ~np.isnan(y_train).flatten()
        x_train_filtered = x_train[mask]
        y_train_filtered = y_train[mask]
    else:
        # Adiciona ruído moderado
        noise = np.random.normal(0, 0.1, y_train.shape)
        y_train_filtered = y_train + noise
        x_train_filtered = x_train

    # Modelo FCNN
    model = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),
        activation='tanh',
        solver='adam',
        max_iter=100000,
        random_state=42,
        learning_rate_init=0.001,
        tol=1e-8
    )

    # Treina o modelo
    model.fit(x_train_filtered, y_train_filtered)

    # Predição
    y_test_true = func(x_test)
    if name == 'tan':
        y_test_true[np.abs(y_test_true) > 10] = np.nan  # assíntotas
    y_test_pred = model.predict(x_test)

    # Avaliação
    mask_eval = ~np.isnan(y_test_true.flatten())
    mse = mean_squared_error(y_test_true[mask_eval], y_test_pred[mask_eval])
    print(f"[{name}] Mean Squared Error on Test Data: {mse:.5f}")

    # Plot
    plt.subplot(3, 1, i)
    plt.scatter(x_train_filtered, y_train_filtered, alpha=0.5, label='Training Data')
    plt.plot(x_test, y_test_true, label=f'True {name}(x)', color='blue')
    plt.plot(x_test, y_test_pred, label=f'Predicted {name}(x)', color='red')
    plt.title(f'FCNN Approximation of {name}(x)')
    plt.xlabel('x (radians)')
    plt.ylabel(f'{name}(x)')
    plt.legend()
    plt.grid(True)
    if name == 'tan':
        plt.ylim(-10, 10)

plt.tight_layout()
plt.show()
