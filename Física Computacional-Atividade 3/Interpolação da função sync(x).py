import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Generate Training Data
np.random.seed(42)  # for reproducibility
num_samples = 100
angles_train = np.random.uniform(-10, 10, num_samples).reshape(-1, 1)
sync_values_train = np.sin(angles_train)/angles_train

x_train = np.random.uniform(-10, 10, num_samples).reshape(-1, 1)
gaus_values_train = np.exp(-x_train**2)

# Add some noise to the training data (optional, but can make it more realistic)
noise = np.random.normal(0, 0.01, sync_values_train.shape)
sync_values_train += noise
gaus_values_train += noise

# 2. Define the model
model = MLPRegressor(
    hidden_layer_sizes=(10,10,10),  # 3 hidden layer with 10 neurons
    activation='tanh',              # Rectified Linear Unit activation function
    solver='adam',                  # Optimization algorithm
    max_iter=100000,                # Maximum number of iterations
    random_state=42,                # For reproducibility,
    learning_rate_init = 0.001,
    tol = 1e-8
)

# 3. Generate Test Data for Evaluation
num_test_samples = 50
angles_test = np.linspace(-10, 10, num_test_samples).reshape(-1, 1)
sync_values_true = np.sin(angles_test)/angles_test

x_test = np.linspace(-10, 10, num_samples).reshape(-1, 1)
gaus_values_true = np.exp(-x_test**2)

# 4. Train the models and make predictions
model.fit(angles_train, sync_values_train)
sync_values_predicted = model.predict(angles_test)

model.fit(x_train, gaus_values_train)
gaus_values_predicted = model.predict(x_test)

# 5. Visualize sync(x)
plt.figure(figsize=(10, 6))
plt.scatter(angles_train, sync_values_train, label='Dados de treinamento', alpha=0.5)
plt.plot(angles_test, sync_values_true, label='Função sync(x) verdadeira', color='blue')
plt.plot(angles_test, sync_values_predicted, label='Interpolação da Rede Neural', color='red')
plt.xlabel('x')
plt.ylabel('sync(x)')
plt.title('Interpolação da função sync(x)')
plt.legend()
plt.grid(True)
plt.show()

