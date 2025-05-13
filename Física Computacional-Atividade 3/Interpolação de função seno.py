import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Generate Training Data
np.random.seed(42)  # for reproducibility
num_samples = 100
angles_train = np.random.uniform(0, 2 * np.pi, num_samples).reshape(-1, 1)
tanh_values_train = np.tanh(angles_train)

# Add some noise to the training data
noise = np.random.normal(0, 0.1, tanh_values_train.shape)
tanh_values_train += noise

# 2. Define and Train the FCNN Model
model = MLPRegressor(
    hidden_layer_sizes=(10, 10, 10),
    activation='tanh',
    solver='adam',
    max_iter=100000,
    random_state=42,
    learning_rate_init=0.001,
    tol=1e-8
)

# Train the model
model.fit(angles_train, tanh_values_train)

# 3. Generate Test Data for Evaluation
num_test_samples = 50
angles_test = np.linspace(0, 4 * np.pi, num_test_samples).reshape(-1, 1)
tanh_values_true = np.tanh(angles_test)

# 4. Make Predictions
tanh_values_predicted = model.predict(angles_test)

# 5. Evaluate the Model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(tanh_values_true, tanh_values_predicted)
print(f"Mean Squared Error on Test Data: {mse}")

# 6. Visualize the Results
plt.figure(figsize=(10, 6))
plt.scatter(angles_train, tanh_values_train, label='Training Data', alpha=0.5)
plt.plot(angles_test, tanh_values_true, label='True tanh(theta)', color='blue')
plt.plot(angles_test, tanh_values_predicted, label='Predicted tanh(theta)', color='red')
plt.xlabel('Angle (radians)')
plt.ylabel('tanh(theta)')
plt.title('FCNN Interpolation of tanh(theta)')
plt.legend()
plt.grid(True)
plt.show()
