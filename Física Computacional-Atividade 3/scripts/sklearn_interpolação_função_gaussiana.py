import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define a função Gaussiana
def gaussian_function(x, a=1, b=0, c=1):
    """
    Função Gaussiana: f(x) = a * exp(-(x-b)^2 / (2*c^2))
    Para simplificar, usamos a=1, b=0, c=1/sqrt(2) -> exp(-x^2)
    """
    return np.exp(-x**2)

# 1. Geração de Dados de Treinamento
np.random.seed(42)  # Para reprodutibilidade
num_samples_train = 100 # Número de amostras de treinamento
# Gera valores de x aleatórios no intervalo [-10, 10]
# No script original, o intervalo para x_train era [-10, 10], mas a gaussiana é mais pronunciada perto de 0.
# Vamos usar um intervalo menor para concentrar os pontos onde a função varia mais, por exemplo [-5, 5].
x_train = np.random.uniform(-5, 5, num_samples_train).reshape(-1, 1)
# Calcula os valores da função Gaussiana para o treinamento
gaus_values_train = gaussian_function(x_train)

# Adiciona um pouco de ruído aos dados de treinamento
noise_level = 0.01
noise = np.random.normal(0, noise_level, gaus_values_train.shape)
gaus_values_train_noisy = gaus_values_train + noise

# 2. Definição do Modelo MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=(10, 10, 10),  # 3 camadas ocultas com 10 neurônios cada
    activation='tanh',                # Função de ativação tangente hiperbólica
    solver='adam',                    # Algoritmo de otimização Adam
    max_iter=100000,                  # Número máximo de iterações
    random_state=42,                  # Semente para reprodutibilidade
    learning_rate_init=0.001,         # Taxa de aprendizado inicial
    tol=1e-8,                         # Tolerância para convergência
    n_iter_no_change=50               # Número de iterações sem melhora para parada antecipada
)

# 3. Treinamento do Modelo
model.fit(x_train, gaus_values_train_noisy.ravel())

# 4. Geração de Dados de Teste para Avaliação e Visualização
num_samples_test = 200 # Número de amostras de teste para uma curva suave
# Gera valores de x uniformemente espaçados no intervalo onde a função é visualmente interessante, ex: [-5, 5]
# O script original usava [-10,10] para x_test, mas a gaussiana exp(-x^2) é quase zero fora de [-4,4]
x_test = np.linspace(-5, 5, num_samples_test).reshape(-1, 1)
# Calcula os valores verdadeiros da função Gaussiana para o conjunto de teste
gaus_values_true_test = gaussian_function(x_test)

# 5. Predição com o Modelo Treinado
gaus_values_predicted_test = model.predict(x_test)

# 6. Avaliação do Modelo
mse = mean_squared_error(gaus_values_true_test, gaus_values_predicted_test)
print(f"Função: Gaussiana - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}")

# 7. Visualização dos Resultados
plt.figure(figsize=(10, 6))
# Plota os dados de treinamento (com ruído)
plt.scatter(x_train, gaus_values_train_noisy, label='Dados de Treinamento (com ruído)', alpha=0.5, color='orange')
# Plota a função Gaussiana verdadeira
plt.plot(x_test, gaus_values_true_test, label='Função $e^{-x^2}$ Verdadeira', color='blue', linewidth=2)
# Plota a interpolação da rede neural
plt.plot(x_test, gaus_values_predicted_test, label='Interpolação da Rede Neural', color='red', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('$e^{-x^2}$')
plt.title('Interpolação da Função Gaussiana $e^{-x^2}$ com Rede Neural')
plt.legend()
plt.grid(True)
plt.show()