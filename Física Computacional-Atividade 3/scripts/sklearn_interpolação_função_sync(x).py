import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define a função sinc
def sinc_function(x):
    # É importante tratar o caso onde x é zero para evitar divisão por zero.
    # np.sinc(x/np.pi) é a versão normalizada. Aqui usamos sin(x)/x diretamente.
    # Para x=0, o limite de sin(x)/x é 1.
    return np.where(x == 0, 1.0, np.sin(x) / x)

# 1. Geração de Dados de Treinamento
np.random.seed(42)  # Para reprodutibilidade
num_samples_train = 100 # Número de amostras de treinamento
# Gera valores de x aleatórios no intervalo [-10, 10]
x_train = np.random.uniform(-10, 10, num_samples_train).reshape(-1, 1)
# Calcula os valores da função sinc para o treinamento
sinc_values_train = sinc_function(x_train)

# Adiciona um pouco de ruído aos dados de treinamento para simular dados reais
noise_level = 0.01
noise = np.random.normal(0, noise_level, sinc_values_train.shape)
sinc_values_train_noisy = sinc_values_train + noise

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
# model.fit espera que y seja um array 1D, por isso o .ravel()
model.fit(x_train, sinc_values_train_noisy.ravel())

# 4. Geração de Dados de Teste para Avaliação e Visualização
num_samples_test = 200 # Número de amostras de teste para uma curva suave
# Gera valores de x uniformemente espaçados no intervalo [-10, 10]
x_test = np.linspace(-10, 10, num_samples_test).reshape(-1, 1)
# Calcula os valores verdadeiros da função sinc para o conjunto de teste
sinc_values_true_test = sinc_function(x_test)

# 5. Predição com o Modelo Treinado
sinc_values_predicted_test = model.predict(x_test)

# 6. Avaliação do Modelo
mse = mean_squared_error(sinc_values_true_test, sinc_values_predicted_test)
print(f"Função: sinc(x) - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}")

# 7. Visualização dos Resultados
plt.figure(figsize=(10, 6))
# Plota os dados de treinamento (com ruído)
plt.scatter(x_train, sinc_values_train_noisy, label='Dados de Treinamento (com ruído)', alpha=0.5, color='orange')
# Plota a função sinc verdadeira
plt.plot(x_test, sinc_values_true_test, label='Função sinc(x) Verdadeira', color='blue', linewidth=2)
# Plota a interpolação da rede neural
plt.plot(x_test, sinc_values_predicted_test, label='Interpolação da Rede Neural', color='red', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('sinc(x)')
plt.title('Interpolação da Função sinc(x) = sin(x)/x com Rede Neural')
plt.legend()
plt.grid(True)
plt.show()