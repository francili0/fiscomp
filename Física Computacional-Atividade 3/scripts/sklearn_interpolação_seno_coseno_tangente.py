import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define as funções matemáticas que queremos que a rede neural aprenda a interpolar
functions = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan
}

# Configurações comuns para geração de dados e plotagem
np.random.seed(42)  # Garante que os resultados aleatórios sejam os mesmos a cada execução
num_samples_train = 100  # Número de pontos para o conjunto de treinamento
num_samples_test = 200   # Número de pontos para o conjunto de teste/visualização
# Gera pontos x de treinamento aleatoriamente no intervalo [0, 2*pi]
x_train_base = np.random.uniform(0, 2 * np.pi, num_samples_train).reshape(-1, 1)
# Gera pontos x de teste uniformemente espaçados no intervalo [0, 2*pi] para uma curva suave
x_test = np.linspace(0, 2 * np.pi, num_samples_test).reshape(-1, 1)

# Cria uma figura para conter os subplots das diferentes funções
plt.figure(figsize=(12, 15)) # Aumentado o tamanho para melhor visualização dos 3 plots

# Itera sobre cada função definida no dicionário 'functions'
for i, (name, func) in enumerate(functions.items(), 1):
    # Gera os valores y de treinamento aplicando a função aos x_train_base
    y_train_base = func(x_train_base)

    # Tratamento especial para a função tangente devido às suas assíntotas
    if name == 'tan':
        # Remove valores muito altos/baixos (próximos às assíntotas) que dificultam o treinamento e a visualização. Define um limite (threshold) de 10.
        y_train_base[np.abs(y_train_base) > 10] = np.nan  # Substitui outliers por NaN (Not a Number)
        # Cria uma máscara para remover os NaNs tanto de x_train quanto de y_train
        mask_train_finite = ~np.isnan(y_train_base).flatten()
        x_train_filtered = x_train_base[mask_train_finite]
        y_train_filtered = y_train_base[mask_train_finite].reshape(-1, 1) # Garante que y_train_filtered seja coluna
    else:
        # Para seno e cosseno, adiciona um ruído gaussiano moderado para simular dados reais
        noise = np.random.normal(0, 0.1, y_train_base.shape) # Média 0, desvio padrão 0.1
        y_train_filtered = y_train_base + noise
        x_train_filtered = x_train_base # Mantém todos os x_train originais

    # Define o modelo de Regressão com Perceptron Multicamadas (MLP)
    model = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),  # 3 camadas ocultas, cada uma com 10 neurônios
        activation='tanh',                # Função de ativação tangente hiperbólica
        solver='adam',                    # Otimizador Adam
        max_iter=100000,                  # Número máximo de iterações (épocas)
        random_state=42,                  # Semente para reprodutibilidade dos pesos iniciais
        learning_rate_init=0.001,         # Taxa de aprendizado inicial
        tol=1e-8,                         # Tolerância para a otimização (critério de parada)
        n_iter_no_change=50               # Número de iterações sem melhora para parada antecipada
    )

    # Treina o modelo com os dados de treinamento filtrados (ou com ruído)
    # É importante que y_train_filtered seja um vetor 1D (ravel) para o MLPRegressor
    model.fit(x_train_filtered, y_train_filtered.ravel())

    # Gera os valores y verdadeiros para o conjunto de teste (sem ruído)
    y_test_true = func(x_test)
    if name == 'tan':
        # Aplica o mesmo tratamento de limite para a tangente nos dados de teste para visualização
        y_test_true[np.abs(y_test_true) > 10] = np.nan

    # Faz as predições com o modelo treinado usando os dados de teste x_test
    y_test_pred = model.predict(x_test)

    # Avalia o desempenho do modelo usando o Erro Quadrático Médio (MSE)
    # É importante calcular o MSE apenas sobre os pontos válidos (não NaN), especialmente para a tangente
    mask_eval_finite = ~np.isnan(y_test_true.flatten())
    mse = mean_squared_error(y_test_true[mask_eval_finite], y_test_pred[mask_eval_finite])
    print(f"Função: {name} - Erro Quadrático Médio (MSE) nos dados de teste: {mse:.5f}")

    # Plotagem dos resultados
    plt.subplot(3, 1, i)  # Cria um subplot para cada função (3 linhas, 1 coluna, i-ésimo plot)
    plt.scatter(x_train_filtered, y_train_filtered, alpha=0.5, label='Dados de Treinamento')
    plt.plot(x_test, y_test_true, label=f'{name}(x) Verdadeira', color='blue', linewidth=2)
    plt.plot(x_test, y_test_pred, label=f'{name}(x) Prevista (Rede Neural)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Aproximação da Função {name}(x) por uma Rede Neural')
    plt.xlabel('x (radianos)')
    plt.ylabel(f'{name}(x)')
    plt.legend()
    plt.grid(True)
    if name == 'tan':
        # Define limites para o eixo y no gráfico da tangente para melhor visualização
        plt.ylim(-10, 10)

plt.tight_layout()  # Ajusta o layout para evitar sobreposição de títulos e legendas
plt.show()          # Mostra o gráfico