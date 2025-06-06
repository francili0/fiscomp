import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42) # Semente para reprodutibilidade

# Gera dados de funções polinomiais aleatórias e suas derivadas
def generate_data(nx, qtde_funcoes, p_max):
    """
    Gera 'qtde_funcoes' pares de (função, derivada) baseadas em polinômios aleatórios.
    Cada função é avaliada em 'nx' pontos no intervalo [-1, 1].
    'p_max' é o grau máximo dos polinômios gerados.
    """
    # Define o domínio x comum para todas as funções
    x_base = np.linspace(-1, 1, nx).reshape(-1, 1) # Vetor coluna com 'nx' pontos
    
    lista_y = []  # Lista para armazenar os valores das funções (entradas da rede)
    lista_dy = [] # Lista para armazenar os valores das derivadas (saídas da rede)
    
    for _ in range(qtde_funcoes):
        # Gera um polinômio aleatório de grau p, onde 0 <= p <= p_max
        p = np.random.randint(0, p_max + 1) 
        coeffs = np.random.randn(p + 1) # Coeficientes aleatórios para o polinômio

        # Avalia o polinômio e sua derivada nos pontos x_base
        # np.polyval avalia o polinômio dado os coeficientes e os valores de x
        polinomio_vals = np.polyval(coeffs, x_base)
        # np.polyder calcula os coeficientes da derivada do polinômio
        derivada_coeffs = np.polyder(coeffs)
        # Avalia a derivada
        derivada_vals = np.polyval(derivada_coeffs, x_base)

        # Normaliza os valores (para facilitar o aprendizado)
        # A normalização aqui é um pouco peculiar: ambos são normalizados pelo máximo do polinômio.
        # Isso pode ser uma fonte de dificuldade se as magnitudes de y e dy forem muito diferentes.
        # Uma alternativa seria normalizar cada um por seu próprio máximo ou usar StandardScaler.
        max_abs_polinomio = np.max(np.abs(polinomio_vals))
        if max_abs_polinomio < 1e-6: # Evita divisão por zero se o polinômio for quase nulo
            max_abs_polinomio = 1.0

        polinomio_vals_norm = polinomio_vals / max_abs_polinomio
        derivada_vals_norm = derivada_vals / max_abs_polinomio # Normalizado pelo mesmo fator

        # Adiciona ruído (simula dados reais e ajuda na generalização)
        # O ruído é relativo à escala normalizada (que é próxima de 1 para o polinômio)
        noise_y = 0.01 * np.random.randn(len(x_base), 1) # Reduzido o nível de ruído de 0.05 para 0.01
        noise_dy = 0.01 * np.random.randn(len(x_base), 1)

        lista_y.append(polinomio_vals_norm + noise_y)
        lista_dy.append(derivada_vals_norm + noise_dy)

    # Converte as listas de arrays em matrizes numpy
    # Cada linha representa uma função (ou derivada) avaliada nos nx pontos
    y_out = np.hstack(lista_y).T  # Transpõe para ter (qtde_funcoes, nx)
    dy_out = np.hstack(lista_dy).T # Transpõe para ter (qtde_funcoes, nx)
    return y_out, dy_out, x_base # Retorna também x_base para uso posterior

# Parâmetros para geração de dados
num_pontos_por_funcao = 50   # nx: Número de pontos para discretizar cada função
num_funcoes_treino = 10000 # qtde: Quantidade de pares (função, derivada) a gerar
grau_max_polinomio = 10    # p_max: Grau máximo dos polinômios (nome da variável local)

# Gera os dados de treinamento: Y são as funções, dY são as derivadas
# Y será a entrada da rede (X_data), dY será a saída que queremos prever (y_data)
Y_data, dY_data, x_grid = generate_data(nx=num_pontos_por_funcao,
                                        qtde_funcoes=num_funcoes_treino,
                                        p_max=grau_max_polinomio) # CORRIGIDO: pmax -> p_max

# Divide os dados em conjuntos de treino e teste
# X_train/X_test são os valores das funções (polinômios)
# y_train/y_test são os valores das derivadas correspondentes
X_train, X_test, y_train, y_test = train_test_split(Y_data, dY_data, test_size=0.2, random_state=42)

# Define e treina a rede neural
# A arquitetura (10 camadas de 10 neurônios) é bastante profunda.
# Pode ser propensa a overfitting ou dificuldades de treinamento (vanishing/exploding gradients),
# embora 'adam' e 'tanh' ajudem a mitigar isso.
model = MLPRegressor(
    hidden_layer_sizes=(10,) * 10,      # 10 camadas ocultas, cada uma com 10 neurônios
    activation='tanh',                  # Função de ativação tangente hiperbólica
    solver='adam',                      # Otimizador Adam
    max_iter=1000,                      # Aumentado de 100000 (do script original que era muito) para um valor mais razoável para teste inicial. Ajustar conforme necessário.
                                        # O script original tinha 100000, o que pode ser muito longo.
                                        # A configuração n_iter_no_change=50 já ajuda a parar antes se não houver melhora.
    random_state=42,
    learning_rate='adaptiv
print("Treinamento concluído.")

# Avalia o desempenho no conjunto de teste (polinômios)
y_pred_test_polinomios = model.predict(X_test)
mse_polinomios = mean_squared_error(y_test, y_pred_test_polinomios)
print(f"\nMean Squared Error (Erro Quadrático Médio) nos polinômios de teste: {mse_polinomios:.6f}")

# --- Teste com funções conhecidas (que não são polinômios) ---
# O objetivo é ver se a rede generaliza para outros tipos de funções.

# Prepara o x para as funções de teste (mesma grade dos dados de treino)
# x_grid já está no formato (num_pontos_por_funcao, 1)
# new_x_test precisa ser (1, num_pontos_por_funcao) para o predict, pois esperamos uma única função como entrada.
# No entanto, as funções de teste são geradas sobre x_grid.flatten() e depois formatadas.

# A normalização aplicada durante o treinamento (dividir por max|polinômio|)
# precisa ser replicada aqui para as funções de teste.
def normalizar_funcao_para_teste(func_vals, x_base_vals):
    """Normaliza os valores da função da mesma forma que no treinamento."""
    # No treinamento, a normalização foi feita pelo max_abs_polinomio.
    # Para funções de teste, vamos normalizar pelo máximo absoluto da *própria função de teste*.
    # Isso é uma aproximação, já que não temos o "polinômio original" para a função de teste.
    max_abs_func = np.max(np.abs(func_vals))
    if max_abs_func < 1e-6:
        max_abs_func = 1.0
    return func_vals / max_abs_func, max_abs_func


plt.figure(figsize=(15, 5)) # Aumentado o tamanho da figura

# Teste 1: seno
plt.subplot(1, 3, 1)
func_sin = np.sin(2 * np.pi * x_grid.flatten()) # Usando 2*pi para ter um período completo em [-1, 1]
                                            
x_plot = x_grid.flatten() # Para plotagem no eixo x

# Funções e derivadas no intervalo [-1, 1]
func_y_sin = np.sin(np.pi * x_plot) # sin(pi*x) para ter um período em [-1,1]
func_dy_sin_real = np.pi * np.cos(np.pi * x_plot)

# Normaliza a entrada sin(pi*x) como foi feito no treino
func_y_sin_norm, max_abs_sin = normalizar_funcao_para_teste(func_y_sin, x_plot)
# A rede espera entrada com shape (1, num_pontos_por_funcao)
predicted_derivative_norm = model.predict(func_y_sin_norm.reshape(1, -1))
# Desnormaliza a saída
predicted_derivative_sin = predicted_derivative_norm.flatten() * max_abs_sin


plt.plot(x_plot, func_y_sin, label='Entrada: sin(πx)', color='black')
plt.plot(x_plot, func_dy_sin_real, label='Derivada Real (πcos(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_sin, label='Derivada Prevista', color='red', linestyle='dashed', linewidth=2)
plt.title("Teste 1: Derivada de sin(πx)")
plt.ylim(-np.pi - 0.5, np.pi + 0.5) # Ajusta o limite y para a escala da derivada
plt.legend()
plt.grid(True)

# Teste 2: cosseno
plt.subplot(1, 3, 2)
func_y_cos = np.cos(np.pi * x_plot)
func_dy_cos_real = -np.pi * np.sin(np.pi * x_plot)

func_y_cos_norm, max_abs_cos = normalizar_funcao_para_teste(func_y_cos, x_plot)
predicted_derivative_norm = model.predict(func_y_cos_norm.reshape(1, -1))
predicted_derivative_cos = predicted_derivative_norm.flatten() * max_abs_cos

plt.plot(x_plot, func_y_cos, label='Entrada: cos(πx)', color='black')
plt.plot(x_plot, func_dy_cos_real, label='Derivada Real (-πsin(πx))', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_cos, label='Derivada Prevista', color='red', linestyle='dashed', linewidth=2)
plt.title("Teste 2: Derivada de cos(πx)")
plt.ylim(-np.pi - 0.5, np.pi + 0.5)
plt.legend()
plt.grid(True)

# Teste 3: x² (um polinômio simples que pode ou não estar no treino)
plt.subplot(1, 3, 3)
func_y_x2 = x_plot**2
func_dy_x2_real = 2 * x_plot

func_y_x2_norm, max_abs_x2 = normalizar_funcao_para_teste(func_y_x2, x_plot)
predicted_derivative_norm = model.predict(func_y_x2_norm.reshape(1, -1))
predicted_derivative_x2 = predicted_derivative_norm.flatten() * max_abs_x2

plt.plot(x_plot, func_y_x2, label='Entrada: x²', color='black')
plt.plot(x_plot, func_dy_x2_real, label='Derivada Real (2x)', color='blue', linewidth=2)
plt.plot(x_plot, predicted_derivative_x2, label='Derivada Prevista', color='red', linestyle='dashed', linewidth=2)
plt.title("Teste 3: Derivada de x²")
plt.ylim(np.min(func_dy_x2_real) - 0.5, np.max(func_dy_x2_real) + 0.5)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle("Teste da Rede Neural para Calcular Derivadas (Treinada com Polinômios)", fontsize=16, y=1.02)
plt.show()