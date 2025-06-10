import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Cria o caminho completo para o arquivo de dados dentro desse diretório
file_path = os.path.join(script_dir, 'dados_treinamento.npz')

# --- Parâmetros ---
r = 0.005
T_amb = 25
T0 = 95
np.random.seed(42) # Para reprodutibilidade do ruído

# --- Gerar Dados ---
# 10 pontos no intervalo de 0 a 200 segundos
t_dados = np.linspace(0, 200, 10)
def solucao_analitica(t):
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

T_verdadeiro = solucao_analitica(t_dados)

# Adicionar ruído gaussiano com média 0 e desvio padrão 0.5
ruido = np.random.normal(loc=0.0, scale=0.5, size=t_dados.shape)
T_ruidoso = T_verdadeiro + ruido

# --- Salvar dados para uso posterior ---
np.savez(file_path, t_dados=t_dados, T_ruidoso=T_ruidoso)
print(f"Dados de treinamento salvos em: {file_path}")

# --- Visualização ---
t_curva = np.linspace(0, 1000, 500)
T_curva = solucao_analitica(t_curva)

plt.figure(figsize=(10, 6))
plt.plot(t_curva, T_curva, 'k--', label='Solução Analítica (Verdade)')
plt.scatter(t_dados, T_ruidoso, color='red', zorder=5, label='Dados Sintéticos de Treinamento')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Dados Sintéticos Gerados para Treinamento')
plt.legend()
plt.grid(True)
plt.show()