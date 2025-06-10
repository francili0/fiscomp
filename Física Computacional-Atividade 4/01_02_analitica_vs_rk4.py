import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parâmetros do Problema ---
r = 0.005      # Taxa de resfriamento [1/s]
T_amb = 25     # Temperatura ambiente [°C]
T0 = 95        # Temperatura inicial do café [°C]
t_span = (0, 1000) # Intervalo de tempo para a solução [s]
t_eval = np.linspace(t_span[0], t_span[1], 500) # Pontos para avaliar a solução

# --- Passo 1: Solução Analítica ---
# T(t) = T_amb + (T0 - T_amb) * exp(-r*t)
def solucao_analitica(t, T0, T_amb, r):
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

T_analitica = solucao_analitica(t_eval, T0, T_amb, r)

# --- Passo 2: Solução Numérica (usando solve_ivp, que implementa RK45) ---
# Define a EDO: dT/dt = r * (T_amb - T)
def edo_resfriamento(t, T):
    return r * (T_amb - T)

# Resolve a EDO
sol_numerica = solve_ivp(edo_resfriamento, t_span, [T0], t_eval=t_eval)

# --- Comparação Gráfica ---
plt.figure(figsize=(12, 7))
plt.plot(t_eval, T_analitica, 'k--', linewidth=2, label='Solução Analítica')
plt.plot(sol_numerica.t, sol_numerica.y[0], 'r-', markersize=4, label='Solução Numérica (solve_ivp)')
plt.xlabel('Tempo (s)', fontsize=14)
plt.ylabel('Temperatura (°C)', fontsize=14)
plt.title('Comparação: Solução Analítica vs. Numérica (RK4)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Imprime a diferença máxima para verificação
diferenca_maxima = np.max(np.abs(T_analitica - sol_numerica.y[0]))
print(f"Diferença máxima entre a solução analítica e numérica: {diferenca_maxima:.2e} °C")