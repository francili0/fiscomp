import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros
r = 0.005             # constante de resfriamento [1/s]
T_amb = 25            # temperatura ambiente [°C]
T0 = 90               # temperatura inicial do café [°C]
t_span = (0, 5000)    # intervalo de tempo em segundos
t_eval = np.linspace(*t_span, 300)  # pontos de avaliação

# Definindo a EDO
def dTdt(t, T):
    return r * (T_amb - T)

# Solução numérica com RK45 (Runge-Kutta 4/5 de Dormand-Prince)
sol_num = solve_ivp(dTdt, t_span, [T0], method='RK45', t_eval=t_eval)

# Solução analítica
def T_analytical(t):
    return T_amb + (T0 - T_amb) * np.exp(-r * t)

T_exact = T_analytical(t_eval)

# Plotando os resultados
plt.plot(sol_num.t, sol_num.y[0], 'b-', label='Solução Numérica (RK4)')
plt.plot(t_eval, T_exact, 'r--', label='Solução Analítica')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Resfriamento do Café')
plt.legend()
plt.grid(True)
plt.show()
