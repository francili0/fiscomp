import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do problema
r = 0.005  # taxa de resfriamento
T_amb = 25  # temperatura ambiente (°C)
T0 = 90     # temperatura inicial do café (°C)
t0 = 0
tf = 2000
h = 10  # passo para RK4 manual
t_eval = np.arange(t0, tf + h, h)

# Função da EDO
def f(t, T):
    return r * (T_amb - T)

# ---------- Método RK4 manual ----------
def rk4_manual(f, t0, T0, tf, h):
    t_values = [t0]
    T_values = [T0]
    t = t0
    T = T0
    while t < tf:
        k1 = h * f(t, T)
        k2 = h * f(t + h/2, T + k1/2)
        k3 = h * f(t + h/2, T + k2/2)
        k4 = h * f(t + h, T + k3)
        T += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        t_values.append(t)
        T_values.append(T)
    return np.array(t_values), np.array(T_values)

# ---------- Método solve_ivp ----------
sol = solve_ivp(f, [t0, tf], [T0], t_eval=t_eval, method='RK45')

# ---------- Solução analítica ----------
T_exact = (T0 - T_amb) * np.exp(-r * t_eval) + T_amb

# ---------- Solução com RK4 manual ----------
t_rk4, T_rk4 = rk4_manual(f, t0, T0, tf, h)

# ---------- Plot ----------
plt.figure(figsize=(10, 6))
plt.plot(t_eval, T_exact, 'k--', label='Solução analítica')
plt.plot(sol.t, sol.y[0], 'g-', label='solve_ivp (RK45)')
plt.plot(t_rk4, T_rk4, 'bo', label='RK4 manual', markersize=4)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Comparação: RK4 Manual vs solve_ivp vs Analítica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
