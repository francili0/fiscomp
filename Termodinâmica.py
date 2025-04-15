import numpy as np
import matplotlib.pyplot as plt

# Constantes
R = 8.314  # constante universal dos gases (J/mol·K)

# Dados do problema
n = 1.0           # mols
T = 200.0         # temperatura em Kelvin
V_i = 10.0        # volume inicial em litros
V_f = 2.0         # volume final em litros

# Convertendo para m³
V_i_m3 = V_i / 1000
V_f_m3 = V_f / 1000

# Calculando o trabalho isotérmico
W = n * R * T * np.log(V_f_m3 / V_i_m3)

print(f"Trabalho realizado sobre o gás: {W:.2f} J")

# Plotando P x V da transformação isotérmica
V = np.linspace(V_f_m3, V_i_m3, 100)
P = (n * R * T) / V

plt.figure(figsize=(10, 5))
plt.plot(V * 1000, P / 1000, label="Isotérmica")  # convertendo para L e kPa
plt.title("Transformação Isotérmica (P x V)")
plt.xlabel("Volume (L)")
plt.ylabel("Pressão (kPa)")
plt.grid(True)
plt.legend()
plt.show()
