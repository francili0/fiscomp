import numpy as np
import matplotlib.pyplot as plt

# Constantes
R = 8.314
n = 1.0         # mols
T1 = 600        # temperatura quente (K)
T2 = 300        # temperatura fria (K)
gamma = 1.4     # para gás diatômico (como o ar)

# Volumes
V1 = 1.0        # volume inicial (m³)
V2 = 2.0        # após expansão isotérmica
V3 = 4.0        # após expansão adiabática

# Isotérmica (T1): V1 -> V2
V_isot1 = np.linspace(V1, V2, 100)
P_isot1 = (n * R * T1) / V_isot1

# Adiabática (T1 -> T2): V2 -> V3
P2 = P_isot1[-1]
V_adiab1 = np.linspace(V2, V3, 100)
P_adiab1 = P2 * (V2 / V_adiab1)**gamma

# Isotérmica (T2): V3 -> V2
V_isot2 = np.linspace(V3, V2, 100)
P_isot2 = (n * R * T2) / V_isot2

# Adiabática (T2 -> T1): V2 -> V1
P4 = P_isot2[-1]
V_adiab2 = np.linspace(V2, V1, 100)
P_adiab2 = P4 * (V2 / V_adiab2)**gamma

# Plotando o ciclo P x V
plt.figure(figsize=(10, 6))
plt.plot(V_isot1, P_isot1, label='Isotérmica T1 (expansão)', color='blue')
plt.plot(V_adiab1, P_adiab1, label='Adiabática (expansão)', color='orange')
plt.plot(V_isot2, P_isot2, label='Isotérmica T2 (compressão)', color='green')
plt.plot(V_adiab2, P_adiab2, label='Adiabática (compressão)', color='red')

plt.title('Ciclo de Carnot - Diagrama P x V')
plt.xlabel('Volume (m³)')
plt.ylabel('Pressão (Pa)')
plt.grid(True)
plt.legend()
plt.show()

# Eficiência do Ciclo de Carnot
efficiency = 1 - T2 / T1
print(f"Eficiência teórica do Ciclo de Carnot: {efficiency * 100:.2f}%")
