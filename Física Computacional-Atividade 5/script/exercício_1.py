import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

#Parâmetros
alpha = 10     # Para potencial mais elevado
N_G = 3         # Considera ondas planas de -3 a + 3
k_points = np.linspace(np.pi, -np.pi, 400)
G_vals = np.arange(-N_G, N_G+1)

#Monta o Hamiltoniano tridiagonal e calcula autovalores
bands = []
for k in k_points:
    diagonal = np.array([(k-2*np.pi*n)**2 for n in G_vals])
    off_diag = np.full(2*N_G, alpha)
    eigvals = eigh_tridiagonal(diagonal, off_diag, select='a')[0]
    bands.append(eigvals)

bands = np.array(bands).T

#Plot das primeiras bandas de energia
plt.figure(figsize=(8,6))
for i in range(3): #Mostra as 3 bandas mais baixas
    plt.plot(k_points,bands[i],label=f"Banda{i+1}")

plt.axvline(-np.pi, color='gray',linestyle='--', linewidth=0.8)
plt.axvline(np.pi, color='gray', linestyle='--',linewidth=0.8)
plt.title("Abertura de gap na borda da 1ª zona de Brillouin(α=0.1)")
plt.xlabel("$k$")
plt.ylabel("Energia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()