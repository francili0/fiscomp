import numpy as np
import matplotlib.pyplot as plt

# Configurações iniciais
tempo = np.linspace(0, 1, 1000)
frequencia_som = 1000  # 1 kHz
amplitude = 1

# Simula bactéria como ponto absorvente parcialmente
class Bacteria:
    def __init__(self, adesao=0.5):
        self.adesao = adesao  # entre 0 e 1, 1 = forte interação com LIG

    def interagir_com_lig(self):
        condutividade = 1 - np.exp(-5 * self.adesao)  # modelo simplificado
        return condutividade

    def absorver_som(self, sinal):
        return sinal * (1 - self.adesao)  # absorve parte da onda sonora

# Emissão de som: onda senoidal
onda_emitida = amplitude * np.sin(2 * np.pi * frequencia_som * tempo)

# Instanciar bactéria e simular interação
bacteria = Bacteria(adesao=0.7)
condutividade = bacteria.interagir_com_lig()
onda_refletida = bacteria.absorver_som(onda_emitida)

# Plotar o resultado
plt.figure(figsize=(10, 5))
plt.plot(tempo, onda_emitida, label='Onda Emitida')
plt.plot(tempo, onda_refletida, label='Onda Após Interação com Bactéria', linestyle='--')
plt.title(f'Interação com LIG | Condutividade simulada: {condutividade:.2f}')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
