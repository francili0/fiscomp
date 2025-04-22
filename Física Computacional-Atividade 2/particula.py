class Particula:
 import matplotlib.pyplot as plt
import math

class particula():
    def __init__(self, x, y, vx, vy, massa):
        self.x = [x]
        self.y = [y]
        self.vx = [vx]
        self.vy = [vy]
        self.massa = massa

    def newton(self, fx, fy, dt):
        if self.y[-1] >= 0:
            ax = fx / self.massa
            ay = fy / self.massa
            self.x.append(self.x[-1] + self.vx[-1] * dt + 0.5 * ax * dt**2)
            self.y.append(self.y[-1] + self.vy[-1] * dt + 0.5 * ay * dt**2)
            self.vx.append(self.vx[-1] + ax * dt)
            self.vy.append(self.vy[-1] + ay * dt)

# 🚀 Dados iniciais do lançamento
massa = 1  # kg
v0 = 20    # m/s
angulo = 45  # graus
g = 9.81   # m/s²

# Convertendo ângulo para radianos e calculando vx0, vy0
vx0 = v0 * math.cos(math.radians(angulo))
vy0 = v0 * math.sin(math.radians(angulo))

# Criando a partícula
p = particula(0, 0, vx0, vy0, massa)

# ⏱ Parâmetros da simulação
dt = 0.01
tempo_total = 5  # s
t = 0

# 🧮 Loop de simulação
while p.y[-1] >= 0:
    p.newton(0, -massa * g, dt)
    t += dt

# 📈 Gráfico da trajetória
plt.plot(p.x, p.y)
plt.title("Lançamento Oblíquo (com Newton)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.show()
