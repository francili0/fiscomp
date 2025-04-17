import matplotlib.pyplot as plt

class Particula:
    def __init__(self, x, y, vx, vy, massa):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.massa = massa

    def newton(self, fx, fy, dt):
        ax = fx / self.massa
        ay = fy / self.massa
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

# Parâmetros iniciais
massa = 1.0  # kg
v0 = 20      # m/s
angulo = 60  # graus
g = 9.8      # m/s²
dt = 0.01    # s

# Componentes da velocidade inicial
from math import radians, cos, sin
v0x = v0 * cos(radians(angulo))
v0y = v0 * sin(radians(angulo))

# Criar partícula
p = Particula(0, 0, v0x, v0y, massa)

# Listas para armazenar os dados
xs = []
ys = []

# Simulação
while p.y >= 0:
    p.newton(fx=0, fy=-massa * g, dt=dt)
    xs.append(p.x)
    ys.append(p.y)

# Plotando a trajetória
plt.plot(xs, ys)
plt.title("Lançamento Oblíquo")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.show()
