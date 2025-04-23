from particula import Particula
import matplotlib.pyplot as plt

# Criar uma partícula com posição (0, 0) e velocidade inicial (10, 10)
p = Particula(x=0, y=0, vx=10, vy=10, massa=1.0)

# Força da gravidade (ex: fx = 0, fy = -9.8 N)
fx = 0
fy = -9.8
dt = 0.1  # intervalo de tempo

tempos = []
xs = []
ys = []

t = 0.0
while True:
    tempos.append(t)
    xs.append(p.x)
    ys.append(p.y)

    p.newton(fx, fy, dt)
    t += dt

    # Parar se a partícula atinge o solo
    if p.y <= 0:
        break

# Plotar trajetória
plt.plot(xs, ys, marker='o')
plt.title("Trajetória da Partícula (até atingir o solo)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.show()
