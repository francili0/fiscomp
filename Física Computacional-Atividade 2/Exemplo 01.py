from particula import Particula
import matplotlib.pyplot as plt

# Criar uma partícula com posição inicial (0, 0) e velocidade inicial (10, 10)
#p = Particula(x=0, y=0, vx=10, vy=10, massa=1.0)
p = Particula(x=0, y=0, vx=10, vy=10, massa=1.0)

# Aplicar uma força constante (ex: gravidade) na direção y
# (ex: fx = 0, fy = -9.8 N, que simula a gravidade)
fx = 0
fy = -9.8  # Força gravitacional
dt = 0.1   # Intervalo de tempo
tempos = []
xs = []
ys = []

t = 0.0
while t <= 2.0:  # Simulando por 2 segundos
    tempos.append(t)
    xs.append(p.x)
    ys.append(p.y)

    # Atualizar a posição e velocidade da partícula usando a segunda lei de Newton
    p.newton(fx, fy, dt)
    
    t += dt

# Plotar a trajetória da partícula
plt.plot(xs, ys, marker='o')
plt.title("Trajetória da Partícula")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.show()
