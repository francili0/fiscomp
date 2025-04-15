class Particula:
    def __init__(self, x, y, vx, vy, massa):
        """
        Inicializa uma nova partícula.

        Parâmetros:
        x, y     -- posição inicial
        vx, vy   -- velocidade inicial
        massa    -- massa da partícula
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.massa = massa

    def newton(self, fx, fy, dt):
        """
        Aplica a Segunda Lei de Newton para atualizar a velocidade e posição da partícula.

        Parâmetros:
        fx, fy -- forças aplicadas nas direções x e y
        dt     -- intervalo de tempo
        """
        # Aceleração: F = m * a -> a = F / m
        ax = fx / self.massa
        ay = fy / self.massa

        # Atualiza a velocidade: v = v0 + a * dt
        self.vx += ax * dt
        self.vy += ay * dt

        # Atualiza a posição: x = x0 + v * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def __repr__(self):
        return f"Particula(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}, massa={self.massa})"


# ===== EXEMPLO DE USO =====

# Criando uma partícula na posição (0, 0), com velocidade (1, 1) e massa 2
p = Particula(x=0, y=0, vx=10, vy=10, massa=2)

# Aplicando uma força de 4N na direção x e 0N na direção y, por 1 segundo
p.newton(fx=4, fy=0, dt=1)

# Mostrando o estado da partícula após a aplicação da força
print(p)

import matplotlib.pyplot as plt

class Particula:
    def __init__(self, x, y, vx, vy, massa):
        """
        Inicializa uma partícula com posição (x, y), velocidade (vx, vy) e massa.
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.massa = massa

    def newton(self, fx, fy, dt):
        """
        Aplica a segunda lei de Newton para atualizar a velocidade e posição.
        fx, fy: forças aplicadas
        dt: intervalo de tempo
        """
        # Calcula a aceleração
        ax = fx / self.massa
        ay = fy / self.massa

        # Atualiza a velocidade
        self.vx += ax * dt
        self.vy += ay * dt

        # Atualiza a posição
        self.x += self.vx * dt
        self.y += self.vy * dt

    def __repr__(self):
        return f"Particula(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}, massa={self.massa})"

# ===== Simulação da trajetória =====

# Cria a partícula: posição (0,0), velocidade inicial (1,1), massa = 2
p = Particula(x=0, y=0, vx=1, vy=1, massa=2)

# Listas para guardar a trajetória
traj_x = [p.x]
traj_y = [p.y]

# Simula 20 passos de tempo aplicando uma força constante de 4N em x
for _ in range(1000):
    p.newton(fx=4, fy=0, dt=1)  # força constante em x, nenhuma em y
    traj_x.append(p.x)
    traj_y.append(p.y)

# ===== Gráfico da trajetória =====

plt.figure(figsize=(18, 10))
plt.plot(traj_x, traj_y, marker='o', linestyle='-')
plt.title("Trajetória da Partícula")
plt.xlabel("Posição X")
plt.ylabel("Posição Y")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

