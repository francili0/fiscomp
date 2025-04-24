class Particula:
    def __init__(self, x, y, vx, vy, massa):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.massa = massa

    def newton(self, fx, fy, dt):
        # Calcula aceleração
        ax = fx / self.massa
        ay = fy / self.massa

        # Atualiza velocidade
        self.vx += ax * dt
        self.vy += ay * dt

        # Atualiza posição
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Se atingir o solo, define y = 0
        if self.y < 0:
            self.y = 0
