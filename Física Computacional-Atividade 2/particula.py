class Particula:
    def __init__(self, x, y, vx, vy, massa):
        # Atributos iniciais da partícula
        self.x = x        # posição x
        self.y = y        # posição y
        self.vx = vx      # velocidade x
        self.vy = vy      # velocidade y
        self.massa = massa  # massa da partícula

    def newton(self, fx, fy, dt):
        """Aplica a segunda lei de Newton para atualizar a velocidade e a posição da partícula"""
        # Aceleração = Força / Massa (Lei de Newton)
        ax = fx / self.massa
        ay = fy / self.massa

        # Atualizar a velocidade
        self.vx += ax * dt
        self.vy += ay * dt

        # Atualizar a posição
        self.x += self.vx * dt
        self.y += self.vy * dt

