# Classe: define o molde para criar objetos
class Cachorro:
    def __init__(self, nome, raca):
        self.nome = nome  # atributo
        self.raca = raca  # atributo

    def latir(self):  # método
        print(f"{self.nome} diz: Au au!")

    def exibir_dados(self):
        print(f"Nome: {self.nome}, Raça: {self.raca}")

# Criando objetos da classe Cachorro
cachorro1 = Cachorro("Rex", "Labrador")
cachorro2 = Cachorro("Bolt", "Pastor Alemão")

# Usando métodos dos objetos
cachorro1.latir()
cachorro2.exibir_dados()

class Carro:
    def __init__(self, modelo, ano):
        self.modelo = modelo
        self.ano = ano
        self.velocidade = 0  # velocidade inicial

    def acelerar(self):
        self.velocidade += 10
        print(f"{self.modelo} acelerou para {self.velocidade} km/h.")

    def frear(self):
        if self.velocidade >= 10:
            self.velocidade -= 10
        else:
            self.velocidade = 0
        print(f"{self.modelo} freou para {self.velocidade} km/h.")

    def exibir_dados(self):
        print(f"Modelo: {self.modelo}, Ano: {self.ano}, Velocidade: {self.velocidade} km/h")

# Criando um carro
carro1 = Carro("Fusca", 1980)

# Usando os métodos
carro1.exibir_dados()
carro1.acelerar()
carro1.acelerar()
carro1.frear()
carro1.exibir_dados()

