# A função print imprime um texto ou número na saída padrão
print('Hello World')
user_name = input('Enter you name: ')
print('Hello', user_name)
# definindo uma string
variable = 'Gomes'
print(variable)
# definindo um valor inteiro
variable = 42
print(variable)
# definindo um ponto flutuante
variable = 46.5
print(variable)
# definindo um valor Booleano
variable = True
print(variable)
a, b = 1, 2
print(a)
print(b)
a = b = 500
print(a)
print(b)
# para saber o tipo de uma variável, basta chamar a função type()
variable = 46.5
type(variable)
variable = 10
my_variable = 10
var1 = 'Texto'
var2 = "2"
print(var1)
print(var2)
type(var2)
print(type(var2))
var3 = "Citação 'fala'"
print(var3)
var4 = 'Citação "fala"'
print(var4)
var3
var5 = var3 + var4
print(var5)
#var5 - var4
### Tamanho de uma string
len(var5)
# Para acessar um caracter espcífico, devemos fornecer o índice correspondente, lembrando que em python o primeiro índice é o zero.
var5[4]
# Para acessar uma sequencia de caracteres, devemos fazer da seguinte forma:
var5[4:10]
# Para repetir strings basta multiplicar por um número
var6 = 'Hi'
print(var6*10)
# Para substituir caracteres utilizamos o método replace
print(var5.replace('fala', 'texto'))
# Inteiros
x = 1
type(x)
x = 10_000_000_000_000_000_000_000
type(x)
# Números reais ou pontos flutuantes
x = 1.0
type(x)
# Números complexos
y = 1+3j
print(y)
print(type(y))
print(y.real)
print(y.imag)
# Valores Booleanos
a = True
b = False
print(type(a))
print(type(b))
# Typo None, usado para representar algo vazio.
a = None
type(a)
a = 3
b = 5
print('soma', a+b)
print('subtração', b-a)
print('multiplicação', b*a)
print('divisão', b/a)
print('divisão inteira', b//a)
print('resto de divisão', b%a)
print('expoente', b**a)
print(3*0.1)
a = (1/3)**10
b = 3**10
print(a*b)
# Tuplas são sequências de valores ordenados e imutáveis. Permite repetição
#a = (1, 2, 'carro', 1+3j)
#b[1] = 4
#print(b)
# Sets são coleções de valores não ordenados e não indexados. Não permite repetição e é mutável.
c = {1, 2, 5, 1, 'carro'}
print(c)
type(c)
#a['gatos'] = 4
#print(a)
# Função que mostra a identificação única de um objeto
id(user_name)
user_name = 'text'
id(user_name)
a = 5
b = 5
print(id(a), id(b))
a = 500
b = 500
print(id(a), id(b))
a = b = 500
print(id(a), id(b))
a = b = [1, 2]
b[0] = 4
print(b)
print(a)
#if (condition):
    # bloco caso a condição seja verdadeira
    #print('verdadeiro')

if True:
    # O texto dentro deste bloco só será executado caso a condição seja verdadeira.
    # Note a identação (espaços) antes dos comandos abaixo do if.
    # Todos os comandos com a mesma identação estão dentro do mesmo bloco.
    print('Teste verdadeiro')
if False:
    print('Teste verdadeiro')
# Testar se duas variáveis são iguais
a = 3
b = 3
if a==b:
    print('variáveis iguais')
# Testar se duas variáveis são diferentes:
a = 3
b = 4
if a!=b:
    print('variáveis diferentes')
a = 3
b = 3
print(a < b)  # Se a é menor que b
print(a > b)  # Se a é maior que b
print(a <= b)  # Se a é menor ou igual a b
print(a >= b)  # Se a é maior ou igual a b
print(3 < 4)
print(5 > 4)
# O operador "and" resulta verdadeiro se as duas comparações forem verdadeiras
print((3 < 4) and (5 > 4))
print(3 < 4)
print(3 > 5)
# O operador "or" resulta verdadeiro se pelo menos uma das comparações for verdadeira
print((3 < 4) or (3 > 5))
print(3 > 4)
# O operador "not" inverte o resultado da comparação
print(not (3 > 4))
a = 5
b = 6
print(b and a)
print(a & b)
a = 5
b = 5
print(a is b)
a = 500
b = 500
print(a is b)
lista = [1, 6, 'casa', True]
print(2 in lista)
print('casa' in lista)
a = 2
b = 1
if (a > b):
    print('Verdadeiro')
else:
    print('falso')
a = 1
b = 2
if (a > b):
    print('Verdadeiro')
else:
    print('falso')
a = 1
b = 2
if (a > b):
    print('a é maior que b')
elif (a < b):
    print('a é menor que b')
else:
    print('a e b são iguais')

idade = int(input('Digite a sua idade'))
if (idade <= 12):
    print('Você é uma criança')
elif (idade > 12) and (idade < 18):
    print('Você é um adolescente')
else:
    print('Você é um adulto')

a = 'texto'
if a:
    print('verdadeiro')
else:
    print('falso')

i = 0
# O bloco abaixo irá rodar repetidamente enquanto o valor de i for menor que 10
while (i < 20):
    print('loop', i)
    i = i + 2
print('final', i)

for i in range(10):
    print(i)

lista = [1, True, 'casa', 1 + 2j]
for i in lista:
    print(i)

for i in range(10):
    print('antes do break ', i)
    if i == 7:
        break
    print('depois do break', i)

for i in range(10):
    print('antes do continue ', i)
    if i == 7:
        continue
    print('depois do continue', i)

# range é um gerador que fornece um número a cada loop sem ocupar memória.
for i in range(10):
    print(i)

# o enumerate permite retornar um valor de uma lista, assim como o seu índice quando necessário.
lista_cores = ['verde', 'vermelho', 'azul', 'rosa', 'preto']
for i, cor in enumerate(lista_cores):
    print(i, cor)

# o zip serve para retornar valores de combinações de listas 1 a 1:
lista_nomes = ['Altair', 'Carla', 'Sirius', 'Minerva', 'Capitu']
lista_cores = ['cinza', 'azul', 'laranja', 'preta', 'cinza']
for nome, cor in zip(lista_nomes, lista_cores):
    print(nome, '->', cor)

number = int(input('Digite o número do fatorial'))
value = 1
for i in range(1, number+1):
    value = value*i
print('O valor de ', number, 'fatorial é ', value)

def soma():
    a = 1 + 1
    print(a)

def soma():
    a = 1 + 1
    print(a)
soma()

def square():
    a = 3*3
    return a
valor = square()
print(valor)

def coordenadas():
    x = 2
    y = 3
    return x, y
# ao receber os resultados, é possível separá-los em variáveis diferentes.
a, b = coordenadas()
print(a, b)
# se não separar os resultados, a variável final é uma tupla com todos os resultados.
a = coordenadas()
print(a)

# n é um parâmetro da função, que é utilizada dentro da função
def square(n):
    a = n*n
    return a
# 4.5 é o argumento passado para a função. O parâmetro n conterá o argumento 4.5 dentro dele.
valor = square(4.5)
print(valor)

def soma(a, b):
    return a + b
valor = soma(24, 35)
print(valor)

# o parâmetro a é "obrigatório", o parâmetro b é "opcional"
def soma(a, b=55):
    return a + b
soma(14)
soma(14, 86)

def soma(*args):
    print(args)
soma(3, 5, 6)
def agradece_nomes(*args):
    for nome in args:
        print('Obrigado', nome)
agradece_nomes('Altair', 'Carla', 'Sirius', 'Minerva')

def uso_kwargs(**kwargs):
    print(kwargs)
uso_kwargs(a=1, b=2, c=3)
def uso_kwargs(**kwargs):
    print(kwargs['a'])
    print(kwargs['b'])
uso_kwargs(a=1, b=2, c=3)

#def teste():
    #var = 3  # variável que só existe dentro da função
#teste()
#print(var)
#def muda_valor(a):
  #  a = 3
  #  print(a)
#a = 2
#muda_valor(a)
#print(a)

def soma(a):
    print(a + b)
b=10
soma(5)

def muda_lista(lista):
    lista[1] = 3
lista = [0, 1, 6, 8]
muda_lista(lista)
print(lista)

import numpy as np
#%%time
a = 0
for i in range(100_000_000):
    a = a + i
print(a)

# np.arange -> Gera um array numpy que vai de um certo valor até outro, em sequência
a = np.arange(100)
print(a)

b = np.array([1, 5, 7, 6, 10, 15])
print(b)
print(type(b))

def gaussian(x, mu=1, sigma=2):
    a = sigma*np.sqrt(2*np.pi)
    b = np.square((x-mu)/sigma)
    fx = (1/a)*np.exp(-0.5*b)
    return fx
gaussian(0.1)
array = np.arange(-5,5,0.1)
print(array)
gaussian(array)

import matplotlib.pyplot as plt
t = np.arange(100)
y = 1 + 1000*t - 9.8*(t**2)
print(t)
print(x)
plt.plot(t, y)
plt.show()
plt.plot(t, y, 'o')
plt.plot(t, y)
plt.show()
plt.plot(t, y)
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.show()

def densidade_esfera(raio, massa):
    """ Calcula a densidade da esfera
    
    Parameters
    ----------
    raio : float
        Raio da esfera. Deve estar em metros
    massa : float
        Massa da esfera. Deve estar em kg.
    
    Returns
    -------
    densidade : float
        Densidade da esfera em kg/m^3
    """
    volume = (4/3)*np.pi*(raio**3)
    densidade = massa / volume
    return densidade
r = 5 # raio da esfera em metros
m = 125 # massa da esfera em kg

d = densidade_esfera(r, m)
print(d)

# Sugiro importar o units da seguinte forma para facilitar a utilização.
# Porém é importante não definir nenhuma outra variável como u
import astropy.units as np
def densidade_esfera(raio, massa):
    """ Calcula a densidade da esfera
    
    Parameters
    ----------
    raio : float
        Raio da esfera. Default: metros
    massa : float
        Massa da esfera. Default: kg.
    
    Returns
    -------
    densidade : float
        Densidade da esfera
    """
    raio = u.Quantity(raio, 'm') # u.Quantity permite definir um float com uma certa unidade, ou verificar se ele está com uma unidade compatível (m, cm, km, etc).
    massa = u.Quantity(massa, 'kg')
    volume = (4/3)*np.pi*(raio**3)
    densidade = massa / volume
    
    # O método "to" permite converter de uma certa unidade para outra compatível.
    return densidade.to(u.kg/u.m**3)
# para definir uma variável que possui uma unidade definida, fazemos como mostrado abaixo.
r = 5*u.m
m = 125*u.kg

d = densidade_esfera(r, m)
print(d)
# Se eu definir unidades diferentes, o código saberá lidar com isso.
r = 500*u.cm
m = 125000*u.g

d = densidade_esfera(r, m)
print(d)
# sCom essas definições, se eu tentar somar valores de unidades diferentes, ele gera um erro.
print(r + m)
print(m)
print(m.to(u.kg))

import numpy as np
import matplotlib.pyplot as plt
def polinomio(x):
    return 3 + 6*x - x**2
# A função pode ser visualizada fazendo:
x = np.linspace(1,8,20)
y = polinomio(x)
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

start = 1
end = 8
step = 0.01
x1 = np.arange(start, end, 0.01)
x = np.arange(start, end, step) + step/2
#print(x)
plt.bar(x, polinomio(x), step, color='red')
plt.plot(x1, polinomio(x1))
plt.show()
# Realizando a integral
start = 1
end = 8
step = 0.001
x = np.arange(start, end, step) + step/2
y = polinomio(x)
res = np.sum(y*step)
print(res)
# Integração por trapézio
x = np.arange(start, end+step, step)
y = polinomio(x)
res = np.sum((y[1:] + y[:-1])*step/2)
print(res)

altair = Person(nome='Altair', idade=32)
print(altair.idade)  # 32
altair.aniversario()
print(altair.idade)  # 33