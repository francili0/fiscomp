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
a, b = 1, 2
print(a)
print(b)
a = b = 500
print(a)
print(b)

a = 3
b = 5
print('soma', a+b)
print('subtração', b-a)
print('multiplicação', b*a)
print('divisão', b/a)
print('divisão inteira', b//a)
print('resto de divisão', b%a)
print('expoente', b**a)

i = 0
# O bloco abaixo irá rodar repetidamente enquanto o valor de i for menor que 10
while (i < 20):
    print('loop', i)
    i = i + 1
print('final', i)

altair = Person(nome='Altair', idade=32)
print(altair.idade)  # 32
altair.aniversario()
print(altair.idade)  # 33