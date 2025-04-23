import numpy as np
import matplotlib.pyplot as plt

def U1(x):
    return x**2 - 1

def U2(x):
    return (x**2)*(x-1)*(x+1)

def U3(x):
    return (x**2)*(x-1)*(x+1) + (x/4)

def grad_U1(x):
    return 2*x #Gradiente da função U1

def grad_U2(x):
    return 4*(x**3) - 2*x #Gradiente da função U2

def grad_U3(x):
    return 4*x**3 - 2*x + 1/4 #Gradiente da função U3


#Função Gradiente Descendente
def gradiente_descendente(grad, x0, alfa, tol, n):
    xp = [x0] #Pontos que vão ser plotados
    for i in range(n):
        xx = x0 - alfa * grad(x0)
        xp.append(xx) #xp é uma lista
        x0 = xx
        if abs(grad_U1(x0)) < tol: #Se o módulo do gradiente for menor que a tolerância
            #print(i) Quantas iterações
            break
    return(np.array(xp),alfa) #retorna a lista como um vetor, retorna alfa


#Exercício 1
#x0 (posição inicial), alfa(taxa de aprendizado), tol(tolerância), n(iterações)
x1 = np.linspace(-5, 5, 400) #Intervalo da função
#                                          x0  alfa  tol   n
xp1,a1 = gradiente_descendente(grad_U1, 5, 0.1, 0.01, 1000) 
xp2,a2 = gradiente_descendente(grad_U1, 5, 0.8, 0.01, 1000)
#Gráfico 1
plt.plot(x1, U1(x1), label=R'$U_1(x) = x^2 - 1$') #Plota o gráfico
plt.plot(xp1, U1(xp1), 'g--', marker='o', label=f'GD (α = {a1} )')  # 'o' adiciona pontos nos dados
plt.plot(xp2, U1(xp2), 'r--', marker='o', label=f'GD (α = {a2} )')
plt.legend() #fontsize=15
plt.xlabel(R'x')
plt.ylabel(R'$U_1(x)$')
plt.title('Exercício 1')
plt.tight_layout()
#plt.savefig('Exercicio1.png', dpi=300)
plt.show()


#Exercício 2
x2 = np.linspace(-1.5, 1.5, 400) #Intervalo da função
#                                            x0  alfa   tol   n
xp1,a1 = gradiente_descendente(grad_U2, 1.5, 0.01, 0.01, 1000) 
xp2,a2 = gradiente_descendente(grad_U2, 1.5, 0.25, 0.01, 1000)
#Gráfico 2
plt.plot(x2, U2(x2), label=R'$U_2(x) = x^4 - x^2$')
plt.plot(xp1, U2(xp1), 'g--', marker='o', label=f'GD (α = {a1} )')
plt.plot(xp2, U2(xp2), 'r--', marker='o', label=f'GD (α = {a2} )')
plt.legend()
plt.xlabel(R'x')
plt.ylabel(R'$U_2(x)$')
plt.title('Exercício 2')
plt.tight_layout()
#plt.savefig('Exercicio2.png', dpi=300)
plt.show()


#Exercício 3
#                                            x0  alfa   tol   n
xp1,a1 = gradiente_descendente(grad_U3, 1.5, 0.01, 0.01, 1000) 
xp2,a2 = gradiente_descendente(grad_U3, 1.5, 0.25, 0.01, 1000)
#Gráfico 3
plt.plot(x2, U3(x2), label=R'$U_2(x) = x^4 - x^2 + x/4$')
plt.plot(xp1, U3(xp1), 'g--', marker='o', label=f'GD (α = {a1} )')
plt.plot(xp2, U3(xp2), 'r--', marker='o', label=f'GD (α = {a2} )')
plt.legend()
plt.xlabel(R'x')
plt.ylabel(R'$U_3(x)$')
plt.title('Exercício 3')
plt.tight_layout()
#plt.savefig('Exercicio3.png', dpi=300)
plt.show()