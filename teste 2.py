# Primeiramente definimos os valores das constantes que serão utilizadas no problema
import numpy as np
import matplotlib.pyplot as plt
g = 9.81
L = 25
m = 40
def euler_method(theta0, w0, dt, n):
    """ Realiza a integração numérica utilizando o método de Euler
    
    Parameters
    ----------
    theta0 : float
        Ângulo inicial    
    w0 : float
        Velocidade Angular inicial
    dt : float
        Passo de tempo
    n : int
        Número de pontos
    """
    # Cria uma lista vazia para salvar os valores de cada passo
    theta = np.zeros(n)
    w = np.zeros(n)
    
    # Defini que o primeiro valor de cada lista é o valor inicial do problema
    theta[0] = theta0
    w[0] = w0
    
    # Fazemos o loop para ir preenchendo cada item da lista utilizando a equação (5)
    for i in range(n-1):
        theta[i+1] = theta[i] + dt*w[i]
        w[i+1] = w[i] + dt*(-np.sin(theta[i])*g/L)
        
    return theta, w
# Vamos definir os parâmetros da integração
theta0 = 60*np.pi/180  # valor inicial do ângulo theta
w0 = 0  # valor inicial da velocidade angular
T = 50  # Instante final da integração
n = 500  # Número de passos da integração
dt = T/n  # Calcula o passo da integração
# Criamos a lista de tempos
t = np.arange(0, T, dt)
# aplicamos o método de Euler com os valores acima
theta, omega = euler_method(theta0, w0, dt, n)
# Fazemos o plot do movimento
plt.plot(t, theta*180/np.pi)
plt.xlabel('Tempo (s)')
plt.ylabel(r'$\theta$ (deg)')


# Primeiramente definimos os valores das constantes que serão utilizadas no problema
import numpy as np
import matplotlib.pyplot as plt
g = 9.81
L = 25
m = 40

def euler_method(theta0, w0, dt, n):
    """ Realiza a integração numérica utilizando o método de Euler
    
    Parameters
    ----------
    theta0 : float
        Ângulo inicial    
    w0 : float
        Velocidade Angular inicial
    dt : float
        Passo de tempo
    n : int
        Número de pontos
    """
    # Cria uma lista vazia para salvar os valores de cada passo
    theta = np.zeros(n)
    w = np.zeros(n)
    
    # Defini que o primeiro valor de cada lista é o valor inicial do problema
    theta[0] = theta0
    w[0] = w0
    
    # Fazemos o loop para ir preenchendo cada item da lista utilizando a equação (5)
    for i in range(n-1):
        theta[i+1] = theta[i] + dt*w[i]
        w[i+1] = w[i] + dt*(-np.sin(theta[i])*g/L)
        
    return theta, w

def euler_method(theta0, w0, dt, n):
    """ Realiza a integração numérica utilizando o método de Euler
    
    Parameters
    ----------
    theta0 : float
        Ângulo inicial    
    w0 : float
        Velocidade Angular inicial
    dt : float
        Passo de tempo
    n : int
        Número de pontos
    """
    # Cria uma lista vazia para salvar os valores de cada passo
    theta = np.zeros(n)
    w = np.zeros(n)
    
    # Defini que o primeiro valor de cada lista é o valor inicial do problema
    theta[0] = theta0
    w[0] = w0
    
    # Fazemos o loop para ir preenchendo cada item da lista utilizando a equação (5)
    for i in range(n-1):
        theta[i+1] = theta[i] + dt*w[i]
        w[i+1] = w[i] + dt*(-np.sin(theta[i])*g/L)
        
    return theta, w

# Criamos a lista de tempos
t = np.arange(0, T, dt)

# aplicamos o método de Euler com os valores acima
theta, omega = euler_method(theta0, w0, dt, n)

# Fazemos o plot do movimento
plt.plot(t, theta*180/np.pi)
plt.xlabel('Tempo (s)')
plt.ylabel(r'$\theta$ (deg)')

n = 50000  # Número de passos da integração
dt = T/n  # Calcula o passo da integração
t = np.arange(0, T, dt)
theta, omega = euler_method(theta0, w0, dt, n)
plt.plot(t, theta*180/np.pi)
plt.xlabel('Tempo (s)')
plt.ylabel(r'$\theta$ (deg)')
plt.show()
# ... (restante do código)
plt.plot(t, theta*180/np.pi)
plt.xlabel('Tempo (s)')
plt.ylabel(r'$\theta$ (deg)')
plt.show()  # <- Essa linha mostra o gráfico

K = (m*(L*omega)**2)/2
U = m*g*L*(1-np.cos(theta))
E = K + U
plt.plot(t, K, label='cinética')
plt.plot(t, U, label='potencial')
plt.plot(t, E, label='total')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.show()

py as np
import matplotlib.pyplot as plt
import num
x=np.linspace (0, 2*np.pi, 100)
y=np.sin(x)


plt.plot(x, y)
plt.show
