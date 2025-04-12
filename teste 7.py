import numpy as np
import matplotlib.pyplot as plt


x=np.linspace (0, 2*np.pi, 100)
y=np.sin(x)

plt.plot(x, y)
plt.xlabel(r'$\theta$')
plt.ylabel(r'\sin\theta$')
plt.show ()

# Primeiramente definimos os valores das constantes que serão utilizadas no problema
import numpy as np
import matplotlib.pyplot as plt
g = 9.81
L = 5
m = 40

# Vamos definir uma função do método de Euler que sirva para qualquer função que queremos integrar independente do número de derivadas.
def euler(func, v0, step, n):
    """ Realiza a integração numérica de uma função utilizando o método de Euler de primeira ordem
    
    Parameters
    ----------
    func : function
        A função que calcula as derivadas que serão utilizadas na integração
    v0 : list
        A lista de condições iniciais do problema
    step : float
        O passo da integração. Será o mesmo para todas as derivadas
    n : int
        O número de pontos que serão gerados durante a integração
    """
    values = np.zeros((n, len(v0)))  # Cria uma tabela de n linhas e o número de colunas equivalente ao número de condições iniciais v0
    values[0] = np.array(v0)  # Define que a primeira linha da tabela são as próprias condições iniciais do problema.
    # Faz o loop para preencher as linhas seguintes
    for i in range(n-1):
        # Em cada linha aplicamos o método de Euler
        values[i+1] = values[i] + step*np.array(func(*values[i]))
    return values

# Definimos a função que calcula as derivadas do movimento do pêndulo
def pendulo(theta, w):
    """ Calcula as derivadas do movimento do pêndulo: Eq. 4 da Aula 12
    
    Parameters
    ----------
    theta : float
        O ângulo inicial, em radianos
    w : float
        A velocidade angular inicial, em radianos por segundo
    """
    dthetadt = w  # A derivada de theta em relação ao tempo
    dwdt = -(g/L)*np.sin(theta)  # A derivada de omega em relação ao tempo
    return dthetadt, dwdt

# Fazemos um teste
T = 50  # Instante final da integração
n = 50000  # Número de passos da integração
dt = T/n  # Calcula o passo da integração
t = np.arange(0, T, dt)

v0 = [60*np.pi/180, 0]  # theta0 e omega0
resultado = euler(pendulo, v0, dt, n)

theta = resultado[:,0]  # pega a primeira coluna da tabela
omega = resultado[:,1]  # pega a segunda coluna da tabela

plt.plot(t, theta*180/np.pi)
plt.show ()


# Primeiramente definimos os valores das constantes que serão utilizadas no problema
import numpy as np
import matplotlib.pyplot as plt
g = 9.81  # m/s²
L = 5  # m
m = 40  #kg
b = 5  # N.s/m

# Vamos definir uma função do método de Euler que sirva para qualquer função que queremos integrar independente do número de derivadas.
def euler(func, v0, step, n):
    """ Realiza a integração numérica de uma função utilizando o método de Euler de primeira ordem
    
    Parameters
    ----------
    func : function
        A função que calcula as derivadas que serão utilizadas na integração.
        Deve receber n valores, com n igual ao número de condições iniciais, e retornar n valores
    v0 : list
        Uma lista com as n condições iniciais do problema
    step : float
        O passo da integração. Será o mesmo para todas as derivadas
    n : int
        O número de pontos que serão gerados durante a integração
    """
    values = np.zeros((n, len(v0)))  # Cria uma tabela de n linhas e o número de colunas equivalente ao número de condições iniciais v0
    values[0] = np.array(v0)  # Define que a primeira linha da tabela são as próprias condições iniciais do problema.
    # Faz o loop para preencher as linhas seguintes
    for i in range(n-1):
        # Em cada linha aplicamos o método de Euler
        values[i+1] = values[i] + step*np.array(func(*values[i]))
    return values

# Definimos a função que calcula as derivadas do movimento do pêndulo
def pendulo(theta, w):
    """ Calcula as derivadas do movimento do pêndulo com resistência do ar: Eq. 4 da Aula 12
    
    Parameters
    ----------
    theta : float
        O ângulo inicial, em radianos
    w : float
        A velocidade angular inicial, em radianos por segundo
    """
    dthetadt = w  # A derivada de theta em relação ao tempo
    dwdt = -(g/L)*np.sin(theta) - (b/m)*w  # A derivada de omega em relação ao tempo considerando resistência do ar
    return dthetadt, dwdt

# Fazemos um teste
T = 50  # Instante final da integração
n = 50000  # Número de passos da integração
dt = T/n  # Calcula o passo da integração
t = np.arange(0, T, dt)

v0 = [60*np.pi/180, 0]  # theta0 e omega0
resultado = euler(pendulo, v0, dt, n)

theta = resultado[:,0]  # pega a primeira coluna da tabela
omega = resultado[:,1]  # pega a segunda coluna da tabela

plt.plot(t, theta*180/np.pi)
plt.show ()

K = (m*(L*omega)**2)/2
U = m*g*L*(1-np.cos(theta))
E = K + U

plt.plot(t, K, label='cinética')
plt.plot(t, U, label='potencial')
plt.plot(t, E, label='mecânica')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.show ()

K = (m*(L*omega)**2)/2
U = m*g*L*(1-np.cos(theta))
E = K + U - W
plt.plot(t, K, label='cinética')
plt.plot(t, U, label='potencial')
plt.plot(t, -W, label='Trabalho')
plt.plot(t, E, label='total')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.show ()