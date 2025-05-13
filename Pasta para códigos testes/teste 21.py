import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def escolher_funcao_ativacao(nome_funcao):
    """Retorna a função de ativação correspondente ao nome."""
    if nome_funcao.lower() == 'relu':
        return nn.ReLU()
    elif nome_funcao.lower() == 'tanh':
        return nn.Tanh()
    elif nome_funcao.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Função de ativação '{nome_funcao}' não suportada.")

class Net(nn.Module):
    def __init__(self, num_camadas, num_neuronios, funcao_ativacao):
        super(Net, self).__init__()
        layers = []
        # Camada de entrada
        layers.append(nn.Linear(1, num_neuronios))
        layers.append(funcao_ativacao)
        # Camadas ocultas
        for _ in range(num_camadas - 1):
            layers.append(nn.Linear(num_neuronios, num_neuronios))
            layers.append(funcao_ativacao)
        # Camada de saída
        layers.append(nn.Linear(num_neuronios, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def gerar_dados_seno(num_pontos=100):
    x = np.linspace(0, 2 * np.pi, num_pontos)
    y = np.sin(x)
    # Adiciona um pouco de ruído para tornar o aprendizado mais robusto
    y += 0.05 * np.random.randn(num_pontos)
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    return x_tensor, y_tensor

def treinar_modelo(modelo, x_treino, y_treino, epocas=1000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
    historico_perdas = []

    for epoca in range(epocas):
        # Forward pass
        outputs = modelo(x_treino)
        loss = criterion(outputs, y_treino)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Histórico de perdas
        historico_perdas.append(loss.item())

    return historico_perdas

def interpolar_seno_pytorch(num_camadas, num_neuronios, nome_funcao_ativacao, num_epocas=1000):
    # Gerar dados de treinamento
    x_treino, y_treino = gerar_dados_seno(num_pontos=100)
    # Escolher a função de ativação
    try:
        funcao_ativacao = escolher_funcao_ativacao(nome_funcao_ativacao)
    except ValueError as e:
        print(f"Erro: {e}")
        return

    # Criar o modelo da rede neural
    modelo = Net(num_camadas, num_neuronios, funcao_ativacao)
    # Treinar o modelo
    historico_perdas = treinar_modelo(modelo, x_treino, y_treino, epocas=num_epocas)
    # Gerar dados para visualização da interpolação
    x_teste_np = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1).astype(np.float32)
    x_teste = torch.tensor(x_teste_np)
    modelo.eval() # Coloca o modelo em modo de avaliação
    with torch.no_grad(): # Desativa o cálculo de gradientes para a inferência
        y_pred = modelo(x_teste).numpy()
    y_verdadeiro = np.sin(x_teste_np)
  
    # Plotar os resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(x_treino.numpy(), y_treino.numpy(), label='Dados de Treinamento', alpha=0.7)
    plt.plot(x_teste_np, y_verdadeiro, label='Função Seno Verdadeira', color='blue')
    plt.plot(x_teste_np, y_pred, label='Interpolação da Rede Neural', color='red')
    plt.xlabel(rf'$\theta$')
    plt.ylabel(rf'sen($\theta$)')
    plt.title('Interpolação da Função Seno')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    # Plotar a curva de perda
    plt.figure(figsize=(10, 4))
    plt.plot(historico_perdas)
    plt.xlabel('Época')
    plt.ylabel('Perda (MSE)')
    plt.title('Curva de Perda durante o Treinamento')
    plt.grid(True)
    plt.show(block=True)

interpolar_seno_pytorch(3, 10, 'tanh', 1000)