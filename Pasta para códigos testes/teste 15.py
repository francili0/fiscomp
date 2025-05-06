import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Gerar os dados: y = sen(x)
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# 2. Converter os dados para tensores
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. Definir a rede neural com ativação tanh
class SimpleTanhNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)    # Camada escondida com 10 neurônios
        self.fc2 = nn.Linear(10, 1)    # Camada de saída com 1 neurônio

    def forward(self, x):
        x = torch.tanh(self.fc1(x))   # Ativação tanh após primeira camada
        x = self.fc2(x)               # Saída sem ativação
        return x

# 4. Criar o modelo, função de perda e otimizador
model = SimpleTanhNN()
criterion = nn.MSELoss()                         # Erro quadrático médio
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Treinar a rede
for epoch in range(1000):
    y_pred = model(X_tensor)                     # Previsão
    loss = criterion(y_pred, y_tensor)           # Cálculo da perda

    optimizer.zero_grad()                        # Zera gradientes anteriores
    loss.backward()                              # Calcula os novos gradientes
    optimizer.step()                             # Atualiza os pesos

# 6. Fazer previsões e visualizar resultado
with torch.no_grad():                            # Desliga gradientes para avaliação
    predictions = model(X_tensor).numpy()

# 7. Plotar o resultado
plt.plot(X, y, label='sen(x)', color='red')
plt.plot(X, predictions, label='Previsão da rede', color='blue')
plt.legend()
plt.title('Rede Neural Simples com ativação tanh')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
