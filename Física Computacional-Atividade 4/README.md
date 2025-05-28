# Física Computacional - Atividade 4

## Redes Neurais Informadas por Física (PINNs)

### O que são PINNs?

As PINNs (do inglês *Physics-Informed Neural Networks*) são um tipo especial de rede neural que usa as leis da física no seu aprendizado, e não apenas dados como uma rede tradicional.

### Para que servem?

Elas são usadas principalmente para resolver equações da física, como as equações diferenciais que aparecem em muitos problemas (ex: calor, movimento, fluidos, etc.). Mesmo quando a solução exata da equação é desconhecida, a PINN pode aprender a resolvê-la.

### Como a PINN aprende?

Ela aprende minimizando um erro, que junta três partes:

1. **Erro dos dados**: compara a resposta da rede com dados reais (como medições ou simulações).  
2. **Erro da física**: garante que a resposta respeite as equações da física.  
3. **Erro nas condições de contorno**: garante que a solução funcione com as condições iniciais e de borda do problema.

Esses três erros viram uma função só (chamada de **função de perda**), e a rede aprende tentando minimizar esse erro total.

### Vantagem principal

Diferente das redes normais, que só “copiam os dados”, a PINN entende como o sistema funciona. Por isso, ela consegue até prever coisas fora dos dados, se a física permitir.

### Exemplo prático

Imagine que você tem dados de um oscilador (tipo uma mola), mas não sabe a frequência ou o atrito. A PINN pode:

1. Aprender a resolver a equação do oscilador;  
2. E descobrir quais são os parâmetros físicos (frequência, atrito) que melhor explicam os dados.


