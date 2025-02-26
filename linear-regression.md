# Regressão Linear com Gradiente Descendente


## Conceitos Fundamentais

### Regressão Linear - Conceito Básico

A regressão linear é um dos algoritmos fundamentais de machine learning. No caso deste projeto, estamos a encontrar uma relação linear entre duas variáveis:
- **Variável independente (X)**: quilometragem do carro (km)
- **Variável dependente (Y)**: preço do carro (price)

A ideia é encontrar uma linha reta que melhor ajusta se aos dados, representada pela equação:

```
price = θ0 + θ1 * km
```

Onde:
- θ0 (theta0) é o intercepto (ponto onde a linha cruza o eixo Y)
- θ1 (theta1) é a inclinação da linha



Leigo
    O total de quilómetros percorridos (kmkm) de um carro influencia o preço.
    Normalmente, carros com mais quilómetros tendem a ser mais baratos, porque têm mais desgaste. 
    Os carros com menos quilómetros costumam ser mais caros,      pois sao mais novos e em melhor estado.
 Como podemos descobrir esta relação?
 A ideia é:
 Variável independente (X) → Número de quilómetros do carro.
 Variável dependente (Y) → Preço do carro.

Queremos encontrar uma linha que melhor represente esta relação, algo como:
Preço=θ0+θ1×KM


### Método do Gradiente Descendente

O gradiente descendente é um algoritmo de otimização que encontra os valores de θ0 e θ1 que minimizam o erro entre as previsões e os valores reais. Queres descobrir como o tamanho de uma casa item nfluencia no preço. Tens uma lista de casas com diferentes tamanhos e preços e queres encontrar uma linha que melhor represente essa relação. Funciona assim:

1. Iniciamos com valores arbitrários para θ0 e θ1 (neste caso, ambos começam em 0)
2. Calculo o erro das previsões usando estes parâmetros
3. Ajusto os parâmetros na direção que reduz o erro
4. Repetir os passos 2-3 até convergir (ou atingir um número máximo de iterações)

Simples
 Começamos com uma linha qualquer, mesmo que esteja errada.
 Calculamos o erro dessa linha ao comparar com os dados reais.
 Ajustamos a linha ligeiramente na direção que reduz o erro.
 Repetimos este processo até encontrarmos a melhor linha.

As fórmulas específicas para atualizar os parâmetros são:

```
tmp_θ0 = θ0 - learning_rate * (1/m) * Σ(previsão[i] - real[i])
tmp_θ1 = θ1 - learning_rate * (1/m) * Σ(previsão[i] - real[i]) * km[i]
```

Onde:
- m é o número de exemplos no conjunto de dados
- Σ representa a soma para todos os exemplos
- learning_rate é um hiperparâmetro que controla o tamanho dos passos do gradiente

### Normalização dos Dados

A normalização é crucial para o gradiente descendente funcionar, especialmente quando as variáveis têm escalas diferentes. Neste caso:

- A quilometragem pode variar de cerca de 20.000 a 240.000 km
- Os preços variam aproximadamente de 3.650 a 8.290 euros

Normalizamos os dados para o intervalo [0, 1] usando a fórmula:

```
x_normalizado = (x - x_min) / (x_max - x_min)
```

Isso ajuda o algoritmo a convergir mais rapidamente e evita problemas numéricos.

## Detalhes da Implementação

### 1. Carregamento e Preparação dos Dados

```python
def load_data(file_path):
    mileage = []
    price = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return np.array(mileage), np.array(price)
```

Este código carrega os dados do arquivo CSV em dois arrays numpy: um para quilometragem e outro para preços.

### 2. Normalização dos Dados

```python
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    
    normalized = (data - min_val) / (max_val - min_val)
    
    return normalized, min_val, max_val
```

Esta função normaliza os dados para o intervalo [0, 1] e retorna também os valores mínimos e máximos para podermos fazer a desnormalização mais tarde.

### 3. Função de Estimativa de Preço

```python
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)
```

Esta é nossa hipótese linear para estimar o preço do  carro com base no km.

### 4. Implementação do Gradiente Descendente

```python
def linear_regression(mileage, price, learning_rate=0.1, iterations=1000): iterations ou epochs "fica mais bonito"
    m = len(mileage)
    theta0 = 0
    theta1 = 0
    
 
    theta_history = []
    cost_history = []
    
    for _ in range(iterations):
 
        tmp_theta0 = 0
        tmp_theta1 = 0
        
        # Calcular os gradientes
        for i in range(m):
            prediction = estimate_price(mileage[i], theta0, theta1)
            error = prediction - price[i]
            
            tmp_theta0 += error
            tmp_theta1 += error * mileage[i]
        
        # Atualizar os valores de theta
        theta0 -= learning_rate * (1/m) * tmp_theta0
        theta1 -= learning_rate * (1/m) * tmp_theta1
        
  
        theta_history.append((theta0, theta1))
        cost_history.append(compute_mse(mileage, price, theta0, theta1))
    
    return theta0, theta1, theta_history, cost_history
```

Este é o coração do algoritmo:

1. Inicializamos θ0 e θ1 com zero
2. Para cada iteração:
   - Calcula as previsões e os erros para cada exemplo
   - Somar os erros (para θ0) e os erros ponderados pela quilometragem (para θ1)
   - Atualiza os parâmetros usando a fórmula do gradiente descendente
   - Armazena o histórico para análise posterior

O detalhe importante aqui é que os gradientes são acumulados para todos os exemplos antes de atualizar os parâmetros. Isso é conhecido como "batch gradient descent", onde consideramos todos os exemplos de uma vez.

Outro ponto crucial é que os parâmetros são atualizados simultaneamente, usando os valores antigos para calcular ambos os novos valores.

### 5. Avaliação do Modelo

Para avaliar a qualidade do modelo, implementamos várias métricas:

- **MSE (Mean Squared Error)**: Média dos quadrados dos erros. É a métrica que o gradiente descendente tenta minimizar.
- **RMSE (Root Mean Squared Error)**: Raiz quadrada do MSE, tem a vantagem de estar na mesma unidade da variável dependente (euros, no nosso caso).
- **MAE (Mean Absolute Error)**: Média dos valores absolutos dos erros.
- **R² (Coeficiente de Determinação)**: Indica quanto da variância na variável dependente é explicada pelo modelo. Varia de 0 a 1, onde 1 indica um ajuste perfeito.

```python
def compute_accuracy_metrics(y_true, y_pred):
    # Erro Absoluto Médio (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Erro Quadrático Médio (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Raiz do Erro Quadrático Médio (RMSE)
    rmse = np.sqrt(mse)
    
    # Coeficiente de Determinação (R²)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return mae, mse, rmse, r2
```

## Visualizações Importantes

1. **Gráfico de Dispersão com Linha de Regressão**: Mostra os dados originais como pontos e a linha de regressão encontrada pelo algoritmo.

```python
def plot_data_and_regression(mileage, price, theta0, theta1, normalized=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, c='blue', label='Dados')
    
    # Gerar pontos para a linha de regressão
    x = np.linspace(min(mileage), max(mileage), 100)
    y = theta0 + theta1 * x
    
    plt.plot(x, y, 'r', label='Linha de Regressão')
    plt.xlabel('Quilometragem' + (' (normalizada)' if normalized else ''))
    plt.ylabel('Preço' + (' (normalizado)' if normalized else ''))
    plt.title('Regressão Linear: Preço x Quilometragem')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_plot.png')
    plt.show()
```

2. **Histórico de Custo**: Mostra como o erro (MSE) diminui ao longo das iterações, permitindo verificar se o algoritmo convergiu.

```python
def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iterações')
    plt.ylabel('Custo (MSE)')
    plt.title('Histórico de Custo durante o Treinamento')
    plt.grid(True)
    plt.savefig('cost_history.png')
    plt.show()
```

3. **Gráfico de Resíduos**: Mostra a diferença entre os valores reais e as previsões (residuais). Um bom modelo teria resíduos distribuídos aleatoriamente em torno de zero.

```python
def plot_residuals(y_true, y_pred, mileage):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico de dispersão dos resíduos
    plt.subplot(2, 1, 1)
    plt.scatter(mileage, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Quilometragem (km)')
    plt.ylabel('Resíduos (€)')
    plt.title('Resíduos vs Quilometragem')
    plt.grid(True)
    
    # Histograma dos resíduos
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Resíduos (€)')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Resíduos')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    plt.show()
```

4. **Previsões vs. Valores Reais**: Um gráfico que compara as previsões com os valores reais. Idealmente, os pontos estariam alinhados em uma linha de 45 graus.

```python
def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred)
    
    # Linha de igualdade perfeita (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Preço Real (€)')
    plt.ylabel('Preço Previsto (€)')
    plt.title('Preço Previsto vs Preço Real')
    plt.grid(True)
    plt.savefig('predictions_vs_actual.png')
    plt.show()
```

## Detalhes Práticos da Implementação

1. **Taxa de Aprendizado (learning_rate)**: Este hiperparâmetro controla o tamanho dos passos do gradiente. Se for muito grande, o algoritmo pode divergir; se for muito pequeno, pode levar muito tempo para convergir. No código, usamos 0.01, que funciona bem para os dados normalizados.

2. **Número de Iterações**: Defeni 10.000 iterações, o que é suficiente para este conjunto de dados. Em conjuntos maiores ou mais complexos, poderia ser necessário aumentar este número.

3. **Salvamento do Modelo**: Após o trenio, os parâmetros θ0, θ1 e os valores usados para normalização são salvos em um arquivo de texto simples, para serem usados posteriormente no programa de previsão.

```python
def save_model(theta0, theta1, km_min, km_max, price_min, price_max, file_path="model.txt"):
    with open(file_path, 'w') as f:
        f.write(f"{theta0}\n{theta1}\n{km_min}\n{km_max}\n{price_min}\n{price_max}")
```

4. **Previsão com Novos Dados**: No programa de previsão, a quilometragem é normalizada usando os mesmos valores do trieno, e então a fórmula de previsão é aplicada. O resultado é desnormalizado para obter o preço em euros.

```python
def estimate_price(mileage, theta0, theta1, km_min, km_max, price_min, price_max):
    # Normalizar o mileage
    mileage_norm = (mileage - km_min) / (km_max - km_min)
    
    # Fazer a previsão normalizada
    price_norm = theta0 + (theta1 * mileage_norm)
    
    # Desnormalizar o preço
    price = price_norm * (price_max - price_min) + price_min
    
    return price
```
##O que são Resíduos?
A distribuição de resíduos é uma ferramenta fundamental para avaliar a qualidade de um modelo de regressão. Vou explicar o que são resíduos, como analisá-los e por que essa análise é crucial. Resíduos são as diferenças entre os valores observados (reais) e os valores previstos pelo modelo: resíduo = valor_real - valor_previsto

No contexto do projeto de previsão de preços de carros com base na quilometragem:

    Valor real: preço real do carro no dataset
    Valor previsto: preço estimado pelo nosso modelo de regressão linear
    Resíduo: diferença entre esses dois valores

Análise da Distribuição de Resíduos
    A análise de resíduos é essencial porque nos diz se o nosso modelo está capturando adequadamente os padrões nos dados. Existem várias formas de visualizar e analisar resíduos.

## Conjuntos de Programas Desenvolvidos

### 1. training.py - Programa de Treinamento
Carrega os dados, treina o modelo usando gradiente descendente e salva os parâmetros.

### 2. predict.py - Programa de Previsão
Carrega os parâmetros treinados e permite ao usuário inserir uma quilometragem para obter uma previsão de preço.

### 3. evaluate.py - Programa de Avaliação
Avalia a precisão do modelo usando várias métricas e gera visualizações para análise.

## Conclusão

A regressão linear com gradiente descendente é um algoritmo fundamental em machine learning. Embora existam métodos analíticos mais eficientes para resolver regressão linear simples (como a equação normal), o gradiente descendente é valioso porque pode ser estendido para problemas mais complexos, como regressão logística e redes neurais.

## Variações do Gradiente Descendente
    Batch Gradient Descent: Usa todas as amostras para calcular o gradiente.
    Stochastic Gradient Descent (SGD): Atualiza os parâmetros a cada amostra.
    Mini-Batch Gradient Descent: Usa pequenos lotes de amostras para atualizar os parâmetros.

## Refrencias
https://stanford.edu/~shervine/teaching/cs-229/
https://mml-book.github.io/
https://didatica.tech/gradiente-descendente-e-regressao-linear/
https://medium.com/@bruno.dorneles/regress%C3%A3o-linear-com-gradiente-descendente-d3420b0b0ff

