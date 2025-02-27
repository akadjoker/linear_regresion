import csv
import matplotlib.pyplot as plt
import numpy as np

 
def load_data(file_path):
    mileage = []
    price = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return np.array(mileage), np.array(price)

 
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    
    normalized = (data - min_val) / (max_val - min_val)
    
    return normalized, min_val, max_val

 
def denormalize_data(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val

 
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

 
def compute_mse(mileage, price, theta0, theta1):
    m = len(mileage)
    total_error = 0
    
    for i in range(m):
        prediction = estimate_price(mileage[i], theta0, theta1)
        total_error += (prediction - price[i]) ** 2
    
    return total_error / m


def linear_regression(mileage, price, learning_rate=0.1,  epochs=1000):
    m = len(mileage)
    theta0 = 0
    theta1 = 0
    
    # Histórico para visualização

    cost_history = []
    
    for _ in range(epochs):
 
        tmp_theta0 = 0
        tmp_theta1 = 0
        
        # Calcular os gradientes
        for i in range(m):
            prediction = estimate_price(mileage[i], theta0, theta1)
            error = prediction - price[i]
            
            tmp_theta0 += error
            tmp_theta1 += error * mileage[i]
        
        # valores de theta
        theta0 -= learning_rate * (1/m) * tmp_theta0
        theta1 -= learning_rate * (1/m) * tmp_theta1
        # Nota:  calculando os valores  tmp_theta0 e tmp_theta1 usando os valores atuais de theta0 e theta1, 
        # e só depois atualiza ambos os parâmetros. Isso garante que o gradiente descendente funcione corretamente, 
        # pois os gradientes são calculados com base no mesmo conjunto de parâmetros.
        
 
      
        cost_history.append(compute_mse(mileage, price, theta0, theta1))
    
    return theta0, theta1, cost_history

 
def save_model(theta0, theta1, km_min, km_max, price_min, price_max, file_path="model.txt"):
    with open(file_path, 'w') as f:
        f.write(f"{theta0}\n{theta1}\n{km_min}\n{km_max}\n{price_min}\n{price_max}")

 
def plot_data_and_regression(mileage, price, theta0, theta1, normalized=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, c='blue', label='Dados')
    
 
    x = np.linspace(min(mileage), max(mileage), 100)
    y = theta0 + theta1 * x
    
    plt.plot(x, y, 'r', label='Linha de Regressão')
    plt.xlabel('Quilometragem' + (' (normalizada)' if normalized else ''))
    plt.ylabel('Preço' + (' (normalizado)' if normalized else ''))
    plt.title('Regressão Linear: Preço x Quilometragem')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_plot.png')
    #plt.show()

 
def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iterações')
    plt.ylabel('Custo (MSE)')
    plt.title('Histórico de Custo durante o Treinamento')
    plt.grid(True)
    plt.savefig('cost_history.png')
    #plt.show()

 
def main():
    file_path = "data.csv"   
    
  
    mileage, price = load_data(file_path)
    
    # Normalizar os dados
    mileage_norm, km_min, km_max = normalize_data(mileage)
    price_norm, price_min, price_max = normalize_data(price)
    
    #   hiperparâmetros
    learning_rate = 0.01
    iterations = 10000
    
 
    print("Treinando o modelo... :D ")
    theta0, theta1, cost_history = linear_regression(
        mileage_norm, price_norm, learning_rate, iterations
    )
    
 
    mse = compute_mse(mileage_norm, price_norm, theta0, theta1)
    print(f"Erro quadrático médio (MSE): {mse}")
    
 
    save_model(theta0, theta1, km_min, km_max, price_min, price_max)
    print(f"Modelo salvo com sucesso! Parâmetros: theta0={theta0}, theta1={theta1}")
 
    plot_data_and_regression(mileage_norm, price_norm, theta0, theta1, normalized=True)
    
 
    plot_cost_history(cost_history)

    # só algumas previsões para testar o modelo
    sample_km = [50000, 100000, 150000, 200000]
    print("\nExemplos de previsões:")
    for km in sample_km:
        # Normalizar o km
        km_norm = (km - km_min) / (km_max - km_min)
        
        # Fazer a previsão
        price_norm_pred = estimate_price(km_norm, theta0, theta1)
        
        # Desnorm para obter o preço real
        price_pred = denormalize_data(price_norm_pred, price_min, price_max)
        
        print(f"Quilometragem: {km} km => Preço previsto: {price_pred:.2f} €")

if __name__ == "__main__":
    main()
