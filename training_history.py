
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

def train_with_validation(mileage, price, learning_rate=0.1, epochs=1000, val_split=0.2):
    """
    Treina o modelo com validação e retorna o histórico de custos
    para treinamento e validação.
    
    Parameters:
    -----------
    mileage : array
        Array de quilometragens 
    price : array
        Array de preços
    learning_rate : float
        Taxa de aprendizado
    epochs : int
        Número de épocas de treinamento
    val_split : float
        Proporção de dados a ser usada para validação (0 a 1)
    
    Returns:
    --------
    theta0 : float
        Parâmetro theta0 final
    theta1 : float
        Parâmetro theta1 final
    train_history : list
        Histórico de custos no conjunto de treinamento
    val_history : list
        Histórico de custos no conjunto de validação
    """
    # Embaralhar os dados
    indices = np.arange(len(mileage))
    np.random.shuffle(indices)
    mileage = mileage[indices]
    price = price[indices]
    
    # Dividir em treino e validação
    val_size = int(len(mileage) * val_split)
    train_size = len(mileage) - val_size
    
    mileage_train, mileage_val = mileage[:train_size], mileage[train_size:]
    price_train, price_val = price[:train_size], price[train_size:]
    
    # Inicializar parâmetros
    theta0 = 0
    theta1 = 0
    
    # Históricos para visualização
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        # Calcular os gradientes no conjunto de treinamento
        tmp_theta0 = 0
        tmp_theta1 = 0
        
        for i in range(len(mileage_train)):
            prediction = estimate_price(mileage_train[i], theta0, theta1)
            error = prediction - price_train[i]
            
            tmp_theta0 += error
            tmp_theta1 += error * mileage_train[i]
        
        # Atualizar parâmetros
        theta0 -= learning_rate * (1/len(mileage_train)) * tmp_theta0
        theta1 -= learning_rate * (1/len(mileage_train)) * tmp_theta1
        
        # Calcular custo nos conjuntos de treino e validação
        train_cost = compute_mse(mileage_train, price_train, theta0, theta1)
        val_cost = compute_mse(mileage_val, price_val, theta0, theta1)
        
        train_history.append(train_cost)
        val_history.append(val_cost)
    
    return theta0, theta1, train_history, val_history

def plot_train_val_history(train_history, val_history, save_path=None):
    """
    Plota o histórico de custos de treinamento e validação.
    
    Parameters:
    -----------
    train_history : list
        Histórico de custos no conjunto de treinamento
    val_history : list
        Histórico de custos no conjunto de validação
    save_path : str, optional
        Caminho para salvar o gráfico
    """
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, 'b-', label='Custo de Treinamento')
    plt.plot(epochs, val_history, 'r-', label='Custo de Validação')
    
    plt.title('Histórico de Custo (MSE) durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.legend()
    plt.grid(True)
    

    plt.annotate(f'Treino final: {train_history[-1]:.4f}', 
                xy=(len(train_history), train_history[-1]),
                xytext=(0.8*len(train_history), 1.1*train_history[-1]),
                arrowprops=dict(arrowstyle="->"))
    
    plt.annotate(f'Validação final: {val_history[-1]:.4f}', 
                xy=(len(val_history), val_history[-1]),
                xytext=(0.8*len(val_history), 1.1*val_history[-1]),
                arrowprops=dict(arrowstyle="->"))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_train_val_data_with_model(mileage, price, val_split=0.2, theta0=None, theta1=None, save_path=None):
    """
    Plota os dados de treino e validação junto com a linha de regressão.
    
    Parameters:
    -----------
    mileage : array
        Array de quilometragens
    price : array
        Array de preços
    val_split : float
        Proporção de dados usada para validação
    theta0 : float, optional
        Parâmetro theta0 do modelo
    theta1 : float, optional
        Parâmetro theta1 do modelo
    save_path : str, optional
        Caminho para salvar o gráfico
    """
    # Embaralhar os dados
    indices = np.arange(len(mileage))
    np.random.shuffle(indices)
    mileage_shuffled = mileage[indices]
    price_shuffled = price[indices]
    
    # Dividir em treino e validação
    val_size = int(len(mileage) * val_split)
    train_size = len(mileage) - val_size
    
    mileage_train, mileage_val = mileage_shuffled[:train_size], mileage_shuffled[train_size:]
    price_train, price_val = price_shuffled[:train_size], price_shuffled[train_size:]
    
    plt.figure(figsize=(12, 6))
    
    # Plotar dados de treino e validação
    plt.scatter(mileage_train, price_train, c='blue', alpha=0.6, label='Dados de Treino')
    plt.scatter(mileage_val, price_val, c='red', alpha=0.6, label='Dados de Validação')
    
    # Plotar linha de regressão, se os parâmetros forem fornecidos
    if theta0 is not None and theta1 is not None:
        x = np.linspace(min(mileage), max(mileage), 100)
        y = theta0 + theta1 * x
        plt.plot(x, y, 'g-', label=f'Modelo: y = {theta0:.2f} + {theta1:.6f} * x')
    
    plt.xlabel('Quilometragem (km)')
    plt.ylabel('Preço (€)')
    plt.title('Dados de Treino/Validação e Modelo de Regressão Linear')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    file_path = "data.csv"
    
    # Carregar e normalizar os dados
    mileage, price = load_data(file_path)
    mileage_norm, km_min, km_max = normalize_data(mileage)
    price_norm, price_min, price_max = normalize_data(price)
    
    # Hiperparâmetros
    learning_rate = 0.01
    epochs = 1000
    validation_split = 0.2
    
    print("Treinando o modelo com validação...")
    theta0, theta1, train_history, val_history = train_with_validation(
        mileage_norm, price_norm, learning_rate, epochs, validation_split
    )
    
    print(f"Treinamento concluído! Parâmetros: theta0={theta0}, theta1={theta1}")
    
    # Plotar histórico de treino e validação
    plot_train_val_history(train_history, val_history, "train_val_history.png")
    
    # Plotar dados e modelo
    plot_train_val_data_with_model(
        mileage_norm, price_norm, validation_split, theta0, theta1, "train_val_data.png"
    )
    
    # Adicionar visualização de dados não normalizados com o modelo
    # Precisamos converter theta0 e theta1 para a escala original
    
    # Fórmula para preço original: 
    # price = price_norm * (price_max - price_min) + price_min
    # price_norm = theta0 + theta1 * mileage_norm
    # mileage_norm = (mileage - km_min) / (km_max - km_min)
    
    # Substituindo:
    # price = (theta0 + theta1 * (mileage - km_min) / (km_max - km_min)) * (price_max - price_min) + price_min
    # price = theta0 * (price_max - price_min) + price_min + theta1 * (mileage - km_min) / (km_max - km_min) * (price_max - price_min)
    # price = (theta0 * (price_max - price_min) + price_min) + (theta1 * (price_max - price_min) / (km_max - km_min)) * (mileage - km_min)
    # price = (theta0 * (price_max - price_min) + price_min) + (theta1 * (price_max - price_min) / (km_max - km_min)) * mileage - (theta1 * (price_max - price_min) / (km_max - km_min)) * km_min
    
    # Então, no espaço original:
    # original_theta0 = (theta0 * (price_max - price_min) + price_min) - (theta1 * (price_max - price_min) / (km_max - km_min)) * km_min
    # original_theta1 = theta1 * (price_max - price_min) / (km_max - km_min)
    
    original_theta1 = theta1 * (price_max - price_min) / (km_max - km_min)
    original_theta0 = (theta0 * (price_max - price_min) + price_min) - original_theta1 * km_min
    
    print(f"Parâmetros na escala original: theta0={original_theta0}, theta1={original_theta1}")
    
    # Plotar dados originais e modelo
    plt.figure(figsize=(12, 6))
    plt.scatter(mileage, price, c='blue', alpha=0.6, label='Dados Originais')
    
    x = np.linspace(min(mileage), max(mileage), 100)
    y = original_theta0 + original_theta1 * x
    plt.plot(x, y, 'r-', label=f'Modelo: Preço = {original_theta0:.2f} + {original_theta1:.6f} * km')
    
    plt.xlabel('Quilometragem (km)')
    plt.ylabel('Preço (€)')
    plt.title('Dados Originais e Modelo de Regressão Linear')
    plt.legend()
    plt.grid(True)
    plt.savefig("original_data_model.png")
    plt.show()

if __name__ == "__main__":
    main()