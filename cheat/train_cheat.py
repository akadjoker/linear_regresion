
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
import pickle

def load_data(file_path):
    mileage = []
    price = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return np.array(mileage).reshape(-1, 1), np.array(price)

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    
    normalized = (data - min_val) / (max_val - min_val)
    
    return normalized, min_val, max_val

def main():
 
    file_path = "data.csv"
    X, y = load_data(file_path)
    
 
    X_norm, X_min, X_max = normalize_data(X.flatten())
    y_norm, y_min, y_max = normalize_data(y)
    X_norm = X_norm.reshape(-1, 1)  # Reshape para o formato do sklearn
    
 
    print("Treinando o modelo com scikit-learn...")
    model = LinearRegression()
    model.fit(X_norm, y_norm)
    
 
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    
 
    y_pred = model.predict(X_norm)
    sklearn_mse = mean_squared_error(y_norm, y_pred)
    
    print(f"Parâmetros do modelo: theta0={theta0}, theta1={theta1}")
    print(f"Erro quadrático médio (MSE): {sklearn_mse}")
    
 
    params = {
        'theta0': theta0,
        'theta1': theta1,
        'X_min': X_min,
        'X_max': X_max,
        'y_min': y_min,
        'y_max': y_max
    }
    
    with open('sklearn_model.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    print("Modelo sklearn salvo com sucesso em 'sklearn_model.pkl'")
    
 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_norm, y_norm, c='blue', label='Dados')
    
 
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    y_line = model.predict(x)
    
    plt.plot(x, y_line, 'r', label='Linha de Regressão (sklearn)')
    plt.xlabel('Quilometragem (normalizada)')
    plt.ylabel('Preço (normalizado)')
    plt.title('Regressão Linear com scikit-learn')
    plt.legend()
    plt.grid(True)
    plt.savefig('sklearn_regression.png')
 
    
    # Fazer algumas previsões
    sample_km = [50000, 100000, 150000, 200000]
    print("\nExemplos de previsões:")
    for km in sample_km:
 
        km_norm = (km - X_min) / (X_max - X_min)
        
 
        km_norm_reshaped = np.array([[km_norm]])
        price_norm_pred = model.predict(km_norm_reshaped)[0]
        
 
        price_pred = price_norm_pred * (y_max - y_min) + y_min
        
        print(f"Quilometragem: {km} km => Preço previsto: {price_pred:.2f} €")

if __name__ == "__main__":
    main()