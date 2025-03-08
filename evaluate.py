import csv
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model, estimate_price

def load_data(file_path):
    mileage = []
    price = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return np.array(mileage), np.array(price)

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
    #plt.show()

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
    #plt.show()

def main():
    file_path = "data.csv"
    
    try:
        print("Programa de Avaliação de Precisão do Modelo")
        print("==========================================")
        theta0, theta1, km_min, km_max, price_min, price_max = load_model()
        print("Modelo carregado com sucesso!")
        
 
        mileage, actual_prices = load_data(file_path)
        
        # Previsões para todos os dados
        predicted_prices = np.array([
            estimate_price(km, theta0, theta1, km_min, km_max, price_min, price_max)
            for km in mileage
        ])
        
        # Calcular métricas de precisão
        mae, mse, rmse, r2 = compute_accuracy_metrics(actual_prices, predicted_prices)
        
        print("\nMétricas de Avaliação do Modelo:")
        print(f"Erro Absoluto Médio (MAE): {mae:.2f} €")
        print(f"Erro Quadrático Médio (MSE): {mse:.2f} €²")
        print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} €")
        print(f"Coeficiente de Determinação (R²): {r2:.4f}")
        
 
        plot_residuals(actual_prices, predicted_prices, mileage)
        
 
        plot_predictions_vs_actual(actual_prices, predicted_prices)
        
    except Exception as e:
        print(f"Erro ao avaliar o modelo: {e}")
        print("Executa primeiro o programa training.py.")

if __name__ == "__main__":
    main()
