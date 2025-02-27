import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

 
def load_sklearn_model(file_path="sklearn_model.pkl"):
    with open(file_path, 'rb') as f:
        params = pickle.load(f)
    
    return params

 
def load_your_model(file_path="model.txt"):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    return {
        'theta0': float(lines[0].strip()),
        'theta1': float(lines[1].strip()),
        'X_min': float(lines[2].strip()),
        'X_max': float(lines[3].strip()),
        'y_min': float(lines[4].strip()),
        'y_max': float(lines[5].strip())
    }
 
def load_data(file_path="data.csv"):
    mileage = []
    price = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return np.array(mileage), np.array(price)

 
def predict_price(mileage, params):
 
    mileage_norm = (mileage - params['X_min']) / (params['X_max'] - params['X_min'])
    
 
    price_norm = params['theta0'] + params['theta1'] * mileage_norm
    
  
    price = price_norm * (params['y_max'] - params['y_min']) + params['y_min']
    
    return price

def main():
    try:
 
        sklearn_params = load_sklearn_model()
        your_params = load_your_model()
        
        print("Parâmetros dos modelos:")
        print(f"scikit-learn: theta0={sklearn_params['theta0']}, theta1={sklearn_params['theta1']}")
        print(f"Meu modelo: theta0={your_params['theta0']}, theta1={your_params['theta1']}")
        
  
        X, y = load_data()
        
     
        sklearn_predictions = [predict_price(km, sklearn_params) for km in X]
        your_predictions = [predict_price(km, your_params) for km in X]
     
        sklearn_mse = np.mean((y - sklearn_predictions) ** 2)
        your_mse = np.mean((y - your_predictions) ** 2)
        
        print("\nMétricas de erro:")
        print(f"scikit-learn MSE: {sklearn_mse:.2f}")
        print(f"Meu modelo MSE: {your_mse:.2f}")
        
     
        plt.figure(figsize=(12, 8))
        
 
        plt.scatter(X, y, c='blue', label='Dados Originais')
        
 
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        sklearn_pred_sorted = np.array(sklearn_predictions)[sort_idx]
        your_pred_sorted = np.array(your_predictions)[sort_idx]
        
   
        plt.plot(X_sorted, sklearn_pred_sorted, 'r-', linewidth=2, label='scikit-learn')
        plt.plot(X_sorted, your_pred_sorted, 'g--', linewidth=2, label='Sua Implementação')
        
        plt.xlabel('Quilometragem (km)')
        plt.ylabel('Preço (€)')
        plt.title('Comparação dos Modelos de Regressão Linear')
        plt.legend()
        plt.grid(True)
        plt.savefig('model_comparison.png')
 
        
    
        sample_km = [50000, 100000, 150000, 200000]
        print("\nComparação de previsões específicas:")
        for km in sample_km:
            sklearn_price = predict_price(km, sklearn_params)
            your_price = predict_price(km, your_params)
            
            print(f"Quilometragem: {km} km")
            print(f"  scikit-learn: {sklearn_price:.2f} €")
            print(f"  meu modelo: {your_price:.2f} €")
            print(f"  Diferença: {abs(sklearn_price - your_price):.2f} €")
        
    except Exception as e:
        print(f"Erro: {e}")
        print("Verifique se ambos os modelos foram treinados.")

if __name__ == "__main__":
    print("Comparação dos Modelos de Regressão Linear")
    print("==========================================")
    main()