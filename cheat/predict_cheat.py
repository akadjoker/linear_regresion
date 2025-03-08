import numpy as np
import pickle

def load_model(file_path="sklearn_model.pkl"):
    with open(file_path, 'rb') as f:
        params = pickle.load(f)
    
    return params

def predict_price(mileage, params):
 
    mileage_norm = (mileage - params['X_min']) / (params['X_max'] - params['X_min'])
    
 
    price_norm = params['theta0'] + params['theta1'] * mileage_norm
    
 
    price = price_norm * (params['y_max'] - params['y_min']) + params['y_min']
    
    return price

def main():
    try:
 
        params = load_model()
        print("Modelo sklearn carregado com sucesso!")
        
        while True:
       
            km_input = input("\nInforme a quilometragem do carro (ou 'q' para sair): ")
            
            if km_input.lower() == 'q':
                break
            
            try:
                mileage = float(km_input)
                
        
                if mileage < params['X_min'] or mileage > params['X_max']:
                    print(f"Atenção: A quilometragem {mileage} está fora do intervalo de treinamento ({params['X_min']} - {params['X_max']}).")
                    print("A previsão pode não ser precisa :( .")
                
 
                price = predict_price(mileage, params)
                
 
                print(f"Preço estimado pelo sklearn para um carro com {mileage} km: {price:.2f} €")
                
            except ValueError:
                print("Erro: Ei, chuta um valor numérico válido.")
                
    except Exception as e:
        print(f"Erro: {e}")
        print("Verifica ses já tens o bixo treinado usando train_sklearn.py.")

if __name__ == "__main__":
    print("Programa de Previsão com scikit-learn")
    print("=====================================")
    main()