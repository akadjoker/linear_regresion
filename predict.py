def estimate_price(mileage, theta0, theta1, km_min, km_max, price_min, price_max):
    # Normalizar o mileage
    mileage_norm = (mileage - km_min) / (km_max - km_min)
    
    # Fazer a previsão normalizada
    price_norm = theta0 + (theta1 * mileage_norm)
    
    # Desnormalizar o preço
    price = price_norm * (price_max - price_min) + price_min
    
    return price

def load_model(file_path="model.txt"):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    theta0 = float(lines[0].strip())
    theta1 = float(lines[1].strip())
    km_min = float(lines[2].strip())
    km_max = float(lines[3].strip())
    price_min = float(lines[4].strip())
    price_max = float(lines[5].strip())
    
    return theta0, theta1, km_min, km_max, price_min, price_max

def main():
    # Carregar os parâmetros do modelo
    try:
        theta0, theta1, km_min, km_max, price_min, price_max = load_model()
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Verifiqua  vque já treinaste o modelo executando o training.py ;) .")
        return
    
 
    while True:
        try:
 
            km_input = input("\Chuta os KM do carro (ou 'q' para sair): ")
            
            if km_input.lower() == 'q':
                break
            
            mileage = float(km_input)
            
            # Verificar se o valor está dentro do intervalo usado no treinamento
            if mileage < km_min or mileage > km_max:
                print(f"Atenção: A quilometragem {mileage} está fora do intervalo de treinamento ({km_min} - {km_max}).")
                print("A previsão pode não ser precisa.")
            
 
            price = estimate_price(mileage, theta0, theta1, km_min, km_max, price_min, price_max)
            
 
            print(f"Preço estimado para um carro com {mileage} km: {price:.2f} €")
            
        except ValueError:
            print("Erro: Por favori, informa um valor numérico válido :) .")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    print("Programa de Previsão de Preço de Carros")
    print("=======================================")
    main()
