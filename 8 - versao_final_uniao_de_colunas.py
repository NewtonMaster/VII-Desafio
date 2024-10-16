import pandas as pd

# Carregando os três arquivos CSV
train_a = pd.read_csv('versao_final/solar_wind_train_a_final.csv')
train_b = pd.read_csv('versao_final/solar_wind_train_b_final.csv')
train_c = pd.read_csv('versao_final/solar_wind_train_c_final.csv')

# Concatenando os três DataFrames na ordem desejada (train_a, depois train_b, depois train_c)
solar_wind_completed = pd.concat([train_a, train_b, train_c], ignore_index=True)

# Salvando o novo arquivo combinado
solar_wind_completed.to_csv('teste/solar_wind_completed.csv', index=False)

print("Arquivo solar_wind_completed.csv foi criado com sucesso, combinando train_a, train_b, e train_c.")
