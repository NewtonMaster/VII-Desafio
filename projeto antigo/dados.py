import os
import polars as pl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definir PlaidML como backend do Keras para usar a GPU AMD
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# Definir o caminho absoluto para o arquivo CSV de entrada e saída
input_file = r'D:\VS CODE\python\solar_wind.csv'  # Ajuste o caminho do arquivo conforme a sua localização
output_file = r'D:\VS CODE\python\solar_wind_complete.csv'  # O arquivo onde o CSV final será salvo

# Ler o arquivo CSV com Polars
solar_wind_data = pl.read_csv(input_file)

# Exibir as primeiras linhas dos dados para verificação
print(solar_wind_data.head())

# Separar colunas numéricas e colunas não numéricas
numeric_columns = solar_wind_data.select(pl.col(pl.Float64))  # Seleciona apenas as colunas numéricas
non_numeric_columns = solar_wind_data.select(pl.exclude(pl.Float64))  # Seleciona as colunas não numéricas

# Verificar as primeiras linhas das colunas numéricas e não numéricas
print(numeric_columns.head())
print(non_numeric_columns.head())

# Converter colunas numéricas para formato NumPy (necessário para usar scikit-learn)
numeric_data_numpy = numeric_columns.to_numpy()

# Criar o imputador iterativo (múltipla imputação) para as colunas numéricas usando Scikit-learn
imputer = IterativeImputer(max_iter=10, random_state=0)

# Aplicar a imputação múltipla nas colunas numéricas
imputed_data = imputer.fit_transform(numeric_data_numpy)

# Converter de volta para um DataFrame Polars para colunas numéricas imputadas
imputed_numeric_df = pl.DataFrame(imputed_data, schema=numeric_columns.columns)

# Combinar as colunas não numéricas com as colunas numéricas imputadas
complete_solar_wind_data = pl.concat([non_numeric_columns, imputed_numeric_df], how="horizontal")

# Salvar o novo arquivo com os valores faltantes imputados no diretório especificado
complete_solar_wind_data.write_csv(output_file)

# Exibir as primeiras linhas dos dados imputados
print(complete_solar_wind_data.head())

# Agora vamos usar TensorFlow (PlaidML) para treinar um modelo simples com os dados após a imputação

# Converter os dados imputados para um formato NumPy
X = imputed_data  # Dados imputados
y = complete_solar_wind_data['source'].to_numpy()  # Ajuste isso para a coluna de destino correta

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar o modelo de rede neural com TensorFlow (PlaidML backend)
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Ajuste conforme o tipo de classificação (binária ou múltipla)

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Perda: {loss}, Acurácia: {accuracy}")

# Fazer previsões
predictions = model.predict(X_test_scaled)
print("Previsões:", predictions)
