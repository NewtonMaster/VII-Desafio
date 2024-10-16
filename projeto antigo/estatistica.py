# Importar as bibliotecas necessárias
import polars as pl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy.stats import skew, kurtosis

# Verificar os dispositivos disponíveis no TensorFlow (CPU será usada)
print("Dispositivos disponíveis:", tf.config.list_physical_devices('CPU'))

# Definir o caminho absoluto para o arquivo CSV de entrada e saída
input_file = r'D:\VS CODE\python\solar_wind.csv'  # Ajuste o caminho conforme necessário
output_file = r'D:\VS CODE\python\solar_wind_complete.csv'  # O arquivo onde o CSV final será salvo

# Ler o arquivo CSV com Polars
solar_wind_data = pl.read_csv(input_file)

# Exibir as primeiras linhas dos dados para verificação
print(solar_wind_data.head())

# Separar colunas numéricas e colunas não numéricas
numeric_columns = solar_wind_data.select(pl.col(pl.Float64))  # Seleciona apenas as colunas numéricas
non_numeric_columns = solar_wind_data.select(pl.exclude(pl.Float64))  # Seleciona as colunas não numéricas

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

# Agora vamos treinar um modelo de machine learning com os dados imputados
# Supondo que a coluna 'source' seja a variável alvo
X = imputed_data  # Dados imputados
y = complete_solar_wind_data['source'].to_numpy()  # Ajuste para a coluna alvo correta

# Converter 'y' para valores numéricos usando LabelEncoder (caso seja categórico)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================== ANÁLISES ESTATÍSTICAS ======================

# 1. Média por coluna
mean_values = np.mean(X_train, axis=0)
print("Médias por coluna:", mean_values)

# 2. Mediana por coluna
median_values = np.median(X_train, axis=0)
print("Medianas por coluna:", median_values)

# 3. Variância por coluna
variance_values = np.var(X_train, axis=0)
print("Variâncias por coluna:", variance_values)

# 4. Desvio Padrão por coluna
std_values = np.std(X_train, axis=0)
print("Desvios padrão por coluna:", std_values)

# 5. Quartis e Intervalos Interquartis (IQR)
Q1 = np.percentile(X_train, 25, axis=0)
Q3 = np.percentile(X_train, 75, axis=0)
IQR = Q3 - Q1
print("Intervalo Interquartil (IQR) por coluna:", IQR)

# 6. Kurtose e Assimetria (Skewness)
skewness = skew(X_train, axis=0)
kurt = kurtosis(X_train, axis=0)
print("Assimetria (Skewness) por coluna:", skewness)
print("Kurtose por coluna:", kurt)

# 7. Correlação entre variáveis
correlation_matrix = np.corrcoef(X_train, rowvar=False)
print("Matriz de correlação:", correlation_matrix)

# ====================== NORMALIZAÇÃO E PADRONIZAÇÃO ======================

# 8. Normalização (Min-Max Scaling)
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
print("Dados normalizados (Min-Max):", X_train_normalized)

# 9. Padronização (Standard Scaling)
scaler_standard = StandardScaler()
X_train_standardized = scaler_standard.fit_transform(X_train)
X_test_standardized = scaler_standard.transform(X_test)
print("Dados padronizados (Standard Scaling):", X_train_standardized)

# ====================== TREINAMENTO DO MODELO DE MACHINE LEARNING ======================

# Criar o modelo de rede neural com TensorFlow (usando CPU)
model = Sequential()

# Definir o modelo usando a camada Input para evitar o warning
model.add(Input(shape=(X_train_standardized.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Ajuste conforme o tipo de classificação (binária ou múltipla)

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train_standardized, y_train, epochs=50, batch_size=32, verbose=1)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test_standardized, y_test)
print(f"Perda: {loss}, Acurácia: {accuracy}")

# Fazer previsões
predictions = model.predict(X_test_standardized)
print("Previsões:", predictions)
