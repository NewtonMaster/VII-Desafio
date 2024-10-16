import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

# Função para normalizar os dados
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Função para criar e treinar o modelo
def train_model(X_train_scaled, X_test_scaled, y_train, y_test, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Saída linear para regressão
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Treinar o modelo com 1 repetição
    history = model.fit(X_train_scaled, y_train, epochs=1, batch_size=32, 
                        validation_data=(X_test_scaled, y_test), verbose=1)
    
    return model

# Função para preencher valores faltantes
def fill_missing_values(df_numeric, model, column):
    missing_values_X = df_numeric[df_numeric[column].isna()].drop(columns=['bx_gse', 'by_gse', 'bz_gse'])
    missing_values_X_scaled = StandardScaler().fit_transform(missing_values_X)
    predictions = model.predict(missing_values_X_scaled)
    
    # Atualizar o DataFrame com os valores preenchidos
    df_numeric.loc[df_numeric[column].isna(), column] = predictions.flatten()

# Função principal
def main():
    # Carregar os dados
    file_path = r'Assembly\TensorFlow\solar_wind_completed.csv'
    df = pd.read_csv(file_path)

    # Remover colunas não numéricas (mantém apenas os dados numéricos)
    df_numeric = df.select_dtypes(include=[float, int])

    # Separar as variáveis independentes e dependentes (para preencher valores faltantes)
    df_to_fill = df_numeric.dropna(subset=['bx_gse', 'by_gse', 'bz_gse'])  # Exclui as linhas com valores completos ausentes
    X_complete = df_to_fill.drop(columns=['bx_gse', 'by_gse', 'bz_gse'])  # Variáveis independentes
    y_complete = df_to_fill[['bx_gse', 'by_gse', 'bz_gse']]  # Variáveis dependentes

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

    # Normalizar os dados
    with ProcessPoolExecutor(max_workers=8) as executor:
        X_train_scaled = list(executor.map(normalize_data, [X_train]))[0]
        X_test_scaled = list(executor.map(normalize_data, [X_test]))[0]

    # Criar e treinar o modelo para cada variável
    for column in ['bx_gse', 'by_gse', 'bz_gse']:
        print(f"Treinando o modelo para {column} com 1 repetição...")
        model = train_model(X_train_scaled, X_test_scaled, y_train[column], y_test[column], X_train_scaled.shape[1])
        fill_missing_values(df_numeric, model, column)

    # Salvar o DataFrame preenchido
    df_numeric.to_csv(r'Assembly\TensorFlow\solar_wind_completed_filled.csv', index=False)

    print("Preenchimento de dados faltantes concluído e salvo.")

if __name__ == '__main__':
    main()
