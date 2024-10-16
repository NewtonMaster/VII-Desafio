import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

# Função para normalizar os dados
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Função para criar o modelo de predição
def build_prediction_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Saída linear para regressão
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Função principal para a predição
def main():
    # 1. Carregar os dados preenchidos
    file_path = r'Assembly\9 - TensorFlow&Prediction\solar_wind_completed_filled.csv'
    df = pd.read_csv(file_path)

    # 2. Selecionar as variáveis independentes (features) e a dependente (label) para predição
    X = df.drop(columns=['bx_gse', 'by_gse', 'bz_gse'])  # Excluir variáveis a serem previstas
    y = df[['bx_gse', 'by_gse', 'bz_gse']]  # Variáveis alvo

    # 3. Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Normalizar os dados usando ProcessPoolExecutor para 8 núcleos
    with ProcessPoolExecutor(max_workers=8) as executor:
        X_train_scaled = list(executor.map(normalize_data, [X_train]))[0]
        X_test_scaled = list(executor.map(normalize_data, [X_test]))[0]

    # 5. Criar e treinar o modelo para cada variável (bx_gse, by_gse, bz_gse)
    predictions = {}
    for column in ['bx_gse', 'by_gse', 'bz_gse']:
        print(f"Treinando o modelo para {column}...")

        model = build_prediction_model(X_train_scaled.shape[1])

        # Treinar o modelo para a variável específica
        model.fit(X_train_scaled, y_train[column], epochs=1, batch_size=32, validation_data=(X_test_scaled, y_test[column]), verbose=1)

        # Fazer previsões no conjunto de teste
        predictions[column] = model.predict(X_test_scaled)

        # Exibir algumas previsões
        print(f"Previsões para {column}: {predictions[column][:5].flatten()}")

    # 6. Retornar ou salvar as previsões
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(r'Assembly\9 - TensorFlow&Prediction\predicted_solar_wind.csv', index=False)

    print("Predições concluídas e salvas.")

if __name__ == '__main__':
    main()
