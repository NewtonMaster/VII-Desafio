import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

# Função para normalizar dados (usando paralelismo)
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def main():
    # 1. Carregar os dados
    file_path = r'Assembly\TensorFlow\solar_wind_completed.csv'
    df = pd.read_csv(file_path)

    # 2. Descartar as últimas 6 colunas da análise
    df_relevant = df.iloc[:, :-6]  # Exclui as últimas 6 colunas

    # 3. Remover colunas não numéricas
    df_numeric = df_relevant.select_dtypes(include=[float, int])  # Mantém apenas colunas numéricas

    # 4. Identificar valores faltantes (somente nas colunas numéricas)
    missing_cols = df_numeric.columns[df_numeric.isnull().any()]
    df_complete = df_numeric.dropna()  # Dados completos (sem valores faltantes)
    df_missing = df_numeric[df_numeric.isnull().any(axis=1)]  # Dados faltantes (com pelo menos um valor NaN)

    # Saída para dados completos e dados faltantes
    print(f"Número de amostras completas: {df_complete.shape[0]}")
    print(f"Número de amostras com valores faltantes: {df_missing.shape[0]}")

    # Salvar os dados completos e faltantes em CSV
    df_complete.to_csv(r'Assembly\TensorFlow\dados_completos.csv', index=False)
    df_missing.to_csv(r'Assembly\TensorFlow\dados_faltantes.csv', index=False)

    # 5. Separar variáveis independentes (X) e dependentes (y)
    X_complete = df_complete.drop(columns=missing_cols)
    y_complete = df_complete[missing_cols]

    # Verificar se as variáveis X e y não estão vazias
    print(f"Dimensão de X_complete: {X_complete.shape}")
    print(f"Dimensão de y_complete: {y_complete.shape}")

    # 6. Divisão do conjunto de dados
    X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

    # 7. Usar todos os núcleos do processador para normalizar os dados
    with ProcessPoolExecutor(max_workers=8) as executor:
        X_train_scaled = list(executor.map(normalize_data, [X_train]))
        X_test_scaled = list(executor.map(normalize_data, [X_test]))

    # Exibir as dimensões dos conjuntos de treino e teste
    print(f"Dimensão de X_train: {X_train.shape}")
    print(f"Dimensão de X_test: {X_test.shape}")
    print(f"Primeiras 5 linhas de X_train_scaled:\n{X_train_scaled[0][:5]}")

    # Salvar os dados normalizados
    pd.DataFrame(X_train_scaled[0]).to_csv(r'Assembly\TensorFlow\X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled[0]).to_csv(r'Assembly\TensorFlow\X_test_scaled.csv', index=False)

    print("Dados normalizados salvos em 'X_train_scaled.csv' e 'X_test_scaled.csv'.")

if __name__ == '__main__':
    main()
