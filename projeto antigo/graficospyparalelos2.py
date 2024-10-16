import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler

# Ler o arquivo CSV 'solar_wind_complete.csv'
file_path = r'solar_wind_complete.csv'
df = pd.read_csv(file_path)

# Selecionar apenas as colunas numéricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Normalizar as colunas numéricas
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # Aplicar normalização em todas as colunas numéricas

# Diretório de saída para os gráficos
output_dir = r'output_images_solar_wind/'  # Ajuste o caminho conforme necessário
output_txt = r'output_images_solar_wind\analise_resultados.txt'

# Verificar se o diretório existe, caso contrário, criá-lo
os.makedirs(output_dir, exist_ok=True)

# Função para salvar resultados no arquivo de texto
def salvar_resultado_txt(texto):
    with open(output_txt, 'a') as f:
        f.write(texto + '\n')

# Função para gerar histogramas (executa em paralelo)
def gerar_histograma(col):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f'Histograma da {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.savefig(f'{output_dir}histograma_{col}.png')  # Salvando o gráfico como PNG
    plt.close()

    # Salvar dados do histograma no txt
    hist, bin_edges = np.histogram(df[col], bins=30)
    salvar_resultado_txt(f'\nHistograma da coluna {col}:\nBins: {bin_edges}\nFrequências: {hist}\n')

# Função para gerar o gráfico de dispersão
def gerar_grafico_dispersao():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['bx_gse'], y=df['by_gse'], hue=df['bz_gse'], palette='coolwarm')
    plt.title('Gráfico de Dispersão entre bx_gse e by_gse')
    plt.xlabel('bx_gse')
    plt.ylabel('by_gse')
    plt.grid(True)
    plt.savefig(f'{output_dir}dispersao_bx_by.png')  # Salvando o gráfico
    plt.close()

    # Salvar dados do gráfico de dispersão
    salvar_resultado_txt(f'\nGráfico de Dispersão entre bx_gse e by_gse:\nValores de bx_gse: {df["bx_gse"].values[:10]}...\nValores de by_gse: {df["by_gse"].values[:10]}...\n')

# Função para gerar o boxplot
def gerar_boxplot():
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df.iloc[:, :-1])  # Assumindo que a última coluna seja categórica
    plt.title('Boxplot das Variáveis')
    plt.grid(True)
    plt.savefig(f'{output_dir}boxplot_variaveis.png')  # Salvando o gráfico
    plt.close()

    # Salvar dados do boxplot
    estatisticas = df[numeric_cols].describe()
    salvar_resultado_txt(f'\nBoxplot das variáveis (dados estatísticos):\n{estatisticas.to_string()}\n')

# Função para gerar a matriz de correlação
def gerar_matriz_correlacao():
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()  # Matriz de correlação entre as variáveis numéricas
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.savefig(f'{output_dir}matriz_correlacao.png')  # Salvando o gráfico
    plt.close()

    # Salvar a matriz de correlação no arquivo txt
    salvar_resultado_txt(f'\nMatriz de correlação:\n{correlation_matrix.to_string()}\n')

# Função para gerar o gráfico de enxame (swarmplot)
def gerar_swarmplot():
    plt.figure(figsize=(8, 6))
    sns.swarmplot(x='source', y='bx_gse', data=df, palette='coolwarm')  # Exemplo usando a variável 'source'
    plt.title('Gráfico de Enxame (bx_gse por source)')
    plt.grid(True)
    plt.savefig(f'{output_dir}swarmplot.png')  # Salvando o gráfico
    plt.close()

    # Salvar resumo dos dados do gráfico de enxame
    salvar_resultado_txt(f'\nGráfico de Enxame (bx_gse por source):\nValores de bx_gse (amostra): {df["bx_gse"].values[:10]}...\nValores de source: {df["source"].values[:10]}...\n')

# Função para dividir o dataset em 11 conjuntos
def dividir_em_11_conjuntos():
    # Dividindo o DataFrame em 11 partes
    conjuntos = np.array_split(df, 11)

    # Salvando cada conjunto em um CSV separado
    for i, conjunto in enumerate(conjuntos):
        output_conjunto = f'{output_dir}conjunto_{i+1}.csv'
        conjunto.to_csv(output_conjunto, index=False)
        salvar_resultado_txt(f'Conjunto {i+1} salvo como {output_conjunto}')

# Função para gerar estatísticas descritivas
def gerar_estatisticas_descritivas():
    estatisticas = df[numeric_cols].describe().transpose()  # Estatísticas das colunas numéricas
    salvar_resultado_txt(f'Estatísticas descritivas:\n{estatisticas.to_string()}')

# Função principal para executar tudo em paralelo
def main():
    # Apagar o conteúdo anterior do arquivo de texto, se existir
    open(output_txt, 'w').close()

    # Utilizar ProcessPoolExecutor para paralelizar o processamento dos gráficos
    with ProcessPoolExecutor(max_workers=14) as executor:  # Usar todos os 8 núcleos do Ryzen 5 5500U
        # Gerar os histogramas em paralelo
        executor.map(gerar_histograma, numeric_cols)  # Gerar histogramas apenas para colunas numéricas
        
        # Gerar os outros gráficos (em paralelo com o ProcessPoolExecutor)
        executor.submit(gerar_grafico_dispersao)
        executor.submit(gerar_boxplot)
        executor.submit(gerar_matriz_correlacao)
        executor.submit(gerar_swarmplot)

    # Dividir os dados em 11 conjuntos
    dividir_em_11_conjuntos()

    # Gerar estatísticas descritivas
    gerar_estatisticas_descritivas()

if __name__ == "__main__":
    main()
