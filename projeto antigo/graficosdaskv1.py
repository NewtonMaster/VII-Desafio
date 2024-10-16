import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar matplotlib para usar o backend Agg para maior eficiência
plt.switch_backend('agg')

# Ler o arquivo CSV com Dask para processar grandes volumes de dados de forma eficiente
file_path = r'D:\VS CODE\python\solar_wind_complete.csv'
df = dd.read_csv(file_path)

# Diretório de saída para os gráficos
output_dir = r'D:\VS CODE\output_images_solar_wind/'

# Verificar se o diretório existe, caso contrário, criá-lo
os.makedirs(output_dir, exist_ok=True)

# Carregar apenas um subconjunto de dados para a visualização
df_sample = df.sample(frac=0.125)  # Pegamos 12.5% dos dados (aproximadamente 1 milhão de linhas)

# Converter para pandas dataframe para visualização com Seaborn
df_sample = df_sample.compute()

# =================== Filtrar Apenas Colunas Numéricas ===================
# Manter apenas colunas numéricas para o cálculo da correlação
df_numeric = df_sample.select_dtypes(include=['float64', 'int64'])

# =================== Gerar Histogramas ===================
cols_to_analyze = ['bx_gse', 'by_gse', 'bz_gse', 'speed', 'density']  # Exemplo de colunas

for col in cols_to_analyze:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_sample[col], kde=True, bins=30, color='blue')
    plt.title(f'Histograma da {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.savefig(f'{output_dir}histograma_{col}.png')  # Salvando o gráfico como PNG
    plt.close()

# =================== Gráfico de Dispersão ===================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_sample['bx_gse'], y=df_sample['by_gse'], hue=df_sample['bz_gse'], palette='coolwarm')
plt.title('Gráfico de Dispersão entre bx_gse e by_gse')
plt.xlabel('bx_gse')
plt.ylabel('by_gse')
plt.grid(True)
plt.savefig(f'{output_dir}dispersao_bx_by.png')  # Salvando o gráfico
plt.close()

# =================== Gráfico de Boxplot ===================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_sample[cols_to_analyze])
plt.title('Boxplot das Variáveis')
plt.grid(True)
plt.savefig(f'{output_dir}boxplot_variaveis.png')  # Salvando o gráfico
plt.close()

# =================== Matriz de Correlação ===================
plt.figure(figsize=(10, 8))
correlation_matrix = df_numeric.corr()  # Matriz de correlação apenas com colunas numéricas
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.savefig(f'{output_dir}matriz_correlacao.png')  # Salvando o gráfico
plt.close()
