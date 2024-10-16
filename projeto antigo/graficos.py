import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo CSV 'solar_wind_complete.csv'
file_path = r'solar_wind_complete.csv'
df = pd.read_csv(file_path)

# Diretório de saída para os gráficos
output_dir = r'output_images_solar_wind/'  # Ajuste o caminho conforme necessário

# Verificar se o diretório existe, caso contrário, criá-lo
os.makedirs(output_dir, exist_ok=True)

# =================== Gerar Histogramas ===================
for col in df.columns[:-1]:  # Excluindo a última coluna se for target ou não numérica
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f'Histograma da {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.savefig(f'{output_dir}histograma_{col}.png')  # Salvando o gráfico como PNG
    plt.close()

# =================== Gráfico de Dispersão ===================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['bx_gse'], y=df['by_gse'], hue=df['bz_gse'], palette='coolwarm')
plt.title('Gráfico de Dispersão entre bx_gse e by_gse')
plt.xlabel('bx_gse')
plt.ylabel('by_gse')
plt.grid(True)
plt.savefig(f'{output_dir}dispersao_bx_by.png')  # Salvando o gráfico
plt.close()

# =================== Gráfico de Boxplot ===================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, :-1])  # Assumindo que a última coluna seja categórica
plt.title('Boxplot das Variáveis')
plt.grid(True)
plt.savefig(f'{output_dir}boxplot_variaveis.png')  # Salvando o gráfico
plt.close()

# =================== Matriz de Correlação ===================
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()  # Matriz de correlação entre todas as variáveis numéricas
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.savefig(f'{output_dir}matriz_correlacao.png')  # Salvando o gráfico
plt.close()
