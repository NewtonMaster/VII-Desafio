import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Definir o caminho para o arquivo CSV
file_path = r'D:\VS CODE\python\solar_wind.csv'

# Ler o arquivo CSV
df = pd.read_csv(file_path)

# Função para salvar cada parte
def salvar_parte(df_part, parte_num):
    df_part.to_csv(rf'D:\VS CODE\python\solar_wind_parte{parte_num}.csv', index=False)

# Dividir o DataFrame em 10 partes iguais
partes = np.array_split(df, 10)

# Paralelizar a tarefa de salvar os arquivos utilizando 8 núcleos
with ProcessPoolExecutor(max_workers=8) as executor:
    for i, parte in enumerate(partes):
        executor.submit(salvar_parte, parte, i + 1)

print("Arquivos divididos e salvos com sucesso!")
