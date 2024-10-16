import csv

# Função para copiar uma coluna e colar em um arquivo solar_wind.csv na próxima coluna disponível
def copy_column_to_solar(input_file, solar_file, column_index=0):
    # Leitura do arquivo de input (train_x_ajustado)
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader_input = csv.reader(infile)
        input_data = [row for row in reader_input]  # Carrega todos os dados do arquivo de input

    # Leitura e escrita do arquivo solar_wind
    with open(solar_file, mode='r', newline='', encoding='utf-8') as solar_infile:
        reader_solar = csv.reader(solar_infile)
        solar_data = [row for row in reader_solar]  # Carrega todos os dados do arquivo solar_wind
    
    # Agora abrindo o arquivo solar_wind em modo de escrita para adicionar a coluna
    with open(solar_file, mode='w', newline='', encoding='utf-8') as solar_outfile:
        writer = csv.writer(solar_outfile)
        
        # Copia a coluna de input e cola no final de cada linha de solar_wind
        for i, row in enumerate(solar_data):
            try:
                # Adiciona o dado da coluna de input na próxima coluna disponível
                row.append(input_data[i][column_index])  # Colando a coluna do arquivo train_x_ajustado
            except IndexError:
                # Caso o input tenha menos linhas que o solar_wind
                row.append('')  # Adiciona célula vazia se não houver mais dados no arquivo de input
            writer.writerow(row)

if __name__ == '__main__':
    # Arquivos de entrada ajustados
    input_file_a = 'train_a_ajustado.csv'
    input_file_b = 'train_b_ajustado.csv'
    input_file_c = 'train_c_ajustado.csv'

    # Arquivos de saída solar_wind
    solar_file_a = 'solar_wind_train_a.csv'
    solar_file_b = 'solar_wind_train_b.csv'
    solar_file_c = 'solar_wind_train_c.csv'

    # Copiar a coluna e colar nos arquivos solar_wind
    copy_column_to_solar(input_file_a, solar_file_a)
    copy_column_to_solar(input_file_b, solar_file_b)
    copy_column_to_solar(input_file_c, solar_file_c)
