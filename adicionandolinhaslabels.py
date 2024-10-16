import csv

# Função para adicionar as linhas 60 vezes e remover as duas primeiras colunas
def repeat_lines_remove_columns(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Copia o cabeçalho, removendo as duas primeiras colunas
        header = next(reader)[2:]  # Remove as duas primeiras colunas
        writer.writerow(header)

        # Para cada linha no arquivo, remove as duas primeiras colunas e adiciona 60 vezes
        for row in reader:
            writer.writerows([row[2:]] * 60)  # Remove as duas primeiras colunas e adiciona a linha 60 vezes

if __name__ == '__main__':
    # Arquivos de entrada já separados
    input_file_a = 'train_a.csv'
    input_file_b = 'train_b.csv'
    input_file_c = 'train_c.csv'

    # Arquivos de saída com as linhas repetidas e colunas removidas
    output_file_a = 'train_a_ajustado.csv'
    output_file_b = 'train_b_ajustado.csv'
    output_file_c = 'train_c_ajustado.csv'

    # Repetir as linhas 60 vezes para cada arquivo, removendo as duas primeiras colunas
    repeat_lines_remove_columns(input_file_a, output_file_a)
    repeat_lines_remove_columns(input_file_b, output_file_b)
    repeat_lines_remove_columns(input_file_c, output_file_c)
