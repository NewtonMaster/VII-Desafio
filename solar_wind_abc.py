import csv

# Arquivo de entrada e de saída
input_file = 'solar_wind.csv'
output_file_a = 'train_a.csv'
output_file_b = 'train_b.csv'
output_file_c = 'train_c.csv'

# Abrindo o arquivo de entrada
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)

    # Abrindo os arquivos de saída
    with open(output_file_a, mode='w', newline='', encoding='utf-8') as outfile_a, \
         open(output_file_b, mode='w', newline='', encoding='utf-8') as outfile_b, \
         open(output_file_c, mode='w', newline='', encoding='utf-8') as outfile_c:

        writer_a = csv.writer(outfile_a)
        writer_b = csv.writer(outfile_b)
        writer_c = csv.writer(outfile_c)

        # Escrevendo cabeçalhos em todos os arquivos de saída (caso o CSV tenha cabeçalho)
        header = next(reader)  # Pulando o cabeçalho do arquivo de entrada
        writer_a.writerow(header)  # Escrevendo o cabeçalho no arquivo de saída train_a
        writer_b.writerow(header)  # Escrevendo o cabeçalho no arquivo de saída train_b
        writer_c.writerow(header)  # Escrevendo o cabeçalho no arquivo de saída train_c

        # Iterando sobre as linhas do arquivo de entrada
        for row in reader:
            # Verificando se a linha começa com 'train_a', 'train_b' ou 'train_c'
            if row[0].startswith('train_a'):
                writer_a.writerow(row)  # Escrevendo no arquivo de saída train_a
            elif row[0].startswith('train_b'):
                writer_b.writerow(row)  # Escrevendo no arquivo de saída train_b
            elif row[0].startswith('train_c'):
                writer_c.writerow(row)  # Escrevendo no arquivo de saída train_c
