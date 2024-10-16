import pandas as pd

# Carregando os arquivos CSV
solar_wind_train_a = pd.read_csv('teste/solar_wind_train_a.csv')
solar_wind_train_b = pd.read_csv('teste/solar_wind_train_b.csv')
solar_wind_train_c = pd.read_csv('teste/solar_wind_train_c.csv')
sunspot_a = pd.read_csv('teste/sunspot_a.csv')
sunspot_b = pd.read_csv('teste/sunspot_b.csv')
sunspot_c = pd.read_csv('teste/sunspot_c.csv')

# Função para processar intervalos
def process_solar_wind_intervals(solar_wind, sunspot, output_file):
    # Extraindo os dias
    solar_wind['days'] = solar_wind['timedelta'].str.extract(r'(\d+)').astype(int)
    sunspot['days'] = sunspot['timedelta'].str.extract(r'(\d+)').astype(int)
    
    # Criando uma coluna para o smoothed_ssn no solar_wind
    solar_wind['smoothed_ssn'] = None

    # Percorrer os valores de sunspot e preencher para os intervalos correspondentes
    for i in range(len(sunspot) - 1):
        start_day = sunspot['days'].iloc[i]
        end_day = sunspot['days'].iloc[i + 1]
        ssn_value = sunspot['smoothed_ssn'].iloc[i]
        
        # Preencher o intervalo no solar_wind para os dias entre start_day e end_day
        mask = (solar_wind['days'] >= start_day) & (solar_wind['days'] < end_day)
        solar_wind.loc[mask, 'smoothed_ssn'] = ssn_value

    # Preencher para o último valor do sunspot (após o último intervalo)
    last_day = sunspot['days'].iloc[-1]
    last_ssn_value = sunspot['smoothed_ssn'].iloc[-1]
    solar_wind.loc[solar_wind['days'] >= last_day, 'smoothed_ssn'] = last_ssn_value

    # Salvar o arquivo atualizado
    solar_wind.to_csv(output_file, index=False)
    print(f"Arquivo {output_file} foi salvo com os intervalos de smoothed_ssn preenchidos.")

# Processando cada conjunto de arquivos solar_wind e sunspot
process_solar_wind_intervals(solar_wind_train_a, sunspot_a, 'teste/solar_wind_train_a_updated.csv')
process_solar_wind_intervals(solar_wind_train_b, sunspot_b, 'teste/solar_wind_train_b_updated.csv')
process_solar_wind_intervals(solar_wind_train_c, sunspot_c, 'teste/solar_wind_train_c_updated.csv')
