import pandas as pd

# Carregando os arquivos CSV
solar_wind_a = pd.read_csv('teste/solar_wind_train_a_updated.csv')
solar_wind_b = pd.read_csv('teste/solar_wind_train_b_updated.csv')
solar_wind_c = pd.read_csv('teste/solar_wind_train_c_updated.csv')

satellite_pos_a = pd.read_csv('teste/satellite_pos_a.csv')
satellite_pos_b = pd.read_csv('teste/satellite_pos_b.csv')
satellite_pos_c = pd.read_csv('teste/satellite_pos_c.csv')

# Função para processar cada par de solar_wind e satellite_pos
def process_solar_wind(solar_wind, satellite_pos, output_file):
    # Extraindo os dias da coluna timedelta em ambos os arquivos
    solar_wind['days'] = solar_wind['timedelta'].str.extract(r'(\d+)').astype(int)
    satellite_pos['days'] = satellite_pos['timedelta'].str.extract(r'(\d+)').astype(int)

    # Função 1: Preencher valores de 'gse_x_ace', 'gse_y_ace', 'gse_z_ace' quando 'source' for 'ac'
    def preencher_ac(solar_wind, satellite_pos):
        ac_mask = solar_wind['source'] == 'ac'
        solar_wind.loc[ac_mask, 'gse_x_ace'] = solar_wind.loc[ac_mask, 'days'].map(satellite_pos.set_index('days')['gse_x_ace'])
        solar_wind.loc[ac_mask, 'gse_y_ace'] = solar_wind.loc[ac_mask, 'days'].map(satellite_pos.set_index('days')['gse_y_ace'])
        solar_wind.loc[ac_mask, 'gse_z_ace'] = solar_wind.loc[ac_mask, 'days'].map(satellite_pos.set_index('days')['gse_z_ace'])

    # Função 2: Preencher valores de 'gse_x_dscovr', 'gse_y_dscovr', 'gse_z_dscovr' quando 'source' for 'ds'
    def preencher_ds(solar_wind, satellite_pos):
        ds_mask = solar_wind['source'] == 'ds'
        solar_wind.loc[ds_mask, 'gse_x_dscovr'] = solar_wind.loc[ds_mask, 'days'].map(satellite_pos.set_index('days')['gse_x_dscovr'])
        solar_wind.loc[ds_mask, 'gse_y_dscovr'] = solar_wind.loc[ds_mask, 'days'].map(satellite_pos.set_index('days')['gse_y_dscovr'])
        solar_wind.loc[ds_mask, 'gse_z_dscovr'] = solar_wind.loc[ds_mask, 'days'].map(satellite_pos.set_index('days')['gse_z_dscovr'])

    # Aplicando as funções
    preencher_ac(solar_wind, satellite_pos)
    preencher_ds(solar_wind, satellite_pos)

    # Salvando o arquivo atualizado
    solar_wind.to_csv(output_file, index=False)
    print(f"Arquivo {output_file} foi salvo com dados de satellite_pos.")

# Processando cada conjunto de arquivos solar_wind e satellite_pos
process_solar_wind(solar_wind_a, satellite_pos_a, 'teste/solar_wind_train_a_final.csv')
process_solar_wind(solar_wind_b, satellite_pos_b, 'teste/solar_wind_train_b_final.csv')
process_solar_wind(solar_wind_c, satellite_pos_c, 'teste/solar_wind_train_c_final.csv')
