import pandas as pd
import re

mof_csv_path = r'C:\Users\GED\PycharmProjects\Aixelo_Project_DataLab_2025\data\qmof_database\features\stoich45_fingerprints.csv'


def filter_mof_with_energy(energy_csv_path: str, molecule: str):
    energy_df = pd.read_csv(energy_csv_path, header=None, names=['file_path', 'energy'])
    mof_df = pd.read_csv(mof_csv_path)

    def extract_folder_name(fp):
        match = re.search(r'pristine_CO2[\\/](.*?)[\\/]', fp)

        if molecule == "H2O":
            match = re.search(r'pristine_H2O[\\/](.*?)[\\/]', fp)

        if match:
            return match.group(1)
        return None

    energy_df['folder_name'] = energy_df['file_path'].apply(extract_folder_name)
    energy_df = energy_df[energy_df['folder_name'].notnull()]

    folder_names = set(energy_df['folder_name'].values)

    def get_matching_folder(mof_name):
        for folder in folder_names:
            if folder in mof_name:
                return folder
        return None

    filtered_mof_df = mof_df.copy()
    filtered_mof_df['matching_folder'] = filtered_mof_df['MOF'].apply(get_matching_folder)
    filtered_mof_df = filtered_mof_df[filtered_mof_df['matching_folder'].notnull()]

    def find_energy(row):
        folder = row['matching_folder']
        matches = energy_df[energy_df['folder_name'] == folder]
        if not matches.empty:
            return matches.iloc[0]['energy']
        return None

    filtered_mof_df['energy'] = filtered_mof_df.apply(find_energy, axis=1)
    filtered_mof_df = filtered_mof_df.drop(columns=['matching_folder'])
    filtered_mof_df.to_csv(f'matched_{molecule}_qmof_odac.csv', index=False)

filter_mof_with_energy(r"C:\Users\GED\PycharmProjects\Aixelo_Project_DataLab_2025\src\fingerprints\average_energy_CO2.csv", 'CO2')
filter_mof_with_energy(r"C:\Users\GED\PycharmProjects\Aixelo_Project_DataLab_2025\src\fingerprints\average_energy_H2O.csv", 'H2O')