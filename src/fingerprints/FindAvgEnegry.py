import pandas as pd
import os


def extract_name(path):
    return os.path.basename(os.path.dirname(path))


def count_average(path, name):
    df = pd.read_csv(path)
    df = df[df['energy'] != 'energy']
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df = df.dropna(subset=['energy'])
    df['name'] = df['file_path'].apply(extract_name)

    mean_df = df.groupby('name', as_index=False).agg({'energy': 'mean'})
    first_paths = df.groupby('name', as_index=False).first()[['name', 'file_path']]
    result = pd.merge(first_paths, mean_df, on='name')
    result[['file_path', 'energy']].to_csv(f'average_energy_{name}.csv', index=False)


csv_file = "matched_CO2_with_energy.csv"
count_average(csv_file, 'CO2')

csv_file = "matched_H2O_with_energy.csv"
count_average(csv_file, 'H2O')
