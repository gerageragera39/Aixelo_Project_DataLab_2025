import os

import pandas as pd
from ase.io import read, write


def cifs_to_xyz(path, name):
    df = pd.read_csv(path)
    file_paths = df['file_path'].tolist()

    refcodes = []
    mofs = []

    for cif_path in file_paths:
        refcode = os.path.splitext(os.path.basename(cif_path))[0]
        refcodes.append(refcode)
        mofs.append(read(cif_path))

    write(f'mofs_{name}.xyz', mofs)

    with open(f'refcodes_{name}.csv', 'w') as w:
        w.write(','.join(refcodes))


csv_path_CO2 = r'../average_energy_CO2.csv'
csv_path_H2O = r'../average_energy_H2O.csv'

cifs_to_xyz(csv_path_CO2, 'CO2')
cifs_to_xyz(csv_path_H2O, 'H2O')
