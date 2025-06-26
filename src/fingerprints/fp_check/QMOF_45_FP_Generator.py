import json

import pandas as pd

from src.fingerprints.generators.FP_Generator import generate_45_fingerprints

xyz_path_QMOF = 'mofs.xyz'
refcodes_path_QMOF = 'refcodes.csv'

generate_45_fingerprints(refcodes_path_QMOF, xyz_path_QMOF, 'QMOF')


df = pd.read_csv('stoich45_fingerprints_QMOF.csv')

with open('../../../data/qmof_database/qmof.json') as f:
    data = json.load(f)

id_to_name = {item['qmof_id']: item['name'] for item in data}

df['MOF'] = df['MOF'].map(id_to_name).fillna(df['MOF'])

df.to_csv('stoich45_fingerprints_QMOF.csv', index=False)