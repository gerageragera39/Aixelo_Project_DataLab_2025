import os
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from scipy.stats import gmean
from tqdm import tqdm
import re

property_mapping = {
    "atomic_num": "Z",
    "group": "group",
    "period": "row",
    "electronegativity": "X",
    "electron_affinity": "electron_affinity",
    "melting_point": "melting_point",
    "boiling_point": "boiling_point",
    "density": "density_of_solid",
    "ionization_energy": "ionization_energies"
}
stat_names = ["mean", "geometric_mean", "standard_deviation", "max", "min"]


def compute_stats(values):
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [np.nan] * 5
    return [
        np.mean(values),
        gmean(values) if np.all(values > 0) else np.nan,
        np.std(values),
        np.max(values),
        np.min(values)
    ]


df = pd.read_csv('../../matched_cifs_with_energy.csv')
fingerprints = []

for path in tqdm(df["file_path"]):
    try:
        structure = Structure.from_file(path)
        elements = [site.specie for site in structure if isinstance(site.specie, Element)]
        stats = []
        for prop_key, attr in property_mapping.items():
            values = []
            for el in elements:
                try:
                    if attr == "ionization_energies":
                        val = el.ionization_energies[0] if el.ionization_energies else np.nan
                    else:
                        val = getattr(el, attr, np.nan)
                    values.append(val)
                except Exception:
                    values.append(np.nan)
            stats.extend(compute_stats(values))
    except Exception as e:
        print(f"Error with file : {path}: {e}")
        stats = [np.nan] * 45
    fingerprints.append(stats)

columns = [f"{prop}_{stat}" for prop in property_mapping.keys() for stat in stat_names]

filenames_only = df["file_path"].apply(os.path.basename)

fingerprints_df = pd.DataFrame(fingerprints, columns=columns)
result = pd.concat([filenames_only.rename("filename"), df["energy"].reset_index(drop=True), fingerprints_df], axis=1)

result.to_csv("45_fingerprints_with_energy.csv", index=False)

# ----------------------------------------------------------------------------------------

df = pd.read_csv("45_fingerprints_with_energy.csv")


def extract_base_mof_name(filename):
    return re.sub(r'(_random_\d+|_\d+)?_relaxed\.cif$', '', filename)


df['MOF'] = df['filename'].map(extract_base_mof_name)

fingerprint_cols = [col for col in df.columns if col not in ['energy', 'filename', 'MOF']]

agg_dict = {col: 'first' for col in fingerprint_cols}
agg_dict['energy'] = 'mean'

df_grouped = df.groupby('MOF').agg(agg_dict).reset_index()

df_grouped.to_csv("45_fingerprints_grouped_avg_energy.csv", index=False)
