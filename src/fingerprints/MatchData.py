import csv
import os
import re

data_dict = {}
with open('../../data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt', "r") as f:
    for line in f:
        key, value = line.strip().split()
        data_dict[key.strip()] = float(value)

print(f"Length of adsorption_energy.txt : {len(data_dict)} ")


def find_matching_cif_files(root_dir):
    pattern = re.compile(r"_(\d+)_relaxed\.cif$")
    matching_cif_files = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".cif") and pattern.search(file):
                name_without_ext = os.path.splitext(file)[0]
                for key in data_dict:
                    if name_without_ext.startswith(key):
                        energy = data_dict[key]
                        full_file_path = os.path.abspath(os.path.join(subdir, file))
                        matching_cif_files.append((full_file_path, energy))
                        break
    return matching_cif_files


root_dir_CO2 = '../../data/odac/pristine_CO2'
matching_cif_files_CO2 = find_matching_cif_files(root_dir_CO2)

root_dir_H2O = '../../data/odac/pristine_H2O'
matching_cif_files_H2O = find_matching_cif_files(root_dir_H2O)

output_full = "matched_cifs_with_energy.csv"
with open(output_full, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_path', 'energy'])
    writer.writerows(matching_cif_files_CO2 + matching_cif_files_H2O)
