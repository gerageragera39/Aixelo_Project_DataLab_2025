import os

import pandas as pd
import re


def extract_mof_id(name):
    match = re.search(r'[A-Z]{6}', name)
    return match.group(0) if match else None


def generate_qmof_fingerprints_odac_energy(path, full_data_path):
    co2_dir = os.path.dirname(path)
    input_filename = os.path.basename(path)
    input_name_wo_ext = os.path.splitext(input_filename)[0]

    output_filename = f'matched_{input_name_wo_ext}_with_energy.csv'
    output_path = os.path.join(co2_dir, output_filename)

    co2_df = pd.read_csv(path)
    co2_df['MOF_ID'] = co2_df['MOF'].apply(extract_mof_id)

    co2_df = co2_df.dropna(subset=['MOF_ID'])

    full_df = pd.read_csv(full_data_path)
    full_df['MOF_ID'] = full_df['MOF'].apply(extract_mof_id)

    matched_df = full_df[full_df['MOF_ID'].isin(co2_df['MOF_ID'])].copy()

    merged_df = matched_df.merge(co2_df[['MOF_ID', 'energy']], on='MOF_ID', how='left')

    # Сохраняем
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Saved in: {output_path}")


def enrich_with_energy_from_qmof(input_fp_path, energy_data_path):
    base_dir = os.path.dirname(input_fp_path)
    input_filename = os.path.basename(input_fp_path)
    input_name_wo_ext = os.path.splitext(input_filename)[0]

    output_all = os.path.join(base_dir, f'{input_name_wo_ext}_with_qmof_energy.csv')
    output_matched = os.path.join(base_dir, f'odac_{input_name_wo_ext}_with_qmof_energy_matched_only.csv')

    co2_df = pd.read_csv(input_fp_path)
    co2_df = co2_df.drop(columns=['energy'])
    fsr_df = pd.read_csv(energy_data_path)

    co2_df["MOF_ID"] = co2_df["MOF"].apply(extract_mof_id)
    fsr_df["MOF_ID"] = fsr_df["refcode"].apply(extract_mof_id)

    merged_df = co2_df.merge(fsr_df[["MOF_ID", "E_PBE"]], on="MOF_ID", how="left")

    matched_df = merged_df[merged_df["E_PBE"].notna()].copy()
    matched_df.drop(columns=["MOF_ID"], inplace=True)
    matched_df.rename(columns={"E_PBE": "energy"}, inplace=True)
    matched_df.to_csv(output_matched, index=False)

    print(f"✅ Saved in: {output_matched}")


def main():
    FP_CO2_45 = '45_fingerprints_CO2.csv'
    FP_H2O_45 = '45_fingerprints_H2O.csv'

    FP_CO2_120 = '120_fingerprints_CO2.csv'
    FP_H2O_120 = '120_fingerprints_H2O.csv'

    full_data_path_45 = '../../../data/qmof_database/features/stoich45_fingerprints.csv'
    full_data_path_120 = '../../../data/qmof_database/features/stoich120_fingerprints.csv'

    energy_qmof = '../../../data/qmof_database/qmof_full/qmof-energies.csv'

    enrich_with_energy_from_qmof(FP_CO2_45, energy_qmof)
    enrich_with_energy_from_qmof(FP_H2O_45, energy_qmof)
    enrich_with_energy_from_qmof(FP_CO2_120, energy_qmof)
    enrich_with_energy_from_qmof(FP_H2O_120, energy_qmof)


if __name__ == '__main__':
    main()
