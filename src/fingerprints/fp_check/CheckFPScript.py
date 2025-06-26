import pandas as pd
import numpy as np

file1 = pd.read_csv('stoich45_fingerprints_QMOF.csv')
file2 = pd.read_csv('../../../data/qmof_database/features/stoich45_fingerprints.csv')

file1 = pd.read_csv('stoich120_fingerprints_QMOF.csv')
file2 = pd.read_csv('../../../data/qmof_database/features/stoich120_fingerprints.csv')

file1 = file1.sort_index(axis=1)
file2 = file2.sort_index(axis=1)

file1 = file1.fillna(0)
file2 = file2.fillna(0)

common_mofs = set(file1['MOF']) & set(file2['MOF'])
print(f'🔍 Found {len(common_mofs)} common MOFs')

for mof in common_mofs:
    row1 = file1[file1['MOF'] == mof].reset_index(drop=True)
    row2 = file2[file2['MOF'] == mof].reset_index(drop=True)

    if row1.shape != row2.shape:
        print(f"⚠️ Different number of records for MOF '{mof}': {row1.shape} vs {row2.shape}")
        continue

    data1 = row1.drop(columns='MOF')
    data2 = row2.drop(columns='MOF')

    data1_numeric = data1.apply(pd.to_numeric, errors='coerce')
    data2_numeric = data2.apply(pd.to_numeric, errors='coerce')

    numeric_mask = ~data1_numeric.isna().all()

    numeric_cols = data1.columns[numeric_mask]
    string_cols = data1.columns[~numeric_mask]

    numeric_equal = True
    if len(numeric_cols) > 0:
        numeric_equal = np.isclose(
            data1_numeric[numeric_cols],
            data2_numeric[numeric_cols],
            atol=1e-6,
            equal_nan=True
        ).all().all()

    string_equal = True
    if len(string_cols) > 0:
        string_equal = (data1[string_cols].values == data2[string_cols].values).all()

    if numeric_equal and string_equal:
        print(f"✅ Full match for MOF '{mof}'")
    else:
        print(f"❌ Data mismatch for MOF '{mof}':")

        if not numeric_equal and len(numeric_cols) > 0:
            diff_numeric = pd.DataFrame({
                'file1': data1_numeric[numeric_cols].stack(),
                'file2': data2_numeric[numeric_cols].stack()
            }).reset_index()
            diff_numeric.columns = ['row', 'column', 'file1', 'file2']
            diff_numeric = diff_numeric[
                ~np.isclose(
                    diff_numeric['file1'],
                    diff_numeric['file2'],
                    atol=1e-6,
                    equal_nan=True
                )
            ]
            if not diff_numeric.empty:
                print("🔢 Differences in numeric columns:")
                print(diff_numeric)

        if not string_equal and len(string_cols) > 0:
            df1_str = data1[string_cols].reset_index()
            df2_str = data2[string_cols].reset_index()
            diff_list = []
            for col in string_cols:
                unequal_mask = df1_str[col] != df2_str[col]
                if unequal_mask.any():
                    diff = pd.DataFrame({
                        'row': df1_str.index[unequal_mask],
                        'column': col,
                        'file1': df1_str.loc[unequal_mask, col],
                        'file2': df2_str.loc[unequal_mask, col]
                    })
                    diff_list.append(diff)
            if diff_list:
                diff_string = pd.concat(diff_list, ignore_index=True)
                print("🔤 Differences in string columns:")
                print(diff_string)
