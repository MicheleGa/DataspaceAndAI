import pandas as pd
import glob
import os

# Directory containing the CSV files
data_dir = './ed'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# Read all CSVs into a list of DataFrames
dfs = []
for f in csv_files:
    # Select which cSV table to use to generate the actual CSV file for MIMIC IV
    file_name = f.split('/')[-1] 
    if file_name == 'diagnosis.csv' or file_name == 'edstays.csv' or file_name == 'vitalsign.csv':
        dfs.append(pd.read_csv(f))

# Merge all DataFrames on 'subject_id' and 'stay_id'
from functools import reduce
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'stay_id'], how='outer'), dfs)

# Remove seq_num / chart_time_vitalsign / rhythm, not sure if they are actually useful
merged_df = merged_df[[
    'subject_id', 
    'in_time',
    'out_time',
    'gender',
    'race',
    'temperature',
    'heart_rate', 
    'resp_rate', 
    'o2_sat', 
    'sbp', 
    'dbp', 
    'icd_code', 
    'icd_version', 
    'icd_title'
]]

# Convert temperature from Fahrenheit to Celsius
merged_df['temperature'] = ((merged_df['temperature'] - 32) * 5 / 9).round(2)

# Save to output file
output_path = './mimic_iv_db.csv'
merged_df.to_csv(output_path, index=False)