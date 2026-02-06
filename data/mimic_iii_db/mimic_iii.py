import pandas as pd
import os

# Load the CSV files
df_admissions = pd.read_csv('./ed/ADMISSIONS.csv')

df_patients = pd.read_csv('./ed/PATIENTS.csv')

# Merge on 'subject_id'
df_merged = pd.merge(df_admissions, df_patients, on='subject_id', how='inner')

# Select which columns to maintain
df_merged = df_merged[[
    'subject_id',
    'gender',
    'admission_type',
    'admission_location',
    'discharge_location',
    'insurance',
    'language',
    'religion',
    'marital_status',
    'ethnicity',
    'ed_reg_time',
    'ed_out_time',
    'diagnosis',
    ]]

# Save to output file
output_path = './mimic_iii_db.csv'
df_merged.to_csv(output_path, index=False)