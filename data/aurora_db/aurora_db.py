import pandas as pd
import glob
import os

# Directory containing the CSV files
df = pd.read_csv('./participants.tsv', sep='\t')

df = df[[
    'pid',
    'age',
    'height',
    'weight',
    'gender',
    'high_blood_pressure',
    'coronary_artery_disease', 
    'diabetes', 
    'arrythmia',
    'previous_heart_attack', 
    'previous_stroke', 
    'heart_failure',
    'aortic_stenosis', 
    'valvular_heart_disease', 
    'other_cv_diseases'
    ]]

# Convert height and weight columns from inches and pounds to meters and kilograms
df['height'] = (df['height'] * 0.0254).round(2)
df['weight'] = (df['weight'] * 0.45359237).round(2)

# Save to output file
output_path = './aurora_db.csv'
df.to_csv(output_path, index=False)