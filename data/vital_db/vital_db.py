import pandas as pd


if __name__ == "__main__":
    # Get dataset meta-data
    df_trks = pd.read_csv('https://api.vitaldb.net/trks')
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")

    # Sample X distinct caseids
    sampled_caseids = df_cases['caseid'].sample(n=10, random_state=0)

    # Filter rows for all sampled caseids
    subject_trks = df_trks[df_trks['caseid'].isin(sampled_caseids)]
    subject_cases = df_cases[df_cases['caseid'].isin(sampled_caseids)]
    
    # Merge on caseid to get all tracks and case information
    df = pd.merge(subject_trks, subject_cases, on='caseid', how='left')
    df = df.reset_index(drop=True)
    
    # Drop `tid' column and merge all the `tname' values into a signle one
    df = df.drop(columns=['tid', 'caseid'])
    df['tname'] = df['tname'].str.cat(sep='-')
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Rename columns
    df = df.rename(columns={
        'tname': 'track_name',
        'subjectid': 'subject_id',
        'casestart': 'case_start',
        'caseend': 'case_end',
        'anestart': 'anesthesia_start',
        'aneend': 'anesthesia_end',
        'ane_type': 'anesthesia_type',
        'opstart': 'operation_start',
        'opend': 'operation_end',
        'optype' : 'operation_type',
        'opname' : 'operation_name',
        'death_inhosp' : 'death_in_hosp',
        'bmi' : 'body_mass_index'
        })
    df = df[[
        'subject_id',
        'anesthesia_start',
        'anesthesia_end',
        'anesthesia_type',
        'operation_start',
        'operation_end',
        'operation_type',
        'operation_name',
        'age',
        'sex',
        'height',
        'weight',
        'body_mass_index',
        'department',
        'death_in_hosp'
        ]]
    
    # Convert height from centimeters to meters
    df['height'] = (df['height'] / 100).round(2)
    
    # Save as csv
    df.to_csv('vital_db.csv', index=False)