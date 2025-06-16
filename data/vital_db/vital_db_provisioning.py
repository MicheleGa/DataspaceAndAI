import pandas as pd


if __name__ == "__main__":
    # Get dataset meta-data
    df_trks = pd.read_csv('https://api.vitaldb.net/trks')
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")

    # Pick a caseid randomly, given the length of df_cases
    caseid = df_cases['caseid'].sample(n=1).values[0]
    
    subject_trks = df_trks[df_trks['caseid'] == caseid]
    subject_cases = df_cases[df_cases['caseid'] == caseid]
    
    # Merge on caseid to get all tracks and case information
    df = pd.merge(subject_trks, subject_cases, on='caseid', how='left')
    df = df.reset_index(drop=True)
    
    # Drop `tid' column and merge all the `tname' values into a signle one
    df = df.drop(columns=['tid', 'caseid'])
    df['tname'] = df['tname'].str.cat(sep='-')
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Save as csv
    df.to_csv('vital_db_provisioning.csv', index=False)