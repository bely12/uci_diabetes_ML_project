# import packages
import pandas as pd
import numpy as np

# read data
df = pd.read_csv('diabetic_data.csv')

# replace empty vals with nan
df = df.replace('?', np.nan)

# drop cols that are not useful
df = df.drop(columns=['encounter_id',
                      'patient_nbr',
                      'weight',
                      'max_glu_serum',
                      'A1Cresult',
                      'payer_code',
                      'medical_specialty',
                      'diag_2',
                      'diag_3'])

# drop the small number of unknown race (<2%)
df = df.dropna(subset=['race'])

# keep only Male & Female entries (there are only 3 instance of "unknown")
df = df[df['gender'] != 'Unknown/Invalid']

# handle age category; convert ranges to the median age in the range, these can stay as is since they are ordinal variables and preserves their relationship 
age_mapping = {
    '[0-10)': 5, 
    '[10-20)': 15, 
    '[20-30)': 25, 
    '[30-40)': 35,
    '[40-50)': 45, 
    '[50-60)': 55, 
    '[60-70)': 65, 
    '[70-80)': 75,
    '[80-90)': 85, 
    '[90-100)': 95}

df['age'] = df['age'].map(age_mapping)

# map ICD-9 diagnosis codes to diag_1 column to simplify things
def map_diag(code):
    if pd.isna(code):
        return 'Other'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        c = float(code)
        if 390 <= c <= 459 or c == 785: return 'Circulatory'
        elif 460 <= c <= 519 or c == 786: return 'Respiratory'
        elif 520 <= c <= 579 or c == 787: return 'Digestive'
        elif c == 250: return 'Diabetes'
        elif 800 <= c <= 999: return 'Injury'
        elif 710 <= c <= 739: return 'Musculoskeletal'
        elif 580 <= c <= 629 or c == 788: return 'Genitourinary'
        elif 140 <= c <= 239: return 'Neoplasms'
        else: return 'Other'
    except:
        return 'Other'

# convert IDC-9 codes to informative categories; otherwise binarizing would be insane since there are so many different codes
df['diag_1'] = df['diag_1'].apply(map_diag)

# convert some categorical vals to binary; this is for categories with just 2 values (ex. yes/no, male/female, etc.)
df['change'] = (df['change'] == 'Ch').astype(int)
df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
df['gender'] = (df['gender'] == 'Male').astype(int)

# specificy cols that are cateogrical and can't be simply binarized
cat_cols = ['race', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'diag_1',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']

# one-hot encode these cols/categories; variables in categories get their own binary col in the df
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# create binary vals for my target feature
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
df['any_readmission_binary'] = (df['readmitted'] == 'NO').astype(int)
df = df.drop(columns=['readmitted'])

# combine visits for a total number of hospital visits
df['total_previous_visits'] = (df['number_outpatient'] + df['number_inpatient'] + df['number_emergency'])


# check the final dataset
print(f"\nFinal shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# save cleaned dataset
df.to_csv("diabetes_cleaned.csv", index=False)