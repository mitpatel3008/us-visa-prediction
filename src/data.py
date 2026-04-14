# src/data.py
"""
Data pipeline for US-Visa Prediction.
Taking raw data and converting it into clean, ready-to-train data
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                    OrdinalEncoder, PowerTransformer,
                                    FunctionTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
 
#Constants
REFERENCE_YEAR = 2024
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
COLS_TO_DROP   = ['case_id', 'requires_job_training']
BINARY_COLS    = ['has_job_experience', 'full_time_position']
ORDINAL_COLS   = ['education_of_employee']
ORDINAL_ORDER  = [['High School', "Bachelor's", "Master's", 'Doctorate']]
ONEHOT_COLS    = ['continent', 'region_of_employment', 'unit_of_wage']
NUM_COLS       = ['prevailing_wage_annual', 'no_of_employees', 'company_age']
 
# Step 1: Load Data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f'Loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df
 
# Step 2: Drop Useless Columns
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')
    return df
 
# Step 3: Clean Data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['no_of_employees'] = df['no_of_employees'].abs()  # fix negatives
    return df
 
# Step 4: Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    wage_mult = {'Hour': 2080, 'Week': 52, 'Month': 12, 'Year': 1}
    df['prevailing_wage_annual'] = df['prevailing_wage'] * df['unit_of_wage'].map(wage_mult)
    df['company_age'] = REFERENCE_YEAR - df['yr_of_estab']
    df = df.drop(columns=['prevailing_wage', 'yr_of_estab'])
    return df
 
# Step 5: Encode Target
def encode_target(df: pd.DataFrame):
    X = df.drop('case_status', axis=1)
    y = np.where(df['case_status'] == 'Certified', 1, 0)
    return X, y
 
# Step 6: Build Preprocessor
def build_preprocessor() -> ColumnTransformer:
    def binary_encode(X):
        X = pd.DataFrame(X)
        return X.apply(lambda col: (col == 'Y').astype(int)).values
 
    numeric_pipe = Pipeline([
        ('transform', PowerTransformer(method='yeo-johnson')),
        ('scale',     StandardScaler())
    ])
 
    return ColumnTransformer([
        ('num',     numeric_pipe,
                    NUM_COLS),
        ('binary',  FunctionTransformer(binary_encode),
                    BINARY_COLS),
        ('ordinal', OrdinalEncoder(categories=ORDINAL_ORDER),
                    ORDINAL_COLS),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'),
                    ONEHOT_COLS),
    ])
 
#  Full Pipeline 
def prepare_data(data_path: str, save_dir: str = 'models/'):
    
    os.makedirs(save_dir, exist_ok=True)
 
    df = load_data(data_path)
    df = drop_columns(df)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = encode_target(df)
 
    # Split FIRST, then preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
 
    # Fit preprocessor on train only
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
 
    # SMOTE only on training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
 
    # Save preprocessor
    joblib.dump(preprocessor, f'{save_dir}/preprocessor.pkl')
    print(f'Preprocessor saved to {save_dir}/preprocessor.pkl')
 
    return X_train_bal, X_test_proc, y_train_bal, y_test
