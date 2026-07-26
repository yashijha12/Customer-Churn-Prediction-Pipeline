import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering to the cleaned dataframe:
    - tenure bins (0-12, 13-24, 25-48, 49-60, 60+)
    - service flags
    - categorical encoding (dummy variables)
    """
    df = df.copy()
    
    # 1. Tenure bins
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True)
    
    # 2. Service flags
    # Multiple lines
    df['has_multiple_services'] = df['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Streaming (StreamingTV or StreamingMovies)
    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # Internet addons (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport)
    addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['internet_addons_count'] = df[addons].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['has_internet_addons'] = (df['internet_addons_count'] > 0).astype(int)
    
    # 3. Categorical Encoding (One-Hot Encoding)
    # We will drop original columns that we engineered into bins/flags to avoid multicollinearity,
    # or keep them based on choice. We'll drop the redundant ones to see the effect.
    cols_to_drop = ['tenure'] # drop tenure since we have tenure_bin
    df = df.drop(columns=cols_to_drop)
    
    # Convert remaining categoricals to dummies
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all boolean columns are int (for XGBoost compatibility)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    return df

def basic_encode(df):
    """
    Basic encoding without the engineered features for ablation study.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df
