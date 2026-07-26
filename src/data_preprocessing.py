import pandas as pd
import numpy as np

def load_and_clean_data(filepath='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Loads raw telco churn dataset and performs basic cleaning.
    """
    df = pd.read_csv(filepath)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Handle TotalCharges (convert to numeric, drop na)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # There are typically 11 missing values for TotalCharges where tenure=0
    df = df.dropna(subset=['TotalCharges'])
    
    # Map Churn to binary if not already
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

if __name__ == '__main__':
    df = load_and_clean_data()
    print(f"Cleaned dataset shape: {df.shape}")
