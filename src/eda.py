import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Running EDA...")
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    sns.set_theme(style="whitegrid")
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Churn Distribution')
    plt.savefig('data/churn_distribution.png')
    plt.close()
    
    # 2. Numeric Features vs Churn
    # Handle TotalCharges for plotting
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=axes[0])
    axes[0].set_title('Tenure Distribution by Churn')
    
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, ax=axes[1])
    axes[1].set_title('Monthly Charges Distribution by Churn')
    
    sns.histplot(data=df, x='TotalCharges', hue='Churn', kde=True, ax=axes[2])
    axes[2].set_title('Total Charges Distribution by Churn')
    
    plt.tight_layout()
    plt.savefig('data/numeric_features_churn.png')
    plt.close()
    
    # 3. Correlation Matrix
    df_corr = df.copy()
    df_corr['Churn_Bin'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
    corr_matrix = df_corr[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Bin']].corr()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('data/correlation_matrix.png')
    plt.close()
    
    print("EDA completed. Plots saved to data/")

if __name__ == "__main__":
    run_eda()
