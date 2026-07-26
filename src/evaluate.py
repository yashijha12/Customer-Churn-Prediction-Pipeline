import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_clean_data
from feature_engineering import engineer_features

def main():
    print("Loading data...")
    df = load_and_clean_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_engineered = engineer_features(df)
    
    X = df_engineered.drop('Churn', axis=1)
    y = df_engineered['Churn']
    
    # Use the same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Loading best model...")
    try:
        model = joblib.load('models/best_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("--- Final Evaluation Metrics ---")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
