import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score, roc_curve
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from data_preprocessing import load_and_clean_data
from feature_engineering import engineer_features, basic_encode

def train_and_evaluate(X_train, y_train, X_test, y_test, model, param_grid, search_type='grid', n_iter=10):
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=1)
    else:
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=3, scoring='recall', n_jobs=-1, random_state=42, verbose=1)
        
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return best_model, {'ROC-AUC': auc, 'Recall': rec, 'Precision': prec, 'F1': f1}

def main():
    print("Loading data...")
    df = load_and_clean_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Engineering features...")
    df_engineered = engineer_features(df)
    
    X = df_engineered.drop('Churn', axis=1)
    y = df_engineered['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTEENN...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    
    print(f"Original train shape: {X_train.shape}, Resampled train shape: {X_train_resampled.shape}")
    
    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'search': 'grid'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'search': 'random',
            'n_iter': 10
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'search': 'random',
            'n_iter': 10
        }
    }
    
    results = {}
    best_model = None
    best_recall = 0
    best_model_name = ""
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        model, metrics = train_and_evaluate(
            X_train_resampled, y_train_resampled, X_test, y_test, 
            config['model'], config['params'], config['search'], config.get('n_iter')
        )
        results[name] = metrics
        print(f"{name} Metrics: {metrics}")
        
        if metrics['ROC-AUC'] > best_recall:
            best_recall = metrics['ROC-AUC']
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_recall:.4f}")
    
    # Save best model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save test data columns to handle feature alignment in Flask
    joblib.dump(list(X_train.columns), 'models/model_columns.pkl')
    
    print("\n--- Ablation Study ---")
    print("Training best model type on basic encoded features without tenure bins/service flags...")
    df_basic = basic_encode(df)
    X_basic = df_basic.drop('Churn', axis=1)
    y_basic = df_basic['Churn']
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_basic, y_basic, test_size=0.2, random_state=42, stratify=y_basic)
    X_train_b_res, y_train_b_res = smote_enn.fit_resample(X_train_b, y_train_b)
    
    best_model_class = models[best_model_name]['model']
    ablation_model = best_model_class.fit(X_train_b_res, y_train_b_res)
    
    y_pred_b = ablation_model.predict(X_test_b)
    rec_b = recall_score(y_test_b, y_pred_b)
    
    improvement = ((best_recall - rec_b) / rec_b) * 100
    print(f"Recall without engineered features: {rec_b:.4f}")
    print(f"Recall with engineered features: {best_recall:.4f}")
    print(f"Improvement: {improvement:.2f}%")

if __name__ == '__main__':
    main()
