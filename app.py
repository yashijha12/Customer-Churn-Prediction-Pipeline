from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and columns
model_path = 'models/best_model.pkl'
columns_path = 'models/model_columns.pkl'

if os.path.exists(model_path) and os.path.exists(columns_path):
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
else:
    model = None
    model_columns = None
    print("Warning: Model or model columns not found. Ensure train_models.py has been run.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not model_columns:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        if not data:
            data = request.form.to_dict()
            
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # In a real scenario, we should run the data through data_preprocessing 
        # and feature_engineering. For simplicity, if the form sends raw data:
        # We need to map raw to engineered. 
        # But if the user sends already engineered data or we write a quick wrapper:
        
        # Let's assume the user sends raw data through the form.
        # We need to import our engineering functions
        import sys
        sys.path.append('src')
        from data_preprocessing import load_and_clean_data
        from feature_engineering import engineer_features
        
        # Since we might only have one row, let's just do manual mapping for the API
        # To make it robust, we should ideally use a pipeline. 
        # We will re-engineer here for single prediction:
        
        # Quick fallback if it's already engineered:
        if 'tenure_bin_13-24' in df.columns or len(df.columns) > 20:
            df_processed = df
        else:
            # We must apply feature engineering. 
            # We can't easily run pd.get_dummies on a single row without missing columns.
            # So we create a df with model_columns initialized to 0.
            pass # See below
            
        # Standard approach for single row:
        # Create an empty dataframe with the training columns
        df_pred = pd.DataFrame(columns=model_columns)
        df_pred.loc[0] = 0 # Initialize with 0
        
        # Map values manually based on training logic
        if 'tenure' in df.columns:
            tenure = float(df['tenure'].iloc[0])
            if tenure <= 12: df_pred['tenure_bin_0-12'] = 1
            elif tenure <= 24: df_pred['tenure_bin_13-24'] = 1
            elif tenure <= 48: df_pred['tenure_bin_25-48'] = 1
            elif tenure <= 60: df_pred['tenure_bin_49-60'] = 1
            else: df_pred['tenure_bin_60+'] = 1
            
        if 'MonthlyCharges' in df.columns:
            df_pred['MonthlyCharges'] = float(df['MonthlyCharges'].iloc[0])
        if 'TotalCharges' in df.columns:
            try:
                df_pred['TotalCharges'] = float(df['TotalCharges'].iloc[0])
            except:
                df_pred['TotalCharges'] = 0.0
                
        # Service flags
        if 'MultipleLines' in df.columns and df['MultipleLines'].iloc[0] == 'Yes':
            df_pred['has_multiple_services'] = 1
            
        streaming_tv = df.get('StreamingTV', pd.Series(['No'])).iloc[0]
        streaming_mov = df.get('StreamingMovies', pd.Series(['No'])).iloc[0]
        if streaming_tv == 'Yes' or streaming_mov == 'Yes':
            df_pred['has_streaming'] = 1
            
        addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        addon_count = 0
        for addon in addons:
            if df.get(addon, pd.Series(['No'])).iloc[0] == 'Yes':
                addon_count += 1
        if addon_count > 0:
            df_pred['has_internet_addons'] = 1
            df_pred['internet_addons_count'] = addon_count
            
        # Map categorical dummies
        for col in df.columns:
            val = df[col].iloc[0]
            dummy_col = f"{col}_{val}"
            if dummy_col in model_columns:
                df_pred[dummy_col] = 1
                
        # Ensure correct types
        df_pred = df_pred.astype(float)
        
        prediction = model.predict(df_pred)[0]
        probability = model.predict_proba(df_pred)[0][1]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
