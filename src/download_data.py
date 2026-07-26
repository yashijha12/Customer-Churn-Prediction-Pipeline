import kagglehub
import shutil
import os
import pandas as pd

def download_data():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    
    # Kagglehub downloads to a cache directory. 
    # We find the csv and copy it to our data folder.
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break
    
    if csv_file:
        dest = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        shutil.copy(csv_file, dest)
        print(f"Data copied to {dest}")
        df = pd.read_csv(dest)
        print(f"Dataset shape: {df.shape}")
    else:
        print("Failed to find CSV file in downloaded dataset.")

if __name__ == "__main__":
    download_data()
