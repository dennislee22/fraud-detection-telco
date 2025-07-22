import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import os

# --- MLflow Imports ---
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

def feature_engineering_pandas(df):
    """
    Performs feature engineering on the raw call data using pandas.
    """
    print("Performing feature engineering with pandas...")
    # Setting the index is not necessary here as groupby handles it
    # df = df.set_index('msisdn')

    user_features_df = df.groupby('msisdn').apply(calculate_all_features_for_group)

    return user_features_df

def calculate_all_features_for_group(group):
    """
    Calculates aggregated features for a single user (a group of records).
    """
    # Ensure the 'is_fraud' column is treated as a boolean
    group['is_fraud'] = group['is_fraud'].astype(bool)

    # Calculate features for the group
    nocturnal_hours = (group['hour_of_day'] >= 22) | (group['hour_of_day'] <= 6)
    features = {
        'total_calls': len(group),
        'outgoing_call_ratio': (group['call_direction'] == 'outgoing').mean(),
        'avg_duration': group['duration'].mean(),
        'std_duration': group['duration'].std(),
        'nocturnal_call_ratio': nocturnal_hours.mean(),
        'mobility': group['cell_tower'].nunique(),
        'is_fraud': group['is_fraud'].iloc[0] # Take the fraud status from the first record
    }
    return pd.Series(features)

if __name__ == '__main__':
    # Define file paths
    raw_data_filename = '3G_cdr_data.csv'

    try:
        print(f"\nReading '{raw_data_filename}' with pandas...")
        # Read the entire CSV into a pandas DataFrame
        raw_df = pd.read_csv(raw_data_filename)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{raw_data_filename}'.")
        print("Please ensure the data file is in the correct directory.")
        exit()

    start_time = time.time()

    # --- Feature Engineering ---
    features_df = feature_engineering_pandas(raw_df)
    
    # --- Data Preparation for Model ---
    print("\nPreparing data for the XGBoost model...")
    # Fill any potential missing values that arose from aggregations
    features_df = features_df.fillna(0)

    X = features_df.drop('is_fraud', axis=1)
    y = features_df['is_fraud']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # --- MLflow Experiment Tracking ---
    mlflow.set_experiment("Fraud Detection XGBoost")

    with mlflow.start_run():
        print("\nStarting MLflow run...")

        # --- Model Hyperparameters & Class Imbalance Handling ---
        scale_pos_weight = (y_train.value_counts()[False] / y_train.value_counts()[True])
        
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }

        # Log parameters
        mlflow.log_params(params)
        print(f"scale_pos_weight determined to be: {scale_pos_weight:.2f}")

        # --- Model Training ---
        print("Training the XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # --- Model Evaluation ---
        print("\nModel Evaluation on Test Set...")
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Print metrics to console
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # --- Log Artifacts ---
        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "evaluation_plots")
        plt.close()

        # 2. Feature Importances
        booster = model.get_booster()
        feature_scores = booster.get_score(importance_type='weight')
        feature_importances = pd.Series(feature_scores).sort_values(ascending=False)
        fi_path = "feature_importances.txt"
        feature_importances.to_csv(fi_path, header=False)
        mlflow.log_artifact(fi_path, "feature_importances")
        print("\nFeature Importances:")
        print(feature_importances)

        # --- Log Model ---
        print("\nLogging the model with MLflow...")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="xgboost-fraud-detection-model"
        )

        print(f"\n✅ MLflow Run successfully completed. Run ID: {mlflow.active_run().info.run_id}")

    print(f"\nProcess complete in {time.time() - start_time:.2f} seconds.")
