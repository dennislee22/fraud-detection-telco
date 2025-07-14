import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import joblib

def feature_engineering_pandas(df):
    """
    Performs feature engineering on the raw call data using pandas.
    """
    print("Performing feature engineering with pandas...")
    df = df.set_index('msisdn')

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

def train_fraud_detection_model_rf_pandas(features_df):
    """
    Trains a fraud detection model using RandomForestClassifier with pandas and scikit-learn.
    """
    print("\nTraining the RandomForest model with pandas/scikit-learn...")
    # Fill any potential missing values that arose from aggregations (like std dev on a single call)
    features_df = features_df.fillna(0)

    X = features_df.drop('is_fraud', axis=1)
    y = features_df['is_fraud']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # Handle class imbalance using the 'class_weight' parameter
    print("Using class_weight='balanced' to handle class imbalance.")

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all available CPU cores
    )

    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    print("\nModel Evaluation on Test Set...")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    return model

if __name__ == '__main__':
    raw_data_filename = '3G_cdr_data.csv'
    model_output_filename = 'fraud_detection_model_rf.joblib'

    try:
        print(f"\nReading '{raw_data_filename}' with pandas...")
        # Read the entire CSV into a pandas DataFrame
        raw_df = pd.read_csv(raw_data_filename)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{raw_data_filename}'.")
        print("Please ensure the data file is in the correct directory.")
        exit()

    start_time = time.time()

    features_df = feature_engineering_pandas(raw_df)
    fraud_model = train_fraud_detection_model_rf_pandas(features_df)

    # Save the trained model using joblib
    joblib.dump(fraud_model, model_output_filename)

    print(f"\nTrained RandomForest model saved to '{model_output_filename}'")
    print(f"Process complete in {time.time() - start_time:.2f} seconds.")