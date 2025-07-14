# Detecting Fraudulent Users in Telco
Answering the call to combat telecommunications fraud, particularly within the prepaid customer segment, requires a sophisticated blend of data analysis and machine learning. Unlike postpaid services where customer information is more readily available, the anonymous nature of prepaid SIMs presents a unique challenge. Here is a real-life example of how a telco would train a model to detect a specific type of prepaid fraud: SIM Box Fraud.

#### The Scenario: Unmasking SIM Box Fraud
SIM box fraud is a prevalent issue where international calls are illegally terminated as local calls. Fraudsters use a device called a SIM box, which houses multiple prepaid SIM cards. When an international call comes in, it's routed over the internet to the SIM box, which then uses a local prepaid SIM to connect the last leg of the call. This bypasses the international gateway of the local operator, depriving them of significant revenue from international call tariffs.

## Step 1: Raw Data
The first step is to gather relevant data for each prepaid SIM card on the network. The primary source of this information is the Call Detail Records (CDRs). For each call and SMS, a CDR is generated containing a wealth of information. Key data points for our fraud detection model would include the following columns to build a behavioral profile.

- Subscriber ID (msisdn): The unique identifier for the SIM card.
- call_direction: Incoming or outgoing.
- duration: The length of the call in seconds.
- hour_of_day: Which hour the call was made.
- imei: The unique identifier of the handset used.
- cell_tower: The location of the cell tower that handled the call.

## Step 2: Feature Engineering - Building the Behavioral DNA
Raw CDR data isn't directly fed into a ML model. Instead, data scientists engage in feature engineering to create meaningful variables that can help distinguish between a regular user and a fraudulent SIM box. For detecting SIM box fraud, the following features are often engineered:

- total_calls: Total calls made by a specific MSISDN over a period of time.
- outgoing_call_ratio: A very high ratio of outgoing to incoming calls (from the perspective of the local network) is a strong indicator. A normal user calls a relatively diverse set of numbers over time. A SIM in a SIM box will call a vast number of unique numbers in a short period.
- avg_duration & std_duration: Calls routed through SIM boxes often have unusually consistent or very short durations. Features like the average call duration and the standard deviation of call durations are crucial.
- nocturnal_call_ratio: SIM boxes often operate during off-peak hours, including late at night, to take advantage of lower network traffic and less scrutiny. A high volume of calls during these hours is suspicious.
- mobility: Geospatial analysis and lack of mobility: A SIM card in a stationary SIM box will always connect to the same one or two cell towers. A legitimate mobile user, by contrast, will show movement across different cell tower locations.
- is_fraud: Label applies on specific MSISDN with it associated features based on historical judgement. 

## Step 3: Model Training
With the engineered features, the next step is to train a ML model. A common and effective approach is to use a supervised learning model, such as Random Forest Classifier (RFC) and XGBoost (Gradient Boosting). 
To train a supervised model, a historical dataset with labeled examples of fraudulent and non-fraudulent SIMs is required. It builds a multitude of decision trees, each based on a random subset of the features. To classify whether a particular MSISDN is fradulent or not, the model runs its features through all the decision trees and final outcome is determined by a majority vote from all the trees.


## The Result & Analysis
- For Step 1, sythentic data can be created using this [synthetic data creation script](create-synthetic-cdr.py).
- Run [RFC script](train-RFC.py) for Step 2 and 3 based on RFC technique. Result as follows.

```
$ python train-RFC.py 

Reading '3G_cdr_data.csv' with pandas...
Performing feature engineering with pandas...

Training the RandomForest model with pandas/scikit-learn...
Using class_weight='balanced' to handle class imbalance.

Model Evaluation on Test Set...
Confusion Matrix:
[[38000     0]
 [    0  2000]]

Classification Report:
              precision    recall  f1-score   support

       False       1.00      1.00      1.00     38000
        True       1.00      1.00      1.00      2000

    accuracy                           1.00     40000
   macro avg       1.00      1.00      1.00     40000
weighted avg       1.00      1.00      1.00     40000


Feature Importances:
total_calls             0.25
nocturnal_call_ratio    0.22
std_duration            0.21
avg_duration            0.20
outgoing_call_ratio     0.12
mobility                0.00
dtype: float64

Trained RandomForest model saved to 'fraud_detection_model_rf.joblib'
Process complete in 200.50 seconds.
```
- Feature Importances: "Which behavioral patterns were most useful for the model's decisions?"
   1. The values you see (e.g., 0.25, 0.22) are the average decrease in node impurity (specifically Gini impurity) that a feature provides across all trees in the forest. It measures how effective a feature is at creating "pure" splits, separating the 'fraud' and 'not-fraud' cases into clean, distinct groups. These values are normalized so that the sum of all importances equals 1.0. They represent a relative contribution to the model's overall decision-making purity. A perfectly pure group would have an impurity of 0 (e.g., all users in the group are fraudulent).
   2. RFC builds hundreds of deep, independent decision trees on random subsets of the data and features. It then averages their predictions. Because each tree is independent, features that are generally good predictors (like nocturnal_call_ratio in your case) will consistently produce pure splits and thus get a high importance score.
    
- Run [XGBoost script](train-XGBoost.py) for Step 2 and 3 based on XGBoost technique. Result as follows.
  
```
$ python train-xgboost.py 

Reading '3G_cdr_data.csv' with pandas...
Performing feature engineering with pandas...

Training the XGBoost model with pandas/scikit-learn...
Calculating scale_pos_weight for class imbalance...
scale_pos_weight determined to be: 19.00

Model Evaluation on Test Set...
Confusion Matrix:
[[38000     0]
 [    0  2000]]

Classification Report:
              precision    recall  f1-score   support

       False       1.00      1.00      1.00     38000
        True       1.00      1.00      1.00      2000

    accuracy                           1.00     40000
   macro avg       1.00      1.00      1.00     40000
weighted avg       1.00      1.00      1.00     40000


Feature Importances:
total_calls             21.0
outgoing_call_ratio     11.0
avg_duration             9.0
std_duration             8.0
nocturnal_call_ratio     7.0
dtype: float64

Trained XGBoost model saved to 'fraud_detection_model_xgb2.json'
Process complete in 201.70 seconds.
```
- Feature Importances: "Which behavioral patterns were most useful for the model's decisions?"
   1. The values (e.g., 21.0, 11.0) are raw counts of how many times a feature was used to make a split across all the trees, as default importance_type='weight' is used. It's a measure of frequency. It doesn't care how good the split was. These values are not normalized and do not sum to 1. An importance of 21 means the total_calls feature was used for a split 21 times.
   2. XGBoost builds trees sequentially. Each new tree is built to correct the errors made by the previous ones. This process can cause it to focus on features differently. For example, if total_calls corrects the most errors early on, it will be used frequently (high 'weight' score). Then, other features like outgoing_call_ratio might be used to clean up the remaining, more subtle errors.

- The confusion matrix table shows common output.
  1. True Negatives (TN = 380). These are the legitimate users that the model correctly identified as legitimate. There were 380, and the model got all of them right.
  2. False Positives (FP = 0). These are legitimate users that the model incorrectly flagged as fraudulent. An FP is a costly mistake because you might block a real customer. The model made zero of these mistakes, which is perfect.
  3. False Negatives (FN = 0). These are fraudulent users that the model failed to catch. This is the most dangerous type of error, as it means fraud is going undetected. Your model had zero of these misses, which is also perfect.
  4. True Positives (TP = 20). These are the fraudulent users that the model correctly identified as fraudulent. There were 20, and the model caught all of them.

- The classification report translates the numbers from the confusion matrix into more intuitive scores:
  1. Precision (for True class): 1.00. TP/(TP + FP) -> 20/(20 + 0) = 1.00. The model has 100% precision. Every single user it flagged as fraud was actually a fraudulent user.
  2. Recall (for True class): 1.00. TP /(TP + FN) -> 20 / (20 + 0) = 1.00. The model has 100% recall. It found every single fraudulent user in the test set.
  3. F1-Score: 1.00. This is the harmonic mean of Precision and Recall. It provides a single score that balances both metrics, and it's particularly useful for imbalanced datasets like this one. A score of 1.00 is perfect. High recall is essential to catch as much fraud as possible.
  4. Accuracy: 1.00. This is the overall percentage of correct predictions. Since the model made no mistakes, the accuracy is 100%.

- The model achieved a perfect score on this test set. This is likely because the synthetic data has very clear and distinct patterns for fraudulent vs. legitimate behavior. In a real-world scenario, the scores would be lower, but this shows the model is learning the intended patterns effectively.
- Both RFC and XGBoost could lead to a perfect score but give very different answers due to different calculation methods and model architectures.

## Step 4: Model Visualization
- Run this [script](inference-viz.ipynb) to generate and visualize the inference result.




