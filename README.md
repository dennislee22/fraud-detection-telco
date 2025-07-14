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
- For Step 1, sythentic data can be created using this script.
- Run for Step 2 and 3 using RFC technique. Result as follows.
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
    1. The values you see (e.g., 0.25, 0.22) are the average decrease in node impurity (specifically Gini impurity) that a feature provides across all trees in the forest. It measures how effective a feature is at creating "pure" splits, separating the 'fraud' and 'not-fraud' cases into clean, distinct groups. These values are normalized so that the sum of all importances equals 1.0. They represent a relative contribution to the model's overall decision-making purity.
    2. RFC builds hundreds of deep, independent decision trees on random subsets of the data and features. It then averages their predictions. Because each tree is independent, features that are generally good predictors (like nocturnal_call_ratio in your case) will consistently produce pure splits and thus get a high importance score.
    
- Run for Step 2 and 3 using XGBoost technique. Result as follows.
```
python train-xgboost.py 

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
    1. The values (e.g., 21.0, 11.0) are raw counts of how many times a feature was used to make a split across all the trees, as default importance_type='weight' is used. It's a measure of frequency. It doesn't care how good the split was. These values are not normalized and do not sum to 1. An importance of 21 means the total_calls feature was used for a split 21 times.
    2. XGBoost builds trees sequentially. Each new tree is built to correct the errors made by the previous ones. This process can cause it to focus on features differently. For example, if total_calls corrects the most errors early on, it will be used frequently (high 'weight' score). Then, other features like outgoing_call_ratio might be used to clean up the remaining, more subtle errors.

Conclusion: Both RFC and XGBoost could lead to a perfect score but give very different answers due to different calculation methods and model architectures.

## Step 4: Model Evaluation and Tuning
Once the model is trained, it's crucial to evaluate its performance on a separate test dataset that it has not seen before. Key metrics for evaluation in fraud detection include:

- Precision: Of all the SIMs the model flagged as fraudulent, what percentage were actually fraudulent? High precision is important to avoid incorrectly blocking legitimate customers.
- Recall (Sensitivity): Of all the truly fraudulent SIMs, what percentage did the model correctly identify? High recall is essential to catch as much fraud as possible.
- F1-Score: The harmonic mean of precision and recall, providing a balanced measure of the model's accuracy.





# Code Explanation

```
def feature_engineering(df):
    """
    Processes the raw CDR data to create aggregated features for each user.
    """
```
This defines the function, which takes one argument: df, a pandas DataFrame containing the raw call detail records (CDRs).
```
Python

    user_features = df.groupby('user_id').agg(
This is the core of the transformation.

df.groupby('user_id'): This method groups all the rows in the DataFrame by their user_id. All records for user_id 1 are put into one group, all for user_id 2 in another, and so on.

.agg(...): This is the aggregation method. It is applied to each of these user-specific groups to calculate summary statistics, creating a single summary row for each user.

Inside the .agg() method, we define the new features (columns) we want to create:

total_calls=('user_id', 'count')

This creates a new column named total_calls.

For each user group, it looks at the user_id column and applies the count function to it, effectively counting the total number of calls made by that user.

outgoing_call_ratio=('call_direction', lambda x: (x == 'outgoing').sum() / len(x))

This creates the outgoing_call_ratio column.

It uses a custom function (lambda) on the call_direction column of each user's records.

(x == 'outgoing').sum() counts the number of calls where the direction was "outgoing".

len(x) gets the total number of calls for that user.

Dividing the two gives the proportion of calls that were outgoing—a very strong indicator for SIM box fraud.

avg_duration=('duration', 'mean')

This creates the avg_duration column by calculating the average (mean) of the duration for all calls made by that user.

std_duration=('duration', 'std')

This creates the std_duration column by calculating the standard deviation (std) of the call duration. A very low standard deviation means the call lengths are highly consistent, which is suspicious and typical of automated systems.

nocturnal_call_ratio=('hour_of_day', lambda h: (h.between(22, 24).sum() + h.between(0, 6).sum()) / len(h))

This creates the nocturnal_call_ratio column.

It uses a lambda function on the hour_of_day column.

It counts the number of calls made late at night (10 PM to 6 AM) and divides by the user's total number of calls to get the ratio of nocturnal activity.

mobility=('cell_tower', 'nunique')

This creates the mobility column.

It counts the number of unique (nunique) cell towers a user has connected to. A low value (like 1) signifies a lack of movement, which is a key characteristic of a stationary SIM box.

is_fraud=('is_fraud', 'first')

This simply carries over the is_fraud label for each user. Since all records for a given user have the same fraud status, we can just take the first value from the group.

Python

    ).reset_index()
After grouping, user_id becomes the index of the new DataFrame. .reset_index() converts this index back into a regular column, which is a more convenient format.

Python

    user_features.fillna(0, inplace=True)
This is a data cleaning step. If a user made only one call, its standard deviation (std_duration) would be undefined (NaN). This line replaces any such NaN values with 0 so the machine learning model can process the data without errors.

Python


will RFC recheck if the answer produced in X is different from is_fraud column in y?



Show thinking
That is a fantastic question, and it shows you're thinking critically about the training process.

The short answer is: Yes, absolutely. That "rechecking" process is the very essence of how the model learns.

However, let's clarify one small but important detail in your question: The "answer" isn't produced in X. X is just the evidence. The model itself produces an internal "guess" or prediction based on the evidence in X.

Here is a more detailed breakdown of that "rechecking" loop during the model.fit(X_train, y_train) step:

The Training Process: A Cycle of Guessing, Checking, and Refining
Think of the Random Forest as building a series of decision trees. For each tree, and for each split in that tree, the model goes through this cycle:

Step 1: Make a "Guess" (Propose a Split)
The model picks a feature from X_train (e.g., outgoing_call_ratio) and a value (e.g., 0.9). It proposes a rule:

"Let's split all users into two groups: Group A where outgoing_call_ratio > 0.9, and Group B where outgoing_call_ratio <= 0.9."

Step 2: The "Recheck" (Compare against y_train)
Now, the model does exactly what you asked about. It looks at the actual is_fraud labels in y_train for the users in each group.

It checks Group A: "Okay, for all the users I put in this group, what are their actual labels in y_train? Are they mostly True (fraud) or False (legit)?"

It checks Group B: "And for this other group, what are their actual labels?"

Step 3: Measure the "Error" (Calculate Impurity)
The model needs a mathematical way to score how good its split was. It calculates a value called "impurity" (often using a metric called Gini Impurity or Entropy).

A perfectly pure group would have an impurity of 0 (e.g., all users in the group are fraudulent).

A completely mixed group would have a high impurity (e.g., the group is a 50/50 mix of fraudulent and legit users).

The model's goal is to find splits that create the purest possible groups.

Step 4: Refine the Guess (Try a Different Split)
The model is not satisfied with its first guess. It will now try many other splits:

What if the rule was avg_duration < 30?

What if the rule was mobility == 1?

For every possible split, it repeats Step 2 and Step 3, "rechecking" its results against the y_train answer key and calculating the impurity score.

Step 5: Choose the Best Split
After checking all its options, the model chooses the one rule (the one split) that resulted in the lowest combined impurity. That rule becomes a permanent node in the decision tree.

This entire 5-step process is then repeated for the new subgroups that were just created, and so on, until the tree is fully built. The entire forest of 100 trees is built this way.

Summary: Training vs. Prediction
During Training (.fit()): The model has access to both the evidence (X_train) and the answers (y_train). It is in a constant loop of proposing rules, checking them against the answers, and refining them to minimize its error. This is where the "rechecking" happens.

During Prediction (.predict()): The model's training is finished. Its rules are locked in. Now, it only gets new evidence (X_test). It no longer has the answers. It simply applies the best rules it learned during training to predict a new answer. It cannot "recheck" because the answer key (y_test) is hidden from it.


    return user_features
Finally, the function returns the user_features DataFrame, which is now perfectly structured for the XGBoost model.


will RFC recheck if the answer produced in X is different from is_fraud column in y?



Show thinking
That is a fantastic question, and it shows you're thinking critically about the training process.

The short answer is: Yes, absolutely. That "rechecking" process is the very essence of how the model learns.

However, let's clarify one small but important detail in your question: The "answer" isn't produced in X. X is just the evidence. The model itself produces an internal "guess" or prediction based on the evidence in X.

Here is a more detailed breakdown of that "rechecking" loop during the model.fit(X_train, y_train) step:

The Training Process: A Cycle of Guessing, Checking, and Refining
Think of the Random Forest as building a series of decision trees. For each tree, and for each split in that tree, the model goes through this cycle:

Step 1: Make a "Guess" (Propose a Split)
The model picks a feature from X_train (e.g., outgoing_call_ratio) and a value (e.g., 0.9). It proposes a rule:

"Let's split all users into two groups: Group A where outgoing_call_ratio > 0.9, and Group B where outgoing_call_ratio <= 0.9."

Step 2: The "Recheck" (Compare against y_train)
Now, the model does exactly what you asked about. It looks at the actual is_fraud labels in y_train for the users in each group.

It checks Group A: "Okay, for all the users I put in this group, what are their actual labels in y_train? Are they mostly True (fraud) or False (legit)?"

It checks Group B: "And for this other group, what are their actual labels?"

Step 3: Measure the "Error" (Calculate Impurity)
The model needs a mathematical way to score how good its split was. It calculates a value called "impurity" (often using a metric called Gini Impurity or Entropy).

A perfectly pure group would have an impurity of 0 (e.g., all users in the group are fraudulent).

A completely mixed group would have a high impurity (e.g., the group is a 50/50 mix of fraudulent and legit users).

The model's goal is to find splits that create the purest possible groups.

Step 4: Refine the Guess (Try a Different Split)
The model is not satisfied with its first guess. It will now try many other splits:

What if the rule was avg_duration < 30?

What if the rule was mobility == 1?

For every possible split, it repeats Step 2 and Step 3, "rechecking" its results against the y_train answer key and calculating the impurity score.

Step 5: Choose the Best Split
After checking all its options, the model chooses the one rule (the one split) that resulted in the lowest combined impurity. That rule becomes a permanent node in the decision tree.

This entire 5-step process is then repeated for the new subgroups that were just created, and so on, until the tree is fully built. The entire forest of 100 trees is built this way.

Summary: Training vs. Prediction
During Training (.fit()): The model has access to both the evidence (X_train) and the answers (y_train). It is in a constant loop of proposing rules, checking them against the answers, and refining them to minimize its error. This is where the "rechecking" happens.

During Prediction (.predict()): The model's training is finished. Its rules are locked in. Now, it only gets new evidence (X_test). It no longer has the answers. It simply applies the best rules it learned during training to predict a new answer. It cannot "recheck" because the answer key (y_test) is hidden from it.

The Confusion Matrix
[[380   0]
 [  0  20]]
This is a table that shows the model's performance in detail:

Top-Left (380): True Negatives (TN)

These are the legitimate users that the model correctly identified as legitimate. There were 380, and the model got all of them right.

Top-Right (0): False Positives (FP)

These are legitimate users that the model incorrectly flagged as fraudulent. An FP is a costly mistake because you might block a real customer. Your model made zero of these mistakes, which is perfect.

Bottom-Left (0): False Negatives (FN)

These are fraudulent users that the model failed to catch. This is the most dangerous type of error, as it means fraud is going undetected. Your model had zero of these misses, which is also perfect.

Bottom-Right (20): True Positives (TP)

These are the fraudulent users that the model correctly identified as fraudulent. There were 20, and the model caught all of them.

The Classification Report
This report translates the numbers from the confusion matrix into more intuitive scores:

Precision (for True class): 1.00

Question: When the model predicts a user is fraudulent, how often is it correct?

Calculation: True Positives / (True Positives + False Positives) -> 20 / (20 + 0) = 1.00

Meaning: Your model has 100% precision. Every single user it flagged as fraud was actually a fraudulent user.

Recall (for True class): 1.00

Question: Of all the actual fraudulent users in the dataset, what percentage did the model successfully catch?

Calculation: True Positives / (True Positives + False Negatives) -> 20 / (20 + 0) = 1.00

Meaning: Your model has 100% recall. It found every single fraudulent user in the test set.

F1-Score: 1.00

This is the harmonic mean of Precision and Recall. It provides a single score that balances both metrics, and it's particularly useful for imbalanced datasets like this one. A score of 1.00 is perfect.

Accuracy: 1.00

This is the overall percentage of correct predictions. Since the model made no mistakes, the accuracy is 100%.

Conclusion for Part 1: The model achieved a perfect score on this test set. This is likely because the synthetic data has very clear and distinct patterns for fraudulent vs. legitimate behavior. In a real-world scenario, the scores would be lower, but this shows the model is learning the intended patterns effectively.

Part 2: Feature Importances
This section answers the question: "Which behavioral patterns were most useful for the model's decisions?"

The model assigns a score to each feature based on how much it helped improve the accuracy of its predictions.

avg_duration (0.413): This was by far the most important feature. The huge difference between the short, consistent calls of fraudsters and the varied calls of legitimate users gave the model the most information.

std_duration (0.223): The standard deviation of call duration was the second most important. This tells the model how consistent the call lengths are. A very low standard deviation is a strong sign of automation (fraud).

total_calls, outgoing_call_ratio, nocturnal_call_ratio: These features were also very helpful, capturing the high volume, one-way direction, and off-hours nature of the fraudulent activity.

mobility (0.000): This feature has an importance of zero. This means the other features were so powerful and clear that the model could perfectly identify all the fraudsters without even needing to check if they were moving between cell towers.
