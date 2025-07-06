# Detecting Fraudulent Users in Telco
Line-by-Line Breakdown

def feature_engineering(df):
    """
    Processes the raw CDR data to create aggregated features for each user.
    """
This defines the function, which takes one argument: df, a pandas DataFrame containing the raw call detail records (CDRs).

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

    return user_features
Finally, the function returns the user_features DataFrame, which is now perfectly structured for the XGBoost model.
