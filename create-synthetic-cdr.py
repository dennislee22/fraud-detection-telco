import pandas as pd
import numpy as np
import os
import time
from typing import List

def generate_batch_data(msisdn_batch: List[int], fraud_msisdns: set):
    batch_records = []
    
    # Loop through only the users assigned to this batch
    for msisdn in msisdn_batch:
        is_fraud = msisdn in fraud_msisdns
        
        if is_fraud:
            # Simulate Fraudulent SIM Box Behavior
            num_records = np.random.randint(500, 2000)
            call_directions = np.random.choice(['outgoing', 'incoming'], size=num_records, p=[0.99, 0.01])
            call_durations = np.random.normal(loc=15, scale=5, size=num_records).clip(5, 60)
            hours = np.random.normal(loc=2, scale=3, size=num_records).astype(int) % 24
            imei = f"fraud_imei_{msisdn}"
            cell_tower = "tower_A"
        else:
            # Simulate Legitimate User Behavior
            num_records = np.random.randint(20, 300)
            call_directions = np.random.choice(['outgoing', 'incoming'], size=num_records, p=[0.6, 0.4])
            call_durations = np.random.exponential(scale=180, size=num_records).clip(10, 1800)
            hours = np.random.normal(loc=14, scale=5, size=num_records).astype(int) % 24
            imei = f"legit_imei_{msisdn}"
            cell_tower = np.random.choice(["tower_A", "tower_B", "tower_C", "tower_D"], p=[0.4, 0.3, 0.2, 0.1])

        for i in range(num_records):
            batch_records.append({
                "msisdn": msisdn,
                "call_direction": call_directions[i],
                "duration": call_durations[i],
                "hour_of_day": hours[i],
                "imei": imei,
                "cell_tower": cell_tower,
                "is_fraud": is_fraud
            })
            
    return pd.DataFrame(batch_records)

if __name__ == '__main__':
    NUM_USERS = 200000
    FRAUD_PERCENTAGE = 0.05
    BATCH_SIZE = 500   # Process 500 users at a time to keep memory low
    OUTPUT_FILENAME = '3G_cdr_data.csv'
    
    start_time = time.time()
    
    print("Step 1: Preparing user lists...")
    base_msisdn = 6590000000
    msisdns = [base_msisdn + i for i in range(NUM_USERS)]
    
    num_fraud_users = int(NUM_USERS * FRAUD_PERCENTAGE)
    fraud_msisdns = set(np.random.choice(msisdns, size=num_fraud_users, replace=False))
    
    # Create batches of MSISDNs
    msisdn_batches = [msisdns[i:i + BATCH_SIZE] for i in range(0, len(msisdns), BATCH_SIZE)]
    num_batches = len(msisdn_batches)
    
    print(f"Divided {NUM_USERS} users into {num_batches} batches of up to {BATCH_SIZE} users each.")
    
    # --- Batch Processing Loop ---
    print("\nStep 2: Generating data in batches and writing to CSV...")
    for i, batch in enumerate(msisdn_batches):
        print(f"  - Processing batch {i+1}/{num_batches}...")
        
        # Generate a DataFrame for the current batch only
        df_batch = generate_batch_data(batch, fraud_msisdns)
        
        # For the first batch, write to a new file with the header
        if i == 0:
            df_batch.to_csv(OUTPUT_FILENAME, index=False, mode='w', header=True)
        # For all subsequent batches, append to the existing file without the header
        else:
            df_batch.to_csv(OUTPUT_FILENAME, index=False, mode='a', header=False)

    print(f"\nRaw synthetic dataset saved to '{OUTPUT_FILENAME}'")
    print(f"Dataset generation complete in {time.time() - start_time:.2f} seconds.")