from locust import HttpUser, task, between, events
import urllib3
import threading
import os
import json

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Define Payloads for API Requests ---
NON_FRAUD_PAYLOAD = {
  "accessKey": "m7f0r82csvg7rl2i1d0eqyyrgtkl6iwq",
  "request": {
    "dataframe_split": {
      "columns": ["total_calls", "outgoing_call_ratio", "avg_duration", "std_duration", "nocturnal_call_ratio", "mobility"],
      "data": [[50, 0.4, 120.5, 45.2, 0.1, 5]]
    }
  }
}
FRAUD_PAYLOAD = {
  "accessKey": "m7f0r82csvg7rl2i1d0eqyyrgtkl6iwq",
  "request": {
    "dataframe_split": {
      "columns": ["total_calls", "outgoing_call_ratio", "avg_duration", "std_duration", "nocturnal_call_ratio", "mobility"],
      "data": [[350, 0.98, 15.5, 4.8, 0.92, 1]]
    }
  }
}

# --- Setup for Logging Responses ---
write_lock = threading.Lock()
line_counter = {"count": 0}
OUTPUT_FILE = "locust_predictions.txt"

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"INFO: Removed existing output file: {OUTPUT_FILE}")

# --- Locust User Class ---
class FraudModelUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        self.client.verify = False

    def log_response(self, request_type, response):
        """
        Parses the nested JSON response and logs the prediction and latency to a file.
        """
        prediction_result = "ERROR"
        # **NEW**: Get latency in milliseconds from the response object
        latency_ms = int(response.elapsed.total_seconds() * 1000)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if "response" in response_data and "prediction" in response_data["response"]:
                    prediction_result = response_data["response"]["prediction"][0]
                else:
                    prediction_result = "ERROR (Key path not found)"
            except (json.JSONDecodeError, KeyError, IndexError):
                prediction_result = "ERROR (Invalid JSON or structure)"
        else:
            prediction_result = f"ERROR (HTTP {response.status_code})"
        
        # Write the final result, now including latency, to the log file
        with write_lock:
            line_counter["count"] += 1
            line_number = line_counter["count"]
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                # **UPDATED**: Added latency to the output string
                f.write(f"{line_number}. Type: {request_type}, Prediction: {prediction_result}, Latency: {latency_ms}ms\n")

    @task(9)
    def predict_non_fraud(self):
        response = self.client.post("/model", json=NON_FRAUD_PAYLOAD)
        self.log_response("Non-Fraud", response)

    @task(1)
    def predict_fraud(self):
        response = self.client.post("/model", json=FRAUD_PAYLOAD)
        self.log_response("Fraud", response)
