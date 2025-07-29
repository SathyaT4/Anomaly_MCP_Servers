import pandas as pd
import requests
import time
import datetime
import random
import json

# --- Configuration ---
FASTAPI_ENDPOINT = "http://localhost:3000/mcp/model_management/predict_anomaly" # Adjust if your endpoint is different
NUM_SETS = 1 # Set to 1 to send only one set at a time
POINTS_PER_SET = 50
MIN_ANOMALIES_PER_10_POINTS = 3 # New requirement: at least 3 anomalies in a set of 10
TIME_DELAY_SECONDS = 30 # Delay between sending each data point (simulates real-time)

# --- Load historical data for normal points ---
try:
    df_metrics = pd.read_csv("uploaded_data/DCS1-Ares_metrics.csv")
    
    # Exclude the timestamp column and the last 'node_uname_info' column as data points
    metric_columns = df_metrics.columns[1:-1].tolist()
    if 'node_uname_info' not in df_metrics.columns: # Fallback if the column is not there
        metric_columns = df_metrics.columns[1:].tolist()
    
    # Store historical data as a list of dictionaries, where each dict is a full data point
    historical_full_data_points = []
    for index, row in df_metrics[metric_columns].iterrows():
        data_dict = {}
        for col in metric_columns:
            try:
                data_dict[col] = float(row[col]) # Convert to float
            except ValueError:
                data_dict[col] = 0.0 # Handle non-numeric gracefully, or implement more robust cleaning
        historical_full_data_points.append(data_dict)
    
    print(f"Loaded {len(historical_full_data_points)} historical normal data points from DCS1-Ares_metrics.csv.")

except FileNotFoundError:
    print("Error: DCS1-Ares_metrics.csv not found. Please ensure it's in the correct path relative to the script.")
    exit()
except Exception as e:
    print(f"Error loading or processing DCS1-Ares_metrics.csv: {e}")
    exit()

# --- Load and Pre-process anomaly data ---
try:
    df_anomalies = pd.read_csv(
        "C:/Users/sathy\Downloads/Anomaly_MCP_Servers/Anomaly_MCP_Servers/anomaly_reports/gmm_model_auto_k/all_anomalies_with_top5_metrics.csv", # Path correct here
        engine='python',
        on_bad_lines='skip'
    )
    
    # Group anomalies by their original timestamp to represent a single anomalous event
    grouped_anomalies = df_anomalies.groupby('Timestamp')

    # Create a list of dictionaries, where each dictionary represents an anomalous event
    # and contains only the 'Metric' and 'Original_Value' for that event.
    anomaly_events_data = []
    for timestamp, group in grouped_anomalies:
        anomaly_dict = {}
        for _, row in group.iterrows():
            try:
                anomaly_dict[row['Metric']] = float(row['Original_Value']) # Ensure value is float
            except ValueError:
                anomaly_dict[row['Metric']] = 0.0 # Handle non-numeric gracefully
        if anomaly_dict: # Only add if it's not empty
            anomaly_events_data.append(anomaly_dict)
    
    print(f"Loaded {len(anomaly_events_data)} distinct anomaly events from all_anomalies_with_top5_metrics.csv.")

except FileNotFoundError:
    print("Error: all_anomalies_with_top5_metrics.csv not found. Please ensure it's in the correct path relative to the script.")
    exit()
except Exception as e:
    print(f"Error loading or processing all_anomalies_with_top5_metrics.csv: {e}")
    exit()

def send_data_to_api(timestamp, data_point_dict):
    """Sends a single data point (as a dictionary) to the FastAPI anomaly prediction endpoint."""
    payload = {
        "timestamp": timestamp,
        "data_point": data_point_dict
    }
    try:
        response = requests.post(FASTAPI_ENDPOINT, json=payload, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Request timed out for timestamp: {timestamp}")
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError as ce:
        print(f"Connection error for timestamp: {timestamp}. Is the FastAPI server running? Error: {ce}")
        return {"error": "ConnectionError"}
    except requests.exceptions.RequestException as e:
        print(f"Error sending data for timestamp {timestamp}: {e}. Response body: {response.text if 'response' in locals() else 'N/A'}")
        return {"error": str(e)}

def simulate_data():
    """Simulates real-time data by sending points, ensuring a minimum number of anomalies."""
    
    num_historical_normal_points = len(historical_full_data_points)
    num_anomaly_events = len(anomaly_events_data)

    if num_historical_normal_points == 0:
        print("Error: No historical normal data points available for simulation.")
        return
    
    # Declare global variables that might be modified within this function
    global POINTS_PER_SET
    global MIN_ANOMALIES_PER_10_POINTS # ADDED THIS LINE

    if num_anomaly_events == 0:
        print("Warning: No anomaly events loaded for simulation. Sending only normal data.")
        MIN_ANOMALIES_PER_10_POINTS = 0 # No anomalies if source is empty

    # Ensure POINTS_PER_SET is not more than available historical data
    if num_historical_normal_points < POINTS_PER_SET:
        print(f"Warning: Not enough historical normal data ({num_historical_normal_points}) for a full set of {POINTS_PER_SET} points. Adjusting POINTS_PER_SET to {num_historical_normal_points}.")
        POINTS_PER_SET = num_historical_normal_points
    
    # Calculate how many 10-point blocks are in a set
    num_10_point_blocks = POINTS_PER_SET // 10
    remaining_points_in_last_block = POINTS_PER_SET % 10

    # This loop will run only once due to NUM_SETS = 1
    for i in range(NUM_SETS): 
        print(f"\n--- Simulating Set {i+1}/{NUM_SETS} (Total {POINTS_PER_SET} points) ---")
        
        simulated_points_count = 0
        
        # Simulate in blocks of 10
        for block_idx in range(num_10_point_blocks):
            print(f"\n  -- Simulating Block {block_idx+1} of 10 points --")
            
            # Determine how many anomalies to inject in this block
            # Ensure at least MIN_ANOMALIES_PER_10_POINTS, but not more than available anomalies or points in block
            anomalies_for_this_block = min(MIN_ANOMALIES_PER_10_POINTS, num_anomaly_events, 10)
            
            # Randomly select actual anomaly events for this block
            if num_anomaly_events > 0:
                selected_anomaly_events_for_block = random.sample(anomaly_events_data, anomalies_for_this_block)
            else:
                selected_anomaly_events_for_block = [] # No anomalies if source is empty

            # Select normal points for the remaining slots in this block
            num_normal_points_for_block = 10 - anomalies_for_this_block
            selected_normal_points_for_block = random.sample(historical_full_data_points, num_normal_points_for_block)
            
            # Create a combined list of points to send for this block, mixing anomalies and normal
            points_to_send_in_order = []
            
            # Add anomaly points
            for anomaly_event_data in selected_anomaly_events_for_block:
                # Start with a random normal data point's structure to ensure all keys are present
                base_data_point = random.choice(historical_full_data_points).copy()
                # Overlay the anomalous metrics (Original_Value directly, no scaling)
                base_data_point.update(anomaly_event_data)
                points_to_send_in_order.append({"type": "ANOMALOUS", "data": base_data_point})
            
            # Add normal points
            for normal_point_data in selected_normal_points_for_block:
                points_to_send_in_order.append({"type": "NORMAL", "data": normal_point_data.copy()}) # .copy() to prevent accidental modification

            random.shuffle(points_to_send_in_order) # Randomize the order within the block
            
            for point_info in points_to_send_in_order:
                current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                point_type = point_info["type"]
                point_to_send_dict = point_info["data"]

                if point_type == "ANOMALOUS":
                    print(f"    Sending ANOMALOUS point (block {block_idx+1}, point {simulated_points_count+1}): {current_timestamp}")
                else:
                    print(f"    Sending normal point (block {block_idx+1}, point {simulated_points_count+1}): {current_timestamp}")
                
                response = send_data_to_api(current_timestamp, point_to_send_dict)
                
                if response and response.get("is_anomaly"):
                    print(f"      -> API detected anomaly with score: {response.get('anomaly_score'):.2f}")
                elif response and not response.get("error"):
                    print(f"      -> API returned normal prediction.")

                time.sleep(TIME_DELAY_SECONDS)
                simulated_points_count += 1
        
        # Handle remaining points if POINTS_PER_SET is not a multiple of 10
        if remaining_points_in_last_block > 0:
            print(f"\n  -- Simulating Remaining {remaining_points_in_last_block} points --")
            # Determine how many anomalies to inject in this remainder block
            # This ensures the overall set of 50 points has the correct anomaly distribution if logic requires
            anomalies_for_remainder = max(0, MIN_ANOMALIES_PER_10_POINTS - (simulated_points_count % 10)) 
            anomalies_for_remainder = min(anomalies_for_remainder, num_anomaly_events, remaining_points_in_last_block)

            selected_anomaly_events_for_remainder = random.sample(anomaly_events_data, anomalies_for_remainder)
            num_normal_points_for_remainder = remaining_points_in_last_block - anomalies_for_remainder
            selected_normal_points_for_remainder = random.sample(historical_full_data_points, num_normal_points_for_remainder)

            remainder_points_to_send = []
            for anomaly_event_data in selected_anomaly_events_for_remainder:
                base_data_point = random.choice(historical_full_data_points).copy()
                base_data_point.update(anomaly_event_data)
                remainder_points_to_send.append({"type": "ANOMALOUS", "data": base_data_point})
            for normal_point_data in selected_normal_points_for_remainder:
                remainder_points_to_send.append({"type": "NORMAL", "data": normal_point_data.copy()})
            
            random.shuffle(remainder_points_to_send)

            for point_info in remainder_points_to_send:
                current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                point_type = point_info["type"]
                point_to_send_dict = point_info["data"]

                if point_type == "ANOMALOUS":
                    print(f"    Sending ANOMALOUS remaining point (point {simulated_points_count+1}): {current_timestamp}")
                else:
                    print(f"    Sending normal remaining point (point {simulated_points_count+1}): {current_timestamp}")
                
                response = send_data_to_api(current_timestamp, point_to_send_dict)
                if response and response.get("is_anomaly"):
                    print(f"      -> API detected anomaly with score: {response.get('anomaly_score'):.2f}")
                elif response and not response.get("error"):
                    print(f"      -> API returned normal prediction.")
                time.sleep(TIME_DELAY_SECONDS)
                simulated_points_count += 1


if __name__ == "__main__":
    print(f"Starting simulation of {NUM_SETS} set, each with {POINTS_PER_SET} data points.")
    print(f"Connecting to FastAPI endpoint: {FASTAPI_ENDPOINT}")
    simulate_data()
    print("\nSimulation complete.")