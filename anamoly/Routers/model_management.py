# other_processes_server/routers/model_management.py
import ast
import base64
import datetime
import io
import os
import plotly.graph_objects as go
import shutil
import uuid
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import cm, pyplot as plt
from pydantic import BaseModel, Field
import regex as re
import pandas as pd
from fastapi import APIRouter, Body, Depends, FastAPI, File, HTTPException, Query, UploadFile, status, Request # Import Request
from typing import List, Literal, Optional, Dict, Any, Union # Ensure Dict and Any are imported for broadcast data
import json # Import json for sending data over WebSocket

from Schemas.model_management import (
    ModelLoadRequest,
    SetActiveModelRequest,
    ModelInfo,
    TrainModelParams,
    RetrainModelParams,
    TrainModelRequest,
)

from Schemas.model_management import (
    ModelLoadRequest,
    SetActiveModelRequest,
    ModelInfo,
    TrainModelParams, # Updated to accept optional uploaded_file_name
    RetrainModelParams,
    TrainModelResponse, # Ensure this is imported for response_model
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    AnomalyResult
)
from Services.model_manager import ModelManager
from Core_ML.gmm_model import GMMAnomalyDetector
from Services.predict_model import AnomalyDetector, AnomalyPredictionResponse, DataPoint

gmm_detector_for_training = GMMAnomalyDetector() # Use a separate instance for training operations
detector = AnomalyDetector()

UPLOAD_DIRECTORY = "./uploaded_data"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

router = APIRouter(
    prefix="/mcp/model_management",
    tags=["Model Management"]
)

# Dependency to get the ModelManager instance
def get_model_manager() -> ModelManager:
    return ModelManager()

def parse_labels_for_transformation(labels_str: str) -> dict:
    labels_dict = {}
    matches = re.findall(r'(\w+)=([^,"]+)', labels_str)
    for key, value in matches:
        labels_dict[key] = value
    return labels_dict

UPLOADED_DATA_DIR = "../uploaded_data"
if not os.path.exists(UPLOADED_DATA_DIR):
    os.makedirs(UPLOADED_DATA_DIR)

REPORTS_BASE_DIR = "../anomaly_reports/gmm_model_auto_k"
if not os.path.exists(REPORTS_BASE_DIR):
    os.makedirs(REPORTS_BASE_DIR)


def process_datetime_for_utc(dt_str: Optional[str]) -> Optional[datetime.datetime]:
    """
    Processes a datetime string, attempting to parse it and convert to UTC.
    Handles 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD' formats.
    """
    if not dt_str:
        return None
    
    try:
        # Try parsing with seconds precision first (ISO format)
        dt_obj = datetime.datetime.fromisoformat(dt_str)
    except ValueError:
        try:
            # If that fails, try parsing just date (e.g., 'YYYY-MM-DD')
            dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d')
        except ValueError as e:
            # Re-raise as HTTPException for FastAPI to handle, as this is a client-side input error
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Invalid datetime format: '{dt_str}'. Please use 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD'. Error: {e}")

    # If datetime object is naive (no timezone info), assume UTC and localize it.
    # If it has timezone info, convert it to UTC.
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=datetime.timezone.utc)
    else:
        return dt_obj.astimezone(datetime.timezone.utc)

@router.get("/get_anomaly_summary")
async def get_detailed_anomaly_summary(
    start_datetime: Optional[str] = Query(None, description="Start date and time (ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)"),
    end_datetime: Optional[str] = Query(None, description="End date and time (ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)"),
    # anomaly_score_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum anomaly score (0.0 to 1.0) to include. Anomalies with lower scores will be filtered out."),
    max_anomalies_to_return: int = Query(5, ge=1, le=10, description="Maximum number of top-scoring anomalies to return. Prioritizes anomalies with the highest scores."),
    num_neighboring_points: Optional[int] = Query(2, ge=0, description="Number of neighboring data points (on each side of the anomaly) to include for context. Capped at a maximum of 6 points per side. Defaults to 2. Set to 0 to exclude context data."),
    metric_filter: Optional[str] = Query(None, description="Comma-separated list of metrics (e.g., 'cpu_usage,memory_utilization') to filter anomalies by. Anomalies must include at least one of these metrics in their top contributions.")
) -> JSONResponse:
    """
    Retrieves a detailed, summarized report of anomalies from the pre-processed anomaly CSV,
    including a specified number of neighboring data points from the raw time series data for context,
    and advanced metric filtering.
    The response is optimized for LLM consumption, providing expanded insights while aiming for conciseness.
    
    Parameters:
    - start_datetime (Optional[str]): Start date and time in ISO format (e.g., '2025-06-25T10:00:00' or '2025-06-25').
    - end_datetime (Optional[str]): End date and time in ISO format.
    - anomaly_score_threshold (float): Minimum anomaly score (0.0 to 1.0) to include.
    - max_anomalies_to_return (int): Maximum number of top anomalies to return.
    - num_neighboring_points (Optional[int]): Number of data points to fetch before and after each anomaly's timestamp for context. Max 6 per side. Defaults to 2.
    - metric_filter (Optional[str]): Comma-separated list of metric names to filter anomalies by.
    """
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Received request for /get_detailed_anomaly_summary:")
    print(f"  start_datetime: {start_datetime}")
    print(f"  end_datetime: {end_datetime}")
    # print(f"  anomaly_score_threshold: {anomaly_score_threshold}")
    print(f"  max_anomalies_to_return: {max_anomalies_to_return}")
    print(f"  num_neighboring_points: {num_neighboring_points}") 
    print(f"  metric_filter: {metric_filter}")

    # Cap num_neighboring_points at a maximum of 6 as per user's request
    capped_num_neighboring_points = min(num_neighboring_points if num_neighboring_points is not None else 0, 6)

    processed_start_datetime = process_datetime_for_utc(start_datetime)
    processed_end_datetime = process_datetime_for_utc(end_datetime)

    # Define paths for the anomaly report and the raw time series data
    anomaly_report_filename = "all_anomalies_with_top5_metrics.csv"
    anomaly_report_path = os.path.join(REPORTS_BASE_DIR, anomaly_report_filename)
    
    # Using the user-provided DCS1-Ares_metrics.csv as the raw time series data
    time_series_data_filename = "DCS1-Ares_metrics.csv" 
    time_series_data_path = os.path.join(UPLOADED_DATA_DIR, time_series_data_filename)

    # --- Validate and Load Anomaly Report ---
    if not os.path.exists(anomaly_report_path) or not os.path.isfile(anomaly_report_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Anomaly report file not found at '{anomaly_report_path}'. Please ensure anomaly reports are generated and available.")
    
    # --- Load Time Series Data for Context (only if neighboring points are requested) ---
    df_ts = pd.DataFrame()
    if capped_num_neighboring_points > 0:
        if not os.path.exists(time_series_data_path) or not os.path.isfile(time_series_data_path):
            print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Warning: Time series data file not found at '{time_series_data_path}'. Cannot provide context points.")
            # Do not raise an error; simply proceed without context points if the file is missing
        else:
            try:
                df_ts = pd.read_csv(time_series_data_path, on_bad_lines='skip')
                if 'timestamp' not in df_ts.columns:
                    raise ValueError("Time series data CSV missing 'timestamp' column. Cannot align context points.")
                
                df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
                
                # Localize or convert timestamps to UTC for consistency
                if df_ts['timestamp'].dt.tz is None:
                    df_ts['timestamp'] = df_ts['timestamp'].dt.tz_localize('UTC')
                else:
                    df_ts['timestamp'] = df_ts['timestamp'].dt.tz_convert('UTC')
                
                # Sort time series data by timestamp and reset index for efficient integer-location based slicing
                df_ts.sort_values(by='timestamp', inplace=True)
                df_ts.reset_index(drop=True, inplace=True) 
                
                print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Successfully loaded time series data with {len(df_ts)} rows.")

            except Exception as e:
                print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Error loading time series data: {e}")
                df_ts = pd.DataFrame() # Clear df_ts if loading fails, so context points won't be attempted

    # --- Main Anomaly Report Processing Logic ---
    try:
        df_anomaly = pd.read_csv(anomaly_report_path, on_bad_lines='skip')
        
        if df_anomaly.empty:
            return JSONResponse(content={"anomalies": [], "message": "Anomaly report CSV is empty. No anomalies to summarize."})
        
        if 'Timestamp' not in df_anomaly.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Required 'Timestamp' column not found in anomaly report CSV. Cannot process anomalies.")
        
        df_anomaly['Timestamp'] = pd.to_datetime(df_anomaly['Timestamp'])
        if df_anomaly['Timestamp'].dt.tz is None:
            df_anomaly['Timestamp'] = df_anomaly['Timestamp'].dt.tz_localize('UTC')
        else:
            df_anomaly['Timestamp'] = df_anomaly['Timestamp'].dt.tz_convert('UTC')

        # Apply time range filters to anomalies
        if processed_start_datetime:
            df_anomaly = df_anomaly[df_anomaly['Timestamp'] >= processed_start_datetime].copy()
        if processed_end_datetime:
            df_anomaly = df_anomaly[df_anomaly['Timestamp'] <= processed_end_datetime].copy()

        # Apply anomaly score threshold filter
        # df_anomaly = df_anomaly[df_anomaly['Anomaly_Score'] >= anomaly_score_threshold].copy()
        
        if df_anomaly.empty:
            return JSONResponse(content={"anomalies": [], "message": "No anomalies found matching the specified time range and score threshold."})
        
        # --- Apply Metric Filter ---
        if metric_filter:
            allowed_metrics = [m.strip().lower() for m in metric_filter.split(',')]
            
            # Helper function to check if anomaly's top metrics (from string representation) contain any of the allowed metrics
            def contains_any_allowed_metric(top_metrics_str: str, allowed_metrics_list: List[str]) -> bool:
                if not top_metrics_str or top_metrics_str == "No specific metrics identified":
                    return False
                try:
                    # Safely evaluate the string representation of a list of dictionaries (e.g., "[{'metric': 'cpu', 'deviation': 0.5}]")
                    metrics_list = ast.literal_eval(top_metrics_str) 
                    for m_dict in metrics_list:
                        if m_dict.get('metric') and m_dict['metric'].lower() in allowed_metrics_list:
                            return True
                except (ValueError, SyntaxError):
                    # Fallback for unexpected formats (e.g., if 'Top_Metrics' is just a comma-separated string of names)
                    for am in allowed_metrics_list:
                        if am in top_metrics_str.lower(): # Simple substring match
                            return True
                return False

            if 'Top_Metrics' not in df_anomaly.columns:
                print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Warning: 'Top_Metrics' column not found in anomaly report CSV. Cannot apply metric filter.")
            else:
                initial_count = len(df_anomaly)
                df_anomaly = df_anomaly[df_anomaly['Top_Metrics'].astype(str).apply(lambda x: contains_any_allowed_metric(x, allowed_metrics))].copy()
                print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] After metric filter: {len(df_anomaly)} anomalies (from {initial_count})")

        # Re-check if df_anomaly is empty after all filters
        if df_anomaly.empty:
            return JSONResponse(content={"anomalies": [], "message": "No anomalies found after applying all specified filters."})

        # --- Optimization: Pre-group and prepare top metrics for each anomaly event ---
        # Group by the anomaly identifying columns and aggregate relevant metric info
        # This reduces redundant filtering inside the main loop for 'top_metrics'
        grouped_anomalies = df_anomaly.groupby(['Timestamp', 'Anomaly_Score', 'Cluster']).apply(
            # For each group (unique anomaly event), extract top 5 metrics by Scaled_Deviation
            lambda x: x[['Metric', 'Scaled_Deviation', 'Original_Value']].sort_values(by='Scaled_Deviation', ascending=False).head(5).to_dict(orient='records')
        ).reset_index(name='top_metrics_data')

        # To get the top N anomalies overall, sort the unique anomaly events by Anomaly_Score
        grouped_anomalies_sorted = grouped_anomalies.sort_values(by='Anomaly_Score', ascending=False).head(max_anomalies_to_return)

        summarized_anomalies: List[Any] = []

        # Iterate through the pre-grouped and sorted unique anomaly events
        for index, row in grouped_anomalies_sorted.iterrows():
            anomaly_timestamp = row['Timestamp']
            
            # Format top_metrics_data for the response from the pre-calculated data
            metrics_contribution_summary = []
            if row['top_metrics_data']: # Ensure there's data to process
                for metric_dict in row['top_metrics_data']:
                    metrics_contribution_summary.append({
                        "metric": metric_dict['Metric'],
                        "deviation": round(float(metric_dict['Scaled_Deviation']), 2),
                        "value": round(float(metric_dict['Original_Value']), 2),
                    })
            
            context_points_data: List[Dict[str, Any]] = []

            # Fetch context data points based on num_neighboring_points if requested and raw data is available
            if capped_num_neighboring_points > 0 and not df_ts.empty:
                # Find the insertion point for the anomaly timestamp in the sorted time series data
                ts_index_loc = df_ts['timestamp'].searchsorted(anomaly_timestamp)

                # Calculate the range of indices for neighboring points
                start_idx = max(0, ts_index_loc - capped_num_neighboring_points)
                end_idx = min(len(df_ts), ts_index_loc + capped_num_neighboring_points + 1)

                context_df = df_ts.iloc[start_idx:end_idx].copy()
                
                if not context_df.empty:
                    # Determine which metrics from the raw data are relevant for context
                    # Prioritize metrics that contributed to the anomaly
                    relevant_metrics_for_context = [m['metric'] for m in metrics_contribution_summary if m['metric'] in context_df.columns]
                    
                    # Fallback: If no top metrics are found in the time series data or if 'top_metrics' was empty,
                    # include a few common high-level metrics if they exist in the raw data.
                    if not relevant_metrics_for_context:
                        common_metrics_candidates = ['node_load1', 'node_memory_MemAvailable_bytes', 
                                                     'node_network_transmit_bytes_total', 'node_disk_written_bytes_total',
                                                     'sda_node_disk_read_bytes_total', 'node_sockstat_sockets_used']
                        # Filter to only columns actually present in the current context_df
                        relevant_metrics_for_context = [m for m in common_metrics_candidates if m in context_df.columns]
                    
                    # Only proceed if there are actual metrics to display in the context
                    if relevant_metrics_for_context:
                        for _, ctx_row in context_df.iterrows():
                            point = {"timestamp": ctx_row['timestamp'].isoformat(timespec='seconds')}
                            for col in relevant_metrics_for_context:
                                # Only include if the value is not NaN and round for conciseness
                                if pd.notna(ctx_row[col]):
                                    point[col] = round(float(ctx_row[col]), 2)
                            # Only add the point if it contains more than just the timestamp (i.e., actual metric values)
                            if len(point) > 1:
                                context_points_data.append(point)
                    
            # Add the summarized anomaly details to the list
            summarized_anomalies.append({
                "timestamp": anomaly_timestamp.isoformat(timespec='seconds'),
                "score": round(float(row['Anomaly_Score']), 4), 
                # "cluster": int(row['Cluster']),
                "top_metrics": metrics_contribution_summary if metrics_contribution_summary else "No specific metrics identified",
                "context_data_points": context_points_data if context_points_data else "No context data points available."
            })
            
            # Stop once the maximum number of desired results are collected
            if len(summarized_anomalies) >= max_anomalies_to_return:
                break

        print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Returning {len(summarized_anomalies)} detailed summarized anomalies.")
        return JSONResponse(content={
            "anomalies": summarized_anomalies,
            "message": f"Success"
        })
        
    except Exception as e:
        print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] Critical Error processing anomaly report: {str(e)}")
        # Provide a general error message for unexpected server-side issues
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An internal server error occurred while processing the anomaly report: {str(e)}")

    
@router.get("/get_system_context")
async def get_system_context() -> JSONResponse:
    """
    Returns a concise, plain-text summary of the system's capabilities
    regarding Broadside email transmission and ServerSage observability.
    This context is optimized for LLM consumption, providing key facts
    without verbose formatting, to efficiently inform the model about
    the system's domain expertise.
    """
    context_text = """
System Context:
Our expertise covers robust, scalable solutions for critical business operations, focusing on mass digital communication (Broadside) and 360° IT observability (ServerSage).

Broadside: High-Volume Email Transmission
Challenges Addressed:
- Massive Email Surges: Handles hundreds of thousands of emails/hour, preventing delays.
- Spam Reputation Risk: Manages millions of emails daily to avoid spam filters.
- Unprecedented Scale & Orchestration: Supports distributed architectures for extreme email processing.

Broadside's Solutions:
- Hyper-Parallel Processing: Uses massive parallel processing and distributed databases for colossal email data surges and tracking.
- Centralized Control: Single console to control entire campaign flow.
- Extreme Scalability & Optimization: Scales seamlessly to 50+ Linux servers; highly optimized for efficiency (single server can handle peak loads).
- Resource Maximization: Stresses all four core resources (CPU, RAM, disk I/O, network) simultaneously, indicating truly high workload handling.

ServerSage: Unified Observability for Heterogeneous Environments
Key Capabilities:
- Agent-Based Monitoring: Lightweight collectors for cloud, on-premise, hybrid.
- Real-Time Insights & Proactive Alerts: Instant notifications and live analytics for problem pinpointing.
- Intelligent Thresholding (Watermark-Based): Alerts only for critical parameter deviations, reducing noise.
- Robust Open-Source Foundation: Built on industry-leading open-source frameworks for reliability and flexibility.
"""
    return JSONResponse(content={"context": context_text})



@router.post("/predict_anomaly")
async def predict_anomaly_endpoint(data_point_request: DataPoint, request: Request): # Add request: Request
    """
    Receives a single data point and returns an anomaly prediction.
    If an anomaly is detected, it is logged to a CSV file AND broadcasted via WebSocket.
    """
    try:
        # Get the timestamp from the request or use current UTC time
        current_timestamp = data_point_request.timestamp if data_point_request.timestamp else datetime.datetime.now(datetime.timezone.utc)

        # Use the _predict_anomaly method from the singleton detector instance
        prediction_details = detector._predict_anomaly(data_point_request.data_point)

        # Log anomaly if detected
        if prediction_details["is_anomaly"]:
            detector._log_anomaly_to_csv(
                timestamp=current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                anomaly_score=prediction_details["anomaly_score"],
                log_likelihood=prediction_details["log_likelihood"],
                probability_density=prediction_details["probability_density"],
                cluster_id=prediction_details["cluster_id"],
                assigned_cluster_probability=prediction_details["assigned_cluster_probability"],
                top_metrics={k: v for k, v in prediction_details["top_contributing_metrics"].items()}
            )
            
# Extract top 1 or 2 contributing metrics for the simplified description
            top_metrics_list = list(prediction_details['top_contributing_metrics'].items())
            summary_metrics_parts = []

            if len(top_metrics_list) >= 1:
                metric1_name = top_metrics_list[0][0]
                metric1_value = top_metrics_list[0][1]['original_value']
                # Changed phrasing for more directness
                summary_metrics_parts.append(f"'{metric1_name}'")

            if len(top_metrics_list) >= 2:
                metric2_name = top_metrics_list[1][0]
                metric2_value = top_metrics_list[1][1]['original_value']
                # Changed phrasing for more directness
                summary_metrics_parts.append(f"'{metric2_name}'")

            if summary_metrics_parts:
                metrics_summary_text = " and ".join(summary_metrics_parts)
            else:
                metrics_summary_text = "some key metrics are showing unusual behavior" # Fallback if no top metrics

            # Generate current timestamp for the notification
            current_timestamp = datetime.datetime.now()

            # Construct the new, more actionable 'description'
            description_for_ws = (
                f"**CRITICAL ALERT:** Anomaly detected in system performance. "
                f"Please investigate the following key indicators: {metrics_summary_text}."
            )

            # Update your anomaly_data_for_ws dictionary with this new description
            anomaly_data_for_ws = {
                "id": current_timestamp.timestamp(),
                "message": "Anomaly Detected!",
                "description": description_for_ws, # THIS IS THE KEY CHANGE
                "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "read": False,
                "details": prediction_details # Still send full details for deeper inspection
            }

            # This 'description' variable now holds the actionable summary
            print(anomaly_data_for_ws["description"])

            # The broadcast logic remains the same:
            websocket_manager = request.app.state.websocket_manager
            await websocket_manager.broadcast(json.dumps(anomaly_data_for_ws))
            print(f"Broadcasted anomaly via WebSocket: {anomaly_data_for_ws['message']} at {anomaly_data_for_ws['timestamp']}")

        return AnomalyPredictionResponse(
            is_anomaly=prediction_details["is_anomaly"],
            anomaly_score=prediction_details["anomaly_score"],
            log_likelihood=prediction_details["log_likelihood"],
            probability_density=prediction_details["probability_density"],
            cluster_id=prediction_details["cluster_id"],
            assigned_cluster_probability=prediction_details["assigned_cluster_probability"],
            top_contributing_metrics=prediction_details["top_contributing_metrics"],
            timestamp=current_timestamp,
            message="Anomaly prediction successful."
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")


@router.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Allows users to upload a data file (e.g., CSV).
    The server saves it and returns the path for subsequent training.
    """
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "message": f"File '{file.filename}' uploaded successfully.", "file_path": file_location}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not upload file: {e}")


# --- API Endpoints ---
# (Note: You have two identical @router.post("/upload_data") definitions.
#  FastAPI will likely use the second one. You should remove the duplicate.)

# @router.post("/upload_data") # DUPLICATE - REMOVE THIS ONE
# async def upload_data(file: UploadFile = File(...)):
#     """
#     Allows users to upload a data file (e.g., CSV).
#     The server saves it and returns the path for subsequent training.
#     """
#     file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
#     try:
#         os.makedirs(os.path.dirname(file_location), exist_ok=True)
#         with open(file_location, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         return {"status": "success", "message": f"File '{file.filename}' uploaded successfully.", "file_path": file_location}
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not upload file: {e}")


@router.post("/train_anomaly_model", response_model=TrainModelResponse)
async def train_anomaly_model(request: TrainModelRequest):
    """
    Trains a new anomaly detection model using GMM.
    - Allows manual or automatic selection of `n_components`.
    - Stores the trained model, anomaly report, and plots.
    """
    if not os.path.exists(request.data_source_path):
        raise HTTPException(
            status_code=404,
            detail=f"Data source file not found: {request.data_source_path}"
        )

    try:
        df_original = pd.read_csv(request.data_source_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read data source: {e}"
        )

    if request.algorithm.lower() != "gmm":
        raise HTTPException(
            status_code=400,
            detail="Only 'GMM' algorithm is supported at the moment."
        )

    try:
        # Call training with both manual and auto n_components support
        training_results = gmm_detector_for_training.train_model(
            df_original=df_original,
            n_components=request.n_components,
            auto_n_components=request.auto_n_components,
            criterion=request.criterion,
            anomaly_threshold_percentile=request.anomaly_threshold_percentile,
            model_id=request.model_id
        )

        return TrainModelResponse(
            status="success",
            message="Model training completed and artifacts stored.",
            model_id=training_results["model_id"],
            trained_model_path=training_results["trained_model_path"],
            anomaly_report_path=training_results["anomaly_report_path"],
            plots_path=str(training_results["plots_path"])
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during model training: {e}"
        )

# In-memory cache for storing chart data
chart_cache: Dict[str, Dict[str, Any]] = {}

# Set up Jinja2 templates.
templates = Jinja2Templates(directory="templates")


# --- Pydantic Models for Data Validation ---

# 3D Data Models
class Series3DData(BaseModel):
    x: List[Any]
    y: List[Any]
    z: List[Any]
    name: str

class Heatmap3DData(BaseModel):
    x: List[Any]
    y: List[Any]
    z: List[List[Any]]
    name: str = "3D Heatmap"

# 2D Data Models
class ChartDataPoint(BaseModel):
    x: List[Any]
    y: List[Any]
    name: str

class PieChartData(BaseModel):
    labels: List[str]
    values: List[float]
    hole: float = 0.4
    hoverinfo: str = "label+percent"

# The main request model is now more comprehensive
class ChartPlottingRequest(BaseModel):
    title: str = "Dynamic Data Visualization"
    description: str = "A powerful, interactive chart generated on-demand."
    chart_type: str = Field(..., description="The type of chart to render (e.g., 'line', 'bar', 'pie', 'scatter', 'heatmap').")
    chart_data: Union[List[ChartDataPoint], PieChartData, List[Series3DData], Heatmap3DData] = Field(..., description="Data payload based on chart type.")
    
    @classmethod
    def __discriminator__(cls, v):
        chart_type = v.get("chart_type")
        if chart_type in ["line", "bar", "stacked_bar", "waterfall", "scatter"]:
            return "chart_data"
        elif chart_type == "pie":
            return "chart_data"
        elif chart_type in ["line3d", "bar3d", "scatter3d"]:
            return "chart_data"
        elif chart_type == "heatmap":
            return "chart_data"
        return None

# --- Helper Function for Chart Generation ---
def generate_plotly_chart_definition(request_data: ChartPlottingRequest) -> Dict[str, Any]:
    """Generates a Plotly chart definition with a futuristic theme."""
    fig = go.Figure()
    
    # A vibrant, futuristic color palette
    color_palette = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF8C00', '#1E90FF', '#FF1493']

    # --- Conditional Logic for All Supported Graph Types (2D & 3D) ---
    chart_type_lower = request_data.chart_type.lower()
    
    if chart_type_lower in ["line", "bar", "stacked_bar", "waterfall", "scatter"]:
        # 2D charts with enhanced styles
        if chart_type_lower == "line":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter(x=series.x, y=series.y, mode='lines+markers', name=series.name,
                    line=dict(color=color_palette[i % len(color_palette)], width=3, shape='spline'),
                    marker=dict(size=10, color='white', line=dict(width=2, color=color_palette[i % len(color_palette)])),
                    hoverinfo='x+y+name'
                ))
        elif chart_type_lower == "bar":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar(x=series.x, y=series.y, name=series.name,
                    marker_color=color_palette[i % len(color_palette)],
                    hoverinfo='x+y+name'
                ))
        elif chart_type_lower == "stacked_bar":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar(x=series.x, y=series.y, name=series.name,
                    marker_color=color_palette[i % len(color_palette)],
                    hoverinfo='x+y+name'
                ))
            fig.update_layout(barmode='stack')
        elif chart_type_lower == "waterfall":
            series = request_data.chart_data[0]
            fig = go.Figure(go.Waterfall(x=series.x, y=series.y, name=series.name,
                connector=dict(line=dict(color="#555555")),
                increasing=dict(marker=dict(color='#2ECC71')),
                decreasing=dict(marker=dict(color='#E74C3C')),
                totals=dict(marker=dict(color='#3498DB')),
                hoverinfo='x+y+name'
            ))
        elif chart_type_lower == "scatter":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter(x=series.x, y=series.y, mode='markers', name=series.name,
                    marker=dict(size=10, color=color_palette[i % len(color_palette)], line=dict(width=2, color='white')),
                    hoverinfo='x+y+name'
                ))
        
        # Apply 2D layout
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14),
            xaxis=dict(title_text="X-Axis", showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.2)'),
            yaxis=dict(title_text="Y-Axis", showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.2)'),
            hovermode='closest'
        )

    elif chart_type_lower in ["line3d", "scatter3d", "bar3d", "heatmap"]:
        # 3D charts with enhanced styles
        if chart_type_lower == "line3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter3d(x=series.x, y=series.y, z=series.z, mode='lines+markers', name=series.name,
                    line=dict(color=color_palette[i % len(color_palette)], width=4),
                    marker=dict(size=5, symbol='circle', color=color_palette[i % len(color_palette)]),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "scatter3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter3d(x=series.x, y=series.y, z=series.z, mode='markers', name=series.name,
                    marker=dict(size=6, symbol='circle', color=color_palette[i % len(color_palette)], opacity=0.9),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "bar3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar3d(x=series.x, y=series.y, z=series.z, name=series.name,
                    marker=dict(color=color_palette[i % len(color_palette)], opacity=0.8),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "heatmap":
            fig.add_trace(go.Surface(
                x=request_data.chart_data.x, y=request_data.chart_data.y, z=request_data.chart_data.z,
                colorscale='Viridis', colorbar_title="Value",
                contours=dict(x=dict(show=True, project=dict(z=True))),
                name=request_data.chart_data.name
            ))

        # Apply 3D layout
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14),
            scene=dict(
                xaxis={"gridcolor": 'rgba(0, 255, 255, 0.2)', "zerolinecolor": '#00FFFF'},
                yaxis={"gridcolor": 'rgba(255, 0, 255, 0.2)', "zerolinecolor": '#FF00FF'},
                zaxis={"gridcolor": 'rgba(255, 255, 0, 0.2)', "zerolinecolor": '#FFFF00'},
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye={"x": 1.5, "y": 1.5, "z": 0.8})
            )
        )

    elif chart_type_lower == "pie":
        # Pie chart with enhanced styles
        fig.add_trace(go.Pie(
            labels=request_data.chart_data.labels, values=request_data.chart_data.values,
            name=request_data.title, hole=request_data.chart_data.hole,
            marker_colors=color_palette, textinfo='label+percent',
            insidetextorientation='radial'
        ))
        fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14)
        )
    
    else:
        raise ValueError(f"Unsupported chart type: {request_data.chart_type}")
    
    return fig.to_json()


# --- FastAPI Endpoints ---
@router.post("/plot_chart")
async def generate_chart_link(request_data: ChartPlottingRequest = Body(...)):
    """Generates a chart definition, stores it, and returns a unique URL."""
    try:
        chart_definition_dict = json.loads(generate_plotly_chart_definition(request_data))
        chart_definition_dict["description"] = request_data.description
        
        chart_id = str(uuid.uuid4())
        chart_cache[chart_id] = {"data": chart_definition_dict}
        chart_url = f"/mcp/model_management/chart_viewer/{chart_id}"
        return JSONResponse(content={"status": "success", "url": chart_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/chart_viewer/{chart_id}")
async def chart_viewer(request: Request, chart_id: str):
    """Renders the HTML page for a specific chart ID."""
    chart_data_entry = chart_cache.get(chart_id)
    if not chart_data_entry:
        raise HTTPException(status_code=404, detail="Chart not found. The link may have expired.")
    
    chart_definition_json = json.dumps(chart_data_entry["data"])
    
    return templates.TemplateResponse(
        "chart_viewer.html",
        {"request": request, "chart_definition": chart_definition_json}
    )

