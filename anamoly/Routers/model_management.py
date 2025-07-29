# other_processes_server/routers/model_management.py
import base64
import datetime
import io
import os
import shutil
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from matplotlib import cm, pyplot as plt
from pydantic import BaseModel, Field
import regex as re
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status, Request # Import Request
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

REPORTS_BASE_DIR = "anamoly/anomaly_reports/gmm_model_auto_k"
if not os.path.exists(REPORTS_BASE_DIR):
    os.makedirs(REPORTS_BASE_DIR)




@router.get("/get_context")
async def get_combined_anomaly_and_dataset(
    start_datetime: Optional[datetime.datetime] = Query(None, description="Start date and time for filtering (e.g., 2025-07-20T10:00:00Z)"),
    end_datetime: Optional[datetime.datetime] = Query(None, description="End date and time for filtering (e.g., 2025-07-21T12:00:00Z)")
):
    """
    Retrieves both the structured anomaly report JSON and the raw dataset content (as JSON).
    Can filter results by a specified date and time range using 'start_datetime' and 'end_datetime' query parameters.
    """
    print(f"Received request for /get_context:")
    print(f"  start_datetime (raw query): {start_datetime}")
    print(f"  end_datetime (raw query): {end_datetime}")

    # Initialize processed datetime variables at the top of the function
    # This prevents NameError if start_datetime or end_datetime are None
    processed_start_datetime = None
    processed_end_datetime = None

    # Process start_datetime and end_datetime for consistent UTC comparison
    if start_datetime:
        if start_datetime.tzinfo is None:
            processed_start_datetime = start_datetime.replace(tzinfo=datetime.timezone.utc)
            print(f"  start_datetime localized to UTC (from naive): {processed_start_datetime}")
        else:
            processed_start_datetime = start_datetime.astimezone(datetime.timezone.utc)
            print(f"  start_datetime converted to UTC: {processed_start_datetime}")

    if end_datetime:
        if end_datetime.tzinfo is None:
            processed_end_datetime = end_datetime.replace(tzinfo=datetime.timezone.utc)
            print(f"  end_datetime localized to UTC (from naive): {processed_end_datetime}")
        else:
            processed_end_datetime = end_datetime.astimezone(datetime.timezone.utc)
            print(f"  end_datetime converted to UTC: {processed_end_datetime}")


    # --- 1. Retrieve and process the Anomaly Report JSON ---
    anomaly_report_filename = "all_anomalies_with_top5_metrics.csv"
    anomaly_report_path = 'C:/Users/sathy\Downloads/Anomaly_MCP_Servers/Anomaly_MCP_Servers/anomaly_reports/gmm_model_auto_k/' + anomaly_report_filename
    anomaly_data_json = []

    print(f"\nChecking for anomaly report at: {anomaly_report_path}")
    if not os.path.exists(anomaly_report_path) or not os.path.isfile(anomaly_report_path):
        print(f"Warning: Anomaly report not found at {anomaly_report_path}")
        # If the anomaly report is essential, consider raising an HTTPException here
        # raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Anomaly report file not found at {anomaly_report_path}.")
    else:
        print(f"Anomaly report found. Attempting to read...")
        try:
            df_anomaly = pd.read_csv(anomaly_report_path, on_bad_lines='skip')
            print(f"Original anomaly dataframe shape: {df_anomaly.shape}")

            if df_anomaly.empty:
                print("Anomaly report CSV is empty after reading.")
            elif 'Timestamp' not in df_anomaly.columns:
                print("Warning: 'Timestamp' column not found in anomaly report CSV. Anomaly filtering will not be applied.")
            else:
                # Convert 'Timestamp' column to datetime objects and ensure UTC
                df_anomaly['Timestamp'] = pd.to_datetime(df_anomaly['Timestamp'])
                if df_anomaly['Timestamp'].dt.tz is None:
                    df_anomaly['Timestamp'] = df_anomaly['Timestamp'].dt.tz_localize('UTC')
                    print("Anomaly Timestamps localized to UTC (from naive).")
                else:
                    df_anomaly['Timestamp'] = df_anomaly['Timestamp'].dt.tz_convert('UTC')
                    print("Anomaly Timestamps converted to UTC (from another timezone).")

                # Apply date and time filtering
                if processed_start_datetime:
                    df_anomaly = df_anomaly[df_anomaly['Timestamp'] >= processed_start_datetime]
                    print(f"Anomaly dataframe shape after start_datetime filter: {df_anomaly.shape}")
                if processed_end_datetime:
                    df_anomaly = df_anomaly[df_anomaly['Timestamp'] <= processed_end_datetime]
                    print(f"Anomaly dataframe shape after end_datetime filter: {df_anomaly.shape}")
                
                if df_anomaly.empty:
                    print("Anomaly dataframe is empty after filtering.")
                else:
                    grouped_anomalies = []
                    for (timestamp, anomaly_score, cluster), group in df_anomaly.groupby(['Timestamp', 'Anomaly_Score', 'Cluster']):
                        metrics_contribution = []
                        for _, row in group.iterrows():
                            metrics_contribution.append({
                                "Metric": row['Metric'],
                                "Scaled_Deviation": float(row['Scaled_Deviation']),
                                "Original_Value": float(row['Original_Value'])
                            })
                        
                        log_likelihood = float(group['Log_Likelihood'].iloc[0])
                        probability_density = float(group['Probability_Density'].iloc[0])
                        assigned_cluster_probability = float(group['Assigned_Cluster_Probability'].iloc[0])

                        grouped_anomalies.append({
                            "Timestamp": timestamp.isoformat(),
                            "Anomaly_Score": float(anomaly_score),
                            "Log_Likelihood": log_likelihood,
                            "Probability_Density": probability_density,
                            "Cluster": int(cluster),
                            "Assigned_Cluster_Probability": assigned_cluster_probability,
                            "Metrics_Contribution": metrics_contribution
                        })
                    anomaly_data_json = grouped_anomalies
                    print(f"Processed {len(anomaly_data_json)} anomaly groups.")
            
        except Exception as e:
            print(f"Error processing anomaly report CSV: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing anomaly report CSV: {str(e)}")

    # --- 2. Retrieve and process the Dataset JSON ---
    dataset_file_name = 'DCS1-Ares_metrics.csv'
    dataset_file_full_path = os.path.join(UPLOADED_DATA_DIR, dataset_file_name)
    dataset_content_json = []

    print(f"\nChecking for dataset file at: {dataset_file_full_path}")
    if not os.path.exists(dataset_file_full_path) or not os.path.isfile(dataset_file_full_path):
        print(f"Error: Dataset file '{dataset_file_name}' not found in uploaded_data at {dataset_file_full_path}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset file '{dataset_file_name}' not found in uploaded_data.")
    else:
        print(f"Dataset file found. Attempting to read...")
        try:
            df_dataset = pd.read_csv(dataset_file_full_path)
            print(f"Original dataset dataframe shape: {df_dataset.shape}")

            if df_dataset.empty:
                print("Dataset CSV is empty after reading.")
            elif 'timestamp' not in df_dataset.columns: # Note: column name is 'timestamp' here (lowercase 't')
                print("Warning: 'timestamp' column not found in dataset CSV. Dataset filtering will not be applied.")
            else:
                df_dataset['timestamp'] = pd.to_datetime(df_dataset['timestamp'])
                if df_dataset['timestamp'].dt.tz is None:
                    df_dataset['timestamp'] = df_dataset['timestamp'].dt.tz_localize('UTC')
                    print("Dataset Timestamps localized to UTC (from naive).")
                else:
                    df_dataset['timestamp'] = df_dataset['timestamp'].dt.tz_convert('UTC')
                    print("Dataset Timestamps converted to UTC (from another timezone).")

                # Apply date and time filtering to dataset
                if processed_start_datetime:
                    df_dataset = df_dataset[df_dataset['timestamp'] >= processed_start_datetime]
                    print(f"Dataset dataframe shape after start_datetime filter: {df_dataset.shape}")
                if processed_end_datetime:
                    df_dataset = df_dataset[df_dataset['timestamp'] <= processed_end_datetime]
                    print(f"Dataset dataframe shape after end_datetime filter: {df_dataset.shape}")

                if df_dataset.empty:
                    print("Dataset dataframe is empty after filtering.")
                else:
                    # Convert float64 and int64 columns to float for JSON serialization
                    for col in df_dataset.select_dtypes(include=['int64', 'float64']).columns:
                        df_dataset[col] = df_dataset[col].astype(float)
                    
                    # Convert datetime columns to ISO format for JSON serialization
                    for col in df_dataset.select_dtypes(include=['datetime64[ns]']).columns:
                        # Ensure timezone is removed before isoformat for clean output if desired, or keep it
                        # If you want timezone in output, remove .dt.tz_convert(None)
                        df_dataset[col] = df_dataset[col].dt.tz_convert(None).dt.isoformat() if df_dataset[col].dt.tz is not None else df_dataset[col].dt.isoformat()
                    
                    dataset_content_json = df_dataset.to_dict(orient="records")
                    print(f"Processed {len(dataset_content_json)} dataset records.")
        except Exception as e:
            print(f"Error processing dataset CSV: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing dataset CSV: {str(e)}")

    # --- 3. Combine and return the response ---
    print(f"\nFinal anomaly_data_json (first 3 items if any): {anomaly_data_json[:3]}")
    print(f"Final dataset_content_json (first 3 items if any): {dataset_content_json[:3]}")
    
    response_content = {
        "message": "Combined anomaly report and dataset retrieved successfully.",
        "anomaly_report": anomaly_data_json,
        # "original_dataset": dataset_content_json # <-- UNCOMMENTED THIS LINE
    }
    print(f"\nReturning JSONResponse with {len(anomaly_data_json)} anomaly records and {len(dataset_content_json)} dataset records.")
    return JSONResponse(content=response_content)


@router.get("/get_system_context")
async def get_system_context():
    """
    Returns the static context about Broadside challenges, solutions, and ServerSage capabilities.
    This context should be fed to Claude Desktop automatically before any other actions.
    """
    context_text = """
    ## **Our Expertise: Tackling Digital Infrastructure's Toughest Challenges**

    At **[Your Company Name - if you want to add one here, otherwise omit]**, we specialize in developing robust, scalable solutions for critical business operations. For over a decade, our flagship SaaS product, **Broadside**, has empowered enterprises to overcome the most demanding challenges in mass digital communication. Complementing this, **ServerSage** provides unparalleled 360° observability across complex IT environments.

    ---

    ## **Broadside: Mastering High-Volume Email Transmission** 🚀

    Sending large volumes of emails reliably and efficiently presents formidable challenges that conventional corporate email systems simply can't handle.

    ### **The Challenges We Solve:**

    * **Massive Email Surges:** Corporate email systems falter under the pressure of hundreds of thousands of emails per hour, leading to delays that directly impact customer experience (e.g., delayed e-tickets).
    * **Spam Reputation Risk:** Daily transmission of millions of emails without proper management inevitably triggers spam filters, jeopardizing legitimate communications and business viability.
    * **Unprecedented Scale & Orchestration:** Processing, filtering, logging, and dispatching emails at such a scale requires sophisticated, distributed architectures. Relying on a few servers is insufficient; orchestrating dozens to seamlessly work as one demands exceptional scalability and cluster management.

    ### **Broadside's Pioneering Solutions:**

    Broadside is engineered from the ground up to defy the limitations of traditional email systems. It's a testament to over a decade of innovation, designed for unparalleled performance:

    * **Hyper-Parallel Processing:** Broadside leverages **massive parallel processing** and **multiple distributed databases** to effortlessly manage colossal email data surges, maintaining complete fidelity and tracking every single message.
    * **Centralized Control:** An intuitive console allows a single executive to **control the entire flow** of any campaign or one batch, providing comprehensive oversight regardless of the underlying server count.
    * **Extreme Scalability & Optimization:** Running on a cluster of Linux servers, Broadside **scales seamlessly to 50+ servers** without performance degradation. Each server is meticulously optimized, so much so that even our largest corporate clients rarely utilize more than two servers at their peak, a testament to its efficiency.
    * **Resource Maximization:** Broadside stands out as the *only* system we've designed where, given the right load, a single server is simultaneously **stressed on all four core resources**: CPU speed, RAM capacity, disk I/O speed, and network bandwidth, signifying its truly non-trivial workload handling.

    ---

    ## **ServerSage: Unified Observability for Heterogeneous Environments** 📊

    ServerSage revolutionizes how you monitor your entire IT landscape, delivering a **360° view** across diverse applications, infrastructure, databases, and other appliances. Our platform consolidates real-time metrics, logs, and traces into one intuitive, unified dashboard.

    ### **Key Capabilities:**

    * **Agent-Based Monitoring:** Deploy **lightweight, easy-to-install collectors** that integrate seamlessly across all your environments—cloud, on-premise, and hybrid setups.
    * **Real-Time Insights & Proactive Alerts:** Stay ahead of potential issues with **instant notifications** and **live analytics**. ServerSage precisely pinpoints problems, enabling you to address them before they escalate into critical incidents.
    * **Intelligent Thresholding (Watermark-Based):** Receive alerts **only when they truly matter**. Our unique watermark-based thresholding ensures you focus solely on critical parameter deviations, eliminating notification fatigue from irrelevant noise.
    * **Robust Open-Source Foundation:** Built upon **industry-leading open-source frameworks**, ServerSage offers the unparalleled reliability and flexibility required for modern, dynamic IT operations.
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

class ChartPlottingRequest(BaseModel):
    """
    Request payload for plotting charts using direct SVG generation.
    Supports 'line', 'bar', 'scatter', 'area', 'stacked_bar', 'horizontal_bar' chart types.
    """
    # x_data can be numbers (for quantitative axes like scatter) or strings (for categorical labels)
    x_data: List[Union[float, str]] = Field(..., description="Data points for the X-axis. Can be numerical for quantitative axes or strings for categorical labels.")
    
    # y_data_series is always a list of lists, where each inner list is a series/layer.
    # For single series charts (line, scatter, basic bar), provide as [[value1, value2, ...]]
    y_data_series: List[List[float]] = Field(..., description="List of Y-axis data series. Each inner list is a series.")
    
    # Updated to include all 6 supported chart types
    chart_type: Literal["line", "bar", "scatter", "area", "stacked_bar", "horizontal_bar"] = Field("line", description="Type of chart to generate.")
    
    title: Optional[str] = Field("Generated Chart", description="Title of the chart.")
    x_label: Optional[str] = Field("X-Axis", description="Label for the X-axis.")
    y_label: Optional[str] = Field("Y-Axis", description="Label for the Y-axis.")
    
    # Optional: labels for multiple series (e.g., for legend)
    series_labels: Optional[List[str]] = Field(None, description="Optional labels for each data series, matching the order of y_data_series. Used for legend.") 

    # Aesthetic parameters for direct SVG
    width: Optional[int] = Field(800, description="Width of the SVG chart in pixels.")
    height: Optional[int] = Field(500, description="Height of the SVG chart in pixels.")
    
    line_width: Optional[float] = Field(2.0, description="Line width for line/area charts.")
    
    # For line/scatter charts: whether to draw markers
    marker_style: Optional[bool] = Field(False, description="Whether to draw markers on line/scatter charts. (True/False).")

    # For area charts
    fill_opacity: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Transparency for filled areas (0.0 to 1.0).")

    # For bar charts (bar, stacked_bar, horizontal_bar)
    bar_gap_ratio: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Ratio of gap between bars to bar width within a category (0.0 to 1.0).")
    
    # Control for X-axis scaling: if x_data contains numbers but should be treated categorically, set to False.
    x_axis_is_numeric: bool = Field(False, description="If true, x_data will be treated as numerical values for scaling (e.g., for true scatter plots); otherwise, it's treated as categorical labels where points are evenly spaced. Applicable for 'line' and 'scatter'.")


# --- Updated Plotting Endpoint ---
@router.post("/plot_chart", response_class=HTMLResponse)
async def plot_chart_html(request: ChartPlottingRequest):
    """
    Generates an SVG chart (line, bar, scatter, area, stacked_bar, horizontal_bar) 
    based on provided data, labels, and dimensions.
    Returns the SVG content embedded in basic HTML.
    """
    if not request.y_data_series or not request.x_data:
        raise HTTPException(status_code=400, detail="Missing x_data or y_data_series.")
    
    # Validate lengths
    for series_index, y_values in enumerate(request.y_data_series):
        if len(y_values) != len(request.x_data):
            raise HTTPException(
                status_code=400, 
                detail=f"Length of y_data_series[{series_index}] ({len(y_values)}) must match length of x_data ({len(request.x_data)})."
            )
    if request.series_labels and len(request.series_labels) != len(request.y_data_series):
        raise HTTPException(status_code=400, detail="Number of series_labels must match y_data_series.")

    chart_type = request.chart_type.lower()
    supported_chart_types = ["line", "bar", "scatter", "area", "stacked_bar", "horizontal_bar"]
    if chart_type not in supported_chart_types:
        raise HTTPException(status_code=400, detail=f"Chart type '{chart_type}' is not supported. Supported types: {', '.join(supported_chart_types)}.")

    # Define SVG dimensions and padding
    svg_width = request.width if request.width is not None else 800
    svg_height = request.height if request.height is not None else 500
    padding = 60 # Padding for axes and labels

    # --- Determine Axis Scaling ---
    # For horizontal bar charts, swap width/height roles for scaling
    is_horizontal = (chart_type == "horizontal_bar")
    plot_width = svg_width - 2 * padding
    plot_height = svg_height - 2 * padding

    # X-axis scaling (dynamic based on x_axis_is_numeric)
    x_min_val, x_max_val = 0, len(request.x_data) - 1 # Default for categorical x-axis (index-based)
    x_is_numeric_scale = request.x_axis_is_numeric and all(isinstance(val, (int, float)) for val in request.x_data)

    if x_is_numeric_scale:
        x_numeric_data = [float(x) for x in request.x_data]
        if x_numeric_data:
            x_min_val = min(x_numeric_data)
            x_max_val = max(x_numeric_data)
            # Add some buffer to x-axis range
            x_range_buffer = (x_max_val - x_min_val) * 0.1
            if x_range_buffer == 0: # Handle single point or all same x value
                x_range_buffer = 1.0 # Default buffer for single point
            x_min_val -= x_range_buffer
            x_max_val += x_range_buffer
            if x_max_val == x_min_val: x_max_val += 1.0 # Ensure range for single point
        else: # If x_axis_is_numeric is true but x_data is empty, fallback
            x_min_val, x_max_val = 0, 1


    # Y-axis scaling (find min/max across all series)
    all_y_values = [item for sublist in request.y_data_series for item in sublist]
    if not all_y_values:
        y_min_val, y_max_val = 0, 1 # Default if no data
    elif chart_type in ["stacked_bar", "stacked_area"]: # Note: stacked_area is commented out in this version.
        # Calculate max cumulative sum for stacked charts
        max_positive_cumulative_y = 0
        min_negative_cumulative_y = 0
        for i in range(len(request.x_data)):
            current_positive_sum = sum(y_series[i] for y_series in request.y_data_series if y_series[i] >= 0)
            current_negative_sum = sum(y_series[i] for y_series in request.y_data_series if y_series[i] < 0)
            if current_positive_sum > max_positive_cumulative_y:
                max_positive_cumulative_y = current_positive_sum
            if current_negative_sum < min_negative_cumulative_y:
                min_negative_cumulative_y = current_negative_sum
        
        y_min_val = min_negative_cumulative_y # Can go negative for stacked
        y_max_val = max_positive_cumulative_y

    else:
        y_min_val = min(all_y_values)
        y_max_val = max(all_y_values)

    y_range_buffer = (y_max_val - y_min_val) * 0.1
    if y_range_buffer == 0: # All identical values
        y_max_scaled = y_max_val + 1.0 if y_max_val == 0 else y_max_val * 1.1
        y_min_scaled = y_min_val - 1.0 if y_min_val == 0 else y_min_val * 0.9
        if y_max_scaled == y_min_scaled: # Still identical, force a range
            y_max_scaled += 1.0
            y_min_scaled -= 1.0
    else:
        y_min_scaled = y_min_val - y_range_buffer
        y_max_scaled = y_max_val + y_range_buffer
    
    # Ensure range is not zero if single point or flat line
    if y_max_scaled == y_min_scaled:
        y_max_scaled += 1.0
        y_min_scaled -= 1.0


    # Scaling functions to map data coordinates to SVG pixel coordinates
    # For horizontal bar, x_data maps to Y-axis and y_data maps to X-axis
    def scale_x_coord(val, for_axis_label=False): # Maps x_data value/index to SVG x-coordinate
        if is_horizontal: # For horizontal bar, x_data maps to vertical position (y-axis)
            idx = request.x_data.index(val) if not isinstance(val, int) else val
            # Calculate position on Y-axis based on category index
            return padding + plot_height - (idx / (len(request.x_data) - 1 if len(request.x_data) > 1 else 1)) * plot_height 
        else: # Normal x-axis
            if x_is_numeric_scale and not for_axis_label: # For actual data points (scatter/line)
                return padding + ((val - x_min_val) / (x_max_val - x_min_val)) * plot_width
            else: # Categorical x-axis or for label positioning
                idx = request.x_data.index(val) if not isinstance(val, int) else val
                return padding + (idx / (len(request.x_data) - 1 if len(request.x_data) > 1 else 1)) * plot_width

    def scale_y_coord(val, for_axis_label=False): # Maps y_data value to SVG y-coordinate
        if is_horizontal: # For horizontal bar, y_data maps to horizontal position (x-axis)
            return padding + ((val - y_min_scaled) / (y_max_scaled - y_min_scaled)) * plot_width
        else: # Normal y-axis
            return (svg_height - padding) - ((val - y_min_scaled) / (y_max_scaled - y_min_scaled)) * plot_height

    svg_elements = []
    
    # Define colors for series
    colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#00bcd4", "#e91e63", "#66aa00", "#b82912", "#31659a"]

    # Draw Axes Lines
    # Horizontal charts swap X and Y axis conceptual roles for drawing.
    # The 'y1' of the line for horizontal charts (Y-axis for categories) runs from top_padding to bottom_padding.
    # The 'x1' of the line for horizontal charts (X-axis for values) runs from left_padding to right_padding.
    
    # Main axes (based on orientation)
    if is_horizontal:
        # X-axis (value axis)
        svg_elements.append(f'<line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>')
        # Y-axis (category axis)
        svg_elements.append(f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>')
    else:
        # Y-axis
        svg_elements.append(f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>')
        # X-axis
        svg_elements.append(f'<line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>')

    # --- Axis Labels and Ticks ---
    # X-axis (labels and ticks)
    if not is_horizontal: # Normal vertical chart X-axis
        num_x_labels = len(request.x_data)
        for i, label_val in enumerate(request.x_data):
            x_pos = scale_x_coord(label_val if x_is_numeric_scale else i, for_axis_label=True)
            svg_elements.append(f'<text x="{x_pos}" y="{svg_height - padding + 15}" text-anchor="middle" font-size="10">{label_val}</text>')
            svg_elements.append(f'<line x1="{x_pos}" y1="{svg_height - padding}" x2="{x_pos}" y2="{svg_height - padding + 5}" stroke="black" stroke-width="1"/>')
            svg_elements.append(f'<line x1="{x_pos}" y1="{padding}" x2="{x_pos}" y2="{svg_height - padding}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')
    
    # Y-axis (labels and ticks)
    num_y_ticks = 5 # Number of ticks for the value axis
    for i in range(num_y_ticks):
        y_tick_val = y_min_scaled + (i / (num_y_ticks - 1)) * (y_max_scaled - y_min_scaled) if num_y_ticks > 1 else y_min_scaled
        
        if is_horizontal: # X-axis is value axis (y_label in request)
            x_tick_pos = scale_y_coord(y_tick_val, for_axis_label=True) # Scale y_tick_val to X-pixel for horizontal
            svg_elements.append(f'<text x="{x_tick_pos}" y="{svg_height - padding + 15}" text-anchor="middle" font-size="10">{y_tick_val:.2f}</text>')
            svg_elements.append(f'<line x1="{x_tick_pos}" y1="{svg_height - padding}" x2="{x_tick_pos}" y2="{svg_height - padding + 5}" stroke="black" stroke-width="1"/>')
            svg_elements.append(f'<line x1="{x_tick_pos}" y1="{padding}" x2="{x_tick_pos}" y2="{svg_height - padding}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')
        else: # Normal Y-axis (y_label in request)
            y_pos = scale_y_coord(y_tick_val, for_axis_label=True)
            svg_elements.append(f'<text x="{padding - 10}" y="{y_pos + 4}" text-anchor="end" font-size="10">{y_tick_val:.2f}</text>')
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos}" x2="{padding - 5}" y2="{y_pos}" stroke="black" stroke-width="1"/>')
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos}" x2="{svg_width - padding}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')

    # Categorical Axis Labels (X-axis for vertical, Y-axis for horizontal)
    if is_horizontal: # Y-axis is now categorical
        # Calculate positions for categorical axis (y-axis for horizontal bar chart)
        num_categories = len(request.x_data)
        for i, label_val in enumerate(request.x_data):
            y_pos_category = padding + plot_height - (i / (num_categories - 1 if num_categories > 1 else 1)) * plot_height
            svg_elements.append(f'<text x="{padding - 10}" y="{y_pos_category + 4}" text-anchor="end" font-size="10">{label_val}</text>')
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos_category}" x2="{padding - 5}" y2="{y_pos_category}" stroke="black" stroke-width="1"/>')
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos_category}" x2="{svg_width - padding}" y2="{y_pos_category}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')


    # --- Plot Data Series ---
    
    # Calculate zero line position for bar/area charts
    # For vertical charts, this is a Y-coordinate. For horizontal, it's an X-coordinate.
    zero_val_pos = scale_y_coord(0) 

    if chart_type in ["line", "scatter", "area"]:
        for i, y_values in enumerate(request.y_data_series):
            current_color = colors[i % len(colors)]
            points = []
            
            if chart_type == "area":
                # Start path from the zero line (or min_scaled) at the first x-point
                x_start_area = scale_x_coord(request.x_data[0] if x_is_numeric_scale else 0)
                points.append(f"{x_start_area},{zero_val_pos}")

            for j, y_val in enumerate(y_values):
                x_val_for_point = request.x_data[j] if x_is_numeric_scale else j
                x_pos = scale_x_coord(x_val_for_point)
                y_pos = scale_y_coord(y_val)
                points.append(f"{x_pos},{y_pos}")

                if chart_type == "scatter" or (chart_type == "line" and request.marker_style):
                    svg_elements.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="4" fill="{current_color}" stroke="none"/>')

            if points:
                if chart_type == "line":
                    path_d = "M " + " L ".join(points)
                    svg_elements.append(f'<path d="{path_d}" stroke="{current_color}" stroke-width="{request.line_width}" fill="none"/>')
                elif chart_type == "area":
                    # Close the path back to the zero line at the last x-point
                    x_end_area = scale_x_coord(request.x_data[-1] if x_is_numeric_scale else len(request.x_data) - 1)
                    points.append(f"{x_end_area},{zero_val_pos}")
                    path_d = "M " + " L ".join(points)
                    svg_elements.append(f'<path d="{path_d}" fill="{current_color}" fill-opacity="{request.fill_opacity}" stroke="{current_color}" stroke-width="{request.line_width}"/>')
    
    elif chart_type in ["bar", "stacked_bar"]:
        num_categories = len(request.x_data)
        total_series = len(request.y_data_series)
        
        # Calculate bar dimensions for vertical bars
        category_total_width = plot_width / num_categories
        bar_gap = category_total_width * (request.bar_gap_ratio if request.bar_gap_ratio is not None else 0.2)
        
        if chart_type == "bar": # Grouped bars
            bar_width = (category_total_width - bar_gap) / total_series
        else: # Stacked bars
            bar_width = category_total_width - bar_gap # Each stack occupies this width

        bar_width = max(1, bar_width) # Ensure minimum bar width

        # Store current accumulated heights for stacked bars
        if chart_type == "stacked_bar":
            current_y_stack = [0.0] * num_categories # For positive stacks
            current_y_stack_neg = [0.0] * num_categories # For negative stacks

        for j in range(num_categories): # Iterate through each x_data point (category)
            x_pos_category_center = scale_x_coord(j, for_axis_label=True) # Center of the current x-category
            
            if chart_type == "bar": # Grouped bars
                # Calculate the starting X position for the first bar in this group
                group_width_effective = (bar_width * total_series) + (bar_gap * (total_series - 1) * 0.5) # Slight adjustment for group centering
                group_start_x = x_pos_category_center - (group_width_effective / 2)
                
                for i, y_values in enumerate(request.y_data_series): # Iterate through each series
                    y_val = y_values[j]
                    current_color = colors[i % len(colors)]
                    
                    x_rect_start = group_start_x + (i * bar_width)
                    
                    y_pos_top = scale_y_coord(y_val)
                    bar_y_start = min(y_pos_top, zero_val_pos)
                    bar_height = abs(y_pos_top - zero_val_pos)

                    if bar_height < 0: bar_height = 0
                    svg_elements.append(f'<rect x="{x_rect_start}" y="{bar_y_start}" width="{bar_width}" height="{bar_height}" fill="{current_color}" stroke="black" stroke-width="0.5"/>')
            
            elif chart_type == "stacked_bar":
                x_rect_start = x_pos_category_center - (bar_width / 2)
                for i, y_values in enumerate(request.y_data_series):
                    y_val = y_values[j]
                    current_color = colors[i % len(colors)]

                    if y_val >= 0:
                        y_pos_top = scale_y_coord(current_y_stack[j] + y_val)
                        y_pos_bottom = scale_y_coord(current_y_stack[j])
                        bar_y_start = y_pos_top
                        bar_height = y_pos_bottom - y_pos_top
                        current_y_stack[j] += y_val
                    else: # Negative stack
                        y_pos_top = scale_y_coord(current_y_stack_neg[j])
                        y_pos_bottom = scale_y_coord(current_y_stack_neg[j] + y_val)
                        bar_y_start = y_pos_top
                        bar_height = y_pos_bottom - y_pos_top
                        current_y_stack_neg[j] += y_val

                    if bar_height < 0: bar_height = 0 # Ensure non-negative height
                    svg_elements.append(f'<rect x="{x_rect_start}" y="{bar_y_start}" width="{bar_width}" height="{bar_height}" fill="{current_color}" stroke="black" stroke-width="0.5"/>')

    elif chart_type == "horizontal_bar":
        num_categories = len(request.x_data)
        total_series = len(request.y_data_series)
        
        category_total_height = plot_height / num_categories
        bar_gap = category_total_height * (request.bar_gap_ratio if request.bar_gap_ratio is not None else 0.2)

        bar_height_per_series = (category_total_height - bar_gap) / total_series
        bar_height_per_series = max(1, bar_height_per_series) # Ensure positive height

        for j in range(num_categories): # Iterate through each x_data point (category for Y-axis)
            y_pos_category_center = scale_x_coord(request.x_data[j], for_axis_label=True) # Center of the current y-category (which is x_data)
            
            # Calculate the starting Y position for the first bar in this group
            group_height_effective = (bar_height_per_series * total_series) + (bar_gap * (total_series - 1) * 0.5)
            group_start_y = y_pos_category_center - (group_height_effective / 2)

            for i, y_values in enumerate(request.y_data_series): # Iterate through each series
                y_val = y_values[j]
                current_color = colors[i % len(colors)]
                
                y_rect_start = group_start_y + (i * bar_height_per_series)
                
                x_pos_end = scale_y_coord(y_val) # X-position of the bar's end (value axis)
                bar_x_start = min(x_pos_end, zero_val_pos)
                bar_width = abs(x_pos_end - zero_val_pos)

                if bar_width < 0: bar_width = 0
                svg_elements.append(f'<rect x="{bar_x_start}" y="{y_rect_start}" width="{bar_width}" height="{bar_height_per_series}" fill="{current_color}" stroke="black" stroke-width="0.5"/>')


    # --- Axis Labels & Title ---
    if is_horizontal:
        svg_elements.append(f'<text x="{svg_width / 2}" y="{svg_height - 10}" text-anchor="middle" font-size="14">{request.y_label}</text>') # X-axis (values)
        svg_elements.append(f'<text x="{15}" y="{svg_height / 2}" text-anchor="middle" transform="rotate(-90 {15},{svg_height / 2})" font-size="14">{request.x_label}</text>') # Y-axis (categories)
    else:
        svg_elements.append(f'<text x="{svg_width / 2}" y="{svg_height - 10}" text-anchor="middle" font-size="14">{request.x_label}</text>')
        svg_elements.append(f'<text x="{15}" y="{svg_height / 2}" text-anchor="middle" transform="rotate(-90 {15},{svg_height / 2})" font-size="14">{request.y_label}</text>')

    # Title
    svg_elements.append(f'<text x="{svg_width / 2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{request.title}</text>')

    # --- Basic Legend (optional, for multiple series) ---
    if request.series_labels and len(request.y_data_series) > 0:
        legend_start_x = svg_width - 150
        legend_start_y = 50
        for i, label in enumerate(request.series_labels):
            color = colors[i % len(colors)]
            svg_elements.append(f'<rect x="{legend_start_x}" y="{legend_start_y + i * 20}" width="15" height="15" fill="{color}"/>')
            svg_elements.append(f'<text x="{legend_start_x + 20}" y="{legend_start_y + i * 20 + 12}" font-size="12">{label}</text>')


    # Combine all SVG elements into a full SVG string
    svg_content = f"""
    <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white"/>
        {" ".join(svg_elements)}
    </svg>
    """

    # Final HTML with embedded SVG
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{request.title}</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; margin: 20px; }}
            svg {{ border: 1px solid #ccc; background-color: #f9f9f9; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>{request.title}</h1>
        {svg_content}
    </body>
    </html>
    """
    return HTMLResponse(content=html)