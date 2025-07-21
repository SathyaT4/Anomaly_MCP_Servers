# other_processes_server/routers/model_management.py
import os
import shutil
import regex as re
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from typing import List
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

gmm_detector_for_training = GMMAnomalyDetector() # Use a separate instance for training operations

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


@router.post("/load_model_into_slot")
async def load_model_into_slot(
    request: ModelLoadRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Loads a model artifact into an available slot, or a specified slot.
    Uses LRU (Least Recently Used) policy if all slots are full.
    """
    result = model_manager.load_model_into_slot(
        model_id=request.model_id,
        model_artifact_path=request.model_artifact_path,
        target_slot_id=request.target_slot_id
    )
    return {"status": "success", **result}

@router.get("/get_loaded_models_info", response_model=List[ModelInfo]) # type: ignore
async def get_loaded_models_info(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Retrieves information about all models currently loaded into slots."""
    info = model_manager.get_loaded_models_info()
    return info

@router.post("/set_active_session_model")
async def set_active_session_model(
    request: SetActiveModelRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Designates a model as 'active' for a specific session."""
    model_manager.set_active_session_model(request.session_id, request.model_id)
    return {"status": "success", "active_model_id": request.model_id, "message": f"Model '{request.model_id}' set as active for session '{request.session_id}'."}
# --- Data Transformation Helper Function ---
# This function centralizes the logic for parsing 'labels' and pivoting data.
def parse_labels_and_transform_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw data from 'long' format (timestamp, value, metric, labels)
    to 'wide' format (timestamp, metric_1, metric_2, ...).
    """
    try:
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

        # Function to parse labels string into a dictionary
        def parse_labels_string(labels_str):
            labels_dict = {}
            # Regex to find key=value pairs, handling quoted values if necessary
            matches = re.findall(r'(\w+)=([^,"]*)', labels_str)
            for key, value in matches:
                labels_dict[key] = value
            return labels_dict

        parsed_labels = df_raw['labels'].apply(parse_labels_string)

        # Extract common labels. Add more .get() calls here if other common labels are expected.
        df_raw['host'] = parsed_labels.apply(lambda x: x.get('host', 'unknown'))
        df_raw['dc'] = parsed_labels.apply(lambda x: x.get('dc', 'unknown'))
        df_raw['mt'] = parsed_labels.apply(lambda x: x.get('mt', 'unknown')) # 'mt' (metric type)

        # Construct a unique full metric name for each combination
        # This makes the transformation generic to any metric and label combination.
        df_raw['full_metric_name'] = df_raw['metric'] + '_' + \
                                     df_raw['mt'] + '_' + \
                                     df_raw['dc'] + '_' + \
                                     df_raw['host']

        # Pivot the table to get metrics as columns
        df_transformed = df_raw.pivot_table(
            index='timestamp',
            columns='full_metric_name',
            values='value',
            aggfunc='mean' # Aggregate duplicate timestamps by mean value
        )

        df_transformed = df_transformed.reset_index()
        df_transformed.columns.name = None # Remove the 'full_metric_name' column name from the DataFrame

        print("Data transformed from 'long' to 'wide' format successfully.")
        return df_transformed

    except Exception as e:
        # Re-raise as HTTPException or log and raise a more generic error
        raise ValueError(f"Error during data transformation: {e}")


# --- API Endpoints ---

@router.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Allows users to upload a data file (e.g., CSV).
    The server saves it and returns the path for subsequent training.
    """
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    try:
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "message": f"File '{file.filename}' uploaded successfully.", "file_path": file_location}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not upload file: {e}")

# This is the modified endpoint you asked for:
@router.post("/train_anomaly_model", response_model=TrainModelResponse)
async def train_anomaly_model(params: TrainModelParams):
    """
    Trains a new anomaly detection model using data from either a specified path
    or a previously uploaded file.
    Automatically transforms raw 'long' format data (timestamp, value, metric, labels)
    into the required 'wide' format before training.
    """
    raw_data_path = None

    if params.data_source_path:
        raw_data_path = params.data_source_path
    elif params.uploaded_file_name:
        raw_data_path = os.path.join(UPLOAD_DIRECTORY, params.uploaded_file_name)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'data_source_path' or 'uploaded_file_name' must be provided.")

    if not os.path.exists(raw_data_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Data source file not found: {raw_data_path}")

    try:
        df_raw = pd.read_csv(raw_data_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read data source '{raw_data_path}': {e}")

    # --- Intelligent Data Loading and Transformation ---
    df_final_for_training: pd.DataFrame
    # Check if the raw data appears to be in the 'long' format (e.g., cpu_busy_percentage)
    if all(col in df_raw.columns for col in ['timestamp', 'value', 'metric', 'labels']):
        print(f"Server: Raw data from '{raw_data_path}' detected as 'long' format. Applying transformation...")
        try:
            df_final_for_training = parse_labels_and_transform_data(df_raw.copy()) # Pass a copy to avoid modifying original df_raw
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data transformation failed: {e}")
    else:
        # Assume data is already in the 'wide' format (e.g., DCS1-Ares_metrics.csv)
        print(f"Server: Data from '{raw_data_path}' appears to be in 'wide' format. Using directly for training.")
        df_final_for_training = df_raw
        # Ensure timestamp is datetime, even if already wide format
        df_final_for_training['timestamp'] = pd.to_datetime(df_final_for_training['timestamp'])

    # --- Algorithm Check ---
    if params.algorithm.lower() != "gmm":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'GMM' algorithm is supported for training at the moment.")

    # --- Model Training ---
    try:
        print(f"Server: Initiating GMM training with {len(df_final_for_training)} records...")
        training_results = gmm_detector_for_training.train_model(
            df_original=df_final_for_training, # Pass the transformed (or direct) DataFrame
            n_components=params.n_components,
            anomaly_threshold_percentile=params.anomaly_threshold_percentile,
            model_id=params.model_id
        )

        print(f"Server: Model training completed for model_id: {training_results['model_id']}")
        return TrainModelResponse(
            status="success",
            message="Model training completed and artifacts stored.",
            model_id=training_results["model_id"],
            trained_model_path=training_results["trained_model_path"],
            anomaly_report_path=training_results["anomaly_report_path"],
            plots_path=training_results["plots_path"] # This will likely be None if plots are not generated
        )
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full traceback for debugging server-side
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during model training: {e}. Check server logs for details.")

# --- Keep existing endpoints below ---
@router.post("/retrain_model")
async def retrain_model(params: RetrainModelParams):
    """Simulates retraining an existing model."""
    print(f"SERVER 2 - Router: Simulating retraining model {params.model_id} with new data {params.new_data_id}")
    # TODO: Call a service method here
    return {"status": "success", "updated_model_id": f"{params.model_id}_retrained", "message": "Model retraining simulated."}

# Import Pydantic models




@router.post("/train_anomaly_model", response_model=TrainModelResponse)
async def train_anomaly_model(request: TrainModelRequest):
    """
    Trains a new anomaly detection model using the specified algorithm and data source.
    Stores the trained model, anomaly reports, and plots.
    """
    if not os.path.exists(request.data_source_path):
        raise HTTPException(status_code=404, detail=f"Data source file not found: {request.data_source_path}")

    try:
        df_original = pd.read_csv(request.data_source_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data source: {e}")

    if request.algorithm.lower() != "gmm":
        raise HTTPException(status_code=400, detail="Only 'GMM' algorithm is supported for training at the moment.")

    try:
        # Pass the pre-initialized gmm_detector_for_training instance
        training_results = gmm_detector_for_training.train_model(
            df_original=df_original,
            n_components=request.n_components,
            anomaly_threshold_percentile=request.anomaly_threshold_percentile,
            model_id=request.model_id
        )

        return TrainModelResponse(
            status="success",
            message="Model training completed and artifacts stored.",
            model_id=training_results["model_id"],
            trained_model_path=training_results["trained_model_path"],
            anomaly_report_path=training_results["anomaly_report_path"],
            plots_path=training_results["plots_path"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {e}")


# @router.post("/load_model_into_slot", response_model=ModelLoadResponse)
# async def load_model_into_slot(request: ModelLoadRequest):
#     """
#     Loads a trained model into an available model slot.
#     Applies LRU policy if all slots are occupied.
#     """
#     # Check if the model artifact base path exists
#     if not os.path.isdir(request.model_artifact_path):
#         raise HTTPException(status_code=404, detail=f"Model artifact directory not found: {request.model_artifact_path}")
    
#     # The ModelManager itself handles the loading of the GMMAnomalyDetector instance
#     result = model_manager.load_model_into_slot(
#         model_id=request.model_id,
#         model_artifact_base_path=request.model_artifact_path,
#         target_slot_id=request.target_slot_id
#     )
#     if result["status"] == "error":
#         raise HTTPException(status_code=500, detail=result["message"])
    
#     return ModelLoadResponse(
#         status=result["status"],
#         slot_id=result["slot_id"],
#         loaded_model_id=result["loaded_model_id"],
#         message=result.get("message", "Model loaded successfully.")
#     )

# @router.get("/mcp/model_management/get_loaded_models_info", response_model=List[ModelInfo])
# async def get_loaded_models_info():
#     """
#     Retrieves information about currently loaded models in slots.
#     """
#     return model_manager.get_loaded_models_info()

# @router.post("\/set_active_session_model", response_model=SetActiveSessionModelResponse)
# async def set_active_session_model(request: SetActiveSessionModelRequest):
#     """
#     Sets a specific loaded model as active for a given session.
#     """
#     result = model_manager.set_active_session_model(request.session_id, request.model_id)
#     if result["status"] == "error":
#         raise HTTPException(status_code=400, detail=result["message"])
#     return SetActiveSessionModelResponse(status=result["status"], message=result["message"])

# @router.post("/mcp/model_management/retrain_model", response_model=RetrainModelResponse)
# async def retrain_model(request: RetrainModelRequest):
#     """
#     Simulates retraining an existing model with new data/feedback.
#     In a real scenario, this would trigger a new training run and update the model.
#     """
#     # For now, this is a simulation. A real retraining would involve:
#     # 1. Fetching old model parameters/features.
#     # 2. Loading new data (`request.new_data_source_path`).
#     # 3. Running a training process (similar to `train_anomaly_model`).
#     # 4. Saving the new model, potentially with a version increment.
#     # 5. User would then call load_model_into_slot with the new model ID.

#     # Simulating a new model ID for the retrained model
#     retrained_model_id = f"{request.model_id}_retrained_{int(time.time())}" 
    
#     # You could actually call the GMMAnomalyDetector.train_model here with new data
#     # For simplicity of this demo, we'll just acknowledge the request.
#     print(f"Retraining simulated for model {request.model_id} with new data from {request.new_data_source_path}")
#     print(f"Feedback: {request.feedback}")
    
#     # In a full implementation, you'd perform the retraining logic here
#     # and get actual trained_model_path, anomaly_report_path, plots_path etc.
    
#     return RetrainModelResponse(
#         status="success",
#         message=f"Retraining request for model '{request.model_id}' received. New model ID: '{retrained_model_id}'. (Simulation)",
#         retrained_model_id=retrained_model_id,
#         new_model_path=f"trained_models/{retrained_model_id}", # Placeholder
#         new_anomaly_report_path=f"anomaly_reports/{retrained_model_id}/report.csv" # Placeholder
#     )


# @router.post("/mcp/model_management/predict_anomaly", response_model=AnomalyDetectionResponse)
# async def predict_anomaly(request: AnomalyDetectionRequest):
#     """
#     Performs real-time anomaly detection on a single data point using a loaded model.
#     NOTE: While placed in MCP for demonstration, this endpoint is typically
#     part of a separate 'Real-Time Anomaly Detection Server' for low-latency inference.
#     """
#     model_instance = model_manager.get_model_instance(request.model_id)
#     if model_instance is None:
#         raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found or not active in any slot.")
    
#     # Convert incoming data_point (dict) to a pandas DataFrame
#     # Ensure correct feature order/columns based on what the model expects
#     data_point_df = pd.DataFrame([request.data_point])

#     try:
#         prediction_results = model_instance.predict_anomaly(data_point_df)
        
#         # Ensure timestamp is a datetime object
#         timestamp = pd.to_datetime(request.data_point.get('timestamp', datetime.now()))


#         return AnomalyDetectionResponse(
#             status="success",
#             message="Anomaly detection performed successfully.",
#             result=AnomalyResult(
#                 timestamp=timestamp,
#                 anomaly_score=prediction_results['anomaly_score'],
#                 is_anomaly=prediction_results['is_anomaly'],
#                 cluster_id=prediction_results['cluster_id'],
#                 top_contributing_metrics=prediction_results['top_contributing_metrics']
#             )
#         )
#     except Exception as e:
#         import traceback
#         traceback.print_exc() # Print full traceback for debugging
#         raise HTTPException(status_code=500, detail=f"Error during anomaly prediction: {e}")