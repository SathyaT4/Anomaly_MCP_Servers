# other_processes_server/schemas/model_management.py
from typing import Optional, Literal
from pydantic import BaseModel, Field, root_validator
from typing import Dict, Any, List, Optional
from datetime import datetime

class ModelLoadRequest(BaseModel):
    model_id: str
    model_artifact_path: str # Path to the saved model file/directory
    target_slot_id: Optional[int] = None

class SetActiveModelRequest(BaseModel):
    model_id: str
    session_id: str

class TrainModelParams(BaseModel):
    algorithm: str
    cluster_range: Optional[List[int]] = None
    model_id: Optional[str] = None # If retraining/updating existing

class RetrainModelParams(BaseModel):
    model_id: str
    new_data_id: str
    feedback: Optional[List[Dict[str, Any]]] = None

# For responses
class ModelInfo(BaseModel):
    slot_id: int
    model_id: Optional[str]
    status: str
    last_used: Optional[float]



class TrainModelRequest(BaseModel):
    """
    Request body for training a new anomaly‑detection model.
    - Either supply `n_components` manually,
      OR set `auto_n_components=True` to let the server choose.
    """
    # Path to CSV on disk (required)
    data_source_path: str = Field(..., example="data/DCS1-Ares_metrics.csv")

    # Optional user‑defined model identifier
    model_id: Optional[str] = Field(
        default=None, example="gmm_model_2025_july"
    )

    # Only 'gmm' supported for now (case‑insensitive)
    algorithm: Literal["gmm", "GMM"] = "gmm"

    # Manual K (optional if auto_n_components=True)
    n_components: Optional[int] = Field(
        default=None, ge=1, example=4,
        description="Number of Gaussian components (ignored when auto_n_components=True)"
    )

    # Turn on auto‑K selection via AIC/BIC
    auto_n_components: bool = Field(
        default=False,
        description="If True, server will pick the best K using `criterion`"
    )

    # Which metric to minimise when auto‑selecting K
    criterion: Literal["bic", "aic"] = Field(
        default="bic", description="Metric used when auto_n_components=True"
    )

    # Percentile for anomaly threshold
    anomaly_threshold_percentile: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Top‑percentile scores flagged as anomalies (0–1)"
    )

    @root_validator
    def check_k_or_auto(cls, values):
        """Ensure either n_components is provided OR auto_n_components is True."""
        n_components = values.get("n_components")
        auto_flag   = values.get("auto_n_components")
        if not auto_flag and n_components is None:
            raise ValueError(
                "Either `n_components` must be supplied, "
                "or `auto_n_components` must be set to True."
            )
        return values

class TrainModelResponse(BaseModel):
    """Response body after training a model."""
    status: str
    message: str
    model_id: str
    trained_model_path: Optional[str] = None
    anomaly_report_path: Optional[str] = None
    plots_path: Optional[str] = None

class ModelLoadRequest(BaseModel):
    """Request body for loading a model into a slot."""
    model_id: str
    model_artifact_path: str # Base path to the directory containing model artifacts
    target_slot_id: Optional[int] = None # 1 or 2

class ModelLoadResponse(BaseModel):
    """Response body after loading a model into a slot."""
    status: str
    slot_id: int
    loaded_model_id: str
    message: Optional[str] = None

class ModelInfo(BaseModel):
    """Information about a loaded model in a slot."""
    slot_id: int
    model_id: Optional[str] = None
    status: str # "empty", "active"
    last_used: Optional[float] = None # Unix timestamp

class SetActiveSessionModelRequest(BaseModel):
    """Request body to set an active model for a session."""
    session_id: str
    model_id: str

class SetActiveSessionModelResponse(BaseModel):
    """Response body after setting an active model for a session."""
    status: str
    message: str

class RetrainModelRequest(BaseModel):
    """Request body for retraining an existing model."""
    model_id: str
    new_data_source_path: str
    feedback: Optional[str] = None # e.g., "false_positive", "false_negative"

class RetrainModelResponse(BaseModel):
    """Response body after retraining a model."""
    status: str
    message: str
    retrained_model_id: str
    new_model_path: Optional[str] = None
    new_anomaly_report_path: Optional[str] = None

# New for Prediction Endpoint
class AnomalyDetectionRequest(BaseModel):
    """Request body for real-time anomaly detection."""
    model_id: str
    data_point: Dict[str, Any] # A single dictionary representing one row of metrics

class AnomalyResult(BaseModel):
    """Details about a detected anomaly."""
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    cluster_id: Optional[int] = None
    top_contributing_metrics: Optional[List[Dict[str, Any]]] = None

class AnomalyDetectionResponse(BaseModel):
    """Response body after real-time anomaly detection."""
    status: str
    message: str
    result: AnomalyResult