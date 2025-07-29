import time
from typing import Dict, Any, Optional, List
import joblib
import os

# Assuming GMMAnomalyDetector is in ml_core/gmm_anomaly_detector.py
# from Core_ML.gmm_model import GMMAnomalyDetector
from Core_ML.gmm_model import GMMAnomalyDetector

class ModelManager:
    MAX_MODEL_SLOTS = 2
    
    def __init__(self):
        # model_pool stores instances of GMMAnomalyDetector, not just raw models
        # Each GMMAnomalyDetector instance will hold its own scaler, gmm_model, pca_model
        self.model_pool: Dict[int, Dict[str, Any]] = {
            1: {"model_id": None, "instance": None, "status": "empty", "last_used": None},
            2: {"model_id": None, "instance": None, "status": "empty", "last_used": None},
        }
        self.active_sessions: Dict[str, str] = {} # session_id -> model_id

    def _find_empty_slot(self) -> Optional[int]:
        for slot_id, slot_info in self.model_pool.items():
            if slot_info["status"] == "empty":
                return slot_id
        return None

    def _find_lru_slot(self) -> int:
        lru_slot_id = None
        oldest_time = float('inf')
        for slot_id, slot_info in self.model_pool.items():
            if slot_info["last_used"] is None: # Should not happen if all are active
                 return slot_id # Should pick this if it somehow happened.
            if slot_info["last_used"] < oldest_time:
                oldest_time = slot_info["last_used"]
                lru_slot_id = slot_id
        return lru_slot_id

    def load_model_into_slot(self, model_id: str, model_artifact_base_path: str, target_slot_id: Optional[int] = None) -> Dict[str, Any]:
        
        # Check if the model is already loaded in any slot
        for slot_id, slot_info in self.model_pool.items():
            if slot_info["model_id"] == model_id and slot_info["status"] == "active":
                slot_info["last_used"] = time.time() # Update last used time
                print(f"Model '{model_id}' already active in slot {slot_id}. Last used time updated.")
                return {"status": "success", "slot_id": slot_id, "loaded_model_id": model_id, "message": "Model already loaded and active."}

        # Determine which slot to use
        chosen_slot_id = None
        if target_slot_id and self.model_pool[target_slot_id]["status"] == "empty":
            chosen_slot_id = target_slot_id
        else:
            chosen_slot_id = self._find_empty_slot()

        if chosen_slot_id is None: # No empty slots, use LRU
            chosen_slot_id = self._find_lru_slot()
            old_model_id = self.model_pool[chosen_slot_id]["model_id"]
            print(f"Slot {chosen_slot_id} is full (model '{old_model_id}'). Applying LRU policy to replace.")

        # Initialize a new GMMAnomalyDetector instance
        detector_instance = GMMAnomalyDetector()
        
        # Load model artifacts into this new instance
        if not detector_instance.load_model_artifacts(model_artifact_base_path):
            return {"status": "error", "message": f"Failed to load model artifacts from {model_artifact_base_path}"}

        # Update the chosen slot
        self.model_pool[chosen_slot_id]["model_id"] = model_id
        self.model_pool[chosen_slot_id]["instance"] = detector_instance
        self.model_pool[chosen_slot_id]["status"] = "active"
        self.model_pool[chosen_slot_id]["last_used"] = time.time()

        print(f"Model '{model_id}' loaded into slot {chosen_slot_id}.")
        return {"status": "success", "slot_id": chosen_slot_id, "loaded_model_id": model_id}

    def get_loaded_models_info(self) -> List[Dict[str, Any]]: # type: ignore
        return [
            {"slot_id": slot_id, "model_id": info["model_id"], "status": info["status"], "last_used": info["last_used"]}
            for slot_id, info in self.model_pool.items()
        ]

    def set_active_session_model(self, session_id: str, model_id: str) -> Dict[str, Any]:
        # Verify if the model is actually loaded in any slot
        model_found = False
        for slot_info in self.model_pool.values():
            if slot_info["model_id"] == model_id and slot_info["status"] == "active":
                model_found = True
                # Update last_used time for the model being set active
                slot_info["last_used"] = time.time()
                break
        
        if not model_found:
            return {"status": "error", "message": f"Model '{model_id}' is not currently loaded in any active slot."}

        self.active_sessions[session_id] = model_id
        print(f"Session '{session_id}' now uses model '{model_id}'.")
        return {"status": "success", "message": f"Model '{model_id}' set as active for session '{session_id}'."}

    def get_model_instance(self, model_id: str) -> Optional[GMMAnomalyDetector]:
        """Retrieves the GMMAnomalyDetector instance for a given model_id."""
        for slot_info in self.model_pool.values():
            if slot_info["model_id"] == model_id and slot_info["status"] == "active":
                # Update last_used when a model instance is requested (implies usage)
                slot_info["last_used"] = time.time()
                return slot_info["instance"]
        return None