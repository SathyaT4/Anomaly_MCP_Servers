# other_processes_server/main.py
from fastapi import FastAPI
# Corrected import path: assuming 'routers' directory is sibling to main.py
from Routers import model_management 
import uvicorn # Imported here for the if __name__ == "__main__": block

app = FastAPI(
    title="Other Anomaly Detection Processes Server",
    description="Handles Data Ingestion, Model Management (with 2 slots), Training, and Visualization."
)

# Include routers here
app.include_router(model_management.router)

# Health check endpoint - Uncommented and active
@app.get("/")
async def read_root():
    return {"message": "Other Processes Server is running!"}


if __name__ == "__main__":
    # Define the host and port for this server
    HOST = "0.0.0.0"  # Listen on all available network interfaces
    PORT = 8000       # Port for the "Other Processes Server"

    print(f"Starting Other Processes Server on http://{HOST}:{PORT}")
    # The 'reload=True' is great for development, automatically restarts server on code changes
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)


