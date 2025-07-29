from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, Request
from fastapi.responses import HTMLResponse # For a simple root endpoint if needed
from typing import List
import uvicorn
import json
from datetime import datetime, timezone # Ensure datetime is imported for broadcast data

from Routers import model_management  # Corrected import path (lowercase 'routers')

# -------------------------------------------------------------
# WebSocket Manager Class
# -------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected: {websocket.client}")
        except ValueError:
            # Handle case where websocket might already be removed (e.g., if connection broke abruptly)
            pass

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        # Create a list of connections to potentially remove
        connections_to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except RuntimeError as e:
                # This exception often indicates a closed connection (e.g., "WebSocket is not connected")
                print(f"Error sending to WebSocket {connection.client}: {e}. Marking for removal.")
                connections_to_remove.append(connection)
            except Exception as e:
                print(f"Unexpected error broadcasting to WebSocket {connection.client}: {e}. Marking for removal.")
                connections_to_remove.append(connection)

        # Remove dead connections outside the loop to avoid modifying list during iteration
        for connection in connections_to_remove:
            self.active_connections.remove(connection)
            print(f"Removed dead WebSocket connection: {connection.client}")


# -------------------------------------------------------------
# FastAPI Application Setup
# -------------------------------------------------------------
app = FastAPI(
    title="Other Anomaly Detection Processes Server",
    description="Handles Data Ingestion, Model Management (with 2 slots), Training, and Visualization."
)

# Instantiate ConnectionManager and store it in app.state
# This makes the manager accessible from any endpoint via request.app.state.websocket_manager
app.state.websocket_manager = ConnectionManager()

# Include routers
app.include_router(model_management.router)

# -------------------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------------------
@app.websocket("/ws/alerts") # Frontend will connect to ws://your-server-ip:3000/ws/alerts
async def websocket_endpoint(websocket: WebSocket):
    # Get the manager instance from app.state
    manager = app.state.websocket_manager
    await manager.connect(websocket)
    try:
        # This loop keeps the connection open.
        # For an alerts-only system, the client might not send messages,
        # but the loop needs to exist to keep the connection alive.
        while True:
            # You can optionally receive messages from the client here if needed
            # data = await websocket.receive_text()
            # print(f"Received message from client: {data}")
            # await manager.send_personal_message(f"You said: {data}", websocket)
            await websocket.receive_text() # This will just wait for any incoming message or disconnect
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error in /ws/alerts: {e}")
        manager.disconnect(websocket)

# -------------------------------------------------------------
# Standard HTTP Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "connections": len(app.state.websocket_manager.active_connections)}

@app.get("/")
async def read_root():
    return {"message": "Other Processes Server is running!"}

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 3000
    print(f"Starting Other Processes Server on http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)