# cloud_server/app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import CloudServerAction, CloudServerObservation
    from cloud_server.cloud_server_environment import CloudServerEnvironment
except ImportError:
    from models import CloudServerAction, CloudServerObservation
    from cloud_server_environment import CloudServerEnvironment

os.environ["ENABLE_WEB_INTERFACE"] = "true"

# Create the main FastAPI app
app = FastAPI(title="Cloud Optimizer Pro", version="2.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = CloudServerEnvironment()

class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    task_id: str = "easy"

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(os.path.dirname(current_dir), "templates")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the enhanced UI"""
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Cloud Optimizer Pro</h1><p>Templates not found. Please ensure templates/index.html exists.</p>")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Cloud Optimizer Pro"}

@app.get("/task-info")
async def task_info():
    return {
        "tasks": ["easy", "medium", "hard"],
        "current_task": env.current_task,
        "description": {
            "easy": "Stable traffic (100-200 req/s)",
            "medium": "Oscillating traffic (120-280 req/s)",
            "hard": "Stress test with spikes (100-500 req/s)"
        }
    }

@app.post("/reset")
async def reset_env(request: ResetRequest):
    """Reset the environment"""
    observation = env.reset(request.task_id)
    return {
        "traffic": observation.traffic,
        "servers": observation.servers,
        "latency": observation.latency,
        "reward": observation.reward,
        "done": observation.done,
        "message": observation.message,
        "step": observation.step,
        "task_id": observation.task_id
    }

@app.post("/step")
async def step_env(request: StepRequest):
    """Execute a step"""
    # Validate action
    if request.action not in [0, 1, 2]:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid action. Must be 0, 1, or 2"}
        )
    
    action = CloudServerAction(action=request.action)
    observation = env.step(action)
    return {
        "traffic": observation.traffic,
        "servers": observation.servers,
        "latency": observation.latency,
        "reward": observation.reward,
        "done": observation.done,
        "message": observation.message,
        "step": observation.step,
        "task_id": observation.task_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)