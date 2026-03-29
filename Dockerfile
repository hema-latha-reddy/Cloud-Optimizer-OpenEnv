from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. THE STATE: What the AI sees
class Observation(BaseModel):
    traffic: int = 10
    servers: int = 1
    latency: int = 100

current_obs = Observation()

# 2. THE RESET: Starting the game over
@app.get("/reset")
def reset():
    global current_obs
    current_obs = Observation()
    return current_obs

# 3. THE STEP: When the AI makes a move
@app.post("/step")
def step(action: int):
    global current_obs
    # Action 0: Remove Server, 1: Stay, 2: Add Server
    if action == 0 and current_obs.servers > 1:
        current_obs.servers -= 1
    elif action == 2:
        current_obs.servers += 1
    
    # Simple logic: More servers = Lower latency
    current_obs.latency = int((current_obs.traffic / current_obs.servers) * 10)
    
    # Reward: 1.0 for good performance, 0.0 for bad
    reward = 1.0 if current_obs.latency < 200 else 0.0
    return {"observation": current_obs, "reward": reward}

# 4. THE TASKS: Required for the hackathon
@app.get("/tasks")
def tasks():
    return [{"id": "easy", "name": "Basic Scaling"}]