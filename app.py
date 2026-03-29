from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import random

app = FastAPI()

# --- MODELS (Defining the "Shape" of  data) ---
class Observation(BaseModel):
    traffic: int
    servers: int
    latency: int
    message: str

class Task(BaseModel):
    id: str
    name: str
    description: str

# --- THE ENVIRONMENT STATE ---
class CloudEnv:
    def __init__(self):
        self.reset("easy")

    def reset(self, task_id: str):
        self.task_id = task_id
        self.servers = 2
        self.step_count = 0
        self.total_reward = 0.0
        
        if task_id == "easy":
            self.traffic = 100
        elif task_id == "medium":
            self.traffic = random.randint(200, 500) # Spiky traffic
        else: # hard
            self.traffic = 1000
        
        return self._get_obs("Environment Reset")

    def _get_obs(self, msg: str):
        return Observation(
            traffic=self.traffic,
            servers=self.servers,
            latency=int((self.traffic / self.servers) * 10),
            message=msg
        )

env = CloudEnv()

# --- CORE ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Cloud Optimizer V1 - OpenEnv Ready", "team": "Cracking"}

@app.get("/reset", response_model=Observation)
def reset_endpoint(task_id: str = "easy"):
    return env.reset(task_id)

@app.post("/step", response_model=Observation)
def step_endpoint(action: int):
    # Action 0: Remove, 1: Stay, 2: Add
    env.step_count += 1
    
    if action == 2: env.servers += 1
    elif action == 0 and env.servers > 1: env.servers -= 1
    
    # Task Logic: Medium/Hard have traffic changes
    if env.task_id != "easy":
        env.traffic += random.randint(-50, 70) 
    
    obs = env._get_obs("Step Successful")
    
    # Calculate Reward (The Grader Logic)
    # High reward if latency < 150ms AND servers < 10
    if obs.latency < 150 and obs.servers < 10:
        env.total_reward += 1.0
        
    return obs


@app.get("/tasks", response_model=List[Task])
def get_tasks():
    return [
        {"id": "easy", "name": "Steady State", "description": "Constant traffic, no surprises."},
        {"id": "medium", "name": "Traffic Spikes", "description": "Traffic changes every step."},
        {"id": "hard", "name": "Scale Crisis", "description": "Massive traffic, high difficulty."}
    ]

@app.get("/grader")
def grader():
    # Returns a score between 0.0 and 1.0 based on efficiency
    if env.step_count == 0: return {"score": 0.0}
    final_score = min(1.0, env.total_reward / env.step_count)
    return {"score": round(final_score, 2), "steps": env.step_count}

@app.get("/baseline")
def baseline():
    return {
        "expected_score": 0.95,
        "strategy": "Always keep servers = traffic / 10"
    }