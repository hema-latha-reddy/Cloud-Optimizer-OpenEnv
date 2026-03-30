from fastapi import FastAPI
from pydantic import BaseModel
import random
import uvicorn  # Moved import to the top

app = FastAPI()

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
            self.traffic = random.randint(200, 500)
        else: # hard
            self.traffic = 1000
        
        return self._get_obs_dict()

    def _get_obs_dict(self):
        # Calculate latency
        latency = int((self.traffic / self.servers) * 10)
        return {
            "traffic": self.traffic,
            "servers": self.servers,
            "latency": latency
        }

env = CloudEnv()

# --- CORE ENDPOINTS ---

@app.get("/")
def home():
    return {
        "status": "openenv_compatible", 
        "env_id": "Cloud-Optimizer-Cracking",
        "team": "Cracking"
    }

@app.post("/reset")
def reset_endpoint(task_id: str = "easy"):
    observation = env.reset(task_id)
    return {"observation": observation, "message": "Environment Reset"}

@app.post("/step")
def step_endpoint(action: int):
    env.step_count += 1
    
    if action == 2: 
        env.servers += 1
    elif action == 0 and env.servers > 1: 
        env.servers -= 1
    
    if env.task_id != "easy":
        env.traffic += random.randint(-50, 70) 
    
    obs = env._get_obs_dict()
    
    reward = 0.0
    if obs["latency"] < 150 and obs["servers"] < 10:
        reward = 1.0
        env.total_reward += 1.0
        
    return {
        "observation": obs,
        "reward": reward,
        "done": False,
        "info": {"step_count": env.step_count}
    }

@app.get("/grader")
def grader():
    if env.step_count == 0: 
        return {"score": 0.0}
    final_score = min(1.0, env.total_reward / env.step_count)
    return {"score": round(final_score, 2), "steps": env.step_count}

# --- THE FIX: ADDING THE MAIN FUNCTION ---
def main():
    """The entry point the validator is looking for."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
