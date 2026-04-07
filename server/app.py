from fastapi import FastAPI, Request
import random
import uvicorn

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
        # Latency logic: higher traffic/lower servers = higher latency
        latency = int((self.traffic / self.servers) * 10)
        return {
            "traffic": self.traffic,
            "servers": self.servers,
            "latency": latency
        }

    def step(self, action: int):
        self.step_count += 1
        
        # Action Logic: 0=Scale Down, 1=Stay, 2=Scale Up
        if action == 2: 
            self.servers += 1
        elif action == 0 and self.servers > 1: 
            self.servers -= 1
        
        # Simulate dynamic traffic for Medium/Hard tasks
        if self.task_id != "easy":
            self.traffic += random.randint(-50, 70) 
            if self.traffic < 50: self.traffic = 50 
        
        obs = self._get_obs_dict()
        
        # --- REWARD LOGIC ---
        reward = 0.0
        if obs["latency"] < 150:
            reward = 1.0 # Base reward
            if obs["servers"] <= 5:
                reward += 0.2 # Efficiency bonus
        elif obs["latency"] > 500:
            reward = -0.5 # Penalty
            
        self.total_reward += reward
            
        return {
            "observation": obs,
            "reward": round(reward, 2),
            "done": False,
            "info": {"step_count": self.step_count}
        }

# Create the instance for direct Python imports
env = CloudEnv()

# --- CORE ENDPOINTS ---

@app.get("/")
def home():
    return {
        "status": "openenv_compatible", 
        "env_id": "Cloud-Optimizer-Cracking",
        "team": "Cracking"
    }

# NEW: Grader endpoint handles both GET and POST to be safe
@app.get("/grader")
@app.post("/grader")
async def grader_endpoint(request: Request):
    """
    Returns the current evaluation score to the Meta/Scaler validator.
    """
    if env.step_count == 0: 
        return {"score": 0.0, "status": "no_steps_completed"}
    
    # Calculate average performance
    final_score = env.total_reward / env.step_count
    return {
        "score": round(final_score, 4), 
        "steps": env.step_count,
        "status": "active"
    }

@app.post("/reset")
async def reset_endpoint(request: Request):
    # Handle optional task_id from JSON or query params
    data = await request.json() if await request.body() else {}
    task_id = data.get("task_id", "easy")
    observation = env.reset(task_id)
    return {"observation": observation, "message": f"Environment Reset for {task_id}"}

@app.post("/step")
async def step_endpoint(request: Request):
    data = await request.json()
    action = data.get("action", 1)
    return env.step(action)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
