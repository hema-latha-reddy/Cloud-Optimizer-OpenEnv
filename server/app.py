import sys
import os
# Force the path so Python finds the server module regardless of working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
import random
import uvicorn

app = FastAPI()

# --- THE ENVIRONMENT STATE ---
class CloudEnv:
    def __init__(self):
        self.env_id = "cloud-optimizer-cracking"
        # Synchronized IDs to match the YAML exactly
        self.available_tasks = ["easy", "medium", "hard"]
        self.reset("easy")

    def reset(self, task_id: str):
        # Normalize the task_id (strip "task-" if the validator sends it that way)
        clean_id = task_id.replace("task-", "").replace("task_", "")
        self.task_id = clean_id if clean_id in self.available_tasks else "easy"
        
        self.servers = 2
        self.step_count = 0
        self.total_reward = 0.0
        
        if self.task_id == "easy":
            self.traffic = 100
        elif self.task_id == "medium":
            self.traffic = random.randint(200, 500)
        else:
            self.traffic = 1000
        
        return self._get_obs_dict()

    def _get_obs_dict(self):
        # Basic latency formula: (traffic / servers) * 10
        latency = int((self.traffic / self.servers) * 10)
        return {"traffic": self.traffic, "servers": self.servers, "latency": latency}

    def step(self, action: int):
        self.step_count += 1
        # 0: Decrease, 1: Stay, 2: Increase
        if action == 2: 
            self.servers += 1
        elif action == 0 and self.servers > 1: 
            self.servers -= 1
        
        # Traffic fluctuates in harder tasks
        if self.task_id != "easy":
            self.traffic += random.randint(-50, 70)
            if self.traffic < 50: self.traffic = 50
        
        obs = self._get_obs_dict()
        
        # Reward Logic
        if obs["latency"] < 150:
            reward = 1.2 if self.servers <= 5 else 1.0
        elif obs["latency"] > 500:
            reward = -0.5
        else:
            reward = 0.0
            
        self.total_reward += reward
        return {
            "observation": obs, 
            "reward": round(reward, 2), 
            "done": False, 
            "info": {"step_count": self.step_count}
        }

env = CloudEnv()

@app.get("/")
def home():
    return {
        "status": "ok", 
        "env_id": env.env_id,
        "tasks": env.available_tasks 
    }

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
        # Handle both 'task_id' and 'id' keys just in case
        task_id = data.get("task_id") or data.get("id") or "easy"
        
        observation = env.reset(str(task_id))
        return observation
    except Exception as e:
        # If anything goes wrong, return a valid starting observation
        return {"traffic": 100, "servers": 2, "latency": 500}

@app.post("/step")
async def step_endpoint(request: Request):
    try:
        body = await request.json()
        action = int(body.get("action", 1))
    except:
        action = 1
    return env.step(action)

# Grader endpoint for discovery
@app.get("/grader")
@app.post("/grader")
async def grader_endpoint(request: Request):
    avg_score = round(env.total_reward / env.step_count, 4) if env.step_count > 0 else 0.0
    return {
        "score": avg_score,
        "tasks": {t: {"score": avg_score} for t in env.available_tasks}
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
