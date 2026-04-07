from fastapi import FastAPI
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
        # Latency increases as traffic rises and decreases as servers are added
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
            if self.traffic < 50: self.traffic = 50 # Floor for traffic
        
        obs = self._get_obs_dict()
        
        # --- PHASE 3 CUSTOM REWARD LOGIC ---
        reward = 0.0
        
        if obs["latency"] < 150:
            # Base reward for meeting the SLA (Service Level Agreement)
            reward = 1.0
            
            # UNIQUE FEATURE: Efficiency Bonus
            # If the AI keeps latency low with 5 or fewer servers, give extra points
            if obs["servers"] <= 5:
                reward += 0.2 
        
        elif obs["latency"] > 500:
            # UNIQUE FEATURE: Critical Failure Penalty
            # If the system is extremely slow, penalize the agent
            reward = -0.5
            
        self.total_reward += reward
            
        return {
            "observation": obs,
            "reward": round(reward, 2),
            "done": False,
            "info": {"step_count": self.step_count}
        }

# Create the instance for direct Python imports (Phase 2 Requirement)
env = CloudEnv()

# --- CORE ENDPOINTS (Phase 1 Requirement) ---

@app.get("/")
def home():
    return {
        "status": "openenv_compatible", 
        "env_id": "Cloud-Optimizer-Cracking",
        "team": "Cracking",
        "features": ["Efficiency Bonus", "Latency Penalty"]
    }

@app.post("/reset")
def reset_endpoint(task_id: str = "easy"):
    observation = env.reset(task_id)
    return {"observation": observation, "message": "Environment Reset"}

@app.post("/step")
def step_endpoint(action: int):
    return env.step(action)

@app.get("/grader")
def grader():
    if env.step_count == 0: 
        return {"score": 0.0}
    # Calculate average reward per step
    final_score = env.total_reward / env.step_count
    return {"score": round(final_score, 2), "steps": env.step_count}

def main():
    # Standard Hugging Face port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
