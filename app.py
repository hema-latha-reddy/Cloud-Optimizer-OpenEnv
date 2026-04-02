from fastapi import FastAPI
import random
import uvicorn

app = FastAPI()

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
        latency = int((self.traffic / self.servers) * 10)
        return {
            "traffic": self.traffic,
            "servers": self.servers,
            "latency": latency
        }

    def step(self, action: int):
        self.step_count += 1
        
        if action == 2: 
            self.servers += 1
        elif action == 0 and self.servers > 1: 
            self.servers -= 1
        
        if self.task_id != "easy":
            self.traffic += random.randint(-50, 70) 
        
        obs = self._get_obs_dict()
        
        reward = 0.0
        if obs["latency"] < 150 and obs["servers"] < 10:
            reward = 1.0
            self.total_reward += 1.0
            
        return {
            "observation": obs,
            "reward": reward,
            "done": False,
            "info": {"step_count": self.step_count}
        }

# Create the instance for direct Python imports
env = CloudEnv()

@app.get("/")
def home():
    return {"status": "openenv_compatible", "env_id": "Cloud-Optimizer-Cracking", "team": "Cracking"}

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
    final_score = min(1.0, env.total_reward / env.step_count)
    return {"score": round(final_score, 2), "steps": env.step_count}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
