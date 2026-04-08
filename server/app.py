import sys
import os
# Add root to path so we can find environment.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
import uvicorn
from environment import CloudEnv

app = FastAPI()
env = CloudEnv()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Cloud Optimizer API is running"}

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
        task_id = data.get("task_id") or data.get("id") or "easy"
        return env.reset(str(task_id))
    except:
        return env.reset("easy")

@app.post("/step")
async def step_endpoint(request: Request):
    try:
        body = await request.json()
        action = int(body.get("action", 1))
    except:
        action = 1
    return env.step(action)

@app.get("/grader")
@app.post("/grader")
async def grader_endpoint(request: Request):
    avg_score = round(env.total_reward / env.step_count, 4) if env.step_count > 0 else 0.0
    return {"score": avg_score, "tasks": {t: {"score": avg_score} for t in env.available_tasks}}

def main():
    # Runs the production server
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
