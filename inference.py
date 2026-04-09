import asyncio
import os
import sys
import textwrap
from typing import List, Optional
from openai import OpenAI

from models import CloudServerObservation
# 1. Add the root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. Specific Imports based on your file tree
try:
    # 1. Import the Action model from models.py (in your root)
    from models import CloudServerAction
    
    # 2. Import the Environment class from the file INSIDE the server folder
    # Based on your screenshot, the file is server/cloud_server_environment.py
    from cloud_server.cloud_server_environment import CloudServerEnvironment
    
    print("SUCCESS: Environment loaded correctly.", file=sys.stderr)
except ImportError as e:
    print(f"DEBUG: Failed to find in 'server' folder. Trying root 'environment.py'...")
    try:
        # Fallback: Maybe your class is actually in the root environment.py?
        from environment import CloudEnv as CloudServerEnvironment
        from models import CloudServerAction
    except ImportError:
        print(f"CRITICAL: Could not find environment files. Error: {e}")
        sys.exit(1)

# --- CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("BENCHMARK", "cloud-optimizer-cracking")
MAX_STEPS = 8

# --- HELPER LOGGERS (Standardized) ---
def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: int, reward: float, done: bool):
    if reward >= 1:
        reward = 0.99
    if reward <= 0:
        reward = 0.01
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    if score >= 1:
        score = 0.99
    if score <= 0:
        score = 0.01
    good_rewards = []
    for i in rewards:
        if i >= 1:
            i = 0.99
        if i <= 0:
            i = 0.01
        good_rewards.append(i)
    rewards_str = ",".join(f"{r:.2f}" for r in good_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# --- CORE INFERENCE LOGIC ---

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = CloudServerEnvironment()
    
    # The 3 tasks required for Round 1
    # tasks = ["easy", "medium", "hard"]
    tasks = ["easy","medium","hard"]

    for task_id in tasks:
        for _ in range(3):
            history = []
            rewards = []
            steps_taken = 0
            
            log_start(task_id)

            try:
                # Reset the environment for the specific task
                result = env.reset(task_id=task_id)
                observation: CloudServerObservation = result 
                for step in range(1, MAX_STEPS + 1):
                    steps_taken = step
                    
                    # LLM Interaction
                    prompt = f"Current Cloud State: {observation}. You can: 0 (Decrease), 1 (Maintain), 2 (Increase). Return ONLY the number."
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a cloud resource optimizer. Output only the integer 0, 1, or 2."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                    )
                    
                    res_text = (completion.choices[0].message.content or "1").strip()
                    # Get the first available digit
                    action_val = int(next((s for s in res_text if s in "012"), 1))
                    
                    # Execute step using the Typed Action
                    result = env.step(CloudServerAction(action=action_val))
                    
                    # Extract results from the OpenEnv result object
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)
                    obs = result
                    
                    rewards.append(reward)
                    log_step(step, action_val, reward, done)

                    if done:
                        break
                    await asyncio.sleep(0.01)

                # Final metrics
                final_score = sum(rewards) / len(rewards) if rewards else 0.0
                is_success = final_score > 0.1 # Threshold for passing
                log_end(is_success, steps_taken, final_score, rewards)

            except Exception as e:
                # Errors to stderr so they don't break the validator
                sys.stderr.write(f"Task {task_id} failed: {e}\n")
                # Ensure an [END] block is still printed if the validator expects it
                log_end(False, steps_taken, 0.0, rewards)

if __name__ == "__main__":
    asyncio.run(main())