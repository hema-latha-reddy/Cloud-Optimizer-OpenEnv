import asyncio
import os
import textwrap
from typing import List, Optional

# --- SAFETY IMPORT WRAPPER ---
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[DEBUG] Missing libraries. Ensure python-dotenv is in requirements.txt")
    class OpenAI: 
        def __init__(self, **kwargs): pass
    def load_dotenv(): pass

# --- THE IMPORT FIX ---
try:
    from server.app import env as cloud_env
except ImportError:
    try:
        from app import env as cloud_env
    except ImportError:
        cloud_env = None

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_Token")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "cloud-optimizer-cracking"
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Cloud Controller. Goal: Keep latency BELOW 150ms.
    RULES:
    - If Latency > 150: Use Action 2 (Scale Up).
    - If Latency < 50: Use Action 0 (Scale Down).
    - Else: Use Action 1 (Maintain).
    Respond with ONLY the digit: 0, 1, or 2.
    """
).strip()

# --- STRUCTURED LOGGING ---
def log_start(task: str, env: str, model: str) -> None:
    # CRITICAL: This is the exact string the scraper looks for 
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
    # Explicit task closure for the scraper
    print(f"[END] task={task}", flush=True)

# --- LLM INTERACTION ---
def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float) -> int:
    user_prompt = f"Step: {step}\nObservation: {obs}\nLast Reward: {last_reward}\nAction (0/1/2):"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=5,
        )
        text = (completion.choices[0].message.content or "").strip()
        for char in text:
            if char in ["0", "1", "2"]:
                return int(char)
        return 1 
    except Exception as exc:
        return 1

# --- INDIVIDUAL TASK RUNNER ---
async def run_single_task(client: OpenAI, task_id: str, max_steps: int) -> None:
    rewards, steps_taken, success, score = [], 0, False, 0.0
    
    # 1. Start Log (Scraper Trigger)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not cloud_env:
            raise ImportError("Environment not found.")
            
        obs_container = cloud_env.reset(task_id)
        obs = obs_container.get("observation", obs_container) if isinstance(obs_container, dict) else obs_container
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            action = get_model_action(client, step, obs, last_reward)
            result = cloud_env.step(action)
            
            obs = result["observation"]
            reward = result["reward"]
            done = result.get("done", False)
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=str(action), reward=reward, done=done, error=None)
            if done: break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
    finally:
        # 2. End Log (Scraper Trigger)
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

# --- MAIN LOOP (3-Task Discovery Fix) ---
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Define tasks with their specific step limits from YAML 
    all_tasks = [
        {"id": "easy", "steps": 10},
        {"id": "medium", "steps": 15},
        {"id": "hard", "steps": 20}
    ]

    for task_info in all_tasks:
        await run_single_task(client, task_info["id"], task_info["steps"])

if __name__ == "__main__":
    asyncio.run(main())
