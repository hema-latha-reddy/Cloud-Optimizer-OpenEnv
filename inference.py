import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- THE IMPORT FIX ---
try:
    from server.app import env as cloud_env
except ImportError:
    from app import env as cloud_env

# --- CONFIGURATION (Checklist Compliant) ---
# This checks both naming conventions to ensure your secret is found
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_Token")

# Defaults are set ONLY for Base URL and Model Name
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "cloud-optimizer-cracking"
MAX_STEPS = 10
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
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 1

# --- MAIN LOOP ---
async def main() -> None:
    # Initializing OpenAI client with checklist-compliant variables
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Check if token exists
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is missing. Ensure it is set in Secrets.")

        obs_container = cloud_env.reset(TASK_NAME)
        # Handle different return formats from reset
        obs = obs_container.get("observation", obs_container) if isinstance(obs_container, dict) else obs_container
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
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
        print(f"[DEBUG] Execution Error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
