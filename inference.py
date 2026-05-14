import asyncio
import os
import sys
import re
from typing import List
from openai import OpenAI
import httpx

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from models import CloudServerAction
    from cloud_server.cloud_server_environment import CloudServerEnvironment
    print("SUCCESS: Environment loaded correctly.", file=sys.stderr)
except ImportError as e:
    print(f"CRITICAL: Could not find environment files. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    print("ERROR: No API token found. Please set HF_TOKEN or API_KEY environment variable", file=sys.stderr)
    print("Example: export HF_TOKEN='your_token_here'", file=sys.stderr)
    sys.exit(1)

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("BENCHMARK", "cloud-optimizer-cracking")
MAX_STEPS = 50

# --- HELPER LOGGERS ---
def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: int, reward: float, done: bool):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def extract_action(response_text: str) -> int:
    """Safely extract action from model response"""
    numbers = re.findall(r'\b[012]\b', response_text)
    if numbers:
        return int(numbers[0])
    
    response_lower = response_text.lower()
    if any(word in response_lower for word in ['decrease', 'scale down', 'reduce', '0']):
        return 0
    elif any(word in response_lower for word in ['increase', 'scale up', 'add', '2']):
        return 2
    else:
        return 1

def get_optimal_action(latency: float, servers: int, traffic: int, task: str) -> int:
    """Fallback rule-based action when API fails"""
    if task == "easy":
        if latency > 250:
            return 2
        elif latency < 150:
            return 0
        return 1
    elif task == "medium":
        # Wider thresholds for oscillating traffic
        if latency > 280:
            return 2
        elif latency < 120:
            return 0
        return 1
    else:  # hard
        if latency > 250:
            return 2
        elif latency < 100 and servers > 3:
            return 0
        return 1

async def main():
    token = API_KEY
    
    http_client = httpx.Client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0
    )

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=token,
        http_client=http_client
    )
    
    env = CloudServerEnvironment()
    
    # The 3 tasks required for Round 1
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        rewards = []
        steps_taken = 0
        log_start(task_id)

        try:
            # Reset the environment for the specific task
            observation = env.reset(task_id=task_id)
            print(f"Initial state for {task_id}: traffic={observation.traffic}, servers={observation.servers}, latency={observation.latency:.2f}ms", file=sys.stderr)
            
            for step in range(1, MAX_STEPS + 1):
                steps_taken = step
                
                # Extract current metrics
                latency = getattr(observation, 'latency', 0)
                servers = getattr(observation, 'servers', 1)
                traffic = getattr(observation, 'traffic', 0)
                
                # Create prompt
                prompt = f"""Current Cloud Server Status:
- Incoming Traffic: {traffic} requests/sec
- Active Servers: {servers}
- Current Latency: {latency:.0f}ms
- Target Latency Range: 150-250ms

Available Actions:
0 = DECREASE SERVERS (when latency is too low, below 150ms)
1 = MAINTAIN (when latency is optimal, between 150-250ms)
2 = INCREASE SERVERS (when latency is too high, above 250ms)

Based on the current latency of {latency:.0f}ms, what action should you take?
Respond with ONLY the number (0, 1, or 2):"""
                
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a cloud infrastructure optimizer. Output only 0, 1, or 2 based on latency. No explanations."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=10,
                        temperature=0.0
                    )
                    
                    response = completion.choices[0].message.content or "1"
                    action_val = extract_action(response)
                    
                except Exception as e:
                    sys.stderr.write(f"API error at step {step}: {e}\n")
                    # Use rule-based fallback
                    action_val = get_optimal_action(latency, servers, traffic, task_id)
                
                # Ensure action is valid
                action_val = max(0, min(2, action_val))
                
                # Execute the action
                result = env.step(CloudServerAction(action=action_val))
                
                # Extract results
                reward = float(getattr(result, 'reward', 0.0))
                done = bool(getattr(result, 'done', False))
                
                rewards.append(reward)
                log_step(step, action_val, reward, done)
                
                print(f"Step {step}: action={action_val}, latency={result.latency:.0f}ms, reward={reward:.2f}, servers={result.servers}", file=sys.stderr)
                
                # Update observation
                observation = result
                
                if done:
                    print(f"Task {task_id} completed at step {step}", file=sys.stderr)
                    break
                    
                await asyncio.sleep(0.01)
            
            # Calculate final metrics
            final_score = sum(rewards) / len(rewards) if rewards else 0.0
            is_success = final_score >= 0.5
            
            print(f"\n{'='*50}", file=sys.stderr)
            print(f"Task: {task_id}", file=sys.stderr)
            print(f"Steps taken: {steps_taken}", file=sys.stderr)
            print(f"Average reward: {final_score:.2f}", file=sys.stderr)
            print(f"Success: {is_success}", file=sys.stderr)
            print(f"Final servers: {observation.servers}", file=sys.stderr)
            print(f"Final latency: {observation.latency:.0f}ms", file=sys.stderr)
            print(f"{'='*50}\n", file=sys.stderr)
            
            log_end(is_success, steps_taken, final_score, rewards)
            
        except Exception as e:
            sys.stderr.write(f"Task {task_id} failed: {e}\n")
            import traceback
            traceback.print_exc()
            log_end(False, steps_taken, 0.0, rewards)

if __name__ == "__main__":
    asyncio.run(main())