import requests

URL = "https://henareddy-openenv-cracking.hf.space"
tasks = ["easy", "medium", "hard"]

for task in tasks:
    print(f"\n--- Testing Task: {task.upper()} ---")
    
    # 1. CHANGED: Reset now uses requests.post instead of requests.get
    r = requests.post(f"{URL}/reset?task_id={task}")
    print(f"Reset: {r.json()}")
    
    # 2. Take 5 steps (Action 2: Add Server)
    print(f"Running 5 steps to optimize...")
    for i in range(5):
        r = requests.post(f"{URL}/step?action=2")
        data = r.json()
        # Note: If your app.py returns the observation inside a dictionary, 
        # you might need data['observation']['traffic']
        print(f"  Step {i+1} | Traffic: {data.get('traffic', 'N/A')} | Latency: {data.get('latency', 'N/A')}")

    # 3. Check the Grader
    g = requests.get(f"{URL}/grader")
    print(f"Final Score for {task}: {g.json()}")

print("\n--- ALL TESTS COMPLETE ---")
