import requests
import sys

def run_inference(url):
    base_url = url.rstrip('/')
    requests.post(f"{base_url}/reset?task_id=easy")
    for _ in range(5):
        requests.post(f"{base_url}/step?action=2")
    response = requests.get(f"{base_url}/grader")
    print(f"Final Score: {response.json()}")

if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    run_inference(target_url)
