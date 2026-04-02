

**Project Name:** Cloud-Optimizer-OpenEnv
**Team Name:** Cracking


**Cloud-Optimizer-OpenEnv** is a Gymnasium-style Reinforcement Learning (RL) environment designed to simulate real-world cloud infrastructure scaling. It provides a standardized interface for AI agents to learn optimal server management under varying traffic loads.

## 🚀 Project Description
Cloud-Optimizer-OpenEnv provides a dynamic simulation where agents manage a variable server pool to maintain optimal latency (< 150ms) while facing stochastic traffic spikes.

### Core Functionality
* **Dynamic Load Balancing:** Agents must balance server costs against performance (latency).
* **Three Complexity Tiers:**
    * **Easy:** Predictable traffic patterns (100 units).
    * **Medium:** Dynamic fluctuations (200-500 units) requiring adaptive scaling.
    * **Hard:** High-intensity stress testing (1000+ units).
* **Standardized API:** Fully compliant with the OpenEnv specification.

## 📊 Environment Specification

### Observation Space
The agent receives a dictionary containing:
* `traffic`: The current number of incoming requests.
* `servers`: The current number of active server instances.
* `latency`: Calculated as `(traffic / servers) * 10`.

### Action Space
The agent can choose one of three discrete actions:
* **0**: Scale Down (Remove 1 server)
* **1**: Maintain (No change)
* **2**: Scale Up (Add 1 server)

### Reward Function
* **+1.0**: Awarded if `latency < 150ms` and `servers < 10`.
* **0.0**: Awarded if the system is unhealthy (high latency or over-provisioned).

## 🛠️ Technical Highlights
* **Architecture:** Built using **FastAPI** for low-latency communication.
* **Deployment:** Containerized via Docker on **Hugging Face Spaces**.
* **Unified Logic:** We deliberately chose a single-file architecture (`app.py`) instead of a multi-file `models.py` approach. 
    * **Why no `models.py`?** By keeping the `CloudEnv` class and FastAPI routes in one file, we ensure zero-latency imports and prevent "circular dependency" errors during high-frequency agent evaluations. This makes the environment more robust for automated Phase 2 grading.

## 🚦 How to Evaluate

1.  **Reset Environment:** `POST /reset?task_id=easy` 
    *Initializes the state and returns the first observation.*

2.  **Take Steps:** `POST /step?action=2` 
    *Executes an action and returns the new observation, reward, and done status.*

3.  **Check Performance:** `GET /grader` 
    *Retrieves the final quantitative performance score based on reward averages.*

## 💻 Local Setup
```bash
pip install -r requirements.txt
python3 inference.py

