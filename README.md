

**Project Name:** Cloud-Optimizer-OpenEnv
**Team Name:** Cracking


### Project Description
Cloud-Optimizer-OpenEnv is a Gymnasium-style Reinforcement Learning (RL) environment designed to simulate real-world cloud infrastructure scaling. 

**Core Functionality:**
* **Dynamic Load Balancing:** Agents are tasked with managing a variable server pool to maintain optimal latency (< 150ms) while facing stochastic traffic spikes.
* **Three Complexity Tiers:**
* ***Easy:** Predictable traffic patterns for agent baseline establishment.
    * **Medium:** Dynamic traffic fluctuations requiring adaptive scaling.
    * **Hard:** High-intensity traffic simulation (1000+ units) to test agent robustness under stress.
* **Standardized API:** Fully compliant with the OpenEnv specification, featuring `/reset` for state initialization, `/step` for action execution, and a dedicated `/grader` endpoint for automated performance evaluation.

---

### Technical Highlights
* **Architecture:** Built using FastAPI to ensure low-latency communication between the environment and the training agent.
* **Deployment:** Containerized via Docker on Hugging Face Spaces for immediate accessibility and reproducibility by the hackathon evaluation team.
* **Evaluation:** The system tracks `score` (efficiency) and `steps` (latency minimization), providing a clear quantitative metric for AI agent optimization.

---

### How to Evaluate
1.  **Reset Environment:** Send a `GET` request to `/reset?task_id=[easy/medium/hard]` to initialize the environment.
2.  **Take Steps:** Use `POST` requests to `/step?action=[0/1/2]` to manipulate the server count.
3.  **Check Performance:** Monitor the returned `observation` (traffic, servers, latency) and query the `/grader` endpoint to retrieve the current agent performance score.

---

