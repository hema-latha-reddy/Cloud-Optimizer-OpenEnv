# cloud_server/cloud_server_environment.py

import random
import math
from typing import Optional, Dict, Any
from models import CloudServerObservation, CloudServerAction

class State:
    """State class for OpenEnv compatibility"""
    def __init__(self, env):
        self.step_count = env.step_count
        self.servers = env.servers
        self.traffic = env.traffic
        self.latency = env.latency
        self.reward = env.reward
        self.done = env.done
        self.message = env.message
        self.task = env.current_task
        self.episode_id = str(getattr(env, 'episode_id', '0'))
        self.current_step = env.step_count
        self.total_reward = env.reward
    
    def model_dump(self) -> Dict[str, Any]:
        """Required by OpenEnv - returns state as dictionary"""
        return {
            "step_count": self.step_count,
            "servers": self.servers,
            "traffic": self.traffic,
            "latency": self.latency,
            "reward": self.reward,
            "done": self.done,
            "message": self.message,
            "task": self.task,
            "episode_id": self.episode_id,
            "current_step": self.current_step,
            "total_reward": self.total_reward
        }
    
    def dict(self) -> Dict[str, Any]:
        """Alternative method for OpenEnv"""
        return self.model_dump()

class CloudServerEnvironment:
    """
    Cloud Server Environment for optimizing server scaling based on traffic.
    """
    
    def __init__(self, max_servers: int = 10, min_servers: int = 1):
        self.max_servers = max_servers
        self.min_servers = min_servers
        
        # Environment state
        self.episode_id = "0"
        self.step_count = 0
        self.servers = 5  # CHANGED: Start with 5 servers instead of 1
        self.traffic = 0
        self.latency = 0.0
        self.reward = 0.0
        self.done = False
        self.message = ""
        self.current_task = "easy"
        self.max_steps = 30
        self._state = None
        
        # Task configurations
        self.tasks = {
            "easy": {
                "traffic_range": (100, 200),
                "max_steps": 30
            },
            "medium": {
                "traffic_range": (120, 280),
                "max_steps": 40
            },
            "hard": {
                "traffic_range": (100, 500),
                "max_steps": 50
            }
        }
    
    @property
    def state(self):
        """Return state object (REQUIRED by OpenEnv)"""
        self._state = State(self)
        return self._state
    
    # Async methods required by OpenEnv
    async def reset_async(self, task_id: str = "easy", seed: Optional[int] = None):
        """Async reset method"""
        return self.reset(task_id, seed)
    
    async def step_async(self, action: CloudServerAction):
        """Async step method"""
        return self.step(action)
    
    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> CloudServerObservation:
        """Reset the environment"""
        if seed is not None:
            random.seed(seed)
        
        if task_id not in self.tasks:
            task_id = "easy"
        
        # Increment episode_id as string
        current_id = int(self.episode_id) if self.episode_id.isdigit() else 0
        self.episode_id = str(current_id + 1)
        
        self.current_task = task_id
        self.max_steps = self.tasks[task_id]["max_steps"]
        self.step_count = 0
        self.servers = 5  # CHANGED: Start with 5 servers
        self.done = False
        self.message = ""
        
        # Generate initial traffic
        min_t, max_t = self.tasks[task_id]["traffic_range"]
        self.traffic = random.randint(min_t, max_t)
        
        # Calculate initial latency
        self._calculate_latency()
        
        # Calculate initial reward
        self._calculate_reward()
        
        print(f"RESET: task={task_id}, traffic={self.traffic}, servers={self.servers}, latency={self.latency:.0f}ms, reward={self.reward}", flush=True)
        
        return self._get_observation()
    
    def step(self, action: CloudServerAction) -> CloudServerObservation:
        """Execute one step"""
        # Validate action - FIX: Ensure action is 0,1,2
        action_value = action.action
        if action_value not in [0, 1, 2]:
            print(f"WARNING: Invalid action {action_value}, defaulting to 1", flush=True)
            action_value = 1
        
        # Store previous state for comparison
        old_servers = self.servers
        old_latency = self.latency
        
        # Update servers
        if action_value == 0:
            self.servers = max(self.min_servers, self.servers - 1)
        elif action_value == 2:
            self.servers = min(self.max_servers, self.servers + 1)
        
        # Update step counter
        self.step_count += 1
        
        # Update traffic
        self._update_traffic()
        
        # Calculate new latency
        self._calculate_latency()
        
        # Calculate reward based on new latency
        self._calculate_reward()
        
        # Generate message
        self._update_message(action_value, old_servers, old_latency)
        
        # Check if done
        if self.step_count >= self.max_steps:
            self.done = True
            self.message = f"Task {self.current_task} completed! Final reward: {self.reward:.2f}"
        else:
            self.done = False
        
        print(f"STEP {self.step_count}: action={action_value}, servers={self.servers}, traffic={self.traffic}, latency={self.latency:.0f}ms, reward={self.reward:.2f}", flush=True)
        
        return self._get_observation()
    
    def _update_traffic(self):
        """Update traffic based on task"""
        min_t, max_t = self.tasks[self.current_task]["traffic_range"]
        
        if self.current_task == "easy":
            # Stable traffic with small variations
            base = (min_t + max_t) // 2
            variation = random.randint(-15, 15)
            self.traffic = max(min_t, min(max_t, base + variation))
            
        elif self.current_task == "medium":
            # Gentle oscillation
            center = (min_t + max_t) // 2
            amplitude = (max_t - min_t) // 4
            self.traffic = center + amplitude * math.sin(self.step_count * 0.1)
            self.traffic = int(max(min_t, min(max_t, self.traffic)))
            
        elif self.current_task == "hard":
            # Random with spikes
            if random.random() < 0.2:
                self.traffic = random.randint(max_t - 80, max_t)
            else:
                self.traffic = random.randint(min_t, min_t + 100)
            self.traffic = max(min_t, min(max_t, self.traffic))
    
    def _calculate_latency(self):
        """Calculate latency based on traffic and servers"""
        if self.servers == 0:
            self.latency = 500
            return
        
        # Latency formula: (traffic / servers) * 8
        # With 5 servers and 170 traffic: (170/5)*8 = 272ms (reward ~0.7)
        raw_latency = (self.traffic / self.servers) * 8
        
        # Add small random variation
        variation = random.uniform(-5, 5)
        
        # Calculate final latency (capped between 20 and 500)
        self.latency = max(20, min(500, raw_latency + variation))
    
    def _calculate_reward(self):
        """Calculate reward based on latency"""
        latency = self.latency
        
        # Target range: 150-250ms for optimal reward
        if 150 <= latency <= 250:
            self.reward = 1.0
        elif 140 <= latency <= 260:
            self.reward = 0.9
        elif 130 <= latency <= 270:
            self.reward = 0.8
        elif 120 <= latency <= 280:
            self.reward = 0.7
        elif 110 <= latency <= 290:
            self.reward = 0.6
        elif 100 <= latency <= 300:
            self.reward = 0.5
        elif 90 <= latency <= 310:
            self.reward = 0.4
        elif 80 <= latency <= 320:
            self.reward = 0.3
        elif 70 <= latency <= 330:
            self.reward = 0.2
        elif 60 <= latency <= 340:
            self.reward = 0.1
        else:
            self.reward = 0.0
    
    def _update_message(self, action: int, old_servers: int, old_latency: float):
        """Update status message"""
        action_names = {0: "scaled down", 1: "maintained", 2: "scaled up"}
        
        if self.latency < old_latency:
            improvement = "✓ Latency improved"
        elif self.latency > old_latency:
            improvement = "✗ Latency worsened"
        else:
            improvement = "Latency unchanged"
        
        self.message = f"Step {self.step_count}: {action_names[action]}. Traffic: {self.traffic}, Servers: {self.servers}, Latency: {self.latency:.0f}ms, Reward: {self.reward:.2f}. {improvement}"
    
    def _get_observation(self) -> CloudServerObservation:
        """Create observation object"""
        return CloudServerObservation(
            traffic=self.traffic,
            servers=self.servers,
            latency=round(self.latency, 2),
            reward=self.reward,
            done=self.done,
            message=self.message,
            step=self.step_count,
            task_id=self.current_task
        )
    
    def render(self, mode: str = "human"):
        """Render the environment state"""
        if mode == "human":
            print(f"\n{'='*50}")
            print(f"Task: {self.current_task.upper()} | Step: {self.step_count}")
            print(f"Traffic: {self.traffic} req/s")
            print(f"Servers: {self.servers}")
            print(f"Latency: {self.latency:.0f}ms (Target: 150-250ms)")
            print(f"Reward: {self.reward:.2f}")
            print(f"Message: {self.message}")
            print(f"{'='*50}\n")
    
    def close(self):
        """Clean up resources"""
        pass