# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cloud Server Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CloudServerAction, CloudServerObservation
except ImportError:
    from models import CloudServerAction, CloudServerObservation


class CloudServerEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = CloudServerEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Cloud Server environment ready!"
        >>>
        >>> obs = env.step(CloudServerAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the cloud_server environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.env_id = "cloud-optimizer-cracking"
        self.available_tasks = ["easy", "medium", "hard"]
        self.reset()
    
    def reset(self) -> CloudServerObservation:
        """
        Reset the environment.

        Returns:
            CloudServerObservation with a ready message
        """
        clean_id = str(0).replace("task-", "").replace("task_", "")
        self.task_id = clean_id if clean_id in self.available_tasks else "easy"
        
        self.servers = 2
        self.step_count = 0
        self.total_reward = 0.0
        
        if self.task_id == "easy":
            self.traffic = 100
        elif self.task_id == "medium":
            self.traffic = random.randint(200, 500)
        else:
            self.traffic = 1000
        
        latency = int((self.traffic / self.servers) * 10)

        return CloudServerObservation(
            traffic=str(self.traffic),
            servers=str(self.servers),
            latency=str(latency)
        )

    def step(self, action: CloudServerAction) -> CloudServerObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: CloudServerAction containing the message to echo

        Returns:
            CloudServerObservation with the echoed message and its length
        """
        self._state.step_count += 1

        if action == 2: self.servers += 1
        elif action == 0 and self.servers > 1: self.servers -= 1
        
        if self.task_id != "easy":
            self.traffic += random.randint(-50, 70)
            if self.traffic < 50: self.traffic = 50
        
        obs = self._get_obs_dict()
        reward = 1.0 if obs["latency"] < 200 else 0.0
        self.total_reward += reward
        
        return CloudServerObservation()
        
        return {
            "observation": obs, 
            "reward": round(reward, 2), 
            "done": False, 
            "info": {"step_count": self._state.step_count}
        }

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
