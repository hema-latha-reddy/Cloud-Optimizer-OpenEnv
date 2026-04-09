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
import sys
import os

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
    """

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the cloud_server environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.env_id = "cloud-optimizer-cracking"
        self.available_tasks = ["easy", "medium", "hard"]
        self.reset()

    def reset(self, task_id: str = "easy") -> CloudServerObservation:
        """
        Reset the environment.

        Args:
            task_id: The ID of the task to initialize (e.g., "easy", "medium", "hard")

        Returns:
            CloudServerObservation with initialized state
        """
        # 1. Properly handle the task_id passed by the validator/server
        clean_id = str(task_id).replace("task-", "").replace("task_", "")
        self.task_id = clean_id if clean_id in self.available_tasks else "easy"
        
        self.servers = 2
        self._state.step_count = 0 # Ensure internal state reset
        self.total_reward = 0.0
        
        # 2. Logic based on the task difficulty (The Grader logic)
        if self.task_id == "easy":
            self.traffic = 100
        elif self.task_id == "medium":
            # Medium uses range to test adaptability
            self.traffic = random.randint(300, 500)
        else:
            # Hard uses high stress load
            self.traffic = 1000
        
        # 3. Calculate latency (using your logic)
        latency = int((self.traffic / self.servers) * 10)

        # 4. Return the Observation with ALL required fields
        return CloudServerObservation(
            traffic=int(self.traffic),
            servers=int(self.servers),
            latency=int(latency),
            message=f"Environment reset for task: {self.task_id}"
        )

    def step(self, action: CloudServerAction) -> CloudServerObservation:
        self._state.step_count += 1
        action_val = action.action

        # 1. Scaling Logic (Existing)
        if action_val == 2: 
            self.servers += 1
        elif action_val == 0 and self.servers > 1: 
            self.servers -= 1
        
        # 2. Traffic Logic (Existing)
        if self.task_id != "easy":
            self.traffic += random.randint(-30, 50)
        if self.traffic < 50: 
            self.traffic = 50
        
        # 3. Calculate Latency
        latency = int((self.traffic / self.servers) * 10)

        # 4. UPDATED REWARD GRADER (Proportional)
        # Perfect range: 100-300
        if 100 <= latency <= 300:
            reward_val = 1.0
        # Acceptable range: 50-100 or 300-500
        elif 50 <= latency <= 500:
            reward_val = 0.5
        # Everything else is a failure
        else:
            reward_val = 0.0
            
        self.total_reward += reward_val
        
        # 5. Create and Return Observation
        obs = CloudServerObservation(
            traffic=int(self.traffic),
            servers=int(self.servers),
            latency=int(latency),
            message=f"Step {self._state.step_count}: Latency is {latency}"
        )

        # Attach reward/done for the UI to see
        obs.reward = float(reward_val)
        obs.done = self._state.step_count >= 8
        
        return obs
    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state