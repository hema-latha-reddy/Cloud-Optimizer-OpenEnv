# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Cloud Server Environment.

The cloud_server environment is a simple test environment that echoes back messages.
"""

from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class CloudServerAction(Action):
    """
    Action for the Cloud Server environment.
    0: Decrease, 1: Maintain, 2: Increase
    """
    action: int = Field(..., description="Action to perform: 0, 1, or 2")


class CloudServerObservation(Observation):
    """
    Observation for the Cloud Server environment.
    Note: We use 'Observation' as the base class to stay compliant with OpenEnv.
    """
    traffic: int = Field(..., description="Current network traffic")
    servers: int = Field(..., description="Current number of active servers")
    latency: int = Field(..., description="Current system latency")
    message: str = Field(..., description="Status message")
    