# models.py
from pydantic import Field, field_validator
from typing import Optional

# Try different imports for OpenEnv compatibility
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback for when openenv is not installed
    from pydantic import BaseModel
    Action = BaseModel
    Observation = BaseModel
    print("Warning: openenv not found, using fallback models", flush=True)


class CloudServerAction(Action):
    """Action for the Cloud Server environment"""
    
    action: int = Field(
        ..., 
        description="Scaling Control: 0=Decrease, 1=Maintain, 2=Increase",
        ge=0, 
        le=2
    )
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v: int) -> int:
        if v not in [0, 1, 2]:
            print(f"WARNING: Invalid action {v} auto-corrected to 1", flush=True)
            return 1
        return v


class CloudServerObservation(Observation):
    """Observation for the Cloud Server environment"""
    
    traffic: int = Field(..., description="Current network traffic")
    servers: int = Field(..., description="Current number of active servers")
    latency: float = Field(..., description="Current system latency in ms")
    reward: float = Field(default=0.0, description="Current step reward")
    done: bool = Field(default=False, description="Whether episode is complete")
    message: str = Field(default="", description="Status message")
    step: int = Field(default=0, description="Current step number")
    task_id: str = Field(default="easy", description="Task difficulty level")