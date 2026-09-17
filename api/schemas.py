"""
Pydantic Schemas for CEdge-XAI FastAPI REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    ambient_temp: float = Field(..., description="Ambient temperature (°C)")
    ambient_humidity: float = Field(..., description="Ambient relative humidity (%)")
    fan_speed: float = Field(..., description="Cooling fan speed (RPM)")
    lubrication_flow: float = Field(..., description="Lubricant flow (L/min)")
    motor_temp: float = Field(..., description="Main motor temperature (°C)")
    spindle_speed: float = Field(..., description="Machining spindle rotational speed (RPM)")
    vibration: float = Field(..., description="Vibration root mean square (mm/s)")
    acoustic_emission: float = Field(..., description="Ultrasonic acoustic emission level (dB)")
    power_draw: float = Field(..., description="Total active electrical power (kW)")
    tool_wear: float = Field(..., description="Normalized tool wear index")
    coolant_pressure: Optional[float] = Field(None, description="Coolant system pressure (bar)")
    bearing_temp: Optional[float] = Field(None, description="Bearing assembly temperature (°C)")
    load_percentage: Optional[float] = Field(
        None, description="Normalized mechanical load percentage (%)"
    )


class DetectRequest(BaseModel):
    timestamp: Optional[str] = Field(None, description="ISO timestamp of telemetry")
    window: Optional[List[List[float]]] = Field(
        None,
        description="60 timesteps x sensor channels telemetry array (scaled or raw)",
    )
    current_reading: Optional[SensorReading] = Field(
        None,
        description="Single snapshot telemetry (if full window not supplied)",
    )


class DetectResponse(BaseModel):
    timestamp: str
    anomaly: bool
    anomaly_score: float
    threshold: float
    confidence: float
    inference_time_ms: float


class ExplainRequest(BaseModel):
    window: List[List[float]] = Field(..., description="60 timesteps x sensor channels window")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    method: str = Field("clamping", description="Counterfactual method: 'clamping' or 'gradient'")
    top_k: int = Field(3, description="Number of top contributing sensors to explain")


class SensorContribution(BaseModel):
    sensor: str
    contribution_score: float
    current_value: float
    counterfactual_value: float
    delta: float


class ExplainResponse(BaseModel):
    timestamp: str
    anomaly: bool
    anomaly_score: float
    threshold: float
    root_cause: str
    explanation: str
    recommended_action: str
    causal_path: List[str]
    top_contributors: List[SensorContribution]
    counterfactual_valid: bool
    iterations: int
    computation_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    num_sensors: int
    sensor_names: List[str]
    window_size: int
    threshold: float
    onnx_deployed: bool
    int8_quantized: bool
    version: str
