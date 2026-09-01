"""
schemas.py

Pydantic request/response models for the CargoAI ML service.
Kept separate from app.py so cargoai-backend's mlClient.js has a single,
stable, documented contract to serialize against (and so /docs renders
clean schemas via FastAPI's automatic OpenAPI generation).
"""

from typing import Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    route: str = Field(..., examples=["Singapore-Rotterdam"])
    cargo_type: str = Field(..., examples=["container"])
    horizon_days: int = Field(30, ge=1, le=180)


class ForecastPoint(BaseModel):
    date: str
    predicted_rate_usd_per_ton: float


class ForecastResponse(BaseModel):
    route: str
    cargo_type: str
    model: str
    predictions: list[ForecastPoint]


class FeasibilityRequest(BaseModel):
    vessel_type: str = Field(..., examples=["panamax"])
    loa_m: float = Field(..., gt=0, description="Length overall, meters")
    beam_m: float = Field(..., gt=0, description="Vessel beam, meters")
    draft_m: float = Field(..., gt=0, description="Vessel draft, meters")
    cargo_weight_tons: float = Field(..., gt=0)
    port: str = Field(..., examples=["Rotterdam"])
    port_max_loa_m: float = Field(..., gt=0)
    port_max_beam_m: float = Field(..., gt=0)
    port_max_draft_m: float = Field(..., gt=0)
    congestion_index: Optional[float] = Field(
        None, ge=0, le=1, description="0 (clear) to 1 (severely congested); omit to use a port default"
    )


class Margins(BaseModel):
    loa: float
    beam: float
    draft: float


class FeasibilityResponse(BaseModel):
    feasible: bool
    risk_level: str
    risk_probability: Optional[dict[str, float]] = None
    model: str
    margins_m: Margins
