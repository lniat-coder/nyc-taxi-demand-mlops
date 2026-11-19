"""The API contract. Invalid input is rejected with a 422 before any model code runs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

Count = Field(ge=0, le=10_000, description="pickups in that past hour")


class PredictRequest(BaseModel):
    # extra="forbid": a renamed upstream field is a 422, not a silently missing feature
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "zone_id": 237, "hour_start": "2020-02-14T18:00:00",
        "lag_1": 310, "lag_2": 295, "lag_3": 260, "lag_24": 280, "lag_168": 301, "roll_mean_24": 190.5}})

    zone_id: int = Field(ge=1, le=263, description="TLC pickup zone (PULocationID)")
    hour_start: datetime = Field(description="start of the hour to forecast, NYC local time")
    lag_1: float = Count
    lag_2: float = Count
    lag_3: float = Count
    lag_24: float = Count
    lag_168: float = Count
    roll_mean_24: float = Field(ge=0, le=10_000, description="mean pickups of the 24 previous hours")


class PredictResponse(BaseModel):
    request_id: str
    zone_id: int
    hour_start: datetime
    predicted_pickups: float
    model_version: str
    model_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    train_window: list[str]
    valid_wape: float
    n_zones: int
