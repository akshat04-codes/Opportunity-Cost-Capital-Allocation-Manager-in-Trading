"""
Core data models for the Opportunity Cost & Capital Allocation Manager.
"""
from agents.baseline import run_baseline_evaluation
from env.env import TradingAllocationEnv

if __name__ == "__main__":
    run_baseline_evaluation(
        env_class=TradingAllocationEnv,
        initial_capital=100_000,
        seed=42,
    )

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# =========================
# Opportunity
# =========================
class Opportunity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    expected_return: float = Field(..., ge=0.0, le=1.0)
    risk: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    name: str = Field(default="Opportunity")
    min_investment: Optional[float] = Field(None, ge=0.0)
    max_investment: Optional[float] = Field(None, ge=0.0)
    duration: Optional[int] = Field(None, ge=1)

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("risk")
    def validate_risk(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("risk must be between 0 and 1")
        return v

    @field_validator("expected_return")
    def validate_return(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("expected_return must be between 0 and 1")
        return v


# =========================
# Open Position
# =========================
class OpenPosition(BaseModel):
    opportunity_id: str
    capital_invested: float = Field(..., gt=0.0)
    entry_step: int = Field(..., ge=0)
    current_value: Optional[float] = Field(None, gt=0.0)
    realized_return: Optional[float] = None


# =========================
# Portfolio  ✅ FIX ADDED
# =========================
class Portfolio(BaseModel):
    """
    Represents full portfolio state.
    """
    cash: float = Field(..., ge=0.0)
    positions: List[OpenPosition] = Field(default_factory=list)
    total_value: float = Field(..., ge=0.0)


# =========================
# Action
# =========================
class Action(BaseModel):
    allocations: List[float] = Field(..., min_items=0)
    hold_cash: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("allocations")
    def validate_allocations(cls, v):
        for i, a in enumerate(v):
            if not (0.0 <= a <= 1.0):
                raise ValueError(f"allocation[{i}] must be in [0,1]")
        return v

    @field_validator("hold_cash")
    def validate_hold_cash(cls, v, info):
        allocations = info.data.get("allocations", [])
        total = sum(allocations)

        if total > 1.0:
            raise ValueError("allocations exceed 1.0")

        if v is None:
            return 1.0 - total

        if abs((total + v) - 1.0) > 1e-6:
            raise ValueError("allocations + hold_cash must equal 1.0")

        return v


# =========================
# Observation
# =========================
class Observation(BaseModel):
    available_capital: float = Field(..., ge=0.0)
    opportunities: List[Opportunity] = Field(default_factory=list)
    open_positions: List[OpenPosition] = Field(default_factory=list)
    step: Optional[int] = Field(None, ge=0)
    total_portfolio_value: Optional[float] = Field(None, ge=0.0)
    performance_metrics: Optional[dict] = Field(default_factory=dict)


# =========================
# Reward
# =========================
class RewardInfo(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    components: dict = Field(default_factory=dict)
    penalties: dict = Field(default_factory=dict)
    step: Optional[int] = Field(None, ge=0)
    details: Optional[str] = None


# =========================
# Config
# =========================
class EnvironmentConfig(BaseModel):
    initial_capital: float = Field(..., gt=0.0)
    max_steps: int = Field(..., gt=0)
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    opportunity_generation_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: Optional[int] = None


# ============================================================================
# END
# ============================================================================