# ============================================================================
# Trading Allocation Environment Module (FINAL CLEAN VERSION)
# ============================================================================

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


# =========================
# Opportunity
# =========================
@dataclass
class Opportunity:
    id: int
    expected_return: float
    risk: float
    confidence: float
    initial_value: float = 1.0


# =========================
# Step Info
# =========================
@dataclass
class StepInfo:
    allocations: Dict[int, float]
    returns: Dict[int, float]
    capital_used: float
    capital_efficiency: float
    risk_concentration: float
    profit: float


# =========================
# ENVIRONMENT
# =========================
class TradingAllocationEnv:

    def __init__(
        self,
        initial_capital: float = 100000.0,
        num_opportunities: int = 5,
        max_steps: int = 252,
        seed: Optional[int] = None,
        risk_penalty_weight: float = 0.3,
        efficiency_penalty_weight: float = 0.1,
    ):

        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if num_opportunities <= 0:
            raise ValueError("num_opportunities must be positive")

        self.initial_capital = initial_capital
        self.num_opportunities = num_opportunities
        self.max_steps = max_steps

        self.risk_penalty_weight = risk_penalty_weight
        self.efficiency_penalty_weight = efficiency_penalty_weight

        self.rng = np.random.RandomState(seed)

        # state
        self.capital = initial_capital
        self.opportunities: List[Opportunity] = []
        self.current_step = 0
        self.cumulative_profit = 0.0
        self.episode_history: List[StepInfo] = []

    # =========================
    # OPPORTUNITY GENERATION
    # =========================
    def generate_opportunities(self) -> List[Opportunity]:

        opportunities = []

        for i in range(self.num_opportunities):

            if i % 3 == 0:
                expected_return = self.rng.uniform(0.05, 0.15)
                risk = self.rng.uniform(0.05, 0.15)
                confidence = self.rng.uniform(0.7, 1.0)

            elif i % 3 == 1:
                expected_return = self.rng.uniform(0.15, 0.35)
                risk = self.rng.uniform(0.20, 0.40)
                confidence = self.rng.uniform(0.5, 0.85)

            else:
                expected_return = self.rng.uniform(0.0, 0.5)
                risk = self.rng.uniform(0.30, 0.60)
                confidence = self.rng.uniform(0.3, 0.7)

            opportunity = Opportunity(
                id=i,
                expected_return=float(np.clip(expected_return, 0, 1)),
                risk=float(np.clip(risk, 0, 1)),
                confidence=float(np.clip(confidence, 0, 1)),
            )

            opportunities.append(opportunity)

        return opportunities

    # =========================
    # RESET
    # =========================
    def reset(self) -> np.ndarray:

        self.capital = self.initial_capital
        self.opportunities = self.generate_opportunities()
        self.current_step = 0
        self.cumulative_profit = 0.0
        self.episode_history = []

        return self.state()

    # =========================
    # STEP
    # =========================
    def step(self, action: np.ndarray):

        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        if len(action) != self.num_opportunities:
            raise ValueError("Invalid action size")

        action = np.clip(action, 0.0, 1.0)

        if action.sum() > 1:
            action = action / action.sum()

        capital_used = self.capital * action.sum()

        allocations = {}
        returns = {}
        profits = {}

        for i, opp in enumerate(self.opportunities):

            allocated = self.capital * action[i]
            allocations[opp.id] = allocated

            if allocated > 0:
                base = opp.expected_return * opp.confidence
                noise = self.rng.normal(0, opp.risk)
                actual_return = base + noise

                profit = allocated * actual_return

                returns[opp.id] = actual_return
                profits[opp.id] = profit
            else:
                returns[opp.id] = 0.0
                profits[opp.id] = 0.0

        total_profit = sum(profits.values())

        self.capital += total_profit
        self.cumulative_profit += total_profit

        capital_efficiency = min(action.sum(), 1.0)

        if capital_used > 0:
            risk_concentration = sum(
                (allocations[opp.id] / capital_used) * opp.risk
                for opp in self.opportunities
                if allocations[opp.id] > 0
            )
        else:
            risk_concentration = 0.0

        reward = self._compute_reward(
            total_profit,
            capital_efficiency,
            risk_concentration
        )

        info = StepInfo(
            allocations=allocations,
            returns=returns,
            capital_used=capital_used,
            capital_efficiency=capital_efficiency,
            risk_concentration=risk_concentration,
            profit=total_profit
        )

        self.episode_history.append(info)
        self.current_step += 1

        done = self.current_step >= self.max_steps

        return self.state(), reward, done, info

    # =========================
    # REWARD
    # =========================
    def _compute_reward(self, profit, efficiency, risk):

        profit_component = profit / self.initial_capital
        efficiency_bonus = efficiency
        risk_penalty = -self.risk_penalty_weight * risk
        idle_penalty = -self.efficiency_penalty_weight * (1 - efficiency)

        return float(profit_component + efficiency_bonus + risk_penalty + idle_penalty)

    # =========================
    # STATE
    # =========================
    def state(self) -> np.ndarray:

        state = []

        state.append(self.capital / self.initial_capital)

        for opp in self.opportunities:
            state.extend([
                opp.expected_return,
                opp.risk,
                opp.confidence
            ])

        state.append(self.current_step / self.max_steps)

        return np.array(state, dtype=np.float32)

    # =========================
    # 🔥 FIX (IMPORTANT)
    # =========================
    def get_opportunities_as_dict(self):
        return [
            {
                "id": opp.id,
                "expected_return": opp.expected_return,
                "risk": opp.risk,
                "confidence": opp.confidence
            }
            for opp in self.opportunities
        ]

    # =========================
    # STATS
    # =========================
    def get_stats(self):

        if not self.episode_history:
            return {}

        profits = [x.profit for x in self.episode_history]

        return {
            "total_profit": sum(profits),
            "steps": len(self.episode_history),
            "final_capital": self.capital,
            "return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100
        }

# ============================================================================
# END
# ============================================================================