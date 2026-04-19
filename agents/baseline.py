# ============================================================================
# FILE START: baseline.py
# ============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from env.env import TradingAllocationEnv


class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AllocationAction:
    opportunity_id: int
    capital_amount: float
    confidence: float


class BaselineAgent:

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _score(self, r, risk, conf):
        if risk < 1e-3:
            risk = 1e-3
        return (r * conf) / risk

    def act(self, env: TradingAllocationEnv):
        """
        Convert env state → action vector
        """

        opportunities = env.get_opportunities_as_dict()
        capital = env.capital

        if not opportunities:
            return np.zeros(env.num_opportunities)

        scores = []
        for opp in opportunities:
            s = self._score(
                opp["expected_return"],
                opp["risk"],
                opp["confidence"]
            )
            scores.append(max(0, s))

        scores = np.array(scores)

        if scores.sum() == 0:
            return np.zeros(env.num_opportunities)

        weights = scores / scores.sum()

        # clip max allocation per opportunity
        weights = np.clip(weights, 0, 0.3)

        # normalize again
        if weights.sum() > 1:
            weights = weights / weights.sum()

        return weights.astype(np.float32)

    def run_episode(self, env, max_steps=100):

        env.reset()
        total_reward = 0

        for _ in range(max_steps):

            action = self.act(env)

            _, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

        return env.capital, total_reward


def run_baseline_evaluation(initial_capital=100000, seed=42):

    agent = BaselineAgent(seed=seed)

    for difficulty in TaskDifficulty:
        print("\n==============================")
        print(f"Running: {difficulty.value}")
        print("==============================")

        env = TradingAllocationEnv(
            initial_capital=initial_capital,
            seed=seed
        )

        final_capital, reward = agent.run_episode(env)

        print(f"Final Capital: {final_capital:.2f}")
        print(f"Reward: {reward:.4f}")



# ============================================================================
# FILE END
# ============================================================================