# ============================================================================
# FILE START: grader.py
# ============================================================================
"""
Grader module for evaluating agent performance in the Capital Allocation environment.

This module provides deterministic scoring functions that evaluate:
1. Capital growth (absolute returns)
2. Allocation efficiency (wise deployment of capital)
3. Risk management (avoidance of catastrophic loss)
4. Opportunity cost (not missing high-value trades)

All grading is deterministic and explainable—no randomness.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math


@dataclass
class Trajectory:
    """
    Represents a complete episode trajectory.
    
    Attributes:
        capital_history: List of capital values at each step
        action_history: List of actions taken (which opportunities invested in)
        reward_history: List of rewards received per step
        opportunity_quality: Dict mapping opportunity_id to true quality score
        max_possible_capital: Maximum capital achievable with perfect allocation
        final_capital: Capital at episode end
        num_steps: Total steps in episode
    """
    capital_history: List[float]
    action_history: List[Dict]
    reward_history: List[float]
    opportunity_quality: Dict[int, float]
    max_possible_capital: float
    final_capital: float
    num_steps: int


class CapitalGrowthGrader:
    """
    Grades capital growth: how much wealth was generated.
    
    Logic:
    - Baseline: 0% return = 0.0 score
    - Target: match market growth = 0.5 score
    - Excellent: beat market significantly = 0.8-1.0 score
    - Max achievable: perfect allocation = 1.0 score
    
    This grader rewards absolute growth but is normalized against max possible.
    """
    
    def __init__(self, market_return_rate: float = 0.05):
        """
        Args:
            market_return_rate: Expected return if capital was deployed passively
                                (e.g., 0.05 = 5% typical market return)
        """
        self.market_return_rate = market_return_rate
    
    def grade(self, trajectory: Trajectory) -> Tuple[float, Dict]:
        """
        Grade capital growth relative to market and max possible.
        
        Args:
            trajectory: Episode trajectory with capital history
            
        Returns:
            Tuple of (score: 0.0-1.0, explanation: dict with breakdown)
        """
        initial_capital = trajectory.capital_history[0]
        final_capital = trajectory.final_capital
        max_possible = trajectory.max_possible_capital
        
        # Calculate actual return
        actual_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0
        
        # Calculate market baseline return
        market_capital = initial_capital * (1 + self.market_return_rate)
        market_return = self.market_return_rate
        
        # Calculate max achievable return
        max_return = (max_possible - initial_capital) / initial_capital if initial_capital > 0 else 0
        
        # Scoring logic:
        # - If actual_return <= 0: score = 0.0 (lost money)
        # - If actual_return == market_return: score = 0.5 (matched market)
        # - If actual_return == max_return: score = 1.0 (perfect)
        # - Linear interpolation between these points
        
        if actual_return <= 0:
            score = 0.0
        elif actual_return >= max_return:
            score = 1.0
        elif actual_return >= market_return:
            # Interpolate between 0.5 (market) and 1.0 (max)
            score = 0.5 + 0.5 * (actual_return - market_return) / (max_return - market_return)
        else:
            # Interpolate between 0.0 (breakeven) and 0.5 (market)
            score = 0.5 * (actual_return / market_return)
        
        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        explanation = {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "actual_return": actual_return,
            "market_return": market_return,
            "max_possible_return": max_return,
            "score": score,
        }
        
        return score, explanation


class AllocationEfficiencyGrader:
    """
    Grades how efficiently capital was deployed.
    
    Logic:
    - If capital was never deployed: 0.0 (wasted opportunity)
    - If all capital was deployed in good opportunities: high score
    - If capital was deployed in poor opportunities: penalized
    - Efficient = deploying capital in high-quality, high-returning opportunities
    
    Measured by: weighted return of capital deployed vs. opportunity quality.
    """
    
    def grade(self, trajectory: Trajectory) -> Tuple[float, Dict]:
        """
        Grade allocation efficiency: how well capital was matched to opportunities.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Tuple of (score: 0.0-1.0, explanation: dict)
        """
        if len(trajectory.action_history) == 0:
            return 0.0, {"error": "No actions taken", "score": 0.0}
        
        initial_capital = trajectory.capital_history[0]
        final_capital = trajectory.final_capital
        
        # Calculate total capital deployed across all steps
        total_capital_deployed = 0.0
        weighted_opportunity_quality = 0.0
        num_deployments = 0
        
        for action in trajectory.action_history:
            if action.get("investment"):  # If investment was made
                opportunities_invested = action.get("opportunities", [])
                for opp_id, amount in opportunities_invested:
                    total_capital_deployed += amount
                    quality = trajectory.opportunity_quality.get(opp_id, 0.0)
                    weighted_opportunity_quality += amount * quality
                    num_deployments += 1
        
        # Metric 1: deployment ratio (how much capital was actually used)
        if initial_capital > 0:
            deployment_ratio = total_capital_deployed / initial_capital
        else:
            deployment_ratio = 0.0
        
        # Metric 2: opportunity quality matching
        if total_capital_deployed > 0:
            avg_quality = weighted_opportunity_quality / total_capital_deployed
        else:
            avg_quality = 0.0
        
        # Metric 3: actual returns achieved
        actual_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0
        
        # Scoring logic:
        # - deployment_ratio contributes 40% (encourages capital deployment)
        # - avg_quality contributes 40% (encourages quality allocation)
        # - actual_return contributes 20% (proves efficiency through results)
        
        # Clamp deployment ratio to [0, 1] (100% deployment is optimal)
        deployment_score = min(1.0, deployment_ratio)
        
        # Quality is already in [0, 1]
        quality_score = avg_quality
        
        # Return score: anything positive is good, scale logarithmically
        # Assume typical return is 0-2x (0-100%)
        if actual_return <= 0:
            return_score = 0.0
        else:
            return_score = min(1.0, actual_return / 2.0)  # 100% return = 1.0 score
        
        # Weighted combination
        score = (
            0.40 * deployment_score +
            0.40 * quality_score +
            0.20 * return_score
        )
        
        explanation = {
            "total_capital_deployed": total_capital_deployed,
            "deployment_ratio": deployment_ratio,
            "avg_opportunity_quality": avg_quality,
            "actual_return": actual_return,
            "deployment_score": deployment_score,
            "quality_score": quality_score,
            "return_score": return_score,
            "score": score,
        }
        
        return score, explanation


class RiskManagementGrader:
    """
    Grades risk management: avoiding catastrophic loss.
    
    Logic:
    - No loss = full score (1.0)
    - Small loss (< 10%) = high score (0.8-0.95)
    - Moderate loss (10-30%) = medium score (0.5-0.8)
    - Large loss (> 30%) = low score (< 0.5)
    - Catastrophic loss (> 50%) = near-zero score
    
    Also considers volatility: smoother capital curves score higher.
    """
    
    def grade(self, trajectory: Trajectory) -> Tuple[float, Dict]:
        """
        Grade risk management through loss avoidance and stability.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Tuple of (score: 0.0-1.0, explanation: dict)
        """
        capital_history = trajectory.capital_history
        initial_capital = capital_history[0]
        final_capital = trajectory.final_capital
        
        # Metric 1: avoid losses
        actual_loss = max(0.0, initial_capital - final_capital)
        loss_ratio = actual_loss / initial_capital if initial_capital > 0 else 0.0
        
        # Loss penalty curve: steeper penalty for bigger losses
        if loss_ratio <= 0.0:
            loss_score = 1.0
        elif loss_ratio <= 0.10:
            loss_score = 0.95 - (loss_ratio / 0.10) * 0.15  # 0.95 -> 0.80
        elif loss_ratio <= 0.30:
            loss_score = 0.80 - ((loss_ratio - 0.10) / 0.20) * 0.30  # 0.80 -> 0.50
        else:
            loss_score = max(0.0, 0.50 - (loss_ratio - 0.30) * 2.0)  # 0.50 -> 0.0
        
        # Metric 2: avoid volatility (smooth capital curve)
        volatility_score = self._calculate_volatility_score(capital_history)
        
        # Metric 3: avoid drawdown (max peak-to-trough decline)
        drawdown_score = self._calculate_drawdown_score(capital_history, initial_capital)
        
        # Weighted combination:
        # - loss_score: 50% (most important)
        # - volatility_score: 30% (smooth is safer)
        # - drawdown_score: 20% (temporary dips are acceptable)
        score = (
            0.50 * loss_score +
            0.30 * volatility_score +
            0.20 * drawdown_score
        )
        
        explanation = {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "loss_ratio": loss_ratio,
            "loss_score": loss_score,
            "volatility_score": volatility_score,
            "drawdown_score": drawdown_score,
            "score": score,
        }
        
        return score, explanation
    
    def _calculate_volatility_score(self, capital_history: List[float]) -> float:
        """
        Calculate volatility as standard deviation of returns.
        Lower volatility = higher score.
        """
        if len(capital_history) < 2:
            return 1.0
        
        # Calculate step-wise returns
        returns = []
        for i in range(1, len(capital_history)):
            if capital_history[i - 1] > 0:
                ret = (capital_history[i] - capital_history[i - 1]) / capital_history[i - 1]
                returns.append(ret)
        
        if len(returns) == 0:
            return 1.0
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        # Score: lower volatility is better
        # Assume typical volatility is 5% per step; >20% is bad
        # Linear mapping: 0% std = 1.0, 20% std = 0.0
        volatility_score = max(0.0, 1.0 - (std_dev / 0.20))
        
        return volatility_score
    
    def _calculate_drawdown_score(self, capital_history: List[float], initial_capital: float) -> float:
        """
        Calculate maximum drawdown (peak-to-trough decline).
        Larger drawdowns = lower score.
        """
        if len(capital_history) < 2:
            return 1.0
        
        max_drawdown = 0.0
        peak = capital_history[0]
        
        for capital in capital_history[1:]:
            if capital < peak:
                drawdown = (peak - capital) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            peak = max(peak, capital)
        
        # Score: 0% drawdown = 1.0, 50% drawdown = 0.0, >50% = negative (clamp to 0)
        drawdown_score = max(0.0, 1.0 - (max_drawdown / 0.50))
        
        return drawdown_score


class OpportunityCostGrader:
    """
    Grades opportunity cost: not missing valuable opportunities.
    
    Logic:
    - If agent skipped high-quality opportunities: penalized
    - If agent captured opportunities proportional to their quality: rewarded
    - Measured by: quality-weighted capture rate
    
    This encourages the agent to identify and invest in good deals.
    """
    
    def grade(self, trajectory: Trajectory) -> Tuple[float, Dict]:
        """
        Grade opportunity cost: how many good opportunities were capitalized.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Tuple of (score: 0.0-1.0, explanation: dict)
        """
        if not trajectory.opportunity_quality:
            return 1.0, {"error": "No opportunities in trajectory", "score": 1.0}
        
        # Track which opportunities were invested in
        invested_opportunities = set()
        for action in trajectory.action_history:
            if action.get("investment"):
                for opp_id, amount in action.get("opportunities", []):
                    if amount > 0:
                        invested_opportunities.add(opp_id)
        
        # Calculate weighted opportunity capture
        total_quality = 0.0
        captured_quality = 0.0
        
        for opp_id, quality in trajectory.opportunity_quality.items():
            total_quality += quality
            if opp_id in invested_opportunities:
                captured_quality += quality
        
        if total_quality > 0:
            capture_ratio = captured_quality / total_quality
        else:
            capture_ratio = 1.0  # No opportunities to miss
        
        # Scoring logic:
        # - Capturing all high-quality opportunities: 1.0
        # - Capturing none: 0.0
        # - Linear interpolation between
        
        score = capture_ratio
        
        explanation = {
            "total_opportunities": len(trajectory.opportunity_quality),
            "invested_opportunities": len(invested_opportunities),
            "total_opportunity_quality": total_quality,
            "captured_opportunity_quality": captured_quality,
            "capture_ratio": capture_ratio,
            "score": score,
        }
        
        return score, explanation


class CompositeGrader:
    """
    Combines multiple graders into a single composite score.
    
    This provides a holistic view of agent performance across multiple dimensions.
    """
    
    def __init__(
        self,
        capital_growth_weight: float = 0.35,
        allocation_efficiency_weight: float = 0.25,
        risk_management_weight: float = 0.25,
        opportunity_cost_weight: float = 0.15,
    ):
        """
        Args:
            capital_growth_weight: Weight for capital growth metric
            allocation_efficiency_weight: Weight for allocation efficiency
            risk_management_weight: Weight for risk management
            opportunity_cost_weight: Weight for opportunity cost
        """
        # Validate weights sum to 1.0
        total_weight = (
            capital_growth_weight
            + allocation_efficiency_weight
            + risk_management_weight
            + opportunity_cost_weight
        )
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        self.capital_growth_weight = capital_growth_weight
        self.allocation_efficiency_weight = allocation_efficiency_weight
        self.risk_management_weight = risk_management_weight
        self.opportunity_cost_weight = opportunity_cost_weight
        
        # Initialize sub-graders
        self.capital_grader = CapitalGrowthGrader()
        self.efficiency_grader = AllocationEfficiencyGrader()
        self.risk_grader = RiskManagementGrader()
        self.opportunity_grader = OpportunityCostGrader()
    
    def grade(self, trajectory: Trajectory) -> Tuple[float, Dict]:
        """
        Compute composite score from all graders.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Tuple of (composite_score: 0.0-1.0, detailed_breakdown: dict)
        """
        # Grade each dimension
        capital_score, capital_details = self.capital_grader.grade(trajectory)
        efficiency_score, efficiency_details = self.efficiency_grader.grade(trajectory)
        risk_score, risk_details = self.risk_grader.grade(trajectory)
        opportunity_score, opportunity_details = self.opportunity_grader.grade(trajectory)
        
        # Weighted combination
        composite_score = (
            self.capital_growth_weight * capital_score
            + self.allocation_efficiency_weight * efficiency_score
            + self.risk_management_weight * risk_score
            + self.opportunity_cost_weight * opportunity_score
        )
        
        # Detailed breakdown
        breakdown = {
            "composite_score": composite_score,
            "capital_growth": {
                "score": capital_score,
                "weight": self.capital_growth_weight,
                "details": capital_details,
            },
            "allocation_efficiency": {
                "score": efficiency_score,
                "weight": self.allocation_efficiency_weight,
                "details": efficiency_details,
            },
            "risk_management": {
                "score": risk_score,
                "weight": self.risk_management_weight,
                "details": risk_details,
            },
            "opportunity_cost": {
                "score": opportunity_score,
                "weight": self.opportunity_cost_weight,
                "details": opportunity_details,
            },
        }
        
        return composite_score, breakdown


def create_easy_task_grader() -> CompositeGrader:
    """
    Create a grader optimized for easy difficulty.
    
    Easy tasks prioritize capital growth and risk avoidance.
    """
    return CompositeGrader(
        capital_growth_weight=0.40,
        allocation_efficiency_weight=0.20,
        risk_management_weight=0.30,
        opportunity_cost_weight=0.10,
    )


def create_medium_task_grader() -> CompositeGrader:
    """
    Create a grader optimized for medium difficulty.
    
    Medium tasks balance all four dimensions equally.
    """
    return CompositeGrader(
        capital_growth_weight=0.35,
        allocation_efficiency_weight=0.25,
        risk_management_weight=0.25,
        opportunity_cost_weight=0.15,
    )


def create_hard_task_grader() -> CompositeGrader:
    """
    Create a grader optimized for hard difficulty.
    
    Hard tasks emphasize allocation efficiency and opportunity cost.
    """
    return CompositeGrader(
        capital_growth_weight=0.25,
        allocation_efficiency_weight=0.35,
        risk_management_weight=0.20,
        opportunity_cost_weight=0.20,
    )