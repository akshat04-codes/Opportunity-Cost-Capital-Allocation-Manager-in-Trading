"""
tasks.py

Defines three trading opportunity allocation tasks with varying difficulty levels:
1. Easy: 2-3 clear opportunities with obvious optimal choice
2. Medium: 4-5 opportunities with mixed risk-return profiles
3. Hard: Dynamic environment with changing opportunities and uncertainty

Each task is a reusable, cleanly structured class following production standards.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class DifficultyLevel(str, Enum):
    """Enumeration of task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Opportunity:
    """
    Represents a single investment opportunity.
    
    Attributes:
        id: Unique identifier for the opportunity
        name: Human-readable name
        min_investment: Minimum capital required (>0)
        expected_return: Expected annualized return (0.0-1.0)
        volatility: Risk measure, std dev of returns (0.0-1.0)
        duration_steps: How many steps the opportunity remains available
        success_probability: Likelihood of achieving expected return (0.0-1.0)
        capital_locked: Whether capital is locked for the duration or can be withdrawn
    """
    id: int
    name: str
    min_investment: float
    expected_return: float
    volatility: float
    duration_steps: int
    success_probability: float = 1.0
    capital_locked: bool = True
    
    def __post_init__(self):
        """Validate opportunity parameters."""
        if self.min_investment <= 0:
            raise ValueError(f"min_investment must be positive, got {self.min_investment}")
        if not (0.0 <= self.expected_return <= 1.0):
            raise ValueError(f"expected_return must be in [0, 1], got {self.expected_return}")
        if not (0.0 <= self.volatility <= 1.0):
            raise ValueError(f"volatility must be in [0, 1], got {self.volatility}")
        if self.duration_steps < 1:
            raise ValueError(f"duration_steps must be >= 1, got {self.duration_steps}")
        if not (0.0 <= self.success_probability <= 1.0):
            raise ValueError(f"success_probability must be in [0, 1], got {self.success_probability}")
    
    def calculate_return(self, capital_invested: float, random_state: np.random.RandomState) -> float:
        """
        Calculate actual return based on expected return, volatility, and success probability.
        
        Args:
            capital_invested: Amount of capital invested in this opportunity
            random_state: NumPy random state for reproducibility
            
        Returns:
            Actual return amount (capital gain/loss)
        """
        if capital_invested <= 0:
            return 0.0
        
        # Check if opportunity succeeds
        if random_state.random() > self.success_probability:
            # Opportunity fails: lose some/all capital
            loss_factor = random_state.uniform(0.2, 1.0)  # Lose 20-100% of invested capital
            return -capital_invested * loss_factor
        
        # Opportunity succeeds: return varies around expected return
        actual_return_rate = random_state.normal(
            loc=self.expected_return,
            scale=self.volatility
        )
        actual_return_rate = np.clip(actual_return_rate, -1.0, 2.0)  # Bound to realistic range
        return capital_invested * actual_return_rate


@dataclass
class TaskConfig:
    """
    Configuration for a task.
    
    Attributes:
        difficulty: Task difficulty level
        initial_capital: Starting capital available
        num_opportunities: Number of opportunities in task
        max_steps: Maximum steps before episode ends
        seed: Random seed for reproducibility
        opportunities: Pre-defined opportunities (if None, generated)
    """
    difficulty: DifficultyLevel
    initial_capital: float
    num_opportunities: int
    max_steps: int
    seed: Optional[int] = None
    opportunities: Optional[List[Opportunity]] = field(default_factory=lambda: None)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if self.num_opportunities < 1:
            raise ValueError(f"num_opportunities must be >= 1, got {self.num_opportunities}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")


class Task(ABC):
    """
    Abstract base class for trading opportunity allocation tasks.
    
    Provides interface for task initialization, opportunity access, and reset.
    Subclasses implement specific difficulty-level tasks.
    """
    
    def __init__(self, config: TaskConfig):
        """
        Initialize task.
        
        Args:
            config: TaskConfig instance with task parameters
        """
        self.config = config
        self.random_state = np.random.RandomState(config.seed)
        self._opportunities: List[Opportunity] = []
        self._setup_opportunities()
    
    @abstractmethod
    def _setup_opportunities(self) -> None:
        """Set up opportunities for this task. Implemented by subclasses."""
        pass
    
    def get_opportunities(self) -> List[Opportunity]:
        """Return list of available opportunities."""
        return self._opportunities.copy()
    
    def reset(self) -> None:
        """Reset task to initial state."""
        self.random_state = np.random.RandomState(self.config.seed)
        self._setup_opportunities()
    
    def get_opportunity_by_id(self, opp_id: int) -> Optional[Opportunity]:
        """Get opportunity by ID."""
        for opp in self._opportunities:
            if opp.id == opp_id:
                return opp
        return None


class EasyTask(Task):
    """
    Easy difficulty task with 2–3 opportunities and one clearly optimal choice.
    
    Scenario:
    - High-return, low-volatility safe investment
    - Lower-return, moderate-volatility risky investment
    - Optional: Medium option for hedge
    
    Expected behavior: Agent should heavily allocate to the safe investment.
    """
    
    def _setup_opportunities(self) -> None:
        """Set up 3 opportunities with clear hierarchy."""
        self._opportunities = [
            Opportunity(
                id=0,
                name="Treasury Bond Fund",
                min_investment=1000.0,
                expected_return=0.06,  # 6% annual
                volatility=0.02,  # Very low risk
                duration_steps=self.config.max_steps,
                success_probability=0.99,
                capital_locked=True,
            ),
            Opportunity(
                id=1,
                name="Blue Chip Dividend Stock",
                min_investment=2000.0,
                expected_return=0.10,  # 10% annual
                volatility=0.08,  # Moderate risk
                duration_steps=self.config.max_steps,
                success_probability=0.95,
                capital_locked=False,
            ),
            Opportunity(
                id=2,
                name="Growth Tech Stock",
                min_investment=1500.0,
                expected_return=0.25,  # 25% annual (risky)
                volatility=0.20,  # High volatility
                duration_steps=self.config.max_steps,
                success_probability=0.70,
                capital_locked=True,
            ),
        ]


class MediumTask(Task):
    """
    Medium difficulty task with 4–5 opportunities and mixed risk-return profiles.
    
    Scenario:
    - Mix of bonds, stocks, and emerging market opportunities
    - Varied risk levels and minimum investments
    - Requires thoughtful allocation to balance portfolio
    
    Expected behavior: Agent should diversify across opportunities,
    balancing high-return risky assets with stable investments.
    """
    
    def _setup_opportunities(self) -> None:
        """Set up 5 opportunities with varied profiles."""
        self._opportunities = [
            Opportunity(
                id=0,
                name="Stable Bond Portfolio",
                min_investment=5000.0,
                expected_return=0.04,
                volatility=0.015,
                duration_steps=self.config.max_steps,
                success_probability=0.98,
                capital_locked=True,
            ),
            Opportunity(
                id=1,
                name="Index Fund S&P 500",
                min_investment=2500.0,
                expected_return=0.10,
                volatility=0.12,
                duration_steps=self.config.max_steps,
                success_probability=0.92,
                capital_locked=False,
            ),
            Opportunity(
                id=2,
                name="Emerging Markets Fund",
                min_investment=3000.0,
                expected_return=0.15,
                volatility=0.18,
                duration_steps=self.config.max_steps - 2,  # Shorter duration
                success_probability=0.80,
                capital_locked=True,
            ),
            Opportunity(
                id=3,
                name="Real Estate Investment Trust",
                min_investment=4000.0,
                expected_return=0.08,
                volatility=0.10,
                duration_steps=self.config.max_steps,
                success_probability=0.88,
                capital_locked=True,
            ),
            Opportunity(
                id=4,
                name="Venture Capital Fund",
                min_investment=10000.0,
                expected_return=0.30,
                volatility=0.35,
                duration_steps=self.config.max_steps,
                success_probability=0.50,
                capital_locked=True,
            ),
        ]


class HardTask(Task):
    """
    Hard difficulty task with dynamic environment.
    
    Scenario:
    - Opportunities change each step (new opportunities emerge, old ones expire)
    - High uncertainty in returns and success probabilities
    - Capital-constrained scenario requiring strategic decisions
    - Agent must adapt to changing landscape and manage opportunity costs
    
    Expected behavior: Agent must continuously reassess opportunities,
    rebalance portfolio, and make decisions under uncertainty about future opportunities.
    """
    
    def _setup_opportunities(self) -> None:
        """Set up initial opportunities pool for dynamic environment."""
        # Create a larger pool of opportunities that will be dynamically revealed
        self._initial_opportunity_pool = [
            # Stable but mature
            Opportunity(id=0, name="Bonds", min_investment=2000, expected_return=0.03, 
                       volatility=0.01, duration_steps=self.config.max_steps, success_probability=0.99),
            # High return but risky
            Opportunity(id=1, name="Crypto", min_investment=5000, expected_return=0.50, 
                       volatility=0.40, duration_steps=self.config.max_steps, success_probability=0.40),
            # Moderate
            Opportunity(id=2, name="Dividend Stocks", min_investment=3000, expected_return=0.10, 
                       volatility=0.12, duration_steps=self.config.max_steps, success_probability=0.85),
            # Emerging
            Opportunity(id=3, name="Startup Equity", min_investment=8000, expected_return=0.60, 
                       volatility=0.50, duration_steps=5, success_probability=0.30),
            # Hedge
            Opportunity(id=4, name="Gold ETF", min_investment=1500, expected_return=0.04, 
                       volatility=0.08, duration_steps=self.config.max_steps, success_probability=0.90),
            # High opportunity cost risk
            Opportunity(id=5, name="Limited IPO Slot", min_investment=15000, expected_return=0.80, 
                       volatility=0.25, duration_steps=2, success_probability=0.75),
            # Late-stage opportunity
            Opportunity(id=6, name="Real Assets", min_investment=10000, expected_return=0.12, 
                       volatility=0.15, duration_steps=self.config.max_steps, success_probability=0.80),
            # Volatile short-term
            Opportunity(id=7, name="Options Trading", min_investment=4000, expected_return=0.35, 
                       volatility=0.45, duration_steps=3, success_probability=0.45),
        ]
        
        # Start with a subset
        self._opportunities = self._initial_opportunity_pool[:4].copy()
    
    def get_opportunities_for_step(self, step: int) -> List[Opportunity]:
        """
        Return opportunities available at a specific step.
        
        In hard mode, opportunities change dynamically:
        - Some expire as duration_steps passes
        - New opportunities become available
        
        Args:
            step: Current environment step
            
        Returns:
            List of available opportunities at this step
        """
        available = []
        
        for opp in self._initial_opportunity_pool:
            # Check if opportunity has appeared yet (staggered introduction)
            appears_at_step = max(0, (opp.id - 4) * 2) if opp.id >= 4 else 0
            
            # Check if opportunity has expired
            if step >= appears_at_step and step < appears_at_step + opp.duration_steps:
                available.append(opp)
        
        return available


class TaskFactory:
    """
    Factory for creating task instances.
    
    Provides convenient methods to instantiate tasks at each difficulty level.
    """
    
    @staticmethod
    def create_easy_task(
        initial_capital: float = 50000.0,
        max_steps: int = 10,
        seed: Optional[int] = None,
    ) -> EasyTask:
        """Create an easy task."""
        config = TaskConfig(
            difficulty=DifficultyLevel.EASY,
            initial_capital=initial_capital,
            num_opportunities=3,
            max_steps=max_steps,
            seed=seed,
        )
        return EasyTask(config)
    
    @staticmethod
    def create_medium_task(
        initial_capital: float = 100000.0,
        max_steps: int = 20,
        seed: Optional[int] = None,
    ) -> MediumTask:
        """Create a medium task."""
        config = TaskConfig(
            difficulty=DifficultyLevel.MEDIUM,
            initial_capital=initial_capital,
            num_opportunities=5,
            max_steps=max_steps,
            seed=seed,
        )
        return MediumTask(config)
    
    @staticmethod
    def create_hard_task(
        initial_capital: float = 150000.0,
        max_steps: int = 20,
        seed: Optional[int] = None,
    ) -> HardTask:
        """Create a hard task."""
        config = TaskConfig(
            difficulty=DifficultyLevel.HARD,
            initial_capital=initial_capital,
            num_opportunities=8,
            max_steps=max_steps,
            seed=seed,
        )
        return HardTask(config)


# Utility functions for task analysis

def get_task_difficulty_info(task: Task) -> Dict:
    """
    Return metadata about a task's difficulty.
    
    Args:
        task: Task instance
        
    Returns:
        Dictionary with difficulty metadata
    """
    opps = task.get_opportunities()
    return {
        "difficulty": task.config.difficulty.value,
        "num_opportunities": len(opps),
        "initial_capital": task.config.initial_capital,
        "max_steps": task.config.max_steps,
        "avg_expected_return": np.mean([o.expected_return for o in opps]),
        "avg_volatility": np.mean([o.volatility for o in opps]),
        "min_total_investment": sum(o.min_investment for o in opps),
    }


def get_task_opportunity_summary(task: Task) -> str:
    """
    Return human-readable summary of opportunities.
    
    Args:
        task: Task instance
        
    Returns:
        Formatted string summary
    """
    opps = task.get_opportunities()
    lines = [f"\n{'='*60}"]
    lines.append(f"Task: {task.config.difficulty.value.upper()}")
    lines.append(f"Initial Capital: ${task.config.initial_capital:,.2f}")
    lines.append(f"Max Steps: {task.config.max_steps}")
    lines.append(f"{'='*60}")
    
    for opp in opps:
        lines.append(f"\n[{opp.id}] {opp.name}")
        lines.append(f"    Min Investment: ${opp.min_investment:,.2f}")
        lines.append(f"    Expected Return: {opp.expected_return*100:.1f}%")
        lines.append(f"    Volatility: {opp.volatility*100:.1f}%")
        lines.append(f"    Success Prob: {opp.success_probability*100:.0f}%")
        lines.append(f"    Duration: {opp.duration_steps} steps")
    
    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)