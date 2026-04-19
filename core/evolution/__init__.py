"""
Evolution system core module
Contains cross-task performance testing and parallel evaluation mechanisms
"""

from .fitness_evaluator import FitnessEvaluator
from .evolution_visualizer import EvolutionVisualizer
from .checkpoint_manager import CheckpointManager
from .genetic_engine import GeneticEngine
from .population_manager import PopulationManager
from .nsga_ii import NSGA2Selector
from .lifelong_learning import LifelongLearningSystem
from .task_evaluators.coinrun_evaluator import CoinRunEvaluator
from .task_evaluators.trading_evaluator import TradingEvaluator
from .task_evaluators.real_world_evaluator import StackingEvaluator

# Backward compatibility aliases
RealWorldEvaluator = StackingEvaluator

__all__ = [
    'FitnessEvaluator',
    'EvolutionVisualizer',
    'CheckpointManager',
    'GeneticEngine',
    'PopulationManager',
    'NSGA2Selector',
    'LifelongLearningSystem',
    'CoinRunEvaluator', 
    'TradingEvaluator',
    'StackingEvaluator',
    'RealWorldEvaluator'
]

__version__ = '1.0.0'
__author__ = '适应度评估系统开发团队'