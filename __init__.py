"""NeuroMinecraft Genesis 主模块"""

# 核心系统组件
from .core.evolution import GeneticEngine, NSGA2Selector, PopulationManager

__version__ = "1.0.0"
__all__ = [
    "GeneticEngine", 
    "NSGA2Selector", 
    "PopulationManager"
]
