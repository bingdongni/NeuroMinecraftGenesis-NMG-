# -*- coding: utf-8 -*-
"""
可视化模块
包含六维认知能力监控、3D脑网络可视化和进化树可视化功能
"""

# 原有组件
try:
    from .brain_network_3d import BrainNetwork3D
    from .neuron_renderer import NeuronRenderer
    from .spike_propagation import SpikePropagation
    from .network_data_handler import NetworkDataHandler
except ImportError:
    BrainNetwork3D = None
    NeuronRenderer = None
    SpikePropagation = None
    NetworkDataHandler = None

# 进化树可视化模块
from .evolution_tree import EvolutionTree, Individual, EvolutionNode
from .tree_renderer import TreeRenderer
from .generation_tracker import GenerationTracker, GenerationStats, GenerationRecord
from .fitness_visualizer import FitnessVisualizer, FitnessMetrics, TrendAnalysis
from .diversity_analyzer import DiversityAnalyzer, DiversityMetrics, GeneticDistanceMatrix

__all__ = [
    # 原有组件
    'BrainNetwork3D',
    'NeuronRenderer', 
    'SpikePropagation',
    'NetworkDataHandler',
    
    # 进化树可视化模块
    'EvolutionTree',
    'Individual',
    'EvolutionNode',
    'TreeRenderer',
    'GenerationTracker',
    'GenerationStats',
    'GenerationRecord',
    'FitnessVisualizer',
    'FitnessMetrics',
    'TrendAnalysis',
    'DiversityAnalyzer',
    'DiversityMetrics',
    'GeneticDistanceMatrix'
]