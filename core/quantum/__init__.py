"""
量子决策系统模块初始化文件

该模块实现了基于量子计算原理的决策系统，包括：
- 3量子比特决策电路
- 量子态叠加与纠缠机制
- 概率分布生成与优化决策
- SIMD指令加速的矩阵运算

主要组件：
- QuantumDecider: 量子决策器主类
- DecisionCircuit: 量子决策电路实现
- QuantumOptimizer: 量子优化算法
"""

from .quantum_decider import QuantumDecider
from .decision_circuit import DecisionCircuit

__version__ = "1.0.0"
__all__ = [
    "QuantumDecider",
    "DecisionCircuit"
]