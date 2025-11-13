"""
任务评估器模块初始化文件

该模块包含了适应度评估系统中各个任务域的专门评估器：
1. CoinRunEvaluator - 游戏性能评估器
2. TradingEvaluator - 交易性能评估器  
3. RealWorldEvaluator - 现实世界性能评估器

每个评估器都实现了独立的任务域评估，并支持：
- 详细性能指标计算
- 学习曲线分析
- 泛化能力测试
- 并行评估支持
- 数据记录和可视化
"""

from .coinrun_evaluator import CoinRunEvaluator
from .trading_evaluator import TradingEvaluator
from .real_world_evaluator import StackingEvaluator

# 为了保持向后兼容，添加别名
RealWorldEvaluator = StackingEvaluator

__all__ = [
    'CoinRunEvaluator',
    'TradingEvaluator',
    'StackingEvaluator',
    'RealWorldEvaluator'
]

__version__ = '1.0.0'