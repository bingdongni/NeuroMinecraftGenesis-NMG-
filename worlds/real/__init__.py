#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理世界策略迁移系统
实现从Minecraft虚拟环境到物理世界的策略迁移机制

主要组件：
- StrategyTransfer: 策略迁移主类
- KnowledgeMapper: 知识映射器  
- TransferEvaluator: 迁移评估器
- AdaptationEngine: 适应引擎
- PerformanceAnalyzer: 性能分析器
"""

import sys
import os

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from strategy_transfer import StrategyTransfer
from knowledge_mapper import KnowledgeMapper
from transfer_evaluator import TransferEvaluator
from adaptation_engine import AdaptationEngine
from performance_analyzer import PerformanceAnalyzer

__version__ = "1.0.0"
__author__ = "Strategy Migration System"
__email__ = "support@strategymigration.com"

# 公开接口
__all__ = [
    'StrategyTransfer',
    'KnowledgeMapper', 
    'TransferEvaluator',
    'AdaptationEngine',
    'PerformanceAnalyzer'
]

# 系统版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

# 系统能力声明
SYSTEM_CAPABILITIES = {
    'strategy_transfer': True,
    'knowledge_mapping': True,
    'transfer_evaluation': True,
    'real_time_adaptation': True,
    'performance_analysis': True,
    'bidirectional_mapping': True,
    'anomaly_detection': True,
    'trend_analysis': True,
    'performance_forecasting': True,
    'optimization_recommendations': True
}

# 默认配置模板
DEFAULT_CONFIG = {
    'strategy_transfer': {
        'similarity_threshold': 0.7,
        'adaptation_rate': 0.1,
        'learning_rate': 0.01
    },
    'knowledge_mapping': {
        'mapping_granularity': 'medium',
        'confidence_weight': 0.8,
        'bidirectional_mapping': True
    },
    'transfer_evaluation': {
        'evaluation_metrics': ['accuracy', 'success_rate', 'execution_time', 'stability'],
        'baseline_comparison': True,
        'statistical_significance': 0.05
    },
    'adaptation_engine': {
        'adaptation_frequency': 1,
        'convergence_criteria': {
            'performance_stability': 0.05,
            'min_adaptations': 5
        }
    },
    'performance_analyzer': {
        'analysis_frequency': 10,
        'window_size': 100,
        'alert_threshold': 0.8
    }
}

def get_version_info():
    """获取版本信息"""
    return VERSION_INFO.copy()

def get_system_capabilities():
    """获取系统能力声明"""
    return SYSTEM_CAPABILITIES.copy()

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()

def create_transfer_system(config=None):
    """
    创建策略迁移系统的便捷函数
    
    Args:
        config: 可选的配置字典
        
    Returns:
        StrategyTransfer: 配置好的迁移系统实例
    """
    return StrategyTransfer(config)

# 导入所需的标准库
from datetime import datetime

if __name__ == "__main__":
    print("物理世界策略迁移系统 v1.0.0")
    print("=" * 40)
    
    try:
        # 测试系统初始化
        transfer_system = StrategyTransfer()
        print("✓ 策略迁移系统初始化成功")
        
        # 测试会话创建
        session_id = transfer_system.start_transfer_session("demo")
        print(f"✓ 测试会话创建成功: {session_id}")
        
        # 获取系统统计
        stats = transfer_system.get_transfer_statistics()
        print(f"✓ 系统统计: {stats}")
        
        print("\n🎉 策略迁移系统运行正常!")
        
    except Exception as e:
        print(f"✗ 系统测试失败: {e}")
        import traceback
        traceback.print_exc()