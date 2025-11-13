"""
符号逻辑推理引擎
Symbolic Logic Reasoning Engine

这是一个完整的符号逻辑推理系统，支持经典逻辑推理、模糊逻辑推理、
不确定性推理和高级推理机制。系统集成了LLM作为符号引擎后端。

主要组件:
- SymbolicReasoner: 符号推理引擎主类
- LogicParser: 逻辑表达式解析器
- KnowledgeBase: 知识库管理
- RuleBase: 规则库管理
- InferenceEngine: 推理引擎核心

Author: AI Assistant
Date: 2025-11-13
"""

from .symbolic_reasoner import SymbolicReasoner, ReasoningMode
from .logic_parser import LogicParser, LogicType
from .knowledge_base import KnowledgeBase, KnowledgeType, CertaintyLevel
from .rule_base import RuleBase, RuleType, RuleStatus
from .inference_engine import InferenceEngine, InferenceType

__all__ = [
    'SymbolicReasoner',
    'ReasoningMode',
    'LogicParser',
    'LogicType',
    'KnowledgeBase',
    'KnowledgeType',
    'CertaintyLevel',
    'RuleBase',
    'RuleType',
    'RuleStatus',
    'InferenceEngine',
    'InferenceType'
]

__version__ = '1.0.0'
__author__ = 'AI Assistant'