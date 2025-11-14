"""
认知科学实验模块
==============

该模块包含了各种认知能力测试和评估系统，用于测试智能体的泛化能力、学习能力和适应能力。

主要组件：
- GeneralizationTest: 泛化能力测试主系统
- ModdedMinecraftTest: 模组Minecraft泛化测试
- PyBulletTest: 物理模拟器泛化测试
- RedditDialogueTest: Reddit对话泛化测试
- ZeroShotEvaluator: 零样本学习评估器

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

from .generalization_test import GeneralizationTest, TestType, GeneralizationResult
from .modded_minecraft_test import ModdedMinecraftTest, NewBlockType, NewSkillType, MinecraftBlock, MinecraftSkill, TestResult
from .pybullet_test import PyBulletTest, PhysicsScene, PhysicalProperty, PhysicalObject, PhysicsTestResult
from .reddit_dialogue_test import RedditDialogueTest, QuestionCategory, ResponseQuality, RedditQuestion, RedditResponse, DialogueTestResult
from .zero_shot_evaluator import ZeroShotEvaluator, EvaluationMetric, DomainType, PerformanceMetrics, EvaluationReport

__all__ = [
    'GeneralizationTest',
    'ModdedMinecraftTest', 
    'PyBulletTest',
    'RedditDialogueTest',
    'ZeroShotEvaluator',
    'TestType',
    'GeneralizationResult',
    'NewBlockType',
    'NewSkillType',
    'MinecraftBlock',
    'MinecraftSkill',
    'TestResult',
    'PhysicsScene',
    'PhysicalProperty',
    'PhysicalObject',
    'PhysicsTestResult',
    'QuestionCategory',
    'ResponseQuality',
    'RedditQuestion',
    'RedditResponse',
    'DialogueTestResult',
    'EvaluationMetric',
    'DomainType',
    'PerformanceMetrics',
    'EvaluationReport'
]

__version__ = "1.0.0"
__author__ = "NeuroMinecraftGenesis Team"