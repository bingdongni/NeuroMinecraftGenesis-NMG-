# -*- coding: utf-8 -*-
"""
学习效率评估器
Learning Efficiency Evaluator

该模块负责评估智能体在不同领域的学习效率。
通过多维度的指标分析，量化学习过程中的效率特征，
为跨域学习提供科学的效率评估和改进建议。

主要功能：
- 学习速度评估
- 学习质量分析
- 记忆保持效率
- 学习曲线分析
- 资源利用效率
- 个性化学习优化

作者: AI系统
日期: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import json
import asyncio
import math
from collections import deque, defaultdict

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LearningMetrics:
    """学习指标数据结构"""
    domain: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    learning_speed: float                   # 学习速度
    learning_quality: float                 # 学习质量
    memory_efficiency: float               # 记忆效率
    accuracy_progression: List[float]      # 精度进展
    speed_progression: List[float]         # 速度进展
    consistency_score: float               # 一致性得分
    final_performance: float               # 最终性能
    resource_usage: Dict[str, float]       # 资源使用情况
    learning_curve_slope: float            # 学习曲线斜率
    plateau_threshold: float               # 平台期阈值


@dataclass
class LearningEfficiencyReport:
    """学习效率报告"""
    overall_efficiency: float              # 总体效率
    speed_efficiency: float                # 速度效率
    quality_efficiency: float              # 质量效率
    resource_efficiency: float             # 资源效率
    consistency_rating: float              # 一致性评级
    learning_pattern: str                  # 学习模式
    improvement_suggestions: List[str]     # 改进建议
    optimal_parameters: Dict[str, Any]     # 最优参数
    benchmark_comparison: Dict[str, float] # 基准比较


class LearningEfficiency:
    """
    学习效率评估器
    
    负责评估和监控智能体在不同领域的学习效率，
    通过多维度指标分析和学习模式识别，提供科学的效率评估。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('LearningEfficiency')
        
        # 学习效率参数配置
        self.efficiency_params = {
            'speed_weight': config.get('speed_weight', 0.3),
            'quality_weight': config.get('quality_weight', 0.4),
            'consistency_weight': config.get('consistency_weight', 0.2),
            'resource_weight': config.get('resource_weight', 0.1),
            'plateau_detection_threshold': config.get('plateau_detection_threshold', 0.01),
            'learning_curve_smoothing': config.get('learning_curve_smoothing', 0.1),
            'min_learning_samples': config.get('min_learning_samples', 10),
            'efficiency_benchmark': config.get('efficiency_benchmark', 0.8)
        }
        
        # 学习模式定义
        self.learning_patterns = {
            'rapid_learner': {
                'description': '快速学习者',
                'characteristics': {
                    'initial_speed': 'high',
                    'plateau_speed': 'medium',
                    'consistency': 'medium'
                },
                'efficiency_signature': [0.8, 0.6, 0.4, 0.9, 0.7, 0.8]
            },
            'steady_learner': {
                'description': '稳健学习者',
                'characteristics': {
                    'initial_speed': 'medium',
                    'plateau_speed': 'high',
                    'consistency': 'high'
                },
                'efficiency_signature': [0.6, 0.7, 0.8, 0.8, 0.85, 0.9]
            },
            'deep_learner': {
                'description': '深度学习者',
                'characteristics': {
                    'initial_speed': 'low',
                    'plateau_speed': 'very_high',
                    'consistency': 'high'
                },
                'efficiency_signature': [0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
            },
            'adaptive_learner': {
                'description': '自适应学习者',
                'characteristics': {
                    'initial_speed': 'variable',
                    'plateau_speed': 'adaptive',
                    'consistency': 'variable'
                },
                'efficiency_signature': [0.6, 0.9, 0.4, 0.8, 0.7, 0.85]
            }
        }
        
        # 学习会话记录
        self.learning_sessions = {}
        self.efficiency_history = defaultdict(list)
        
        # 基准性能数据
        self.performance_benchmarks = self._initialize_benchmarks()
        
        self.logger.info("学习效率评估器初始化完成")
    
    def _initialize_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """初始化性能基准"""
        return {
            'game': {
                'average_accuracy': 0.75,
                'learning_speed': 0.8,
                'memory_retention': 0.7,
                'transfer_efficiency': 0.65
            },
            'physics': {
                'average_accuracy': 0.8,
                'learning_speed': 0.6,
                'memory_retention': 0.8,
                'transfer_efficiency': 0.7
            },
            'social': {
                'average_accuracy': 0.7,
                'learning_speed': 0.7,
                'memory_retention': 0.75,
                'transfer_efficiency': 0.6
            },
            'language': {
                'average_accuracy': 0.8,
                'learning_speed': 0.65,
                'memory_retention': 0.85,
                'transfer_efficiency': 0.75
            },
            'spatial': {
                'average_accuracy': 0.75,
                'learning_speed': 0.75,
                'memory_retention': 0.7,
                'transfer_efficiency': 0.7
            }
        }
    
    async def evaluate_learning_efficiency(self,
                                         domain: str,
                                         knowledge_base: Dict[str, Any],
                                         evaluation_tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估学习效率
        
        这是系统的核心方法，用于评估智能体在特定领域的学习效率。
        
        Args:
            domain: 学习领域
            knowledge_base: 知识库
            evaluation_tasks: 评估任务
            
        Returns:
            Dict: 学习效率评估结果
        """
        self.logger.info(f"开始评估领域 {domain} 的学习效率")
        
        session_id = f"session_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # 1. 初始化学习会话
            session = await self._initialize_learning_session(
                session_id, domain, start_time
            )
            
            # 2. 收集学习数据
            learning_data = await self._collect_learning_data(
                domain, knowledge_base, evaluation_tasks, session
            )
            
            # 3. 计算学习速度
            learning_speed = await self._calculate_learning_speed(
                learning_data['accuracy_progression']
            )
            
            # 4. 计算学习质量
            learning_quality = await self._calculate_learning_quality(
                learning_data['final_accuracy'], domain
            )
            
            # 5. 计算记忆效率
            memory_efficiency = await self._calculate_memory_efficiency(
                learning_data['retention_data']
            )
            
            # 6. 分析学习曲线
            learning_curve_analysis = await self._analyze_learning_curve(
                learning_data['accuracy_progression']
            )
            
            # 7. 评估资源利用效率
            resource_efficiency = await self._calculate_resource_efficiency(
                learning_data['resource_usage']
            )
            
            # 8. 识别学习模式
            learning_pattern = await self._identify_learning_pattern(
                learning_data['efficiency_signature']
            )
            
            # 9. 计算综合效率评分
            overall_efficiency = await self._calculate_overall_efficiency(
                learning_speed, learning_quality, memory_efficiency, 
                resource_efficiency, learning_curve_analysis['consistency_score']
            )
            
            # 10. 生成改进建议
            improvement_suggestions = await self._generate_improvement_suggestions(
                domain, learning_speed, learning_quality, learning_pattern
            )
            
            # 11. 优化学习参数
            optimal_parameters = await self._optimize_learning_parameters(
                domain, learning_pattern, overall_efficiency
            )
            
            # 12. 完成会话
            session.end_time = datetime.now()
            session.learning_speed = learning_speed
            session.learning_quality = learning_quality
            session.memory_efficiency = memory_efficiency
            session.accuracy_progression = learning_data['accuracy_progression']
            session.speed_progression = learning_data['speed_progression']
            session.consistency_score = learning_curve_analysis['consistency_score']
            session.final_performance = learning_data['final_accuracy']
            session.resource_usage = learning_data['resource_usage']
            session.learning_curve_slope = learning_curve_analysis['slope']
            session.plateau_threshold = learning_curve_analysis['plateau_threshold']
            
            # 保存会话
            self.learning_sessions[session_id] = session
            self.efficiency_history[domain].append(overall_efficiency)
            
            # 构建效率报告
            efficiency_report = LearningEfficiencyReport(
                overall_efficiency=overall_efficiency,
                speed_efficiency=learning_speed,
                quality_efficiency=learning_quality,
                resource_efficiency=resource_efficiency,
                consistency_rating=learning_curve_analysis['consistency_score'],
                learning_pattern=learning_pattern,
                improvement_suggestions=improvement_suggestions,
                optimal_parameters=optimal_parameters,
                benchmark_comparison=await self._compare_with_benchmarks(
                    domain, learning_speed, learning_quality, overall_efficiency
                )
            )
            
            result = {
                'session_id': session_id,
                'domain': domain,
                'metrics': session,
                'efficiency_report': efficiency_report,
                'learning_data': learning_data,
                'curve_analysis': learning_curve_analysis,
                'timestamp': start_time.isoformat(),
                'evaluation_duration': (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"学习效率评估完成，领域: {domain}, 效率: {overall_efficiency:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"学习效率评估失败: {str(e)}")
            raise
    
    async def _initialize_learning_session(self, session_id: str, domain: str, 
                                         start_time: datetime) -> LearningMetrics:
        """初始化学习会话"""
        
        return LearningMetrics(
            domain=domain,
            session_id=session_id,
            start_time=start_time,
            end_time=None,
            learning_speed=0.0,
            learning_quality=0.0,
            memory_efficiency=0.0,
            accuracy_progression=[],
            speed_progression=[],
            consistency_score=0.0,
            final_performance=0.0,
            resource_usage={},
            learning_curve_slope=0.0,
            plateau_threshold=0.0
        )
    
    async def _collect_learning_data(self, domain: str,
                                   knowledge_base: Dict[str, Any],
                                   evaluation_tasks: Dict[str, Any],
                                   session: LearningMetrics) -> Dict[str, Any]:
        """收集学习数据"""
        
        # 模拟学习数据收集过程
        learning_data = {
            'accuracy_progression': [],
            'speed_progression': [],
            'resource_usage': {},
            'final_accuracy': 0.0,
            'retention_data': [],
            'efficiency_signature': []
        }
        
        # 模拟学习过程
        num_iterations = 20
        initial_accuracy = 0.3
        final_accuracy = 0.85
        
        for i in range(num_iterations):
            # 模拟准确率进展
            progress_ratio = i / (num_iterations - 1)
            # 使用对数增长曲线模拟学习
            accuracy = initial_accuracy + (final_accuracy - initial_accuracy) * (1 - math.exp(-progress_ratio * 3))
            learning_data['accuracy_progression'].append(accuracy)
            
            # 模拟速度进展
            speed = 1.0 - (accuracy - initial_accuracy) / (final_accuracy - initial_accuracy)
            learning_data['speed_progression'].append(speed)
            
            # 生成效率签名
            efficiency_signature = self._generate_efficiency_signature(i, num_iterations, accuracy)
            learning_data['efficiency_signature'].append(efficiency_signature)
        
        # 最终准确率
        learning_data['final_accuracy'] = learning_data['accuracy_progression'][-1]
        
        # 资源使用情况
        learning_data['resource_usage'] = {
            'cpu_usage': np.random.uniform(0.3, 0.8),
            'memory_usage': np.random.uniform(0.4, 0.9),
            'time_spent': num_iterations * 0.1,  # 每个迭代0.1秒
            'energy_consumption': np.random.uniform(0.2, 0.7)
        }
        
        # 记忆保持数据
        for retention_point in [1, 5, 10, 20]:  # 不同时间点的保持率
            if retention_point <= len(learning_data['accuracy_progression']):
                retention_rate = learning_data['accuracy_progression'][-retention_point] * np.random.uniform(0.8, 0.95)
                learning_data['retention_data'].append(retention_rate)
        
        return learning_data
    
    def _generate_efficiency_signature(self, iteration: int, total_iterations: int, 
                                     accuracy: float) -> float:
        """生成效率签名"""
        
        # 基于迭代次数和准确率生成效率特征
        progress = iteration / total_iterations
        
        # 模拟学习效率变化
        if progress < 0.3:
            # 初期效率较低
            efficiency = 0.5 + accuracy * 0.3
        elif progress < 0.7:
            # 中期效率提升
            efficiency = 0.6 + accuracy * 0.4
        else:
            # 后期效率稳定
            efficiency = 0.7 + accuracy * 0.3
        
        return min(1.0, max(0.0, efficiency))
    
    async def _calculate_learning_speed(self, accuracy_progression: List[float]) -> float:
        """计算学习速度"""
        
        if len(accuracy_progression) < 2:
            return 0.0
        
        # 计算初始速度
        initial_samples = min(5, len(accuracy_progression))
        initial_slope = self._calculate_linear_slope(
            range(initial_samples), 
            accuracy_progression[:initial_samples]
        )
        
        # 计算总体速度
        total_slope = self._calculate_linear_slope(
            range(len(accuracy_progression)),
            accuracy_progression
        )
        
        # 计算最终速度
        final_samples = min(5, len(accuracy_progression))
        final_start = len(accuracy_progression) - final_samples
        final_slope = self._calculate_linear_slope(
            range(final_start, len(accuracy_progression)),
            accuracy_progression[final_start:]
        )
        
        # 综合速度评分
        speed_score = (
            initial_slope * 0.3 +      # 初始学习速度
            total_slope * 0.4 +        # 总体学习速度
            final_slope * 0.3          # 最终学习速度
        )
        
        return max(0.0, min(1.0, speed_score * 10))  # 标准化到[0,1]
    
    def _calculate_linear_slope(self, x_values: range, y_values: List[float]) -> float:
        """计算线性回归斜率"""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        x_array = np.array(list(x_values))
        y_array = np.array(y_values)
        
        # 计算线性回归斜率
        x_mean = np.mean(x_array)
        y_mean = np.mean(y_array)
        
        numerator = np.sum((x_array - x_mean) * (y_array - y_mean))
        denominator = np.sum((x_array - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    async def _calculate_learning_quality(self, final_accuracy: float, domain: str) -> float:
        """计算学习质量"""
        
        # 获取领域基准
        benchmark = self.performance_benchmarks.get(domain, {'average_accuracy': 0.7})
        target_accuracy = benchmark['average_accuracy']
        
        # 计算质量得分
        if final_accuracy >= target_accuracy:
            quality_score = 1.0
        else:
            quality_score = final_accuracy / target_accuracy
        
        # 应用质量调整因子
        quality_adjustments = {
            'stability': 0.1,  # 稳定性奖励
            'consistency': 0.1  # 一致性奖励
        }
        
        adjusted_quality = quality_score + sum(quality_adjustments.values())
        
        return max(0.0, min(1.0, adjusted_quality))
    
    async def _calculate_memory_efficiency(self, retention_data: List[float]) -> float:
        """计算记忆效率"""
        
        if not retention_data:
            return 0.5  # 默认中等效率
        
        # 计算记忆保持率
        initial_retention = retention_data[0] if retention_data else 1.0
        final_retention = retention_data[-1] if retention_data else 0.5
        
        # 计算保持稳定性
        retention_variance = np.var(retention_data) if len(retention_data) > 1 else 0.0
        stability_factor = 1.0 - min(retention_variance, 0.2)  # 方差越小，稳定性越好
        
        # 计算记忆效率
        efficiency = (
            final_retention * 0.6 +           # 最终保持率权重
            stability_factor * 0.4            # 稳定性权重
        )
        
        return max(0.0, min(1.0, efficiency))
    
    async def _analyze_learning_curve(self, accuracy_progression: List[float]) -> Dict[str, float]:
        """分析学习曲线"""
        
        if len(accuracy_progression) < 2:
            return {
                'slope': 0.0,
                'plateau_point': len(accuracy_progression),
                'plateau_threshold': 0.0,
                'consistency_score': 0.5,
                'curve_type': 'insufficient_data'
            }
        
        # 计算学习曲线斜率
        slope = self._calculate_linear_slope(range(len(accuracy_progression)), accuracy_progression)
        
        # 检测平台期
        plateau_point, plateau_threshold = self._detect_learning_plateau(accuracy_progression)
        
        # 计算一致性得分
        consistency_score = self._calculate_consistency_score(accuracy_progression)
        
        # 确定曲线类型
        curve_type = self._classify_learning_curve(accuracy_progression)
        
        return {
            'slope': slope,
            'plateau_point': plateau_point,
            'plateau_threshold': plateau_threshold,
            'consistency_score': consistency_score,
            'curve_type': curve_type
        }
    
    def _detect_learning_plateau(self, accuracy_progression: List[float]) -> Tuple[int, float]:
        """检测学习平台期"""
        
        if len(accuracy_progression) < 3:
            return len(accuracy_progression), 0.0
        
        # 计算滑动窗口的改进率
        window_size = 3
        improvement_rates = []
        
        for i in range(window_size, len(accuracy_progression)):
            window_data = accuracy_progression[i - window_size:i]
            if len(window_data) >= 2:
                window_slope = self._calculate_linear_slope(range(len(window_data)), window_data)
                improvement_rates.append(window_slope)
        
        # 检测平台期开始点
        plateau_threshold = self.efficiency_params['plateau_detection_threshold']
        
        plateau_point = len(accuracy_progression)
        for i, rate in enumerate(improvement_rates):
            if rate < plateau_threshold:
                plateau_point = i + window_size
                break
        
        return plateau_point, plateau_threshold
    
    def _calculate_consistency_score(self, accuracy_progression: List[float]) -> float:
        """计算一致性得分"""
        
        if len(accuracy_progression) < 3:
            return 0.5
        
        # 计算准确率变化的标准差
        accuracy_std = np.std(accuracy_progression)
        
        # 计算趋势一致性
        direction_changes = 0
        for i in range(1, len(accuracy_progression)):
            prev_change = accuracy_progression[i] - accuracy_progression[i-1]
            if i > 1:
                curr_change = accuracy_progression[i] - accuracy_progression[i-1]
                if (prev_change > 0 and curr_change < 0) or (prev_change < 0 and curr_change > 0):
                    direction_changes += 1
        
        # 一致性得分（变化越小，一致性越高）
        direction_consistency = 1.0 - (direction_changes / max(len(accuracy_progression) - 2, 1))
        std_consistency = max(0.0, 1.0 - accuracy_std)
        
        consistency_score = (direction_consistency * 0.6 + std_consistency * 0.4)
        
        return max(0.0, min(1.0, consistency_score))
    
    def _classify_learning_curve(self, accuracy_progression: List[float]) -> str:
        """分类学习曲线类型"""
        
        if len(accuracy_progression) < 5:
            return 'insufficient_data'
        
        initial_improvement = accuracy_progression[4] - accuracy_progression[0]
        final_improvement = accuracy_progression[-1] - accuracy_progression[-5]
        
        if initial_improvement > 0.3 and final_improvement < 0.1:
            return 'rapid_then_plateau'
        elif initial_improvement < 0.1 and final_improvement > 0.2:
            return 'slow_then_accelerate'
        elif abs(initial_improvement - final_improvement) < 0.1:
            return 'consistent_growth'
        else:
            return 'irregular_pattern'
    
    async def _calculate_resource_efficiency(self, resource_usage: Dict[str, float]) -> float:
        """计算资源利用效率"""
        
        # 定义理想资源使用率
        ideal_usage = {
            'cpu_usage': 0.6,
            'memory_usage': 0.7,
            'time_spent': 1.0,  # 标准化时间消耗
            'energy_consumption': 0.5
        }
        
        efficiency_components = []
        
        for resource_type, actual_usage in resource_usage.items():
            ideal_usage_rate = ideal_usage.get(resource_type, 0.7)
            
            # 计算资源利用效率
            if actual_usage <= ideal_usage_rate:
                resource_efficiency = 1.0 - (ideal_usage_rate - actual_usage) / ideal_usage_rate * 0.3
            else:
                resource_efficiency = ideal_usage_rate / actual_usage * 0.7
            
            efficiency_components.append(max(0.0, min(1.0, resource_efficiency)))
        
        # 计算总体资源效率
        total_efficiency = np.mean(efficiency_components) if efficiency_components else 0.5
        
        return total_efficiency
    
    async def _identify_learning_pattern(self, efficiency_signature: List[float]) -> str:
        """识别学习模式"""
        
        if len(efficiency_signature) < 6:
            return 'unknown'
        
        # 计算与已知模式的相似度
        pattern_similarities = {}
        
        for pattern_name, pattern_info in self.learning_patterns.items():
            signature = pattern_info['efficiency_signature']
            similarity = self._calculate_pattern_similarity(efficiency_signature, signature)
            pattern_similarities[pattern_name] = similarity
        
        # 选择最相似的模式
        best_pattern = max(pattern_similarities.items(), key=lambda x: x[1])
        
        # 如果相似度太低，标记为混合模式
        if best_pattern[1] < 0.6:
            return 'hybrid_pattern'
        
        return best_pattern[0]
    
    def _calculate_pattern_similarity(self, signature1: List[float], signature2: List[float]) -> float:
        """计算模式相似度"""
        
        if len(signature1) != len(signature2):
            min_len = min(len(signature1), len(signature2))
            signature1 = signature1[:min_len]
            signature2 = signature2[:min_len]
        
        # 使用余弦相似度
        dot_product = sum(a * b for a, b in zip(signature1, signature2))
        magnitude1 = math.sqrt(sum(a * a for a in signature1))
        magnitude2 = math.sqrt(sum(b * b for b in signature2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _calculate_overall_efficiency(self, learning_speed: float,
                                          learning_quality: float,
                                          memory_efficiency: float,
                                          resource_efficiency: float,
                                          consistency_score: float) -> float:
        """计算综合学习效率"""
        
        # 应用权重
        weights = self.efficiency_params
        
        overall_efficiency = (
            learning_speed * weights['speed_weight'] +
            learning_quality * weights['quality_weight'] +
            memory_efficiency * weights['memory_weight'] +
            resource_efficiency * weights['resource_weight'] +
            consistency_score * weights['consistency_weight']
        )
        
        return max(0.0, min(1.0, overall_efficiency))
    
    async def _generate_improvement_suggestions(self, domain: str,
                                              learning_speed: float,
                                              learning_quality: float,
                                              learning_pattern: str) -> List[str]:
        """生成改进建议"""
        
        suggestions = []
        
        # 基于学习速度的建议
        if learning_speed < 0.6:
            suggestions.append("增加学习率以加快初期学习速度")
            suggestions.append("使用更有效的预训练策略")
            suggestions.append("考虑引入课程学习方法")
        
        # 基于学习质量的建议
        if learning_quality < 0.7:
            suggestions.append("优化模型架构以提高表达能力")
            suggestions.append("增加训练数据的质量和多样性")
            suggestions.append("使用正则化技术防止过拟合")
        
        # 基于学习模式的建议
        if learning_pattern == 'rapid_learner':
            suggestions.append("稳定学习过程，避免过度拟合")
            suggestions.append("增加验证机制确保学习稳定性")
        elif learning_pattern == 'steady_learner':
            suggestions.append("继续保持稳定的学习节奏")
            suggestions.append("在稳定期后可尝试加速策略")
        elif learning_pattern == 'deep_learner':
            suggestions.append("尝试更激进的学习策略以提高初期速度")
            suggestions.append("考虑渐进式深度学习")
        elif learning_pattern == 'adaptive_learner':
            suggestions.append("优化自适应机制，减少不必要的变化")
            suggestions.append("建立更稳定的适应性基础")
        
        # 领域特定建议
        domain_specific = {
            'game': ["增加策略多样性的训练", "使用强化学习方法"],
            'physics': ["加强数学基础训练", "使用仿真环境学习"],
            'social': ["增加社交场景模拟", "使用多智能体学习"],
            'language': ["扩大语料库规模", "使用预训练语言模型"],
            'spatial': ["增加空间推理训练", "使用3D环境学习"]
        }
        
        if domain in domain_specific:
            suggestions.extend(domain_specific[domain])
        
        return suggestions
    
    async def _optimize_learning_parameters(self, domain: str,
                                          learning_pattern: str,
                                          overall_efficiency: float) -> Dict[str, Any]:
        """优化学习参数"""
        
        # 基于学习模式优化参数
        pattern_configs = {
            'rapid_learner': {
                'learning_rate': 0.01,
                'batch_size': 32,
                'regularization': 0.001,
                'patience': 5
            },
            'steady_learner': {
                'learning_rate': 0.005,
                'batch_size': 64,
                'regularization': 0.01,
                'patience': 10
            },
            'deep_learner': {
                'learning_rate': 0.002,
                'batch_size': 16,
                'regularization': 0.1,
                'patience': 15
            },
            'adaptive_learner': {
                'learning_rate': 0.008,
                'batch_size': 48,
                'regularization': 0.005,
                'patience': 8
            }
        }
        
        base_config = pattern_configs.get(learning_pattern, pattern_configs['steady_learner'])
        
        # 基于总体效率调整参数
        if overall_efficiency < 0.5:
            # 低效率情况，增加学习稳定性和正则化
            base_config['regularization'] *= 1.5
            base_config['patience'] += 3
        elif overall_efficiency > 0.8:
            # 高效率情况，可以尝试更激进的参数
            base_config['learning_rate'] *= 1.2
            base_config['batch_size'] = min(base_config['batch_size'] * 2, 128)
        
        return base_config
    
    async def _compare_with_benchmarks(self, domain: str,
                                     learning_speed: float,
                                     learning_quality: float,
                                     overall_efficiency: float) -> Dict[str, float]:
        """与基准性能比较"""
        
        benchmark = self.performance_benchmarks.get(domain, {})
        
        return {
            'speed_vs_benchmark': learning_speed / benchmark.get('learning_speed', 0.7),
            'quality_vs_benchmark': learning_quality / benchmark.get('average_accuracy', 0.7),
            'overall_vs_benchmark': overall_efficiency / self.efficiency_params['efficiency_benchmark'],
            'retention_vs_benchmark': 1.0  # 默认值，实际中需要计算
        }
    
    def get_efficiency_statistics(self) -> Dict[str, Any]:
        """获取效率统计信息"""
        
        if not self.learning_sessions:
            return {
                'total_sessions': 0,
                'average_efficiency': 0.0,
                'efficiency_trend': [],
                'common_patterns': []
            }
        
        # 计算总体统计
        all_efficiencies = [
            session.learning_speed * 0.3 + session.learning_quality * 0.4 +
            session.memory_efficiency * 0.2 + session.consistency_score * 0.1
            for session in self.learning_sessions.values()
        ]
        
        average_efficiency = np.mean(all_efficiencies)
        
        # 效率趋势
        efficiency_trend = []
        for domain, efficiencies in self.efficiency_history.items():
            if len(efficiencies) >= 2:
                trend_slope = self._calculate_linear_slope(
                    range(len(efficiencies)), efficiencies
                )
                efficiency_trend.append({
                    'domain': domain,
                    'trend_slope': trend_slope,
                    'current_efficiency': efficiencies[-1]
                })
        
        # 常见学习模式统计
        patterns = []
        # 这里可以基于历史数据统计常见模式
        
        return {
            'total_sessions': len(self.learning_sessions),
            'average_efficiency': average_efficiency,
            'efficiency_trend': efficiency_trend,
            'common_patterns': patterns,
            'domain_performance': {
                domain: {
                    'average_efficiency': np.mean(effs),
                    'session_count': len(effs),
                    'best_efficiency': max(effs),
                    'improvement_trend': self._calculate_improvement_trend(effs)
                }
                for domain, effs in self.efficiency_history.items()
            }
        }
    
    def _calculate_improvement_trend(self, efficiencies: List[float]) -> float:
        """计算改进趋势"""
        if len(efficiencies) < 2:
            return 0.0
        
        recent_efficiency = np.mean(efficiencies[-3:])  # 最近3次平均
        early_efficiency = np.mean(efficiencies[:3])    # 前3次平均
        
        return recent_efficiency - early_efficiency


def create_learning_efficiency(config: Optional[Dict[str, Any]] = None) -> LearningEfficiency:
    """创建学习效率评估器实例的便捷函数"""
    return LearningEfficiency(config or {})


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建学习效率评估器
        evaluator = create_learning_efficiency({
            'speed_weight': 0.3,
            'quality_weight': 0.4,
            'plateau_detection_threshold': 0.01
        })
        
        # 评估学习效率
        result = await evaluator.evaluate_learning_efficiency(
            domain='game',
            knowledge_base={'concepts': ['strategy', 'tactics'], 'examples': []},
            evaluation_tasks={'tasks': ['classification', 'prediction']}
        )
        
        print(f"学习效率评估完成: {result['efficiency_report'].overall_efficiency:.3f}")
    
    # 运行示例
    # asyncio.run(main())