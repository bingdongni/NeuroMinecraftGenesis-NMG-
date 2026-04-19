# -*- coding: utf-8 -*-
"""
适应指标计算器
Adaptation Metrics Calculator

该模块负责计算和评估智能体在新领域中的适应能力。
通过多维度的适应指标分析，全面评估智能体的适应速度、
适应质量和长期适应效果，为跨域学习提供科学的适应评估。

主要功能：
- 适应速度测量
- 适应质量评估
- 适应稳定性分析
- 适应模式识别
- 长期适应效果预测
- 个性化适应策略优化

作者: AI系统
日期: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
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
class AdaptationMetrics:
    """适应指标数据结构"""
    domain: str
    adaptation_session_id: str
    start_time: datetime
    completion_time: Optional[datetime]
    adaptation_speed: float                  # 适应速度
    adaptation_quality: float               # 适应质量
    stability_score: float                  # 稳定性得分
    success_rate: float                     # 成功率
    error_rate: float                       # 错误率
    adaptation_curve: List[Dict[str, float]] # 适应曲线
    performance_evolution: List[float]      # 性能演化
    consistency_measure: float              # 一致性度量
    robustness_score: float                 # 鲁棒性得分
    transfer_readiness: float              # 迁移就绪度
    adaptation_efficiency: float           # 适应效率


@dataclass
class AdaptationReport:
    """适应报告"""
    overall_adaptation_score: float          # 总体适应评分
    adaptation_velocity: float               # 适应速度
    adaptation_quality_score: float          # 适应质量评分
    stability_rating: float                  # 稳定性评级
    adaptability_index: float                # 适应能力指数
    adaptation_pattern: str                  # 适应模式
    resilience_score: float                  # 弹性得分
    optimization_recommendations: List[str]  # 优化建议
    risk_assessment: Dict[str, Any]          # 风险评估
    future_adaptation_potential: float       # 未来适应潜力


class AdaptationMetrics:
    """
    适应指标计算器
    
    负责评估和测量智能体在不同领域的适应能力，
    通过多维度指标分析，为跨域学习提供科学的适应评估。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('AdaptationMetrics')
        
        # 适应指标参数配置
        self.adaptation_params = {
            'speed_threshold': config.get('speed_threshold', 0.1),      # 适应速度阈值
            'quality_threshold': config.get('quality_threshold', 0.8),  # 适应质量阈值
            'stability_window': config.get('stability_window', 5),      # 稳定性分析窗口
            'adaptation_phases': config.get('adaptation_phases', ['initial', 'transition', 'stabilization']),
            'performance_targets': config.get('performance_targets', {
                'initial_performance': 0.3,
                'target_performance': 0.9,
                'minimum_performance': 0.6
            }),
            'stress_test_levels': config.get('stress_test_levels', [0.1, 0.3, 0.5, 0.7, 0.9])
        }
        
        # 适应模式定义
        self.adaptation_patterns = {
            'immediate_adapter': {
                'description': '即时适应者',
                'characteristics': {
                    'initial_response': 'rapid',
                    'performance_stability': 'high',
                    'error_recovery': 'fast'
                },
                'typical_curve': [0.8, 0.85, 0.9, 0.92, 0.95],
                'time_to_stability': 'short'
            },
            'gradual_adapter': {
                'description': '渐进适应者',
                'characteristics': {
                    'initial_response': 'moderate',
                    'performance_stability': 'medium',
                    'error_recovery': 'steady'
                },
                'typical_curve': [0.4, 0.5, 0.65, 0.75, 0.85],
                'time_to_stability': 'medium'
            },
            'oscillating_adapter': {
                'description': '震荡适应者',
                'characteristics': {
                    'initial_response': 'variable',
                    'performance_stability': 'low',
                    'error_recovery': 'inconsistent'
                },
                'typical_curve': [0.6, 0.4, 0.7, 0.5, 0.8],
                'time_to_stability': 'long'
            },
            'resilient_adapter': {
                'description': '弹性适应者',
                'characteristics': {
                    'initial_response': 'adaptive',
                    'performance_stability': 'high',
                    'error_recovery': 'excellent'
                },
                'typical_curve': [0.5, 0.3, 0.7, 0.9, 0.95],
                'time_to_stability': 'short'
            }
        }
        
        # 适应历史记录
        self.adaptation_sessions = {}
        self.adaptation_performance_history = defaultdict(list)
        
        # 领域适应基准
        self.adaptation_benchmarks = self._initialize_adaptation_benchmarks()
        
        self.logger.info("适应指标计算器初始化完成")
    
    def _initialize_adaptation_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """初始化适应基准数据"""
        return {
            'game': {
                'average_adaptation_time': 120.0,  # 秒
                'adaptation_success_rate': 0.85,
                'stability_threshold': 0.8,
                'performance_improvement_rate': 0.05
            },
            'physics': {
                'average_adaptation_time': 180.0,
                'adaptation_success_rate': 0.8,
                'stability_threshold': 0.85,
                'performance_improvement_rate': 0.04
            },
            'social': {
                'average_adaptation_time': 150.0,
                'adaptation_success_rate': 0.75,
                'stability_threshold': 0.7,
                'performance_improvement_rate': 0.06
            },
            'language': {
                'average_adaptation_time': 200.0,
                'adaptation_success_rate': 0.9,
                'stability_threshold': 0.9,
                'performance_improvement_rate': 0.03
            },
            'spatial': {
                'average_adaptation_time': 100.0,
                'adaptation_success_rate': 0.8,
                'stability_threshold': 0.75,
                'performance_improvement_rate': 0.07
            }
        }
    
    async def evaluate_adaptation_speed(self,
                                      target_domain: str,
                                      transferred_knowledge: Dict[str, Any],
                                      adaptation_tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估适应速度
        
        这是系统的核心方法，用于评估智能体在新领域中的适应速度。
        
        Args:
            target_domain: 目标领域
            transferred_knowledge: 迁移的知识
            adaptation_tasks: 适应任务
            
        Returns:
            Dict: 适应速度评估结果
        """
        self.logger.info(f"开始评估领域 {target_domain} 的适应速度")
        
        session_id = f"adaptation_{target_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # 1. 初始化适应会话
            session = await self._initialize_adaptation_session(
                session_id, target_domain, start_time
            )
            
            # 2. 收集适应数据
            adaptation_data = await self._collect_adaptation_data(
                target_domain, transferred_knowledge, adaptation_tasks, session
            )
            
            # 3. 计算适应速度
            adaptation_speed = await self._calculate_adaptation_speed(
                adaptation_data['performance_evolution'], 
                adaptation_data['time_points']
            )
            
            # 4. 评估适应质量
            adaptation_quality = await self._evaluate_adaptation_quality(
                target_domain, adaptation_data['final_performance']
            )
            
            # 5. 分析适应稳定性
            stability_analysis = await self._analyze_adaptation_stability(
                adaptation_data['performance_evolution']
            )
            
            # 6. 计算成功率
            success_rate = await self._calculate_success_rate(
                adaptation_data['task_results']
            )
            
            # 7. 计算错误率
            error_rate = await self._calculate_error_rate(
                adaptation_data['error_log']
            )
            
            # 8. 评估鲁棒性
            robustness_score = await self._evaluate_robustness(
                target_domain, adaptation_data['stress_test_results']
            )
            
            # 9. 评估迁移就绪度
            transfer_readiness = await self._evaluate_transfer_readiness(
                target_domain, adaptation_data
            )
            
            # 10. 计算适应效率
            adaptation_efficiency = await self._calculate_adaptation_efficiency(
                adaptation_speed, adaptation_quality, stability_analysis['consistency_score']
            )
            
            # 11. 识别适应模式
            adaptation_pattern = await self._identify_adaptation_pattern(
                adaptation_data['performance_evolution']
            )
            
            # 12. 完成会话
            session.completion_time = datetime.now()
            session.adaptation_speed = adaptation_speed
            session.adaptation_quality = adaptation_quality
            session.stability_score = stability_analysis['stability_score']
            session.success_rate = success_rate
            session.error_rate = error_rate
            session.adaptation_curve = adaptation_data['adaptation_curve']
            session.performance_evolution = adaptation_data['performance_evolution']
            session.consistency_measure = stability_analysis['consistency_score']
            session.robustness_score = robustness_score
            session.transfer_readiness = transfer_readiness
            session.adaptation_efficiency = adaptation_efficiency
            
            # 保存会话
            self.adaptation_sessions[session_id] = session
            self.adaptation_performance_history[target_domain].append(adaptation_efficiency)
            
            # 构建适应报告
            adaptation_report = AdaptationReport(
                overall_adaptation_score=(
                    adaptation_speed * 0.25 +
                    adaptation_quality * 0.3 +
                    stability_analysis['stability_score'] * 0.2 +
                    robustness_score * 0.15 +
                    transfer_readiness * 0.1
                ),
                adaptation_velocity=adaptation_speed,
                adaptation_quality_score=adaptation_quality,
                stability_rating=stability_analysis['stability_score'],
                adaptability_index=await self._calculate_adaptability_index(session),
                adaptation_pattern=adaptation_pattern,
                resilience_score=robustness_score,
                optimization_recommendations=await self._generate_optimization_recommendations(
                    target_domain, adaptation_speed, adaptation_quality, adaptation_pattern
                ),
                risk_assessment=await self._assess_adaptation_risks(
                    target_domain, adaptation_data
                ),
                future_adaptation_potential=await self._predict_future_adaptation_potential(
                    target_domain, adaptation_data
                )
            )
            
            result = {
                'session_id': session_id,
                'target_domain': target_domain,
                'metrics': session,
                'adaptation_report': adaptation_report,
                'adaptation_data': adaptation_data,
                'stability_analysis': stability_analysis,
                'timestamp': start_time.isoformat(),
                'evaluation_duration': (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"适应速度评估完成，领域: {target_domain}, 速度: {adaptation_speed:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"适应速度评估失败: {str(e)}")
            raise
    
    async def _initialize_adaptation_session(self, session_id: str, domain: str, 
                                           start_time: datetime) -> AdaptationMetrics:
        """初始化适应会话"""
        
        return AdaptationMetrics(
            domain=domain,
            adaptation_session_id=session_id,
            start_time=start_time,
            completion_time=None,
            adaptation_speed=0.0,
            adaptation_quality=0.0,
            stability_score=0.0,
            success_rate=0.0,
            error_rate=0.0,
            adaptation_curve=[],
            performance_evolution=[],
            consistency_measure=0.0,
            robustness_score=0.0,
            transfer_readiness=0.0,
            adaptation_efficiency=0.0
        )
    
    async def _collect_adaptation_data(self, target_domain: str,
                                     transferred_knowledge: Dict[str, Any],
                                     adaptation_tasks: Dict[str, Any],
                                     session: AdaptationMetrics) -> Dict[str, Any]:
        """收集适应数据"""
        
        adaptation_data = {
            'performance_evolution': [],
            'time_points': [],
            'task_results': [],
            'error_log': [],
            'adaptation_curve': [],
            'final_performance': 0.0,
            'stress_test_results': {}
        }
        
        # 模拟适应过程数据
        num_assessments = 15
        adaptation_duration = 300  # 5分钟适应过程
        
        for i in range(num_assessments):
            time_point = (i / (num_assessments - 1)) * adaptation_duration
            adaptation_data['time_points'].append(time_point)
            
            # 模拟性能进展（使用logistic增长模型）
            progress_ratio = i / (num_assessments - 1)
            initial_performance = self.adaptation_params['performance_targets']['initial_performance']
            target_performance = self.adaptation_params['performance_targets']['target_performance']
            
            # 添加一些噪声和波动
            noise = np.random.normal(0, 0.05)
            performance = initial_performance + (target_performance - initial_performance) * (
                1 - math.exp(-progress_ratio * 2)
            ) + noise
            performance = max(0.0, min(1.0, performance))
            
            adaptation_data['performance_evolution'].append(performance)
            
            # 适应曲线数据
            curve_point = {
                'time': time_point,
                'performance': performance,
                'confidence': 0.8 + 0.1 * math.sin(i * 0.5),
                'error_rate': max(0.0, 1.0 - performance + np.random.normal(0, 0.02)),
                'stability': 0.9 - abs(performance - 0.5) * 0.1
            }
            adaptation_data['adaptation_curve'].append(curve_point)
            
            # 任务结果
            task_success = np.random.choice([True, False], p=[performance, 1-performance])
            adaptation_data['task_results'].append(task_success)
            
            # 错误日志
            if not task_success:
                error_types = ['timeout', 'incorrect_response', 'processing_error', 'memory_overflow']
                error = {
                    'time': time_point,
                    'type': np.random.choice(error_types),
                    'severity': np.random.choice(['low', 'medium', 'high']),
                    'recovery_time': np.random.uniform(0.1, 2.0)
                }
                adaptation_data['error_log'].append(error)
        
        # 最终性能
        adaptation_data['final_performance'] = adaptation_data['performance_evolution'][-1]
        
        # 压力测试结果
        for level in self.adaptation_params['stress_test_levels']:
            stress_performance = adaptation_data['final_performance'] * (1 - level * 0.3)
            adaptation_data['stress_test_results'][level] = {
                'performance': stress_performance,
                'degradation': level * 0.3,
                'stress_resistance': max(0.0, 1.0 - level * 0.5)
            }
        
        return adaptation_data
    
    async def _calculate_adaptation_speed(self, performance_evolution: List[float],
                                        time_points: List[float]) -> float:
        """计算适应速度"""
        
        if len(performance_evolution) < 2 or len(time_points) < 2:
            return 0.0
        
        # 计算早期适应速度（前25%的数据）
        early_end = max(2, len(performance_evolution) // 4)
        early_performance = performance_evolution[:early_end]
        early_time = time_points[:early_end]
        
        early_slope = self._calculate_regression_slope(early_time, early_performance)
        
        # 计算总体适应速度
        total_slope = self._calculate_regression_slope(time_points, performance_evolution)
        
        # 计算达到目标性能的所需时间
        target_performance = self.adaptation_params['performance_targets']['target_performance']
        time_to_target = self._calculate_time_to_target(performance_evolution, time_points, target_performance)
        
        # 计算适应速度评分
        speed_score = (
            early_slope * 0.4 +                # 早期适应速度
            total_slope * 0.4 +                # 总体适应速度
            (1.0 / max(time_to_target, 1.0)) * 0.2  # 目标达成速度
        )
        
        # 标准化到[0,1]范围
        normalized_speed = max(0.0, min(1.0, speed_score * 10))
        
        return normalized_speed
    
    def _calculate_regression_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """计算回归斜率"""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        # 计算皮尔逊相关系数和斜率
        correlation_matrix = np.corrcoef(x_array, y_array)
        correlation = correlation_matrix[0, 1]
        
        # 计算斜率
        x_std = np.std(x_array)
        y_std = np.std(y_array)
        
        if x_std == 0 or y_std == 0 or np.isnan(correlation):
            return 0.0
        
        slope = correlation * (y_std / x_std)
        
        return slope
    
    def _calculate_time_to_target(self, performance_evolution: List[float],
                                time_points: List[float], target: float) -> float:
        """计算达到目标性能所需时间"""
        
        for i, performance in enumerate(performance_evolution):
            if performance >= target:
                return time_points[i]
        
        # 如果未达到目标，返回总时间
        return time_points[-1] if time_points else 0.0
    
    async def _evaluate_adaptation_quality(self, target_domain: str, final_performance: float) -> float:
        """评估适应质量"""
        
        # 获取领域基准
        benchmark = self.adaptation_benchmarks.get(target_domain, {})
        target_threshold = benchmark.get('stability_threshold', 0.8)
        
        # 计算质量得分
        if final_performance >= target_threshold:
            quality_score = 1.0
        else:
            quality_score = final_performance / target_threshold
        
        # 质量调整因子
        if final_performance >= 0.9:
            quality_bonus = 0.1  # 高性能奖励
        elif final_performance >= 0.8:
            quality_bonus = 0.05  # 中高性能奖励
        else:
            quality_bonus = 0.0
        
        adjusted_quality = min(1.0, quality_score + quality_bonus)
        
        return adjusted_quality
    
    async def _analyze_adaptation_stability(self, performance_evolution: List[float]) -> Dict[str, float]:
        """分析适应稳定性"""
        
        if len(performance_evolution) < 3:
            return {
                'stability_score': 0.5,
                'consistency_score': 0.5,
                'volatility_index': 1.0,
                'stabilization_point': len(performance_evolution)
            }
        
        # 计算性能方差
        performance_variance = np.var(performance_evolution)
        
        # 计算稳定性得分（方差越小，稳定性越高）
        stability_score = max(0.0, 1.0 - performance_variance * 2)
        
        # 计算一致性得分
        consistency_score = self._calculate_consistency_score(performance_evolution)
        
        # 计算波动性指数
        performance_changes = [abs(performance_evolution[i] - performance_evolution[i-1]) 
                              for i in range(1, len(performance_evolution))]
        volatility_index = np.mean(performance_changes) if performance_changes else 1.0
        
        # 找到稳定化点
        stabilization_point = self._find_stabilization_point(performance_evolution)
        
        return {
            'stability_score': stability_score,
            'consistency_score': consistency_score,
            'volatility_index': volatility_index,
            'stabilization_point': stabilization_point
        }
    
    def _calculate_consistency_score(self, performance_evolution: List[float]) -> float:
        """计算一致性得分"""
        
        if len(performance_evolution) < 3:
            return 0.5
        
        # 计算方向变化次数
        direction_changes = 0
        for i in range(1, len(performance_evolution)):
            prev_change = performance_evolution[i] - performance_evolution[i-1]
            if i > 1:
                curr_change = performance_evolution[i] - performance_evolution[i-1]
                if (prev_change > 0 and curr_change < 0) or (prev_change < 0 and curr_change > 0):
                    direction_changes += 1
        
        # 一致性得分
        total_possible_changes = len(performance_evolution) - 2
        consistency_score = 1.0 - (direction_changes / max(total_possible_changes, 1))
        
        return max(0.0, min(1.0, consistency_score))
    
    def _find_stabilization_point(self, performance_evolution: List[float]) -> int:
        """找到稳定化点"""
        
        if len(performance_evolution) < 5:
            return len(performance_evolution)
        
        # 寻找性能变化小于阈值的连续区间
        stability_threshold = 0.02
        window_size = 3
        
        for i in range(len(performance_evolution) - window_size + 1):
            window = performance_evolution[i:i + window_size]
            if np.std(window) < stability_threshold:
                return i + window_size
        
        return len(performance_evolution)  # 未找到稳定区间
    
    async def _calculate_success_rate(self, task_results: List[bool]) -> float:
        """计算成功率"""
        
        if not task_results:
            return 0.0
        
        successful_tasks = sum(task_results)
        success_rate = successful_tasks / len(task_results)
        
        return success_rate
    
    async def _calculate_error_rate(self, error_log: List[Dict[str, Any]]) -> float:
        """计算错误率"""
        
        # 简化计算：错误数量与总评估次数的比例
        total_evaluations = len(self.adaptation_params['adaptation_phases']) * 10  # 假设每次评估10次
        error_rate = len(error_log) / max(total_evaluations, 1)
        
        return min(1.0, error_rate)
    
    async def _evaluate_robustness(self, target_domain: str,
                                 stress_test_results: Dict[str, float]) -> float:
        """评估鲁棒性"""
        
        if not stress_test_results:
            return 0.5
        
        # 计算在不同压力水平下的性能保持能力
        robustness_scores = []
        
        for stress_level, results in stress_test_results.items():
            performance = results.get('performance', 0.0)
            degradation = results.get('degradation', 0.0)
            
            # 鲁棒性评分：高性能和低退化率的组合
            robustness_score = performance * (1.0 - degradation)
            robustness_scores.append(robustness_score)
        
        # 综合鲁棒性评分
        overall_robustness = np.mean(robustness_scores) if robustness_scores else 0.5
        
        return max(0.0, min(1.0, overall_robustness))
    
    async def _evaluate_transfer_readiness(self, target_domain: str,
                                         adaptation_data: Dict[str, Any]) -> float:
        """评估迁移就绪度"""
        
        # 基于适应质量、稳定性、鲁棒性评估迁移就绪度
        final_performance = adaptation_data['final_performance']
        stability_score = self._calculate_adaptation_stability_simple(
            adaptation_data['performance_evolution']
        )
        
        # 模拟基于stress test的鲁棒性评估
        robustness_estimate = self._estimate_robustness_from_performance(final_performance)
        
        # 计算迁移就绪度
        transfer_readiness = (
            final_performance * 0.4 +
            stability_score * 0.3 +
            robustness_estimate * 0.3
        )
        
        return max(0.0, min(1.0, transfer_readiness))
    
    def _calculate_adaptation_stability_simple(self, performance_evolution: List[float]) -> float:
        """简化的适应稳定性计算"""
        
        if len(performance_evolution) < 2:
            return 0.5
        
        # 计算最后几个点的方差作为稳定性指标
        recent_points = performance_evolution[-3:] if len(performance_evolution) >= 3 else performance_evolution
        stability = max(0.0, 1.0 - np.std(recent_points) * 2)
        
        return stability
    
    def _estimate_robustness_from_performance(self, performance: float) -> float:
        """从性能估算鲁棒性"""
        
        # 简化的鲁棒性估算：高性能通常意味着更好的鲁棒性
        # 但也要考虑性能是否是通过过度拟合获得的
        if performance > 0.8:
            robustness = 0.7 + (performance - 0.8) * 1.5
        else:
            robustness = performance * 0.7
        
        return min(1.0, robustness)
    
    async def _calculate_adaptation_efficiency(self, adaptation_speed: float,
                                             adaptation_quality: float,
                                             consistency_score: float) -> float:
        """计算适应效率"""
        
        # 适应效率 = 适应速度 * 适应质量 * 一致性
        efficiency = adaptation_speed * adaptation_quality * consistency_score
        
        return max(0.0, min(1.0, efficiency))
    
    async def _identify_adaptation_pattern(self, performance_evolution: List[float]) -> str:
        """识别适应模式"""
        
        if len(performance_evolution) < 5:
            return 'insufficient_data'
        
        # 提取适应特征
        initial_performance = performance_evolution[0]
        final_performance = performance_evolution[-1]
        peak_performance = max(performance_evolution)
        
        # 计算变化特征
        initial_slope = self._calculate_initial_slope(performance_evolution)
        stability = self._calculate_final_stability(performance_evolution)
        
        # 模式分类逻辑
        if initial_slope > 0.3 and stability > 0.7:
            return 'immediate_adapter'
        elif initial_slope < 0.1 and final_performance > 0.8:
            return 'gradual_adapter'
        elif stability < 0.5:
            return 'oscillating_adapter'
        elif peak_performance > 0.9 and self._has_recovery_pattern(performance_evolution):
            return 'resilient_adapter'
        else:
            return 'mixed_adapter'
    
    def _calculate_initial_slope(self, performance_evolution: List[float]) -> float:
        """计算初始斜率"""
        
        if len(performance_evolution) < 3:
            return 0.0
        
        early_points = performance_evolution[:3]
        return (early_points[-1] - early_points[0]) / 2
    
    def _calculate_final_stability(self, performance_evolution: List[float]) -> float:
        """计算最终稳定性"""
        
        if len(performance_evolution) < 5:
            return 0.5
        
        final_points = performance_evolution[-3:]
        return max(0.0, 1.0 - np.std(final_points) * 2)
    
    def _has_recovery_pattern(self, performance_evolution: List[float]) -> bool:
        """检查是否有恢复模式"""
        
        if len(performance_evolution) < 6:
            return False
        
        # 寻找下降后的上升模式
        mid_point = len(performance_evolution) // 2
        before_mid = performance_evolution[:mid_point]
        after_mid = performance_evolution[mid_point:]
        
        return (min(before_mid) < 0.6 and 
                max(after_mid) > min(before_mid) + 0.2)
    
    async def _calculate_adaptability_index(self, session: AdaptationMetrics) -> float:
        """计算适应能力指数"""
        
        # 综合考虑多个维度
        index = (
            session.adaptation_speed * 0.3 +
            session.adaptation_quality * 0.3 +
            session.stability_score * 0.2 +
            session.robustness_score * 0.2
        )
        
        return max(0.0, min(1.0, index))
    
    async def _generate_optimization_recommendations(self, target_domain: str,
                                                   adaptation_speed: float,
                                                   adaptation_quality: float,
                                                   adaptation_pattern: str) -> List[str]:
        """生成优化建议"""
        
        recommendations = []
        
        # 基于适应速度的建议
        if adaptation_speed < 0.6:
            recommendations.extend([
                "增加适应任务的复杂度以提高适应速度",
                "使用渐进式适应策略",
                "优化初始参数设置"
            ])
        
        # 基于适应质量的建议
        if adaptation_quality < 0.7:
            recommendations.extend([
                "增强领域知识的前期准备",
                "使用质量导向的适应算法",
                "增加适应验证机制"
            ])
        
        # 基于适应模式的建议
        pattern_specific = {
            'immediate_adapter': [
                "利用快速适应优势，尝试更复杂的任务",
                "建立适应稳定性监控机制",
                "避免过度依赖初始适应速度"
            ],
            'gradual_adapter': [
                "采用长期适应策略规划",
                "设置阶段性适应目标",
                "增强适应过程的可视化监控"
            ],
            'oscillating_adapter': [
                "分析震荡原因并针对性改进",
                "增加适应过程中的稳定性机制",
                "考虑使用适应性学习率调整"
            ],
            'resilient_adapter': [
                "复制成功的恢复策略到其他场景",
                "研究鲁棒性机制并进一步增强",
                "作为其他智能体的参考模型"
            ]
        }
        
        if adaptation_pattern in pattern_specific:
            recommendations.extend(pattern_specific[adaptation_pattern])
        
        # 领域特定建议
        domain_recommendations = {
            'game': ["使用游戏特定的适应策略", "增加游戏经验迁移"],
            'physics': ["强化物理概念理解", "使用物理仿真辅助适应"],
            'social': ["增加社交场景模拟", "提高情境感知能力"],
            'language': ["扩大语言环境适应", "增强语义理解能力"],
            'spatial': ["加强空间认知训练", "使用空间推理辅助"]
        }
        
        if target_domain in domain_recommendations:
            recommendations.extend(domain_recommendations[target_domain])
        
        return recommendations
    
    async def _assess_adaptation_risks(self, target_domain: str,
                                     adaptation_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估适应风险"""
        
        risk_assessment = {
            'high_risks': [],
            'medium_risks': [],
            'low_risks': [],
            'risk_mitigation_strategies': [],
            'overall_risk_level': 'low'
        }
        
        # 分析高性能波动风险
        performance_evolution = adaptation_data['performance_evolution']
        volatility = np.std(performance_evolution)
        
        if volatility > 0.15:
            risk_assessment['high_risks'].append('性能波动过大')
            risk_assessment['risk_mitigation_strategies'].append('增加性能稳定性监控')
        
        # 分析错误累积风险
        error_rate = len(adaptation_data['error_log']) / max(len(performance_evolution), 1)
        
        if error_rate > 0.2:
            risk_assessment['high_risks'].append('错误率过高')
            risk_assessment['risk_mitigation_strategies'].append('实施错误预防机制')
        elif error_rate > 0.1:
            risk_assessment['medium_risks'].append('错误率偏高')
            risk_assessment['risk_mitigation_strategies'].append('增强错误处理能力')
        
        # 分析适应速度风险
        adaptation_speed = await self._calculate_adaptation_speed_simple(performance_evolution)
        if adaptation_speed < 0.4:
            risk_assessment['medium_risks'].append('适应速度过慢')
            risk_assessment['risk_mitigation_strategies'].append('优化适应算法参数')
        
        # 确定总体风险等级
        if len(risk_assessment['high_risks']) > 0:
            risk_assessment['overall_risk_level'] = 'high'
        elif len(risk_assessment['medium_risks']) > 0:
            risk_assessment['overall_risk_level'] = 'medium'
        else:
            risk_assessment['overall_risk_level'] = 'low'
        
        return risk_assessment
    
    def _calculate_adaptation_speed_simple(self, performance_evolution: List[float]) -> float:
        """简化的适应速度计算"""
        
        if len(performance_evolution) < 2:
            return 0.0
        
        # 计算前25%到后25%的性能提升
        quarter = len(performance_evolution) // 4
        initial_performance = np.mean(performance_evolution[:quarter])
        final_performance = np.mean(performance_evolution[-quarter:])
        
        improvement = final_performance - initial_performance
        
        return max(0.0, min(1.0, improvement))
    
    async def _predict_future_adaptation_potential(self, target_domain: str,
                                                 adaptation_data: Dict[str, Any]) -> float:
        """预测未来适应潜力"""
        
        # 基于当前适应表现预测未来适应能力
        final_performance = adaptation_data['final_performance']
        stability = self._calculate_adaptation_stability_simple(
            adaptation_data['performance_evolution']
        )
        
        # 鲁棒性预测
        robustness_estimate = self._estimate_robustness_from_performance(final_performance)
        
        # 计算适应潜力
        adaptation_potential = (
            final_performance * 0.4 +
            stability * 0.3 +
            robustness_estimate * 0.3
        )
        
        # 增长潜力评估
        if final_performance < 0.6:
            growth_potential = 0.8  # 有很大增长空间
        elif final_performance < 0.8:
            growth_potential = 0.5  # 有中等增长空间
        else:
            growth_potential = 0.2  # 增长空间有限
        
        # 综合适应潜力
        total_potential = adaptation_potential * 0.7 + growth_potential * 0.3
        
        return max(0.0, min(1.0, total_potential))
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取适应统计信息"""
        
        if not self.adaptation_sessions:
            return {
                'total_sessions': 0,
                'average_adaptation_speed': 0.0,
                'success_rate': 0.0,
                'common_patterns': []
            }
        
        # 计算总体统计
        all_sessions = list(self.adaptation_sessions.values())
        total_sessions = len(all_sessions)
        
        # 平均适应速度
        adaptation_speeds = [session.adaptation_speed for session in all_sessions]
        average_speed = np.mean(adaptation_speeds)
        
        # 成功率统计
        success_rates = [session.success_rate for session in all_sessions]
        overall_success_rate = np.mean(success_rates)
        
        # 常见适应模式统计
        pattern_counts = {}
        for session in all_sessions:
            pattern = self._identify_adaptation_pattern(session.performance_evolution)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 领域适应性能统计
        domain_performance = {}
        for domain, efficiencies in self.adaptation_performance_history.items():
            domain_performance[domain] = {
                'average_efficiency': np.mean(efficiencies),
                'session_count': len(efficiencies),
                'best_efficiency': max(efficiencies),
                'improvement_trend': self._calculate_improvement_trend(efficiencies)
            }
        
        return {
            'total_sessions': total_sessions,
            'average_adaptation_speed': average_speed,
            'overall_success_rate': overall_success_rate,
            'common_patterns': common_patterns,
            'domain_performance': domain_performance,
            'adaptation_quality_distribution': {
                'excellent': len([s for s in all_sessions if s.adaptation_quality >= 0.9]),
                'good': len([s for s in all_sessions if 0.7 <= s.adaptation_quality < 0.9]),
                'fair': len([s for s in all_sessions if 0.5 <= s.adaptation_quality < 0.7]),
                'poor': len([s for s in all_sessions if s.adaptation_quality < 0.5])
            }
        }
    
    def _calculate_improvement_trend(self, efficiencies: List[float]) -> float:
        """计算改进趋势"""
        if len(efficiencies) < 2:
            return 0.0
        
        recent_efficiency = np.mean(efficiencies[-3:])  # 最近3次平均
        early_efficiency = np.mean(efficiencies[:3])    # 前3次平均
        
        return recent_efficiency - early_efficiency
    
    def _identify_adaptation_pattern_simple(self, performance_evolution: List[float]) -> str:
        """简化的适应模式识别"""
        
        if len(performance_evolution) < 5:
            return 'unknown'
        
        # 简化的模式识别逻辑
        initial_performance = performance_evolution[0]
        final_performance = performance_evolution[-1]
        improvement = final_performance - initial_performance
        
        # 根据改进程度和最终性能简单分类
        if improvement > 0.4 and final_performance > 0.8:
            return 'high_performer'
        elif improvement > 0.2:
            return 'steady_improver'
        elif improvement < 0.1:
            return 'slow_adaptor'
        else:
            return 'moderate_adaptor'
    
    def _identify_adaptation_pattern(self, performance_evolution: List[float]) -> str:
        """适应模式识别（完整版本）"""
        return self._identify_adaptation_pattern_simple(performance_evolution)


def create_adaptation_metrics(config: Optional[Dict[str, Any]] = None) -> AdaptationMetrics:
    """创建适应指标计算器实例的便捷函数"""
    return AdaptationMetrics(config or {})


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建适应指标计算器
        calculator = create_adaptation_metrics({
            'speed_threshold': 0.1,
            'quality_threshold': 0.8,
            'stability_window': 5
        })
        
        # 评估适应速度
        result = await calculator.evaluate_adaptation_speed(
            target_domain='game',
            transferred_knowledge={'concepts': ['strategy', 'tactics'], 'rules': []},
            adaptation_tasks={'tasks': ['task1', 'task2']}
        )
        
        print(f"适应速度评估完成: {result['adaptation_report'].overall_adaptation_score:.3f}")
    
    # 运行示例
    # asyncio.run(main())