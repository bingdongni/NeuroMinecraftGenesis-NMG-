"""
自适应难度调节系统

该模块实现智能的自适应难度调节功能，包括：
1. 多种难度调节策略
2. 性能驱动的难度调整
3. 个性化适应算法
4. 多智能体难度平衡
5. 难度梯度控制
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifficultyStrategy(Enum):
    """难度调节策略枚举"""
    PERFORMANCE_BASED = "performance_based"  # 基于表现
    CAPABILITY_BASED = "capability_based"    # 基于能力
    CHALLENGE_BASED = "challenge_based"      # 基于挑战
    GRADUAL_INCREASE = "gradual_increase"    # 逐步增加
    ADAPTIVE_BALANCED = "adaptive_balanced"  # 自适应平衡
    MULTI_OBJECTIVE = "multi_objective"      # 多目标优化


@dataclass
class DifficultyParameters:
    """难度参数"""
    # 基础难度参数
    base_complexity: float = 0.3
    terrain_variance: float = 0.5
    resource_density: float = 1.0
    danger_multiplier: float = 1.0
    
    # 动态参数
    temporal_changes: float = 0.1
    spatial_constraints: float = 0.5
    survival_challenge: float = 0.3
    
    # 智能体特定参数
    capability_floor: float = 0.2
    capability_ceiling: float = 0.9
    adaptation_sensitivity: float = 1.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'base_complexity': self.base_complexity,
            'terrain_variance': self.terrain_variance,
            'resource_density': self.resource_density,
            'danger_multiplier': self.danger_multiplier,
            'temporal_changes': self.temporal_changes,
            'spatial_constraints': self.spatial_constraints,
            'survival_challenge': self.survival_challenge,
            'capability_floor': self.capability_floor,
            'capability_ceiling': self.capability_ceiling,
            'adaptation_sensitivity': self.adaptation_sensitivity
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    agent_id: str
    timestamp: float
    success_rate: float = 0.0
    task_completion_time: float = 0.0
    resource_efficiency: float = 0.0
    survival_score: float = 0.0
    learning_rate: float = 0.0
    challenge_level: float = 0.0
    stress_level: float = 0.0
    
    def overall_score(self) -> float:
        """计算综合性能分数"""
        weights = {
            'success_rate': 0.25,
            'task_completion_time': 0.15,
            'resource_efficiency': 0.15,
            'survival_score': 0.25,
            'learning_rate': 0.15,
            'stress_penalty': -0.05
        }
        
        score = (weights['success_rate'] * self.success_rate +
                weights['task_completion_time'] * (1.0 - min(1.0, self.task_completion_time / 100.0)) +
                weights['resource_efficiency'] * self.resource_efficiency +
                weights['survival_score'] * self.survival_score +
                weights['learning_rate'] * self.learning_rate +
                weights['stress_penalty'] * (1.0 - self.stress_level))
        
        return max(0.0, min(1.0, score))
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'success_rate': self.success_rate,
            'task_completion_time': self.task_completion_time,
            'resource_efficiency': self.resource_efficiency,
            'survival_score': self.survival_score,
            'learning_rate': self.learning_rate,
            'challenge_level': self.challenge_level,
            'stress_level': self.stress_level,
            'overall_score': self.overall_score()
        }


@dataclass
class DifficultyAdjustment:
    """难度调整记录"""
    timestamp: float
    strategy: DifficultyStrategy
    previous_difficulty: float
    new_difficulty: float
    adjustment_reason: str
    performance_impact: Optional[float] = None
    agent_feedback: Optional[Dict] = None
    implementation_time: float = 0.0


class AdaptiveDifficultyEngine:
    """自适应难度引擎"""
    
    def __init__(self, 
                 initial_difficulty: float = 0.3,
                 adaptation_strategies: List[DifficultyStrategy] = None,
                 learning_window: int = 50,
                 min_adjustment_interval: float = 30.0):
        """
        初始化自适应难度引擎
        
        Args:
            initial_difficulty: 初始难度
            adaptation_strategies: 适应策略列表
            learning_window: 学习窗口大小
            min_adjustment_interval: 最小调整间隔（秒）
        """
        self.current_difficulty = initial_difficulty
        self.strategies = adaptation_strategies or [DifficultyStrategy.ADAPTIVE_BALANCED]
        self.learning_window = learning_window
        self.min_adjustment_interval = min_adjustment_interval
        
        # 性能历史数据
        self.performance_history = defaultdict(deque)
        self.difficulty_history = []
        
        # 调整历史
        self.adjustment_history = []
        self.last_adjustment_time = 0
        
        # 并发控制
        self._lock = threading.RLock()
        
        # 参数
        self.parameters = DifficultyParameters()
        
        # 统计信息
        self.statistics = {
            'total_adjustments': 0,
            'strategy_usage': defaultdict(int),
            'avg_adjustment_size': 0.0,
            'performance_correlation': 0.0
        }
        
        logger.info(f"初始化自适应难度引擎 - 策略: {[s.value for s in self.strategies]}")
    
    def evaluate_and_adjust(self, 
                          performance_data: Dict[str, PerformanceMetrics],
                          environment_state: Dict,
                          force_adjustment: bool = False) -> Dict:
        """
        评估并调整难度
        
        Args:
            performance_data: 性能数据字典
            environment_state: 环境状态
            force_adjustment: 是否强制调整
            
        Returns:
            Dict: 调整结果
        """
        with self._lock:
            try:
                current_time = time.time()
                
                # 检查是否可以进行调整
                if not force_adjustment:
                    if current_time - self.last_adjustment_time < self.min_adjustment_interval:
                        return {
                            'adjustment_made': False,
                            'reason': 'adjustment_interval_restriction',
                            'current_difficulty': self.current_difficulty
                        }
                
                # 分析当前表现
                performance_analysis = self._analyze_performance(performance_data)
                
                # 选择最佳策略
                selected_strategy = self._select_strategy(performance_analysis, environment_state)
                
                # 计算新的难度
                new_difficulty = self._calculate_target_difficulty(
                    performance_analysis, environment_state, selected_strategy
                )
                
                # 验证调整合理性
                adjustment_valid, validation_reason = self._validate_adjustment(
                    self.current_difficulty, new_difficulty, performance_analysis
                )
                
                if not adjustment_valid:
                    logger.info(f"调整被阻止: {validation_reason}")
                    return {
                        'adjustment_made': False,
                        'reason': validation_reason,
                        'current_difficulty': self.current_difficulty,
                        'proposed_difficulty': new_difficulty
                    }
                
                # 执行调整
                old_difficulty = self.current_difficulty
                self.current_difficulty = new_difficulty
                self.last_adjustment_time = current_time
                
                # 记录调整历史
                adjustment = DifficultyAdjustment(
                    timestamp=current_time,
                    strategy=selected_strategy,
                    previous_difficulty=old_difficulty,
                    new_difficulty=new_difficulty,
                    adjustment_reason=validation_reason
                )
                
                self._record_adjustment(adjustment, performance_analysis)
                
                # 更新统计信息
                self._update_statistics(adjustment, performance_analysis)
                
                logger.info(f"难度调整: {old_difficulty:.3f} -> {new_difficulty:.3f} "
                          f"(策略: {selected_strategy.value}, 原因: {validation_reason})")
                
                return {
                    'adjustment_made': True,
                    'strategy': selected_strategy.value,
                    'previous_difficulty': old_difficulty,
                    'new_difficulty': new_difficulty,
                    'adjustment_amount': new_difficulty - old_difficulty,
                    'reason': validation_reason,
                    'performance_analysis': performance_analysis,
                    'adjustment_time': current_time
                }
                
            except Exception as e:
                logger.error(f"难度调整失败: {str(e)}")
                raise
    
    def _analyze_performance(self, 
                           performance_data: Dict[str, PerformanceMetrics]) -> Dict:
        """分析智能体表现"""
        if not performance_data:
            return {'status': 'no_data', 'average_score': 0.0}
        
        # 计算各项指标的平均值
        metrics = list(performance_data.values())
        
        avg_success_rate = np.mean([p.success_rate for p in metrics])
        avg_completion_time = np.mean([p.task_completion_time for p in metrics])
        avg_efficiency = np.mean([p.resource_efficiency for p in metrics])
        avg_survival = np.mean([p.survival_score for p in metrics])
        avg_learning = np.mean([p.learning_rate for p in metrics])
        avg_stress = np.mean([p.stress_level for p in metrics])
        avg_overall = np.mean([p.overall_score() for p in metrics])
        
        # 计算表现趋势
        trend_analysis = self._calculate_performance_trend(performance_data)
        
        # 计算能力分布
        capability_variance = np.var([p.overall_score() for p in metrics])
        
        # 识别表现问题
        issues = self._identify_performance_issues(performance_data)
        
        return {
            'status': 'analyzed',
            'agent_count': len(performance_data),
            'average_metrics': {
                'success_rate': avg_success_rate,
                'task_completion_time': avg_completion_time,
                'resource_efficiency': avg_efficiency,
                'survival_score': avg_survival,
                'learning_rate': avg_learning,
                'stress_level': avg_stress,
                'overall_score': avg_overall
            },
            'trend_analysis': trend_analysis,
            'capability_variance': capability_variance,
            'performance_issues': issues,
            'requires_adjustment': self._determine_adjustment_need(
                avg_success_rate, avg_overall, avg_stress, capability_variance
            )
        }
    
    def _calculate_performance_trend(self, 
                                   performance_data: Dict[str, PerformanceMetrics]) -> Dict:
        """计算表现趋势"""
        if len(performance_data) < 2:
            return {'trend': 'stable', 'confidence': 0.0}
        
        # 收集历史数据（这里简化处理，实际需要从performance_history中获取）
        current_scores = [p.overall_score() for p in performance_data.values()]
        avg_current = np.mean(current_scores)
        
        # 模拟趋势计算（需要实际的历史数据）
        # 这里使用简单的随机模拟，实际应用中应该使用真实历史数据
        historical_avg = 0.5  # 假设的历史平均值
        
        if avg_current > historical_avg + 0.05:
            trend = 'improving'
            confidence = min(1.0, (avg_current - historical_avg) * 2)
        elif avg_current < historical_avg - 0.05:
            trend = 'declining'
            confidence = min(1.0, (historical_avg - avg_current) * 2)
        else:
            trend = 'stable'
            confidence = 0.3
        
        return {
            'trend': trend,
            'confidence': confidence,
            'current_average': avg_current,
            'baseline': historical_avg
        }
    
    def _identify_performance_issues(self, 
                                   performance_data: Dict[str, PerformanceMetrics]) -> List[str]:
        """识别性能问题"""
        issues = []
        
        metrics = list(performance_data.values())
        
        # 检查成功率
        avg_success = np.mean([p.success_rate for p in metrics])
        if avg_success < 0.3:
            issues.append('low_success_rate')
        elif avg_success > 0.9:
            issues.append('high_success_rate')
        
        # 检查学习率
        avg_learning = np.mean([p.learning_rate for p in metrics])
        if avg_learning < 0.1:
            issues.append('low_learning_rate')
        
        # 检查压力水平
        avg_stress = np.mean([p.stress_level for p in metrics])
        if avg_stress > 0.7:
            issues.append('high_stress_level')
        
        # 检查完成时间
        avg_time = np.mean([p.task_completion_time for p in metrics])
        if avg_time > 60.0:
            issues.append('slow_completion')
        
        # 检查能力差异
        scores = [p.overall_score() for p in metrics]
        score_variance = np.var(scores)
        if score_variance > 0.1:
            issues.append('high_capability_variance')
        
        return issues
    
    def _determine_adjustment_need(self, 
                                 success_rate: float, 
                                 overall_score: float,
                                 stress_level: float,
                                 capability_variance: float) -> bool:
        """确定是否需要调整"""
        # 基于多个指标判断是否需要调整
        need_adjustment = False
        reasons = []
        
        # 成功率过低或过高
        if success_rate < 0.3 or success_rate > 0.9:
            need_adjustment = True
            reasons.append('success_rate_extreme')
        
        # 综合表现不佳
        if overall_score < 0.4:
            need_adjustment = True
            reasons.append('poor_overall_performance')
        
        # 压力过大
        if stress_level > 0.8:
            need_adjustment = True
            reasons.append('high_stress')
        
        # 能力差异过大
        if capability_variance > 0.15:
            need_adjustment = True
            reasons.append('high_capability_variance')
        
        return need_adjustment
    
    def _select_strategy(self, 
                       performance_analysis: Dict, 
                       environment_state: Dict) -> DifficultyStrategy:
        """选择适应策略"""
        issues = performance_analysis.get('performance_issues', [])
        trend = performance_analysis.get('trend_analysis', {}).get('trend', 'stable')
        
        # 根据表现问题选择策略
        if 'low_success_rate' in issues:
            return DifficultyStrategy.PERFORMANCE_BASED
        elif 'high_capability_variance' in issues:
            return DifficultyStrategy.CAPABILITY_BASED
        elif 'high_stress' in issues:
            return DifficultyStrategy.CHALLENGE_BASED
        elif trend == 'improving':
            return DifficultyStrategy.GRADUAL_INCREASE
        elif len(issues) > 2:
            return DifficultyStrategy.MULTI_OBJECTIVE
        else:
            return DifficultyStrategy.ADAPTIVE_BALANCED
    
    def _calculate_target_difficulty(self, 
                                   performance_analysis: Dict,
                                   environment_state: Dict,
                                   strategy: DifficultyStrategy) -> float:
        """计算目标难度"""
        current_difficulty = self.current_difficulty
        metrics = performance_analysis.get('average_metrics', {})
        
        if strategy == DifficultyStrategy.PERFORMANCE_BASED:
            return self._calculate_performance_based_difficulty(metrics, current_difficulty)
        elif strategy == DifficultyStrategy.CAPABILITY_BASED:
            return self._calculate_capability_based_difficulty(metrics, current_difficulty)
        elif strategy == DifficultyStrategy.CHALLENGE_BASED:
            return self._calculate_challenge_based_difficulty(metrics, current_difficulty)
        elif strategy == DifficultyStrategy.GRADUAL_INCREASE:
            return self._calculate_gradual_increase_difficulty(metrics, current_difficulty)
        elif strategy == DifficultyStrategy.MULTI_OBJECTIVE:
            return self._calculate_multi_objective_difficulty(metrics, current_difficulty)
        else:  # ADAPTIVE_BALANCED
            return self._calculate_balanced_difficulty(metrics, current_difficulty)
    
    def _calculate_performance_based_difficulty(self, 
                                              metrics: Dict, 
                                              current_difficulty: float) -> float:
        """基于表现的难度计算"""
        success_rate = metrics.get('success_rate', 0.5)
        overall_score = metrics.get('overall_score', 0.5)
        
        # 目标成功率区间：[0.4, 0.8]
        target_success_rate = 0.6
        
        # 计算成功率差距
        success_gap = target_success_rate - success_rate
        
        # 基于成功率差距调整难度
        # 成功率过低 -> 降低难度
        # 成功率过高 -> 增加难度
        difficulty_adjustment = -success_gap * 0.5
        
        # 考虑整体表现
        score_factor = (overall_score - 0.5) * 0.3
        total_adjustment = difficulty_adjustment + score_factor
        
        new_difficulty = current_difficulty + total_adjustment
        
        return max(0.1, min(1.0, new_difficulty))
    
    def _calculate_capability_based_difficulty(self, 
                                             metrics: Dict, 
                                             current_difficulty: float) -> float:
        """基于能力的难度计算"""
        learning_rate = metrics.get('learning_rate', 0.5)
        survival_score = metrics.get('survival_score', 0.5)
        
        # 高学习率和高生存能力 -> 可以承受更高难度
        capability_score = (learning_rate + survival_score) / 2
        
        # 能力上限和下限
        capability_ceiling = self.parameters.capability_ceiling
        capability_floor = self.parameters.capability_floor
        
        # 计算能力利用率
        if capability_score > capability_ceiling:
            # 能力未被充分利用，可以增加难度
            utilization_factor = 0.8  # 只利用80%的能力
            difficulty_increase = (capability_score - capability_ceiling) * 0.2
        elif capability_score < capability_floor:
            # 能力不足，需要降低难度
            utilization_factor = 0.5
            difficulty_decrease = (capability_floor - capability_score) * 0.3
            difficulty_increase = -difficulty_decrease
        else:
            # 能力适中，正常调整
            utilization_factor = 0.7
            difficulty_increase = 0
        
        # 敏感性调整
        sensitivity = self.parameters.adaptation_sensitivity
        adjustment = difficulty_increase * sensitivity
        
        new_difficulty = current_difficulty + adjustment
        
        return max(0.1, min(1.0, new_difficulty))
    
    def _calculate_challenge_based_difficulty(self, 
                                            metrics: Dict, 
                                            current_difficulty: float) -> float:
        """基于挑战的难度计算"""
        stress_level = metrics.get('stress_level', 0.3)
        survival_score = metrics.get('survival_score', 0.5)
        
        # 目标压力水平：[0.2, 0.6]
        target_stress = 0.4
        
        # 计算压力差距
        stress_gap = target_stress - stress_level
        
        # 压力过高 -> 降低挑战性
        # 压力过低 -> 增加挑战性
        challenge_adjustment = stress_gap * 0.4
        
        # 考虑生存表现
        survival_factor = (survival_score - 0.5) * 0.2
        
        total_adjustment = challenge_adjustment + survival_factor
        new_difficulty = current_difficulty + total_adjustment
        
        return max(0.1, min(1.0, new_difficulty))
    
    def _calculate_gradual_increase_difficulty(self, 
                                             metrics: Dict, 
                                             current_difficulty: float) -> float:
        """逐步增加难度"""
        success_rate = metrics.get('success_rate', 0.5)
        
        # 只有在表现良好时才逐步增加
        if success_rate > 0.7:
            # 每次增加不超过当前难度的10%
            max_increase = current_difficulty * 0.1
            gradual_increase = 0.02  # 固定增加0.02
            adjustment = min(max_increase, gradual_increase)
        else:
            # 表现不佳时适度降低
            adjustment = -0.01
        
        new_difficulty = current_difficulty + adjustment
        return max(0.1, min(1.0, new_difficulty))
    
    def _calculate_multi_objective_difficulty(self, 
                                            metrics: Dict, 
                                            current_difficulty: float) -> float:
        """多目标优化难度"""
        success_rate = metrics.get('success_rate', 0.5)
        stress_level = metrics.get('stress_level', 0.3)
        learning_rate = metrics.get('learning_rate', 0.5)
        survival_score = metrics.get('survival_score', 0.5)
        
        # 多个目标权重
        weights = {
            'success_rate': 0.3,
            'stress_level': 0.25,
            'learning_rate': 0.25,
            'survival_score': 0.2
        }
        
        # 目标值
        targets = {
            'success_rate': 0.6,
            'stress_level': 0.4,
            'learning_rate': 0.6,
            'survival_score': 0.7
        }
        
        # 计算每个目标的差距
        gaps = {
            'success_rate': targets['success_rate'] - success_rate,
            'stress_level': targets['stress_level'] - stress_level,
            'learning_rate': targets['learning_rate'] - learning_rate,
            'survival_score': targets['survival_score'] - survival_score
        }
        
        # 加权调整
        total_adjustment = sum(weights[obj] * gaps[obj] for obj in weights)
        
        # 限制调整幅度
        max_adjustment = 0.15
        adjusted_change = max(-max_adjustment, min(max_adjustment, total_adjustment))
        
        new_difficulty = current_difficulty + adjusted_change
        return max(0.1, min(1.0, new_difficulty))
    
    def _calculate_balanced_difficulty(self, 
                                     metrics: Dict, 
                                     current_difficulty: float) -> float:
        """平衡自适应难度"""
        # 综合考虑多个因素
        success_rate = metrics.get('success_rate', 0.5)
        overall_score = metrics.get('overall_score', 0.5)
        stress_level = metrics.get('stress_level', 0.3)
        
        # 目标综合分数
        target_score = 0.6
        score_gap = target_score - overall_score
        
        # 适应强度（越接近目标，变化越小）
        adaptation_intensity = 1.0 - abs(score_gap) * 2
        
        # 基础调整
        base_adjustment = score_gap * 0.3 * adaptation_intensity
        
        # 压力平衡
        if stress_level > 0.7:
            base_adjustment -= 0.05  # 减压
        elif stress_level < 0.2:
            base_adjustment += 0.03  # 增加挑战
        
        new_difficulty = current_difficulty + base_adjustment
        return max(0.1, min(1.0, new_difficulty))
    
    def _validate_adjustment(self, 
                           old_difficulty: float, 
                           new_difficulty: float,
                           performance_analysis: Dict) -> Tuple[bool, str]:
        """验证调整合理性"""
        change_magnitude = abs(new_difficulty - old_difficulty)
        
        # 检查变化幅度（一次调整不超过30%）
        if change_magnitude > 0.3:
            return False, 'excessive_change_magnitude'
        
        # 检查是否跳过了最优区间 [0.3, 0.8]
        if old_difficulty < 0.3 and new_difficulty > 0.8:
            return False, 'skipping_optimal_range'
        if old_difficulty > 0.8 and new_difficulty < 0.3:
            return False, 'skipping_optimal_range'
        
        # 检查是否需要调整
        requires_adjustment = performance_analysis.get('requires_adjustment', False)
        if not requires_adjustment and change_magnitude > 0.05:
            return False, 'unnecessary_large_adjustment'
        
        # 检查智能体数量适应性
        agent_count = performance_analysis.get('agent_count', 1)
        if agent_count > 5 and change_magnitude > 0.2:
            return False, 'too_dramatic_for_group'
        
        return True, 'validation_passed'
    
    def _record_adjustment(self, 
                          adjustment: DifficultyAdjustment, 
                          performance_analysis: Dict):
        """记录调整历史"""
        # 更新智能体表现历史
        self.performance_history['adjustments'].append(adjustment)
        
        # 保持历史记录在合理范围内
        max_history = 100
        if len(self.performance_history['adjustments']) > max_history:
            self.performance_history['adjustments'] = \
                self.performance_history['adjustments'][-max_history:]
        
        self.adjustment_history.append(adjustment)
        if len(self.adjustment_history) > 50:
            self.adjustment_history = self.adjustment_history[-50:]
        
        # 更新难度历史
        self.difficulty_history.append({
            'timestamp': adjustment.timestamp,
            'difficulty': adjustment.new_difficulty,
            'strategy': adjustment.strategy.value
        })
    
    def _update_statistics(self, 
                          adjustment: DifficultyAdjustment,
                          performance_analysis: Dict):
        """更新统计信息"""
        self.statistics['total_adjustments'] += 1
        self.statistics['strategy_usage'][adjustment.strategy.value] += 1
        
        # 更新平均调整大小
        current_avg = self.statistics['avg_adjustment_size']
        total_count = self.statistics['total_adjustments']
        change_magnitude = abs(adjustment.new_difficulty - adjustment.previous_difficulty)
        
        self.statistics['avg_adjustment_size'] = \
            (current_avg * (total_count - 1) + change_magnitude) / total_count
    
    def get_difficulty_recommendations(self) -> Dict:
        """获取难度建议"""
        if len(self.adjustment_history) < 5:
            return {
                'status': 'insufficient_data',
                'message': '调整历史不足，无法提供建议',
                'current_difficulty': self.current_difficulty
            }
        
        recent_adjustments = self.adjustment_history[-10:]
        
        # 分析调整模式
        strategies_used = [adj.strategy.value for adj in recent_adjustments]
        strategy_frequency = {strategy: strategies_used.count(strategy) 
                            for strategy in set(strategies_used)}
        
        # 计算调整趋势
        difficulties = [adj.new_difficulty for adj in recent_adjustments]
        if len(difficulties) >= 2:
            if difficulties[-1] > difficulties[0]:
                trend = 'increasing'
            elif difficulties[-1] < difficulties[0]:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # 生成建议
        recommendations = []
        
        if trend == 'increasing' and self.current_difficulty > 0.7:
            recommendations.append('当前难度较高，考虑在下次调整时适度降低')
        elif trend == 'decreasing' and self.current_difficulty < 0.3:
            recommendations.append('当前难度较低，可以考虑逐步增加挑战性')
        
        most_used_strategy = max(strategy_frequency, key=strategy_frequency.get)
        if strategy_frequency[most_used_strategy] > len(recent_adjustments) * 0.7:
            recommendations.append(f'过度依赖策略 {most_used_strategy}，建议平衡使用多种策略')
        
        if self.statistics['avg_adjustment_size'] > 0.15:
            recommendations.append('调整幅度较大，建议采用更渐进的方式')
        
        return {
            'current_difficulty': self.current_difficulty,
            'trend': trend,
            'strategy_distribution': strategy_frequency,
            'most_used_strategy': most_used_strategy,
            'avg_adjustment_size': self.statistics['avg_adjustment_size'],
            'total_adjustments': self.statistics['total_adjustments'],
            'recommendations': recommendations,
            'parameters': self.parameters.to_dict()
        }
    
    def export_adjustment_history(self, filepath: str):
        """导出调整历史"""
        export_data = {
            'adjustment_history': [
                {
                    'timestamp': adj.timestamp,
                    'strategy': adj.strategy.value,
                    'previous_difficulty': adj.previous_difficulty,
                    'new_difficulty': adj.new_difficulty,
                    'adjustment_amount': adj.new_difficulty - adj.previous_difficulty,
                    'reason': adj.adjustment_reason
                }
                for adj in self.adjustment_history
            ],
            'difficulty_history': self.difficulty_history,
            'statistics': self.statistics,
            'current_parameters': self.parameters.to_dict(),
            'strategies': [s.value for s in self.strategies]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"调整历史已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出调整历史失败: {str(e)}")
            raise


# 工厂函数
def create_adaptive_difficulty_engine(config: Dict) -> AdaptiveDifficultyEngine:
    """
    创建自适应难度引擎的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        AdaptiveDifficultyEngine: 自适应难度引擎实例
    """
    strategies = []
    if 'strategies' in config:
        for strategy_name in config['strategies']:
            try:
                strategies.append(DifficultyStrategy(strategy_name))
            except ValueError:
                logger.warning(f"未知的策略: {strategy_name}")
    
    return AdaptiveDifficultyEngine(
        initial_difficulty=config.get('initial_difficulty', 0.3),
        adaptation_strategies=strategies or [DifficultyStrategy.ADAPTIVE_BALANCED],
        learning_window=config.get('learning_window', 50),
        min_adjustment_interval=config.get('min_adjustment_interval', 30.0)
    )


if __name__ == "__main__":
    # 演示用法
    logger.info("自适应难度引擎演示")
    
    # 创建引擎
    engine = AdaptiveDifficultyEngine()
    
    # 模拟智能体性能数据
    performance_data = {
        'agent_1': PerformanceMetrics(
            agent_id='agent_1',
            timestamp=time.time(),
            success_rate=0.7,
            task_completion_time=30.0,
            resource_efficiency=0.6,
            survival_score=0.8,
            learning_rate=0.4,
            challenge_level=0.5,
            stress_level=0.3
        ),
        'agent_2': PerformanceMetrics(
            agent_id='agent_2',
            timestamp=time.time(),
            success_rate=0.6,
            task_completion_time=45.0,
            resource_efficiency=0.5,
            survival_score=0.7,
            learning_rate=0.3,
            challenge_level=0.5,
            stress_level=0.4
        )
    }
    
    # 模拟环境状态
    environment_state = {
        'terrain_complexity': 0.6,
        'resource_availability': 0.7,
        'danger_level': 0.4
    }
    
    # 执行难度调整
    result = engine.evaluate_and_adjust(performance_data, environment_state)
    print("难度调整结果:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取建议
    recommendations = engine.get_difficulty_recommendations()
    print("难度建议:", json.dumps(recommendations, indent=2, ensure_ascii=False))