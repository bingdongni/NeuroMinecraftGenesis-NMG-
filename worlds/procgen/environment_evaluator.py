"""
环境评估器和监控系统

该模块提供全面的环境评估和监控功能，包括：
1. 环境复杂度实时监控
2. 智能体适应性能评估
3. 环境变化趋势分析
4. 进化效果评估
5. 反馈机制和优化建议
6. 多维度性能指标计算
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """评估类型枚举"""
    REAL_TIME = "real_time"        # 实时评估
    PERIODIC = "periodic"          # 定期评估
    TRIGGERED = "triggered"        # 触发式评估
    COMPREHENSIVE = "comprehensive" # 全面评估
    PERFORMANCE_BASED = "performance_based" # 基于性能的评估


class MetricCategory(Enum):
    """指标类别枚举"""
    SPATIAL = "spatial"            # 空间指标
    TEMPORAL = "temporal"          # 时间指标
    RESOURCES = "resources"        # 资源指标
    DANGERS = "dangers"           # 危险指标
    ADAPTABILITY = "adaptability"  # 适应性指标
    PERFORMANCE = "performance"    # 性能指标


@dataclass
class EnvironmentSnapshot:
    """环境快照"""
    timestamp: float
    world_state: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, float]
    system_load: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """评估结果"""
    evaluation_id: str
    evaluation_type: EvaluationType
    timestamp: float
    duration: float
    
    # 整体评分
    overall_score: float
    
    # 分类评分
    category_scores: Dict[MetricCategory, float]
    
    # 具体指标
    metrics: Dict[str, float]
    
    # 分析结果
    analysis: Dict[str, Any]
    
    # 建议
    recommendations: List[str]
    
    # 风险评估
    risk_assessment: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'evaluation_id': self.evaluation_id,
            'evaluation_type': self.evaluation_type.value,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'overall_score': self.overall_score,
            'category_scores': {k.value: v for k, v in self.category_scores.items()},
            'metrics': self.metrics,
            'analysis': self.analysis,
            'recommendations': self.recommendations,
            'risk_assessment': self.risk_assessment
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化性能监控器
        
        Args:
            window_size: 监控窗口大小
        """
        self.window_size = window_size
        self.metrics_history = defaultdict(deque)
        self.performance_thresholds = {
            'success_rate': {'min': 0.3, 'max': 0.9},
            'learning_rate': {'min': 0.1, 'max': 1.0},
            'stress_level': {'min': 0.0, 'max': 0.7},
            'survival_score': {'min': 0.2, 'max': 1.0},
            'efficiency': {'min': 0.2, 'max': 1.0}
        }
    
    def record_metric(self, metric_name: str, value: float, timestamp: float = None):
        """记录指标"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history[metric_name].append((timestamp, value))
        
        # 保持窗口大小
        if len(self.metrics_history[metric_name]) > self.window_size:
            self.metrics_history[metric_name].popleft()
    
    def get_metric_trend(self, metric_name: str, time_window: float = 300.0) -> Dict[str, float]:
        """获取指标趋势"""
        if metric_name not in self.metrics_history:
            return {'trend': 'no_data', 'slope': 0.0, 'stability': 0.0}
        
        current_time = time.time()
        window_start = current_time - time_window
        
        # 过滤时间窗口内的数据
        recent_data = [(ts, val) for ts, val in self.metrics_history[metric_name] 
                      if ts >= window_start]
        
        if len(recent_data) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'stability': 0.0}
        
        # 计算趋势
        timestamps = [ts - recent_data[0][0] for ts, _ in recent_data]  # 相对时间
        values = [val for _, val in recent_data]
        
        # 线性回归计算斜率
        if len(values) >= 2:
            slope = np.polyfit(timestamps, values, 1)[0]
        else:
            slope = 0.0
        
        # 计算稳定性（标准差的倒数）
        if len(values) > 1:
            stability = 1.0 / (1.0 + np.std(values))
        else:
            stability = 0.0
        
        # 判断趋势
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'stability': stability,
            'current_value': values[-1],
            'average_value': np.mean(values),
            'data_points': len(recent_data)
        }
    
    def assess_performance_health(self) -> Dict[str, Any]:
        """评估性能健康度"""
        health_scores = {}
        issues = []
        
        for metric_name, threshold in self.performance_thresholds.items():
            if metric_name in self.metrics_history:
                recent_data = list(self.metrics_history[metric_name])
                if recent_data:
                    current_value = recent_data[-1][1]
                    
                    # 计算健康分数
                    if threshold['min'] <= current_value <= threshold['max']:
                        # 在正常范围内，计算偏离中心的程度
                        center = (threshold['min'] + threshold['max']) / 2
                        deviation = abs(current_value - center) / (threshold['max'] - threshold['min'])
                        health_score = 1.0 - deviation * 0.5  # 最大扣分50%
                    else:
                        # 超出范围，严重扣分
                        if current_value < threshold['min']:
                            health_score = max(0.0, current_value / threshold['min'])
                        else:  # current_value > threshold['max']
                            health_score = max(0.0, 1.0 - (current_value - threshold['max']) / (1.0 - threshold['max']))
                        
                        issues.append(f'{metric_name}超出正常范围')
                    
                    health_scores[metric_name] = health_score
                    
                    # 添加趋势信息
                    trend_info = self.get_metric_trend(metric_name)
                    health_scores[f'{metric_name}_trend'] = trend_info['trend']
        
        overall_health = np.mean(list(health_scores.values())) if health_scores else 0.0
        
        return {
            'overall_health': overall_health,
            'metric_scores': health_scores,
            'issues': issues,
            'healthy_metrics': sum(1 for score in health_scores.values() 
                                 if score >= 0.7) / len(health_scores) if health_scores else 0.0
        }


class EnvironmentEvaluator:
    """环境评估器"""
    
    def __init__(self, 
                 evaluation_interval: float = 30.0,
                 performance_window: int = 50,
                 enable_async: bool = True):
        """
        初始化环境评估器
        
        Args:
            evaluation_interval: 评估间隔（秒）
            performance_window: 性能窗口大小
            enable_async: 是否启用异步评估
        """
        self.evaluation_interval = evaluation_interval
        self.enable_async = enable_async
        
        # 评估历史和快照
        self.evaluation_history = deque(maxlen=1000)
        self.snapshot_history = deque(maxlen=500)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor(window_size=performance_window)
        
        # 评估器
        self.spatial_evaluator = self._init_spatial_evaluator()
        self.temporal_evaluator = self._init_temporal_evaluator()
        self.resource_evaluator = self._init_resource_evaluator()
        self.danger_evaluator = self._init_danger_evaluator()
        self.adaptability_evaluator = self._init_adaptability_evaluator()
        
        # 线程控制
        self._lock = threading.RLock()
        self._evaluation_thread = None
        self._stop_evaluation = False
        
        # 评估统计
        self.evaluation_stats = {
            'total_evaluations': 0,
            'evaluation_types_used': defaultdict(int),
            'avg_evaluation_time': 0.0,
            'last_evaluation': None
        }
        
        logger.info(f"环境评估器初始化完成 - 评估间隔: {evaluation_interval}秒")
    
    def _init_spatial_evaluator(self) -> Callable:
        """初始化空间评估器"""
        def evaluate_spatial(world_state: Dict, agent_states: Dict) -> Dict[str, float]:
            """评估空间相关指标"""
            metrics = {}
            
            # 地形复杂度
            terrain = world_state.get('terrain', {})
            height_variance = terrain.get('height_variance', 0.5)
            cave_density = terrain.get('cave_density', 0.3)
            connectivity = terrain.get('connectivity', 0.8)
            
            metrics['terrain_complexity'] = (0.4 * height_variance + 
                                           0.3 * min(0.8, cave_density + 0.3) + 
                                           0.3 * (1 - connectivity))
            
            # 空间利用率
            total_area = world_state.get('total_area', 1000)
            accessible_area = world_state.get('accessible_area', 800)
            metrics['spatial_utilization'] = accessible_area / total_area if total_area > 0 else 0.8
            
            # 路径复杂度
            path_metrics = world_state.get('paths', {})
            avg_path_length = path_metrics.get('average_length', 10.0)
            path_difficulty = path_metrics.get('difficulty', 0.5)
            metrics['path_complexity'] = min(1.0, avg_path_length / 50.0 * 0.6 + path_difficulty * 0.4)
            
            # 智能体分布密度
            if agent_states:
                positions = [state.get('position', [0, 0, 0]) for state in agent_states.values()]
                if len(positions) > 1:
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                            distances.append(dist)
                    avg_distance = np.mean(distances)
                    # 归一化：距离越近，密度越大
                    metrics['agent_density'] = min(1.0, 20.0 / avg_distance) if avg_distance > 0 else 1.0
                else:
                    metrics['agent_density'] = 0.5
            else:
                metrics['agent_density'] = 0.0
            
            return metrics
        
        return evaluate_spatial
    
    def _init_temporal_evaluator(self) -> Callable:
        """初始化时间评估器"""
        def evaluate_temporal(world_state: Dict, agent_states: Dict) -> Dict[str, float]:
            """评估时间相关指标"""
            metrics = {}
            
            # 环境变化频率
            dynamics = world_state.get('dynamics', {})
            change_frequency = dynamics.get('change_frequency', 0.1)
            change_amplitude = dynamics.get('change_amplitude', 0.1)
            metrics['temporal_volatility'] = change_frequency * change_amplitude
            
            # 事件密度
            event_log = dynamics.get('events', [])
            time_window = 300.0  # 5分钟窗口
            current_time = time.time()
            recent_events = [event for event in event_log 
                           if current_time - event.get('timestamp', 0) < time_window]
            metrics['event_density'] = len(recent_events) / 10.0  # 归一化
            
            # 环境稳定性
            stability_metrics = world_state.get('stability', {})
            terrain_stability = stability_metrics.get('terrain', 0.8)
            resource_stability = stability_metrics.get('resources', 0.8)
            metrics['environmental_stability'] = (terrain_stability + resource_stability) / 2
            
            # 智能体活动频率
            if agent_states:
                activity_scores = []
                for state in agent_states.values():
                    actions = state.get('recent_actions', [])
                    if actions:
                        # 计算最近活动的频率
                        recent_activity = sum(1 for action in actions 
                                             if current_time - action.get('timestamp', 0) < 60)
                        activity_scores.append(recent_activity / 60.0)  # 每分钟活动数
                
                metrics['agent_activity_level'] = np.mean(activity_scores) if activity_scores else 0.0
            else:
                metrics['agent_activity_level'] = 0.0
            
            # 周期性模式检测
            cycle_metrics = world_state.get('cycles', {})
            day_night_cycle = cycle_metrics.get('day_night_strength', 0.0)
            seasonal_cycle = cycle_metrics.get('seasonal_strength', 0.0)
            metrics['temporal_patterning'] = (day_night_cycle + seasonal_cycle) / 2
            
            return metrics
        
        return evaluate_temporal
    
    def _init_resource_evaluator(self) -> Callable:
        """初始化资源评估器"""
        def evaluate_resources(world_state: Dict, agent_states: Dict) -> Dict[str, float]:
            """评估资源相关指标"""
            metrics = {}
            
            # 资源丰富度
            resources = world_state.get('resources', {})
            total_resources = resources.get('total_count', 100)
            resource_types = resources.get('type_count', 5)
            metrics['resource_abundance'] = min(1.0, total_resources / 1000.0) * (resource_types / 10.0)
            
            # 资源稀缺度
            scarcity = resources.get('scarcity_level', 0.5)
            metrics['resource_scarcity'] = scarcity
            
            # 资源分布均匀度
            distribution = resources.get('distribution', {})
            uniformity = distribution.get('uniformity', 0.5)
            clustering = distribution.get('clustering', 0.5)
            # 高均匀度 = 低集群度
            metrics['resource_distribution'] = uniformity * (1 - clustering)
            
            # 资源可访问性
            accessibility = distribution.get('accessibility', 0.7)
            connectivity = distribution.get('terrain_connectivity', 0.8)
            metrics['resource_accessibility'] = accessibility * connectivity
            
            # 资源质量
            quality_metrics = resources.get('quality', {})
            avg_quality = quality_metrics.get('average', 0.7)
            quality_variance = quality_metrics.get('variance', 0.1)
            metrics['resource_quality'] = avg_quality * (1 - quality_variance)
            
            # 资源竞争度
            if agent_states:
                agent_counts = len(agent_states)
                # 资源数量 / 智能体数量 的比值
                metrics['resource_competition'] = max(0.0, 1.0 - (total_resources / agent_counts / 100.0))
            else:
                metrics['resource_competition'] = 0.0
            
            # 资源再生速度
            regeneration = resources.get('regeneration', {})
            regeneration_rate = regeneration.get('rate', 0.1)
            regeneration_delay = regeneration.get('average_delay', 60.0)
            metrics['resource_regeneration'] = regeneration_rate * (60.0 / (regeneration_delay + 1.0))
            
            return metrics
        
        return evaluate_resources
    
    def _init_danger_evaluator(self) -> Callable:
        """初始化危险评估器"""
        def evaluate_dangers(world_state: Dict, agent_states: Dict) -> Dict[str, float]:
            """评估危险相关指标"""
            metrics = {}
            
            # 敌对实体密度
            dangers = world_state.get('dangers', {})
            hostile_density = dangers.get('hostile_density', 0.1)
            metrics['hostile_density'] = hostile_density
            
            # 环境危险
            environmental = dangers.get('environmental', {})
            hazard_count = environmental.get('hazard_count', 0)
            hazard_intensity = environmental.get('intensity', 0.1)
            metrics['environmental_danger'] = min(1.0, hazard_count / 10.0 + hazard_intensity)
            
            # 生存挑战度
            survival_challenge = dangers.get('survival_challenge', 0.3)
            metrics['survival_challenge'] = survival_challenge
            
            # 预警系统
            warning_system = dangers.get('warning_system', {})
            warning_coverage = warning_system.get('coverage', 0.5)
            warning_accuracy = warning_system.get('accuracy', 0.8)
            metrics['danger_preparedness'] = warning_coverage * warning_accuracy
            
            # 逃生通道
            escape_routes = dangers.get('escape_routes', {})
            route_count = escape_routes.get('count', 3)
            route_quality = escape_routes.get('quality', 0.7)
            metrics['escape_accessibility'] = min(1.0, route_count / 10.0) * route_quality
            
            # 危险升级速度
            escalation = dangers.get('escalation', {})
            escalation_rate = escalation.get('rate', 0.05)
            escalation_threshold = escalation.get('threshold', 0.7)
            metrics['danger_escalation'] = escalation_rate * escalation_threshold
            
            # 智能体防护能力
            if agent_states:
                protection_levels = []
                for state in agent_states.values():
                    equipment = state.get('equipment', {})
                    protection = equipment.get('protection_level', 0.3)
                    protection_levels.append(protection)
                metrics['agent_protection'] = np.mean(protection_levels) if protection_levels else 0.3
            else:
                metrics['agent_protection'] = 0.3
            
            return metrics
        
        return evaluate_dangers
    
    def _init_adaptability_evaluator(self) -> Callable:
        """初始化适应性评估器"""
        def evaluate_adaptability(world_state: Dict, agent_states: Dict) -> Dict[str, float]:
            """评估适应性相关指标"""
            metrics = {}
            
            # 智能体学习能力
            if agent_states:
                learning_rates = []
                for agent_id, state in agent_states.items():
                    performance = state.get('performance', {})
                    learning_rate = performance.get('learning_rate', 0.3)
                    adaptation_speed = performance.get('adaptation_speed', 0.5)
                    # 综合学习能力
                    learning_ability = (learning_rate + adaptation_speed) / 2
                    learning_rates.append(learning_ability)
                
                metrics['agent_learning_ability'] = np.mean(learning_rates) if learning_rates else 0.0
            else:
                metrics['agent_learning_ability'] = 0.0
            
            # 环境变化适应度
            change_metrics = world_state.get('change_adaptation', {})
            adaptation_frequency = change_metrics.get('frequency', 0.1)
            adaptation_success_rate = change_metrics.get('success_rate', 0.7)
            metrics['adaptation_fitness'] = adaptation_frequency * adaptation_success_rate
            
            # 压力承受力
            stress_metrics = world_state.get('stress_tolerance', {})
            max_tolerable_stress = stress_metrics.get('max_tolerable', 0.8)
            stress_recovery_speed = stress_metrics.get('recovery_speed', 0.5)
            metrics['stress_tolerance'] = max_tolerable_stress * stress_recovery_speed
            
            # 创新解决能力
            innovation_metrics = world_state.get('innovation', {})
            creativity_level = innovation_metrics.get('creativity', 0.3)
            problem_solving_efficiency = innovation_metrics.get('problem_solving', 0.5)
            metrics['innovative_capability'] = (creativity_level + problem_solving_efficiency) / 2
            
            # 协作适应性
            if agent_states and len(agent_states) > 1:
                collaboration_metrics = world_state.get('collaboration', {})
                cooperation_level = collaboration_metrics.get('cooperation', 0.4)
                coordination_efficiency = collaboration_metrics.get('coordination', 0.6)
                metrics['collaborative_adaptation'] = cooperation_level * coordination_efficiency
            else:
                metrics['collaborative_adaptation'] = 0.0
            
            # 资源转换能力
            resource_conversion = world_state.get('resource_conversion', {})
            conversion_efficiency = resource_conversion.get('efficiency', 0.5)
            conversion_speed = resource_conversion.get('speed', 0.4)
            metrics['resource_conversion_ability'] = conversion_efficiency * conversion_speed
            
            # 环境预测能力
            prediction_metrics = world_state.get('prediction', {})
            prediction_accuracy = prediction_metrics.get('accuracy', 0.6)
            prediction_horizon = prediction_metrics.get('horizon', 10.0)  # 预测时间范围
            normalized_horizon = min(1.0, prediction_horizon / 60.0)  # 归一化到1小时
            metrics['predictive_ability'] = prediction_accuracy * normalized_horizon
            
            return metrics
        
        return evaluate_adaptability
    
    def create_snapshot(self, 
                       world_state: Dict[str, Any],
                       agent_states: Dict[str, Dict[str, Any]] = None) -> EnvironmentSnapshot:
        """创建环境快照"""
        if agent_states is None:
            agent_states = {}
        
        # 收集性能指标
        performance_metrics = {}
        for metric_name in self.performance_monitor.metrics_history:
            if self.performance_monitor.metrics_history[metric_name]:
                recent_data = list(self.performance_monitor.metrics_history[metric_name])
                if recent_data:
                    performance_metrics[metric_name] = recent_data[-1][1]
        
        # 系统负载信息
        system_load = {
            'cpu_usage': world_state.get('system_stats', {}).get('cpu_usage', 0.3),
            'memory_usage': world_state.get('system_stats', {}).get('memory_usage', 0.4),
            'process_count': world_state.get('system_stats', {}).get('process_count', 10)
        }
        
        snapshot = EnvironmentSnapshot(
            timestamp=time.time(),
            world_state=world_state,
            agent_states=agent_states,
            performance_metrics=performance_metrics,
            system_load=system_load
        )
        
        self.snapshot_history.append(snapshot)
        return snapshot
    
    def evaluate_environment(self, 
                           world_state: Dict[str, Any],
                           agent_states: Dict[str, Dict[str, Any]] = None,
                           evaluation_type: EvaluationType = EvaluationType.REAL_TIME,
                           custom_metrics: Dict[str, float] = None) -> EvaluationResult:
        """
        执行环境评估
        
        Args:
            world_state: 世界状态
            agent_states: 智能体状态
            evaluation_type: 评估类型
            custom_metrics: 自定义指标
            
        Returns:
            EvaluationResult: 评估结果
        """
        start_time = time.time()
        evaluation_id = f"eval_{int(start_time)}_{np.random.randint(1000, 9999)}"
        
        with self._lock:
            try:
                # 创建快照
                snapshot = self.create_snapshot(world_state, agent_states)
                
                if agent_states is None:
                    agent_states = {}
                
                # 执行分类评估
                category_scores = {}
                all_metrics = {}
                
                # 空间评估
                spatial_metrics = self.spatial_evaluator(world_state, agent_states)
                category_scores[MetricCategory.SPATIAL] = np.mean(list(spatial_metrics.values()))
                all_metrics.update({f"spatial_{k}": v for k, v in spatial_metrics.items()})
                
                # 时间评估
                temporal_metrics = self.temporal_evaluator(world_state, agent_states)
                category_scores[MetricCategory.TEMPORAL] = np.mean(list(temporal_metrics.values()))
                all_metrics.update({f"temporal_{k}": v for k, v in temporal_metrics.items()})
                
                # 资源评估
                resource_metrics = self.resource_evaluator(world_state, agent_states)
                category_scores[MetricCategory.RESOURCES] = np.mean(list(resource_metrics.values()))
                all_metrics.update({f"resource_{k}": v for k, v in resource_metrics.items()})
                
                # 危险评估
                danger_metrics = self.danger_evaluator(world_state, agent_states)
                category_scores[MetricCategory.DANGERS] = np.mean(list(danger_metrics.values()))
                all_metrics.update({f"danger_{k}": v for k, v in danger_metrics.items()})
                
                # 适应性评估
                adaptability_metrics = self.adaptability_evaluator(world_state, agent_states)
                category_scores[MetricCategory.ADAPTABILITY] = np.mean(list(adaptability_metrics.values()))
                all_metrics.update({f"adaptability_{k}": v for k, v in adaptability_metrics.items()})
                
                # 性能评估
                performance_health = self.performance_monitor.assess_performance_health()
                performance_score = performance_health['overall_health']
                category_scores[MetricCategory.PERFORMANCE] = performance_score
                all_metrics.update(performance_health['metric_scores'])
                
                # 添加自定义指标
                if custom_metrics:
                    all_metrics.update(custom_metrics)
                
                # 计算整体评分
                overall_score = np.mean(list(category_scores.values()))
                
                # 生成分析结果
                analysis = self._generate_analysis(category_scores, all_metrics, performance_health)
                
                # 生成建议
                recommendations = self._generate_recommendations(category_scores, all_metrics, analysis)
                
                # 风险评估
                risk_assessment = self._assess_risks(category_scores, all_metrics)
                
                # 创建评估结果
                evaluation_result = EvaluationResult(
                    evaluation_id=evaluation_id,
                    evaluation_type=evaluation_type,
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    overall_score=overall_score,
                    category_scores=category_scores,
                    metrics=all_metrics,
                    analysis=analysis,
                    recommendations=recommendations,
                    risk_assessment=risk_assessment
                )
                
                # 更新统计信息
                self._update_evaluation_stats(evaluation_type, evaluation_result.duration)
                
                # 保存评估历史
                self.evaluation_history.append(evaluation_result)
                
                logger.debug(f"环境评估完成: {evaluation_id}, 整体评分: {overall_score:.3f}")
                return evaluation_result
                
            except Exception as e:
                logger.error(f"环境评估失败: {str(e)}")
                raise
    
    def _generate_analysis(self, 
                         category_scores: Dict[MetricCategory, float],
                         all_metrics: Dict[str, float],
                         performance_health: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析结果"""
        analysis = {}
        
        # 识别强项和弱项
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['strengths'] = [cat.value for cat, score in sorted_categories[:2] if score > 0.7]
        analysis['weaknesses'] = [cat.value for cat, score in sorted_categories[-2:] if score < 0.5]
        
        # 环境类型判断
        high_spatial = category_scores.get(MetricCategory.SPATIAL, 0.5)
        high_resource = category_scores.get(MetricCategory.RESOURCES, 0.5)
        high_danger = category_scores.get(MetricCategory.DANGERS, 0.5)
        
        if high_spatial > 0.8 and high_danger > 0.6:
            analysis['environment_type'] = 'high_risk_complex'
        elif high_spatial > 0.7 and high_resource > 0.7:
            analysis['environment_type'] = 'resource_rich_complex'
        elif high_danger > 0.8:
            analysis['environment_type'] = 'extreme_danger'
        elif high_resource < 0.3:
            analysis['environment_type'] = 'resource_scarce'
        else:
            analysis['environment_type'] = 'moderate_balanced'
        
        # 智能体适应状况
        agent_learning = all_metrics.get('adaptability_agent_learning_ability', 0.5)
        stress_tolerance = all_metrics.get('adaptability_stress_tolerance', 0.5)
        
        if agent_learning > 0.7 and stress_tolerance > 0.6:
            analysis['agent_adaptation_status'] = 'well_adapted'
        elif agent_learning < 0.4 or stress_tolerance < 0.3:
            analysis['agent_adaptation_status'] = 'poorly_adapted'
        else:
            analysis['agent_adaptation_status'] = 'moderately_adapted'
        
        # 趋势分析
        analysis['trend'] = self._analyze_trends(all_metrics)
        
        # 稳定性评估
        stability_score = all_metrics.get('temporal_environmental_stability', 0.5)
        if stability_score > 0.8:
            analysis['stability'] = 'high_stability'
        elif stability_score < 0.4:
            analysis['stability'] = 'low_stability'
        else:
            analysis['stability'] = 'moderate_stability'
        
        # 性能健康状况
        health_issues = performance_health.get('issues', [])
        analysis['performance_issues'] = health_issues
        
        return analysis
    
    def _generate_recommendations(self, 
                                category_scores: Dict[MetricCategory, float],
                                all_metrics: Dict[str, float],
                                analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于弱项的建议
        for weakness in analysis.get('weaknesses', []):
            if weakness == 'spatial':
                recommendations.append('考虑降低地形复杂度或改善路径连通性')
            elif weakness == 'resources':
                recommendations.append('增加资源生成速度或改善资源分布')
            elif weakness == 'dangers':
                recommendations.append('减少敌对实体密度或增强预警系统')
            elif weakness == 'adaptability':
                recommendations.append('优化智能体学习机制或降低变化频率')
            elif weakness == 'performance':
                recommendations.append('监控并优化智能体性能瓶颈')
        
        # 基于环境类型的建议
        env_type = analysis.get('environment_type', 'moderate_balanced')
        if env_type == 'high_risk_complex':
            recommendations.append('环境风险较高，建议加强智能体防护或提供更多逃生通道')
        elif env_type == 'resource_scarce':
            recommendations.append('资源稀缺，建议增加资源点或优化资源分配机制')
        elif env_type == 'extreme_danger':
            recommendations.append('极度危险环境，建议立即降低敌对实体密度')
        
        # 基于适应状况的建议
        adaptation_status = analysis.get('agent_adaptation_status', 'moderately_adapted')
        if adaptation_status == 'poorly_adapted':
            recommendations.append('智能体适应不良，建议调整环境参数或增强学习算法')
        elif adaptation_status == 'well_adapted':
            recommendations.append('智能体适应良好，可以考虑适当增加挑战性')
        
        # 基于趋势的建议
        trend = analysis.get('trend', 'stable')
        if trend == 'deteriorating':
            recommendations.append('环境呈现恶化趋势，需要及时干预')
        elif trend == 'improving' and analysis.get('environment_type') != 'extreme_danger':
            recommendations.append('环境正在改善，可以考虑增加复杂度')
        
        return recommendations[:5]  # 最多返回5个建议
    
    def _assess_risks(self, 
                     category_scores: Dict[MetricCategory, float],
                     all_metrics: Dict[str, float]) -> Dict[str, float]:
        """评估风险"""
        risks = {}
        
        # 危险相关风险
        danger_score = category_scores.get(MetricCategory.DANGERS, 0.5)
        risks['danger_risk'] = danger_score
        
        # 资源枯竭风险
        resource_score = category_scores.get(MetricCategory.RESOURCES, 0.5)
        if resource_score < 0.3:
            risks['resource_depletion_risk'] = 0.8
        elif resource_score < 0.5:
            risks['resource_depletion_risk'] = 0.5
        else:
            risks['resource_depletion_risk'] = 0.2
        
        # 智能体适应风险
        adaptation_score = category_scores.get(MetricCategory.ADAPTABILITY, 0.5)
        if adaptation_score < 0.4:
            risks['adaptation_failure_risk'] = 0.8
        elif adaptation_score < 0.6:
            risks['adaptation_failure_risk'] = 0.4
        else:
            risks['adaptation_failure_risk'] = 0.1
        
        # 性能退化风险
        performance_score = category_scores.get(MetricCategory.PERFORMANCE, 0.5)
        if performance_score < 0.4:
            risks['performance_degradation_risk'] = 0.7
        else:
            risks['performance_degradation_risk'] = 1.0 - performance_score
        
        # 环境不稳定风险
        stability = all_metrics.get('temporal_environmental_stability', 0.5)
        risks['instability_risk'] = 1.0 - stability
        
        return risks
    
    def _analyze_trends(self, all_metrics: Dict[str, float]) -> str:
        """分析趋势"""
        # 收集关键指标的趋势
        key_metrics = [
            'performance_overall_health',
            'adaptability_agent_learning_ability',
            'spatial_terrain_complexity',
            'resource_resource_scarcity'
        ]
        
        trends = []
        for metric in key_metrics:
            if metric in self.performance_monitor.metrics_history:
                trend_info = self.performance_monitor.get_metric_trend(metric)
                if trend_info['trend'] == 'increasing':
                    trends.append('up')
                elif trend_info['trend'] == 'decreasing':
                    trends.append('down')
                else:
                    trends.append('stable')
        
        # 综合判断
        if not trends:
            return 'stable'
        
        up_count = trends.count('up')
        down_count = trends.count('down')
        
        if up_count > len(trends) * 0.6:
            return 'improving'
        elif down_count > len(trends) * 0.6:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _update_evaluation_stats(self, evaluation_type: EvaluationType, duration: float):
        """更新评估统计"""
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['evaluation_types_used'][evaluation_type.value] += 1
        
        # 更新平均评估时间
        current_avg = self.evaluation_stats['avg_evaluation_time']
        total_count = self.evaluation_stats['total_evaluations']
        self.evaluation_stats['avg_evaluation_time'] = \
            (current_avg * (total_count - 1) + duration) / total_count
        
        self.evaluation_stats['last_evaluation'] = time.time()
    
    def start_monitoring(self, 
                        evaluation_callback: Callable[[EvaluationResult], None] = None):
        """启动监控"""
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            logger.warning("监控已在运行中")
            return
        
        self._stop_evaluation = False
        self._evaluation_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(evaluation_callback,),
            daemon=True
        )
        self._evaluation_thread.start()
        
        logger.info("环境监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            self._stop_evaluation = True
            self._evaluation_thread.join(timeout=5.0)
            logger.info("环境监控已停止")
    
    def _monitoring_loop(self, evaluation_callback: Callable[[EvaluationResult], None]):
        """监控循环"""
        while not self._stop_evaluation:
            try:
                # 这里应该获取当前世界状态和智能体状态
                # 简化实现：使用占位符
                world_state = {
                    'terrain': {'height_variance': 0.5, 'cave_density': 0.3, 'connectivity': 0.8},
                    'resources': {'total_count': 500, 'scarcity_level': 0.4},
                    'dangers': {'hostile_density': 0.2, 'survival_challenge': 0.3}
                }
                
                # 执行评估
                result = self.evaluate_environment(
                    world_state=world_state,
                    evaluation_type=EvaluationType.PERIODIC
                )
                
                # 调用回调函数
                if evaluation_callback:
                    evaluation_callback(result)
                
                # 等待下一个评估周期
                time.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {str(e)}")
                time.sleep(5.0)  # 错误后等待5秒再继续
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        with self._lock:
            return {
                'evaluation_stats': self.evaluation_stats,
                'history_size': len(self.evaluation_history),
                'snapshot_size': len(self.snapshot_history),
                'monitoring_active': self._evaluation_thread and self._evaluation_thread.is_alive(),
                'performance_monitor_stats': {
                    'metrics_tracked': len(self.performance_monitor.metrics_history),
                    'window_size': self.performance_monitor.window_size
                }
            }
    
    def export_evaluation_data(self, filepath: str, limit: int = 100):
        """导出评估数据"""
        export_data = {
            'export_timestamp': time.time(),
            'statistics': self.get_evaluation_statistics(),
            'evaluations': [
                result.to_dict() for result in list(self.evaluation_history)[-limit:]
            ],
            'snapshots': [
                {
                    'timestamp': snap.timestamp,
                    'world_state_size': len(str(snap.world_state)),
                    'agent_count': len(snap.agent_states),
                    'performance_metrics_count': len(snap.performance_metrics)
                }
                for snap in list(self.snapshot_history)[-limit:]
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"评估数据已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出评估数据失败: {str(e)}")
            raise


# 工厂函数
def create_environment_evaluator(config: Dict) -> EnvironmentEvaluator:
    """
    创建环境评估器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        EnvironmentEvaluator: 环境评估器实例
    """
    return EnvironmentEvaluator(
        evaluation_interval=config.get('evaluation_interval', 30.0),
        performance_window=config.get('performance_window', 50),
        enable_async=config.get('enable_async', True)
    )


if __name__ == "__main__":
    # 演示用法
    logger.info("环境评估器演示")
    
    # 创建评估器
    evaluator = EnvironmentEvaluator(evaluation_interval=10.0)
    
    # 模拟世界状态
    world_state = {
        'terrain': {
            'height_variance': 0.6,
            'cave_density': 0.4,
            'connectivity': 0.7
        },
        'resources': {
            'total_count': 800,
            'scarcity_level': 0.3,
            'type_count': 10
        },
        'dangers': {
            'hostile_density': 0.2,
            'survival_challenge': 0.4
        },
        'dynamics': {
            'change_frequency': 0.1,
            'change_amplitude': 0.2
        }
    }
    
    # 模拟智能体状态
    agent_states = {
        'agent_1': {
            'position': [10, 20, 30],
            'performance': {'learning_rate': 0.6, 'adaptation_speed': 0.7},
            'equipment': {'protection_level': 0.6}
        }
    }
    
    # 执行评估
    result = evaluator.evaluate_environment(world_state, agent_states)
    print("评估结果:", json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    
    # 获取统计信息
    stats = evaluator.get_evaluation_statistics()
    print("评估统计:", json.dumps(stats, indent=2, ensure_ascii=False))