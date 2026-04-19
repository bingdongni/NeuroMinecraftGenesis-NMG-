"""
环境动态复杂度调节器

该模块实现环境的动态复杂度调节功能，包括：
1. 环境复杂度的实时评估
2. 智能体能力评估
3. 动态难度调节算法
4. 挑战性维持机制
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainComplexity(Enum):
    """地形复杂度枚举"""
    SIMPLE = 0  # 简单地形
    MODERATE = 1  # 中等复杂
    COMPLEX = 2  # 复杂地形
    EXTREME = 3  # 极端复杂


class ResourceScarcity(Enum):
    """资源稀缺度枚举"""
    ABUNDANT = 0  # 资源丰富
    NORMAL = 1  # 正常
    SCARCE = 2  # 稀缺
    EXTREME_SCARCE = 3  # 极度稀缺


class DangerLevel(Enum):
    """危险系数枚举"""
    SAFE = 0  # 安全
    LOW = 1  # 低危险
    MEDIUM = 2  # 中等危险
    HIGH = 3  # 高危险
    EXTREME = 4  # 极端危险


@dataclass
class EnvironmentMetrics:
    """环境复杂度指标"""
    terrain_complexity: float  # 地形复杂度 (0-1)
    resource_scarcity: float   # 资源稀缺度 (0-1)
    danger_level: float        # 危险系数 (0-1)
    spatial_variance: float    # 空间方差
    temporal_stability: float  # 时间稳定性
    accessibility: float       # 可访问性 (0-1)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'terrain_complexity': self.terrain_complexity,
            'resource_scarcity': self.resource_scarcity,
            'danger_level': self.danger_level,
            'spatial_variance': self.spatial_variance,
            'temporal_stability': self.temporal_stability,
            'accessibility': self.accessibility
        }


@dataclass
class AgentCapabilities:
    """智能体能力评估"""
    spatial_reasoning: float   # 空间推理能力 (0-1)
    resource_management: float # 资源管理能力 (0-1)
    danger_assessment: float   # 危险评估能力 (0-1)
    adaptation_speed: float    # 适应速度 (0-1)
    problem_solving: float     # 问题解决能力 (0-1)
    survival_rate: float       # 生存率 (0-1)
    
    def overall_score(self) -> float:
        """计算综合能力分数"""
        return (self.spatial_reasoning + self.resource_management + 
                self.danger_assessment + self.adaptation_speed + 
                self.problem_solving + self.survival_rate) / 6
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'spatial_reasoning': self.spatial_reasoning,
            'resource_management': self.resource_management,
            'danger_assessment': self.danger_assessment,
            'adaptation_speed': self.adaptation_speed,
            'problem_solving': self.problem_solving,
            'survival_rate': self.survival_rate,
            'overall_score': self.overall_score()
        }


class DynamicComplexityController:
    """动态复杂度控制器"""
    
    def __init__(self, 
                 initial_complexity: float = 0.3,
                 adaptation_rate: float = 0.1,
                 complexity_range: Tuple[float, float] = (0.1, 1.0)):
        """
        初始化动态复杂度控制器
        
        Args:
            initial_complexity: 初始复杂度
            adaptation_rate: 适应速率
            complexity_range: 复杂度范围
        """
        self.current_complexity = initial_complexity
        self.adaptation_rate = adaptation_rate
        self.complexity_range = complexity_range
        self.evaluation_history = []
        self.adaptation_history = []
        
        # 性能监控参数
        self.monitoring_window = 100  # 监控窗口大小
        self.performance_threshold = 0.6  # 性能阈值
        
        logger.info(f"初始化动态复杂度控制器 - 初始复杂度: {initial_complexity}")
    
    def evaluate_environment_complexity(self, world_state: Dict) -> EnvironmentMetrics:
        """
        评估环境复杂度
        
        Args:
            world_state: 世界状态数据
            
        Returns:
            EnvironmentMetrics: 环境复杂度指标
        """
        try:
            # 地形复杂度评估
            terrain_complexity = self._calculate_terrain_complexity(world_state)
            
            # 资源稀缺度评估
            resource_scarcity = self._calculate_resource_scarcity(world_state)
            
            # 危险系数评估
            danger_level = self._calculate_danger_level(world_state)
            
            # 空间方差计算
            spatial_variance = self._calculate_spatial_variance(world_state)
            
            # 时间稳定性计算
            temporal_stability = self._calculate_temporal_stability(world_state)
            
            # 可访问性计算
            accessibility = self._calculate_accessibility(world_state)
            
            metrics = EnvironmentMetrics(
                terrain_complexity=terrain_complexity,
                resource_scarcity=resource_scarcity,
                danger_level=danger_level,
                spatial_variance=spatial_variance,
                temporal_stability=temporal_stability,
                accessibility=accessibility
            )
            
            self.evaluation_history.append({
                'timestamp': time.time(),
                'metrics': metrics.to_dict()
            })
            
            # 保持历史记录在合理范围内
            if len(self.evaluation_history) > self.monitoring_window:
                self.evaluation_history = self.evaluation_history[-self.monitoring_window:]
            
            logger.debug(f"环境复杂度评估完成: {metrics.to_dict()}")
            return metrics
            
        except Exception as e:
            logger.error(f"环境复杂度评估失败: {str(e)}")
            raise
    
    def _calculate_terrain_complexity(self, world_state: Dict) -> float:
        """
        计算地形复杂度
        
        计算依据：
        - 地形起伏程度
        - 洞穴数量和深度
        - 垂直空间变化
        - 可通行路径的复杂程度
        """
        terrain_data = world_state.get('terrain', {})
        
        # 基础地形复杂度计算
        height_variance = terrain_data.get('height_variance', 0.5)
        cave_density = terrain_data.get('cave_density', 0.3)
        vertical_range = terrain_data.get('vertical_range', 1.0)
        
        # 权重计算：洞穴密度从0.3逐步提升到0.8
        cave_complexity = min(0.8, cave_density + 0.5 * self.current_complexity)
        
        # 地形复杂度 = 高度变化 + 洞穴复杂度 + 垂直空间复杂度
        complexity = (0.4 * height_variance + 
                     0.3 * cave_complexity + 
                     0.3 * min(1.0, vertical_range))
        
        return min(1.0, complexity)
    
    def _calculate_resource_scarcity(self, world_state: Dict) -> float:
        """
        计算资源稀缺度
        
        计算依据：
        - 可用资源数量
        - 资源分布均匀度
        - 资源质量等级
        - 资源再生速度
        """
        resource_data = world_state.get('resources', {})
        
        # 资源数量指标
        total_resources = resource_data.get('total_count', 100)
        resource_types = resource_data.get('type_count', 5)
        distribution_uniformity = resource_data.get('uniformity', 0.5)
        
        # 基础稀缺度计算
        scarcity = 1.0 - (total_resources / 1000)  # 归一化到0-1
        
        # 根据当前复杂度动态调整资源稀缺度
        # 从1.0逐步降低到0.3
        dynamic_scarcity = max(0.3, scarcity * (1 - self.current_complexity * 0.7))
        
        # 考虑分布均匀度
        adjusted_scarcity = dynamic_scarcity + (1 - distribution_uniformity) * 0.2
        
        return min(1.0, adjusted_scarcity)
    
    def _calculate_danger_level(self, world_state: Dict) -> float:
        """
        计算危险系数
        
        计算依据：
        - 敌对生物密度
        - 环境危险因素
        - 生存挑战程度
        """
        danger_data = world_state.get('dangers', {})
        
        # 敌对生物密度
        hostile_density = danger_data.get('hostile_density', 0.1)
        
        # 环境危险因素
        environmental_hazards = danger_data.get('environmental_hazards', 0.1)
        
        # 生存挑战程度
        survival_challenge = danger_data.get('survival_challenge', 0.1)
        
        # 综合危险系数
        danger_level = (0.4 * hostile_density + 
                       0.3 * environmental_hazards + 
                       0.3 * survival_challenge)
        
        # 根据复杂度动态调整
        adjusted_danger = min(1.0, danger_level + self.current_complexity * 0.3)
        
        return adjusted_danger
    
    def _calculate_spatial_variance(self, world_state: Dict) -> float:
        """计算空间方差"""
        # 简化计算：基于地形和资源分布的空间变化
        terrain_variance = world_state.get('terrain', {}).get('height_variance', 0.5)
        resource_clustering = world_state.get('resources', {}).get('clustering', 0.5)
        
        return (terrain_variance + (1 - resource_clustering)) / 2
    
    def _calculate_temporal_stability(self, world_state: Dict) -> float:
        """计算时间稳定性"""
        # 基于环境变化的频率和幅度
        change_frequency = world_state.get('dynamics', {}).get('change_frequency', 0.1)
        change_magnitude = world_state.get('dynamics', {}).get('change_magnitude', 0.1)
        
        # 稳定性 = 1 - 变化程度
        stability = 1.0 - (change_frequency + change_magnitude) / 2
        return max(0.0, stability)
    
    def _calculate_accessibility(self, world_state: Dict) -> float:
        """计算可访问性"""
        # 基于地形连通性和资源分布
        connectivity = world_state.get('terrain', {}).get('connectivity', 0.8)
        resource_accessibility = world_state.get('resources', {}).get('accessibility', 0.8)
        
        return (connectivity + resource_accessibility) / 2
    
    def evaluate_agent_capabilities(self, agent_performance: Dict) -> AgentCapabilities:
        """
        评估智能体能力
        
        Args:
            agent_performance: 智能体性能数据
            
        Returns:
            AgentCapabilities: 智能体能力评估
        """
        try:
            # 从性能数据中提取各项能力指标
            capabilities = AgentCapabilities(
                spatial_reasoning=self._assess_spatial_reasoning(agent_performance),
                resource_management=self._assess_resource_management(agent_performance),
                danger_assessment=self._assess_danger_assessment(agent_performance),
                adaptation_speed=self._assess_adaptation_speed(agent_performance),
                problem_solving=self._assess_problem_solving(agent_performance),
                survival_rate=self._assess_survival_rate(agent_performance)
            )
            
            logger.debug(f"智能体能力评估完成: {capabilities.to_dict()}")
            return capabilities
            
        except Exception as e:
            logger.error(f"智能体能力评估失败: {str(e)}")
            raise
    
    def _assess_spatial_reasoning(self, performance: Dict) -> float:
        """评估空间推理能力"""
        navigation_success = performance.get('navigation_success_rate', 0.5)
        path_efficiency = performance.get('path_efficiency', 0.5)
        spatial_memory = performance.get('spatial_memory_accuracy', 0.5)
        
        return (navigation_success + path_efficiency + spatial_memory) / 3
    
    def _assess_resource_management(self, performance: Dict) -> float:
        """评估资源管理能力"""
        resource_collection_rate = performance.get('resource_collection_rate', 0.5)
        resource_efficiency = performance.get('resource_efficiency', 0.5)
        inventory_management = performance.get('inventory_management', 0.5)
        
        return (resource_collection_rate + resource_efficiency + inventory_management) / 3
    
    def _assess_danger_assessment(self, performance: Dict) -> float:
        """评估危险评估能力"""
        danger_avoidance = performance.get('danger_avoidance_rate', 0.5)
        threat_recognition = performance.get('threat_recognition_accuracy', 0.5)
        safety_decisions = performance.get('safety_decision_accuracy', 0.5)
        
        return (danger_avoidance + threat_recognition + safety_decisions) / 3
    
    def _assess_adaptation_speed(self, performance: Dict) -> float:
        """评估适应速度"""
        learning_rate = performance.get('learning_rate', 0.5)
        adaptation_time = performance.get('adaptation_time', 10.0)
        # 适应时间越短，适应速度越快
        speed_score = max(0, 1.0 - adaptation_time / 20.0)
        
        return (learning_rate + speed_score) / 2
    
    def _assess_problem_solving(self, performance: Dict) -> float:
        """评估问题解决能力"""
        challenge_success = performance.get('challenge_success_rate', 0.5)
        solution_efficiency = performance.get('solution_efficiency', 0.5)
        creativity_score = performance.get('creativity_score', 0.5)
        
        return (challenge_success + solution_efficiency + creativity_score) / 3
    
    def _assess_survival_rate(self, performance: Dict) -> float:
        """评估生存率"""
        time_alive = performance.get('average_survival_time', 10.0)
        death_avoidance = performance.get('death_avoidance_rate', 0.5)
        
        # 归一化生存时间分数（假设最大生存时间为60分钟）
        time_score = min(1.0, time_alive / 60.0)
        
        return (time_score + death_avoidance) / 2
    
    def calculate_optimal_complexity(self, 
                                   env_metrics: EnvironmentMetrics,
                                   agent_caps: AgentCapabilities) -> float:
        """
        计算最优复杂度
        
        Args:
            env_metrics: 环境指标
            agent_caps: 智能体能力
            
        Returns:
            float: 最优复杂度值
        """
        try:
            # 基于能力-挑战匹配的最优复杂度计算
            agent_score = agent_caps.overall_score()
            
            # 目标性能范围：0.4-0.8（保持适度挑战）
            target_performance = 0.6
            
            # 当前环境复杂度下的预期性能
            current_challenge = self._calculate_challenge_level(env_metrics)
            expected_performance = self._predict_performance(agent_score, current_challenge)
            
            # 计算复杂度调整量
            performance_gap = target_performance - expected_performance
            
            # 根据表现差距调整复杂度
            # 如果表现太好，增加复杂度；如果表现太差，降低复杂度
            complexity_adjustment = self.adaptation_rate * performance_gap
            
            # 考虑智能体的特定能力
            adjustment_factor = self._calculate_adjustment_factor(agent_caps)
            final_adjustment = complexity_adjustment * adjustment_factor
            
            # 计算新的目标复杂度
            target_complexity = self.current_complexity + final_adjustment
            
            # 限制在允许范围内
            min_complexity, max_complexity = self.complexity_range
            optimal_complexity = max(min_complexity, 
                                  min(max_complexity, target_complexity))
            
            # 添加随机性以增加不可预测性
            noise_factor = 0.05 * (np.random.random() - 0.5)  # ±5%随机波动
            final_complexity = optimal_complexity * (1 + noise_factor)
            final_complexity = max(min_complexity, min(max_complexity, final_complexity))
            
            logger.debug(f"复杂度计算: 当前={self.current_complexity:.3f}, "
                        f"预期={expected_performance:.3f}, 目标={target_performance:.3f}, "
                        f"调整={final_adjustment:.3f}, 最优={final_complexity:.3f}")
            
            return final_complexity
            
        except Exception as e:
            logger.error(f"最优复杂度计算失败: {str(e)}")
            return self.current_complexity
    
    def _calculate_challenge_level(self, env_metrics: EnvironmentMetrics) -> float:
        """计算挑战水平"""
        # 挑战水平 = 地形复杂度 + 资源稀缺度 + 危险系数
        # 权重：地形40%，资源30%，危险30%
        challenge = (0.4 * env_metrics.terrain_complexity + 
                    0.3 * env_metrics.resource_scarcity + 
                    0.3 * env_metrics.danger_level)
        return challenge
    
    def _predict_performance(self, agent_capability: float, challenge_level: float) -> float:
        """
        预测智能体表现
        
        Args:
            agent_capability: 智能体能力
            challenge_level: 挑战水平
            
        Returns:
            float: 预期表现
        """
        # 使用S型函数来模拟能力-表现关系
        # 当挑战接近能力时，表现最高
        capability_difference = agent_capability - challenge_level
        
        # S型函数，峰值在能力=挑战时
        if capability_difference >= 0:
            # 能力 >= 挑战：表现随挑战增加而提升，但有上限
            performance = 0.3 + 0.7 * (1 - math.exp(-3 * capability_difference))
        else:
            # 能力 < 挑战：表现急剧下降
            performance = 0.3 * math.exp(2 * capability_difference)
        
        return max(0.0, min(1.0, performance))
    
    def _calculate_adjustment_factor(self, agent_caps: AgentCapabilities) -> float:
        """计算调整因子"""
        # 根据智能体的特殊能力调整复杂度变化幅度
        adaptation_bonus = agent_caps.adaptation_speed * 0.2
        problem_solving_bonus = agent_caps.problem_solving * 0.2
        
        # 如果智能体适应性强且善于解决问题，可以承受更大的复杂度变化
        factor = 1.0 + adaptation_bonus + problem_solving_bonus
        
        return min(1.5, max(0.5, factor))  # 限制在0.5-1.5范围内
    
    def adapt_complexity(self, 
                        world_state: Dict, 
                        agent_performance: Dict,
                        update_immediately: bool = False) -> Dict:
        """
        执行复杂度自适应
        
        Args:
            world_state: 当前世界状态
            agent_performance: 智能体表现
            update_immediately: 是否立即更新
            
        Returns:
            Dict: 适应结果
        """
        try:
            # 评估当前环境复杂度
            env_metrics = self.evaluate_environment_complexity(world_state)
            
            # 评估智能体能力
            agent_caps = self.evaluate_agent_capabilities(agent_performance)
            
            # 计算新的目标复杂度
            new_complexity = self.calculate_optimal_complexity(env_metrics, agent_caps)
            
            # 计算变化量
            complexity_change = new_complexity - self.current_complexity
            complexity_change_pct = (complexity_change / self.current_complexity) * 100 if self.current_complexity > 0 else 0
            
            # 记录适应历史
            adaptation_record = {
                'timestamp': time.time(),
                'old_complexity': self.current_complexity,
                'new_complexity': new_complexity,
                'change_amount': complexity_change,
                'change_percentage': complexity_change_pct,
                'env_metrics': env_metrics.to_dict(),
                'agent_capabilities': agent_caps.to_dict(),
                'agent_score': agent_caps.overall_score(),
                'update_immediately': update_immediately
            }
            
            self.adaptation_history.append(adaptation_record)
            
            # 保持历史记录在合理范围内
            if len(self.adaptation_history) > 50:
                self.adaptation_history = self.adaptation_history[-50:]
            
            # 如果需要立即更新或变化显著，则更新当前复杂度
            change_threshold = 0.05  # 5%变化阈值
            if update_immediately or abs(complexity_change_pct) > change_threshold * 100:
                self.current_complexity = new_complexity
                adaptation_record['actually_updated'] = True
                logger.info(f"复杂度自适应: {self.current_complexity:.3f} "
                          f"(变化: {complexity_change_pct:+.1f}%)")
            else:
                adaptation_record['actually_updated'] = False
                logger.debug(f"复杂度适应记录: {self.current_complexity:.3f} "
                           f"(变化: {complexity_change_pct:+.1f}%，未达阈值)")
            
            return {
                'old_complexity': self.current_complexity,
                'new_complexity': new_complexity,
                'complexity_change': complexity_change,
                'complexity_change_pct': complexity_change_pct,
                'adaptation_applied': adaptation_record['actually_updated'],
                'env_metrics': env_metrics.to_dict(),
                'agent_capabilities': agent_caps.to_dict(),
                'adaptation_record': adaptation_record
            }
            
        except Exception as e:
            logger.error(f"复杂度自适应失败: {str(e)}")
            raise
    
    def get_complexity_recommendations(self) -> Dict:
        """获取复杂度建议"""
        if len(self.adaptation_history) < 5:
            return {'status': 'insufficient_data', 'message': '历史数据不足'}
        
        recent_adaptations = self.adaptation_history[-10:]
        
        # 分析趋势
        changes = [adapt['change_amount'] for adapt in recent_adaptations]
        avg_change = np.mean(changes)
        change_std = np.std(changes)
        
        # 判断趋势方向
        if avg_change > 0.01:
            trend = 'increasing'
            recommendation = '环境正在趋向更复杂，考虑平衡进化压力与智能体能力'
        elif avg_change < -0.01:
            trend = 'decreasing'
            recommendation = '环境正在趋向更简单，可能需要增强挑战性'
        else:
            trend = 'stable'
            recommendation = '环境复杂度保持稳定，当前状态良好'
        
        # 变化性分析
        variability = 'high' if change_std > 0.05 else 'normal'
        if variability == 'high':
            recommendation += ' 注意：环境变化较大，建议增加稳定性'
        
        return {
            'current_complexity': self.current_complexity,
            'trend': trend,
            'variability': variability,
            'avg_change': avg_change,
            'change_std': change_std,
            'recommendation': recommendation,
            'adaptation_count': len(self.adaptation_history)
        }
    
    def get_performance_statistics(self) -> Dict:
        """获取性能统计"""
        if not self.evaluation_history:
            return {'status': 'no_data'}
        
        # 环境指标统计
        recent_evaluations = self.evaluation_history[-20:]
        
        complexity_scores = [eval['metrics']['terrain_complexity'] for eval in recent_evaluations]
        resource_scores = [eval['metrics']['resource_scarcity'] for eval in recent_evaluations]
        danger_scores = [eval['metrics']['danger_level'] for eval in recent_evaluations]
        
        return {
            'evaluation_count': len(self.evaluation_history),
            'monitoring_window': self.monitoring_window,
            'environment_stats': {
                'terrain_complexity': {
                    'mean': np.mean(complexity_scores),
                    'std': np.std(complexity_scores),
                    'min': np.min(complexity_scores),
                    'max': np.max(complexity_scores)
                },
                'resource_scarcity': {
                    'mean': np.mean(resource_scores),
                    'std': np.std(resource_scores),
                    'min': np.min(resource_scores),
                    'max': np.max(resource_scores)
                },
                'danger_level': {
                    'mean': np.mean(danger_scores),
                    'std': np.std(danger_scores),
                    'min': np.min(danger_scores),
                    'max': np.max(danger_scores)
                }
            },
            'complexity_range': self.complexity_range,
            'adaptation_rate': self.adaptation_rate
        }
    
    def export_history(self, filepath: str):
        """导出历史数据"""
        export_data = {
            'evaluation_history': self.evaluation_history,
            'adaptation_history': self.adaptation_history,
            'statistics': self.get_performance_statistics(),
            'current_state': {
                'current_complexity': self.current_complexity,
                'complexity_range': self.complexity_range,
                'adaptation_rate': self.adaptation_rate
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"历史数据已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出历史数据失败: {str(e)}")
            raise


# 工厂函数
def create_complexity_controller(config: Dict) -> DynamicComplexityController:
    """
    创建动态复杂度控制器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        DynamicComplexityController: 复杂度控制器实例
    """
    return DynamicComplexityController(
        initial_complexity=config.get('initial_complexity', 0.3),
        adaptation_rate=config.get('adaptation_rate', 0.1),
        complexity_range=tuple(config.get('complexity_range', (0.1, 1.0)))
    )


if __name__ == "__main__":
    # 演示用法
    logger.info("环境动态复杂度调节器演示")
    
    # 创建控制器
    controller = DynamicComplexityController()
    
    # 模拟世界状态
    world_state = {
        'terrain': {
            'height_variance': 0.7,
            'cave_density': 0.5,
            'vertical_range': 0.8
        },
        'resources': {
            'total_count': 500,
            'type_count': 8,
            'uniformity': 0.6
        },
        'dangers': {
            'hostile_density': 0.2,
            'environmental_hazards': 0.1,
            'survival_challenge': 0.3
        }
    }
    
    # 模拟智能体表现
    agent_performance = {
        'navigation_success_rate': 0.8,
        'resource_collection_rate': 0.7,
        'danger_avoidance_rate': 0.9,
        'average_survival_time': 45.0
    }
    
    # 执行复杂度自适应
    result = controller.adapt_complexity(world_state, agent_performance)
    print("复杂度自适应结果:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取建议
    recommendations = controller.get_complexity_recommendations()
    print("复杂度建议:", json.dumps(recommendations, indent=2, ensure_ascii=False))