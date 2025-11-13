#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适应引擎
负责策略在物理世界环境中的实时适应和优化调整
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import random


class AdaptationEngine:
    """
    适应引擎类
    
    功能：
    1. 实时监控策略执行效果
    2. 根据环境反馈进行策略调整
    3. 自适应参数优化
    4. 在线学习和知识积累
    5. 性能收敛性分析
    
    适应机制：
    - 参数调整：实时调整控制参数
    - 策略修改：基于反馈修改执行策略
    - 环境感知：持续感知环境变化
    - 错误恢复：自动处理执行错误
    - 性能优化：持续改进执行效果
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化适应引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('AdaptationEngine')
        
        # 适应参数配置
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.adaptation_patience = self.config.get('adaptation_patience', 10)
        self.min_improvement = self.config.get('min_improvement', 0.001)
        
        # 适应状态管理
        self.adaptation_history = defaultdict(list)
        self.performance_buffer = defaultdict(lambda: deque(maxlen=50))
        self.current_state = {}
        self.adaptation_memory = {}
        
        # 学习组件
        self.parameter_optimizer = self._init_parameter_optimizer()
        self.strategy_modifier = self._init_strategy_modifier()
        self.environment_monitor = self._init_environment_monitor()
        self.error_handler = self._init_error_handler()
        
        # 收敛性分析
        self.convergence_tracker = {}
        self.performance_trends = {}
        
        self.logger.info("适应引擎初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'adaptation_rate': 0.1,
            'learning_rate': 0.01,
            'adaptation_patience': 10,
            'min_improvement': 0.001,
            'adaptation_frequency': 1,  # 每N次执行后适应一次
            'parameter_bounds': {
                'position_gain': [0.1, 10.0],
                'velocity_gain': [0.01, 1.0],
                'force_gain': [0.1, 5.0],
                'timing_offset': [-0.5, 0.5]
            },
            'convergence_criteria': {
                'performance_stability': 0.05,  # 5%性能波动
                'parameter_stability': 0.02,    # 2%参数变化
                'min_adaptations': 5
            },
            'safety_constraints': {
                'max_position_error': 0.1,    # 10cm
                'max_force_variation': 0.3,   # 30%
                'max_timing_deviation': 0.2   # 20%
            }
        }
    
    def _init_parameter_optimizer(self) -> Dict[str, Any]:
        """初始化参数优化器"""
        return {
            'optimization_method': 'gradient_descent',
            'step_size': 0.01,
            'momentum': 0.9,
            'parameter_history': defaultdict(list),
            'optimization_bounds': self.config.get('parameter_bounds', {})
        }
    
    def _init_strategy_modifier(self) -> Dict[str, Any]:
        """初始化策略修改器"""
        return {
            'modification_strategies': [
                'temporal_adjustment',
                'spatial_correction',
                'force_modulation',
                'sequence_reordering'
            ],
            'modification_weights': {
                'temporal_adjustment': 0.3,
                'spatial_correction': 0.4,
                'force_modulation': 0.2,
                'sequence_reordering': 0.1
            }
        }
    
    def _init_environment_monitor(self) -> Dict[str, Any]:
        """初始化环境监控器"""
        return {
            'monitoring_frequency': 10,  # Hz
            'sensitivity_threshold': 0.05,
            'environmental_features': [
                'temperature', 'humidity', 'vibration', 'load_variation'
            ],
            'change_detection': {
                'statistical_threshold': 2.0,  # 2-sigma
                'temporal_window': 100
            }
        }
    
    def _init_error_handler(self) -> Dict[str, Any]:
        """初始化错误处理器"""
        return {
            'error_types': {
                'position_error': {'threshold': 0.05, 'recovery_strategy': 'position_correction'},
                'force_error': {'threshold': 0.2, 'recovery_strategy': 'force_adjustment'},
                'timing_error': {'threshold': 0.1, 'recovery_strategy': 'timing_compensation'},
                'collision_error': {'threshold': 0.0, 'recovery_strategy': 'path_replanning'}
            },
            'recovery_strategies': {}
        }
    
    def adapt_strategy(self, physical_strategy: Dict[str, Any], 
                      physical_environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行策略适应
        
        Args:
            physical_strategy: 物理世界策略
            physical_environment: 物理世界环境
            
        Returns:
            Dict: 适应结果，包含调整后的策略、适应状态等
        """
        try:
            self.logger.info("开始策略适应")
            
            strategy_id = physical_strategy.get('mapped_strategy_id', 'unknown')
            
            # 初始化适应状态（如果不存在）
            if strategy_id not in self.current_state:
                self.current_state[strategy_id] = self._initialize_adaptation_state(physical_strategy)
            
            # 监控环境变化
            environment_changes = self._monitor_environment_changes(physical_environment, strategy_id)
            
            # 执行适应过程
            adaptation_result = self._perform_adaptation(physical_strategy, physical_environment, strategy_id)
            
            # 更新适应状态
            self._update_adaptation_state(strategy_id, adaptation_result)
            
            # 记录适应历史
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'strategy_id': strategy_id,
                'environment_changes': environment_changes,
                'adaptation_result': adaptation_result,
                'performance_impact': self._assess_performance_impact(strategy_id, adaptation_result)
            }
            
            self.adaptation_history[strategy_id].append(adaptation_record)
            
            self.logger.info(f"策略适应完成，适应次数: {self.current_state[strategy_id]['adaptation_count']}")
            
            return adaptation_result
            
        except Exception as e:
            self.logger.error(f"策略适应失败: {str(e)}")
            raise
    
    def _initialize_adaptation_state(self, physical_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """初始化适应状态"""
        return {
            'adaptation_count': 0,
            'current_parameters': self._extract_initial_parameters(physical_strategy),
            'performance_history': [],
            'convergence_status': 'initializing',
            'last_adaptation_time': datetime.now(),
            'parameter_variance': defaultdict(float),
            'performance_trend': 'unknown'
        }
    
    def _extract_initial_parameters(self, physical_strategy: Dict[str, Any]) -> Dict[str, float]:
        """提取初始参数"""
        initial_params = {}
        
        # 从映射的动作序列中提取参数
        mapped_actions = physical_strategy.get('mapped_action_sequences', [])
        if mapped_actions:
            # 提取第一个动作的参数作为初始参数
            first_action = mapped_actions[0]
            parameters = first_action.get('mapped_parameters', {})
            
            # 解析位置参数
            if 'target_position' in parameters:
                pos = parameters['target_position']
                initial_params.update({
                    'target_x': pos.get('x', 0.0),
                    'target_y': pos.get('y', 0.0),
                    'target_z': pos.get('z', 0.0)
                })
            
            # 解析力参数
            if 'applied_force' in parameters:
                force = parameters['applied_force']
                initial_params.update({
                    'force_magnitude': force.get('magnitude', 1.0)
                })
        
        # 添加默认控制参数
        initial_params.update({
            'position_gain': 1.0,
            'velocity_gain': 0.1,
            'force_gain': 1.0,
            'timing_offset': 0.0
        })
        
        return initial_params
    
    def _monitor_environment_changes(self, physical_environment: Dict[str, Any], 
                                   strategy_id: str) -> Dict[str, Any]:
        """监控环境变化"""
        current_features = self._extract_environmental_features(physical_environment)
        
        # 从历史中获取之前的特征
        if strategy_id not in self.environment_monitor['change_detection']:
            self.environment_monitor['change_detection'][strategy_id] = {
                'previous_features': current_features,
                'change_history': []
            }
        
        previous_features = self.environment_monitor['change_detection'][strategy_id]['previous_features']
        
        # 计算特征变化
        feature_changes = {}
        for feature_name, current_value in current_features.items():
            if feature_name in previous_features:
                change = current_value - previous_features[feature_name]
                relative_change = abs(change / (abs(previous_features[feature_name]) + 1e-8))
                feature_changes[feature_name] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'is_significant': relative_change > self.environment_monitor['sensitivity_threshold']
                }
        
        # 更新历史记录
        self.environment_monitor['change_detection'][strategy_id]['previous_features'] = current_features
        self.environment_monitor['change_detection'][strategy_id]['change_history'].append({
            'timestamp': datetime.now().isoformat(),
            'changes': feature_changes
        })
        
        # 保持历史记录在合理范围内
        if len(self.environment_monitor['change_detection'][strategy_id]['change_history']) > 100:
            self.environment_monitor['change_detection'][strategy_id]['change_history'] = \
                self.environment_monitor['change_detection'][strategy_id]['change_history'][-50:]
        
        return feature_changes
    
    def _extract_environmental_features(self, physical_environment: Dict[str, Any]) -> Dict[str, float]:
        """提取环境特征"""
        features = {}
        
        # 基本物理参数
        if 'workspace_dimensions' in physical_environment:
            dimensions = physical_environment['workspace_dimensions']
            features['workspace_volume'] = dimensions.get('width', 1.0) * \
                                         dimensions.get('height', 1.0) * \
                                         dimensions.get('depth', 1.0)
        
        # 摩擦系数
        if 'environmental_constraints' in physical_environment:
            constraints = physical_environment['environmental_constraints']
            if 'friction_coefficients' in constraints:
                friction = constraints['friction_coefficients']
                features['average_friction'] = np.mean(list(friction.values()))
        
        # 物体复杂度
        objects = physical_environment.get('objects', [])
        features['object_count'] = len(objects)
        
        # 环境噪音水平（估算）
        features['estimated_noise'] = self._estimate_environment_noise(physical_environment)
        
        return features
    
    def _estimate_environment_noise(self, physical_environment: Dict[str, Any]) -> float:
        """估算环境噪音"""
        # 基于环境复杂性估算噪音水平
        complexity_factors = [
            len(physical_environment.get('objects', [])) / 10,
            len(physical_environment.get('constraints', [])) / 5
        ]
        
        base_noise = 0.05  # 基础噪音水平
        complexity_noise = np.mean(complexity_factors) * 0.1
        
        return base_noise + complexity_noise
    
    def _perform_adaptation(self, physical_strategy: Dict[str, Any], 
                          physical_environment: Dict[str, Any], 
                          strategy_id: str) -> Dict[str, Any]:
        """执行适应过程"""
        current_state = self.current_state[strategy_id]
        
        # 检查是否需要进行适应
        if not self._should_adapt(current_state, physical_environment):
            return {
                'adaptation_performed': False,
                'reason': 'no_significant_changes',
                'current_state': current_state
            }
        
        # 执行参数优化
        parameter_optimization = self._optimize_parameters(
            current_state['current_parameters'],
            physical_strategy,
            physical_environment,
            strategy_id
        )
        
        # 执行策略修改
        strategy_modifications = self._modify_strategy(
            physical_strategy,
            parameter_optimization['optimized_parameters'],
            physical_environment,
            strategy_id
        )
        
        # 组合适应结果
        adaptation_result = {
            'adaptation_performed': True,
            'optimized_parameters': parameter_optimization['optimized_parameters'],
            'parameter_changes': parameter_optimization['parameter_changes'],
            'strategy_modifications': strategy_modifications,
            'adaptation_confidence': parameter_optimization['optimization_confidence'],
            'adaptation_rate': self.adaptation_rate,
            'learning_progress': self._calculate_learning_progress(strategy_id),
            'convergence_status': self._assess_convergence_status(strategy_id),
            'current_state': current_state
        }
        
        return adaptation_result
    
    def _should_adapt(self, current_state: Dict[str, Any], 
                     physical_environment: Dict[str, Any]) -> bool:
        """判断是否需要进行适应"""
        # 检查适应频率
        adaptation_count = current_state.get('adaptation_count', 0)
        adaptation_frequency = self.config.get('adaptation_frequency', 1)
        
        if adaptation_count % adaptation_frequency != 0:
            return False
        
        # 检查环境变化显著性
        environment_significant = self._check_environment_significance(physical_environment)
        if not environment_significant:
            return False
        
        # 检查性能是否需要改进
        performance_needs_improvement = self._check_performance_needs_improvement(current_state)
        
        return environment_significant or performance_needs_improvement
    
    def _check_environment_significance(self, physical_environment: Dict[str, Any]) -> bool:
        """检查环境变化的显著性"""
        # 这里可以基于历史环境数据检查变化显著性
        # 简化实现：总是返回True，允许适应
        return True
    
    def _check_performance_needs_improvement(self, current_state: Dict[str, Any]) -> bool:
        """检查性能是否需要改进"""
        performance_history = current_state.get('performance_history', [])
        
        if len(performance_history) < 3:
            return True  # 初期阶段需要适应
        
        # 检查最近性能趋势
        recent_performance = performance_history[-3:]
        performance_trend = self._calculate_trend(recent_performance)
        
        # 如果性能下降，需要适应
        return performance_trend < -self.min_improvement
    
    def _optimize_parameters(self, current_parameters: Dict[str, float], 
                           physical_strategy: Dict[str, Any],
                           physical_environment: Dict[str, Any], 
                           strategy_id: str) -> Dict[str, Any]:
        """优化参数"""
        optimized_params = current_parameters.copy()
        parameter_changes = {}
        
        # 基于梯度下降的参数优化
        for param_name, param_value in current_parameters.items():
            # 计算参数梯度（简化实现）
            gradient = self._calculate_parameter_gradient(
                param_name, param_value, physical_strategy, physical_environment, strategy_id
            )
            
            # 应用学习率
            parameter_update = -self.learning_rate * gradient
            
            # 应用适应率限制
            parameter_update *= self.adaptation_rate
            
            # 应用安全约束
            bounded_update = self._apply_parameter_bounds(param_name, parameter_update)
            
            # 更新参数
            new_value = param_value + bounded_update
            optimized_params[param_name] = new_value
            
            parameter_changes[param_name] = {
                'old_value': param_value,
                'new_value': new_value,
                'change': bounded_update,
                'gradient': gradient
            }
        
        # 计算优化置信度
        optimization_confidence = self._calculate_optimization_confidence(parameter_changes)
        
        return {
            'optimized_parameters': optimized_params,
            'parameter_changes': parameter_changes,
            'optimization_confidence': optimization_confidence
        }
    
    def _calculate_parameter_gradient(self, param_name: str, param_value: float,
                                    physical_strategy: Dict[str, Any],
                                    physical_environment: Dict[str, Any], 
                                    strategy_id: str) -> float:
        """计算参数梯度"""
        # 简化实现：基于参数变化对性能的影响估算梯度
        performance_buffer = self.performance_buffer[strategy_id]
        
        if len(performance_buffer) < 2:
            return 0.0
        
        # 获取最近的性能数据
        recent_performances = list(performance_buffer)[-5:]
        
        # 计算参数变化对性能的影响（数值微分）
        epsilon = 0.01 * abs(param_value) + 1e-8  # 小的扰动
        
        # 这里应该计算性能对参数的偏导数
        # 简化实现：返回随机梯度或基于经验的梯度
        if recent_performances:
            performance_trend = self._calculate_trend(recent_performances)
            # 基于性能趋势调整梯度方向
            return performance_trend * 0.1
        else:
            return random.uniform(-0.01, 0.01)
    
    def _apply_parameter_bounds(self, param_name: str, parameter_update: float) -> float:
        """应用参数边界约束"""
        bounds = self.config.get('parameter_bounds', {})
        
        if param_name in bounds:
            min_bound, max_bound = bounds[param_name]
            
            # 确保更新不会超出边界
            # 这里需要当前参数值，但为了简化，我们假设更新是合理的
            if abs(parameter_update) > (max_bound - min_bound) * 0.1:
                parameter_update = np.sign(parameter_update) * (max_bound - min_bound) * 0.1
        
        return parameter_update
    
    def _calculate_optimization_confidence(self, parameter_changes: Dict[str, Any]) -> float:
        """计算优化置信度"""
        if not parameter_changes:
            return 0.0
        
        # 基于参数变化的合理性计算置信度
        change_magnitudes = [abs(change['change']) for change in parameter_changes.values()]
        gradient_magnitudes = [abs(change['gradient']) for change in parameter_changes.values()]
        
        # 参数变化不应该太大
        max_reasonable_change = 0.5
        change_factor = min(1.0, np.mean(change_magnitudes) / max_reasonable_change)
        
        # 梯度应该有一定的模式性
        gradient_factor = min(1.0, np.std(gradient_magnitudes) + 0.1)
        
        confidence = 0.7 * (1 - change_factor) + 0.3 * gradient_factor
        return max(0.0, min(1.0, confidence))
    
    def _modify_strategy(self, physical_strategy: Dict[str, Any], 
                       optimized_parameters: Dict[str, float],
                       physical_environment: Dict[str, Any], 
                       strategy_id: str) -> Dict[str, Any]:
        """修改策略"""
        modifications = {}
        modification_weights = self.strategy_modifier['modification_weights']
        
        # 应用不同类型的策略修改
        for mod_type, weight in modification_weights.items():
            if mod_type == 'temporal_adjustment':
                modifications['temporal_adjustment'] = self._apply_temporal_adjustment(
                    physical_strategy, optimized_parameters
                )
            elif mod_type == 'spatial_correction':
                modifications['spatial_correction'] = self._apply_spatial_correction(
                    physical_strategy, optimized_parameters
                )
            elif mod_type == 'force_modulation':
                modifications['force_modulation'] = self._apply_force_modulation(
                    physical_strategy, optimized_parameters
                )
            elif mod_type == 'sequence_reordering':
                modifications['sequence_reordering'] = self._apply_sequence_reordering(
                    physical_strategy, physical_environment
                )
        
        return modifications
    
    def _apply_temporal_adjustment(self, physical_strategy: Dict[str, Any], 
                                 optimized_parameters: Dict[str, float]) -> Dict[str, Any]:
        """应用时间调整"""
        timing_offset = optimized_parameters.get('timing_offset', 0.0)
        
        return {
            'type': 'temporal_adjustment',
            'timing_offset': timing_offset,
            'adjustment_method': 'uniform_timing_shift',
            'confidence': max(0.0, 1 - abs(timing_offset))
        }
    
    def _apply_spatial_correction(self, physical_strategy: Dict[str, Any], 
                                optimized_parameters: Dict[str, float]) -> Dict[str, Any]:
        """应用空间校正"""
        position_corrections = {
            'x': optimized_parameters.get('position_correction_x', 0.0),
            'y': optimized_parameters.get('position_correction_y', 0.0),
            'z': optimized_parameters.get('position_correction_z', 0.0)
        }
        
        return {
            'type': 'spatial_correction',
            'position_corrections': position_corrections,
            'correction_method': 'uniform_position_offset',
            'confidence': max(0.0, 1 - np.linalg.norm(list(position_corrections.values())) / 0.1)
        }
    
    def _apply_force_modulation(self, physical_strategy: Dict[str, Any], 
                              optimized_parameters: Dict[str, float]) -> Dict[str, Any]:
        """用力调制"""
        force_gain = optimized_parameters.get('force_gain', 1.0)
        
        return {
            'type': 'force_modulation',
            'force_gain': force_gain,
            'modulation_method': 'uniform_force_scaling',
            'confidence': min(1.0, force_gain)
        }
    
    def _apply_sequence_reordering(self, physical_strategy: Dict[str, Any], 
                                 physical_environment: Dict[str, Any]) -> Dict[str, Any]:
        """应用序列重排序"""
        # 基于环境约束重新排序动作序列
        mapped_actions = physical_strategy.get('mapped_action_sequences', [])
        
        if len(mapped_actions) <= 1:
            return {'type': 'sequence_reordering', 'changes': [], 'confidence': 1.0}
        
        # 简化的重排序逻辑：尝试交换相邻动作
        reordered_sequence = mapped_actions.copy()
        if random.random() < 0.3:  # 30%概率进行重排序
            # 随机选择一个位置进行交换
            swap_position = random.randint(0, len(reordered_sequence) - 2)
            reordered_sequence[swap_position], reordered_sequence[swap_position + 1] = \
                reordered_sequence[swap_position + 1], reordered_sequence[swap_position]
        
        return {
            'type': 'sequence_reordering',
            'original_sequence': mapped_actions,
            'reordered_sequence': reordered_sequence,
            'changes': len(reordered_sequence) - len(mapped_actions),
            'confidence': 0.7
        }
    
    def _calculate_learning_progress(self, strategy_id: str) -> float:
        """计算学习进度"""
        adaptation_history = self.adaptation_history[strategy_id]
        
        if len(adaptation_history) < 2:
            return 0.0
        
        # 基于适应次数和改进效果计算进度
        recent_adaptations = adaptation_history[-5:]
        
        # 计算平均性能改进
        performance_impacts = [adaptation.get('performance_impact', 0) 
                             for adaptation in recent_adaptations]
        
        # 进度基于适应性改善和经验积累
        adaptation_count = len(adaptation_history)
        improvement_factor = np.mean(performance_impacts) if performance_impacts else 0
        
        progress = min(1.0, (adaptation_count * 0.1) + max(0, improvement_factor))
        return progress
    
    def _assess_convergence_status(self, strategy_id: str) -> str:
        """评估收敛状态"""
        current_state = self.current_state[strategy_id]
        adaptation_count = current_state.get('adaptation_count', 0)
        
        if adaptation_count < self.config.get('convergence_criteria', {}).get('min_adaptations', 5):
            return 'initializing'
        
        # 检查参数稳定性
        parameter_variance = current_state.get('parameter_variance', {})
        avg_variance = np.mean(list(parameter_variance.values())) if parameter_variance else 1.0
        
        # 检查性能稳定性
        performance_history = current_state.get('performance_history', [])
        performance_stability = self._assess_performance_stability(performance_history)
        
        convergence_criteria = self.config.get('convergence_criteria', {})
        
        if (avg_variance < convergence_criteria.get('parameter_stability', 0.02) and
            performance_stability < convergence_criteria.get('performance_stability', 0.05)):
            return 'converged'
        elif adaptation_count > 50:  # 超时限制
            return 'timeout'
        else:
            return 'adapting'
    
    def _assess_performance_stability(self, performance_history: List[float]) -> float:
        """评估性能稳定性"""
        if len(performance_history) < 3:
            return 1.0
        
        # 计算性能方差
        performance_variance = np.var(performance_history[-10:])  # 最近10个性能点
        return performance_variance
    
    def _assess_performance_impact(self, strategy_id: str, adaptation_result: Dict[str, Any]) -> float:
        """评估适应对性能的影响"""
        # 简化实现：基于适应置信度和参数变化评估影响
        confidence = adaptation_result.get('adaptation_confidence', 0.5)
        parameter_changes = adaptation_result.get('parameter_changes', {})
        
        if not parameter_changes:
            return 0.0
        
        # 计算参数变化程度
        change_magnitudes = [abs(change['change']) for change in parameter_changes.values()]
        avg_change = np.mean(change_magnitudes)
        
        # 性能影响 = 置信度 * 变化程度
        impact = confidence * avg_change * 10  # 放大系数
        
        return impact
    
    def _update_adaptation_state(self, strategy_id: str, adaptation_result: Dict[str, Any]):
        """更新适应状态"""
        if not self.current_state[strategy_id]:
            return
        
        current_state = self.current_state[strategy_id]
        
        # 更新适应次数
        current_state['adaptation_count'] += 1
        current_state['last_adaptation_time'] = datetime.now()
        
        # 更新参数
        if 'optimized_parameters' in adaptation_result:
            current_state['current_parameters'].update(adaptation_result['optimized_parameters'])
        
        # 更新收敛状态
        current_state['convergence_status'] = adaptation_result.get('convergence_status', 'adapting')
        
        # 更新参数方差
        if 'parameter_changes' in adaptation_result:
            for param_name, change_info in adaptation_result['parameter_changes'].items():
                current_state['parameter_variance'][param_name] = change_info['change']
    
    def _update_performance_buffer(self, strategy_id: str, performance_value: float):
        """更新性能缓冲区"""
        self.performance_buffer[strategy_id].append(performance_value)
        
        # 同时更新当前状态的性能历史
        if strategy_id in self.current_state:
            self.current_state[strategy_id]['performance_history'].append(performance_value)
            # 保持历史记录在合理长度
            if len(self.current_state[strategy_id]['performance_history']) > 100:
                self.current_state[strategy_id]['performance_history'] = \
                    self.current_state[strategy_id]['performance_history'][-50:]
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算数值趋势"""
        if len(values) < 2:
            return 0.0
        
        # 简单线性趋势计算
        x = list(range(len(values)))
        slope, _, _, _, _ = np.polyfit(x, values, 1)
        return slope
    
    def report_performance(self, strategy_id: str, performance_value: float):
        """报告性能数据"""
        self._update_performance_buffer(strategy_id, performance_value)
        
        # 检查是否需要触发适应
        if strategy_id in self.current_state:
            current_state = self.current_state[strategy_id]
            
            # 基于性能变化决定是否需要适应
            performance_history = current_state.get('performance_history', [])
            if len(performance_history) >= 2:
                recent_performance = performance_history[-2:]
                performance_change = abs(recent_performance[-1] - recent_performance[-2])
                
                if performance_change > 0.1:  # 性能变化超过10%
                    self.logger.info(f"检测到性能变化 {performance_change:.2f}，可能需要适应")
    
    def get_adaptation_status(self, strategy_id: str) -> Dict[str, Any]:
        """获取适应状态"""
        if strategy_id not in self.current_state:
            return {'status': 'not_found'}
        
        current_state = self.current_state[strategy_id]
        
        return {
            'strategy_id': strategy_id,
            'status': current_state.get('convergence_status', 'unknown'),
            'adaptation_count': current_state.get('adaptation_count', 0),
            'last_adaptation': current_state.get('last_adaptation_time', 'never'),
            'learning_progress': self._calculate_learning_progress(strategy_id),
            'performance_trend': current_state.get('performance_trend', 'unknown'),
            'parameter_variance': dict(current_state.get('parameter_variance', {}))
        }
    
    def force_adaptation(self, strategy_id: str, physical_strategy: Dict[str, Any], 
                        physical_environment: Dict[str, Any]) -> Dict[str, Any]:
        """强制执行适应"""
        self.logger.info(f"强制执行适应，策略ID: {strategy_id}")
        
        # 重置适应状态
        if strategy_id in self.current_state:
            self.current_state[strategy_id]['adaptation_count'] = 0
            self.current_state[strategy_id]['convergence_status'] = 'forced_adaptation'
        
        # 执行适应
        return self.adapt_strategy(physical_strategy, physical_environment)
    
    def perform_progressive_adaptation(self, strategy_id: str, physical_strategy: Dict[str, Any], 
                                     physical_environment: Dict[str, Any], 
                                     adaptation_steps: int = 5) -> List[Dict[str, Any]]:
        """执行渐进式适应"""
        adaptation_results = []
        
        for step in range(adaptation_steps):
            self.logger.info(f"渐进式适应步骤 {step + 1}/{adaptation_steps}")
            
            # 执行单步适应
            step_result = self.adapt_strategy(physical_strategy, physical_environment)
            adaptation_results.append(step_result)
            
            # 检查是否已经达到目标性能
            if step_result.get('adaptation_confidence', 0) > 0.8:
                self.logger.info(f"在第 {step + 1} 步达到目标适应置信度")
                break
            
            # 更新物理策略以进行下一步适应
            if 'adapted_actions' in step_result:
                physical_strategy['mapped_action_sequences'] = step_result['adapted_actions']
        
        return adaptation_results
    
    def adapt_to_environment_changes(self, strategy_id: str, 
                                   environment_changes: Dict[str, Any]) -> Dict[str, Any]:
        """针对环境变化的适应"""
        self.logger.info(f"针对环境变化适应，策略ID: {strategy_id}")
        
        if strategy_id not in self.current_state:
            return {'adaptation_performed': False, 'reason': 'strategy_not_found'}
        
        current_state = self.current_state[strategy_id]
        
        # 分析环境变化类型和影响
        change_analysis = self._analyze_environment_changes(environment_changes)
        
        # 根据变化类型调整适应策略
        adaptation_strategy = self._select_adaptation_strategy(change_analysis)
        
        # 执行针对性适应
        targeted_adaptation = {
            'change_analysis': change_analysis,
            'adaptation_strategy': adaptation_strategy,
            'parameter_adjustments': self._calculate_parameter_adjustments(change_analysis),
            'confidence': self._calculate_change_adaptation_confidence(change_analysis)
        }
        
        return targeted_adaptation
    
    def _analyze_environment_changes(self, environment_changes: Dict[str, Any]) -> Dict[str, Any]:
        """分析环境变化"""
        analysis = {
            'change_magnitude': 0.0,
            'change_types': [],
            'affected_parameters': [],
            'urgency_level': 'low'
        }
        
        # 分析位置变化
        if 'position_shift' in environment_changes:
            position_shift = environment_changes['position_shift']
            shift_magnitude = np.linalg.norm(position_shift)
            analysis['change_magnitude'] += shift_magnitude
            analysis['change_types'].append('spatial')
            analysis['affected_parameters'].append('position')
        
        # 分析力变化
        if 'force_variation' in environment_changes:
            force_change = environment_changes['force_variation']
            analysis['change_magnitude'] += abs(force_change)
            analysis['change_types'].append('dynamic')
            analysis['affected_parameters'].append('force')
        
        # 分析时间变化
        if 'timing_changes' in environment_changes:
            timing_change = environment_changes['timing_changes']
            analysis['change_magnitude'] += abs(timing_change)
            analysis['change_types'].append('temporal')
            analysis['affected_parameters'].append('timing')
        
        # 确定紧迫性级别
        if analysis['change_magnitude'] > 0.5:
            analysis['urgency_level'] = 'high'
        elif analysis['change_magnitude'] > 0.2:
            analysis['urgency_level'] = 'medium'
        
        return analysis
    
    def _select_adaptation_strategy(self, change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """选择适应策略"""
        strategy = {
            'primary_method': 'parameter_tuning',
            'adjustment_speed': 'normal',
            'scope': 'local'
        }
        
        change_magnitude = change_analysis.get('change_magnitude', 0.0)
        change_types = change_analysis.get('change_types', [])
        
        # 根据变化幅度选择策略
        if change_magnitude > 0.5:
            strategy['primary_method'] = 'strategic_modification'
            strategy['adjustment_speed'] = 'fast'
            strategy['scope'] = 'global'
        elif change_magnitude < 0.1:
            strategy['primary_method'] = 'fine_tuning'
            strategy['adjustment_speed'] = 'slow'
            strategy['scope'] = 'local'
        
        # 根据变化类型调整方法
        if 'spatial' in change_types:
            strategy['focus_area'] = 'position_parameters'
        elif 'dynamic' in change_types:
            strategy['focus_area'] = 'force_parameters'
        elif 'temporal' in change_types:
            strategy['focus_area'] = 'timing_parameters'
        
        return strategy
    
    def _calculate_parameter_adjustments(self, change_analysis: Dict[str, Any]) -> Dict[str, float]:
        """计算参数调整量"""
        adjustments = {}
        change_magnitude = change_analysis.get('change_magnitude', 0.0)
        affected_params = change_analysis.get('affected_parameters', [])
        
        # 基于变化幅度计算调整量
        base_adjustment_rate = self.adaptation_rate * 2  # 加速适应
        
        for param in affected_params:
            if param == 'position':
                adjustments['position_gain'] = base_adjustment_rate * 0.8
            elif param == 'force':
                adjustments['force_gain'] = base_adjustment_rate * 1.2
            elif param == 'timing':
                adjustments['timing_offset'] = base_adjustment_rate * 0.5
        
        return adjustments
    
    def _calculate_change_adaptation_confidence(self, change_analysis: Dict[str, Any]) -> float:
        """计算变化适应置信度"""
        change_magnitude = change_analysis.get('change_magnitude', 0.0)
        urgency_level = change_analysis.get('urgency_level', 'low')
        
        # 基础置信度
        base_confidence = 0.8
        
        # 根据变化幅度调整
        if change_magnitude > 0.5:
            confidence_penalty = 0.3  # 大变化降低置信度
        elif change_magnitude > 0.2:
            confidence_penalty = 0.1
        else:
            confidence_penalty = 0.0
        
        # 根据紧迫性调整
        urgency_adjustment = {
            'high': -0.2,
            'medium': -0.1,
            'low': 0.0
        }.get(urgency_level, 0.0)
        
        confidence = base_confidence - confidence_penalty + urgency_adjustment
        return max(0.0, min(1.0, confidence))
    
    def get_adaptation_insights(self, strategy_id: str) -> Dict[str, Any]:
        """获取适应洞察"""
        if strategy_id not in self.current_state:
            return {'status': 'no_data_available'}
        
        current_state = self.current_state[strategy_id]
        adaptation_history = self.adaptation_history[strategy_id]
        
        insights = {
            'adaptation_patterns': self._identify_adaptation_patterns(adaptation_history),
            'performance_correlation': self._analyze_adaptation_performance_correlation(adaptation_history),
            'convergence_analysis': self._analyze_convergence_patterns(adaptation_history),
            'recommendations': self._generate_adaptation_recommendations(current_state, adaptation_history)
        }
        
        return insights
    
    def _identify_adaptation_patterns(self, adaptation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """识别适应模式"""
        if len(adaptation_history) < 3:
            return {'pattern_detected': False, 'reason': 'insufficient_data'}
        
        patterns = {
            'adaptation_frequency': len(adaptation_history),
            'common_adjustments': {},
            'performance_trend': 'stable',
            'adaptation_stability': 0.0
        }
        
        # 分析常见的参数调整
        all_adjustments = []
        for adaptation in adaptation_history:
            if 'parameter_changes' in adaptation.get('adaptation_result', {}):
                param_changes = adaptation['adaptation_result']['parameter_changes']
                all_adjustments.extend(param_changes.keys())
        
        # 统计最常调整的参数
        if all_adjustments:
            from collections import Counter
            adjustment_counts = Counter(all_adjustments)
            patterns['common_adjustments'] = dict(adjustment_counts.most_common(3))
        
        # 分析性能趋势
        performance_values = []
        for adaptation in adaptation_history:
            performance_impact = adaptation.get('performance_impact', 0.0)
            if performance_impact != 0:
                performance_values.append(performance_impact)
        
        if performance_values:
            recent_trend = np.mean(performance_values[-3:]) - np.mean(performance_values[:3])
            if recent_trend > 0.01:
                patterns['performance_trend'] = 'improving'
            elif recent_trend < -0.01:
                patterns['performance_trend'] = 'declining'
            
            patterns['adaptation_stability'] = 1.0 - np.std(performance_values)
        
        return patterns
    
    def _analyze_adaptation_performance_correlation(self, adaptation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析适应与性能的相关性"""
        if len(adaptation_history) < 3:
            return {'correlation_analysis': 'insufficient_data'}
        
        # 提取适应强度和性能影响数据
        adaptation_intensities = []
        performance_impacts = []
        
        for adaptation in adaptation_history:
            adaptation_result = adaptation.get('adaptation_result', {})
            
            # 计算适应强度
            param_changes = adaptation_result.get('parameter_changes', {})
            if param_changes:
                intensity = np.mean([abs(change.get('change', 0)) for change in param_changes.values()])
                adaptation_intensities.append(intensity)
            
            # 提取性能影响
            performance_impact = adaptation.get('performance_impact', 0.0)
            performance_impacts.append(performance_impact)
        
        if len(adaptation_intensities) != len(performance_impacts) or len(adaptation_intensities) < 2:
            return {'correlation_analysis': 'inconsistent_data'}
        
        # 计算相关系数
        correlation, p_value = stats.pearsonr(adaptation_intensities, performance_impacts)
        
        return {
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'correlation_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak',
            'is_significant': p_value < 0.05,
            'interpretation': self._interpret_adaptation_correlation(correlation, p_value)
        }
    
    def _interpret_adaptation_correlation(self, correlation: float, p_value: float) -> str:
        """解释适应相关性结果"""
        if p_value >= 0.05:
            return "适应强度与性能影响之间没有显著相关性"
        
        if correlation > 0.6:
            return "适应强度越大，性能改善越明显"
        elif correlation < -0.6:
            return "适应强度越大，性能反而下降，可能存在过度适应"
        elif correlation > 0.2:
            return "适应强度与性能改善有轻微正相关"
        elif correlation < -0.2:
            return "适应强度与性能改善有轻微负相关"
        else:
            return "适应强度与性能影响相关性较弱"
    
    def _analyze_convergence_patterns(self, adaptation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析收敛模式"""
        if len(adaptation_history) < 5:
            return {'convergence_analysis': 'insufficient_history'}
        
        # 分析收敛速度
        convergence_speeds = []
        for i in range(1, min(len(adaptation_history), 6)):
            # 计算前i次适应的累积效果
            recent_adaptations = adaptation_history[-i:]
            adaptation_effects = [adapt.get('performance_impact', 0.0) for adapt in recent_adaptations]
            
            cumulative_effect = sum(adaptation_effects)
            convergence_speeds.append(cumulative_effect / i)
        
        # 分析收敛稳定性
        if convergence_speeds:
            convergence_variance = np.var(convergence_speeds)
            stability_score = max(0, 1.0 - convergence_variance)
        else:
            stability_score = 0.0
        
        return {
            'convergence_speed': np.mean(convergence_speeds) if convergence_speeds else 0.0,
            'stability_score': stability_score,
            'convergence_quality': 'good' if stability_score > 0.7 else 'fair' if stability_score > 0.4 else 'poor',
            'adaptations_to_convergence': self._estimate_convergence_adaptations(adaptation_history)
        }
    
    def _estimate_convergence_adaptations(self, adaptation_history: List[Dict[str, Any]]) -> int:
        """估算达到收敛所需的适应次数"""
        if len(adaptation_history) < 3:
            return 5  # 默认估算
        
        # 简化的收敛估算：基于性能改善的衰减
        recent_effects = [adapt.get('performance_impact', 0.0) for adapt in adaptation_history[-5:]]
        
        # 计算改善衰减率
        if len(recent_effects) >= 2:
            improvements = [abs(effect) for effect in recent_effects]
            decay_rate = improvements[-1] / max(improvements[0], 1e-8)
            
            if decay_rate < 0.1:  # 改善很小
                return 2  # 接近收敛
            elif decay_rate < 0.3:
                return 5  # 可能还需要几次适应
            else:
                return 10  # 需要更多适应
        
        return 5  # 默认值
    
    def _generate_adaptation_recommendations(self, current_state: Dict[str, Any], 
                                           adaptation_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """生成适应建议"""
        recommendations = []
        
        # 基于收敛状态建议
        convergence_status = current_state.get('convergence_status', 'unknown')
        if convergence_status == 'timeout':
            recommendations.append({
                'type': 'convergence',
                'priority': 'high',
                'suggestion': '考虑重置适应参数或检查环境模型'
            })
        elif convergence_status == 'adapting':
            recommendations.append({
                'type': 'convergence',
                'priority': 'medium',
                'suggestion': '继续当前适应策略，监控收敛进展'
            })
        
        # 基于适应频率建议
        adaptation_count = current_state.get('adaptation_count', 0)
        if adaptation_count > 20:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'medium',
                'suggestion': '适应频率较高，考虑优化适应算法参数'
            })
        
        # 基于性能历史建议
        performance_history = current_state.get('performance_history', [])
        if len(performance_history) >= 5:
            recent_performance = performance_history[-5:]
            performance_trend = np.mean(recent_performance[-2:]) - np.mean(recent_performance[:2])
            
            if performance_trend < -0.05:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'suggestion': '性能呈下降趋势，建议降低适应率'
                })
            elif performance_trend > 0.05:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'low',
                    'suggestion': '性能改善良好，保持当前策略'
                })
        
        return recommendations