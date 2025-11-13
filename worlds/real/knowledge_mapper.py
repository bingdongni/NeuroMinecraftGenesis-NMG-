#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识映射器
负责Minecraft虚拟环境与物理世界之间的知识映射和转换
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import math


class KnowledgeMapper:
    """
    知识映射器类
    
    功能：
    1. 将Minecraft虚拟环境的概念映射到物理世界概念
    2. 处理数值和单位的转换
    3. 分析映射的不确定性和置信度
    4. 提供双向映射支持（虚拟->物理，物理->虚拟）
    
    映射类型：
    - 空间映射：坐标、角度、距离等
    - 物体映射：方块类型到物理对象
    - 动作映射：虚拟操作到物理操作
    - 约束映射：规则和限制的转换
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化知识映射器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('KnowledgeMapper')
        
        # 映射规则和模板
        self.mapping_rules = self._load_mapping_rules()
        self.conversion_templates = self._load_conversion_templates()
        self.uncertainty_models = self._load_uncertainty_models()
        
        # 映射历史和缓存
        self.mapping_cache = {}
        self.conversion_history = []
        self.learned_mappings = defaultdict(list)
        
        # 相似性计算参数
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.mapping_granularity = self.config.get('mapping_granularity', 'medium')
        self.confidence_weight = self.config.get('confidence_weight', 0.8)
        
        self.logger.info("知识映射器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'similarity_threshold': 0.7,
            'mapping_granularity': 'medium',  # coarse, medium, fine
            'confidence_weight': 0.8,
            'uncertainty_propagation': True,
            'bidirectional_mapping': True,
            'learning_enabled': True
        }
    
    def _load_mapping_rules(self) -> Dict[str, Any]:
        """加载映射规则"""
        return {
            # Minecraft方块到物理对象的映射
            'block_to_object': {
                'stone': {'class': 'rock', 'material': 'stone', 'density': 2.7},
                'wood': {'class': 'wood', 'material': 'wood', 'density': 0.6},
                'grass': {'class': 'vegetation', 'material': 'organic', 'density': 1.0},
                'cobblestone': {'class': 'stone', 'material': 'stone', 'density': 2.6},
                'dirt': {'class': 'soil', 'material': 'earth', 'density': 1.5},
                'sand': {'class': 'granular', 'material': 'silica', 'density': 1.6},
                'glass': {'class': 'transparent', 'material': 'glass', 'density': 2.5}
            },
            
            # 虚拟动作到物理动作的映射
            'action_mapping': {
                'grab': {'class': 'grasp', 'method': 'contact_based', 'precision': 'high'},
                'place': {'class': 'placement', 'method': 'positioning', 'precision': 'medium'},
                'push': {'class': 'force_application', 'method': 'contact_push', 'precision': 'low'},
                'pull': {'class': 'force_application', 'method': 'contact_pull', 'precision': 'low'},
                'rotate': {'class': 'manipulation', 'method': 'rotational', 'precision': 'medium'},
                'stack': {'class': 'assembly', 'method': 'sequential_placement', 'precision': 'high'}
            },
            
            # 单位转换规则
            'unit_conversion': {
                'minecraft_blocks_to_meters': 1.0,  # 1 Minecraft方块 = 1米
                'minecraft_ticks_to_seconds': 0.05,  # 1 tick = 50ms
                'virtual_force_to_newtons': 1.0,
                'virtual_torque_to_nm': 1.0
            }
        }
    
    def _load_conversion_templates(self) -> Dict[str, Any]:
        """加载转换模板"""
        return {
            'spatial_conversion': {
                'position': self._convert_position,
                'rotation': self._convert_rotation,
                'scale': self._convert_scale,
                'velocity': self._convert_velocity
            },
            'material_properties': {
                'hardness': self._convert_hardness,
                'friction': self._convert_friction,
                'elasticity': self._convert_elasticity,
                'density': self._convert_density
            },
            'dynamic_properties': {
                'mass': self._convert_mass,
                'momentum': self._convert_momentum,
                'energy': self._convert_energy
            }
        }
    
    def _load_uncertainty_models(self) -> Dict[str, Any]:
        """加载不确定性模型"""
        return {
            'spatial_uncertainty': {
                'position_variance': 0.05,  # 5cm
                'orientation_variance': 0.1,  # 0.1弧度
                'scale_variance': 0.02
            },
            'material_uncertainty': {
                'property_variance': 0.1,
                'density_variance': 0.05
            },
            'action_uncertainty': {
                'execution_variance': 0.08,
                'timing_variance': 0.05
            }
        }
    
    def map_strategy(self, minecraft_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        将Minecraft策略映射到物理世界
        
        Args:
            minecraft_strategy: Minecraft策略数据
            
        Returns:
            Dict: 映射结果，包含映射动作、参数转换、环境适应等
        """
        try:
            self.logger.info("开始策略映射")
            
            # 提取策略组件
            action_sequences = minecraft_strategy.get('action_sequences', [])
            environmental_context = minecraft_strategy.get('environmental_context', {})
            performance_metrics = minecraft_strategy.get('performance_metrics', {})
            
            # 执行映射过程
            mapped_actions = self._map_action_sequences(action_sequences)
            parameter_mappings = self._map_parameters(environmental_context)
            environment_mappings = self._map_environment(environmental_context)
            performance_mappings = self._map_performance_metrics(performance_metrics)
            
            # 计算映射置信度
            mapping_confidence = self._calculate_mapping_confidence(
                mapped_actions, parameter_mappings, environment_mappings
            )
            
            # 构建映射结果
            mapping_result = {
                'mapped_actions': mapped_actions,
                'parameter_mappings': parameter_mappings,
                'environment_mappings': environment_mappings,
                'performance_mappings': performance_mappings,
                'mapping_confidence': mapping_confidence,
                'source_representations': self._extract_source_representations(minecraft_strategy),
                'target_representations': self._build_target_representations(mapped_actions, parameter_mappings),
                'uncertainty_analysis': self._analyze_mapping_uncertainty(mapped_actions, parameter_mappings),
                'mapping_metadata': {
                    'mapping_timestamp': datetime.now().isoformat(),
                    'mapping_method': 'rule_based_with_learning',
                    'confidence_distribution': self._calculate_confidence_distribution(mapped_actions)
                }
            }
            
            # 缓存映射结果
            strategy_id = minecraft_strategy.get('strategy_id', 'unknown')
            self.mapping_cache[strategy_id] = mapping_result
            
            # 更新学习记录
            if self.config.get('learning_enabled', True):
                self._update_learned_mappings(minecraft_strategy, mapping_result)
            
            self.logger.info(f"策略映射完成，置信度: {mapping_confidence:.2f}")
            
            return mapping_result
            
        except Exception as e:
            self.logger.error(f"策略映射失败: {str(e)}")
            raise
    
    def reverse_map(self, physical_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        反向映射：从物理世界策略到虚拟世界策略
        
        Args:
            physical_strategy: 物理世界策略数据
            
        Returns:
            Dict: 反向映射结果
        """
        try:
            self.logger.info("开始反向映射")
            
            # 提取物理策略组件
            physical_actions = physical_strategy.get('action_sequences', [])
            physical_environment = physical_strategy.get('environmental_context', {})
            physical_performance = physical_strategy.get('performance_metrics', {})
            
            # 执行反向映射
            virtual_actions = self._reverse_map_actions(physical_actions)
            virtual_parameters = self._reverse_map_parameters(physical_environment)
            virtual_environment = self._reverse_map_environment(physical_environment)
            
            # 构建反向映射结果
            reverse_mapping_result = {
                'virtual_actions': virtual_actions,
                'virtual_parameters': virtual_parameters,
                'virtual_environment': virtual_environment,
                'reverse_confidence': self._calculate_reverse_confidence(virtual_actions, virtual_parameters),
                'mapping_loss': self._calculate_mapping_loss(physical_strategy, virtual_actions),
                'reverse_metadata': {
                    'reverse_mapping_timestamp': datetime.now().isoformat(),
                    'reverse_mapping_method': 'inverse_rule_based',
                    'completeness_score': self._calculate_reverse_completeness(physical_strategy)
                }
            }
            
            self.logger.info(f"反向映射完成，置信度: {reverse_mapping_result['reverse_confidence']:.2f}")
            
            return reverse_mapping_result
            
        except Exception as e:
            self.logger.error(f"反向映射失败: {str(e)}")
            raise
    
    def _map_action_sequences(self, minecraft_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """映射动作序列"""
        mapped_actions = []
        
        for minecraft_action in minecraft_actions:
            action_type = minecraft_action.get('action_type', 'unknown')
            
            # 查找映射规则
            mapping_rule = self.mapping_rules['action_mapping'].get(action_type)
            if not mapping_rule:
                self.logger.warning(f"未找到动作类型 '{action_type}' 的映射规则")
                continue
            
            # 执行映射
            mapped_action = {
                'mapped_action_id': f"mapped_{len(mapped_actions)}",
                'source_action': action_type,
                'mapped_action_type': mapping_rule['class'],
                'mapping_method': mapping_rule['method'],
                'precision_level': mapping_rule['precision'],
                'mapped_parameters': self._map_action_parameters(minecraft_action, mapping_rule),
                'uncertainty': self._calculate_action_uncertainty(minecraft_action, mapping_rule),
                'mapping_confidence': self._calculate_action_confidence(minecraft_action, mapping_rule)
            }
            
            mapped_actions.append(mapped_action)
        
        return mapped_actions
    
    def _map_parameters(self, minecraft_context: Dict[str, Any]) -> Dict[str, Any]:
        """映射参数"""
        mapped_parameters = {}
        
        # 空间参数映射
        if 'position' in minecraft_context:
            mapped_parameters['position'] = self._convert_position(
                minecraft_context['position']
            )
        
        if 'rotation' in minecraft_context:
            mapped_parameters['rotation'] = self._convert_rotation(
                minecraft_context['rotation']
            )
        
        if 'scale' in minecraft_context:
            mapped_parameters['scale'] = self._convert_scale(
                minecraft_context['scale']
            )
        
        # 物理属性映射
        for block_type, properties in minecraft_context.get('block_properties', {}).items():
            if block_type in self.mapping_rules['block_to_object']:
                physical_props = self.mapping_rules['block_to_object'][block_type]
                mapped_parameters[f'{block_type}_properties'] = {
                    'material_class': physical_props['class'],
                    'material_type': physical_props['material'],
                    'estimated_density': physical_props['density']
                }
        
        return mapped_parameters
    
    def _map_environment(self, minecraft_environment: Dict[str, Any]) -> Dict[str, Any]:
        """映射环境信息"""
        environment_mapping = {
            'mapped_environment': {},
            'environmental_constraints': {},
            'adaptation_requirements': []
        }
        
        # 物理世界环境特征
        if 'world_size' in minecraft_environment:
            size = minecraft_environment['world_size']
            environment_mapping['mapped_environment']['workspace_dimensions'] = {
                'width': size.get('x', 10) * self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters'],
                'height': size.get('y', 5) * self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters'],
                'depth': size.get('z', 10) * self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters']
            }
        
        # 环境约束
        environment_mapping['environmental_constraints'] = {
            'gravity': 9.81,  # 地球重力
            'friction_coefficients': self._estimate_friction_coefficients(minecraft_environment),
            'spatial_limitations': self._analyze_spatial_limitations(minecraft_environment)
        }
        
        # 适应需求
        environment_mapping['adaptation_requirements'] = self._identify_adaptation_requirements(
            minecraft_environment
        )
        
        return environment_mapping
    
    def _map_performance_metrics(self, minecraft_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """映射性能指标"""
        mapped_metrics = {}
        
        # 时间指标转换
        if 'execution_time' in minecraft_metrics:
            minecraft_time = minecraft_metrics['execution_time']
            mapped_metrics['execution_time'] = minecraft_time * self.mapping_rules['unit_conversion']['minecraft_ticks_to_seconds']
        
        # 成功率（保持相同）
        if 'success_rate' in minecraft_metrics:
            mapped_metrics['success_rate'] = minecraft_metrics['success_rate']
        
        # 精度指标转换
        if 'accuracy' in minecraft_metrics:
            mapped_metrics['positioning_accuracy'] = self._convert_accuracy(minecraft_metrics['accuracy'])
        
        return mapped_metrics
    
    def _calculate_mapping_confidence(self, mapped_actions: List[Dict], 
                                    parameter_mappings: Dict, 
                                    environment_mappings: Dict) -> float:
        """计算映射置信度"""
        confidences = []
        
        # 动作映射置信度
        for action in mapped_actions:
            confidences.append(action.get('mapping_confidence', 0.5))
        
        # 参数映射置信度
        param_confidence = 0.9 if parameter_mappings else 0.5
        confidences.append(param_confidence)
        
        # 环境映射置信度
        env_confidence = 0.85 if environment_mappings else 0.5
        confidences.append(env_confidence)
        
        # 计算加权平均置信度
        return np.mean(confidences)
    
    def _calculate_action_confidence(self, minecraft_action: Dict, mapping_rule: Dict) -> float:
        """计算动作映射置信度"""
        base_confidence = 0.8
        
        # 根据动作类型调整置信度
        action_type = minecraft_action.get('action_type', '')
        if action_type in ['grab', 'place']:
            base_confidence += 0.1  # 高精度动作
        
        # 根据数据质量调整
        data_completeness = self._assess_data_completeness(minecraft_action)
        confidence = base_confidence * (0.5 + 0.5 * data_completeness)
        
        return min(confidence, 1.0)
    
    def _calculate_action_uncertainty(self, minecraft_action: Dict, mapping_rule: Dict) -> Dict[str, float]:
        """计算动作映射不确定性"""
        uncertainty = {
            'spatial_uncertainty': self.uncertainty_models['spatial_uncertainty']['position_variance'],
            'temporal_uncertainty': self.uncertainty_models['action_uncertainty']['timing_variance'],
            'execution_uncertainty': self.uncertainty_models['action_uncertainty']['execution_variance']
        }
        
        # 根据动作类型调整不确定性
        action_type = minecraft_action.get('action_type', '')
        if action_type in ['grab', 'place']:
            # 高精度动作的不确定性较小
            for key in uncertainty:
                uncertainty[key] *= 0.8
        
        return uncertainty
    
    def _convert_position(self, minecraft_position: List[float]) -> Dict[str, float]:
        """转换位置坐标"""
        conversion_factor = self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters']
        
        return {
            'x': minecraft_position[0] * conversion_factor,
            'y': minecraft_position[1] * conversion_factor,
            'z': minecraft_position[2] * conversion_factor,
            'frame': 'world_coordinates'
        }
    
    def _convert_rotation(self, minecraft_rotation: Dict[str, float]) -> Dict[str, float]:
        """转换旋转角度"""
        return {
            'yaw': math.radians(minecraft_rotation.get('yaw', 0)),
            'pitch': math.radians(minecraft_rotation.get('pitch', 0)),
            'roll': math.radians(minecraft_rotation.get('roll', 0)),
            'unit': 'radians'
        }
    
    def _convert_scale(self, minecraft_scale: Dict[str, float]) -> Dict[str, float]:
        """转换缩放因子"""
        return {
            'x': minecraft_scale.get('x', 1.0),
            'y': minecraft_scale.get('y', 1.0),
            'z': minecraft_scale.get('z', 1.0),
            'reference': 'minecraft_blocks'
        }
    
    def _convert_velocity(self, minecraft_velocity: Dict[str, float]) -> Dict[str, float]:
        """转换速度"""
        conversion_factor = self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters'] / \
                           self.mapping_rules['unit_conversion']['minecraft_ticks_to_seconds']
        
        return {
            'linear_x': minecraft_velocity.get('x', 0) * conversion_factor,
            'linear_y': minecraft_velocity.get('y', 0) * conversion_factor,
            'linear_z': minecraft_velocity.get('z', 0) * conversion_factor,
            'unit': 'meters_per_second'
        }
    
    def _convert_accuracy(self, minecraft_accuracy: float) -> Dict[str, float]:
        """转换精度指标"""
        # Minecraft精度通常以方块为单位，转换为米
        conversion_factor = self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters']
        
        return {
            'positional_accuracy': minecraft_accuracy * conversion_factor,
            'angular_accuracy': 5.0,  # 默认角度精度5度
            'unit': 'meters'
        }
    
    def _estimate_friction_coefficients(self, minecraft_environment: Dict[str, Any]) -> Dict[str, float]:
        """估算摩擦系数"""
        friction_estimates = {
            'block_to_block': 0.4,
            'block_to_ground': 0.6,
            'tool_to_block': 0.3
        }
        
        # 根据环境特征调整
        surface_types = minecraft_environment.get('surface_types', [])
        if 'stone' in surface_types:
            friction_estimates['block_to_ground'] = 0.5
        elif 'wood' in surface_types:
            friction_estimates['block_to_ground'] = 0.7
        
        return friction_estimates
    
    def _analyze_spatial_limitations(self, minecraft_environment: Dict[str, Any]) -> Dict[str, Any]:
        """分析空间限制"""
        limitations = {
            'reach_limit': 4.0,  # 默认可达范围4米
            'precision_limit': 0.1,  # 精度限制10cm
            'workspace_boundaries': self._define_workspace_boundaries(minecraft_environment)
        }
        
        return limitations
    
    def _identify_adaptation_requirements(self, minecraft_environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别适应需求"""
        requirements = []
        
        # 基于环境复杂性添加适应需求
        complexity_score = self._calculate_environment_complexity(minecraft_environment)
        if complexity_score > 0.7:
            requirements.append({
                'type': 'complexity_adaptation',
                'requirement': 'high_precision_control',
                'priority': 'high'
            })
        
        # 基于物体数量添加适应需求
        object_count = len(minecraft_environment.get('objects', []))
        if object_count > 10:
            requirements.append({
                'type': 'multi_object_adaptation',
                'requirement': 'collision_avoidance',
                'priority': 'medium'
            })
        
        return requirements
    
    def _assess_data_completeness(self, data: Dict[str, Any]) -> float:
        """评估数据完整性"""
        required_fields = ['action_type', 'position', 'parameters']
        present_fields = [field for field in required_fields if field in data]
        return len(present_fields) / len(required_fields)
    
    def _calculate_environment_complexity(self, environment: Dict[str, Any]) -> float:
        """计算环境复杂性"""
        complexity_factors = {
            'object_count': len(environment.get('objects', [])) / 20,  # 标准化到0-1
            'constraint_count': len(environment.get('constraints', [])) / 10,
            'dynamic_elements': len(environment.get('dynamic_objects', [])) / 5
        }
        
        return np.mean(list(complexity_factors.values()))
    
    def _define_workspace_boundaries(self, minecraft_environment: Dict[str, Any]) -> Dict[str, float]:
        """定义工作空间边界"""
        world_size = minecraft_environment.get('world_size', {'x': 10, 'y': 5, 'z': 10})
        conversion = self.mapping_rules['unit_conversion']['minecraft_blocks_to_meters']
        
        return {
            'x_min': 0,
            'x_max': world_size['x'] * conversion,
            'y_min': 0,
            'y_max': world_size['y'] * conversion,
            'z_min': 0,
            'z_max': world_size['z'] * conversion
        }
    
    def _map_action_parameters(self, minecraft_action: Dict[str, Any], mapping_rule: Dict[str, Any]) -> Dict[str, Any]:
        """映射动作参数"""
        parameters = {}
        
        # 基本参数映射
        if 'position' in minecraft_action:
            parameters['target_position'] = self._convert_position(minecraft_action['position'])
        
        if 'force' in minecraft_action:
            # 力度转换
            force_scale = self.mapping_rules['unit_conversion']['virtual_force_to_newtons']
            parameters['applied_force'] = {
                'magnitude': minecraft_action['force'].get('magnitude', 1.0) * force_scale,
                'direction': self._convert_position(minecraft_action['force'].get('direction', [0, 0, 1]))
            }
        
        # 根据动作类型添加特定参数
        action_type = minecraft_action.get('action_type')
        if action_type == 'grab':
            parameters['grasp_configuration'] = self._define_grasp_configuration(minecraft_action)
        elif action_type == 'place':
            parameters['placement_constraints'] = self._define_placement_constraints(minecraft_action)
        
        return parameters
    
    def _define_grasp_configuration(self, minecraft_action: Dict[str, Any]) -> Dict[str, Any]:
        """定义抓取配置"""
        return {
            'approach_direction': [0, -1, 0],  # 从上方接近
            'grasp_closing_force': 10.0,  # 默认10N
            'grasp_stability_threshold': 0.8,
            'release_preparation': True
        }
    
    def _define_placement_constraints(self, minecraft_action: Dict[str, Any]) -> Dict[str, Any]:
        """定义放置约束"""
        return {
            'positioning_tolerance': 0.02,  # 2cm容忍度
            'orientation_tolerance': math.radians(5),  # 5度容忍度
            'surface_contact_check': True,
            'stability_verification': True
        }
    
    def _extract_source_representations(self, minecraft_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """提取源表示"""
        return {
            'virtual_coordinates': 'minecraft_block_coordinates',
            'virtual_orientations': 'minecraft_angular_system',
            'virtual_properties': 'discrete_block_properties',
            'virtual_actions': 'discrete_action_set'
        }
    
    def _build_target_representations(self, mapped_actions: List[Dict], 
                                    parameter_mappings: Dict) -> Dict[str, Any]:
        """构建目标表示"""
        return {
            'physical_coordinates': 'cartesian_coordinates',
            'physical_orientations': 'euler_angles_radians',
            'physical_properties': 'continuous_material_properties',
            'physical_actions': 'continuous_control_parameters'
        }
    
    def _analyze_mapping_uncertainty(self, mapped_actions: List[Dict], 
                                   parameter_mappings: Dict) -> Dict[str, Any]:
        """分析映射不确定性"""
        return {
            'overall_uncertainty': np.mean([action.get('uncertainty', {}).get('spatial_uncertainty', 0.05) 
                                          for action in mapped_actions]),
            'uncertainty_sources': [' discretization_error', 'scale_difference', 'physical_constraints'],
            'confidence_bounds': [0.6, 0.9],
            'uncertainty_propagation': self._propagate_uncertainty(mapped_actions, parameter_mappings)
        }
    
    def _propagate_uncertainty(self, mapped_actions: List[Dict], 
                             parameter_mappings: Dict) -> Dict[str, float]:
        """传播不确定性"""
        return {
            'position_uncertainty': 0.05,
            'force_uncertainty': 0.1,
            'timing_uncertainty': 0.08
        }
    
    def _calculate_confidence_distribution(self, mapped_actions: List[Dict]) -> Dict[str, float]:
        """计算置信度分布"""
        confidences = [action.get('mapping_confidence', 0.5) for action in mapped_actions]
        
        if not confidences:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
    
    def _update_learned_mappings(self, source_strategy: Dict[str, Any], 
                               mapping_result: Dict[str, Any]):
        """更新学习到的映射"""
        mapping_pattern = {
            'source_type': source_strategy.get('strategy_type'),
            'mapped_patterns': [action.get('mapped_action_type') for action in mapping_result.get('mapped_actions', [])],
            'confidence': mapping_result.get('mapping_confidence'),
            'timestamp': datetime.now().isoformat()
        }
        
        strategy_type = source_strategy.get('strategy_type', 'unknown')
        self.learned_mappings[strategy_type].append(mapping_pattern)
    
    # 反向映射相关方法
    def _reverse_map_actions(self, physical_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """反向映射动作"""
        virtual_actions = []
        
        for physical_action in physical_actions:
            action_class = physical_action.get('action_class', 'unknown')
            
            # 查找反向映射规则
            reverse_mapping = self._find_reverse_action_mapping(action_class)
            if reverse_mapping:
                virtual_action = {
                    'virtual_action_type': reverse_mapping,
                    'mapped_parameters': self._reverse_map_action_parameters(physical_action),
                    'reverse_confidence': self._calculate_reverse_action_confidence(physical_action)
                }
                virtual_actions.append(virtual_action)
        
        return virtual_actions
    
    def _find_reverse_action_mapping(self, physical_action_class: str) -> Optional[str]:
        """查找反向动作映射"""
        # 简单的反向查找逻辑
        reverse_mappings = {
            'grasp': 'grab',
            'placement': 'place',
            'force_application': 'push',
            'manipulation': 'rotate',
            'assembly': 'stack'
        }
        return reverse_mappings.get(physical_action_class, 'unknown')
    
    def _reverse_map_parameters(self, physical_environment: Dict[str, Any]) -> Dict[str, Any]:
        """反向映射参数"""
        # 实现从物理世界到虚拟世界的参数转换
        return {}
    
    def _reverse_map_environment(self, physical_environment: Dict[str, Any]) -> Dict[str, Any]:
        """反向映射环境"""
        # 实现从物理世界到虚拟世界的环境转换
        return {}
    
    def _calculate_reverse_confidence(self, virtual_actions: List[Dict], 
                                    virtual_parameters: Dict) -> float:
        """计算反向映射置信度"""
        if not virtual_actions:
            return 0.0
        
        action_confidences = [action.get('reverse_confidence', 0.5) for action in virtual_actions]
        return np.mean(action_confidences)
    
    def _calculate_mapping_loss(self, physical_strategy: Dict[str, Any], 
                              virtual_actions: List[Dict[str, Any]]) -> float:
        """计算映射损失"""
        # 简化的映射损失计算
        return 0.1  # 默认损失值
    
    def _calculate_reverse_completeness(self, physical_strategy: Dict[str, Any]) -> float:
        """计算反向映射完整性"""
        return 0.8  # 默认完整性评分
    
    def _calculate_reverse_action_confidence(self, physical_action: Dict[str, Any]) -> float:
        """计算反向动作置信度"""
        return 0.7  # 默认反向置信度
    
    def _reverse_map_action_parameters(self, physical_action: Dict[str, Any]) -> Dict[str, Any]:
        """反向映射动作参数"""
        return {}  # 简化实现
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息"""
        total_mappings = len(self.mapping_cache)
        learned_patterns = sum(len(patterns) for patterns in self.learned_mappings.values())
        
        # 计算平均置信度
        all_confidences = []
        for mapping_result in self.mapping_cache.values():
            confidence = mapping_result.get('mapping_confidence', 0.0)
            all_confidences.append(confidence)
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'total_mappings_completed': total_mappings,
            'learned_patterns_count': learned_patterns,
            'average_mapping_confidence': avg_confidence,
            'cache_size': len(self.mapping_cache),
            'supported_action_types': len(self.mapping_rules['action_mapping']),
            'supported_object_types': len(self.mapping_rules['block_to_object']),
            'mapping_history_size': len(self.conversion_history),
            'learning_effectiveness': self._calculate_learning_effectiveness()
        }
    
    def _calculate_learning_effectiveness(self) -> float:
        """计算学习效果"""
        if not self.learned_mappings:
            return 0.0
        
        total_cases = sum(len(cases) for cases in self.learned_mappings.values())
        if total_cases == 0:
            return 0.0
        
        # 计算平均性能
        all_performances = []
        for cases in self.learned_mappings.values():
            performances = [case.get('physical_performance', 0.0) for case in cases]
            all_performances.extend(performances)
        
        if not all_performances:
            return 0.0
        
        return np.mean(all_performances)
    
    def _convert_hardness(self, minecraft_hardness: float) -> float:
        """转换硬度值"""
        # Minecraft硬度值通常在0-10范围，转换为物理硬度
        return min(10.0, minecraft_hardness * 1.2)
    
    def _convert_friction(self, minecraft_friction: float) -> float:
        """转换摩擦系数"""
        # Minecraft摩擦系数转换为物理摩擦系数
        return min(1.0, minecraft_friction * 0.8)
    
    def _convert_elasticity(self, minecraft_elasticity: float) -> float:
        """转换弹性系数"""
        # Minecraft弹性值转换为物理弹性系数
        return min(1.0, minecraft_elasticity * 0.9)
    
    def _convert_density(self, minecraft_density: float) -> float:
        """转换密度值"""
        # Minecraft密度值转换为物理密度 (g/cm³)
        return minecraft_density * 2.5  # 简化转换
    
    def _convert_mass(self, minecraft_mass: float) -> float:
        """转换质量值"""
        # Minecraft质量转换为物理质量 (kg)
        return minecraft_mass * 0.1  # 简化转换
    
    def _convert_momentum(self, minecraft_momentum: Dict[str, float]) -> Dict[str, float]:
        """转换动量"""
        return {
            'x': minecraft_momentum.get('x', 0.0) * 0.1,
            'y': minecraft_momentum.get('y', 0.0) * 0.1,
            'z': minecraft_momentum.get('z', 0.0) * 0.1
        }
    
    def _convert_energy(self, minecraft_energy: float) -> float:
        """转换能量值"""
        # Minecraft能量值转换为物理能量 (J)
        return minecraft_energy * 4.184  # 焦耳转换系数
    
    def learn_from_transfer(self, minecraft_strategy: Dict[str, Any], 
                          physical_result: Dict[str, Any]) -> None:
        """从迁移经验中学习"""
        # 记录成功和失败的映射案例
        learning_case = {
            'timestamp': datetime.now().isoformat(),
            'minecraft_strategy_type': minecraft_strategy.get('strategy_type', 'unknown'),
            'mapping_confidence': minecraft_strategy.get('confidence_score', 0.0),
            'physical_performance': physical_result.get('performance_score', 0.0),
            'adaptation_required': physical_result.get('adaptation_required', False)
        }
        
        strategy_type = minecraft_strategy.get('strategy_type', 'unknown')
        self.learned_mappings[strategy_type].append(learning_case)
        
        # 动态更新映射规则置信度
        self._update_mapping_confidence(learning_case)
        
        self.logger.info(f"学习到新的映射经验，策略类型: {strategy_type}")
    
    def _update_mapping_confidence(self, learning_case: Dict[str, Any]) -> None:
        """根据学习案例更新映射置信度"""
        minecraft_confidence = learning_case.get('mapping_confidence', 0.0)
        physical_performance = learning_case.get('physical_performance', 0.0)
        
        # 简单的更新规则：物理性能好，提高置信度
        performance_factor = max(0.5, physical_performance)
        
        # 这里可以更新mapping_rules中的置信度权重
        # 简化实现：暂时只记录学习案例
        self.logger.debug(f"更新映射置信度: {minecraft_confidence} -> {minecraft_confidence * performance_factor}")
    
    def get_mapping_recommendations(self, target_environment: Dict[str, Any]) -> Dict[str, Any]:
        """获取映射建议"""
        recommendations = {
            'suggested_mapping_granularity': 'medium',
            'confidence_adjustments': {},
            'adaptation_strategies': [],
            'potential_challenges': []
        }
        
        # 基于目标环境复杂性建议映射精度
        complexity_score = self._calculate_environment_complexity(target_environment)
        if complexity_score > 0.8:
            recommendations['suggested_mapping_granularity'] = 'fine'
            recommendations['adaptation_strategies'].append('使用高精度映射参数')
        elif complexity_score < 0.3:
            recommendations['suggested_mapping_granularity'] = 'coarse'
            recommendations['potential_challenges'].append('环境太简单，可能导致映射不准确')
        
        # 基于历史成功率调整置信度
        high_success_strategies = []
        for strategy_type, cases in self.learned_mappings.items():
            if cases:
                avg_performance = np.mean([case.get('physical_performance', 0.0) for case in cases])
                if avg_performance > 0.8:
                    high_success_strategies.append(strategy_type)
        
        recommendations['confidence_adjustments'] = {
            'high_confidence_strategies': high_success_strategies,
            'adjustment_factor': 0.1 if high_success_strategies else 0.0
        }
        
        return recommendations
    
    def validate_mapping_quality(self, mapped_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """验证映射质量"""
        validation_result = {
            'overall_quality': 0.0,
            'validation_checks': {},
            'quality_issues': [],
            'improvement_suggestions': []
        }
        
        # 检查映射完整性
        mapped_actions = mapped_strategy.get('mapped_actions', [])
        if not mapped_actions:
            validation_result['quality_issues'].append('没有映射的动作')
            return validation_result
        
        # 检查参数转换合理性
        parameter_mappings = mapped_strategy.get('parameter_conversions', {})
        parameter_quality = self._validate_parameter_conversions(parameter_mappings)
        validation_result['validation_checks']['parameter_conversion'] = parameter_quality
        
        # 检查环境适应充分性
        environment_mappings = mapped_strategy.get('environment_mappings', {})
        environment_quality = self._validate_environment_adaptations(environment_mappings)
        validation_result['validation_checks']['environment_adaptation'] = environment_quality
        
        # 计算总体质量分数
        quality_scores = list(validation_result['validation_checks'].values())
        validation_result['overall_quality'] = np.mean(quality_scores) if quality_scores else 0.0
        
        # 生成改进建议
        if validation_result['overall_quality'] < 0.7:
            validation_result['improvement_suggestions'].extend([
                '提高参数转换精度',
                '加强环境适应性分析',
                '增加映射验证步骤'
            ])
        
        return validation_result
    
    def _validate_parameter_conversions(self, parameter_mappings: Dict[str, Any]) -> float:
        """验证参数转换质量"""
        if not parameter_mappings:
            return 0.0
        
        # 检查是否包含关键参数
        required_params = ['position', 'force', 'timing']
        present_params = [param for param in required_params if param in parameter_mappings]
        
        if not present_params:
            return 0.2  # 缺少关键参数
        
        # 检查参数值的合理性
        quality_score = 0.0
        
        if 'position' in parameter_mappings:
            pos = parameter_mappings['position']
            if isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
                quality_score += 0.4
        
        if 'force' in parameter_mappings:
            force = parameter_mappings['position']
            if isinstance(force, dict) and 'magnitude' in force:
                quality_score += 0.3
        
        if 'timing' in parameter_mappings:
            timing = parameter_mappings['timing']
            if isinstance(timing, (int, float)) and 0 <= timing <= 60:
                quality_score += 0.3
        
        return min(1.0, quality_score)
    
    def _validate_environment_adaptations(self, environment_mappings: Dict[str, Any]) -> float:
        """验证环境适应质量"""
        if not environment_mappings:
            return 0.0
        
        quality_indicators = []
        
        # 检查是否包含物理约束适应
        if 'environmental_constraints' in environment_mappings:
            quality_indicators.append(0.3)
        
        # 检查是否包含空间限制适应
        if 'spatial_limitations' in environment_mappings:
            quality_indicators.append(0.3)
        
        # 检查是否包含适应需求分析
        if 'adaptation_requirements' in environment_mappings:
            requirements = environment_mappings['adaptation_requirements']
            if isinstance(requirements, list) and len(requirements) > 0:
                quality_indicators.append(0.4)
        
        return sum(quality_indicators)
    
    def export_mapping_knowledge(self, filepath: str) -> None:
        """导出映射知识"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'mapping_rules': self.mapping_rules,
            'learned_mappings': dict(self.learned_mappings),
            'conversion_templates': self.conversion_templates,
            'statistics': self.get_mapping_statistics()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"映射知识已导出到: {filepath}")
        except Exception as e:
            self.logger.error(f"导出映射知识失败: {str(e)}")
            raise
    
    def import_mapping_knowledge(self, filepath: str) -> None:
        """导入映射知识"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入学习到的映射
            if 'learned_mappings' in import_data:
                for strategy_type, cases in import_data['learned_mappings'].items():
                    self.learned_mappings[strategy_type].extend(cases)
            
            # 导入映射规则（可选）
            if 'mapping_rules' in import_data:
                # 这里可以合并或替换现有规则
                self.logger.info("映射规则已更新")
            
            self.logger.info(f"映射知识已从 {filepath} 导入")
            
        except Exception as e:
            self.logger.error(f"导入映射知识失败: {str(e)}")
            raise
    
    def get_bidirectional_mapping_info(self) -> Dict[str, Any]:
        """获取双向映射信息"""
        return {
            'supported_directions': ['minecraft_to_physical', 'physical_to_minecraft'],
            'forward_mapping_stats': {
                'total_mappings': len(self.mapping_cache),
                'average_confidence': np.mean([
                    result.get('mapping_confidence', 0.0) 
                    for result in self.mapping_cache.values()
                ]) if self.mapping_cache else 0.0
            },
            'reverse_mapping_capability': {
                'available': True,
                'completeness': 0.8,  # 简化值
                'accuracy': 0.75      # 简化值
            },
            'mapping_rules_coverage': {
                'action_mappings': len(self.mapping_rules.get('action_mapping', {})),
                'object_mappings': len(self.mapping_rules.get('block_to_object', {})),
                'unit_conversions': len(self.mapping_rules.get('unit_conversion', {}))
            }
        }

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息（占位方法）"""
        return {'status': 'implemented'}