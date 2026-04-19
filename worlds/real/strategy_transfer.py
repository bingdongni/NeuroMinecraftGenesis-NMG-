#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略迁移主类
负责管理从Minecraft到物理世界的策略迁移流程
"""

import json
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import pickle
try:
    # 尝试相对导入
    from .knowledge_mapper import KnowledgeMapper
    from .transfer_evaluator import TransferEvaluator
    from .adaptation_engine import AdaptationEngine
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from knowledge_mapper import KnowledgeMapper
    from transfer_evaluator import TransferEvaluator
    from adaptation_engine import AdaptationEngine
    from performance_analyzer import PerformanceAnalyzer


class StrategyTransfer:
    """
    策略迁移主类
    
    功能：
    1. 管理从Minecraft虚拟环境到物理世界的策略迁移
    2. 协调各个组件进行策略提取、映射、适应和评估
    3. 实时监控迁移效果并进行优化调整
    
    迁移流程：
    Minecraft策略提取 -> 知识映射 -> 策略适应 -> 效果评估 -> 性能优化
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化策略迁移系统
        
        Args:
            config: 配置参数，包含各组件的参数设置
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        
        # 初始化各个核心组件
        self.knowledge_mapper = KnowledgeMapper(self.config.get('knowledge_mapper', {}))
        self.transfer_evaluator = TransferEvaluator(self.config.get('transfer_evaluator', {}))
        self.adaptation_engine = AdaptationEngine(self.config.get('adaptation_engine', {}))
        self.performance_analyzer = PerformanceAnalyzer(self.config.get('performance_analyzer', {}))
        
        # 迁移状态和历史记录
        self.transfer_history = []
        self.current_strategy = None
        self.adaptation_state = {}
        self.performance_metrics = {}
        
        # 迁移会话管理
        self.active_sessions = {}
        self.session_counter = 0
        
        self.logger.info("策略迁移系统初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置参数"""
        return {
            'knowledge_mapper': {
                'similarity_threshold': 0.7,
                'mapping_granularity': 'medium',
                'confidence_weight': 0.8
            },
            'transfer_evaluator': {
                'evaluation_metrics': ['accuracy', 'success_rate', 'execution_time'],
                'baseline_comparison': True,
                'statistical_significance': 0.05
            },
            'adaptation_engine': {
                'adaptation_rate': 0.1,
                'learning_rate': 0.01,
                'adaptation_patience': 10,
                'min_improvement': 0.001
            },
            'performance_analyzer': {
                'window_size': 100,
                'analysis_frequency': 10,
                'alert_threshold': 0.8
            },
            'logging': {
                'level': 'INFO',
                'save_history': True,
                'history_file': 'transfer_history.json'
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('StrategyTransfer')
        logger.setLevel(getattr(logging, self.config.get('logging', {}).get('level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_transfer_session(self, session_id: str = None) -> str:
        """
        开始一个新的策略迁移会话
        
        Args:
            session_id: 会话ID，如果为None则自动生成
            
        Returns:
            str: 会话ID
        """
        if session_id is None:
            session_id = f"transfer_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.session_counter += 1
        
        session_data = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'status': 'initialized',
            'strategies_processed': [],
            'adaptation_history': [],
            'evaluation_results': []
        }
        
        self.active_sessions[session_id] = session_data
        self.logger.info(f"开始策略迁移会话: {session_id}")
        
        return session_id
    
    def extract_minecraft_strategy(self, minecraft_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        从Minecraft环境中提取策略信息
        
        Args:
            minecraft_data: Minecraft环境数据，包含：
                - 场景信息：方块类型、位置、环境状态
                - 动作序列：玩家执行的操作序列
                - 状态变化：环境状态的改变记录
                - 性能数据：成功率、执行时间等指标
            session_id: 当前会话ID
            
        Returns:
            Dict: 提取的策略信息，包含：
                - 策略类型：抓取、放置、搬运等
                - 策略参数：具体的参数配置
                - 适用场景：适用的环境条件
                - 成功模式：成功的执行模式
                - 约束条件：执行约束和限制
        """
        try:
            self.logger.info(f"开始提取Minecraft策略，会话: {session_id}")
            
            # 验证输入数据
            if not self._validate_minecraft_data(minecraft_data):
                raise ValueError("Minecraft数据格式不正确")
            
            # 提取策略组件
            strategy_patterns = self._extract_strategy_patterns(minecraft_data)
            action_sequences = self._extract_action_sequences(minecraft_data)
            environmental_context = self._extract_environmental_context(minecraft_data)
            performance_metrics = self._extract_performance_metrics(minecraft_data)
            
            # 构建策略表示
            extracted_strategy = {
                'strategy_id': f"minecraft_strategy_{datetime.now().timestamp()}",
                'strategy_type': self._classify_strategy_type(action_sequences),
                'source_environment': 'minecraft',
                'action_sequences': action_sequences,
                'environmental_context': environmental_context,
                'performance_metrics': performance_metrics,
                'strategy_patterns': strategy_patterns,
                'constraints': self._extract_constraints(minecraft_data),
                'extraction_time': datetime.now().isoformat(),
                'confidence_score': self._calculate_extraction_confidence(minecraft_data)
            }
            
            # 更新会话数据
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['strategies_processed'].append(extracted_strategy['strategy_id'])
                self.active_sessions[session_id]['status'] = 'strategy_extracted'
            
            self.logger.info(f"策略提取完成，类型: {extracted_strategy['strategy_type']}, 置信度: {extracted_strategy['confidence_score']:.2f}")
            
            return extracted_strategy
            
        except Exception as e:
            self.logger.error(f"策略提取失败: {str(e)}")
            raise
    
    def map_to_physical_world(self, minecraft_strategy: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        将Minecraft策略映射到物理世界
        
        Args:
            minecraft_strategy: 提取的Minecraft策略
            session_id: 当前会话ID
            
        Returns:
            Dict: 物理世界映射策略，包含：
                - 映射的动作序列：物理世界中的执行动作
                - 参数转换：数值和单位的转换
                - 环境适应：适应物理世界的环境条件
                - 映射置信度：映射的可靠性评分
        """
        try:
            self.logger.info(f"开始策略映射到物理世界，会话: {session_id}")
            
            # 使用知识映射器进行映射
            mapping_result = self.knowledge_mapper.map_strategy(minecraft_strategy)
            
            # 构建映射策略
            physical_strategy = {
                'mapped_strategy_id': f"physical_strategy_{datetime.now().timestamp()}",
                'source_minecraft_strategy': minecraft_strategy['strategy_id'],
                'mapped_action_sequences': mapping_result['mapped_actions'],
                'parameter_conversions': mapping_result['parameter_mappings'],
                'environmental_adaptations': mapping_result['environment_mappings'],
                'confidence_score': mapping_result['mapping_confidence'],
                'mapping_metadata': {
                    'mapping_method': 'knowledge_based_mapping',
                    'source_representations': mapping_result['source_representations'],
                    'target_representations': mapping_result['target_representations'],
                    'mapping_uncertainty': mapping_result['uncertainty_analysis']
                },
                'mapping_time': datetime.now().isoformat()
            }
            
            # 更新会话数据
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'strategy_mapped'
            
            self.logger.info(f"策略映射完成，映射置信度: {physical_strategy['confidence_score']:.2f}")
            
            return physical_strategy
            
        except Exception as e:
            self.logger.error(f"策略映射失败: {str(e)}")
            raise
    
    def adapt_strategy(self, physical_strategy: Dict[str, Any], physical_environment: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        适应物理世界的环境条件
        
        Args:
            physical_strategy: 映射的物理世界策略
            physical_environment: 物理世界环境数据
            session_id: 当前会话ID
            
        Returns:
            Dict: 适应后的策略，包含：
                - 调整后的动作序列
                - 环境特定参数
                - 适应效果评估
                - 实时优化建议
        """
        try:
            self.logger.info(f"开始策略适应，会话: {session_id}")
            
            # 使用适应引擎进行策略适应
            adaptation_result = self.adaptation_engine.adapt_strategy(
                physical_strategy, 
                physical_environment
            )
            
            # 构建适应后的策略
            adapted_strategy = {
                'adapted_strategy_id': f"adapted_strategy_{datetime.now().timestamp()}",
                'source_physical_strategy': physical_strategy['mapped_strategy_id'],
                'adapted_action_sequences': adaptation_result['adapted_actions'],
                'environmental_adjustments': adaptation_result['environment_adjustments'],
                'parameter_tuning': adaptation_result['parameter_optimizations'],
                'adaptation_metadata': {
                    'adaptation_method': 'real_time_learning',
                    'adaptation_rate': adaptation_result['adaptation_rate'],
                    'learning_progress': adaptation_result['learning_progress'],
                    'convergence_status': adaptation_result['convergence_status']
                },
                'adaptation_confidence': adaptation_result['adaptation_confidence'],
                'adaptation_time': datetime.now().isoformat()
            }
            
            # 更新适应状态
            self.adaptation_state[session_id] = adaptation_result['current_state']
            
            # 更新会话数据
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['adaptation_history'].append(adapted_strategy)
                self.active_sessions[session_id]['status'] = 'strategy_adapted'
            
            self.logger.info(f"策略适应完成，适应置信度: {adapted_strategy['adaptation_confidence']:.2f}")
            
            return adapted_strategy
            
        except Exception as e:
            self.logger.error(f"策略适应失败: {str(e)}")
            raise
    
    def evaluate_transfer(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        评估策略迁移效果
        
        Args:
            adapted_strategy: 适应后的策略
            execution_results: 执行结果数据
            session_id: 当前会话ID
            
        Returns:
            Dict: 评估结果，包含：
                - 成功率和准确率
                - 性能指标对比
                - 改进建议
                - 统计显著性分析
        """
        try:
            self.logger.info(f"开始迁移效果评估，会话: {session_id}")
            
            # 使用迁移评估器进行评估
            evaluation_result = self.transfer_evaluator.evaluate_transfer(
                adapted_strategy,
                execution_results
            )
            
            # 构建评估报告
            evaluation_report = {
                'evaluation_id': f"eval_{datetime.now().timestamp()}",
                'adapted_strategy_id': adapted_strategy['adapted_strategy_id'],
                'session_id': session_id,
                'evaluation_metrics': evaluation_result['metrics'],
                'performance_comparison': evaluation_result['performance_comparison'],
                'statistical_analysis': evaluation_result['statistical_analysis'],
                'improvement_suggestions': evaluation_result['improvement_suggestions'],
                'overall_score': evaluation_result['overall_score'],
                'evaluation_time': datetime.now().isoformat(),
                'evaluation_confidence': evaluation_result['evaluation_confidence']
            }
            
            # 更新性能指标
            self.performance_metrics[session_id] = evaluation_result['metrics']
            
            # 更新会话数据
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['evaluation_results'].append(evaluation_report)
                self.active_sessions[session_id]['status'] = 'transfer_evaluated'
            
            self.logger.info(f"迁移评估完成，总体评分: {evaluation_report['overall_score']:.2f}")
            
            return evaluation_report
            
        except Exception as e:
            self.logger.error(f"迁移评估失败: {str(e)}")
            raise
    
    def optimize_transfer(self, evaluation_report: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        优化迁移性能和效果
        
        Args:
            evaluation_report: 迁移评估报告
            session_id: 当前会话ID
            
        Returns:
            Dict: 优化建议和配置更新，包含：
                - 参数调整建议
                - 策略改进方案
                - 系统配置优化
                - 预期改进效果
        """
        try:
            self.logger.info(f"开始迁移性能优化，会话: {session_id}")
            
            # 使用性能分析器进行优化分析
            optimization_analysis = self.performance_analyzer.analyze_performance(
                self.transfer_history,
                evaluation_report,
                self.performance_metrics.get(session_id, {})
            )
            
            # 生成优化建议
            optimization_suggestions = {
                'parameter_adjustments': optimization_analysis['parameter_recommendations'],
                'strategy_improvements': optimization_analysis['strategy_improvements'],
                'configuration_updates': optimization_analysis['configuration_updates'],
                'learning_optimizations': optimization_analysis['learning_optimizations'],
                'expected_improvements': optimization_analysis['expected_improvements']
            }
            
            # 构建优化报告
            optimization_report = {
                'optimization_id': f"opt_{datetime.now().timestamp()}",
                'source_evaluation': evaluation_report['evaluation_id'],
                'session_id': session_id,
                'optimization_suggestions': optimization_suggestions,
                'confidence_score': optimization_analysis['optimization_confidence'],
                'implementation_priority': optimization_analysis['priority_ranking'],
                'optimization_time': datetime.now().isoformat()
            }
            
            # 应用优化配置（可选）
            if optimization_analysis['auto_apply']:
                self._apply_optimization_changes(optimization_suggestions)
            
            self.logger.info(f"迁移优化完成，置信度: {optimization_report['confidence_score']:.2f}")
            
            return optimization_report
            
        except Exception as e:
            self.logger.error(f"迁移优化失败: {str(e)}")
            raise
    
    def complete_transfer_session(self, session_id: str) -> Dict[str, Any]:
        """
        完成策略迁移会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话总结报告
        """
        try:
            self.logger.info(f"完成迁移会话: {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"会话 {session_id} 不存在")
            
            session_data = self.active_sessions[session_id]
            
            # 生成会话总结
            session_summary = {
                'session_id': session_id,
                'start_time': session_data['start_time'],
                'end_time': datetime.now().isoformat(),
                'status': 'completed',
                'strategies_processed': len(session_data['strategies_processed']),
                'adaptations_performed': len(session_data['adaptation_history']),
                'evaluations_completed': len(session_data['evaluation_results']),
                'overall_performance': self.performance_metrics.get(session_id, {}),
                'recommendations': self._generate_session_recommendations(session_data)
            }
            
            # 保存到历史记录
            self.transfer_history.append(session_summary)
            
            # 清理会话数据
            if self.config.get('logging', {}).get('save_history', True):
                self._save_transfer_history()
            
            # 清理活跃会话
            del self.active_sessions[session_id]
            
            self.logger.info(f"会话 {session_id} 完成，总体性能: {session_summary.get('overall_performance', {})}")
            
            return session_summary
            
        except Exception as e:
            self.logger.error(f"完成会话失败: {str(e)}")
            raise
    
    def get_transfer_status(self, session_id: str) -> Dict[str, Any]:
        """
        获取迁移状态信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 状态信息
        """
        if session_id not in self.active_sessions:
            return {'status': 'not_found'}
        
        return {
            'session_id': session_id,
            'status': self.active_sessions[session_id]['status'],
            'strategies_processed': len(self.active_sessions[session_id]['strategies_processed']),
            'adaptations_performed': len(self.active_sessions[session_id]['adaptation_history']),
            'evaluations_completed': len(self.active_sessions[session_id]['evaluation_results'])
        }
    
    def _validate_minecraft_data(self, data: Dict[str, Any]) -> bool:
        """验证Minecraft数据格式"""
        required_fields = ['scene_info', 'action_sequences', 'performance_metrics']
        return all(field in data for field in required_fields)
    
    def _extract_strategy_patterns(self, minecraft_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取策略模式"""
        patterns = []
        
        # 提取抓取模式
        grab_sequences = [seq for seq in minecraft_data.get('action_sequences', []) 
                         if seq.get('action_type') == 'grab']
        if grab_sequences:
            patterns.append({
                'pattern_type': 'grab_sequence',
                'frequency': len(grab_sequences),
                'success_rate': self._calculate_sequence_success_rate(grab_sequences),
                'typical_duration': self._calculate_average_duration(grab_sequences),
                'common_positions': self._extract_common_positions(grab_sequences)
            })
        
        # 提取放置模式
        place_sequences = [seq for seq in minecraft_data.get('action_sequences', []) 
                          if seq.get('action_type') == 'place']
        if place_sequences:
            patterns.append({
                'pattern_type': 'place_sequence',
                'frequency': len(place_sequences),
                'success_rate': self._calculate_sequence_success_rate(place_sequences),
                'typical_duration': self._calculate_average_duration(place_sequences),
                'precision_patterns': self._analyze_placement_precision(place_sequences)
            })
        
        # 提取搬运模式
        move_sequences = [seq for seq in minecraft_data.get('action_sequences', []) 
                         if seq.get('action_type') == 'move']
        if move_sequences:
            patterns.append({
                'pattern_type': 'transport_sequence',
                'frequency': len(move_sequences),
                'path_patterns': self._analyze_transport_paths(move_sequences),
                'efficiency_metrics': self._calculate_transport_efficiency(move_sequences)
            })
        
        # 提取堆叠模式
        stack_sequences = [seq for seq in minecraft_data.get('action_sequences', []) 
                          if seq.get('action_type') == 'stack']
        if stack_sequences:
            patterns.append({
                'pattern_type': 'stacking_sequence',
                'frequency': len(stack_sequences),
                'stability_patterns': self._analyze_stack_stability(stack_sequences),
                'height_patterns': self._analyze_stack_height_patterns(stack_sequences)
            })
        
        return patterns
    
    def _extract_action_sequences(self, minecraft_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取动作序列"""
        return minecraft_data.get('action_sequences', [])
    
    def _extract_environmental_context(self, minecraft_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取环境上下文"""
        return minecraft_data.get('scene_info', {})
    
    def _extract_performance_metrics(self, minecraft_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取性能指标"""
        return minecraft_data.get('performance_metrics', {})
    
    def _extract_constraints(self, minecraft_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取约束条件"""
        constraints = {
            'physical_constraints': {},
            'temporal_constraints': {},
            'spatial_constraints': {},
            'resource_constraints': {}
        }
        
        # 提取物理约束
        scene_info = minecraft_data.get('scene_info', {})
        if 'gravity' in scene_info:
            constraints['physical_constraints']['gravity'] = scene_info['gravity']
        
        if 'friction_coefficients' in scene_info:
            constraints['physical_constraints']['friction'] = scene_info['friction_coefficients']
        
        # 提取空间约束
        if 'world_bounds' in scene_info:
            constraints['spatial_constraints']['world_bounds'] = scene_info['world_bounds']
        
        if 'reach_limits' in scene_info:
            constraints['spatial_constraints']['reach_limits'] = scene_info['reach_limits']
        
        # 提取时间约束
        performance_metrics = minecraft_data.get('performance_metrics', {})
        if 'max_execution_time' in performance_metrics:
            constraints['temporal_constraints']['max_execution_time'] = performance_metrics['max_execution_time']
        
        if 'response_time_requirement' in performance_metrics:
            constraints['temporal_constraints']['response_time'] = performance_metrics['response_time_requirement']
        
        # 提取资源约束
        if 'energy_limits' in scene_info:
            constraints['resource_constraints']['energy'] = scene_info['energy_limits']
        
        if 'tool_limitations' in scene_info:
            constraints['resource_constraints']['tools'] = scene_info['tool_limitations']
        
        return constraints
    
    def _classify_strategy_type(self, action_sequences: List[Dict[str, Any]]) -> str:
        """分类策略类型"""
        # 基于动作序列分析策略类型
        if not action_sequences:
            return 'unknown'
        
        # 简单的策略类型分类逻辑
        grab_actions = sum(1 for seq in action_sequences if seq.get('action_type') == 'grab')
        place_actions = sum(1 for seq in action_sequences if seq.get('action_type') == 'place')
        
        if grab_actions > place_actions:
            return 'grab_and_move'
        elif place_actions > grab_actions:
            return 'placement_strategy'
        else:
            return 'mixed_strategy'
    
    def _calculate_extraction_confidence(self, minecraft_data: Dict[str, Any]) -> float:
        """计算策略提取置信度"""
        # 基于数据完整性、一致性和质量计算置信度
        completeness_score = self._calculate_data_completeness(minecraft_data)
        consistency_score = self._calculate_data_consistency(minecraft_data)
        quality_score = self._calculate_data_quality(minecraft_data)
        
        # 加权计算总体置信度
        overall_confidence = 0.5 * completeness_score + 0.3 * consistency_score + 0.2 * quality_score
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_data_quality(self, data: Dict[str, Any]) -> float:
        """计算数据质量"""
        # 检查动作序列的质量
        action_sequences = data.get('action_sequences', [])
        if not action_sequences:
            return 0.0
        
        quality_factors = []
        
        # 检查动作序列的长度合理性
        avg_sequence_length = np.mean([len(seq.get('steps', [])) for seq in action_sequences])
        length_quality = max(0.0, 1.0 - abs(avg_sequence_length - 5.0) / 5.0)  # 5步为理想长度
        quality_factors.append(length_quality)
        
        # 检查成功率和失败率
        success_rate = data.get('performance_metrics', {}).get('success_rate', 0.0)
        success_quality = success_rate
        quality_factors.append(success_quality)
        
        # 检查执行时间合理性
        execution_times = [seq.get('execution_time', 0) for seq in action_sequences if seq.get('execution_time')]
        if execution_times:
            avg_time = np.mean(execution_times)
            time_quality = max(0.0, min(1.0, 1.0 - (avg_time - 10.0) / 20.0))  # 10秒为理想时间
            quality_factors.append(time_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """计算数据完整性"""
        required_fields = ['scene_info', 'action_sequences', 'performance_metrics']
        missing_fields = len([f for f in required_fields if f not in data])
        return 1.0 - (missing_fields / len(required_fields))
    
    def _calculate_data_consistency(self, data: Dict[str, Any]) -> float:
        """计算数据一致性"""
        # 检查动作序列的时间一致性
        action_sequences = data.get('action_sequences', [])
        if not action_sequences:
            return 0.0
        
        # 检查时间戳连续性
        timestamps = [seq.get('timestamp', 0) for seq in action_sequences]
        time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # 检查位置变化的合理性
        position_consistency = self._check_position_consistency(action_sequences)
        
        # 检查动作类型转换的合理性
        action_consistency = self._check_action_transition_consistency(action_sequences)
        
        # 综合一致性评分
        if time_gaps:
            time_factor = 1.0 - min(0.5, np.std(time_gaps) / max(np.mean(time_gaps), 1e-8))
        else:
            time_factor = 1.0
        
        overall_consistency = 0.3 * time_factor + 0.4 * position_consistency + 0.3 * action_consistency
        return max(0.0, min(1.0, overall_consistency))
    
    def _check_position_consistency(self, action_sequences: List[Dict]) -> float:
        """检查位置变化一致性"""
        positions = [seq.get('position', [0, 0, 0]) for seq in action_sequences]
        if len(positions) < 2:
            return 1.0
        
        # 计算连续位置间的距离变化
        distances = []
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
            distances.append(dist)
        
        if distances:
            # 距离变化的标准差较小表示位置变化更一致
            consistency = 1.0 - min(1.0, np.std(distances) / max(np.mean(distances), 1e-8))
            return consistency
        return 1.0
    
    def _check_action_transition_consistency(self, action_sequences: List[Dict]) -> float:
        """检查动作转换一致性"""
        action_types = [seq.get('action_type', 'unknown') for seq in action_sequences]
        
        # 检查是否出现不合理的动作转换
        invalid_transitions = 0
        total_transitions = len(action_types) - 1
        
        for i in range(total_transitions):
            current_action = action_types[i]
            next_action = action_types[i + 1]
            
            # 检查不合理转换（如grab后直接grab无中间步骤）
            if (current_action == 'grab' and next_action == 'grab' and 
                i + 1 < len(action_sequences) - 1):
                # 如果两个grab动作之间没有其他动作，可能不合理
                invalid_transitions += 1
        
        if total_transitions == 0:
            return 1.0
        
        consistency = 1.0 - (invalid_transitions / total_transitions)
        return consistency
    
    def _apply_optimization_changes(self, suggestions: Dict[str, Any]):
        """应用优化配置变化"""
        # 更新组件配置
        param_updates = suggestions.get('parameter_adjustments', {})
        
        if 'adaptation_rate' in param_updates:
            self.adaptation_engine.adaptation_rate = param_updates['adaptation_rate']
        
        if 'mapping_threshold' in param_updates:
            self.knowledge_mapper.similarity_threshold = param_updates['mapping_threshold']
        
        self.logger.info("已应用优化配置变化")
    
    def _generate_session_recommendations(self, session_data: Dict[str, Any]) -> List[str]:
        """生成会话建议"""
        recommendations = []
        
        if len(session_data['strategies_processed']) < 3:
            recommendations.append("建议处理更多策略样例以提高迁移效果")
        
        if len(session_data['evaluation_results']) > 0:
            latest_eval = session_data['evaluation_results'][-1]
            if latest_eval.get('overall_score', 0) < 0.7:
                recommendations.append("迁移效果有待提高，建议优化参数设置")
        
        return recommendations
    
    def _save_transfer_history(self):
        """保存迁移历史记录"""
        try:
            history_file = self.config.get('logging', {}).get('history_file', 'transfer_history.json')
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.transfer_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存迁移历史失败: {str(e)}")
    
    # 添加辅助方法
    def _calculate_sequence_success_rate(self, sequences: List[Dict]) -> float:
        """计算序列成功率"""
        if not sequences:
            return 0.0
        
        successful = sum(1 for seq in sequences if seq.get('success', False))
        return successful / len(sequences)
    
    def _calculate_average_duration(self, sequences: List[Dict]) -> float:
        """计算平均持续时间"""
        durations = [seq.get('duration', 0) for seq in sequences if seq.get('duration')]
        return np.mean(durations) if durations else 0.0
    
    def _extract_common_positions(self, sequences: List[Dict]) -> List[Dict]:
        """提取常见位置模式"""
        positions = [seq.get('position', [0, 0, 0]) for seq in sequences]
        if not positions:
            return []
        
        # 简单的聚类分析
        position_clusters = {}
        for pos in positions:
            # 量化位置到网格
            grid_pos = tuple(round(p, 1) for p in pos)
            if grid_pos not in position_clusters:
                position_clusters[grid_pos] = 0
            position_clusters[grid_pos] += 1
        
        # 返回最常见的几个位置
        sorted_positions = sorted(position_clusters.items(), key=lambda x: x[1], reverse=True)
        return [{'position': list(pos), 'frequency': freq} for pos, freq in sorted_positions[:3]]
    
    def _analyze_placement_precision(self, sequences: List[Dict]) -> Dict[str, float]:
        """分析放置精度模式"""
        precision_scores = [seq.get('precision_score', 0.0) for seq in sequences if seq.get('precision_score')]
        if not precision_scores:
            return {'average_precision': 0.0, 'precision_variance': 0.0}
        
        return {
            'average_precision': np.mean(precision_scores),
            'precision_variance': np.var(precision_scores),
            'high_precision_ratio': sum(1 for p in precision_scores if p > 0.8) / len(precision_scores)
        }
    
    def _analyze_transport_paths(self, sequences: List[Dict]) -> List[Dict]:
        """分析搬运路径模式"""
        paths = []
        for seq in sequences:
            if 'path' in seq:
                path_info = {
                    'start_position': seq.get('start_position', [0, 0, 0]),
                    'end_position': seq.get('end_position', [0, 0, 0]),
                    'path_length': seq.get('path_length', 0),
                    'efficiency': seq.get('efficiency_score', 0.0)
                }
                paths.append(path_info)
        return paths
    
    def _calculate_transport_efficiency(self, sequences: List[Dict]) -> Dict[str, float]:
        """计算搬运效率"""
        efficiencies = [seq.get('efficiency_score', 0.0) for seq in sequences if seq.get('efficiency_score')]
        if not efficiencies:
            return {'average_efficiency': 0.0, 'efficiency_variance': 0.0}
        
        return {
            'average_efficiency': np.mean(efficiencies),
            'efficiency_variance': np.var(efficiencies),
            'max_efficiency': np.max(efficiencies)
        }
    
    def _analyze_stack_stability(self, sequences: List[Dict]) -> Dict[str, float]:
        """分析堆叠稳定性"""
        stability_scores = [seq.get('stability_score', 0.0) for seq in sequences if seq.get('stability_score')]
        if not stability_scores:
            return {'average_stability': 0.0, 'stability_variance': 0.0}
        
        return {
            'average_stability': np.mean(stability_scores),
            'stability_variance': np.var(stability_scores),
            'collapse_count': sum(1 for seq in sequences if seq.get('collapsed', False))
        }
    
    def _analyze_stack_height_patterns(self, sequences: List[Dict]) -> Dict[str, int]:
        """分析堆叠高度模式"""
        heights = [seq.get('stack_height', 1) for seq in sequences if seq.get('stack_height')]
        if not heights:
            return {'common_height': 1, 'max_height': 1, 'height_distribution': {1: 1}}
        
        height_distribution = {}
        for height in heights:
            height_distribution[height] = height_distribution.get(height, 0) + 1
        
        common_height = max(height_distribution.items(), key=lambda x: x[1])[0]
        
        return {
            'common_height': common_height,
            'max_height': max(heights),
            'height_distribution': height_distribution
        }

    def get_transfer_statistics(self) -> Dict[str, Any]:
        """获取迁移系统统计信息"""
        total_sessions = len(self.transfer_history)
        active_sessions = len(self.active_sessions)
        
        if total_sessions == 0:
            return {
                'total_sessions': 0,
                'active_sessions': active_sessions,
                'average_performance': 0,
                'success_rate': 0
            }
        
        # 计算统计指标
        total_performance = sum(
            session.get('overall_performance', {}).get('overall_score', 0) 
            for session in self.transfer_history
        )
        
        avg_performance = total_performance / total_sessions
        
        successful_sessions = sum(
            1 for session in self.transfer_history
            if session.get('overall_performance', {}).get('success_rate', 0) > 0.8
        )
        
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'average_performance': avg_performance,
            'success_rate': success_rate,
            'last_session': self.transfer_history[-1] if self.transfer_history else None
        }