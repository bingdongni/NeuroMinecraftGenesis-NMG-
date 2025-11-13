#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移评估器
负责评估策略从Minecraft到物理世界的迁移效果和性能
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy import stats
from collections import defaultdict
import math


class TransferEvaluator:
    """
    迁移评估器类
    
    功能：
    1. 评估策略迁移的效果和性能
    2. 对比迁移前后的表现指标
    3. 进行统计显著性分析
    4. 生成改进建议和优化方案
    
    评估维度：
    - 准确性：动作执行的准确性和精度
    - 成功率：任务完成的比例
    - 效率：执行时间和资源消耗
    - 稳定性：多次执行的一致性
    - 适应性：适应新环境的能力
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化迁移评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('TransferEvaluator')
        
        # 评估指标配置
        self.evaluation_metrics = self.config.get('evaluation_metrics', [
            'accuracy', 'success_rate', 'execution_time', 'stability', 'adaptability'
        ])
        self.baseline_comparison = self.config.get('baseline_comparison', True)
        self.statistical_significance = self.config.get('statistical_significance', 0.05)
        
        # 评估基准和历史数据
        self.benchmarks = self._load_benchmarks()
        self.evaluation_history = []
        self.performance_baselines = {}
        
        # 评估模型和算法
        self.accuracy_evaluator = self._init_accuracy_evaluator()
        self.stability_analyzer = self._init_stability_analyzer()
        self.efficiency_calculator = self._init_efficiency_calculator()
        self.adaptation_evaluator = self._init_adaptation_evaluator()
        
        self.logger.info("迁移评估器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'evaluation_metrics': [
                'accuracy', 'success_rate', 'execution_time', 'stability', 'adaptability'
            ],
            'baseline_comparison': True,
            'statistical_significance': 0.05,
            'evaluation_window': 100,  # 评估窗口大小
            'comparison_methods': ['absolute', 'relative', 'statistical'],
            'report_granularity': 'detailed'  # summary, detailed, comprehensive
        }
    
    def _load_benchmarks(self) -> Dict[str, Any]:
        """加载性能基准"""
        return {
            # Minecraft环境基准
            'minecraft_baseline': {
                'accuracy': 0.85,
                'success_rate': 0.90,
                'execution_time_normalized': 1.0,
                'stability': 0.80,
                'adaptability': 0.75
            },
            
            # 物理世界基准
            'physical_world_baseline': {
                'accuracy': 0.75,
                'success_rate': 0.80,
                'execution_time_normalized': 1.2,
                'stability': 0.70,
                'adaptability': 0.65
            },
            
            # 目标性能指标
            'target_performance': {
                'accuracy': 0.80,
                'success_rate': 0.85,
                'execution_time_ratio': 1.1,  # 允许比Minecraft慢10%
                'stability': 0.75,
                'adaptability': 0.70
            }
        }
    
    def _init_accuracy_evaluator(self) -> Dict[str, Any]:
        """初始化准确度评估器"""
        return {
            'positional_tolerance': 0.05,  # 5cm位置容忍度
            'angular_tolerance': math.radians(5),  # 5度角度容忍度
            'force_tolerance': 0.1,  # 10%力度容忍度
            'evaluation_methods': ['euclidean_distance', 'angular_difference', 'force_deviation']
        }
    
    def _init_stability_analyzer(self) -> Dict[str, Any]:
        """初始化稳定性分析器"""
        return {
            'variance_threshold': 0.1,
            'consistency_measure': 'coefficient_of_variation',
            'stability_metrics': ['position_variance', 'timing_variance', 'success_consistency']
        }
    
    def _init_efficiency_calculator(self) -> Dict[str, Any]:
        """初始化效率计算器"""
        return {
            'time_weight': 0.4,
            'energy_weight': 0.3,
            'resource_weight': 0.3,
            'efficiency_metrics': ['execution_time', 'energy_consumption', 'resource_utilization']
        }
    
    def _init_adaptation_evaluator(self) -> Dict[str, Any]:
        """初始化适应性评估器"""
        return {
            'adaptation_time': 10.0,  # 适应时间阈值10秒
            'performance_degradation': 0.2,  # 允许的性能下降20%
            'adaptation_metrics': ['learning_rate', 'performance_recovery', 'environmental适应性']
        }
    
    def evaluate_transfer(self, adapted_strategy: Dict[str, Any], 
                        execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估策略迁移效果
        
        Args:
            adapted_strategy: 适应后的策略
            execution_results: 执行结果数据
            
        Returns:
            Dict: 评估结果，包含指标值、性能对比、统计分析等
        """
        try:
            self.logger.info("开始迁移效果评估")
            
            # 执行多维度评估
            metrics_evaluation = self._evaluate_all_metrics(adapted_strategy, execution_results)
            
            # 性能对比分析
            performance_comparison = self._compare_performance(metrics_evaluation)
            
            # 统计显著性分析
            statistical_analysis = self._perform_statistical_analysis(execution_results)
            
            # 生成改进建议
            improvement_suggestions = self._generate_improvement_suggestions(
                metrics_evaluation, performance_comparison
            )
            
            # 计算总体评分
            overall_score = self._calculate_overall_score(metrics_evaluation)
            
            # 构建完整评估报告
            evaluation_result = {
                'evaluation_id': f"eval_{datetime.now().timestamp()}",
                'evaluation_timestamp': datetime.now().isoformat(),
                'metrics': metrics_evaluation,
                'performance_comparison': performance_comparison,
                'statistical_analysis': statistical_analysis,
                'improvement_suggestions': improvement_suggestions,
                'overall_score': overall_score,
                'evaluation_confidence': self._calculate_evaluation_confidence(metrics_evaluation),
                'evaluation_metadata': {
                    'metrics_count': len(metrics_evaluation),
                    'comparison_baselines': list(performance_comparison.keys()),
                    'statistical_tests_performed': list(statistical_analysis.keys())
                }
            }
            
            # 保存评估历史
            self.evaluation_history.append(evaluation_result)
            
            self.logger.info(f"迁移评估完成，总体评分: {overall_score:.2f}")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"迁移评估失败: {str(e)}")
            raise
    
    def _evaluate_all_metrics(self, adapted_strategy: Dict[str, Any], 
                            execution_results: Dict[str, Any]) -> Dict[str, float]:
        """评估所有指标"""
        metrics = {}
        
        # 准确度评估
        if 'accuracy' in self.evaluation_metrics:
            metrics['accuracy'] = self._evaluate_accuracy(adapted_strategy, execution_results)
        
        # 成功率评估
        if 'success_rate' in self.evaluation_metrics:
            metrics['success_rate'] = self._evaluate_success_rate(adapted_strategy, execution_results)
        
        # 执行时间评估
        if 'execution_time' in self.evaluation_metrics:
            metrics['execution_time'] = self._evaluate_execution_time(adapted_strategy, execution_results)
        
        # 稳定性评估
        if 'stability' in self.evaluation_metrics:
            metrics['stability'] = self._evaluate_stability(adapted_strategy, execution_results)
        
        # 适应性评估
        if 'adaptability' in self.evaluation_metrics:
            metrics['adaptability'] = self._evaluate_adaptability(adapted_strategy, execution_results)
        
        return metrics
    
    def _evaluate_accuracy(self, adapted_strategy: Dict[str, Any], 
                         execution_results: Dict[str, Any]) -> float:
        """评估准确度"""
        try:
            # 获取实际执行结果
            actual_results = execution_results.get('execution_data', [])
            if not actual_results:
                return 0.0
            
            accuracy_scores = []
            
            for result in actual_results:
                # 位置准确度
                actual_position = result.get('actual_position', [0, 0, 0])
                target_position = result.get('target_position', [0, 0, 0])
                positional_error = np.linalg.norm(np.array(actual_position) - np.array(target_position))
                positional_accuracy = max(0, 1 - positional_error / self.accuracy_evaluator['positional_tolerance'])
                
                # 角度准确度
                actual_orientation = result.get('actual_orientation', [0, 0, 0])
                target_orientation = result.get('target_orientation', [0, 0, 0])
                angular_error = self._calculate_angular_difference(actual_orientation, target_orientation)
                angular_accuracy = max(0, 1 - angular_error / self.accuracy_evaluator['angular_tolerance'])
                
                # 综合准确度
                overall_accuracy = 0.6 * positional_accuracy + 0.4 * angular_accuracy
                accuracy_scores.append(overall_accuracy)
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"准确度评估失败: {str(e)}")
            return 0.0
    
    def _evaluate_success_rate(self, adapted_strategy: Dict[str, Any], 
                             execution_results: Dict[str, Any]) -> float:
        """评估成功率"""
        try:
            execution_data = execution_results.get('execution_data', [])
            if not execution_data:
                return 0.0
            
            successful_executions = 0
            total_executions = len(execution_data)
            
            for result in execution_data:
                # 判断执行是否成功
                success_criteria = result.get('success_criteria', {})
                
                is_successful = (
                    result.get('completed', False) and
                    result.get('error_count', 0) == 0 and
                    result.get('final_state_correct', True)
                )
                
                if is_successful:
                    successful_executions += 1
            
            return successful_executions / total_executions if total_executions > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"成功率评估失败: {str(e)}")
            return 0.0
    
    def _evaluate_execution_time(self, adapted_strategy: Dict[str, Any], 
                               execution_results: Dict[str, Any]) -> float:
        """评估执行时间"""
        try:
            execution_data = execution_results.get('execution_data', [])
            if not execution_data:
                return 0.0
            
            execution_times = []
            for result in execution_data:
                execution_time = result.get('execution_time', 0)
                if execution_time > 0:
                    execution_times.append(execution_time)
            
            if not execution_times:
                return 0.0
            
            avg_execution_time = np.mean(execution_times)
            
            # 标准化执行时间（相对于Minecraft基准）
            minecraft_baseline = self.benchmarks['minecraft_baseline']['execution_time_normalized']
            normalized_time = avg_execution_time / minecraft_baseline
            
            # 时间效率评分（越短越好）
            if normalized_time <= 1.0:
                return 1.0
            else:
                return max(0.3, 1.0 / normalized_time)
            
        except Exception as e:
            self.logger.error(f"执行时间评估失败: {str(e)}")
            return 0.0
    
    def _evaluate_stability(self, adapted_strategy: Dict[str, Any], 
                          execution_results: Dict[str, Any]) -> float:
        """评估稳定性"""
        try:
            execution_data = execution_results.get('execution_data', [])
            if len(execution_data) < 2:
                return 0.0  # 需要至少两次执行来评估稳定性
            
            # 分析性能变异性
            performance_values = []
            for result in execution_data:
                # 计算每次执行的综合性能分数
                performance_score = self._calculate_performance_score(result)
                performance_values.append(performance_score)
            
            # 计算变异系数
            mean_performance = np.mean(performance_values)
            std_performance = np.std(performance_values)
            
            if mean_performance == 0:
                return 0.0
            
            coefficient_of_variation = std_performance / mean_performance
            
            # 稳定性评分（变异系数越小，稳定性越高）
            stability_score = max(0, 1 - coefficient_of_variation / self.stability_analyzer['variance_threshold'])
            
            return stability_score
            
        except Exception as e:
            self.logger.error(f"稳定性评估失败: {str(e)}")
            return 0.0
    
    def _evaluate_adaptability(self, adapted_strategy: Dict[str, Any], 
                             execution_results: Dict[str, Any]) -> float:
        """评估适应性"""
        try:
            adaptation_data = execution_results.get('adaptation_performance', [])
            if not adaptation_data:
                return 0.0
            
            adaptability_scores = []
            
            for adaptation_result in adaptation_data:
                # 适应时间评分
                adaptation_time = adaptation_result.get('adaptation_time', 0)
                time_score = max(0, 1 - adaptation_time / self.adaptation_evaluator['adaptation_time'])
                
                # 性能恢复评分
                initial_performance = adaptation_result.get('initial_performance', 0)
                final_performance = adaptation_result.get('final_performance', 0)
                performance_recovery = final_performance / initial_performance if initial_performance > 0 else 0
                recovery_score = max(0, min(1, performance_recovery))
                
                # 适应性评分
                adaptability_score = 0.5 * time_score + 0.5 * recovery_score
                adaptability_scores.append(adaptability_score)
            
            return np.mean(adaptability_scores) if adaptability_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"适应性评估失败: {str(e)}")
            return 0.0
    
    def _compare_performance(self, metrics_evaluation: Dict[str, float]) -> Dict[str, Any]:
        """性能对比分析"""
        comparison_results = {}
        
        # 与Minecraft基准对比
        minecraft_comparison = self._compare_with_baseline(
            metrics_evaluation, self.benchmarks['minecraft_baseline'], 'minecraft'
        )
        comparison_results['minecraft_baseline'] = minecraft_comparison
        
        # 与物理世界基准对比
        physical_comparison = self._compare_with_baseline(
            metrics_evaluation, self.benchmarks['physical_world_baseline'], 'physical'
        )
        comparison_results['physical_baseline'] = physical_comparison
        
        # 与目标性能对比
        target_comparison = self._compare_with_baseline(
            metrics_evaluation, self.benchmarks['target_performance'], 'target'
        )
        comparison_results['target_performance'] = target_comparison
        
        return comparison_results
    
    def _compare_with_baseline(self, metrics: Dict[str, float], 
                             baseline: Dict[str, float], baseline_name: str) -> Dict[str, Any]:
        """与特定基准进行对比"""
        comparison = {
            'baseline_name': baseline_name,
            'absolute_differences': {},
            'relative_differences': {},
            'performance_ratios': {},
            'improvement_areas': [],
            'degradation_areas': []
        }
        
        for metric_name, metric_value in metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                
                # 绝对差异
                absolute_diff = metric_value - baseline_value
                comparison['absolute_differences'][metric_name] = absolute_diff
                
                # 相对差异（百分比）
                relative_diff = (absolute_diff / baseline_value) * 100 if baseline_value != 0 else 0
                comparison['relative_differences'][metric_name] = relative_diff
                
                # 性能比值
                performance_ratio = metric_value / baseline_value if baseline_value != 0 else 0
                comparison['performance_ratios'][metric_name] = performance_ratio
                
                # 改进和退化区域
                if absolute_diff > 0:
                    comparison['improvement_areas'].append({
                        'metric': metric_name,
                        'improvement': absolute_diff,
                        'percentage': relative_diff
                    })
                else:
                    comparison['degradation_areas'].append({
                        'metric': metric_name,
                        'degradation': abs(absolute_diff),
                        'percentage': abs(relative_diff)
                    })
        
        return comparison
    
    def _perform_statistical_analysis(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行统计分析"""
        statistical_results = {}
        
        execution_data = execution_results.get('execution_data', [])
        if len(execution_data) < 3:
            # 数据不足时返回基本信息
            return {'insufficient_data': True, 'message': '数据量不足，无法进行统计检验'}
        
        # 收集性能指标数据
        performance_data = []
        for result in execution_data:
            score = self._calculate_performance_score(result)
            performance_data.append(score)
        
        # 正态性检验
        normality_test = stats.shapiro(performance_data)
        statistical_results['normality_test'] = {
            'test_type': 'shapiro_wilk',
            'statistic': normality_test.statistic,
            'p_value': normality_test.pvalue,
            'is_normal': normality_test.pvalue > self.statistical_significance
        }
        
        # 计算置信区间
        confidence_level = 1 - self.statistical_significance
        confidence_interval = stats.t.interval(
            confidence_level, 
            len(performance_data) - 1,
            loc=np.mean(performance_data),
            scale=stats.sem(performance_data)
        )
        statistical_results['confidence_interval'] = {
            'level': confidence_level,
            'lower_bound': confidence_interval[0],
            'upper_bound': confidence_interval[1],
            'margin_of_error': (confidence_interval[1] - confidence_interval[0]) / 2
        }
        
        # 与基准的显著性差异检验（如果基准数据可用）
        if hasattr(self, 'baseline_performance_data') and self.baseline_performance_data:
            t_test = stats.ttest_ind(performance_data, self.baseline_performance_data)
            statistical_results['significance_test'] = {
                'test_type': 'independent_t_test',
                'statistic': t_test.statistic,
                'p_value': t_test.pvalue,
                'is_significantly_different': t_test.pvalue < self.statistical_significance,
                'effect_size': self._calculate_cohens_d(performance_data, self.baseline_performance_data)
            }
        
        return statistical_results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """计算Cohen's d效应大小"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                            (len(group1) + len(group2) - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
    
    def _generate_improvement_suggestions(self, metrics_evaluation: Dict[str, float],
                                        performance_comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        suggestions = []
        
        # 基于指标评估生成建议
        for metric_name, score in metrics_evaluation.items():
            if score < 0.7:  # 性能低于70%
                suggestion = {
                    'metric': metric_name,
                    'current_score': score,
                    'priority': 'high' if score < 0.5 else 'medium',
                    'suggested_actions': self._get_metric_specific_suggestions(metric_name, score),
                    'expected_improvement': self._estimate_improvement_potential(metric_name, score)
                }
                suggestions.append(suggestion)
        
        # 基于性能对比生成建议
        target_comparison = performance_comparison.get('target_performance', {})
        for area in target_comparison.get('degradation_areas', []):
            if area['percentage'] > 10:  # 性能下降超过10%
                suggestion = {
                    'metric': area['metric'],
                    'degradation': area['degradation'],
                    'priority': 'high',
                    'focus_area': 'performance_recovery',
                    'suggested_actions': self._get_recovery_suggestions(area['metric'], area['degradation'])
                }
                suggestions.append(suggestion)
        
        # 排序建议（优先级高的在前）
        suggestions.sort(key=lambda x: (x.get('priority') == 'high', -x.get('current_score', 0)))
        
        return suggestions
    
    def _get_metric_specific_suggestions(self, metric_name: str, score: float) -> List[str]:
        """获取指标特定建议"""
        suggestions_map = {
            'accuracy': [
                '提高传感器精度和校准',
                '优化控制系统参数',
                '改进视觉定位算法',
                '增强力反馈机制'
            ],
            'success_rate': [
                '完善错误检测和恢复机制',
                '提高任务规划质量',
                '优化执行序列',
                '增加冗余检查点'
            ],
            'execution_time': [
                '优化算法效率',
                '并行化处理流程',
                '缓存常用计算结果',
                '简化不必要的计算'
            ],
            'stability': [
                '降低控制系统增益',
                '增加滤波和去噪',
                '改进传感器融合',
                '优化机械结构减振'
            ],
            'adaptability': [
                '增加在线学习能力',
                '改进环境感知算法',
                '优化参数自调整',
                '增加适应性控制器'
            ]
        }
        
        return suggestions_map.get(metric_name, ['需要进一步分析'])
    
    def _estimate_improvement_potential(self, metric_name: str, current_score: float) -> Dict[str, float]:
        """估算改进潜力"""
        # 基于历史数据和理论上限估算改进潜力
        theoretical_max = 1.0
        current_gap = theoretical_max - current_score
        
        # 根据指标类型调整潜力估计
        potential_multipliers = {
            'accuracy': 0.8,
            'success_rate': 0.9,
            'execution_time': 0.7,
            'stability': 0.6,
            'adaptability': 0.8
        }
        
        multiplier = potential_multipliers.get(metric_name, 0.7)
        
        return {
            'max_improvement': current_gap * multiplier,
            'realistic_improvement': current_gap * multiplier * 0.6,
            'difficulty_level': 'medium' if current_score > 0.5 else 'high'
        }
    
    def _get_recovery_suggestions(self, metric_name: str, degradation: float) -> List[str]:
        """获取恢复建议"""
        recovery_suggestions = {
            'accuracy': [
                '重新校准所有传感器',
                '检查机械精度',
                '更新控制算法参数'
            ],
            'success_rate': [
                '检查执行环境变化',
                '更新任务执行逻辑',
                '增加异常处理机制'
            ],
            'execution_time': [
                '分析性能瓶颈',
                '优化关键路径算法',
                '升级硬件配置'
            ]
        }
        
        return recovery_suggestions.get(metric_name, ['需要详细分析原因'])
    
    def _calculate_overall_score(self, metrics_evaluation: Dict[str, float]) -> float:
        """计算总体评分"""
        if not metrics_evaluation:
            return 0.0
        
        # 指标权重
        weights = {
            'accuracy': 0.25,
            'success_rate': 0.25,
            'execution_time': 0.20,
            'stability': 0.15,
            'adaptability': 0.15
        }
        
        # 加权平均
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, score in metrics_evaluation.items():
            weight = weights.get(metric_name, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_evaluation_confidence(self, metrics_evaluation: Dict[str, float]) -> float:
        """计算评估置信度"""
        if not metrics_evaluation:
            return 0.0
        
        # 基于指标数量和分布计算置信度
        metric_count = len(metrics_evaluation)
        score_variance = np.var(list(metrics_evaluation.values()))
        
        # 指标越多，置信度越高；方差越小，置信度越高
        count_factor = min(1.0, metric_count / 5)  # 5个指标为满分
        variance_factor = max(0.5, 1 - score_variance)  # 方差越小越好
        
        confidence = 0.6 * count_factor + 0.4 * variance_factor
        return min(1.0, max(0.0, confidence))
    
    def _calculate_performance_score(self, execution_result: Dict[str, Any]) -> float:
        """计算单次执行的性能分数"""
        # 综合多个指标的简单评分
        components = []
        
        if 'accuracy' in execution_result:
            components.append(execution_result['accuracy'])
        
        if 'completion' in execution_result:
            components.append(1.0 if execution_result['completion'] else 0.0)
        
        if 'efficiency' in execution_result:
            components.append(execution_result['efficiency'])
        
        return np.mean(components) if components else 0.0
    
    def _calculate_angular_difference(self, actual: List[float], target: List[float]) -> float:
        """计算角度差异"""
        # 简化的角度差异计算
        diff_x = abs(actual[0] - target[0])
        diff_y = abs(actual[1] - target[1])
        diff_z = abs(actual[2] - target[2])
        
        # 归一化到0-π范围
        diff_x = min(diff_x, 2 * math.pi - diff_x)
        diff_y = min(diff_y, 2 * math.pi - diff_y)
        diff_z = min(diff_z, 2 * math.pi - diff_z)
        
        return math.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    
    def update_baseline_data(self, baseline_performance: List[float]):
        """更新基准数据"""
        self.baseline_performance_data = baseline_performance
        self.logger.info("基准数据已更新")
    
    def analyze_transfer_quality(self, adapted_strategy: Dict[str, Any], 
                               execution_results: Dict[str, Any],
                               quality_dimensions: List[str] = None) -> Dict[str, Any]:
        """深入分析迁移质量
        
        Args:
            adapted_strategy: 适应后的策略
            execution_results: 执行结果数据
            quality_dimensions: 分析维度列表，默认包含所有维度
            
        Returns:
            Dict: 详细的质量分析报告
        """
        try:
            self.logger.info("开始深入迁移质量分析")
            
            quality_dimensions = quality_dimensions or [
                'precision', 'consistency', 'robustness', 'efficiency', 'scalability'
            ]
            
            # 执行多维度质量分析
            quality_analysis = {}
            
            # 精度分析
            if 'precision' in quality_dimensions:
                quality_analysis['precision'] = self._analyze_precision(adapted_strategy, execution_results)
            
            # 一致性分析
            if 'consistency' in quality_dimensions:
                quality_analysis['consistency'] = self._analyze_consistency(adapted_strategy, execution_results)
            
            # 鲁棒性分析
            if 'robustness' in quality_dimensions:
                quality_analysis['robustness'] = self._analyze_robustness(adapted_strategy, execution_results)
            
            # 效率分析
            if 'efficiency' in quality_dimensions:
                quality_analysis['efficiency'] = self._analyze_transfer_efficiency(adapted_strategy, execution_results)
            
            # 可扩展性分析
            if 'scalability' in quality_dimensions:
                quality_analysis['scalability'] = self._analyze_scalability(adapted_strategy, execution_results)
            
            # 计算综合质量分数
            overall_quality_score = self._calculate_overall_quality_score(quality_analysis)
            
            # 生成质量等级评估
            quality_grade = self._determine_quality_grade(overall_quality_score)
            
            # 识别质量问题和风险
            quality_issues = self._identify_quality_issues(quality_analysis)
            
            # 生成改进建议
            quality_suggestions = self._generate_quality_improvement_suggestions(quality_analysis, quality_issues)
            
            quality_report = {
                'analysis_id': f"quality_{datetime.now().timestamp()}",
                'analysis_timestamp': datetime.now().isoformat(),
                'quality_dimensions': quality_dimensions,
                'dimension_analysis': quality_analysis,
                'overall_quality_score': overall_quality_score,
                'quality_grade': quality_grade,
                'quality_issues': quality_issues,
                'improvement_suggestions': quality_suggestions,
                'analysis_confidence': self._calculate_quality_analysis_confidence(quality_analysis),
                'metadata': {
                    'analysis_depth': len(quality_dimensions),
                    'data_sufficiency': self._assess_data_sufficiency(execution_results)
                }
            }
            
            self.logger.info(f"迁移质量分析完成，综合质量分数: {overall_quality_score:.2f}")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"迁移质量分析失败: {str(e)}")
            raise
    
    def compare_strategies(self, strategies_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """对比多个迁移策略的效果
        
        Args:
            strategies_data: 策略数据字典，格式为 {strategy_name: {'strategy': strategy_data, 'results': results_data}}
            
        Returns:
            Dict: 策略对比分析报告
        """
        try:
            self.logger.info(f"开始对比 {len(strategies_data)} 个迁移策略")
            
            if not strategies_data:
                raise ValueError("策略数据不能为空")
            
            # 评估所有策略
            strategy_evaluations = {}
            for strategy_name, strategy_data in strategies_data.items():
                evaluation = self.evaluate_transfer(
                    strategy_data.get('strategy', {}),
                    strategy_data.get('results', {})
                )
                strategy_evaluations[strategy_name] = evaluation
            
            # 进行对比分析
            comparison_results = self._perform_strategy_comparison(strategy_evaluations)
            
            # 排序策略（按总体评分）
            ranked_strategies = self._rank_strategies(strategy_evaluations)
            
            # 识别最佳策略和特征
            best_strategy = ranked_strategies[0] if ranked_strategies else None
            strategy_features = self._analyze_strategy_features(strategy_evaluations)
            
            # 生成策略选择建议
            selection_recommendations = self._generate_strategy_selection_recommendations(
                ranked_strategies, comparison_results
            )
            
            comparison_report = {
                'comparison_id': f"comparison_{datetime.now().timestamp()}",
                'comparison_timestamp': datetime.now().isoformat(),
                'strategies_count': len(strategies_data),
                'strategy_evaluations': strategy_evaluations,
                'comparison_results': comparison_results,
                'ranked_strategies': ranked_strategies,
                'best_strategy': best_strategy,
                'strategy_features': strategy_features,
                'selection_recommendations': selection_recommendations,
                'analysis_summary': {
                    'top_performing_strategy': best_strategy,
                    'average_performance': self._calculate_average_performance(strategy_evaluations),
                    'performance_variance': self._calculate_performance_variance(strategy_evaluations),
                    'most_common_strength': self._identify_common_strengths(strategy_evaluations),
                    'most_common_weakness': self._identify_common_weaknesses(strategy_evaluations)
                }
            }
            
            self.logger.info(f"策略对比分析完成，最佳策略: {best_strategy}")
            return comparison_report
            
        except Exception as e:
            self.logger.error(f"策略对比分析失败: {str(e)}")
            raise
    
    def generate_improvement_suggestions(self, evaluation_result: Dict[str, Any],
                                       suggestion_type: str = "comprehensive") -> Dict[str, Any]:
        """基于评估结果生成改进建议（公有的改进建议生成方法）
        
        Args:
            evaluation_result: 评估结果数据
            suggestion_type: 建议类型 ('comprehensive', 'focused', 'prioritized')
            
        Returns:
            Dict: 详细的改进建议报告
        """
        try:
            self.logger.info(f"开始生成 {suggestion_type} 类型的改进建议")
            
            # 基于评估结果提取核心信息
            metrics = evaluation_result.get('metrics', {})
            performance_comparison = evaluation_result.get('performance_comparison', {})
            statistical_analysis = evaluation_result.get('statistical_analysis', {})
            overall_score = evaluation_result.get('overall_score', 0.0)
            
            # 生成多层次改进建议
            improvement_suggestions = {
                'immediate_actions': self._generate_immediate_actions(metrics, overall_score),
                'short_term_improvements': self._generate_short_term_improvements(metrics, performance_comparison),
                'long_term_optimizations': self._generate_long_term_optimizations(statistical_analysis),
                'strategic_recommendations': self._generate_strategic_recommendations(evaluation_result)
            }
            
            # 根据建议类型进行筛选和优先级排序
            if suggestion_type == "focused":
                improvement_suggestions = {
                    k: v for k, v in improvement_suggestions.items() 
                    if k in ['immediate_actions', 'short_term_improvements']
                }
            elif suggestion_type == "prioritized":
                improvement_suggestions = self._prioritize_suggestions(improvement_suggestions)
            
            # 计算实施优先级和时间预期
            implementation_plan = self._create_implementation_plan(improvement_suggestions)
            
            # 预估改进效果
            improvement_forecasts = self._forecast_improvement_potential(improvement_suggestions, overall_score)
            
            suggestion_report = {
                'suggestions_id': f"suggestions_{datetime.now().timestamp()}",
                'generation_timestamp': datetime.now().isoformat(),
                'suggestion_type': suggestion_type,
                'current_performance': {
                    'overall_score': overall_score,
                    'key_metrics': metrics
                },
                'improvement_suggestions': improvement_suggestions,
                'implementation_plan': implementation_plan,
                'improvement_forecasts': improvement_forecasts,
                'confidence_level': self._calculate_suggestion_confidence(improvement_suggestions),
                'recommendations_summary': self._summarize_recommendations(improvement_suggestions)
            }
            
            self.logger.info(f"改进建议生成完成，建议数量: {sum(len(v) for v in improvement_suggestions.values())}")
            return suggestion_report
            
        except Exception as e:
            self.logger.error(f"改进建议生成失败: {str(e)}")
            raise
    
    def _calculate_improvement_trends(self) -> Dict[str, float]:
        """计算改进趋势"""
        if len(self.evaluation_history) < 2:
            return {}
        
        trends = {}
        for metric in self.evaluation_metrics:
            scores = [eval_data['metrics'].get(metric, 0) 
                     for eval_data in self.evaluation_history]
            
            # 线性回归计算趋势
            x = list(range(len(scores)))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
            
            trends[metric] = {
                'slope': slope,
                'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                'r_squared': r_value**2
            }
        
        return trends
    
    # 新增的质量分析辅助方法
    def _analyze_precision(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析精度维度"""
        execution_data = execution_results.get('execution_data', [])
        if not execution_data:
            return {'precision_score': 0.0, 'analysis': '无数据可供分析'}
        
        precision_errors = []
        for result in execution_data:
            actual = result.get('actual_position', [0, 0, 0])
            target = result.get('target_position', [0, 0, 0])
            error = np.linalg.norm(np.array(actual) - np.array(target))
            precision_errors.append(error)
        
        mean_error = np.mean(precision_errors)
        max_error = np.max(precision_errors)
        
        return {
            'precision_score': max(0, 1 - mean_error / 0.1),  # 10cm基准
            'mean_error': mean_error,
            'max_error': max_error,
            'error_std': np.std(precision_errors),
            'precision_grade': self._grade_precision(max(0, 1 - mean_error / 0.1))
        }
    
    def _analyze_consistency(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析一致性维度"""
        execution_data = execution_results.get('execution_data', [])
        if len(execution_data) < 2:
            return {'consistency_score': 0.0, 'analysis': '数据不足，无法分析一致性'}
        
        performance_scores = []
        for result in execution_data:
            score = self._calculate_performance_score(result)
            performance_scores.append(score)
        
        cv = np.std(performance_scores) / np.mean(performance_scores) if np.mean(performance_scores) > 0 else float('inf')
        
        return {
            'consistency_score': max(0, 1 - cv),
            'coefficient_of_variation': cv,
            'score_range': max(performance_scores) - min(performance_scores),
            'consistency_grade': self._grade_consistency(max(0, 1 - cv))
        }
    
    def _analyze_robustness(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析鲁棒性维度"""
        execution_data = execution_results.get('execution_data', [])
        if not execution_data:
            return {'robustness_score': 0.0, 'analysis': '无数据可供分析'}
        
        # 分析在干扰情况下的表现
        robustness_scores = []
        for result in execution_data:
            interference_level = result.get('environmental_interference', 0)
            performance = self._calculate_performance_score(result)
            
            # 在干扰情况下的性能保持度
            robustness = performance / (1 + interference_level * 0.5)
            robustness_scores.append(robustness)
        
        mean_robustness = np.mean(robustness_scores)
        
        return {
            'robustness_score': mean_robustness,
            'performance_degradation': 1 - mean_robustness,
            'interference_tolerance': self._assess_interference_tolerance(execution_data),
            'robustness_grade': self._grade_robustness(mean_robustness)
        }
    
    def _analyze_transfer_efficiency(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析迁移效率维度"""
        minecraft_performance = adapted_strategy.get('minecraft_performance', 0.8)
        physical_performance = execution_results.get('overall_score', 0.0)
        
        efficiency_ratio = physical_performance / minecraft_performance if minecraft_performance > 0 else 0
        adaptation_cost = execution_results.get('adaptation_cost', 0.0)
        
        return {
            'efficiency_score': efficiency_ratio,
            'adaptation_cost': adaptation_cost,
            'cost_effectiveness': physical_performance / (adaptation_cost + 0.01),  # 避免除零
            'efficiency_grade': self._grade_efficiency(efficiency_ratio)
        }
    
    def _analyze_scalability(self, adapted_strategy: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析可扩展性维度"""
        # 分析策略在不同规模任务下的表现
        task_complexities = execution_results.get('task_complexities', [1, 2, 3])
        performance_at_complexity = execution_results.get('performance_by_complexity', {})
        
        if not performance_at_complexity:
            return {'scalability_score': 0.5, 'analysis': '缺少复杂度数据'}
        
        # 计算性能随复杂度变化的趋势
        complexities = list(performance_at_complexity.keys())
        performances = list(performance_at_complexity.values())
        
        if len(complexities) < 2:
            return {'scalability_score': 0.5, 'analysis': '数据点不足'}
        
        # 线性回归分析趋势
        slope, _, r_value, _, _ = stats.linregress(complexities, performances)
        
        # 斜率接近0表示较好的可扩展性（性能不会因复杂度增加而急剧下降）
        scalability_score = max(0, 1 - abs(slope) / 0.1)
        
        return {
            'scalability_score': scalability_score,
            'performance_trend_slope': slope,
            'scalability_correlation': r_value,
            'scalability_grade': self._grade_scalability(scalability_score)
        }
    
    def _calculate_overall_quality_score(self, quality_analysis: Dict[str, Any]) -> float:
        """计算综合质量分数"""
        if not quality_analysis:
            return 0.0
        
        weights = {
            'precision': 0.25,
            'consistency': 0.20,
            'robustness': 0.20,
            'efficiency': 0.20,
            'scalability': 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, analysis in quality_analysis.items():
            if dimension in weights and isinstance(analysis, dict):
                score = analysis.get(f'{dimension}_score', 0.0)
                weight = weights[dimension]
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_grade(self, quality_score: float) -> str:
        """确定质量等级"""
        if quality_score >= 0.9:
            return 'A+'
        elif quality_score >= 0.8:
            return 'A'
        elif quality_score >= 0.7:
            return 'B+'
        elif quality_score >= 0.6:
            return 'B'
        elif quality_score >= 0.5:
            return 'C+'
        elif quality_score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def _identify_quality_issues(self, quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别质量问题"""
        issues = []
        
        for dimension, analysis in quality_analysis.items():
            if isinstance(analysis, dict):
                score = analysis.get(f'{dimension}_score', 0.0)
                if score < 0.6:
                    issues.append({
                        'dimension': dimension,
                        'severity': 'high' if score < 0.4 else 'medium',
                        'score': score,
                        'issue_description': self._get_dimension_issue_description(dimension, score)
                    })
        
        return issues
    
    def _generate_quality_improvement_suggestions(self, quality_analysis: Dict[str, Any], 
                                                quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成质量改进建议"""
        suggestions = []
        
        for issue in quality_issues:
            dimension = issue['dimension']
            severity = issue['severity']
            
            suggestion = {
                'dimension': dimension,
                'severity': severity,
                'priority': 1 if severity == 'high' else 2,
                'actions': self._get_quality_improvement_actions(dimension),
                'expected_impact': self._estimate_quality_improvement_impact(dimension, issue['score']),
                'implementation_difficulty': self._assess_implementation_difficulty(dimension)
            }
            suggestions.append(suggestion)
        
        return sorted(suggestions, key=lambda x: x['priority'])
    
    # 新增的策略对比辅助方法
    def _perform_strategy_comparison(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """执行策略对比分析"""
        if len(strategy_evaluations) < 2:
            return {'comparison_status': 'insufficient_strategies'}
        
        # 提取所有指标进行比较
        all_metrics = set()
        for evaluation in strategy_evaluations.values():
            all_metrics.update(evaluation.get('metrics', {}).keys())
        
        comparison_results = {}
        
        for metric in all_metrics:
            metric_scores = {}
            for strategy_name, evaluation in strategy_evaluations.items():
                metric_scores[strategy_name] = evaluation.get('metrics', {}).get(metric, 0.0)
            
            comparison_results[metric] = self._compare_metric_across_strategies(metric_scores)
        
        return {
            'metric_comparisons': comparison_results,
            'overall_rankings': self._calculate_overall_rankings(strategy_evaluations),
            'statistical_significance': self._test_strategic_differences(strategy_evaluations)
        }
    
    def _rank_strategies(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略排序"""
        strategies_ranked = []
        
        for strategy_name, evaluation in strategy_evaluations.items():
            strategies_ranked.append({
                'strategy_name': strategy_name,
                'overall_score': evaluation.get('overall_score', 0.0),
                'metrics': evaluation.get('metrics', {}),
                'rank': 0  # 待填充
            })
        
        # 按总体评分排序
        strategies_ranked.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # 添加排名信息
        for i, strategy in enumerate(strategies_ranked):
            strategy['rank'] = i + 1
        
        return strategies_ranked
    
    def _analyze_strategy_features(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析策略特征"""
        features = {
            'strengths': defaultdict(list),
            'weaknesses': defaultdict(list),
            'unique_characteristics': {},
            'common_patterns': []
        }
        
        for strategy_name, evaluation in strategy_evaluations.items():
            metrics = evaluation.get('metrics', {})
            
            # 识别每个策略的优势和劣势
            for metric, score in metrics.items():
                if score > 0.8:
                    features['strengths'][metric].append(strategy_name)
                elif score < 0.5:
                    features['weaknesses'][metric].append(strategy_name)
            
            # 识别独特特征
            unique_metrics = [m for m, s in metrics.items() if s > 0.9]
            if unique_metrics:
                features['unique_characteristics'][strategy_name] = unique_metrics
        
        # 转换为普通字典
        features['strengths'] = dict(features['strengths'])
        features['weaknesses'] = dict(features['weaknesses'])
        
        return features
    
    # 新增的改进建议辅助方法
    def _generate_immediate_actions(self, metrics: Dict[str, float], overall_score: float) -> List[Dict[str, Any]]:
        """生成即时行动建议"""
        actions = []
        
        # 基于低分指标生成即时行动
        for metric, score in metrics.items():
            if score < 0.5:
                actions.append({
                    'action': f'立即修复{metric}问题',
                    'priority': 'high',
                    'estimated_time': '1-3天',
                    'expected_impact': f'将{metric}提升至{score + 0.3:.1f}',
                    'resource_requirement': '低'
                })
        
        # 基于总体分数的通用建议
        if overall_score < 0.6:
            actions.append({
                'action': '进行系统性能诊断',
                'priority': 'critical',
                'estimated_time': '1周',
                'expected_impact': '识别所有性能瓶颈',
                'resource_requirement': '中'
            })
        
        return actions
    
    def _generate_short_term_improvements(self, metrics: Dict[str, float], 
                                        performance_comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成短期改进建议"""
        improvements = []
        
        # 基于性能对比的改进建议
        target_comparison = performance_comparison.get('target_performance', {})
        for area in target_comparison.get('degradation_areas', []):
            if area['percentage'] > 5:
                improvements.append({
                    'improvement': f'优化{area["metric"]}性能',
                    'target_improvement': f'{area["percentage"]}%',
                    'timeframe': '2-4周',
                    'difficulty': 'medium',
                    'specific_actions': self._get_metric_improvement_actions(area['metric'])
                })
        
        return improvements
    
    def _generate_long_term_optimizations(self, statistical_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成长期优化建议"""
        optimizations = []
        
        # 基于统计分析的长期建议
        confidence_interval = statistical_analysis.get('confidence_interval', {})
        if confidence_interval:
            margin = confidence_interval.get('margin_of_error', 0)
            if margin > 0.1:
                optimizations.append({
                    'optimization': '提高系统一致性和稳定性',
                    'rationale': f'当前性能波动较大（误差{margin:.2f}）',
                    'timeframe': '1-3个月',
                    'investment_level': '高'
                })
        
        return optimizations
    
    def _generate_strategic_recommendations(self, evaluation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成战略性建议"""
        recommendations = []
        
        overall_score = evaluation_result.get('overall_score', 0.0)
        
        # 基于总体表现给出战略建议
        if overall_score < 0.7:
            recommendations.append({
                'recommendation': '考虑重新设计迁移策略框架',
                'reason': '当前策略整体表现不理想',
                'timeline': '3-6个月',
                'scope': 'comprehensive'
            })
        elif overall_score > 0.85:
            recommendations.append({
                'recommendation': '扩展策略应用范围',
                'reason': '当前策略表现优秀，可推广至更多场景',
                'timeline': '1-2个月',
                'scope': 'expansion'
            })
        
        return recommendations
    
    # 新增的辅助方法实现
    def _grade_precision(self, score: float) -> str:
        """精度等级评估"""
        if score >= 0.9: return '优秀'
        elif score >= 0.8: return '良好'
        elif score >= 0.7: return '中等'
        else: return '待改进'
    
    def _grade_consistency(self, score: float) -> str:
        """一致性等级评估"""
        if score >= 0.8: return '高度一致'
        elif score >= 0.6: return '基本一致'
        elif score >= 0.4: return '一般一致'
        else: return '不一致'
    
    def _grade_robustness(self, score: float) -> str:
        """鲁棒性等级评估"""
        if score >= 0.8: return '强鲁棒性'
        elif score >= 0.6: return '中等鲁棒性'
        elif score >= 0.4: return '弱鲁棒性'
        else: return '无鲁棒性'
    
    def _grade_efficiency(self, score: float) -> str:
        """效率等级评估"""
        if score >= 1.0: return '高效迁移'
        elif score >= 0.8: return '较高效迁移'
        elif score >= 0.6: return '一般迁移'
        else: return '低效迁移'
    
    def _grade_scalability(self, score: float) -> str:
        """可扩展性等级评估"""
        if score >= 0.8: return '高度可扩展'
        elif score >= 0.6: return '中度可扩展'
        elif score >= 0.4: return '低度可扩展'
        else: return '不可扩展'
    
    def _assess_interference_tolerance(self, execution_data: List[Dict[str, Any]]) -> float:
        """评估干扰容忍度"""
        if not execution_data:
            return 0.0
        
        tolerance_scores = []
        for result in execution_data:
            interference = result.get('environmental_interference', 0)
            performance = self._calculate_performance_score(result)
            # 在给定干扰水平下的性能保持度
            tolerance = performance / (1 + interference * 0.3)
            tolerance_scores.append(tolerance)
        
        return np.mean(tolerance_scores) if tolerance_scores else 0.0
    
    def _get_dimension_issue_description(self, dimension: str, score: float) -> str:
        """获取维度问题描述"""
        descriptions = {
            'precision': f'精度不足（{score:.2f}），误差过大',
            'consistency': f'一致性差（{score:.2f}），性能波动明显',
            'robustness': f'鲁棒性弱（{score:.2f}），受环境影响大',
            'efficiency': f'迁移效率低（{score:.2f}），资源消耗过多',
            'scalability': f'可扩展性差（{score:.2f}），无法适应复杂任务'
        }
        return descriptions.get(dimension, f'{dimension}存在性能问题')
    
    def _get_quality_improvement_actions(self, dimension: str) -> List[str]:
        """获取质量改进行动"""
        actions_map = {
            'precision': ['提高传感器精度', '优化控制算法', '改进机械结构'],
            'consistency': ['标准化操作流程', '增加质量控制', '减少随机因素'],
            'robustness': ['增强系统鲁棒性', '改进容错机制', '优化自适应算法'],
            'efficiency': ['优化资源分配', '改进算法效率', '减少不必要计算'],
            'scalability': ['模块化设计', '优化算法复杂度', '增加并行处理']
        }
        return actions_map.get(dimension, ['需要进一步分析'])
    
    def _estimate_quality_improvement_impact(self, dimension: str, current_score: float) -> Dict[str, Any]:
        """估算质量改进影响"""
        improvement_potential = {
            'precision': 0.3,
            'consistency': 0.4,
            'robustness': 0.35,
            'efficiency': 0.25,
            'scalability': 0.2
        }
        
        potential = improvement_potential.get(dimension, 0.2)
        
        return {
            'max_improvement': min(0.5, potential),
            'realistic_improvement': potential * 0.6,
            'implementation_time': '2-4周' if potential > 0.3 else '1-2周'
        }
    
    def _assess_implementation_difficulty(self, dimension: str) -> str:
        """评估实施难度"""
        difficulty_map = {
            'precision': 'medium',
            'consistency': 'low',
            'robustness': 'high',
            'efficiency': 'medium',
            'scalability': 'high'
        }
        return difficulty_map.get(dimension, 'medium')
    
    def _calculate_quality_analysis_confidence(self, quality_analysis: Dict[str, Any]) -> float:
        """计算质量分析置信度"""
        if not quality_analysis:
            return 0.0
        
        dimension_count = len(quality_analysis)
        confidence_factors = []
        
        for dimension, analysis in quality_analysis.items():
            if isinstance(analysis, dict):
                score = analysis.get(f'{dimension}_score', 0.0)
                # 分数越高质量分析越可信
                confidence_factors.append(score)
        
        if not confidence_factors:
            return 0.5
        
        avg_confidence = np.mean(confidence_factors)
        dimension_factor = min(1.0, dimension_count / 5)
        
        return avg_confidence * dimension_factor
    
    def _assess_data_sufficiency(self, execution_results: Dict[str, Any]) -> str:
        """评估数据充分性"""
        execution_data = execution_results.get('execution_data', [])
        data_count = len(execution_data)
        
        if data_count >= 10:
            return '充分'
        elif data_count >= 5:
            return '基本充足'
        elif data_count >= 2:
            return '不足'
        else:
            return '严重不足'
    
    def _compare_metric_across_strategies(self, metric_scores: Dict[str, float]) -> Dict[str, Any]:
        """比较指标跨策略的表现"""
        if not metric_scores:
            return {}
        
        values = list(metric_scores.values())
        
        return {
            'scores': metric_scores,
            'best_strategy': max(metric_scores.items(), key=lambda x: x[1])[0],
            'worst_strategy': min(metric_scores.items(), key=lambda x: x[1])[0],
            'mean_score': np.mean(values),
            'std_score': np.std(values),
            'range': max(values) - min(values),
            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        }
    
    def _calculate_overall_rankings(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算整体排名"""
        rankings = []
        
        for strategy_name, evaluation in strategy_evaluations.items():
            rankings.append({
                'strategy_name': strategy_name,
                'overall_score': evaluation.get('overall_score', 0.0),
                'rank_position': 0  # 待填充
            })
        
        # 按分数排序
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # 填充排名
        for i, strategy in enumerate(rankings):
            strategy['rank_position'] = i + 1
        
        return rankings
    
    def _test_strategic_differences(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """测试策略差异的统计显著性"""
        if len(strategy_evaluations) < 2:
            return {'test_status': 'insufficient_strategies'}
        
        # 提取所有策略的总体分数
        strategy_scores = []
        strategy_names = []
        
        for strategy_name, evaluation in strategy_evaluations.items():
            strategy_scores.append(evaluation.get('overall_score', 0.0))
            strategy_names.append(strategy_name)
        
        if len(set(strategy_scores)) < 2:
            return {'test_status': 'no_variance'}
        
        # 简化的方差分析（实际应用中需要更复杂的统计检验）
        mean_score = np.mean(strategy_scores)
        variance = np.var(strategy_scores)
        
        return {
            'test_status': 'completed',
            'mean_difference': variance,
            'significance_level': 'low' if variance < 0.1 else 'medium' if variance < 0.2 else 'high',
            'conclusion': '策略间存在显著差异' if variance > 0.1 else '策略间差异不显著'
        }
    
    def _generate_strategy_selection_recommendations(self, ranked_strategies: List[Dict[str, Any]], 
                                                  comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成策略选择建议"""
        recommendations = []
        
        if not ranked_strategies:
            return recommendations
        
        best_strategy = ranked_strategies[0]
        
        # 推荐最佳策略
        recommendations.append({
            'recommendation': f'推荐使用{best_strategy["strategy_name"]}策略',
            'rationale': f'该策略综合评分最高（{best_strategy["overall_score"]:.2f}）',
            'confidence': 'high' if best_strategy['overall_score'] > 0.8 else 'medium'
        })
        
        # 根据不同需求场景给出建议
        if len(ranked_strategies) >= 2:
            second_best = ranked_strategies[1]
            recommendations.append({
                'recommendation': f'备选策略：{second_best["strategy_name"]}',
                'rationale': f'性能稳定，可作为备选方案（{second_best["overall_score"]:.2f}）',
                'confidence': 'medium'
            })
        
        return recommendations
    
    def _calculate_average_performance(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> float:
        """计算平均性能"""
        if not strategy_evaluations:
            return 0.0
        
        scores = [eval_data.get('overall_score', 0.0) for eval_data in strategy_evaluations.values()]
        return np.mean(scores)
    
    def _calculate_performance_variance(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> float:
        """计算性能方差"""
        if not strategy_evaluations:
            return 0.0
        
        scores = [eval_data.get('overall_score', 0.0) for eval_data in strategy_evaluations.values()]
        return np.var(scores)
    
    def _identify_common_strengths(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> List[str]:
        """识别共同优势"""
        metric_strengths = defaultdict(int)
        
        for evaluation in strategy_evaluations.values():
            metrics = evaluation.get('metrics', {})
            for metric, score in metrics.items():
                if score > 0.8:
                    metric_strengths[metric] += 1
        
        # 返回有多个策略表现优秀的指标
        common_strengths = [metric for metric, count in metric_strengths.items() if count >= 2]
        return common_strengths
    
    def _identify_common_weaknesses(self, strategy_evaluations: Dict[str, Dict[str, Any]]) -> List[str]:
        """识别共同劣势"""
        metric_weaknesses = defaultdict(int)
        
        for evaluation in strategy_evaluations.values():
            metrics = evaluation.get('metrics', {})
            for metric, score in metrics.items():
                if score < 0.5:
                    metric_weaknesses[metric] += 1
        
        # 返回有多个策略表现不佳的指标
        common_weaknesses = [metric for metric, count in metric_weaknesses.items() if count >= 2]
        return common_weaknesses
    
    def _prioritize_suggestions(self, improvement_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """对建议进行优先级排序"""
        prioritized = {}
        
        for category, suggestions in improvement_suggestions.items():
            if isinstance(suggestions, list):
                # 按优先级排序（简单的基于严重性）
                sorted_suggestions = sorted(suggestions, 
                                          key=lambda x: (x.get('priority', 2), -x.get('score', 0)))
                prioritized[category] = sorted_suggestions
            else:
                prioritized[category] = suggestions
        
        return prioritized
    
    def _create_implementation_plan(self, improvement_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """创建实施计划"""
        plan = {
            'immediate_actions': [],
            'short_term_tasks': [],
            'long_term_goals': []
        }
        
        for category, suggestions in improvement_suggestions.items():
            if category == 'immediate_actions':
                plan['immediate_actions'] = suggestions
            elif category in ['short_term_improvements']:
                plan['short_term_tasks'].extend(suggestions)
            elif category in ['long_term_optimizations', 'strategic_recommendations']:
                plan['long_term_goals'].extend(suggestions)
        
        return plan
    
    def _forecast_improvement_potential(self, improvement_suggestions: Dict[str, Any], 
                                      current_score: float) -> Dict[str, Any]:
        """预测改进潜力"""
        total_improvement = 0.0
        confidence_factors = []
        
        for category, suggestions in improvement_suggestions.items():
            if isinstance(suggestions, list):
                for suggestion in suggestions:
                    if isinstance(suggestion, dict):
                        # 估算每个建议的改进潜力
                        estimated_improvement = 0.1  # 默认改进10%
                        total_improvement += estimated_improvement
                        confidence_factors.append(0.7)  # 默认70%置信度
        
        return {
            'projected_overall_score': min(1.0, current_score + total_improvement),
            'improvement_magnitude': total_improvement,
            'confidence_level': np.mean(confidence_factors) if confidence_factors else 0.5,
            'implementation_timeline': '2-6个月'
        }
    
    def _calculate_suggestion_confidence(self, improvement_suggestions: Dict[str, Any]) -> float:
        """计算建议置信度"""
        total_suggestions = 0
        confident_suggestions = 0
        
        for category, suggestions in improvement_suggestions.items():
            if isinstance(suggestions, list):
                for suggestion in suggestions:
                    total_suggestions += 1
                    # 基于建议的详细程度和特异性评估置信度
                    if isinstance(suggestion, dict):
                        if 'specific_actions' in suggestion and len(suggestion.get('specific_actions', [])) > 0:
                            confident_suggestions += 1
                        elif 'estimated_time' in suggestion:
                            confident_suggestions += 0.8
                        else:
                            confident_suggestions += 0.5
        
        return confident_suggestions / total_suggestions if total_suggestions > 0 else 0.0
    
    def _summarize_recommendations(self, improvement_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """总结建议"""
        total_count = 0
        category_counts = {}
        priority_high_count = 0
        
        for category, suggestions in improvement_suggestions.items():
            if isinstance(suggestions, list):
                category_counts[category] = len(suggestions)
                total_count += len(suggestions)
                
                for suggestion in suggestions:
                    if isinstance(suggestion, dict) and suggestion.get('priority') in ['high', 'critical']:
                        priority_high_count += 1
        
        return {
            'total_recommendations': total_count,
            'category_distribution': category_counts,
            'high_priority_count': priority_high_count,
            'priority_ratio': priority_high_count / total_count if total_count > 0 else 0
        }
    
    def _get_metric_improvement_actions(self, metric: str) -> List[str]:
        """获取指标改进行动"""
        actions_map = {
            'accuracy': ['提高传感器精度', '优化控制系统', '改进算法参数'],
            'success_rate': ['完善异常处理', '优化任务规划', '增加容错机制'],
            'execution_time': ['优化算法效率', '并行化处理', '减少计算复杂度'],
            'stability': ['增加滤波机制', '优化控制参数', '改进机械设计'],
            'adaptability': ['增强学习算法', '改进环境感知', '优化参数自适应']
        }
        return actions_map.get(metric, ['需要进一步分析'])
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {'status': 'no_evaluations_completed'}
        
        latest_evaluation = self.evaluation_history[-1]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'latest_overall_score': latest_evaluation['overall_score'],
            'average_performance': {
                metric: np.mean([eval_data['metrics'].get(metric, 0) 
                               for eval_data in self.evaluation_history])
                for metric in self.evaluation_metrics
            },
            'improvement_trends': self._calculate_improvement_trends(),
            'last_evaluation_time': latest_evaluation['evaluation_timestamp']
        }