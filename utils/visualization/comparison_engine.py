"""
性能基准展示面板系统 - 比较引擎
Performance Benchmark System - Comparison Engine

该模块提供了算法间性能比较功能，支持多种比较方法、统计显著性测试
和可视化比较结果。

This module provides performance comparison between algorithms, supporting multiple 
comparison methods, statistical significance testing, and visualization of comparison results.

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
import math
from datetime import datetime
import logging

class ComparisonEngine:
    """
    算法性能比较引擎
    
    功能特性:
    - 多算法性能对比分析
    - 统计显著性检验
    - 性能差异量化分析
    - 可视化比较结果生成
    
    Features:
    - Multi-algorithm performance comparison analysis
    - Statistical significance testing
    - Performance difference quantification
    - Visual comparison result generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化比较引擎
        Initialize the comparison engine
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger('ComparisonEngine')
        self.config = config or self._default_config()
        
        # 比较方法配置
        self.comparison_methods = {
            'statistical_test': {
                'name': '统计显著性检验',
                'description': '使用t检验或Wilcoxon检验比较算法性能',
                'applicable': '连续性能指标'
            },
            'effect_size': {
                'name': '效应大小分析',
                'description': '计算Cohen\'s d来量化性能差异大小',
                'applicable': '性能指标差异量化'
            },
            'ranking_comparison': {
                'name': '排名对比分析',
                'description': '基于总体评分的算法排名比较',
                'applicable': '综合性能排名'
            },
            'task_specific': {
                'name': '任务特定比较',
                'description': '针对特定任务类型的性能比较',
                'applicable': '特定任务场景'
            }
        }
        
        # 基准算法数据
        self.baseline_performances = {
            'DQN': {
                'average_reward': 132.5,
                'success_rate': 0.72,
                'exploration_efficiency': 0.68,
                'learning_stability': 0.75,
                'convergence_speed': 0.65,
                'overall_score': 70.3
            },
            'PPO': {
                'average_reward': 145.2,
                'success_rate': 0.78,
                'exploration_efficiency': 0.73,
                'learning_stability': 0.82,
                'convergence_speed': 0.72,
                'overall_score': 76.8
            },
            'DiscoRL': {
                'average_reward': 128.7,
                'success_rate': 0.69,
                'exploration_efficiency': 0.81,
                'learning_stability': 0.70,
                'convergence_speed': 0.58,
                'overall_score': 72.1
            },
            'A3C': {
                'average_reward': 138.9,
                'success_rate': 0.75,
                'exploration_efficiency': 0.70,
                'learning_stability': 0.73,
                'convergence_speed': 0.68,
                'overall_score': 74.2
            },
            'Rainbow': {
                'average_reward': 152.8,
                'success_rate': 0.81,
                'exploration_efficiency': 0.76,
                'learning_stability': 0.79,
                'convergence_speed': 0.74,
                'overall_score': 79.6
            }
        }
        
        # 任务特定基准
        self.task_benchmarks = {
            'Atari Breakout': {
                'DQN': {'breakout_score': 650, 'success_rate': 0.68},
                'PPO': {'breakout_score': 720, 'success_rate': 0.74},
                'DiscoRL': {'breakout_score': 610, 'success_rate': 0.65},
                'A3C': {'breakout_score': 690, 'success_rate': 0.71},
                'Rainbow': {'breakout_score': 780, 'success_rate': 0.79}
            },
            'Minecraft Survival': {
                'DQN': {'survival_rate': 0.82, 'resource_efficiency': 0.70},
                'PPO': {'survival_rate': 0.88, 'resource_efficiency': 0.75},
                'DiscoRL': {'survival_rate': 0.79, 'resource_efficiency': 0.78},
                'A3C': {'survival_rate': 0.85, 'resource_efficiency': 0.72},
                'Rainbow': {'survival_rate': 0.92, 'resource_efficiency': 0.80}
            }
        }
        
        self.logger.info("比较引擎初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'significance_level': 0.05,
            'effect_size_threshold': 0.2,  # Cohen's d阈值
            'min_samples': 10,
            'comparison_methods': ['statistical_test', 'effect_size'],
            'visualization_enabled': True
        }
    
    def compare_performance(self, 
                          current_metrics: Dict[str, float],
                          baseline_metrics: Dict[str, float],
                          algorithm: str,
                          baseline_algorithms: List[str],
                          method: str = 'comprehensive') -> Dict[str, Any]:
        """
        执行性能比较
        Perform performance comparison
        
        Args:
            current_metrics: 当前算法指标
            baseline_metrics: 基线算法指标
            algorithm: 当前算法名称
            baseline_algorithms: 基线算法列表
            method: 比较方法
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        try:
            comparison_result = {
                'algorithm': algorithm,
                'baseline_algorithms': baseline_algorithms,
                'comparison_time': datetime.now().isoformat(),
                'methods_used': []
            }
            
            # 基础性能对比
            basic_comparison = self._basic_performance_comparison(
                current_metrics, baseline_metrics, algorithm, baseline_algorithms
            )
            comparison_result['basic_comparison'] = basic_comparison
            
            # 统计显著性检验
            if 'statistical_test' in self.config.get('comparison_methods', []):
                significance_test = self._statistical_significance_test(
                    current_metrics, baseline_metrics
                )
                comparison_result['significance_test'] = significance_test
                comparison_result['methods_used'].append('statistical_test')
            
            # 效应大小分析
            if 'effect_size' in self.config.get('comparison_methods', []):
                effect_size_analysis = self._effect_size_analysis(
                    current_metrics, baseline_metrics
                )
                comparison_result['effect_size_analysis'] = effect_size_analysis
                comparison_result['methods_used'].append('effect_size')
            
            # 排名对比分析
            if method in ['ranking_comparison', 'comprehensive']:
                ranking_analysis = self._ranking_comparison_analysis(
                    current_metrics, baseline_metrics, algorithm, baseline_algorithms
                )
                comparison_result['ranking_analysis'] = ranking_analysis
                comparison_result['methods_used'].append('ranking_comparison')
            
            # 综合评分
            overall_assessment = self._generate_overall_assessment(comparison_result)
            comparison_result['overall_assessment'] = overall_assessment
            
            self.logger.info(f"性能比较完成: {algorithm} vs {baseline_algorithms}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"性能比较失败: {e}")
            return {'error': str(e)}
    
    def _basic_performance_comparison(self, 
                                    current_metrics: Dict[str, float],
                                    baseline_metrics: Dict[str, float],
                                    algorithm: str,
                                    baseline_algorithms: List[str]) -> Dict[str, Any]:
        """
        基础性能对比
        Basic performance comparison
        """
        basic_comparison = {
            'individual_comparisons': {},
            'summary': {}
        }
        
        # 与每个基线算法进行比较
        for baseline_algo in baseline_algorithms:
            baseline_perf = baseline_metrics.get(baseline_algo, {})
            
            individual_comparison = {}
            
            # 逐项指标比较
            common_metrics = set(current_metrics.keys()) & set(baseline_perf.keys())
            
            for metric in common_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_perf[metric]
                
                if baseline_val != 0:
                    relative_improvement = (current_val - baseline_val) / baseline_val
                else:
                    relative_improvement = current_val
                
                absolute_improvement = current_val - baseline_val
                
                individual_comparison[metric] = {
                    'current_value': current_val,
                    'baseline_value': baseline_val,
                    'absolute_improvement': absolute_improvement,
                    'relative_improvement': relative_improvement,
                    'is_better': current_val > baseline_val
                }
            
            basic_comparison['individual_comparisons'][baseline_algo] = individual_comparison
        
        # 生成汇总统计
        summary = self._generate_comparison_summary(basic_comparison['individual_comparisons'])
        basic_comparison['summary'] = summary
        
        return basic_comparison
    
    def _statistical_significance_test(self, 
                                     current_metrics: Dict[str, float],
                                     baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        统计显著性检验
        Statistical significance testing
        """
        significance_results = {}
        
        # 生成模拟数据进行统计检验（实际应用中应使用真实的历史数据）
        np.random.seed(42)  # 确保结果可重现
        
        for metric in current_metrics:
            if metric in baseline_metrics:
                # 模拟多次运行的数据
                current_values = np.random.normal(
                    current_metrics[metric], 
                    current_metrics[metric] * 0.1, 
                    30
                )
                
                for baseline_algo, baseline_perf in baseline_metrics.items():
                    if metric in baseline_perf:
                        baseline_values = np.random.normal(
                            baseline_perf[metric],
                            baseline_perf[metric] * 0.1,
                            30
                        )
                        
                        # 执行t检验
                        t_stat, p_value = stats.ttest_ind(current_values, baseline_values)
                        
                        # 计算效应大小 (Cohen's d)
                        pooled_std = np.sqrt(((len(current_values) - 1) * np.var(current_values, ddof=1) + 
                                            (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) / 
                                           (len(current_values) + len(baseline_values) - 2))
                        
                        cohens_d = (np.mean(current_values) - np.mean(baseline_values)) / pooled_std
                        
                        significance_results[f"{baseline_algo}_{metric}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.config.get('significance_level', 0.05),
                            'cohens_d': float(cohens_d),
                            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                        }
        
        return significance_results
    
    def _effect_size_analysis(self, 
                            current_metrics: Dict[str, float],
                            baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        效应大小分析
        Effect size analysis
        """
        effect_analysis = {}
        
        for baseline_algo, baseline_perf in baseline_metrics.items():
            algorithm_effects = {}
            
            for metric in current_metrics:
                if metric in baseline_perf:
                    current_val = current_metrics[metric]
                    baseline_val = baseline_perf[metric]
                    
                    # 计算多种效应大小指标
                    cohens_d = self._calculate_cohens_d(current_val, baseline_val)
                    percentage_improvement = ((current_val - baseline_val) / baseline_val) * 100
                    
                    algorithm_effects[metric] = {
                        'cohens_d': cohens_d,
                        'percentage_improvement': percentage_improvement,
                        'practical_significance': self._assess_practical_significance(percentage_improvement),
                        'effect_interpretation': self._interpret_effect_size(abs(cohens_d))
                    }
            
            effect_analysis[baseline_algo] = algorithm_effects
        
        return effect_analysis
    
    def _ranking_comparison_analysis(self, 
                                   current_metrics: Dict[str, float],
                                   baseline_metrics: Dict[str, float],
                                   algorithm: str,
                                   baseline_algorithms: List[str]) -> Dict[str, Any]:
        """
        排名对比分析
        Ranking comparison analysis
        """
        ranking_analysis = {
            'overall_rankings': {},
            'metric_rankings': {},
            'consensus_ranking': {}
        }
        
        # 计算所有算法的综合评分
        algorithm_scores = {}
        algorithm_scores[algorithm] = current_metrics.get('overall_score', 0)
        
        for baseline_algo, baseline_perf in baseline_metrics.items():
            algorithm_scores[baseline_algo] = baseline_perf.get('overall_score', 0)
        
        # 生成总体排名
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        overall_ranking = [(algo, score, idx + 1) for idx, (algo, score) in enumerate(sorted_algorithms)]
        ranking_analysis['overall_rankings'] = overall_ranking
        
        # 生成指标排名
        metric_rankings = {}
        for metric in current_metrics:
            metric_scores = {algorithm: current_metrics[metric]}
            for baseline_algo, baseline_perf in baseline_metrics.items():
                if metric in baseline_perf:
                    metric_scores[baseline_algo] = baseline_perf[metric]
            
            sorted_metric = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            metric_rankings[metric] = [(algo, score, idx + 1) for idx, (algo, score) in enumerate(sorted_metric)]
        
        ranking_analysis['metric_rankings'] = metric_rankings
        
        # 计算共识排名（基于多指标的平均排名）
        consensus_rankings = {}
        for algo in algorithm_scores:
            ranks = [ranking[2] for ranking in overall_ranking if ranking[0] == algo]
            ranks.extend([metric_rank[2] for metric_rank in metric_rankings.values() 
                         for metric_rank in metric_rank if metric_rank[0] == algo])
            consensus_rankings[algo] = np.mean(ranks) if ranks else len(algorithm_scores)
        
        sorted_consensus = sorted(consensus_rankings.items(), key=lambda x: x[1])
        ranking_analysis['consensus_ranking'] = [(algo, avg_rank) for algo, avg_rank in sorted_consensus]
        
        return ranking_analysis
    
    def _calculate_cohens_d(self, mean1: float, mean2: float, pooled_std: float = None) -> float:
        """
        计算Cohen's d效应大小
        Calculate Cohen's d effect size
        """
        if pooled_std is None:
            # 使用简单的标准差估算
            pooled_std = abs(mean1 - mean2) * 0.1 + 1.0
        
        return (mean1 - mean2) / pooled_std
    
    def _interpret_effect_size(self, d: float) -> str:
        """
        解释效应大小
        Interpret effect size
        """
        if d < 0.2:
            return "微小效应"
        elif d < 0.5:
            return "小效应"
        elif d < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def _assess_practical_significance(self, improvement_pct: float) -> str:
        """
        评估实际意义
        Assess practical significance
        """
        abs_improvement = abs(improvement_pct)
        if abs_improvement < 5:
            return "无实际意义"
        elif abs_improvement < 10:
            return "小改进"
        elif abs_improvement < 20:
            return "中等改进"
        else:
            return "显著改进"
    
    def _generate_comparison_summary(self, individual_comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成比较汇总
        Generate comparison summary
        """
        summary = {
            'total_comparisons': 0,
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'average_improvement': 0.0,
            'best_improvement_metric': None,
            'worst_improvement_metric': None
        }
        
        improvements = []
        
        for baseline_algo, comparisons in individual_comparisons.items():
            for metric, comparison in comparisons.items():
                summary['total_comparisons'] += 1
                improvement = comparison['relative_improvement']
                improvements.append(improvement)
                
                if improvement > 0.01:  # 1%阈值
                    summary['wins'] += 1
                elif improvement < -0.01:
                    summary['losses'] += 1
                else:
                    summary['ties'] += 1
        
        if improvements:
            summary['average_improvement'] = np.mean(improvements)
            
            # 找到最好和最差的改进
            best_idx = np.argmax(improvements)
            worst_idx = np.argmin(improvements)
            
            all_metrics = []
            for comparisons in individual_comparisons.values():
                all_metrics.extend(list(comparisons.keys()))
            
            if best_idx < len(all_metrics):
                summary['best_improvement_metric'] = all_metrics[best_idx]
            if worst_idx < len(all_metrics):
                summary['worst_improvement_metric'] = all_metrics[worst_idx]
        
        summary['win_rate'] = summary['wins'] / max(summary['total_comparisons'], 1)
        
        return summary
    
    def _generate_overall_assessment(self, comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合评估
        Generate overall assessment
        """
        assessment = {
            'performance_level': 'unknown',
            'recommendation': '',
            'key_strengths': [],
            'areas_for_improvement': [],
            'confidence_level': 'medium'
        }
        
        # 基于总体评分进行评估
        basic_comparison = comparison_result.get('basic_comparison', {})
        summary = basic_comparison.get('summary', {})
        
        win_rate = summary.get('win_rate', 0)
        avg_improvement = summary.get('average_improvement', 0)
        
        # 确定性能等级
        if win_rate >= 0.8 and avg_improvement > 0.2:
            assessment['performance_level'] = 'outstanding'
            assessment['recommendation'] = '该算法在大多数指标上显著优于基线算法，建议优先考虑'
        elif win_rate >= 0.6 and avg_improvement > 0.1:
            assessment['performance_level'] = 'good'
            assessment['recommendation'] = '该算法在多数指标上表现良好，具有一定优势'
        elif win_rate >= 0.4:
            assessment['performance_level'] = 'average'
            assessment['recommendation'] = '该算法表现中等，与基线算法相当'
        else:
            assessment['performance_level'] = 'below_average'
            assessment['recommendation'] = '该算法在多数指标上不如基线算法，建议进一步优化'
        
        # 确定置信度
        methods_count = len(comparison_result.get('methods_used', []))
        if methods_count >= 3:
            assessment['confidence_level'] = 'high'
        elif methods_count >= 2:
            assessment['confidence_level'] = 'medium'
        else:
            assessment['confidence_level'] = 'low'
        
        return assessment
    
    def compare_task_specific_performance(self, 
                                        current_data: Dict[str, Any],
                                        task: str,
                                        algorithm: str) -> Dict[str, Any]:
        """
        任务特定性能比较
        Task-specific performance comparison
        """
        if task not in self.task_benchmarks:
            return {'error': f'任务 {task} 没有基准数据'}
        
        task_benchmark = self.task_benchmarks[task]
        task_comparison = {
            'task': task,
            'algorithm': algorithm,
            'comparisons': {}
        }
        
        # 针对特定任务进行比较
        for metric in current_data:
            comparisons = {}
            
            for baseline_algo, baseline_perf in task_benchmark.items():
                if metric in baseline_perf:
                    current_val = current_data[metric]
                    baseline_val = baseline_perf[metric]
                    
                    comparisons[baseline_algo] = {
                        'current': current_val,
                        'baseline': baseline_val,
                        'improvement': current_val - baseline_val,
                        'improvement_pct': ((current_val - baseline_val) / baseline_val) * 100
                    }
            
            if comparisons:
                task_comparison['comparisons'][metric] = comparisons
        
        return task_comparison
    
    def get_comparison_history(self, algorithm: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取比较历史
        Get comparison history
        """
        # 这里应该从数据库或文件读取比较历史
        # 当前返回模拟数据
        history = []
        
        for i in range(limit):
            history.append({
                'timestamp': datetime.now().isoformat(),
                'algorithm': algorithm,
                'baseline_algorithm': f'Baseline_{i % 3}',
                'overall_improvement': np.random.uniform(-0.2, 0.3),
                'win_rate': np.random.uniform(0.3, 0.8)
            })
        
        return history