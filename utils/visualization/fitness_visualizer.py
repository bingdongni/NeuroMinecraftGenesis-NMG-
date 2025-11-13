#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适应度可视化器
Fitness Visualizer

该模块负责计算、统计和可视化适应度相关的数据，包括适应度分布、
趋势分析、最优个体追踪等功能。支持多种图表类型和交互式展示。

功能特性：
- 适应度分布图和统计图表
- 趋势线和预测分析
- 最优个体变化追踪
- 适应度热力图
- 多维度适应度对比

Author: 进化树可视化系统
Date: 2025-11-13
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import math

try:
    from .evolution_tree import Individual
except ImportError:
    from utils.visualization.evolution_tree import Individual


@dataclass
class FitnessMetrics:
    """
    适应度指标
    Fitness Metrics
    """
    generation: int
    individual_count: int
    
    # 基础统计
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    median_fitness: float
    std_fitness: float
    
    # 百分位数
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    percentile_99: float
    
    # 高级统计
    skewness: float
    kurtosis: float
    coefficient_of_variation: float
    
    # 时间戳
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'generation': self.generation,
            'individual_count': self.individual_count,
            'best_fitness': self.best_fitness,
            'worst_fitness': self.worst_fitness,
            'average_fitness': self.average_fitness,
            'median_fitness': self.median_fitness,
            'std_fitness': self.std_fitness,
            'percentile_25': self.percentile_25,
            'percentile_75': self.percentile_75,
            'percentile_90': self.percentile_90,
            'percentile_95': self.percentile_95,
            'percentile_99': self.percentile_99,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'coefficient_of_variation': self.coefficient_of_variation,
            'timestamp': self.timestamp
        }


@dataclass
class TrendAnalysis:
    """
    趋势分析结果
    Trend Analysis Result
    """
    generation_range: Tuple[int, int]
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence_level: float
    prediction_intervals: Dict[str, Tuple[float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'generation_range': self.generation_range,
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared,
            'p_value': self.p_value,
            'trend_direction': self.trend_direction,
            'confidence_level': self.confidence_level,
            'prediction_intervals': {
                k: [v[0], v[1]] for k, v in self.prediction_intervals.items()
            }
        }


class FitnessVisualizer:
    """
    适应度可视化器
    Fitness Visualizer
    
    负责计算适应度统计数据、生成趋势分析、
    创建可视化图表数据等功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化适应度可视化器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._default_config()
        
        # 图表配置
        self.chart_config = self.config.get('chart', {})
        self.chart_width = self.chart_config.get('width', 800)
        self.chart_height = self.chart_config.get('height', 400)
        self.chart_type = self.chart_config.get('type', 'line')  # 'line', 'bar', 'heatmap', 'scatter'
        
        # 统计配置
        self.stats_config = self.config.get('statistics', {})
        self.calculate_percentiles = self.stats_config.get('calculate_percentiles', True)
        self.calculate_advanced_stats = self.stats_config.get('calculate_advanced_stats', True)
        
        # 趋势分析配置
        self.trend_config = self.config.get('trend_analysis', {})
        self.trend_window_size = self.trend_config.get('window_size', 10)
        self.confidence_level = self.trend_config.get('confidence_level', 0.95)
        
        # 数据存储
        self.fitness_history = deque(maxlen=self.config.get('history_size', 1000))
        self.metrics_history = {}
        self.best_individuals = {}  # 追踪每个世代的最佳个体
        self.generation_statistics = {}  # 世代统计数据缓存
        
        # 预测模型
        self.trend_model = None
        self.prediction_enabled = self.config.get('enable_predictions', True)
        
        print("适应度可视化器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'chart': {
                'width': 800,
                'height': 400,
                'type': 'line',
                'show_trend': True,
                'show_annotations': True
            },
            'statistics': {
                'calculate_percentiles': True,
                'calculate_advanced_stats': True,
                'percentiles': [25, 75, 90, 95, 99]
            },
            'trend_analysis': {
                'window_size': 10,
                'confidence_level': 0.95,
                'enable_predictions': True
            },
            'history_size': 1000,
            'enable_predictions': True
        }
    
    def update_generation_stats(self, 
                               individuals: List[Individual],
                               generation: int) -> FitnessMetrics:
        """
        更新世代适应度统计
        
        Args:
            individuals: 当前世代的个体列表
            generation: 世代编号
            
        Returns:
            适应度指标
        """
        try:
            if not individuals:
                # 返回默认指标
                return FitnessMetrics(
                    generation=generation,
                    individual_count=0,
                    best_fitness=0,
                    worst_fitness=0,
                    average_fitness=0,
                    median_fitness=0,
                    std_fitness=0,
                    percentile_25=0,
                    percentile_75=0,
                    percentile_90=0,
                    percentile_95=0,
                    percentile_99=0,
                    skewness=0,
                    kurtosis=0,
                    coefficient_of_variation=0,
                    timestamp=time.time()
                )
            
            # 提取适应度值
            fitness_values = np.array([ind.fitness for ind in individuals])
            
            # 计算基础统计
            best_fitness = np.max(fitness_values)
            worst_fitness = np.min(fitness_values)
            average_fitness = np.mean(fitness_values)
            median_fitness = np.median(fitness_values)
            std_fitness = np.std(fitness_values)
            
            # 计算百分位数
            percentiles = [25, 75, 90, 95, 99]
            percentile_values = np.percentile(fitness_values, percentiles)
            percentile_dict = dict(zip([f'percentile_{p}' for p in percentiles], percentile_values))
            
            # 计算高级统计
            skewness = self._calculate_skewness(fitness_values)
            kurtosis = self._calculate_kurtosis(fitness_values)
            coefficient_of_variation = std_fitness / (average_fitness + 1e-10)
            
            # 创建适应度指标
            metrics = FitnessMetrics(
                generation=generation,
                individual_count=len(individuals),
                best_fitness=best_fitness,
                worst_fitness=worst_fitness,
                average_fitness=average_fitness,
                median_fitness=median_fitness,
                std_fitness=std_fitness,
                percentile_25=percentile_dict['percentile_25'],
                percentile_75=percentile_dict['percentile_75'],
                percentile_90=percentile_dict['percentile_90'],
                percentile_95=percentile_dict['percentile_95'],
                percentile_99=percentile_dict['percentile_99'],
                skewness=skewness,
                kurtosis=kurtosis,
                coefficient_of_variation=coefficient_of_variation,
                timestamp=time.time()
            )
            
            # 更新缓存
            self.metrics_history[generation] = metrics
            self.fitness_history.append((generation, average_fitness, best_fitness))
            
            # 追踪最佳个体
            best_individual = max(individuals, key=lambda x: x.fitness)
            self.best_individuals[generation] = best_individual
            
            # 存储世代统计
            self.generation_statistics[generation] = {
                'individual_count': len(individuals),
                'fitness_range': best_fitness - worst_fitness,
                'diversity_index': coefficient_of_variation,
                'timestamp': time.time()
            }
            
            return metrics
            
        except Exception as e:
            print(f"更新世代 {generation} 适应度统计失败：{str(e)}")
            raise
    
    def get_fitness_trend(self, 
                         individuals: Dict[int, Individual],
                         generations: List[int] = None,
                         metric: str = 'best') -> Dict[str, Any]:
        """
        获取适应度趋势数据
        
        Args:
            individuals: 个体数据字典
            generations: 要分析的世代列表
            metric: 趋势指标 ('best', 'average', 'worst', 'std')
            
        Returns:
            趋势数据
        """
        try:
            # 确定要分析的世代
            if generations is None:
                generations = sorted(set(ind.generation for ind in individuals.values()))
            
            if not generations:
                return {
                    'success': False,
                    'message': '没有可用的世代数据'
                }
            
            # 准备趋势数据
            trend_data = {
                'generations': [],
                'values': [],
                'metric': metric
            }
            
            # 按世代收集数据
            for gen in generations:
                generation_individuals = [ind for ind in individuals.values() if ind.generation == gen]
                
                if generation_individuals:
                    fitness_values = [ind.fitness for ind in generation_individuals]
                    
                    # 根据指标选择值
                    if metric == 'best':
                        value = max(fitness_values)
                    elif metric == 'average':
                        value = np.mean(fitness_values)
                    elif metric == 'worst':
                        value = min(fitness_values)
                    elif metric == 'std':
                        value = np.std(fitness_values)
                    elif metric == 'median':
                        value = np.median(fitness_values)
                    else:
                        value = np.mean(fitness_values)
                    
                    trend_data['generations'].append(gen)
                    trend_data['values'].append(value)
            
            # 执行趋势分析
            if len(trend_data['values']) >= 2:
                analysis = self._analyze_fitness_trend(trend_data)
            else:
                analysis = None
            
            result = {
                'success': True,
                'trend_data': trend_data,
                'analysis': analysis,
                'chart_config': self._get_chart_config(metric),
                'metadata': {
                    'total_points': len(trend_data['values']),
                    'generation_range': (min(generations), max(generations)) if generations else None,
                    'metric_type': metric
                }
            }
            
            return result
            
        except Exception as e:
            print(f"获取适应度趋势失败：{str(e)}")
            return {
                'success': False,
                'message': f'获取适应度趋势失败：{str(e)}'
            }
    
    def create_fitness_distribution_chart(self, 
                                        individuals: List[Individual],
                                        generation: int,
                                        chart_type: str = 'histogram') -> Dict[str, Any]:
        """
        创建适应度分布图表
        
        Args:
            individuals: 个体列表
            generation: 世代编号
            chart_type: 图表类型 ('histogram', 'box', 'violin', 'density')
            
        Returns:
            图表数据
        """
        try:
            if not individuals:
                return {
                    'success': False,
                    'message': '没有个体数据用于创建分布图'
                }
            
            fitness_values = np.array([ind.fitness for ind in individuals])
            
            chart_data = {
                'generation': generation,
                'chart_type': chart_type,
                'individual_count': len(individuals),
                'timestamp': time.time()
            }
            
            if chart_type == 'histogram':
                # 创建直方图数据
                bins = self._calculate_optimal_bins(fitness_values)
                hist, bin_edges = np.histogram(fitness_values, bins=bins)
                
                chart_data.update({
                    'bins': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'x_label': '适应度值',
                    'y_label': '个体数量'
                })
            
            elif chart_type == 'box':
                # 创建箱线图数据
                q1, q3 = np.percentile(fitness_values, [25, 75])
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                
                # 检测异常值
                outliers = fitness_values[(fitness_values < lower_fence) | (fitness_values > upper_fence)]
                
                chart_data.update({
                    'min': np.min(fitness_values),
                    'q1': q1,
                    'median': np.median(fitness_values),
                    'q3': q3,
                    'max': np.max(fitness_values),
                    'outliers': outliers.tolist(),
                    'whisker_low': max(fitness_values[fitness_values >= lower_fence]) if len(fitness_values[fitness_values >= lower_fence]) > 0 else np.min(fitness_values),
                    'whisker_high': min(fitness_values[fitness_values <= upper_fence]) if len(fitness_values[fitness_values <= upper_fence]) > 0 else np.max(fitness_values)
                })
            
            elif chart_type == 'violin':
                # 创建小提琴图数据
                density_points = self._calculate_density_points(fitness_values)
                chart_data.update({
                    'density_x': density_points['x'].tolist(),
                    'density_y': density_points['y'].tolist(),
                    'median': np.median(fitness_values)
                })
            
            elif chart_type == 'density':
                # 创建密度图数据
                density = self._calculate_kde_density(fitness_values)
                chart_data.update({
                    'x_values': density['x'].tolist(),
                    'density_values': density['y'].tolist(),
                    'peak': density['peak'],
                    'area_under_curve': density['auc']
                })
            
            # 添加统计信息
            chart_data['statistics'] = {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'skewness': self._calculate_skewness(fitness_values),
                'kurtosis': self._calculate_kurtosis(fitness_values)
            }
            
            return {
                'success': True,
                'chart_data': chart_data,
                'config': self._get_distribution_chart_config(chart_type)
            }
            
        except Exception as e:
            print(f"创建适应度分布图表失败：{str(e)}")
            return {
                'success': False,
                'message': f'创建适应度分布图表失败：{str(e)}'
            }
    
    def create_heatmap_data(self, 
                           generations: List[int],
                           individuals: Dict[int, Individual]) -> Dict[str, Any]:
        """
        创建适应度热力图数据
        
        Args:
            generations: 世代列表
            individuals: 个体数据字典
            
        Returns:
            热力图数据
        """
        try:
            # 准备热力图数据矩阵
            heatmap_data = []
            generation_labels = []
            
            for generation in generations:
                generation_individuals = [
                    ind for ind in individuals.values() 
                    if ind.generation == generation
                ]
                
                if generation_individuals:
                    # 按适应度排序
                    sorted_individuals = sorted(generation_individuals, key=lambda x: x.fitness, reverse=True)
                    
                    # 创建适应度值序列
                    fitness_sequence = [ind.fitness for ind in sorted_individuals]
                    
                    # 标准化处理（将适应度值映射到0-1范围）
                    if fitness_sequence:
                        min_fitness = min(fitness_sequence)
                        max_fitness = max(fitness_sequence)
                        if max_fitness > min_fitness:
                            normalized_fitness = [(f - min_fitness) / (max_fitness - min_fitness) 
                                                for f in fitness_sequence]
                        else:
                            normalized_fitness = [1.0] * len(fitness_sequence)
                    else:
                        normalized_fitness = []
                    
                    heatmap_data.append(normalized_fitness)
                    generation_labels.append(generation)
            
            if not heatmap_data:
                return {
                    'success': False,
                    'message': '没有可用的数据创建热力图'
                }
            
            # 计算热力图统计信息
            all_values = [val for row in heatmap_data for val in row]
            stats = {
                'min_value': min(all_values),
                'max_value': max(all_values),
                'mean_value': np.mean(all_values),
                'total_cells': len(all_values)
            }
            
            return {
                'success': True,
                'heatmap_data': {
                    'data': heatmap_data,
                    'generation_labels': generation_labels,
                    'individual_rank_labels': list(range(1, max(len(row) for row in heatmap_data) + 1))
                },
                'statistics': stats,
                'config': {
                    'color_scheme': 'viridis',
                    'show_color_bar': True,
                    'x_label': '个体排名',
                    'y_label': '世代',
                    'title': '适应度热力图'
                }
            }
            
        except Exception as e:
            print(f"创建适应度热力图失败：{str(e)}")
            return {
                'success': False,
                'message': f'创建适应度热力图失败：{str(e)}'
            }
    
    def compare_fitness_metrics(self, 
                              generations: List[int],
                              metric_types: List[str] = ['best', 'average', 'worst']) -> Dict[str, Any]:
        """
        比较不同世代的适应度指标
        
        Args:
            generations: 要比较的世代列表
            metric_types: 要比较的指标类型列表
            
        Returns:
            比较结果
        """
        try:
            comparison_data = {
                'generations': generations,
                'metrics': {},
                'comparisons': {}
            }
            
            # 为每个指标类型准备数据
            for metric_type in metric_types:
                metric_data = []
                
                for generation in generations:
                    if generation in self.metrics_history:
                        metrics = self.metrics_history[generation]
                        
                        # 根据指标类型获取值
                        if metric_type == 'best':
                            value = metrics.best_fitness
                        elif metric_type == 'average':
                            value = metrics.average_fitness
                        elif metric_type == 'worst':
                            value = metrics.worst_fitness
                        elif metric_type == 'std':
                            value = metrics.std_fitness
                        elif metric_type == 'median':
                            value = metrics.median_fitness
                        elif metric_type == 'cv':  # coefficient of variation
                            value = metrics.coefficient_of_variation
                        else:
                            value = metrics.average_fitness
                        
                        metric_data.append(value)
                    else:
                        metric_data.append(None)
                
                comparison_data['metrics'][metric_type] = metric_data
            
            # 计算比较统计
            for metric_type in metric_types:
                data = comparison_data['metrics'][metric_type]
                valid_data = [x for x in data if x is not None]
                
                if valid_data:
                    comparison_data['comparisons'][metric_type] = {
                        'improvement_rate': self._calculate_improvement_rate(valid_data),
                        'volatility': np.std(valid_data),
                        'trend': self._determine_trend(valid_data),
                        'peak_generation': generations[data.index(max(valid_data))] if data else None,
                        'current_value': valid_data[-1] if valid_data else None,
                        'best_value': max(valid_data) if valid_data else None
                    }
            
            return {
                'success': True,
                'comparison_data': comparison_data,
                'analysis': self._analyze_metric_comparisons(comparison_data)
            }
            
        except Exception as e:
            print(f"比较适应度指标失败：{str(e)}")
            return {
                'success': False,
                'message': f'比较适应度指标失败：{str(e)}'
            }
    
    def predict_fitness_trend(self, 
                            target_generations: int = 5) -> Dict[str, Any]:
        """
        预测适应度趋势
        
        Args:
            target_generations: 预测的世代数量
            
        Returns:
            预测结果
        """
        try:
            if len(self.fitness_history) < 3:
                return {
                    'success': False,
                    'message': '数据不足，无法进行预测'
                }
            
            # 获取历史数据
            generations = [item[0] for item in self.fitness_history]
            best_fitness = [item[2] for item in self.fitness_history]  # 最佳适应度
            avg_fitness = [item[1] for item in self.fitness_history]  # 平均适应度
            
            # 预测最佳适应度
            best_predictions = self._linear_prediction(generations, best_fitness, target_generations)
            
            # 预测平均适应度
            avg_predictions = self._linear_prediction(generations, avg_fitness, target_generations)
            
            # 计算预测置信区间
            confidence_intervals = self._calculate_prediction_intervals(
                best_fitness, best_predictions
            )
            
            # 生成预测报告
            prediction_report = {
                'target_generations': target_generations,
                'current_generation': generations[-1],
                'predicted_generations': [generations[-1] + i + 1 for i in range(target_generations)],
                'best_fitness_predictions': best_predictions,
                'average_fitness_predictions': avg_predictions,
                'confidence_intervals': confidence_intervals,
                'model_info': {
                    'method': 'linear_regression',
                    'training_data_points': len(generations),
                    'last_updated': time.time()
                }
            }
            
            return {
                'success': True,
                'predictions': prediction_report,
                'confidence_score': self._calculate_prediction_confidence(generations, best_fitness)
            }
            
        except Exception as e:
            print(f"预测适应度趋势失败：{str(e)}")
            return {
                'success': False,
                'message': f'预测适应度趋势失败：{str(e)}'
            }
    
    def get_chart_config(self, chart_type: str = 'trend') -> Dict[str, Any]:
        """
        获取图表配置
        
        Args:
            chart_type: 图表类型
            
        Returns:
            图表配置字典
        """
        return self._get_chart_config(chart_type)
    
    def clear(self):
        """清除历史数据"""
        self.fitness_history.clear()
        self.metrics_history.clear()
        self.best_individuals.clear()
        self.generation_statistics.clear()
        print("适应度可视化器数据已清除")
    
    def _analyze_fitness_trend(self, trend_data: Dict[str, Any]) -> TrendAnalysis:
        """分析适应度趋势"""
        generations = trend_data['generations']
        values = trend_data['values']
        
        if len(values) < 2:
            return TrendAnalysis(
                generation_range=(0, 0),
                slope=0,
                intercept=0,
                r_squared=0,
                p_value=1.0,
                trend_direction='insufficient_data',
                confidence_level=0,
                prediction_intervals={}
            )
        
        # 线性回归
        x = np.array(generations)
        y = np.array(values)
        
        # 计算回归系数
        slope, intercept = np.polyfit(x, y, 1)
        
        # 计算R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算p值（简化版本）
        n = len(values)
        if n > 2:
            mse = ss_res / (n - 2)
            se_slope = math.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
            t_stat = slope / se_slope
            # 简化的p值估计
            p_value = 2 * (1 - 0.95) if abs(t_stat) > 2 else 0.1
        else:
            p_value = 1.0
        
        # 确定趋势方向
        if slope > 0.001:
            trend_direction = 'increasing'
        elif slope < -0.001:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # 计算置信区间
        prediction_intervals = self._calculate_prediction_intervals(values, generations)
        
        return TrendAnalysis(
            generation_range=(min(generations), max(generations)),
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            trend_direction=trend_direction,
            confidence_level=self.confidence_level,
            prediction_intervals=prediction_intervals
        )
    
    def _calculate_optimal_bins(self, data: np.ndarray) -> int:
        """计算最优的分组数量"""
        n = len(data)
        if n <= 1:
            return 1
        
        # 使用Freedman-Diaconis规则
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            return int(math.sqrt(n))
        
        bin_width = 2 * iqr * (n ** (-1/3))
        bin_count = int((np.max(data) - np.min(data)) / bin_width)
        
        return max(1, min(bin_count, 50))  # 限制在1-50之间
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        kurtosis = np.sum(((data - mean) / std) ** 4) / n
        return kurtosis - 3  # 超额峰度
    
    def _calculate_density_points(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """计算密度点（用于小提琴图）"""
        # 简化的密度计算
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        # 创建密度点
        density_y = np.linspace(0, 1, n)
        density_x = sorted_data
        
        return {
            'x': density_x,
            'y': density_y
        }
    
    def _calculate_kde_density(self, data: np.ndarray) -> Dict[str, Any]:
        """计算核密度估计"""
        # 简化的核密度估计
        x_min, x_max = np.min(data), np.max(data)
        x_range = x_max - x_min
        
        if x_range == 0:
            return {
                'x': np.array([x_min]),
                'y': np.array([1.0]),
                'peak': x_min,
                'auc': 1.0
            }
        
        # 创建密度估计点
        x = np.linspace(x_min, x_max, 100)
        
        # 高斯核密度估计（简化版本）
        bandwidth = 1.06 * np.std(data) * (len(data) ** (-1/5))
        density = np.zeros_like(x)
        
        for point in data:
            kernel = np.exp(-0.5 * ((x - point) / bandwidth) ** 2)
            density += kernel / (bandwidth * math.sqrt(2 * math.pi))
        
        density /= len(data)
        
        # 找到峰值
        peak_index = np.argmax(density)
        peak_value = x[peak_index]
        
        # 计算曲线下面积
        auc = np.trapz(density, x)
        
        return {
            'x': x,
            'y': density,
            'peak': peak_value,
            'auc': auc
        }
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """计算改进率"""
        if len(values) < 2:
            return 0.0
        
        initial_value = values[0]
        final_value = values[-1]
        
        if initial_value == 0:
            return float('inf') if final_value > 0 else 0.0
        
        return (final_value - initial_value) / abs(initial_value)
    
    def _determine_trend(self, values: List[float]) -> str:
        """确定趋势方向"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # 计算线性趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def _linear_prediction(self, 
                          generations: List[int], 
                          values: List[float], 
                          steps: int) -> List[float]:
        """线性预测"""
        if len(generations) < 2 or len(values) < 2:
            return []
        
        # 拟合线性模型
        x = np.array(generations)
        y = np.array(values)
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # 预测未来值
        last_generation = generations[-1]
        predictions = []
        
        for i in range(1, steps + 1):
            pred_generation = last_generation + i
            pred_value = slope * pred_generation + intercept
            predictions.append(pred_value)
        
        return predictions
    
    def _calculate_prediction_intervals(self, 
                                      historical_values: List[float],
                                      predictions: List[float],
                                      confidence: float = 0.95) -> Dict[str, List[Tuple[float, float]]]:
        """计算预测置信区间"""
        if len(historical_values) < 3:
            return {'95%': [(0, 0)] * len(predictions)}
        
        # 计算残差标准误差
        residuals = np.array(historical_values[1:]) - np.array(historical_values[:-1])
        std_error = np.std(residuals)
        
        # 计算置信区间
        from scipy import stats
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha/2, len(residuals) - 1)
        
        intervals = []
        for pred in predictions:
            margin = t_value * std_error
            intervals.append((pred - margin, pred + margin))
        
        return {'95%': intervals}
    
    def _calculate_prediction_confidence(self, 
                                       generations: List[int], 
                                       values: List[float]) -> float:
        """计算预测置信度"""
        if len(generations) < 3:
            return 0.0
        
        # 基于R²计算置信度
        x = np.array(generations)
        y = np.array(values)
        
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return max(0, min(1, r_squared))
    
    def _analyze_metric_comparisons(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析指标比较结果"""
        analysis = {
            'overall_performance': 'unknown',
            'most_improved_metric': None,
            'least_improved_metric': None,
            'recommendations': []
        }
        
        # 分析改进率
        improvement_rates = {}
        for metric, comparisons in comparison_data['comparisons'].items():
            improvement_rates[metric] = comparisons.get('improvement_rate', 0)
        
        if improvement_rates:
            # 找出改进最大和最小的指标
            best_metric = max(improvement_rates, key=improvement_rates.get)
            worst_metric = min(improvement_rates, key=improvement_rates.get)
            
            analysis['most_improved_metric'] = best_metric
            analysis['least_improved_metric'] = worst_metric
            
            # 生成建议
            avg_improvement = np.mean(list(improvement_rates.values()))
            if avg_improvement > 0.1:
                analysis['overall_performance'] = 'excellent'
                analysis['recommendations'].append('进化效果很好，建议保持当前参数')
            elif avg_improvement > 0.05:
                analysis['overall_performance'] = 'good'
                analysis['recommendations'].append('进化效果良好，可适当调整参数以进一步提升')
            else:
                analysis['overall_performance'] = 'poor'
                analysis['recommendations'].append('进化效果不佳，建议调整选择压力或变异率')
        
        return analysis
    
    def _get_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """获取图表配置"""
        base_config = {
            'width': self.chart_width,
            'height': self.chart_height,
            'show_grid': True,
            'show_legend': True,
            'responsive': True
        }
        
        if chart_type == 'trend':
            base_config.update({
                'x_label': '世代',
                'y_label': '适应度值',
                'title': '适应度趋势图',
                'line_style': 'solid',
                'marker_size': 6
            })
        elif chart_type == 'distribution':
            base_config.update({
                'x_label': '适应度值',
                'y_label': '频率',
                'title': '适应度分布图',
                'show_stats': True
            })
        
        return base_config
    
    def _get_distribution_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """获取分布图配置"""
        configs = {
            'histogram': {
                'show_bins': True,
                'show_density': False,
                'color': '#4CAF50'
            },
            'box': {
                'show_outliers': True,
                'show_whiskers': True,
                'color': '#2196F3'
            },
            'violin': {
                'show_kernel_density': True,
                'bandwidth': 'auto',
                'color': '#FF9800'
            },
            'density': {
                'show_auc': True,
                'show_peak': True,
                'color': '#E91E63'
            }
        }
        
        return configs.get(chart_type, {})


# 导入必要的库
from scipy import stats