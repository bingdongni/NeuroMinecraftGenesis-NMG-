#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多样性分析器
Diversity Analyzer

该模块负责计算和分析进化过程中的遗传多样性和表型多样性，
包括Shannon指数、Simpson指数、遗传距离、聚类分析等指标。

功能特性：
- 遗传多样性指标计算（Shannon、Simpson、Richness等）
- 表型多样性分析
- 遗传距离和相似度计算
- 群体结构分析
- 进化动态追踪

Author: 进化树可视化系统
Date: 2025-11-13
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
import math

try:
    from .evolution_tree import Individual
except ImportError:
    from utils.visualization.evolution_tree import Individual


@dataclass
class DiversityMetrics:
    """
    多样性指标
    Diversity Metrics
    """
    generation: int
    individual_count: int
    
    # Shannon多样性指数
    shannon_index: float
    shannon_evenness: float
    
    # Simpson多样性指数
    simpson_index: float
    simpson_evenness: float
    
    # 丰富度指标
    richness: int
    chao1_estimator: float
    
    # 遗传距离指标
    average_genetic_distance: float
    minimum_genetic_distance: float
    maximum_genetic_distance: float
    
    # 表型多样性
    phenotype_diversity: float
    fitness_variance: float
    
    # 群体结构
    effective_population_size: float
    inbreeding_coefficient: float
    
    # 聚类信息
    cluster_count: int
    cluster_sizes: List[int]
    cluster_diversity: List[float]
    
    # 时间戳
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'generation': self.generation,
            'individual_count': self.individual_count,
            'shannon_index': self.shannon_index,
            'shannon_evenness': self.shannon_evenness,
            'simpson_index': self.simpson_index,
            'simpson_evenness': self.simpson_evenness,
            'richness': self.richness,
            'chao1_estimator': self.chao1_estimator,
            'average_genetic_distance': self.average_genetic_distance,
            'minimum_genetic_distance': self.minimum_genetic_distance,
            'maximum_genetic_distance': self.maximum_genetic_distance,
            'phenotype_diversity': self.phenotype_diversity,
            'fitness_variance': self.fitness_variance,
            'effective_population_size': self.effective_population_size,
            'inbreeding_coefficient': self.inbreeding_coefficient,
            'cluster_count': self.cluster_count,
            'cluster_sizes': self.cluster_sizes,
            'cluster_diversity': self.cluster_diversity,
            'timestamp': self.timestamp
        }


@dataclass
class GeneticDistanceMatrix:
    """
    遗传距离矩阵
    Genetic Distance Matrix
    """
    individual_ids: List[int]
    distance_matrix: np.ndarray
    distance_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'individual_ids': self.individual_ids,
            'distance_matrix': self.distance_matrix.tolist(),
            'distance_type': self.distance_type
        }


class DiversityAnalyzer:
    """
    多样性分析器
    Diversity Analyzer
    
    负责计算和分析种群的多样性指标，包括遗传多样性、
    表型多样性和群体结构分析。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多样性分析器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._default_config()
        
        # 分析配置
        self.analysis_config = self.config.get('analysis', {})
        self.metrics_to_calculate = self.analysis_config.get('metrics', [
            'shannon', 'simpson', 'richness', 'genetic_distance'
        ])
        self.clustering_enabled = self.analysis_config.get('enable_clustering', True)
        self.distance_metric = self.analysis_config.get('distance_metric', 'euclidean')
        
        # 聚类配置
        self.clustering_config = self.config.get('clustering', {})
        self.max_clusters = self.clustering_config.get('max_clusters', 10)
        self.clustering_method = self.clustering_config.get('method', 'hierarchical')
        
        # 数据存储
        self.diversity_history = {}  # 多样性历史数据
        self.genetic_distance_matrices = {}  # 遗传距离矩阵缓存
        self.clustering_results = {}  # 聚类结果缓存
        
        # 统计缓存
        self.population_statistics = {}
        self.evolutionary_trends = {}
        
        print("多样性分析器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'analysis': {
                'metrics': ['shannon', 'simpson', 'richness', 'genetic_distance'],
                'enable_clustering': True,
                'distance_metric': 'euclidean',
                'window_size': 10
            },
            'clustering': {
                'method': 'hierarchical',
                'max_clusters': 10,
                'min_cluster_size': 3,
                'linkage': 'ward'
            },
            'diversity_thresholds': {
                'low_diversity': 0.3,
                'high_diversity': 0.8,
                'critical_diversity': 0.1
            }
        }
    
    def analyze_diversity(self, 
                         individuals: List[Individual],
                         generation: int,
                         metric_types: List[str] = None) -> DiversityMetrics:
        """
        分析种群多样性
        
        Args:
            individuals: 个体列表
            generation: 世代编号
            metric_types: 要计算的指标类型列表
            
        Returns:
            多样性指标
        """
        try:
            logger.info(f"开始分析世代 {generation} 的多样性，包含 {len(individuals)} 个个体")
            
            # 确定要计算的指标
            if metric_types is None:
                metric_types = self.metrics_to_calculate
            
            if not individuals:
                # 返回默认指标
                return self._create_default_metrics(generation)
            
            # 计算基础统计
            fitness_values = np.array([ind.fitness for ind in individuals])
            genomes = np.array([ind.genome for ind in individuals])
            
            # 计算多样性指标
            metrics = DiversityMetrics(
                generation=generation,
                individual_count=len(individuals),
                shannon_index=0.0,
                shannon_evenness=0.0,
                simpson_index=0.0,
                simpson_evenness=0.0,
                richness=0,
                chao1_estimator=0.0,
                average_genetic_distance=0.0,
                minimum_genetic_distance=0.0,
                maximum_genetic_distance=0.0,
                phenotype_diversity=0.0,
                fitness_variance=0.0,
                effective_population_size=0.0,
                inbreeding_coefficient=0.0,
                cluster_count=0,
                cluster_sizes=[],
                cluster_diversity=[],
                timestamp=time.time()
            )
            
            # 计算Shannon多样性指数
            if 'shannon' in metric_types:
                shannon_result = self._calculate_shannon_diversity(genomes)
                metrics.shannon_index = shannon_result['index']
                metrics.shannon_evenness = shannon_result['evenness']
            
            # 计算Simpson多样性指数
            if 'simpson' in metric_types:
                simpson_result = self._calculate_simpson_diversity(genomes)
                metrics.simpson_index = simpson_result['index']
                metrics.simpson_evenness = simpson_result['evenness']
            
            # 计算丰富度
            if 'richness' in metric_types:
                richness_result = self._calculate_richness(genomes)
                metrics.richness = richness_result['richness']
                metrics.chao1_estimator = richness_result['chao1']
            
            # 计算遗传距离
            if 'genetic_distance' in metric_types:
                distance_result = self._calculate_genetic_distances(genomes)
                metrics.average_genetic_distance = distance_result['average']
                metrics.minimum_genetic_distance = distance_result['minimum']
                metrics.maximum_genetic_distance = distance_result['maximum']
            
            # 计算表型多样性
            if 'phenotype' in metric_types:
                phenotype_result = self._calculate_phenotype_diversity(fitness_values)
                metrics.phenotype_diversity = phenotype_result['diversity']
                metrics.fitness_variance = phenotype_result['variance']
            
            # 群体结构分析
            if self.clustering_enabled and len(individuals) >= 3:
                clustering_result = self._analyze_population_structure(genomes, individuals)
                metrics.cluster_count = clustering_result['cluster_count']
                metrics.cluster_sizes = clustering_result['cluster_sizes']
                metrics.cluster_diversity = clustering_result['cluster_diversity']
                
                # 有效种群大小和近交系数
                metrics.effective_population_size = self._estimate_effective_population_size(
                    metrics.cluster_count, len(individuals)
                )
                metrics.inbreeding_coefficient = self._calculate_inbreeding_coefficient(
                    distance_result['average']
                )
            
            # 更新历史记录
            self.diversity_history[generation] = metrics
            
            logger.info(f"世代 {generation} 多样性分析完成")
            return metrics
            
        except Exception as e:
            logger.error(f"分析世代 {generation} 多样性失败：{str(e)}")
            raise
    
    def get_diversity_trend(self, generations: List[int] = None) -> Dict[str, Any]:
        """
        获取多样性趋势数据
        
        Args:
            generations: 要分析的世代列表
            
        Returns:
            趋势数据
        """
        try:
            # 确定要分析的世代
            if generations is None:
                generations = sorted(self.diversity_history.keys())
            
            if not generations:
                return {
                    'success': False,
                    'message': '没有多样性历史数据'
                }
            
            # 准备趋势数据
            trend_data = {
                'generations': generations,
                'metrics': {
                    'shannon_index': [],
                    'simpson_index': [],
                    'richness': [],
                    'average_genetic_distance': [],
                    'phenotype_diversity': [],
                    'effective_population_size': []
                }
            }
            
            # 收集数据
            for generation in generations:
                if generation in self.diversity_history:
                    metrics = self.diversity_history[generation]
                    trend_data['metrics']['shannon_index'].append(metrics.shannon_index)
                    trend_data['metrics']['simpson_index'].append(metrics.simpson_index)
                    trend_data['metrics']['richness'].append(metrics.richness)
                    trend_data['metrics']['average_genetic_distance'].append(metrics.average_genetic_distance)
                    trend_data['metrics']['phenotype_diversity'].append(metrics.phenotype_diversity)
                    trend_data['metrics']['effective_population_size'].append(metrics.effective_population_size)
            
            # 分析趋势
            trend_analysis = self._analyze_diversity_trends(trend_data)
            
            return {
                'success': True,
                'trend_data': trend_data,
                'analysis': trend_analysis,
                'summary': self._generate_diversity_summary(generations)
            }
            
        except Exception as e:
            logger.error(f"获取多样性趋势失败：{str(e)}")
            return {
                'success': False,
                'message': f'获取多样性趋势失败：{str(e)}'
            }
    
    def compare_populations(self, 
                          population1: List[Individual],
                          population2: List[Individual],
                          generation1: int,
                          generation2: int) -> Dict[str, Any]:
        """
        比较两个种群的多样性
        
        Args:
            population1: 第一个种群
            population2: 第二个种群
            generation1: 第一个种群的世代
            generation2: 第二个种群的世代
            
        Returns:
            比较结果
        """
        try:
            # 分析两个种群
            metrics1 = self.analyze_diversity(population1, generation1)
            metrics2 = self.analyze_diversity(population2, generation2)
            
            # 计算差异
            comparison_results = {
                'generation_comparison': f"{generation1} vs {generation2}",
                'population_sizes': {
                    'population1': len(population1),
                    'population2': len(population2)
                },
                'diversity_differences': {
                    'shannon_difference': metrics2.shannon_index - metrics1.shannon_index,
                    'simpson_difference': metrics2.simpson_index - metrics1.simpson_index,
                    'richness_difference': metrics2.richness - metrics1.richness,
                    'genetic_distance_difference': metrics2.average_genetic_distance - metrics1.average_genetic_distance
                },
                'relative_changes': {
                    'shannon_relative_change': self._calculate_relative_change(
                        metrics1.shannon_index, metrics2.shannon_index
                    ),
                    'simpson_relative_change': self._calculate_relative_change(
                        metrics1.simpson_index, metrics2.simpson_index
                    ),
                    'richness_relative_change': self._calculate_relative_change(
                        metrics1.richness, metrics2.richness
                    ) if metrics1.richness > 0 else 0
                }
            }
            
            # 计算遗传相似度
            genomes1 = np.array([ind.genome for ind in population1])
            genomes2 = np.array([ind.genome for ind in population2])
            genetic_similarity = self._calculate_inter_population_similarity(genomes1, genomes2)
            comparison_results['genetic_similarity'] = genetic_similarity
            
            # 生成比较结论
            conclusions = self._generate_comparison_conclusions(comparison_results)
            comparison_results['conclusions'] = conclusions
            
            return {
                'success': True,
                'comparison_data': comparison_results,
                'population1_metrics': metrics1.to_dict(),
                'population2_metrics': metrics2.to_dict()
            }
            
        except Exception as e:
            logger.error(f"比较种群多样性失败：{str(e)}")
            return {
                'success': False,
                'message': f'比较种群多样性失败：{str(e)}'
            }
    
    def detect_diversity_changes(self, 
                               window_size: int = 5,
                               significance_threshold: float = 0.05) -> Dict[str, Any]:
        """
        检测多样性变化点
        
        Args:
            window_size: 检测窗口大小
            significance_threshold: 显著性阈值
            
        Returns:
            变化点检测结果
        """
        try:
            if len(self.diversity_history) < window_size * 2:
                return {
                    'success': False,
                    'message': '数据不足，无法进行变化点检测'
                }
            
            generations = sorted(self.diversity_history.keys())
            change_points = []
            
            # 滑动窗口检测
            for i in range(window_size, len(generations) - window_size):
                window_start = i - window_size
                window_end = i + window_size
                
                # 获取窗口数据
                before_data = [
                    self.diversity_history[gen].shannon_index 
                    for gen in generations[window_start:i]
                ]
                after_data = [
                    self.diversity_history[gen].shannon_index 
                    for gen in generations[i:window_end]
                ]
                
                # 统计检验（简化版t检验）
                t_stat, p_value = self._simple_t_test(before_data, after_data)
                
                if p_value < significance_threshold:
                    change_points.append({
                        'generation': generations[i],
                        'change_type': 'significant_shift',
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'before_mean': np.mean(before_data),
                        'after_mean': np.mean(after_data),
                        'change_magnitude': np.mean(after_data) - np.mean(before_data)
                    })
            
            # 计算变化趋势
            trend_analysis = self._analyze_diversity_changes(change_points)
            
            return {
                'success': True,
                'change_points': change_points,
                'trend_analysis': trend_analysis,
                'detection_summary': {
                    'total_generations': len(generations),
                    'change_points_found': len(change_points),
                    'most_significant_change': max(change_points, key=lambda x: abs(x['change_magnitude'])) if change_points else None
                }
            }
            
        except Exception as e:
            logger.error(f"检测多样性变化失败：{str(e)}")
            return {
                'success': False,
                'message': f'检测多样性变化失败：{str(e)}'
            }
    
    def create_diversity_heatmap(self, generations: List[int] = None) -> Dict[str, Any]:
        """
        创建多样性热力图
        
        Args:
            generations: 要包含的世代列表
            
        Returns:
            热力图数据
        """
        try:
            # 确定世代范围
            if generations is None:
                generations = sorted(self.diversity_history.keys())
            
            if not generations:
                return {
                    'success': False,
                    'message': '没有多样性数据用于创建热力图'
                }
            
            # 准备热力图数据
            heatmap_data = []
            metric_labels = [
                'shannon_index',
                'simpson_index', 
                'richness',
                'average_genetic_distance',
                'phenotype_diversity'
            ]
            
            for generation in generations:
                if generation in self.diversity_history:
                    metrics = self.diversity_history[generation]
                    row_data = [
                        metrics.shannon_index,
                        metrics.simpson_index,
                        metrics.richness / 100.0,  # 标准化处理
                        metrics.average_genetic_distance,
                        metrics.phenotype_diversity
                    ]
                    heatmap_data.append(row_data)
            
            if not heatmap_data:
                return {
                    'success': False,
                    'message': '没有可用的多样性数据'
                }
            
            # 标准化数据（按列）
            heatmap_array = np.array(heatmap_data)
            normalized_data = (heatmap_array - np.min(heatmap_array, axis=0)) / (
                np.max(heatmap_array, axis=0) - np.min(heatmap_array, axis=0) + 1e-10
            )
            
            # 计算统计信息
            stats = {
                'min_values': np.min(heatmap_array, axis=0).tolist(),
                'max_values': np.max(heatmap_array, axis=0).tolist(),
                'mean_values': np.mean(heatmap_array, axis=0).tolist(),
                'std_values': np.std(heatmap_array, axis=0).tolist()
            }
            
            return {
                'success': True,
                'heatmap_data': {
                    'data': normalized_data.tolist(),
                    'generation_labels': generations,
                    'metric_labels': metric_labels
                },
                'original_data': {
                    'data': heatmap_data,
                    'generations': generations,
                    'metrics': metric_labels
                },
                'statistics': stats,
                'config': {
                    'color_scheme': 'viridis',
                    'show_color_bar': True,
                    'x_label': '多样性指标',
                    'y_label': '世代',
                    'title': '多样性指标热力图'
                }
            }
            
        except Exception as e:
            logger.error(f"创建多样性热力图失败：{str(e)}")
            return {
                'success': False,
                'message': f'创建多样性热力图失败：{str(e)}'
            }
    
    def generate_diversity_report(self, generations: List[int] = None) -> Dict[str, Any]:
        """
        生成多样性分析报告
        
        Args:
            generations: 要包含的世代列表
            
        Returns:
            报告数据
        """
        try:
            # 确定分析范围
            if generations is None:
                generations = sorted(self.diversity_history.keys())
            
            if not generations:
                return {
                    'success': False,
                    'message': '没有数据生成报告'
                }
            
            # 计算总体统计
            total_generations = len(generations)
            generation_range = (min(generations), max(generations))
            
            # 收集所有多样性指标
            all_shannon = [self.diversity_history[g].shannon_index for g in generations]
            all_simpson = [self.diversity_history[g].simpson_index for g in generations]
            all_richness = [self.diversity_history[g].richness for g in generations]
            all_genetic_dist = [self.diversity_history[g].average_genetic_distance for g in generations]
            
            # 生成统计摘要
            summary_stats = {
                'shannon_index': {
                    'mean': np.mean(all_shannon),
                    'std': np.std(all_shannon),
                    'min': np.min(all_shannon),
                    'max': np.max(all_shannon),
                    'trend': self._determine_trend(all_shannon)
                },
                'simpson_index': {
                    'mean': np.mean(all_simpson),
                    'std': np.std(all_simpson),
                    'min': np.min(all_simpson),
                    'max': np.max(all_simpson),
                    'trend': self._determine_trend(all_simpson)
                },
                'richness': {
                    'mean': np.mean(all_richness),
                    'std': np.std(all_richness),
                    'min': np.min(all_richness),
                    'max': np.max(all_richness),
                    'trend': self._determine_trend(all_richness)
                },
                'genetic_distance': {
                    'mean': np.mean(all_genetic_dist),
                    'std': np.std(all_genetic_dist),
                    'min': np.min(all_genetic_dist),
                    'max': np.max(all_genetic_dist),
                    'trend': self._determine_trend(all_genetic_dist)
                }
            }
            
            # 多样性健康评估
            health_assessment = self._assess_diversity_health(generations)
            
            # 生成建议
            recommendations = self._generate_diversity_recommendations(summary_stats, health_assessment)
            
            # 生成报告
            report = {
                'metadata': {
                    'report_generated': time.time(),
                    'analysis_period': generation_range,
                    'total_generations': total_generations,
                    'generations_analyzed': generations
                },
                'summary_statistics': summary_stats,
                'health_assessment': health_assessment,
                'recommendations': recommendations,
                'trends': self._analyze_diversity_trends({
                    'generations': generations,
                    'metrics': {
                        'shannon_index': all_shannon,
                        'simpson_index': all_simpson,
                        'richness': all_richness,
                        'average_genetic_distance': all_genetic_dist
                    }
                })
            }
            
            return {
                'success': True,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"生成多样性报告失败：{str(e)}")
            return {
                'success': False,
                'message': f'生成多样性报告失败：{str(e)}'
            }
    
    def clear(self):
        """清除所有数据"""
        self.diversity_history.clear()
        self.genetic_distance_matrices.clear()
        self.clustering_results.clear()
        self.population_statistics.clear()
        self.evolutionary_trends.clear()
        print("多样性分析器数据已清除")
    
    def _calculate_shannon_diversity(self, genomes: np.ndarray) -> Dict[str, float]:
        """计算Shannon多样性指数"""
        if len(genomes) == 0:
            return {'index': 0.0, 'evenness': 0.0}
        
        # 将基因组数字化（量化为bin）
        n_bins = min(10, len(genomes[0]) * 2)  # 动态确定bin数量
        digitized_genomes = []
        
        for genome in genomes:
            digitized = []
            for gene in genome:
                bin_index = min(int(abs(gene) * n_bins), n_bins - 1)
                digitized.append(bin_index)
            digitized_genomes.append(digitized)
        
        # 计算每个位点的多样性
        locus_diversities = []
        for locus_idx in range(len(digitized_genomes[0])):
            locus_values = [genome[locus_idx] for genome in digitized_genomes]
            value_counts = Counter(locus_values)
            total = len(locus_values)
            
            # Shannon指数计算
            shannon_locus = 0.0
            for count in value_counts.values():
                if count > 0:
                    p = count / total
                    shannon_locus -= p * math.log(p)
            
            locus_diversities.append(shannon_locus)
        
        # 整体Shannon指数
        overall_shannon = np.mean(locus_diversities)
        
        # Shannon均匀度
        max_possible_shannon = math.log(len(set().union(*[set(g) for g in digitized_genomes])))
        evenness = overall_shannon / max_possible_shannon if max_possible_shannon > 0 else 0
        
        return {
            'index': overall_shannon,
            'evenness': evenness
        }
    
    def _calculate_simpson_diversity(self, genomes: np.ndarray) -> Dict[str, float]:
        """计算Simpson多样性指数"""
        if len(genomes) == 0:
            return {'index': 0.0, 'evenness': 0.0}
        
        # 将基因组数字化
        n_bins = min(10, len(genomes[0]) * 2)
        digitized_genomes = []
        
        for genome in genomes:
            digitized = []
            for gene in genome:
                bin_index = min(int(abs(gene) * n_bins), n_bins - 1)
                digitized.append(bin_index)
            digitized_genomes.append(digitized)
        
        # 计算每个位点的Simpson指数
        locus_simpson = []
        for locus_idx in range(len(digitized_genomes[0])):
            locus_values = [genome[locus_idx] for genome in digitized_genomes]
            value_counts = Counter(locus_values)
            total = len(locus_values)
            
            # Simpson指数计算
            simpson_locus = 0.0
            for count in value_counts.values():
                if count > 0:
                    p = count / total
                    simpson_locus += p * p
            
            locus_simpson.append(simpson_locus)
        
        # 整体Simpson指数
        overall_simpson = 1 - np.mean(locus_simpson)  # 转换为多样性指数
        
        # Simpson均匀度
        max_possible_simpson = 1 - (1 / len(set().union(*[set(g) for g in digitized_genomes])))
        evenness = overall_simpson / max_possible_simpson if max_possible_simpson > 0 else 0
        
        return {
            'index': overall_simpson,
            'evenness': evenness
        }
    
    def _calculate_richness(self, genomes: np.ndarray) -> Dict[str, float]:
        """计算丰富度"""
        if len(genomes) == 0:
            return {'richness': 0, 'chao1': 0.0}
        
        # 计算唯一基因型数量
        unique_genotypes = set()
        for genome in genomes:
            genotype = tuple(round(g, 3) for g in genome)  # 四舍五入以避免浮点精度问题
            unique_genotypes.add(genotype)
        
        richness = len(unique_genotypes)
        
        # Chao1估计器（基于重复基因型）
        genotype_counts = Counter()
        for genome in genomes:
            genotype = tuple(round(g, 3) for g in genome)
            genotype_counts[genotype] += 1
        
        singleton_count = sum(1 for count in genotype_counts.values() if count == 1)
        double_count = sum(1 for count in genotype_counts.values() if count == 2)
        
        if double_count > 0:
            chao1 = richness + (singleton_count ** 2) / (2 * double_count)
        else:
            chao1 = richness
        
        return {
            'richness': richness,
            'chao1': chao1
        }
    
    def _calculate_genetic_distances(self, genomes: np.ndarray) -> Dict[str, float]:
        """计算遗传距离"""
        if len(genomes) <= 1:
            return {
                'average': 0.0,
                'minimum': 0.0,
                'maximum': 0.0
            }
        
        # 计算距离矩阵
        try:
            if self.distance_metric == 'euclidean':
                distances = pdist(genomes, metric='euclidean')
            elif self.distance_metric == 'manhattan':
                distances = pdist(genomes, metric='cityblock')
            elif self.distance_metric == 'hamming':
                # 将连续值转换为二进制进行比较
                binary_genomes = []
                for genome in genomes:
                    binary_genome = [1 if g > np.median(genomes[:, i]) else 0 
                                   for i, g in enumerate(genome)]
                    binary_genomes.append(binary_genome)
                distances = pdist(binary_genomes, metric='hamming')
            else:
                distances = pdist(genomes, metric='euclidean')
            
            return {
                'average': np.mean(distances),
                'minimum': np.min(distances),
                'maximum': np.max(distances)
            }
        except Exception as e:
            print(f"计算遗传距离失败：{str(e)}")
            return {
                'average': 0.0,
                'minimum': 0.0,
                'maximum': 0.0
            }
    
    def _calculate_phenotype_diversity(self, fitness_values: np.ndarray) -> Dict[str, float]:
        """计算表型多样性"""
        if len(fitness_values) <= 1:
            return {
                'diversity': 0.0,
                'variance': 0.0
            }
        
        variance = np.var(fitness_values)
        mean_fitness = np.mean(fitness_values)
        
        # 变异系数作为多样性指标
        if mean_fitness != 0:
            diversity = variance / (mean_fitness ** 2)
        else:
            diversity = 0.0
        
        return {
            'diversity': diversity,
            'variance': variance
        }
    
    def _analyze_population_structure(self, genomes: np.ndarray, individuals: List[Individual]) -> Dict[str, Any]:
        """分析种群结构"""
        try:
            if len(genomes) < 3:
                return {
                    'cluster_count': 1,
                    'cluster_sizes': [len(individuals)],
                    'cluster_diversity': [0.0]
                }
            
            # 聚类分析
            if self.clustering_method == 'hierarchical':
                return self._hierarchical_clustering(genomes)
            elif self.clustering_method == 'kmeans':
                return self._kmeans_clustering(genomes)
            else:
                return self._hierarchical_clustering(genomes)
                
        except Exception as e:
            print(f"分析种群结构失败：{str(e)}")
            return {
                'cluster_count': 1,
                'cluster_sizes': [len(individuals)],
                'cluster_diversity': [0.0]
            }
    
    def _hierarchical_clustering(self, genomes: np.ndarray) -> Dict[str, Any]:
        """层次聚类"""
        try:
            # 计算距离矩阵
            distances = pdist(genomes, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # 层次聚类
            linkage_matrix = linkage(distances, method='ward')
            
            # 确定最优聚类数（肘部法则）
            max_k = min(self.max_clusters, len(genomes) - 1)
            inertias = []
            
            for k in range(2, max_k + 1):
                clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                inertia = 0
                for cluster_id in range(1, k + 1):
                    cluster_points = genomes[clusters == cluster_id]
                    if len(cluster_points) > 1:
                        centroid = np.mean(cluster_points, axis=0)
                        inertia += np.sum((cluster_points - centroid) ** 2)
                inertias.append(inertia)
            
            # 选择最优k值
            if len(inertias) >= 3:
                # 计算二阶导数
                second_derivatives = []
                for i in range(1, len(inertias) - 1):
                    second_deriv = inertias[i+1] - 2*inertias[i] + inertias[i-1]
                    second_derivatives.append(second_deriv)
                
                optimal_k = second_derivatives.index(min(second_derivatives)) + 3
            else:
                optimal_k = 2
            
            optimal_k = min(optimal_k, max_k)
            
            # 使用最优k值进行聚类
            final_clusters = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
            
            # 计算聚类统计
            cluster_sizes = []
            cluster_diversities = []
            
            for cluster_id in range(1, optimal_k + 1):
                cluster_points = genomes[final_clusters == cluster_id]
                cluster_sizes.append(len(cluster_points))
                
                # 计算聚类内多样性
                if len(cluster_points) > 1:
                    cluster_distances = pdist(cluster_points, metric='euclidean')
                    diversity = np.mean(cluster_distances)
                else:
                    diversity = 0.0
                
                cluster_diversities.append(diversity)
            
            return {
                'cluster_count': optimal_k,
                'cluster_sizes': cluster_sizes,
                'cluster_diversity': cluster_diversities,
                'linkage_matrix': linkage_matrix.tolist(),
                'cluster_labels': final_clusters.tolist()
            }
            
        except Exception as e:
            print(f"层次聚类失败：{str(e)}")
            return {
                'cluster_count': 1,
                'cluster_sizes': [len(genomes)],
                'cluster_diversity': [0.0]
            }
    
    def _kmeans_clustering(self, genomes: np.ndarray) -> Dict[str, Any]:
        """K-means聚类"""
        try:
            max_k = min(self.max_clusters, len(genomes) // 2)
            
            if max_k < 2:
                return {
                    'cluster_count': 1,
                    'cluster_sizes': [len(genomes)],
                    'cluster_diversity': [0.0]
                }
            
            # 尝试不同的k值，选择最优的
            best_k = 2
            best_score = -1
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(genomes)
                
                # 计算轮廓系数（简化版）
                silhouette_score = self._calculate_silhouette_score(genomes, cluster_labels)
                
                if silhouette_score > best_score:
                    best_score = silhouette_score
                    best_k = k
            
            # 使用最优k值重新聚类
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(genomes)
            
            # 计算聚类统计
            cluster_sizes = []
            cluster_diversities = []
            
            for cluster_id in range(best_k):
                cluster_points = genomes[cluster_labels == cluster_id]
                cluster_sizes.append(len(cluster_points))
                
                if len(cluster_points) > 1:
                    cluster_distances = pdist(cluster_points, metric='euclidean')
                    diversity = np.mean(cluster_distances)
                else:
                    diversity = 0.0
                
                cluster_diversities.append(diversity)
            
            return {
                'cluster_count': best_k,
                'cluster_sizes': cluster_sizes,
                'cluster_diversity': cluster_diversities,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_labels': cluster_labels.tolist()
            }
            
        except Exception as e:
            print(f"K-means聚类失败：{str(e)}")
            return {
                'cluster_count': 1,
                'cluster_sizes': [len(genomes)],
                'cluster_diversity': [0.0]
            }
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except ImportError:
            # 简化版轮廓系数计算
            n_samples = len(data)
            if n_samples <= 1:
                return 0.0
            
            total_score = 0.0
            valid_samples = 0
            
            for i in range(min(n_samples, 100)):  # 限制计算量
                if len(np.unique(labels)) <= 1:
                    continue
                
                point = data[i]
                same_cluster = data[labels == labels[i]]
                other_clusters = data[labels != labels[i]]
                
                if len(same_cluster) > 1:
                    a = np.mean([np.linalg.norm(point - p) for p in same_cluster if not np.array_equal(point, p)])
                else:
                    a = 0.0
                
                if len(other_clusters) > 0:
                    b = np.min([np.mean([np.linalg.norm(point - p) for p in other_cluster]) 
                               for other_cluster in [data[labels == label] for label in np.unique(labels) if label != labels[i]]])
                else:
                    b = 0.0
                
                if max(a, b) > 0:
                    score = (b - a) / max(a, b)
                    total_score += score
                    valid_samples += 1
            
            return total_score / valid_samples if valid_samples > 0 else 0.0
        
        except Exception:
            return 0.0
    
    def _estimate_effective_population_size(self, cluster_count: int, total_individuals: int) -> float:
        """估算有效种群大小"""
        if cluster_count <= 0 or total_individuals <= 0:
            return 0.0
        
        # 基于聚类结构的有效种群大小估计
        # Ne = (N^2) / sum(ni^2) 其中ni是第i个聚类的大小
        ni_squared = sum(size ** 2 for size in [total_individuals // cluster_count] * cluster_count)
        ne = (total_individuals ** 2) / ni_squared if ni_squared > 0 else 0
        
        return ne
    
    def _calculate_inbreeding_coefficient(self, average_distance: float) -> float:
        """计算近交系数"""
        # 基于遗传距离的近交系数估计
        # F = 1 - (observed_heterozygosity / expected_heterozygosity)
        # 简化版本，基于平均距离推导
        max_possible_distance = 10.0  # 假设的最大距离
        relative_distance = min(average_distance / max_possible_distance, 1.0)
        
        # 近交系数与距离成反比
        inbreeding_coeff = 1.0 - relative_distance
        
        return max(0, min(1, inbreeding_coeff))
    
    def _analyze_diversity_trends(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析多样性趋势"""
        analysis = {}
        
        for metric_name, values in trend_data['metrics'].items():
            if len(values) < 2:
                analysis[metric_name] = {
                    'trend': 'insufficient_data',
                    'slope': 0.0,
                    'change_rate': 0.0
                }
                continue
            
            # 计算线性趋势
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # 确定趋势方向
            if slope > 0.001:
                trend = 'increasing'
            elif slope < -0.001:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # 计算变化率
            if values[0] != 0:
                change_rate = (values[-1] - values[0]) / abs(values[0])
            else:
                change_rate = 0.0
            
            analysis[metric_name] = {
                'trend': trend,
                'slope': slope,
                'change_rate': change_rate,
                'current_value': values[-1] if values else 0,
                'initial_value': values[0] if values else 0,
                'volatility': np.std(values) if len(values) > 1 else 0
            }
        
        return analysis
    
    def _generate_diversity_summary(self, generations: List[int]) -> Dict[str, Any]:
        """生成多样性摘要"""
        if not generations:
            return {}
        
        # 获取最新的几个世代
        recent_generations = generations[-min(5, len(generations)):]
        recent_metrics = [self.diversity_history[g] for g in recent_generations if g in self.diversity_history]
        
        if not recent_metrics:
            return {}
        
        # 计算平均值
        avg_shannon = np.mean([m.shannon_index for m in recent_metrics])
        avg_simpson = np.mean([m.simpson_index for m in recent_metrics])
        avg_richness = np.mean([m.richness for m in recent_metrics])
        avg_genetic_dist = np.mean([m.average_genetic_distance for m in recent_metrics])
        
        # 健康状态评估
        thresholds = self.config.get('diversity_thresholds', {})
        
        health_status = 'unknown'
        if avg_shannon > thresholds.get('high_diversity', 0.8):
            health_status = 'excellent'
        elif avg_shannon > thresholds.get('low_diversity', 0.3):
            health_status = 'good'
        elif avg_shannon > thresholds.get('critical_diversity', 0.1):
            health_status = 'concerning'
        else:
            health_status = 'critical'
        
        return {
            'current_status': health_status,
            'recent_averages': {
                'shannon_index': avg_shannon,
                'simpson_index': avg_simpson,
                'richness': avg_richness,
                'genetic_distance': avg_genetic_dist
            },
            'generation_count': len(recent_generations),
            'trend_summary': self._get_trend_summary(recent_generations)
        }
    
    def _get_trend_summary(self, generations: List[int]) -> Dict[str, str]:
        """获取趋势摘要"""
        summary = {}
        
        if len(generations) < 2:
            return {'message': '数据不足，无法确定趋势'}
        
        recent_metrics = [self.diversity_history[g] for g in generations if g in self.diversity_history]
        
        if not recent_metrics:
            return {'message': '没有可用的多样性数据'}
        
        # 比较开头和结尾的值
        first = recent_metrics[0]
        last = recent_metrics[-1]
        
        # Shannon指数趋势
        if last.shannon_index > first.shannon_index * 1.05:
            summary['shannon_trend'] = '显著提升'
        elif last.shannon_index < first.shannon_index * 0.95:
            summary['shannon_trend'] = '显著下降'
        else:
            summary['shannon_trend'] = '基本稳定'
        
        # 丰富度趋势
        if last.richness > first.richness:
            summary['richness_trend'] = '提升'
        elif last.richness < first.richness:
            summary['richness_trend'] = '下降'
        else:
            summary['richness_trend'] = '稳定'
        
        return summary
    
    def _create_default_metrics(self, generation: int) -> DiversityMetrics:
        """创建默认的多样性指标"""
        return DiversityMetrics(
            generation=generation,
            individual_count=0,
            shannon_index=0.0,
            shannon_evenness=0.0,
            simpson_index=0.0,
            simpson_evenness=0.0,
            richness=0,
            chao1_estimator=0.0,
            average_genetic_distance=0.0,
            minimum_genetic_distance=0.0,
            maximum_genetic_distance=0.0,
            phenotype_diversity=0.0,
            fitness_variance=0.0,
            effective_population_size=0.0,
            inbreeding_coefficient=0.0,
            cluster_count=0,
            cluster_sizes=[],
            cluster_diversity=[],
            timestamp=time.time()
        )
    
    def _calculate_relative_change(self, old_value: float, new_value: float) -> float:
        """计算相对变化"""
        if old_value == 0:
            return float('inf') if new_value > 0 else 0.0
        return (new_value - old_value) / abs(old_value)
    
    def _calculate_inter_population_similarity(self, genomes1: np.ndarray, genomes2: np.ndarray) -> float:
        """计算种群间相似度"""
        if len(genomes1) == 0 or len(genomes2) == 0:
            return 0.0
        
        # 计算种群间平均距离
        distances = []
        for genome1 in genomes1:
            for genome2 in genomes2:
                dist = np.linalg.norm(genome1 - genome2)
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # 转换为相似度（归一化）
        max_distance = 10.0  # 假设最大距离
        similarity = 1.0 - (avg_distance / max_distance)
        
        return max(0, min(1, similarity))
    
    def _generate_comparison_conclusions(self, comparison_results: Dict[str, Any]) -> List[str]:
        """生成比较结论"""
        conclusions = []
        
        diversity_diffs = comparison_results['diversity_differences']
        relative_changes = comparison_results['relative_changes']
        
        # Shannon指数分析
        shannon_change = relative_changes.get('shannon_relative_change', 0)
        if shannon_change > 0.1:
            conclusions.append("Shannon多样性指数显著提升，种群遗传多样性增加")
        elif shannon_change < -0.1:
            conclusions.append("Shannon多样性指数显著下降，存在遗传多样性丧失风险")
        
        # 丰富度分析
        richness_change = relative_changes.get('richness_relative_change', 0)
        if richness_change > 0.05:
            conclusions.append("基因型丰富度增加，进化创新活跃")
        elif richness_change < -0.05:
            conclusions.append("基因型丰富度减少，可能存在选择压力过强")
        
        # 遗传距离分析
        genetic_dist_change = diversity_diffs.get('genetic_distance_difference', 0)
        if genetic_dist_change > 0.1:
            conclusions.append("种群间遗传差异增大，进化分化明显")
        elif genetic_dist_change < -0.1:
            conclusions.append("种群间遗传差异减小，存在基因流动或同质化趋势")
        
        # 遗传相似度分析
        genetic_similarity = comparison_results.get('genetic_similarity', 0)
        if genetic_similarity > 0.7:
            conclusions.append("两个种群遗传相似度较高，可能来自共同祖先")
        elif genetic_similarity < 0.3:
            conclusions.append("两个种群遗传差异显著，可能是独立的进化谱系")
        
        return conclusions
    
    def _simple_t_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """简化的t检验"""
        if len(sample1) < 2 or len(sample2) < 2:
            return 0.0, 1.0
        
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        var1 = np.var(sample1, ddof=1) if len(sample1) > 1 else 0
        var2 = np.var(sample2, ddof=1) if len(sample2) > 1 else 0
        
        # 合并方差
        n1, n2 = len(sample1), len(sample2)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        if pooled_var == 0:
            return 0.0, 1.0
        
        # t统计量
        t_stat = (mean1 - mean2) / math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # 简化的p值计算
        df = n1 + n2 - 2
        if abs(t_stat) > 2.0:  # 简化阈值
            p_value = 0.01
        else:
            p_value = 0.1
        
        return t_stat, p_value
    
    def _analyze_diversity_changes(self, change_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析多样性变化模式"""
        if not change_points:
            return {
                'pattern': 'no_significant_changes',
                'stability': 'high',
                'recommendations': ['多样性变化平稳，继续监控']
            }
        
        # 分析变化模式
        increasing_changes = [cp for cp in change_points if cp['change_magnitude'] > 0]
        decreasing_changes = [cp for cp in change_points if cp['change_magnitude'] < 0]
        
        if len(increasing_changes) > len(decreasing_changes):
            pattern = 'increasing_diversity'
            stability = 'moderate'
            recommendations = ['多样性总体呈上升趋势，这是积极信号']
        elif len(decreasing_changes) > len(increasing_changes):
            pattern = 'decreasing_diversity'
            stability = 'low'
            recommendations = ['多样性下降趋势，需要关注基因库保护']
        else:
            pattern = 'mixed_changes'
            stability = 'moderate'
            recommendations = ['多样性变化复杂，建议深入分析具体原因']
        
        return {
            'pattern': pattern,
            'stability': stability,
            'change_count': len(change_points),
            'most_significant_change': max(change_points, key=lambda x: abs(x['change_magnitude'])),
            'recommendations': recommendations
        }
    
    def _assess_diversity_health(self, generations: List[int]) -> Dict[str, Any]:
        """评估多样性健康状态"""
        if not generations:
            return {'health_status': 'unknown', 'score': 0}
        
        # 获取最新的多样性数据
        recent_metrics = [self.diversity_history[g] for g in generations[-min(10, len(generations)):] 
                         if g in self.diversity_history]
        
        if not recent_metrics:
            return {'health_status': 'unknown', 'score': 0}
        
        # 计算健康评分
        avg_shannon = np.mean([m.shannon_index for m in recent_metrics])
        avg_simpson = np.mean([m.simpson_index for m in recent_metrics])
        avg_richness = np.mean([m.richness for m in recent_metrics])
        
        # 综合评分（0-100）
        shannon_score = min(100, avg_shannon * 50)  # Shannon指数评分
        simpson_score = min(100, avg_simpson * 100)  # Simpson指数评分
        richness_score = min(100, (avg_richness / 10) * 100)  # 丰富度评分
        
        overall_score = (shannon_score + simpson_score + richness_score) / 3
        
        # 确定健康状态
        if overall_score >= 80:
            health_status = 'excellent'
        elif overall_score >= 60:
            health_status = 'good'
        elif overall_score >= 40:
            health_status = 'fair'
        elif overall_score >= 20:
            health_status = 'poor'
        else:
            health_status = 'critical'
        
        return {
            'health_status': health_status,
            'score': overall_score,
            'component_scores': {
                'shannon_score': shannon_score,
                'simpson_score': simpson_score,
                'richness_score': richness_score
            },
            'trends': {
                'shannon_trend': self._determine_trend([m.shannon_index for m in recent_metrics]),
                'richness_trend': self._determine_trend([m.richness for m in recent_metrics])
            }
        }
    
    def _generate_diversity_recommendations(self, summary_stats: Dict, health_assessment: Dict) -> List[str]:
        """生成多样性管理建议"""
        recommendations = []
        
        health_status = health_assessment.get('health_status', 'unknown')
        score = health_assessment.get('score', 0)
        
        # 基于健康状态的建议
        if health_status == 'critical':
            recommendations.extend([
                "紧急：种群多样性处于危险水平，建议立即增加基因流动",
                "考虑引入外部基因或增加变异率以防止近交衰退",
                "监控适应度变化，防止遗传漂变导致的适应性丧失"
            ])
        elif health_status == 'poor':
            recommendations.extend([
                "多样性水平偏低，建议采取保护措施",
                "增加种群规模或改善基因流动",
                "定期监测遗传健康指标"
            ])
        elif health_status == 'fair':
            recommendations.extend([
                "多样性水平一般，可适当优化进化参数",
                "关注种群结构变化，避免过度选择",
                "维持适度的基因流动"
            ])
        elif health_status in ['good', 'excellent']:
            recommendations.extend([
                "多样性水平良好，继续保持当前进化策略",
                "可适当进行精细化调整",
                "为长期进化保持基因库多样性"
            ])
        
        # 基于趋势的建议
        trends = health_assessment.get('trends', {})
        for metric, trend in trends.items():
            if trend == 'decreasing':
                if 'shannon' in metric:
                    recommendations.append("Shannon指数下降趋势，需要增加基因多样性")
                elif 'richness' in metric:
                    recommendations.append("基因型丰富度下降，建议增加变异压力")
            elif trend == 'increasing':
                if 'shannon' in metric:
                    recommendations.append("Shannon指数上升趋势良好，继续保持")
                elif 'richness' in metric:
                    recommendations.append("基因型丰富度增加，进化创新活跃")
        
        return recommendations
    
    def _determine_trend(self, values: List[float]) -> str:
        """确定数值序列的趋势"""
        if len(values) < 3:
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


# 配置日志
import logging
logger = logging.getLogger(__name__)