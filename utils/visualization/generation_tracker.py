#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
世代跟踪器
Generation Tracker

该模块负责跟踪和记录进化过程中的每一代数据，包括个体信息、
适应度统计、遗传多样性变化等。支持数据持久化和历史回溯功能。

功能特性：
- 世代数据自动记录和存储
- 历史数据查询和回溯
- 数据压缩和清理
- 自动保存和恢复
- 统计指标计算

Author: 进化树可视化系统
Date: 2025-11-13
"""

import json
import time
import pickle
import gzip
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging

try:
    from .evolution_tree import Individual
except ImportError:
    from utils.visualization.evolution_tree import Individual

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """
    世代统计数据
    Generation Statistics
    """
    generation: int
    individual_count: int
    timestamp: float
    
    # 适应度统计
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    median_fitness: float
    std_fitness: float
    
    # 多样性统计
    genetic_diversity: float
    phenotype_diversity: float
    
    # 其他指标
    convergence_rate: float
    innovation_rate: float
    extinction_rate: float
    
    # 元数据
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class GenerationRecord:
    """
    世代记录
    Generation Record
    """
    generation: int
    individuals: List[Individual]
    stats: GenerationStats
    parent_generation: Optional[int] = None
    evolution_operators: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evolution_operators is None:
            self.evolution_operators = {}


class GenerationTracker:
    """
    世代跟踪器
    Generation Tracker
    
    负责记录、存储和管理进化过程中的世代数据，
    提供历史查询、统计分析等功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化世代跟踪器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._default_config()
        
        # 数据存储
        self.generation_records = {}  # 世代记录字典
        self.generation_history = deque(maxlen=self.config.get('max_history_size', 1000))
        
        # 统计缓存
        self.fitness_history = deque(maxlen=self.config.get('fitness_history_size', 100))
        self.diversity_history = deque(maxlen=self.config.get('diversity_history_size', 100))
        
        # 文件存储配置
        self.storage_config = self.config.get('storage', {})
        self.auto_save = self.storage_config.get('auto_save', True)
        self.save_interval = self.storage_config.get('save_interval', 50)
        self.compress_data = self.storage_config.get('compress_data', True)
        self.storage_path = Path(self.storage_config.get('storage_path', './evolution_data'))
        
        # 确保存储目录存在
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # 状态变量
        self.last_save_generation = 0
        self.total_saved_generations = 0
        
        logger.info("世代跟踪器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_size': 1000,
            'fitness_history_size': 100,
            'diversity_history_size': 100,
            'storage': {
                'auto_save': True,
                'save_interval': 50,
                'compress_data': True,
                'storage_path': './evolution_data',
                'max_file_size_mb': 100
            },
            'statistics': {
                'calculate_diversity': True,
                'track_evolutionary_operators': True,
                'enable_convergence_analysis': True
            }
        }
    
    def add_generation(self, 
                      individuals: List[Individual],
                      generation: int,
                      metadata: Optional[Dict[str, Any]] = None) -> GenerationRecord:
        """
        添加一代数据
        
        Args:
            individuals: 当前世代的个体列表
            generation: 世代编号
            metadata: 额外的元数据
            
        Returns:
            创建的世代记录
        """
        try:
            logger.info(f"添加世代 {generation} 数据，包含 {len(individuals)} 个个体")
            
            # 创建世代统计
            stats = self._calculate_generation_stats(individuals, generation)
            
            # 创建世代记录
            record = GenerationRecord(
                generation=generation,
                individuals=individuals.copy(),  # 深拷贝以防止外部修改
                stats=stats,
                metadata=metadata or {}
            )
            
            # 存储记录
            self.generation_records[generation] = record
            self.generation_history.append(generation)
            
            # 更新历史缓存
            self._update_history_caches(stats)
            
            # 自动保存
            if self.auto_save and generation % self.save_interval == 0:
                self._auto_save()
            
            logger.info(f"世代 {generation} 数据添加完成")
            return record
            
        except Exception as e:
            logger.error(f"添加世代 {generation} 数据失败：{str(e)}")
            raise
    
    def get_generation(self, generation: int) -> Optional[GenerationRecord]:
        """
        获取指定世代的数据
        
        Args:
            generation: 世代编号
            
        Returns:
            世代记录，如果不存在则返回None
        """
        return self.generation_records.get(generation)
    
    def get_generations_range(self, 
                             start_generation: int,
                             end_generation: int,
                             step: int = 1) -> List[GenerationRecord]:
        """
        获取指定范围的世代数据
        
        Args:
            start_generation: 开始世代
            end_generation: 结束世代
            step: 步长
            
        Returns:
            世代记录列表
        """
        records = []
        for gen in range(start_generation, end_generation + 1, step):
            record = self.generation_records.get(gen)
            if record:
                records.append(record)
        
        return records
    
    def get_latest_generations(self, count: int = 10) -> List[GenerationRecord]:
        """
        获取最新的N代数据
        
        Args:
            count: 要获取的世代数量
            
        Returns:
            世代记录列表
        """
        if not self.generation_history:
            return []
        
        # 获取最新的世代编号
        latest_generations = sorted(self.generation_history)[-count:]
        return [self.generation_records[gen] for gen in latest_generations]
    
    def calculate_fitness_trend(self, 
                               generations: List[int] = None,
                               metric: str = 'best') -> Dict[str, Any]:
        """
        计算适应度趋势
        
        Args:
            generations: 要分析的世代列表
            metric: 趋势指标 ('best', 'average', 'worst', 'std')
            
        Returns:
            趋势数据
        """
        try:
            # 确定要分析的世代
            if generations is None:
                generations = sorted(self.generation_history)
            
            trend_data = {
                'generations': [],
                'values': [],
                'metric': metric
            }
            
            for gen in generations:
                record = self.generation_records.get(gen)
                if record:
                    # 根据指标获取值
                    if metric == 'best':
                        value = record.stats.best_fitness
                    elif metric == 'average':
                        value = record.stats.average_fitness
                    elif metric == 'worst':
                        value = record.stats.worst_fitness
                    elif metric == 'std':
                        value = record.stats.std_fitness
                    else:
                        value = record.stats.average_fitness
                    
                    trend_data['generations'].append(gen)
                    trend_data['values'].append(value)
            
            return {
                'success': True,
                'trend_data': trend_data,
                'analysis': self._analyze_trend(trend_data)
            }
            
        except Exception as e:
            logger.error(f"计算适应度趋势失败：{str(e)}")
            return {
                'success': False,
                'message': f'计算适应度趋势失败：{str(e)}'
            }
    
    def analyze_convergence(self, 
                           window_size: int = 10,
                           convergence_threshold: float = 0.01) -> Dict[str, Any]:
        """
        分析收敛性
        
        Args:
            window_size: 分析窗口大小
            convergence_threshold: 收敛阈值
            
        Returns:
            收敛分析结果
        """
        try:
            if len(self.fitness_history) < window_size:
                return {
                    'success': False,
                    'message': '数据不足，无法进行收敛分析'
                }
            
            # 计算移动平均和标准差
            recent_fitness = list(self.fitness_history)[-window_size:]
            moving_avg = np.mean(recent_fitness)
            moving_std = np.std(recent_fitness)
            
            # 计算收敛指标
            convergence_score = 1.0 - (moving_std / max(abs(moving_avg), 1e-10))
            is_converged = convergence_score > (1.0 - convergence_threshold)
            
            # 计算趋势方向
            if len(recent_fitness) >= 2:
                recent_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
            else:
                recent_trend = 0
            
            result = {
                'success': True,
                'convergence_analysis': {
                    'is_converged': is_converged,
                    'convergence_score': convergence_score,
                    'moving_average': moving_avg,
                    'moving_std': moving_std,
                    'recent_trend': recent_trend,
                    'window_size': window_size,
                    'threshold': convergence_threshold
                },
                'recommendations': self._generate_convergence_recommendations(
                    is_converged, convergence_score, recent_trend
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"收敛性分析失败：{str(e)}")
            return {
                'success': False,
                'message': f'收敛性分析失败：{str(e)}'
            }
    
    def export_generation_data(self, 
                              generations: List[int] = None,
                              format: str = 'json',
                              include_individuals: bool = True) -> Dict[str, Any]:
        """
        导出世代数据
        
        Args:
            generations: 要导出的世代列表
            format: 导出格式 ('json', 'csv', 'pickle')
            include_individuals: 是否包含个体详细数据
            
        Returns:
            导出结果
        """
        try:
            # 确定要导出的世代
            if generations is None:
                generations = sorted(self.generation_history)
            
            # 准备导出数据
            export_data = {
                'metadata': {
                    'export_time': time.time(),
                    'total_generations': len(generations),
                    'format': format,
                    'include_individuals': include_individuals
                },
                'generations': {}
            }
            
            for gen in generations:
                record = self.generation_records.get(gen)
                if record:
                    generation_data = {
                        'stats': record.stats.to_dict(),
                        'metadata': record.metadata
                    }
                    
                    if include_individuals:
                        generation_data['individuals'] = [
                            asdict(ind) for ind in record.individuals
                        ]
                    
                    export_data['generations'][str(gen)] = generation_data
            
            # 根据格式导出
            timestamp = int(time.time())
            file_name = f"evolution_generations_{timestamp}"
            
            if format.lower() == 'json':
                file_path = self.storage_path / f"{file_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == 'pickle':
                file_path = self.storage_path / f"{file_name}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(export_data, f)
            
            else:
                raise ValueError(f"不支持的导出格式：{format}")
            
            logger.info(f"世代数据导出成功：{file_path}")
            return {
                'success': True,
                'file_path': str(file_path),
                'exported_generations': len(generations),
                'data_size': len(json.dumps(export_data)) if format == 'json' else None
            }
            
        except Exception as e:
            logger.error(f"导出世代数据失败：{str(e)}")
            return {
                'success': False,
                'message': f'导出世代数据失败：{str(e)}'
            }
    
    def load_generation_data(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载世代数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载结果
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'message': f'文件不存在：{file_path}'
                }
            
            # 根据文件扩展名选择加载方式
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                return {
                    'success': False,
                    'message': f'不支持的文件格式：{file_path.suffix}'
                }
            
            # 重建世代记录
            loaded_count = 0
            for gen_str, gen_data in data.get('generations', {}).items():
                gen = int(gen_str)
                
                # 重建统计数据
                stats_data = gen_data['stats']
                stats = GenerationStats(**stats_data)
                
                # 重建个体数据（如果包含）
                individuals = []
                if 'individuals' in gen_data:
                    for ind_data in gen_data['individuals']:
                        individual = Individual(**ind_data)
                        individuals.append(individual)
                
                # 重建世代记录
                record = GenerationRecord(
                    generation=gen,
                    individuals=individuals,
                    stats=stats,
                    metadata=gen_data.get('metadata', {})
                )
                
                self.generation_records[gen] = record
                self.generation_history.append(gen)
                loaded_count += 1
            
            logger.info(f"成功加载 {loaded_count} 个世代的数据")
            return {
                'success': True,
                'loaded_generations': loaded_count,
                'total_records': len(self.generation_records)
            }
            
        except Exception as e:
            logger.error(f"加载世代数据失败：{str(e)}")
            return {
                'success': False,
                'message': f'加载世代数据失败：{str(e)}'
            }
    
    def clear_history(self, 
                     keep_recent: int = 100,
                     auto_save_before_clear: bool = True):
        """
        清除历史数据
        
        Args:
            keep_recent: 保留最近的N代数据
            auto_save_before_clear: 清除前是否自动保存
        """
        try:
            if auto_save_before_clear:
                self._auto_save()
            
            # 保留最新的世代
            recent_generations = sorted(self.generation_history)[-keep_recent:]
            
            # 清除旧数据
            old_generations = set(self.generation_history) - set(recent_generations)
            for gen in old_generations:
                del self.generation_records[gen]
            
            # 更新历史记录
            self.generation_history = deque(recent_generations, maxlen=self.config.get('max_history_size', 1000))
            
            # 清除缓存
            self.fitness_history.clear()
            self.diversity_history.clear()
            
            logger.info(f"已清除 {len(old_generations)} 个历史世代，保留 {len(recent_generations)} 个最新世代")
            
        except Exception as e:
            logger.error(f"清除历史数据失败：{str(e)}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取总体统计信息
        
        Returns:
            统计摘要
        """
        if not self.generation_records:
            return {
                'total_generations': 0,
                'message': '没有世代数据'
            }
        
        # 计算基本统计
        total_generations = len(self.generation_records)
        generation_range = (min(self.generation_history), max(self.generation_history))
        total_individuals = sum(record.individual_count for record in self.generation_records.values())
        
        # 适应度统计
        all_best_fitness = [record.stats.best_fitness for record in self.generation_records.values()]
        best_overall = max(all_best_fitness)
        
        # 计算平均适应度提升率
        improvement_rates = []
        sorted_generations = sorted(self.generation_records.keys())
        for i in range(1, len(sorted_generations)):
            prev_gen = sorted_generations[i-1]
            curr_gen = sorted_generations[i]
            if prev_gen in self.generation_records and curr_gen in self.generation_records:
                prev_best = self.generation_records[prev_gen].stats.best_fitness
                curr_best = self.generation_records[curr_gen].stats.best_fitness
                if prev_best > 0:
                    improvement_rate = (curr_best - prev_best) / prev_best
                    improvement_rates.append(improvement_rate)
        
        avg_improvement_rate = np.mean(improvement_rates) if improvement_rates else 0
        
        return {
            'total_generations': total_generations,
            'generation_range': generation_range,
            'total_individuals': total_individuals,
            'best_overall_fitness': best_overall,
            'average_improvement_rate': avg_improvement_rate,
            'last_updated': max(record.stats.timestamp for record in self.generation_records.values()),
            'storage_path': str(self.storage_path),
            'auto_save_enabled': self.auto_save
        }
    
    def _calculate_generation_stats(self, 
                                  individuals: List[Individual],
                                  generation: int) -> GenerationStats:
        """计算世代统计数据"""
        if not individuals:
            # 返回默认统计
            return GenerationStats(
                generation=generation,
                individual_count=0,
                timestamp=time.time(),
                best_fitness=0,
                worst_fitness=0,
                average_fitness=0,
                median_fitness=0,
                std_fitness=0,
                genetic_diversity=0,
                phenotype_diversity=0,
                convergence_rate=0,
                innovation_rate=0,
                extinction_rate=0
            )
        
        # 适应度统计
        fitness_values = [ind.fitness for ind in individuals]
        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)
        average_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)
        std_fitness = np.std(fitness_values)
        
        # 遗传多样性计算
        genetic_diversity = self._calculate_genetic_diversity(individuals)
        phenotype_diversity = self._calculate_phenotype_diversity(individuals)
        
        # 进化指标计算
        convergence_rate = self._calculate_convergence_rate(individuals)
        innovation_rate = self._calculate_innovation_rate(individuals, generation)
        extinction_rate = self._calculate_extinction_rate(individuals, generation)
        
        return GenerationStats(
            generation=generation,
            individual_count=len(individuals),
            timestamp=time.time(),
            best_fitness=best_fitness,
            worst_fitness=worst_fitness,
            average_fitness=average_fitness,
            median_fitness=median_fitness,
            std_fitness=std_fitness,
            genetic_diversity=genetic_diversity,
            phenotype_diversity=phenotype_diversity,
            convergence_rate=convergence_rate,
            innovation_rate=innovation_rate,
            extinction_rate=extinction_rate
        )
    
    def _calculate_genetic_diversity(self, individuals: List[Individual]) -> float:
        """计算遗传多样性"""
        if len(individuals) < 2:
            return 0.0
        
        # 收集所有基因位点
        genomes = np.array([ind.genome for ind in individuals])
        
        # 计算每个位点的方差
        locus_variances = np.var(genomes, axis=0)
        
        # 计算平均方差作为多样性指标
        return np.mean(locus_variances)
    
    def _calculate_phenotype_diversity(self, individuals: List[Individual]) -> float:
        """计算表型多样性"""
        if len(individuals) < 2:
            return 0.0
        
        fitness_values = np.array([ind.fitness for ind in individuals])
        return np.std(fitness_values)
    
    def _calculate_convergence_rate(self, individuals: List[Individual]) -> float:
        """计算收敛率"""
        if len(individuals) < 2:
            return 0.0
        
        # 基于适应度分布的收敛指标
        fitness_values = np.array([ind.fitness for ind in individuals])
        coefficient_of_variation = np.std(fitness_values) / (np.mean(fitness_values) + 1e-10)
        
        # 收敛率 = 1 - 变异系数
        return max(0, 1 - coefficient_of_variation)
    
    def _calculate_innovation_rate(self, individuals: List[Individual], generation: int) -> float:
        """计算创新率"""
        if generation == 0:
            return 0.0
        
        # 计算有父代的个体比例
        individuals_with_parents = sum(1 for ind in individuals if ind.parent_id is not None)
        return individuals_with_parents / len(individuals) if individuals else 0
    
    def _calculate_extinction_rate(self, individuals: List[Individual], generation: int) -> float:
        """计算灭绝率"""
        if generation == 0:
            return 0.0
        
        # 计算适应度为0或负值的个体比例
        extinct_individuals = sum(1 for ind in individuals if ind.fitness <= 0)
        return extinct_individuals / len(individuals) if individuals else 0
    
    def _update_history_caches(self, stats: GenerationStats):
        """更新历史缓存"""
        self.fitness_history.append(stats.average_fitness)
        self.diversity_history.append(stats.genetic_diversity)
    
    def _analyze_trend(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析趋势"""
        values = trend_data['values']
        
        if len(values) < 2:
            return {'trend_direction': 'insufficient_data'}
        
        # 计算线性趋势
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # 计算趋势强度
        trend_strength = abs(slope)
        
        # 确定趋势方向
        if slope > 0.001:
            direction = 'increasing'
        elif slope < -0.001:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'trend_direction': direction,
            'trend_slope': slope,
            'trend_strength': trend_strength,
            'confidence': self._calculate_trend_confidence(values)
        }
    
    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """计算趋势置信度"""
        if len(values) < 3:
            return 0.0
        
        # 基于数值变化的一致性计算置信度
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        positive_changes = sum(1 for c in changes if c > 0)
        negative_changes = sum(1 for c in changes if c < 0)
        
        total_changes = len(changes)
        consistency = max(positive_changes, negative_changes) / total_changes
        
        return consistency
    
    def _generate_convergence_recommendations(self, 
                                            is_converged: bool,
                                            convergence_score: float,
                                            trend: float) -> List[str]:
        """生成收敛建议"""
        recommendations = []
        
        if is_converged:
            recommendations.append("系统已收敛，建议考虑停止进化或调整参数")
        else:
            if convergence_score < 0.5:
                recommendations.append("收敛较慢，建议增加变异率或种群规模")
            if trend < 0:
                recommendations.append("适应度呈下降趋势，建议检查选择策略")
        
        return recommendations
    
    def _auto_save(self):
        """自动保存数据"""
        try:
            timestamp = int(time.time())
            file_name = f"auto_save_generations_{timestamp}.json"
            file_path = self.storage_path / file_name
            
            # 准备要保存的数据
            save_data = {
                'metadata': {
                    'save_time': time.time(),
                    'total_generations': len(self.generation_records),
                    'auto_save': True
                },
                'generation_records': {
                    str(gen): {
                        'stats': record.stats.to_dict(),
                        'individual_count': record.individual_count,
                        'generation': record.generation
                    }
                    for gen, record in self.generation_records.items()
                }
            }
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.last_save_generation = max(self.generation_history) if self.generation_history else 0
            self.total_saved_generations += 1
            
            logger.info(f"自动保存完成：{file_path}")
            
        except Exception as e:
            logger.error(f"自动保存失败：{str(e)}")


# 导入必要的库
import numpy as np