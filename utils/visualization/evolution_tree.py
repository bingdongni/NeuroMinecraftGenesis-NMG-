#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化树可视化系统主类
Evolution Tree Visualization System Main Class

该模块实现了动态进化过程的图形化展示，支持实时显示种群变化、
适应度曲线、遗传多样性等功能。提供完整的交互式进化树操作界面。

主要功能：
- 进化树的动态渲染和更新
- 多代进化历史记录和回放
- 实时适应度和多样性监控
- 交互式节点操作和筛选
- 数据导出和保存功能

Author: 进化树可视化系统
Date: 2025-11-13
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .tree_renderer import TreeRenderer
from .generation_tracker import GenerationTracker
from .fitness_visualizer import FitnessVisualizer
from .diversity_analyzer import DiversityAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    个体数据结构
    Individual Data Structure
    
    表示进化过程中的一个个体，包含基因组、适应度和其他相关信息
    """
    id: int  # 个体唯一标识符
    genome: Union[List[float], np.ndarray]  # 基因组序列
    fitness: float  # 适应度值
    generation: int  # 所属世代
    parent_id: Optional[int] = None  # 父代个体ID
    mutations: List[int] = None  # 变异发生的位置
    timestamp: float = None  # 时间戳
    
    def __post_init__(self):
        if self.mutations is None:
            self.mutations = []
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class EvolutionNode:
    """
    进化树节点数据结构
    Evolution Tree Node Data Structure
    
    表示进化树中的节点，包含个体信息和树结构信息
    """
    individual: Individual
    children: List['EvolutionNode'] = None
    depth: int = 0
    branch_length: float = 1.0
    cluster_id: Optional[int] = None  # 聚类ID
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'EvolutionNode'):
        """添加子节点"""
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_all_descendants(self) -> List['EvolutionNode']:
        """获取所有后代节点"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants


class EvolutionTree:
    """
    进化树可视化主类
    Evolution Tree Visualization Main Class
    
    这是整个进化树可视化系统的核心类，负责：
    - 维护进化历史数据
    - 协调各个可视化组件
    - 提供统一的接口给前端
    - 处理用户交互和操作
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化进化树可视化系统
        
        Args:
            config: 配置参数字典，包含可视化设置
        """
        self.config = config or self._default_config()
        self.root = None  # 进化树根节点
        self.all_individuals = {}  # 所有个体数据
        self.current_generation = 0  # 当前世代
        self.generation_history = []  # 世代历史记录
        
        # 初始化各个组件
        self.tree_renderer = TreeRenderer(self.config.get('tree_renderer', {}))
        self.generation_tracker = GenerationTracker(self.config.get('generation_tracker', {}))
        self.fitness_visualizer = FitnessVisualizer(self.config.get('fitness_visualizer', {}))
        self.diversity_analyzer = DiversityAnalyzer(self.config.get('diversity_analyzer', {}))
        
        # 动画控制
        self.animation_speed = self.config.get('animation_speed', 1.0)
        self.is_animating = False
        
        logger.info("进化树可视化系统初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'tree_renderer': {
                'canvas_width': 1200,
                'canvas_height': 800,
                'node_radius': 8,
                'branch_width': 2,
                'color_scheme': 'viridis',
                'show_labels': True
            },
            'generation_tracker': {
                'max_history_size': 1000,
                'auto_save': True,
                'save_interval': 50
            },
            'fitness_visualizer': {
                'chart_width': 800,
                'chart_height': 400,
                'show_trend': True,
                'show_annotations': True
            },
            'diversity_analyzer': {
                'metrics': ['shannon', 'simpson', 'richness'],
                'window_size': 10
            },
            'animation_speed': 1.0,
            'data_export': {
                'formats': ['json', 'csv'],
                'include_genomes': True,
                'include_tree_structure': True
            }
        }
    
    def render_evolution_tree(self, 
                            start_generation: int = 0,
                            end_generation: Optional[int] = None,
                            filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        渲染进化树
        
        Args:
            start_generation: 开始渲染的世代
            end_generation: 结束渲染的世代（None表示到最新世代）
            filter_criteria: 筛选条件
            
        Returns:
            包含进化树数据的字典
        """
        try:
            logger.info(f"开始渲染进化树：世代 {start_generation} 到 {end_generation}")
            
            # 确定渲染范围
            if end_generation is None:
                end_generation = self.current_generation
            
            # 获取渲染数据
            render_data = self.tree_renderer.render_tree(
                self.all_individuals,
                start_generation,
                end_generation,
                filter_criteria
            )
            
            # 添加元数据
            result = {
                'tree_data': render_data,
                'metadata': {
                    'start_generation': start_generation,
                    'end_generation': end_generation,
                    'total_individuals': len(self.all_individuals),
                    'timestamp': time.time(),
                    'config': self.config
                },
                'success': True,
                'message': '进化树渲染成功'
            }
            
            logger.info(f"进化树渲染完成，包含 {render_data.get('node_count', 0)} 个节点")
            return result
            
        except Exception as e:
            logger.error(f"进化树渲染失败：{str(e)}")
            return {
                'success': False,
                'message': f'进化树渲染失败：{str(e)}',
                'tree_data': None
            }
    
    def update_generation(self, 
                         individuals: List[Individual],
                         generation: int,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        更新世代数据
        
        Args:
            individuals: 当前世代的个体列表
            generation: 世代编号
            metadata: 额外的元数据
            
        Returns:
            更新结果
        """
        try:
            logger.info(f"更新世代 {generation} 数据，包含 {len(individuals)} 个个体")
            
            # 更新世代跟踪器
            generation_data = self.generation_tracker.add_generation(
                individuals, generation, metadata
            )
            
            # 更新个体数据库
            for individual in individuals:
                self.all_individuals[individual.id] = individual
            
            # 更新当前世代
            self.current_generation = max(self.current_generation, generation)
            
            # 重新构建进化树结构
            self._rebuild_tree()
            
            # 计算适应度统计
            fitness_stats = self.fitness_visualizer.update_generation_stats(
                individuals, generation
            )
            
            # 计算多样性指标
            diversity_stats = self.diversity_analyzer.analyze_diversity(
                individuals, generation
            )
            
            result = {
                'success': True,
                'generation': generation,
                'individual_count': len(individuals),
                'fitness_stats': fitness_stats,
                'diversity_stats': diversity_stats,
                'tree_structure_updated': True,
                'message': f'世代 {generation} 数据更新成功'
            }
            
            logger.info(f"世代 {generation} 数据更新完成")
            return result
            
        except Exception as e:
            logger.error(f"更新世代 {generation} 失败：{str(e)}")
            return {
                'success': False,
                'message': f'更新世代 {generation} 失败：{str(e)}'
            }
    
    def animate_evolution(self, 
                         generations: List[int] = None,
                         speed: float = 1.0,
                         on_frame: Optional[callable] = None) -> Dict[str, Any]:
        """
        动画化进化过程
        
        Args:
            generations: 要动画化的世代列表
            speed: 动画速度倍数
            on_frame: 每帧回调函数
            
        Returns:
            动画控制结果
        """
        try:
            if self.is_animating:
                return {
                    'success': False,
                    'message': '动画正在进行中，请等待完成'
                }
            
            # 设置动画参数
            self.animation_speed = speed
            self.is_animating = True
            
            # 确定要动画化的世代
            if generations is None:
                generations = list(range(self.current_generation + 1))
            
            logger.info(f"开始动画化进化过程，世代范围：{min(generations)}-{max(generations)}")
            
            # 动画控制参数
            animation_data = {
                'generations': generations,
                'speed': speed,
                'total_frames': len(generations),
                'current_frame': 0,
                'on_frame': on_frame
            }
            
            # 这里可以集成实际的动画库，如 matplotlib.animation
            # 目前提供接口框架
            result = {
                'success': True,
                'animation_data': animation_data,
                'message': '进化动画启动成功'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"进化动画启动失败：{str(e)}")
            self.is_animating = False
            return {
                'success': False,
                'message': f'进化动画启动失败：{str(e)}'
            }
    
    def analyze_diversity(self, 
                         generation: int = None,
                         metric_types: List[str] = None) -> Dict[str, Any]:
        """
        分析遗传多样性
        
        Args:
            generation: 要分析的世代（None表示最新世代）
            metric_types: 要计算的指标类型列表
            
        Returns:
            多样性分析结果
        """
        try:
            # 确定要分析的世代
            if generation is None:
                generation = self.current_generation
            
            # 获取该世代的个体数据
            generation_individuals = [
                ind for ind in self.all_individuals.values() 
                if ind.generation == generation
            ]
            
            if not generation_individuals:
                return {
                    'success': False,
                    'message': f'世代 {generation} 没有个体数据'
                }
            
            # 执行多样性分析
            diversity_result = self.diversity_analyzer.analyze_diversity(
                generation_individuals,
                generation,
                metric_types
            )
            
            logger.info(f"世代 {generation} 多样性分析完成")
            return {
                'success': True,
                'generation': generation,
                'diversity_metrics': diversity_result,
                'individual_count': len(generation_individuals)
            }
            
        except Exception as e:
            logger.error(f"多样性分析失败：{str(e)}")
            return {
                'success': False,
                'message': f'多样性分析失败：{str(e)}'
            }
    
    def export_tree_data(self, 
                        file_path: Union[str, Path],
                        format: str = 'json',
                        include_genomes: bool = True,
                        include_tree_structure: bool = True) -> Dict[str, Any]:
        """
        导出进化树数据
        
        Args:
            file_path: 输出文件路径
            format: 导出格式 ('json', 'csv', 'xlsx')
            include_genomes: 是否包含基因组数据
            include_tree_structure: 是否包含树结构数据
            
        Returns:
            导出结果
        """
        try:
            logger.info(f"开始导出进化树数据到 {file_path}")
            
            # 准备导出数据
            export_data = {
                'metadata': {
                    'export_time': time.time(),
                    'total_individuals': len(self.all_individuals),
                    'current_generation': self.current_generation,
                    'config': self.config
                },
                'individuals': {},
                'tree_structure': {},
                'generation_history': self.generation_history
            }
            
            # 添加个体数据
            for ind_id, individual in self.all_individuals.items():
                if include_genomes:
                    export_data['individuals'][str(ind_id)] = asdict(individual)
                else:
                    # 只导出基本信息
                    export_data['individuals'][str(ind_id)] = {
                        'id': individual.id,
                        'fitness': individual.fitness,
                        'generation': individual.generation,
                        'parent_id': individual.parent_id,
                        'timestamp': individual.timestamp
                    }
            
            # 添加树结构数据
            if include_tree_structure and self.root:
                export_data['tree_structure'] = self._serialize_tree(self.root)
            
            # 根据格式导出
            file_path = Path(file_path)
            
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == 'csv':
                # 导出个体数据为CSV
                df = pd.DataFrame.from_dict(export_data['individuals'], orient='index')
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            elif format.lower() == 'xlsx':
                # 导出为Excel格式
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 个体数据表
                    df = pd.DataFrame.from_dict(export_data['individuals'], orient='index')
                    df.to_excel(writer, sheet_name='个体数据', index=False)
                    
                    # 世代历史表
                    if self.generation_history:
                        history_df = pd.DataFrame(self.generation_history)
                        history_df.to_excel(writer, sheet_name='世代历史', index=False)
            
            else:
                raise ValueError(f"不支持的导出格式：{format}")
            
            logger.info(f"进化树数据导出成功：{file_path}")
            return {
                'success': True,
                'file_path': str(file_path),
                'format': format,
                'data_size': len(export_data),
                'message': f'数据导出成功到 {file_path}'
            }
            
        except Exception as e:
            logger.error(f"导出进化树数据失败：{str(e)}")
            return {
                'success': False,
                'message': f'导出进化树数据失败：{str(e)}'
            }
    
    def get_fitness_trend(self, 
                         generations: List[int] = None,
                         metric: str = 'best') -> Dict[str, Any]:
        """
        获取适应度趋势数据
        
        Args:
            generations: 要分析的世代列表
            metric: 趋势指标 ('best', 'average', 'worst', 'std')
            
        Returns:
            适应度趋势数据
        """
        try:
            # 获取趋势数据
            trend_data = self.fitness_visualizer.get_fitness_trend(
                self.all_individuals, generations, metric
            )
            
            return {
                'success': True,
                'metric': metric,
                'trend_data': trend_data,
                'message': '适应度趋势数据获取成功'
            }
            
        except Exception as e:
            logger.error(f"获取适应度趋势失败：{str(e)}")
            return {
                'success': False,
                'message': f'获取适应度趋势失败：{str(e)}'
            }
    
    def _rebuild_tree(self):
        """重新构建进化树结构"""
        try:
            # 按世代排序个体
            sorted_individuals = sorted(
                self.all_individuals.values(),
                key=lambda x: x.generation
            )
            
            # 构建树结构
            if not sorted_individuals:
                self.root = None
                return
            
            # 创建根节点
            root_individual = sorted_individuals[0]
            self.root = EvolutionNode(root_individual)
            
            # 构建子树
            node_map = {root_individual.id: self.root}
            
            for individual in sorted_individuals[1:]:
                # 创建节点
                node = EvolutionNode(individual)
                node_map[individual.id] = node
                
                # 连接父子关系
                if individual.parent_id is not None and individual.parent_id in node_map:
                    parent_node = node_map[individual.parent_id]
                    parent_node.add_child(node)
            
            logger.info("进化树结构重建完成")
            
        except Exception as e:
            logger.error(f"重建进化树结构失败：{str(e)}")
    
    def _serialize_tree(self, node: EvolutionNode) -> Dict[str, Any]:
        """序列化树结构"""
        if node is None:
            return None
        
        return {
            'individual_id': node.individual.id,
            'generation': node.individual.generation,
            'fitness': node.individual.fitness,
            'depth': node.depth,
            'branch_length': node.branch_length,
            'children': [self._serialize_tree(child) for child in node.children]
        }
    
    def get_generation_statistics(self, generation: int = None) -> Dict[str, Any]:
        """
        获取世代统计信息
        
        Args:
            generation: 世代编号（None表示最新世代）
            
        Returns:
            世代统计信息
        """
        try:
            # 确定要统计的世代
            if generation is None:
                generation = self.current_generation
            
            # 获取该世代的个体
            generation_individuals = [
                ind for ind in self.all_individuals.values() 
                if ind.generation == generation
            ]
            
            if not generation_individuals:
                return {
                    'success': False,
                    'message': f'世代 {generation} 没有个体数据'
                }
            
            # 计算统计指标
            fitness_values = [ind.fitness for ind in generation_individuals]
            
            stats = {
                'generation': generation,
                'individual_count': len(generation_individuals),
                'fitness_stats': {
                    'best': max(fitness_values),
                    'worst': min(fitness_values),
                    'average': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'median': np.median(fitness_values)
                },
                'genome_stats': self._calculate_genome_stats(generation_individuals),
                'timestamp': time.time()
            }
            
            return {
                'success': True,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"获取世代统计失败：{str(e)}")
            return {
                'success': False,
                'message': f'获取世代统计失败：{str(e)}'
            }
    
    def _calculate_genome_stats(self, individuals: List[Individual]) -> Dict[str, Any]:
        """计算基因组统计信息"""
        if not individuals:
            return {}
        
        # 收集所有基因位点
        all_genomes = np.array([ind.genome for ind in individuals])
        
        stats = {
            'genome_length': len(individuals[0].genome),
            'gene_stats': {
                'mean': np.mean(all_genomes, axis=0).tolist(),
                'std': np.std(all_genomes, axis=0).tolist(),
                'min': np.min(all_genomes, axis=0).tolist(),
                'max': np.max(all_genomes, axis=0).tolist()
            },
            'population_diversity': {
                'gene_variance': np.var(all_genomes, axis=0).tolist(),
                'overall_variance': np.var(all_genomes)
            }
        }
        
        return stats
    
    def filter_individuals(self, 
                          criteria: Dict[str, Any]) -> List[Individual]:
        """
        根据条件筛选个体
        
        Args:
            criteria: 筛选条件
            
        Returns:
            符合条件的个体列表
        """
        filtered_individuals = []
        
        for individual in self.all_individuals.values():
            # 检查各项条件
            include = True
            
            # 适应度范围
            if 'min_fitness' in criteria:
                if individual.fitness < criteria['min_fitness']:
                    include = False
            
            if 'max_fitness' in criteria:
                if individual.fitness > criteria['max_fitness']:
                    include = False
            
            # 世代范围
            if 'min_generation' in criteria:
                if individual.generation < criteria['min_generation']:
                    include = False
            
            if 'max_generation' in criteria:
                if individual.generation > criteria['max_generation']:
                    include = False
            
            # 特定基因位点
            if 'gene_constraints' in criteria:
                for gene_idx, (min_val, max_val) in criteria['gene_constraints'].items():
                    if individual.genome[gene_idx] < min_val or individual.genome[gene_idx] > max_val:
                        include = False
                        break
            
            if include:
                filtered_individuals.append(individual)
        
        return filtered_individuals
    
    def stop_animation(self):
        """停止动画"""
        self.is_animating = False
        logger.info("进化动画已停止")
    
    def clear_data(self):
        """清除所有数据"""
        self.root = None
        self.all_individuals.clear()
        self.generation_history.clear()
        self.current_generation = 0
        self.is_animating = False
        
        # 重置各个组件
        self.tree_renderer.clear()
        self.generation_tracker.clear()
        self.fitness_visualizer.clear()
        self.diversity_analyzer.clear()
        
        logger.info("所有数据已清除")