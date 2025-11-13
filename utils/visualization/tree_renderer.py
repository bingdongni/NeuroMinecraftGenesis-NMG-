#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化树渲染器
Evolution Tree Renderer

该模块负责将进化树数据渲染为可视化格式，支持多种布局算法、
颜色映射和交互功能。主要用于生成D3.js可用的数据格式。

功能特性：
- 多种树布局算法（径向、层次、圆形等）
- 动态颜色映射和大小编码
- 节点交互和筛选功能
- 动画过渡效果
- 多层次细节展示

Author: 进化树可视化系统
Date: 2025-11-13
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict

# 导入进化树主模块的数据结构
try:
    from .evolution_tree import Individual, EvolutionNode
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from utils.visualization.evolution_tree import Individual, EvolutionNode


@dataclass
class TreeLayout:
    """
    树布局配置
    Tree Layout Configuration
    """
    width: float = 1200.0
    height: float = 800.0
    margin: Dict[str, float] = None
    layout_type: str = 'radial'  # 'radial', 'hierarchical', 'circular'
    node_spacing: float = 100.0
    level_spacing: float = 80.0
    
    def __post_init__(self):
        if self.margin is None:
            self.margin = {'top': 50, 'right': 50, 'bottom': 50, 'left': 50}


@dataclass
class NodeStyle:
    """
    节点样式配置
    Node Style Configuration
    """
    radius: float = 8.0
    fill_color: str = '#4CAF50'
    stroke_color: str = '#2E7D32'
    stroke_width: float = 1.5
    opacity: float = 0.8
    label_font_size: float = 12.0
    label_color: str = '#000000'
    show_labels: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'radius': self.radius,
            'fillColor': self.fill_color,
            'strokeColor': self.stroke_color,
            'strokeWidth': self.stroke_width,
            'opacity': self.opacity,
            'labelFontSize': self.label_font_size,
            'labelColor': self.label_color,
            'showLabels': self.show_labels
        }


@dataclass
class LinkStyle:
    """
    连接线样式配置
    Link Style Configuration
    """
    stroke_color: str = '#666666'
    stroke_width: float = 2.0
    stroke_opacity: float = 0.6
    stroke_dasharray: Optional[str] = None
    hover_stroke_width: float = 3.0
    hover_stroke_color: str = '#FF5722'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'strokeColor': self.stroke_color,
            'strokeWidth': self.stroke_width,
            'strokeOpacity': self.stroke_opacity,
            'strokeDasharray': self.stroke_dasharray,
            'hoverStrokeWidth': self.hover_stroke_width,
            'hoverStrokeColor': self.hover_stroke_color
        }


class TreeRenderer:
    """
    进化树渲染器
    Evolution Tree Renderer
    
    负责将进化树数据结构转换为前端可视化的数据格式，
    支持多种布局算法和样式配置。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化树渲染器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._default_config()
        
        # 布局和样式配置
        self.layout = TreeLayout(**self.config.get('layout', {}))
        self.node_style = NodeStyle(**self.config.get('node_style', {}))
        self.link_style = LinkStyle(**self.config.get('link_style', {}))
        
        # 颜色映射器
        self.color_schemes = {
            'viridis': self._viridis_color_map,
            'plasma': self._plasma_color_map,
            'heat': self._heat_color_map,
            'genetic': self._genetic_color_map
        }
        self.color_scheme = self.config.get('color_scheme', 'viridis')
        
        # 状态变量
        self._current_scale = None
        self._layout_cache = {}
        
        print("进化树渲染器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'layout': {
                'width': 1200,
                'height': 800,
                'layout_type': 'radial',
                'node_spacing': 100,
                'level_spacing': 80
            },
            'node_style': {
                'radius': 8,
                'fill_color': '#4CAF50',
                'stroke_color': '#2E7D32',
                'show_labels': True
            },
            'link_style': {
                'stroke_color': '#666666',
                'stroke_width': 2,
                'stroke_opacity': 0.6
            },
            'color_scheme': 'viridis',
            'animation': {
                'duration': 750,
                'ease': 'ease-in-out'
            }
        }
    
    def render_tree(self, 
                   individuals: Dict[int, Individual],
                   start_generation: int = 0,
                   end_generation: Optional[int] = None,
                   filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        渲染进化树
        
        Args:
            individuals: 个体数据字典
            start_generation: 开始世代
            end_generation: 结束世代
            filter_criteria: 筛选条件
            
        Returns:
            渲染后的树数据
        """
        try:
            print(f"开始渲染进化树：世代 {start_generation} 到 {end_generation}")
            
            # 筛选个体数据
            filtered_individuals = self._filter_individuals(
                individuals, start_generation, end_generation, filter_criteria
            )
            
            if not filtered_individuals:
                return {
                    'nodes': [],
                    'links': [],
                    'layout': self.layout.__dict__,
                    'styles': {
                        'node': self.node_style.to_dict(),
                        'link': self.link_style.to_dict()
                    },
                    'metadata': {
                        'filtered_count': 0,
                        'total_count': len(individuals)
                    }
                }
            
            # 构建树结构
            tree_structure = self._build_tree_structure(filtered_individuals)
            
            # 计算布局
            layout_result = self._calculate_layout(tree_structure)
            
            # 应用颜色映射
            colored_result = self._apply_color_mapping(layout_result)
            
            # 添加交互数据和元数据
            final_result = self._add_interactive_data(colored_result)
            
            print(f"进化树渲染完成：{len(final_result['nodes'])} 个节点")
            return final_result
            
        except Exception as e:
            print(f"进化树渲染失败：{str(e)}")
            return {
                'nodes': [],
                'links': [],
                'error': str(e)
            }
    
    def _filter_individuals(self,
                           individuals: Dict[int, Individual],
                           start_generation: int,
                           end_generation: Optional[int],
                           filter_criteria: Optional[Dict[str, Any]]) -> List[Individual]:
        """筛选个体数据"""
        filtered = []
        
        for individual in individuals.values():
            # 世代筛选
            if individual.generation < start_generation:
                continue
            
            if end_generation is not None and individual.generation > end_generation:
                continue
            
            # 其他筛选条件
            if filter_criteria:
                # 适应度范围筛选
                if 'min_fitness' in filter_criteria:
                    if individual.fitness < filter_criteria['min_fitness']:
                        continue
                
                if 'max_fitness' in filter_criteria:
                    if individual.fitness > filter_criteria['max_fitness']:
                        continue
                
                # 基因组约束筛选
                if 'gene_constraints' in filter_criteria:
                    constraints = filter_criteria['gene_constraints']
                    for gene_idx, (min_val, max_val) in constraints.items():
                        if individual.genome[gene_idx] < min_val or individual.genome[gene_idx] > max_val:
                            continue
            
            filtered.append(individual)
        
        return filtered
    
    def _build_tree_structure(self, individuals: List[Individual]) -> EvolutionNode:
        """构建树结构"""
        if not individuals:
            return None
        
        # 按世代排序
        sorted_individuals = sorted(individuals, key=lambda x: x.generation)
        
        # 创建节点映射
        node_map = {}
        
        # 创建根节点
        root_individual = sorted_individuals[0]
        root_node = EvolutionNode(root_individual)
        node_map[root_individual.id] = root_node
        
        # 构建子树
        for individual in sorted_individuals[1:]:
            node = EvolutionNode(individual)
            node_map[individual.id] = node
            
            # 连接父子关系
            if individual.parent_id is not None:
                parent_node = node_map.get(individual.parent_id)
                if parent_node:
                    parent_node.add_child(node)
        
        return root_node
    
    def _calculate_layout(self, root: EvolutionNode) -> Dict[str, Any]:
        """计算树布局"""
        if root is None:
            return {'nodes': [], 'links': []}
        
        # 根据布局类型选择算法
        if self.layout.layout_type == 'radial':
            return self._radial_layout(root)
        elif self.layout.layout_type == 'hierarchical':
            return self._hierarchical_layout(root)
        elif self.layout.layout_type == 'circular':
            return self._circular_layout(root)
        else:
            return self._radial_layout(root)  # 默认使用径向布局
    
    def _radial_layout(self, root: EvolutionNode) -> Dict[str, Any]:
        """径向布局算法"""
        nodes = []
        links = []
        
        # 计算树的深度和宽度
        max_depth = self._calculate_max_depth(root)
        
        # 为每个节点分配角度和半径
        self._assign_radial_coordinates(root, 0, 0, max_depth)
        
        # 递归提取节点和连接
        self._extract_nodes_and_links(root, nodes, links)
        
        return {
            'nodes': nodes,
            'links': links,
            'layout_type': 'radial'
        }
    
    def _hierarchical_layout(self, root: EvolutionNode) -> Dict[str, Any]:
        """层次布局算法"""
        nodes = []
        links = []
        
        # 使用力导向布局的简化版本
        self._assign_hierarchical_coordinates(root, 0, 0)
        self._extract_nodes_and_links(root, nodes, links)
        
        return {
            'nodes': nodes,
            'links': links,
            'layout_type': 'hierarchical'
        }
    
    def _circular_layout(self, root: EvolutionNode) -> Dict[str, Any]:
        """圆形布局算法"""
        nodes = []
        links = []
        
        # 将树映射到圆形
        self._assign_circular_coordinates(root, 0, 0, 2 * math.pi)
        self._extract_nodes_and_links(root, nodes, links)
        
        return {
            'nodes': nodes,
            'links': links,
            'layout_type': 'circular'
        }
    
    def _assign_radial_coordinates(self, 
                                 node: EvolutionNode, 
                                 angle: float, 
                                 radius: float, 
                                 max_depth: int):
        """为径向布局分配坐标"""
        if node is None:
            return
        
        # 计算节点位置
        node_angle = angle
        node_radius = self.layout.margin['top'] + radius * self.layout.level_spacing
        
        # 保存径向坐标
        if not hasattr(node, 'radial_coords'):
            node.radial_coords = {}
        
        node.radial_coords['angle'] = node_angle
        node.radial_coords['radius'] = node_radius
        node.radial_coords['x'] = node_radius * math.cos(node_angle)
        node.radial_coords['y'] = node_radius * math.sin(node_angle)
        
        # 递归处理子节点
        if node.children:
            children_count = len(node.children)
            angle_step = 2 * math.pi / children_count
            child_angle_start = angle - math.pi / 2
            
            for i, child in enumerate(node.children):
                child_angle = child_angle_start + i * angle_step
                self._assign_radial_coordinates(
                    child, child_angle, radius + 1, max_depth
                )
    
    def _assign_hierarchical_coordinates(self, 
                                       node: EvolutionNode, 
                                       x: float, 
                                       y: float):
        """为层次布局分配坐标"""
        if node is None:
            return
        
        # 简化的层次布局 - 使用广度优先搜索
        queue = [(node, x, y)]
        
        while queue:
            current_node, current_x, current_y = queue.pop(0)
            
            # 保存层次坐标
            if not hasattr(current_node, 'hierarchical_coords'):
                current_node.hierarchical_coords = {}
            
            current_node.hierarchical_coords['x'] = current_x
            current_node.hierarchical_coords['y'] = current_y
            
            # 处理子节点
            child_y = current_y + self.layout.level_spacing
            for i, child in enumerate(current_node.children):
                child_x = current_x + (i - len(current_node.children)/2) * self.layout.node_spacing
                queue.append((child, child_x, child_y))
    
    def _assign_circular_coordinates(self, 
                                   node: EvolutionNode, 
                                   angle_start: float, 
                                   angle_end: float, 
                                   total_angle: float):
        """为圆形布局分配坐标"""
        if node is None:
            return
        
        # 节点角度
        node_angle = (angle_start + angle_end) / 2
        node_radius = self.layout.width / 4  # 使用固定半径
        
        # 保存圆形坐标
        if not hasattr(node, 'circular_coords'):
            node.circular_coords = {}
        
        node.circular_coords['angle'] = node_angle
        node.circular_coords['radius'] = node_radius
        node.circular_coords['x'] = node_radius * math.cos(node_angle)
        node.circular_coords['y'] = node_radius * math.sin(node_angle)
        
        # 递归处理子节点
        if node.children:
            angle_per_child = total_angle / len(node.children)
            for i, child in enumerate(node.children):
                child_angle_start = angle_start + i * angle_per_child
                child_angle_end = child_angle_start + angle_per_child
                self._assign_circular_coordinates(
                    child, child_angle_start, child_angle_end, angle_per_child
                )
    
    def _extract_nodes_and_links(self, 
                               node: EvolutionNode, 
                               nodes: List[Dict[str, Any]], 
                               links: List[Dict[str, Any]]):
        """提取节点和连接数据"""
        if node is None:
            return
        
        # 获取节点坐标（根据布局类型）
        if self.layout.layout_type == 'radial' and hasattr(node, 'radial_coords'):
            coords = node.radial_coords
        elif self.layout.layout_type == 'hierarchical' and hasattr(node, 'hierarchical_coords'):
            coords = node.hierarchical_coords
        elif self.layout.layout_type == 'circular' and hasattr(node, 'circular_coords'):
            coords = node.circular_coords
        else:
            # 默认坐标
            coords = {'x': 0, 'y': 0}
        
        # 创建节点数据
        node_data = {
            'id': node.individual.id,
            'generation': node.individual.generation,
            'fitness': node.individual.fitness,
            'x': coords['x'],
            'y': coords['y'],
            'depth': node.depth,
            'branch_length': node.branch_length,
            'parent_id': node.individual.parent_id,
            'timestamp': node.individual.timestamp,
            'mutations': node.individual.mutations
        }
        
        nodes.append(node_data)
        
        # 创建连接数据
        for child in node.children:
            link_data = {
                'source': node.individual.id,
                'target': child.individual.id,
                'fitness_change': child.individual.fitness - node.individual.fitness,
                'generation_gap': child.individual.generation - node.individual.generation
            }
            links.append(link_data)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_nodes_and_links(child, nodes, links)
    
    def _calculate_max_depth(self, node: EvolutionNode) -> int:
        """计算树的最大深度"""
        if node is None:
            return 0
        
        if not node.children:
            return 0
        
        max_child_depth = max(self._calculate_max_depth(child) for child in node.children)
        return max_child_depth + 1
    
    def _apply_color_mapping(self, layout_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用颜色映射"""
        nodes = layout_result['nodes']
        
        if not nodes:
            return layout_result
        
        # 选择颜色映射函数
        color_func = self.color_schemes.get(self.color_scheme, self._viridis_color_map)
        
        # 获取适应度值范围
        fitness_values = [node['fitness'] for node in nodes]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        
        # 为每个节点分配颜色
        for node in nodes:
            # 标准化适应度值
            normalized_fitness = (node['fitness'] - min_fitness) / (max_fitness - min_fitness)
            node['color'] = color_func(normalized_fitness)
            
            # 添加渐变强度（基于深度）
            depth_factor = 1.0 - (node['depth'] / 10.0)  # 假设最大深度为10
            node['opacity'] = max(0.3, min(1.0, depth_factor))
        
        layout_result['nodes'] = nodes
        return layout_result
    
    def _viridis_color_map(self, value: float) -> str:
        """Viridis 颜色映射"""
        # 简化的Viridis色彩方案
        if value <= 0.25:
            return '#440154'  # 深紫色
        elif value <= 0.5:
            return '#31688e'  # 蓝色
        elif value <= 0.75:
            return '#35b779'  # 绿色
        else:
            return '#fde725'  # 黄色
    
    def _plasma_color_map(self, value: float) -> str:
        """Plasma 颜色映射"""
        if value <= 0.25:
            return '#0d0887'
        elif value <= 0.5:
            return '#7e03a8'
        elif value <= 0.75:
            return '#f0f921'
        else:
            return '#ffffff'
    
    def _heat_color_map(self, value: float) -> str:
        """热力图颜色映射"""
        if value <= 0.2:
            return '#0000ff'
        elif value <= 0.4:
            return '#00ffff'
        elif value <= 0.6:
            return '#00ff00'
        elif value <= 0.8:
            return '#ffff00'
        else:
            return '#ff0000'
    
    def _genetic_color_map(self, value: float) -> str:
        """遗传多样性颜色映射"""
        # 基于基因型的颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_index = int(value * len(colors)) % len(colors)
        return colors[color_index]
    
    def _add_interactive_data(self, layout_result: Dict[str, Any]) -> Dict[str, Any]:
        """添加交互数据"""
        nodes = layout_result['nodes']
        links = layout_result['links']
        
        # 为节点添加交互属性
        for node in nodes:
            # 添加悬停文本
            node['hover_text'] = f"ID: {node['id']}<br>适应度: {node['fitness']:.4f}<br>世代: {node['generation']}<br>深度: {node['depth']}"
            
            # 添加点击事件数据
            node['click_data'] = {
                'id': node['id'],
                'generation': node['generation'],
                'fitness': node['fitness'],
                'parent_id': node['parent_id']
            }
            
            # 添加大小属性（基于适应度）
            fitness_range = max(n['fitness'] for n in nodes) - min(n['fitness'] for n in nodes)
            if fitness_range > 0:
                size_factor = (node['fitness'] - min(n['fitness'] for n in nodes)) / fitness_range
                node['radius'] = self.node_style.radius + size_factor * 5  # 在基础大小上增加最多5像素
            else:
                node['radius'] = self.node_style.radius
        
        # 为连接添加交互属性
        for link in links:
            # 添加悬停文本
            if 'fitness_change' in link:
                link['hover_text'] = f"适应度变化: {link['fitness_change']:+.4f}<br>世代间隔: {link['generation_gap']}"
            
            # 添加样式属性
            if link['fitness_change'] > 0:
                link['stroke_color'] = '#4CAF50'  # 绿色表示适应度提升
            else:
                link['stroke_color'] = '#F44336'  # 红色表示适应度下降
            
            link['stroke_width'] = abs(link['fitness_change']) * 10 + 1  # 基于适应度变化调整线宽
        
        layout_result['nodes'] = nodes
        layout_result['links'] = links
        
        # 添加样式和元数据
        layout_result['styles'] = {
            'node': self.node_style.to_dict(),
            'link': self.link_style.to_dict()
        }
        
        layout_result['metadata'] = {
            'total_nodes': len(nodes),
            'total_links': len(links),
            'layout_type': self.layout.layout_type,
            'color_scheme': self.color_scheme,
            'render_timestamp': time.time()
        }
        
        return layout_result
    
    def set_color_scheme(self, scheme_name: str):
        """设置颜色方案"""
        if scheme_name in self.color_schemes:
            self.color_scheme = scheme_name
            print(f"颜色方案已设置为: {scheme_name}")
        else:
            print(f"未知的颜色方案: {scheme_name}, 使用默认方案")
    
    def set_layout_type(self, layout_type: str):
        """设置布局类型"""
        valid_types = ['radial', 'hierarchical', 'circular']
        if layout_type in valid_types:
            self.layout.layout_type = layout_type
            print(f"布局类型已设置为: {layout_type}")
        else:
            print(f"无效的布局类型: {layout_type}, 使用默认径向布局")
    
    def clear(self):
        """清除缓存和状态"""
        self._layout_cache.clear()
        self._current_scale = None
        print("渲染器状态已清除")
    
    def get_layout_config(self) -> Dict[str, Any]:
        """获取当前布局配置"""
        return {
            'layout': self.layout.__dict__,
            'node_style': self.node_style.to_dict(),
            'link_style': self.link_style.to_dict(),
            'color_scheme': self.color_scheme
        }
    
    def update_layout_config(self, config_updates: Dict[str, Any]):
        """更新布局配置"""
        # 更新布局参数
        if 'layout' in config_updates:
            for key, value in config_updates['layout'].items():
                if hasattr(self.layout, key):
                    setattr(self.layout, key, value)
        
        # 更新节点样式
        if 'node_style' in config_updates:
            for key, value in config_updates['node_style'].items():
                if hasattr(self.node_style, key):
                    setattr(self.node_style, key, value)
        
        # 更新连接样式
        if 'link_style' in config_updates:
            for key, value in config_updates['link_style'].items():
                if hasattr(self.link_style, key):
                    setattr(self.link_style, key, value)
        
        # 更新其他参数
        if 'color_scheme' in config_updates:
            self.set_color_scheme(config_updates['color_scheme'])
        
        print("布局配置已更新")


# 导入时间模块（用于时间戳）
import time