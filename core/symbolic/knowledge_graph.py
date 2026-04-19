#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱核心模块

本模块实现知识图谱的主类，支持动态知识抽取和图谱演化。
基于NetworkX构建，支持实体、关系、属性三种节点类型，
支持多层知识层次结构和大规模知识图谱。

主要功能：
- 知识图谱的构建和维护
- 动态知识更新和演化
- 多模态知识融合
- 知识一致性检查和冲突解决
- 支持百万节点的规模

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from collections import defaultdict, deque
import threading
import weakref


class NodeType(Enum):
    """节点类型枚举"""
    ENTITY = "entity"          # 实体节点
    RELATION = "relation"      # 关系节点
    ATTRIBUTE = "attribute"    # 属性节点


class ConflictStrategy(Enum):
    """冲突解决策略"""
    OVERWRITE = "overwrite"        # 覆盖策略
    MERGE = "merge"                # 合并策略
    IGNORE = "ignore"              # 忽略策略
    REJECT = "reject"              # 拒绝策略


@dataclass
class NodeInfo:
    """节点信息数据结构"""
    node_type: NodeType
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: datetime = None
    updated_at: datetime = None
    sources: Set[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.sources is None:
            self.sources = set()


@dataclass 
class EdgeInfo:
    """边信息数据结构"""
    relation_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: datetime = None
    updated_at: datetime = None
    sources: Set[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.sources is None:
            self.sources = set()


class KnowledgeGraph:
    """
    知识图谱主类
    
    基于NetworkX实现的动态知识图谱，支持：
    - 实体、关系、属性三种节点类型
    - 多层知识层次结构
    - 实时知识更新和演化
    - 大规模图谱处理（百万节点）
    - 多模态知识融合
    - 知识一致性检查和冲突解决
    """
    
    def __init__(self, 
                 name: str = "KnowledgeGraph",
                 max_nodes: int = 1000000,
                 enable_versioning: bool = True,
                 conflict_strategy: ConflictStrategy = ConflictStrategy.MERGE,
                 cache_size: int = 10000):
        """
        初始化知识图谱
        
        Args:
            name: 图谱名称
            max_nodes: 最大节点数量限制
            enable_versioning: 是否启用版本控制
            conflict_strategy: 冲突解决策略
            cache_size: 缓存大小
        """
        self.name = name
        self.max_nodes = max_nodes
        self.enable_versioning = enable_versioning
        self.conflict_strategy = conflict_strategy
        self.cache_size = cache_size
        
        # 初始化NetworkX图
        self.graph = nx.MultiDiGraph()  # 使用多方向图支持多重关系
        
        # 节点和边的元数据存储
        self.node_metadata = {}  # node_id -> NodeInfo
        self.edge_metadata = {}  # (source, target, key) -> EdgeInfo
        
        # 层次结构存储
        self.hierarchy_levels = defaultdict(set)  # level -> set of nodes
        self.concept_relations = defaultdict(set)  # concept -> set of related concepts
        
        # 缓存和性能优化
        self.node_cache = {}  # LRU缓存
        self.subgraph_cache = {}  # 子图缓存
        self.recent_updates = deque(maxlen=cache_size)  # 最近更新记录
        
        # 版本控制
        self.version_history = [] if enable_versioning else None
        self.current_version = 0
        
        # 统计信息
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'entity_nodes': 0,
            'relation_nodes': 0,
            'attribute_nodes': 0,
            'last_update': None,
            'memory_usage': 0
        }
        
        # 并发控制
        self._lock = threading.RLock()
        self._update_callbacks = []
        
        # 设置日志
        self.logger = logging.getLogger(f"KnowledgeGraph.{name}")
        
        self.logger.info(f"知识图谱 '{name}' 初始化完成，最大节点数: {max_nodes}")
    
    def add_node(self, 
                 node_id: str, 
                 node_type: NodeType, 
                 properties: Dict[str, Any] = None,
                 confidence: float = 1.0,
                 sources: Set[str] = None) -> bool:
        """
        添加节点到知识图谱
        
        Args:
            node_id: 节点唯一标识符
            node_type: 节点类型
            properties: 节点属性字典
            confidence: 置信度 (0-1)
            sources: 数据源集合
            
        Returns:
            bool: 是否成功添加节点
        """
        with self._lock:
            try:
                # 检查节点数量限制
                if len(self.graph.nodes) >= self.max_nodes:
                    self.logger.warning(f"节点数量已达上限 {self.max_nodes}")
                    return False
                
                # 检查节点是否已存在
                if node_id in self.graph.nodes:
                    if self.conflict_strategy == ConflictStrategy.OVERWRITE:
                        self._update_existing_node(node_id, node_type, properties, confidence, sources)
                    elif self.conflict_strategy == ConflictStrategy.IGNORE:
                        return False
                    elif self.conflict_strategy == ConflictStrategy.MERGE:
                        return self._merge_node(node_id, node_type, properties, confidence, sources)
                    # REJECT策略通过抛出异常实现
                    else:
                        raise ValueError(f"节点 {node_id} 已存在，冲突策略: {self.conflict_strategy}")
                
                # 创建新节点
                if properties is None:
                    properties = {}
                if sources is None:
                    sources = set()
                    
                node_info = NodeInfo(
                    node_type=node_type,
                    properties=properties,
                    confidence=confidence,
                    sources=sources
                )
                
                # 添加到NetworkX图
                self.graph.add_node(node_id, **properties)
                
                # 存储元数据
                self.node_metadata[node_id] = node_info
                
                # 更新层次结构
                level = self._get_node_level(node_type, properties)
                self.hierarchy_levels[level].add(node_id)
                
                # 更新统计信息
                self._update_stats()
                
                # 记录更新
                self._record_update('add_node', {'node_id': node_id, 'node_type': node_type})
                
                # 触发回调
                self._trigger_callbacks('add_node', node_id, node_info)
                
                self.logger.debug(f"成功添加节点: {node_id}, 类型: {node_type}")
                return True
                
            except Exception as e:
                self.logger.error(f"添加节点失败: {node_id}, 错误: {str(e)}")
                return False
    
    def add_edge(self,
                 source_id: str,
                 target_id: str, 
                 relation_type: str,
                 properties: Dict[str, Any] = None,
                 confidence: float = 1.0,
                 sources: Set[str] = None,
                 key: str = None) -> bool:
        """
        添加边到知识图谱
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relation_type: 关系类型
            properties: 边属性字典
            confidence: 置信度 (0-1)
            sources: 数据源集合
            key: 边的键值（用于多重边）
            
        Returns:
            bool: 是否成功添加边
        """
        with self._lock:
            try:
                # 检查节点是否存在
                if source_id not in self.graph.nodes:
                    self.logger.warning(f"源节点不存在: {source_id}")
                    return False
                if target_id not in self.graph.nodes:
                    self.logger.warning(f"目标节点不存在: {target_id}")
                    return False
                
                # 处理多重边的键值
                if key is None:
                    key = f"{source_id}->{target_id}->{relation_type}"
                
                # 检查边是否已存在
                if self.graph.has_edge(source_id, target_id, key=key):
                    if self.conflict_strategy == ConflictStrategy.OVERWRITE:
                        self._update_existing_edge(source_id, target_id, key, relation_type, properties, confidence, sources)
                    elif self.conflict_strategy == ConflictStrategy.IGNORE:
                        return False
                    elif self.conflict_strategy == ConflictStrategy.MERGE:
                        return self._merge_edge(source_id, target_id, key, relation_type, properties, confidence, sources)
                    else:
                        raise ValueError(f"边已存在，冲突策略: {self.conflict_strategy}")
                
                # 创建新边
                if properties is None:
                    properties = {}
                if sources is None:
                    sources = set()
                    
                edge_info = EdgeInfo(
                    relation_type=relation_type,
                    properties=properties,
                    confidence=confidence,
                    sources=sources
                )
                
                # 添加到NetworkX图
                self.graph.add_edge(source_id, target_id, key=key, **properties)
                
                # 存储边元数据
                edge_key = (source_id, target_id, key)
                self.edge_metadata[edge_key] = edge_info
                
                # 更新概念关系
                if relation_type in ['is_a', 'part_of', 'belongs_to']:
                    self.concept_relations[source_id].add(target_id)
                
                # 更新统计信息
                self._update_stats()
                
                # 记录更新
                self._record_update('add_edge', {
                    'source': source_id, 
                    'target': target_id, 
                    'relation': relation_type
                })
                
                # 触发回调
                self._trigger_callbacks('add_edge', edge_key, edge_info)
                
                self.logger.debug(f"成功添加边: {source_id} -> {target_id}, 关系: {relation_type}")
                return True
                
            except Exception as e:
                self.logger.error(f"添加边失败: {source_id}->{target_id}, 错误: {str(e)}")
                return False
    
    def remove_node(self, node_id: str) -> bool:
        """
        从知识图谱中删除节点
        
        Args:
            node_id: 要删除的节点ID
            
        Returns:
            bool: 是否成功删除节点
        """
        with self._lock:
            try:
                if node_id not in self.graph.nodes:
                    self.logger.warning(f"节点不存在: {node_id}")
                    return False
                
                # 收集相关的边信息
                related_edges = []
                for source, target, key in self.graph.edges(keys=True):
                    if source == node_id or target == node_id:
                        related_edges.append((source, target, key))
                
                # 删除相关边
                for source, target, key in related_edges:
                    self.graph.remove_edge(source, target, key=key)
                    edge_key = (source, target, key)
                    if edge_key in self.edge_metadata:
                        del self.edge_metadata[edge_key]
                
                # 删除节点和元数据
                self.graph.remove_node(node_id)
                if node_id in self.node_metadata:
                    del self.node_metadata[node_id]
                
                # 从层次结构中移除
                for level_nodes in self.hierarchy_levels.values():
                    level_nodes.discard(node_id)
                
                # 清理概念关系
                if node_id in self.concept_relations:
                    del self.concept_relations[node_id]
                for relations in self.concept_relations.values():
                    relations.discard(node_id)
                
                # 更新统计信息
                self._update_stats()
                
                # 记录更新
                self._record_update('remove_node', {'node_id': node_id})
                
                self.logger.info(f"成功删除节点: {node_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"删除节点失败: {node_id}, 错误: {str(e)}")
                return False
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """
        获取节点信息
        
        Args:
            node_id: 节点ID
            
        Returns:
            NodeInfo: 节点信息，如果节点不存在则返回None
        """
        return self.node_metadata.get(node_id)
    
    def get_neighbors(self, node_id: str, relation_type: str = None) -> List[str]:
        """
        获取节点的邻居节点
        
        Args:
            node_id: 节点ID
            relation_type: 关系类型过滤（可选）
            
        Returns:
            List[str]: 邻居节点ID列表
        """
        neighbors = []
        
        for source, target, key in self.graph.edges(keys=True):
            if source == node_id:
                if relation_type is None or self._get_edge_relation_type(source, target, key) == relation_type:
                    neighbors.append(target)
            elif target == node_id:
                if relation_type is None or self._get_edge_relation_type(source, target, key) == relation_type:
                    neighbors.append(source)
        
        return list(set(neighbors))  # 去重
    
    def find_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """
        查找两个节点之间的最短路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            List[str]: 路径节点列表，如果不存在路径则返回空列表
        """
        try:
            if nx.has_path(self.graph, source_id, target_id):
                return nx.shortest_path(self.graph, source_id, target_id)
            else:
                return []
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, node_ids: Set[str]) -> nx.MultiDiGraph:
        """
        获取指定节点的子图
        
        Args:
            node_ids: 节点ID集合
            
        Returns:
            nx.MultiDiGraph: 子图
        """
        return self.graph.subgraph(node_ids).copy()
    
    def merge_graph(self, other_graph: 'KnowledgeGraph', 
                   strategy: ConflictStrategy = None) -> bool:
        """
        合并另一个知识图谱
        
        Args:
            other_graph: 要合并的知识图谱
            strategy: 冲突解决策略
            
        Returns:
            bool: 是否成功合并
        """
        if strategy is None:
            strategy = self.conflict_strategy
        
        with self._lock:
            try:
                # 保存当前策略
                original_strategy = self.conflict_strategy
                self.conflict_strategy = strategy
                
                # 合并节点
                for node_id, node_info in other_graph.node_metadata.items():
                    self.add_node(
                        node_id=node_id,
                        node_type=node_info.node_type,
                        properties=node_info.properties.copy(),
                        confidence=node_info.confidence,
                        sources=node_info.sources.copy()
                    )
                
                # 合并边
                for (source, target, key), edge_info in other_graph.edge_metadata.items():
                    self.add_edge(
                        source_id=source,
                        target_id=target,
                        relation_type=edge_info.relation_type,
                        properties=edge_info.properties.copy(),
                        confidence=edge_info.confidence,
                        sources=edge_info.sources.copy(),
                        key=key
                    )
                
                # 恢复原策略
                self.conflict_strategy = original_strategy
                
                self.logger.info(f"成功合并图谱: {other_graph.name}")
                return True
                
            except Exception as e:
                self.logger.error(f"合并图谱失败: {str(e)}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识图谱统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        self._update_stats()
        return self.stats.copy()
    
    def save_to_file(self, filepath: str, format: str = 'json') -> bool:
        """
        保存知识图谱到文件
        
        Args:
            filepath: 文件路径
            format: 保存格式 ('json', 'gexf', 'graphml')
            
        Returns:
            bool: 是否成功保存
        """
        try:
            if format == 'json':
                return self._save_json(filepath)
            elif format == 'gexf':
                return self._save_gexf(filepath)
            elif format == 'graphml':
                return self._save_graphml(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
                
        except Exception as e:
            self.logger.error(f"保存图谱失败: {str(e)}")
            return False
    
    def load_from_file(self, filepath: str, format: str = 'json') -> bool:
        """
        从文件加载知识图谱
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('json', 'gexf', 'graphml')
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if format == 'json':
                return self._load_json(filepath)
            elif format == 'gexf':
                return self._load_gexf(filepath)
            elif format == 'graphml':
                return self._load_graphml(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
                
        except Exception as e:
            self.logger.error(f"加载图谱失败: {str(e)}")
            return False
    
    def export_visualization(self, output_path: str, layout: str = 'spring') -> bool:
        """
        导出可视化数据
        
        Args:
            output_path: 输出文件路径
            layout: 布局算法 ('spring', 'circular', 'random', 'shell')
            
        Returns:
            bool: 是否成功导出
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 计算布局
            if layout == 'spring':
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(self.graph)
            elif layout == 'random':
                pos = nx.random_layout(self.graph)
            elif layout == 'shell':
                pos = nx.shell_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 定义颜色映射
            color_map = {
                NodeType.ENTITY: 'lightblue',
                NodeType.RELATION: 'lightgreen', 
                NodeType.ATTRIBUTE: 'lightyellow'
            }
            
            # 绘制节点
            for node_type, color in color_map.items():
                nodes = [n for n, data in self.node_metadata.items() 
                        if data.node_type == node_type]
                if nodes:
                    nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, 
                                         node_color=color, node_size=300, alpha=0.8)
            
            # 绘制边
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True)
            
            # 绘制标签（只显示部分以避免混乱）
            labels = {}
            for i, node in enumerate(self.graph.nodes()):
                if i < 50:  # 只显示前50个节点的标签
                    labels[node] = node
            
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # 添加图例
            legend_elements = [
                patches.Patch(color='lightblue', label='实体'),
                patches.Patch(color='lightgreen', label='关系'),
                patches.Patch(color='lightyellow', label='属性')
            ]
            plt.legend(handles=legend_elements)
            
            plt.title(f"知识图谱可视化 - {self.name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"可视化导出成功: {output_path}")
            return True
            
        except ImportError:
            self.logger.error("matplotlib未安装，无法生成可视化")
            return False
        except Exception as e:
            self.logger.error(f"导出可视化失败: {str(e)}")
            return False
    
    def add_update_callback(self, callback: callable):
        """
        添加更新回调函数
        
        Args:
            callback: 回调函数，签名: callback(event_type, node_id, info)
        """
        self._update_callbacks.append(callback)
    
    def _update_existing_node(self, node_id: str, node_type: NodeType, 
                            properties: Dict[str, Any], confidence: float, sources: Set[str]):
        """更新现有节点"""
        existing_info = self.node_metadata[node_id]
        existing_info.properties.update(properties)
        existing_info.confidence = max(existing_info.confidence, confidence)
        existing_info.sources.update(sources)
        existing_info.updated_at = datetime.now()
    
    def _merge_node(self, node_id: str, node_type: NodeType, 
                   properties: Dict[str, Any], confidence: float, sources: Set[str]) -> bool:
        """合并节点"""
        if node_id in self.node_metadata:
            existing_info = self.node_metadata[node_id]
            # 检查类型兼容性
            if existing_info.node_type != node_type:
                self.logger.warning(f"节点类型冲突: {node_id}")
                return False
            
            # 合并属性
            for key, value in properties.items():
                if key in existing_info.properties:
                    # 冲突检测
                    if existing_info.properties[key] != value:
                        self.logger.warning(f"属性冲突: {node_id}.{key}")
                existing_info.properties[key] = value
            
            existing_info.sources.update(sources)
            existing_info.updated_at = datetime.now()
            return True
        else:
            return self.add_node(node_id, node_type, properties, confidence, sources)
    
    def _update_existing_edge(self, source: str, target: str, key: str,
                            relation_type: str, properties: Dict[str, Any], 
                            confidence: float, sources: Set[str]):
        """更新现有边"""
        edge_key = (source, target, key)
        if edge_key in self.edge_metadata:
            existing_info = self.edge_metadata[edge_key]
            existing_info.properties.update(properties)
            existing_info.confidence = max(existing_info.confidence, confidence)
            existing_info.sources.update(sources)
            existing_info.updated_at = datetime.now()
    
    def _merge_edge(self, source: str, target: str, key: str,
                   relation_type: str, properties: Dict[str, Any], 
                   confidence: float, sources: Set[str]) -> bool:
        """合并边"""
        edge_key = (source, target, key)
        if edge_key in self.edge_metadata:
            existing_info = self.edge_metadata[edge_key]
            if existing_info.relation_type != relation_type:
                self.logger.warning(f"关系类型冲突: {source}->{target}")
                return False
            
            for key, value in properties.items():
                if key in existing_info.properties:
                    if existing_info.properties[key] != value:
                        self.logger.warning(f"边属性冲突: {source}->{target}.{key}")
                existing_info.properties[key] = value
            
            existing_info.sources.update(sources)
            existing_info.updated_at = datetime.now()
            return True
        else:
            return self.add_edge(source, target, relation_type, properties, confidence, sources, key)
    
    def _get_node_level(self, node_type: NodeType, properties: Dict[str, Any]) -> int:
        """获取节点的层次级别"""
        if node_type == NodeType.ENTITY:
            return properties.get('level', 0)
        elif node_type == NodeType.RELATION:
            return 1
        else:  # ATTRIBUTE
            return 2
    
    def _get_edge_relation_type(self, source: str, target: str, key: str) -> str:
        """获取边的关系类型"""
        edge_key = (source, target, key)
        if edge_key in self.edge_metadata:
            return self.edge_metadata[edge_key].relation_type
        return "unknown"
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats['total_nodes'] = len(self.graph.nodes)
        self.stats['total_edges'] = len(self.graph.edges(keys=True))
        
        # 按类型统计节点
        entity_count = sum(1 for info in self.node_metadata.values() 
                          if info.node_type == NodeType.ENTITY)
        relation_count = sum(1 for info in self.node_metadata.values() 
                           if info.node_type == NodeType.RELATION)
        attribute_count = sum(1 for info in self.node_metadata.values() 
                            if info.node_type == NodeType.ATTRIBUTE)
        
        self.stats['entity_nodes'] = entity_count
        self.stats['relation_nodes'] = relation_count
        self.stats['attribute_nodes'] = attribute_count
        self.stats['last_update'] = datetime.now()
        
        # 估算内存使用量
        import sys
        self.stats['memory_usage'] = sys.getsizeof(self.graph) + \
                                   sys.getsizeof(self.node_metadata) + \
                                   sys.getsizeof(self.edge_metadata)
    
    def _record_update(self, event_type: str, data: Dict[str, Any]):
        """记录更新事件"""
        if self.enable_versioning:
            version_entry = {
                'version': self.current_version,
                'timestamp': datetime.now(),
                'event_type': event_type,
                'data': data
            }
            self.version_history.append(version_entry)
            self.current_version += 1
            
            # 限制历史记录大小
            if len(self.version_history) > 1000:
                self.version_history = self.version_history[-500:]
        
        # 记录到最近更新队列
        self.recent_updates.append({
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': data
        })
    
    def _trigger_callbacks(self, event_type: str, node_id: str, info: Any):
        """触发更新回调"""
        for callback in self._update_callbacks:
            try:
                callback(event_type, node_id, info)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {str(e)}")
    
    def _save_json(self, filepath: str) -> bool:
        """保存为JSON格式"""
        data = {
            'name': self.name,
            'metadata': {
                'max_nodes': self.max_nodes,
                'enable_versioning': self.enable_versioning,
                'conflict_strategy': self.conflict_strategy.value
            },
            'nodes': {},
            'edges': {}
        }
        
        # 序列化节点
        for node_id, node_info in self.node_metadata.items():
            data['nodes'][node_id] = {
                'node_type': node_info.node_type.value,
                'properties': node_info.properties,
                'confidence': node_info.confidence,
                'created_at': node_info.created_at.isoformat(),
                'updated_at': node_info.updated_at.isoformat(),
                'sources': list(node_info.sources)
            }
        
        # 序列化边
        for (source, target, key), edge_info in self.edge_metadata.items():
            edge_key = f"{source}->{target}->{key}"
            data['edges'][edge_key] = {
                'source': source,
                'target': target,
                'key': key,
                'relation_type': edge_info.relation_type,
                'properties': edge_info.properties,
                'confidence': edge_info.confidence,
                'created_at': edge_info.created_at.isoformat(),
                'updated_at': edge_info.updated_at.isoformat(),
                'sources': list(edge_info.sources)
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    
    def _load_json(self, filepath: str) -> bool:
        """从JSON格式加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建图谱
        self.name = data.get('name', self.name)
        metadata = data.get('metadata', {})
        self.max_nodes = metadata.get('max_nodes', self.max_nodes)
        self.enable_versioning = metadata.get('enable_versioning', self.enable_versioning)
        
        strategy_value = metadata.get('conflict_strategy', self.conflict_strategy.value)
        self.conflict_strategy = ConflictStrategy(strategy_value)
        
        # 重建节点
        for node_id, node_data in data.get('nodes', {}).items():
            node_info = NodeInfo(
                node_type=NodeType(node_data['node_type']),
                properties=node_data['properties'],
                confidence=node_data['confidence'],
                sources=set(node_data['sources'])
            )
            node_info.created_at = datetime.fromisoformat(node_data['created_at'])
            node_info.updated_at = datetime.fromisoformat(node_data['updated_at'])
            
            self.node_metadata[node_id] = node_info
            self.graph.add_node(node_id, **node_data['properties'])
        
        # 重建边
        for edge_key, edge_data in data.get('edges', {}).items():
            edge_info = EdgeInfo(
                relation_type=edge_data['relation_type'],
                properties=edge_data['properties'],
                confidence=edge_data['confidence'],
                sources=set(edge_data['sources'])
            )
            edge_info.created_at = datetime.fromisoformat(edge_data['created_at'])
            edge_info.updated_at = datetime.fromisoformat(edge_data['updated_at'])
            
            self.edge_metadata[edge_key] = edge_info
            self.graph.add_edge(
                edge_data['source'], 
                edge_data['target'], 
                key=edge_data['key'],
                **edge_data['properties']
            )
        
        self._update_stats()
        return True
    
    def _save_gexf(self, filepath: str) -> bool:
        """保存为GEXF格式"""
        # 转换为简单的节点边格式
        nodes_data = []
        for node_id, node_info in self.node_metadata.items():
            nodes_data.append({
                'id': node_id,
                'label': node_id,
                'type': node_info.node_type.value,
                'attributes': node_info.properties
            })
        
        edges_data = []
        for (source, target, key), edge_info in self.edge_metadata.items():
            edges_data.append({
                'source': source,
                'target': target,
                'label': edge_info.relation_type,
                'weight': edge_info.confidence
            })
        
        # 简化的GEXF格式
        gexf_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <meta lastmodifieddate="{datetime.now().isoformat()}">
    <creator>KnowledgeGraph System</creator>
    <description>{self.name}</description>
  </meta>
  <graph mode="static" defaultedgetype="directed">
    <nodes>
'''
        
        for node in nodes_data:
            gexf_content += f'      <node id="{node["id"]}" label="{node["label"]}"/>\n'
        
        gexf_content += '    </nodes>\n    <edges>\n'
        
        for edge in edges_data:
            gexf_content += f'      <edge source="{edge["source"]}" target="{edge["target"]}" label="{edge["label"]}"/>\n'
        
        gexf_content += '    </edges>\n  </graph>\n</gexf>'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(gexf_content)
        
        return True
    
    def _load_gexf(self, filepath: str) -> bool:
        """从GEXF格式加载（简化实现）"""
        # 这里实现一个简化版本的GEXF解析器
        # 在实际应用中可能需要更完整的XML解析
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # 解析节点
            for node_elem in root.findall('.//node'):
                node_id = node_elem.get('id')
                label = node_elem.get('label', node_id)
                self.add_node(node_id, NodeType.ENTITY, {'label': label})
            
            # 解析边
            for edge_elem in root.findall('.//edge'):
                source = edge_elem.get('source')
                target = edge_elem.get('target')
                label = edge_elem.get('label', 'related')
                self.add_edge(source, target, label)
            
            return True
            
        except Exception as e:
            self.logger.error(f"GEXF解析失败: {str(e)}")
            return False
    
    def _save_graphml(self, filepath: str) -> bool:
        """保存为GraphML格式"""
        try:
            nx.write_graphml(self.graph, filepath)
            return True
        except Exception as e:
            self.logger.error(f"GraphML保存失败: {str(e)}")
            return False
    
    def _load_graphml(self, filepath: str) -> bool:
        """从GraphML格式加载"""
        try:
            loaded_graph = nx.read_graphml(filepath)
            self.graph = loaded_graph
            
            # 重建元数据（简化实现）
            for node in self.graph.nodes():
                self.node_metadata[node] = NodeInfo(
                    node_type=NodeType.ENTITY,
                    properties=self.graph.nodes[node]
                )
            
            for source, target, key in self.graph.edges(keys=True):
                edge_key = (source, target, key)
                self.edge_metadata[edge_key] = EdgeInfo(
                    relation_type="related",
                    properties=self.graph.edges[source, target, key]
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"GraphML加载失败: {str(e)}")
            return False