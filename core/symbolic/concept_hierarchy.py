#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念层次结构模块

本模块实现知识图谱的概念层次结构构建和管理。
支持多层次的概念分类、自动层次归纳、概念演化等功能。

主要功能：
- 概念层次结构构建
- 自动概念分类
- 层次路径查找
- 概念相似度计算
- 动态层次演化
- 多维度层次分析

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class HierarchyType(Enum):
    """层次类型枚举"""
    TAXONOMY = "taxonomy"           # 分类层次
    PARTONOMY = "partonomy"         # 部分-整体层次
    MERONOMY = "meronomy"           # 组成层次
    HYPERNYMY = "hypernymy"         # 上位词层次
    ASSOCIATION = "association"     # 关联层次
    CUSTOM = "custom"               # 自定义层次


class ClassificationMethod(Enum):
    """分类方法枚举"""
    STATISTICAL = "statistical"     # 统计分类
    LINGUISTIC = "linguistic"       # 语言学分类
    SEMANTIC = "semantic"           # 语义分类
    FREQUENCY = "frequency"         # 频率分类
    CONTEXT = "context"             # 上下文分类


@dataclass
class ConceptNode:
    """概念节点"""
    concept_id: str
    name: str
    description: str = ""
    level: int = 0
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sources: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    frequency: int = 0
    popularity: float = 0.0
    
    def __post_init__(self):
        if not self.description:
            self.description = self.name


@dataclass
class HierarchyRelation:
    """层次关系"""
    source_concept: str
    target_concept: str
    relation_type: HierarchyType
    strength: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.evidence:
            self.evidence = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class ClassificationResult:
    """分类结果"""
    concept_id: str
    assigned_category: str
    confidence: float
    method: ClassificationMethod
    features: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


class ConceptHierarchy:
    """
    概念层次结构管理器
    
    实现概念层次结构的构建和管理功能：
    - 多层次概念分类
    - 自动层次归纳
    - 概念相似度计算
    - 层次路径分析
    - 动态结构演化
    
    特性：
    - 支持大规模概念集合
    - 多种分类算法
    - 层次一致性检查
    - 增量式结构更新
    - 可视化支持
    """
    
    def __init__(self, 
                 max_levels: int = 10,
                 min_cluster_size: int = 3,
                 similarity_threshold: float = 0.7,
                 confidence_threshold: float = 0.5,
                 enable_fuzzy: bool = True):
        """
        初始化概念层次结构管理器
        
        Args:
            max_levels: 最大层次深度
            min_cluster_size: 最小聚类大小
            similarity_threshold: 相似度阈值
            confidence_threshold: 置信度阈值
            enable_fuzzy: 是否启用模糊分类
        """
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_fuzzy = enable_fuzzy
        
        # 概念存储
        self.concepts = {}  # concept_id -> ConceptNode
        self.hierarchy_relations = []  # 层次关系列表
        self.hierarchy_graph = nx.DiGraph()  # 层次图
        
        # 层次结构
        self.levels = defaultdict(set)  # level -> set of concept_ids
        self.category_index = defaultdict(set)  # category -> set of concept_ids
        self.taxonomy = defaultdict(dict)  # 分类体系
        
        # 分类器
        self.classifiers = {}
        self.classification_rules = []
        
        # 统计信息
        self.hierarchy_stats = {
            'total_concepts': 0,
            'total_relations': 0,
            'max_depth': 0,
            'avg_branching_factor': 0.0,
            'classification_accuracy': 0.0,
            'last_update': None
        }
        
        self.logger = logging.getLogger("ConceptHierarchy")
        
        # 初始化默认分类器
        self._initialize_default_classifiers()
        
        self.logger.info(f"概念层次结构初始化完成，最大层次: {max_levels}")
    
    def add_concept(self, 
                   concept_id: str, 
                   name: str, 
                   description: str = "",
                   level: int = 0,
                   properties: Dict[str, Any] = None,
                   confidence: float = 1.0,
                   sources: Set[str] = None) -> bool:
        """
        添加概念到层次结构
        
        Args:
            concept_id: 概念唯一标识符
            name: 概念名称
            description: 概念描述
            level: 概念层次级别
            properties: 概念属性
            confidence: 置信度
            sources: 数据源集合
            
        Returns:
            bool: 是否成功添加概念
        """
        try:
            if concept_id in self.concepts:
                self.logger.warning(f"概念已存在: {concept_id}")
                return False
            
            if properties is None:
                properties = {}
            if sources is None:
                sources = set()
            
            concept = ConceptNode(
                concept_id=concept_id,
                name=name,
                description=description,
                level=level,
                properties=properties,
                confidence=confidence,
                sources=sources
            )
            
            # 添加到存储
            self.concepts[concept_id] = concept
            self.levels[level].add(concept_id)
            
            # 更新层次图
            self.hierarchy_graph.add_node(concept_id, **properties)
            
            # 更新统计
            self._update_hierarchy_stats()
            
            self.logger.debug(f"成功添加概念: {concept_id}, 级别: {level}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加概念失败: {concept_id}, 错误: {str(e)}")
            return False
    
    def add_hierarchy_relation(self, 
                             source_concept: str,
                             target_concept: str,
                             relation_type: HierarchyType,
                             strength: float = 1.0,
                             confidence: float = 1.0,
                             evidence: List[str] = None,
                             metadata: Dict[str, Any] = None) -> bool:
        """
        添加层次关系
        
        Args:
            source_concept: 源概念
            target_concept: 目标概念
            relation_type: 关系类型
            strength: 关系强度
            confidence: 置信度
            evidence: 证据列表
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加关系
        """
        try:
            # 检查概念是否存在
            if source_concept not in self.concepts:
                self.logger.warning(f"源概念不存在: {source_concept}")
                return False
            
            if target_concept not in self.concepts:
                self.logger.warning(f"目标概念不存在: {target_concept}")
                return False
            
            if evidence is None:
                evidence = []
            if metadata is None:
                metadata = {}
            
            relation = HierarchyRelation(
                source_concept=source_concept,
                target_concept=target_concept,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                metadata=metadata
            )
            
            # 添加到存储
            self.hierarchy_relations.append(relation)
            
            # 更新概念节点
            source_node = self.concepts[source_concept]
            target_node = self.concepts[target_concept]
            
            source_node.child_concepts.add(target_concept)
            target_node.parent_concepts.add(source_concept)
            
            # 更新层次图
            self.hierarchy_graph.add_edge(source_concept, target_concept, 
                                        relation_type=relation_type.value,
                                        strength=strength,
                                        confidence=confidence)
            
            # 更新统计
            self._update_hierarchy_stats()
            
            self.logger.debug(f"成功添加层次关系: {source_concept} -> {target_concept}, 类型: {relation_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加层次关系失败: {str(e)}")
            return False
    
    def build_hierarchy(self, 
                       entities: List[Tuple[str, str, Dict[str, Any]]],
                       method: ClassificationMethod = ClassificationMethod.STATISTICAL,
                       **kwargs) -> bool:
        """
        构建概念层次结构
        
        Args:
            entities: 实体列表，格式: [(实体名, 实体类型, 属性字典)]
            method: 分类方法
            **kwargs: 额外参数
            
        Returns:
            bool: 是否成功构建层次结构
        """
        try:
            self.logger.info(f"开始构建概念层次结构，方法: {method}")
            
            # 第一步：添加基础概念
            self._add_base_concepts(entities)
            
            # 第二步：执行分类
            if method == ClassificationMethod.STATISTICAL:
                self._build_statistical_hierarchy(entities, **kwargs)
            elif method == ClassificationMethod.LINGUISTIC:
                self._build_linguistic_hierarchy(entities, **kwargs)
            elif method == ClassificationMethod.SEMANTIC:
                self._build_semantic_hierarchy(entities, **kwargs)
            elif method == ClassificationMethod.FREQUENCY:
                self._build_frequency_hierarchy(entities, **kwargs)
            elif method == ClassificationMethod.CONTEXT:
                self._build_context_hierarchy(entities, **kwargs)
            else:
                raise ValueError(f"不支持的分类方法: {method}")
            
            # 第三步：优化层次结构
            self._optimize_hierarchy()
            
            # 第四步：验证层次一致性
            self._validate_hierarchy_consistency()
            
            self.logger.info("概念层次结构构建完成")
            return True
            
        except Exception as e:
            self.logger.error(f"构建概念层次结构失败: {str(e)}")
            return False
    
    def find_concept_path(self, 
                         source_concept: str, 
                         target_concept: str,
                         relation_type: HierarchyType = None) -> List[str]:
        """
        查找概念间的层次路径
        
        Args:
            source_concept: 源概念
            target_concept: 目标概念
            relation_type: 关系类型过滤（可选）
            
        Returns:
            List[str]: 路径概念列表
        """
        try:
            if relation_type:
                # 按关系类型过滤
                filtered_edges = [(u, v) for u, v, d in self.hierarchy_graph.edges(data=True)
                                if d.get('relation_type') == relation_type.value]
                
                temp_graph = self.hierarchy_graph.copy()
                temp_graph.remove_edges_from([e for e in temp_graph.edges() 
                                           if e not in filtered_edges])
            else:
                temp_graph = self.hierarchy_graph
            
            # 使用NetworkX查找最短路径
            if nx.has_path(temp_graph, source_concept, target_concept):
                return nx.shortest_path(temp_graph, source_concept, target_concept)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"查找概念路径失败: {str(e)}")
            return []
    
    def get_concept_similarity(self, 
                             concept1: str, 
                             concept2: str,
                             method: str = 'hierarchical') -> float:
        """
        计算概念相似度
        
        Args:
            concept1: 概念1
            concept2: 概念2
            method: 相似度计算方法
            
        Returns:
            float: 相似度值 (0-1)
        """
        try:
            if concept1 not in self.concepts or concept2 not in self.concepts:
                return 0.0
            
            if method == 'hierarchical':
                return self._hierarchical_similarity(concept1, concept2)
            elif method == 'path':
                return self._path_similarity(concept1, concept2)
            elif method == 'feature':
                return self._feature_similarity(concept1, concept2)
            elif method == 'context':
                return self._context_similarity(concept1, concept2)
            else:
                raise ValueError(f"不支持的相似度计算方法: {method}")
                
        except Exception as e:
            self.logger.error(f"概念相似度计算失败: {str(e)}")
            return 0.0
    
    def get_concept_ancestors(self, concept_id: str, max_depth: int = None) -> Set[str]:
        """
        获取概念的祖先节点
        
        Args:
            concept_id: 概念ID
            max_depth: 最大深度
            
        Returns:
            Set[str]: 祖先概念集合
        """
        try:
            if concept_id not in self.concepts:
                return set()
            
            ancestors = set()
            
            def dfs_ancestors(node_id: str, current_depth: int):
                if max_depth and current_depth >= max_depth:
                    return
                
                for parent_id in self.concepts[node_id].parent_concepts:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        dfs_ancestors(parent_id, current_depth + 1)
            
            dfs_ancestors(concept_id, 0)
            return ancestors
            
        except Exception as e:
            self.logger.error(f"获取概念祖先失败: {concept_id}, 错误: {str(e)}")
            return set()
    
    def get_concept_descendants(self, concept_id: str, max_depth: int = None) -> Set[str]:
        """
        获取概念的后代节点
        
        Args:
            concept_id: 概念ID
            max_depth: 最大深度
            
        Returns:
            Set[str]: 后代概念集合
        """
        try:
            if concept_id not in self.concepts:
                return set()
            
            descendants = set()
            
            def dfs_descendants(node_id: str, current_depth: int):
                if max_depth and current_depth >= max_depth:
                    return
                
                for child_id in self.concepts[node_id].child_concepts:
                    if child_id not in descendants:
                        descendants.add(child_id)
                        dfs_descendants(child_id, current_depth + 1)
            
            dfs_descendants(concept_id, 0)
            return descendants
            
        except Exception as e:
            self.logger.error(f"获取概念后代失败: {concept_id}, 错误: {str(e)}")
            return set()
    
    def classify_concept(self, 
                        concept_data: Dict[str, Any],
                        method: ClassificationMethod = ClassificationMethod.STATISTICAL) -> ClassificationResult:
        """
        对概念进行分类
        
        Args:
            concept_data: 概念数据
            method: 分类方法
            
        Returns:
            ClassificationResult: 分类结果
        """
        try:
            if method == ClassificationMethod.STATISTICAL:
                return self._statistical_classification(concept_data)
            elif method == ClassificationMethod.LINGUISTIC:
                return self._linguistic_classification(concept_data)
            elif method == ClassificationMethod.SEMANTIC:
                return self._semantic_classification(concept_data)
            elif method == ClassificationMethod.FREQUENCY:
                return self._frequency_classification(concept_data)
            elif method == ClassificationMethod.CONTEXT:
                return self._context_classification(concept_data)
            else:
                raise ValueError(f"不支持的分类方法: {method}")
                
        except Exception as e:
            self.logger.error(f"概念分类失败: {str(e)}")
            return ClassificationResult(
                concept_id=concept_data.get('id', ''),
                assigned_category='unknown',
                confidence=0.0,
                method=method
            )
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """
        获取层次结构统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.hierarchy_stats.copy()
        
        # 概念分布统计
        level_distribution = {level: len(concepts) for level, concepts in self.levels.items()}
        stats['level_distribution'] = level_distribution
        
        # 关系类型分布
        relation_type_dist = defaultdict(int)
        for relation in self.hierarchy_relations:
            relation_type_dist[relation.relation_type.value] += 1
        stats['relation_type_distribution'] = dict(relation_type_dist)
        
        # 概念复杂度统计
        if self.concepts:
            branching_factors = [len(concept.child_concepts) for concept in self.concepts.values()]
            stats['avg_branching_factor'] = np.mean(branching_factors) if branching_factors else 0.0
            stats['max_branching_factor'] = max(branching_factors) if branching_factors else 0.0
            
            depths = [concept.level for concept in self.concepts.values()]
            stats['avg_depth'] = np.mean(depths) if depths else 0.0
            stats['max_depth'] = max(depths) if depths else 0.0
        
        return stats
    
    def export_hierarchy(self, format: str = 'json') -> Dict[str, Any]:
        """
        导出层次结构
        
        Args:
            format: 导出格式 ('json', 'tree', 'network')
            
        Returns:
            Dict[str, Any]: 导出的层次结构数据
        """
        if format == 'json':
            return self._export_json()
        elif format == 'tree':
            return self._export_tree()
        elif format == 'network':
            return self._export_network()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _initialize_default_classifiers(self):
        """初始化默认分类器"""
        # 统计分类器
        self.classifiers[ClassificationMethod.STATISTICAL] = {
            'type': 'clustering',
            'algorithm': 'agglomerative',
            'parameters': {'n_clusters': None, 'linkage': 'ward'}
        }
        
        # 语言学分类器
        self.classifiers[ClassificationMethod.LINGUISTIC] = {
            'type': 'rule_based',
            'patterns': {
                'person': ['人', '人民', '群众', '市民', '居民', '个体'],
                'location': ['地方', '地区', '城市', '国家', '省份', '县域'],
                'organization': ['公司', '企业', '机构', '组织', '部门', '单位']
            }
        }
    
    def _add_base_concepts(self, entities: List[Tuple[str, str, Dict[str, Any]]]):
        """添加基础概念"""
        for entity_name, entity_type, properties in entities:
            concept_id = f"{entity_type}_{entity_name}"
            
            # 确定概念级别（基于类型）
            level = self._get_concept_level(entity_type)
            
            self.add_concept(
                concept_id=concept_id,
                name=entity_name,
                description=f"{entity_type}类型的概念",
                level=level,
                properties=properties,
                confidence=0.8
            )
    
    def _get_concept_level(self, entity_type: str) -> int:
        """根据实体类型确定概念级别"""
        level_mapping = {
            'PERSON': 3,
            'LOCATION': 2,
            'ORGANIZATION': 2,
            'EVENT': 3,
            'CONCEPT': 1,
            'PRODUCT': 3,
            'TIME': 2,
            'NUMBER': 3
        }
        return level_mapping.get(entity_type.upper(), 3)
    
    def _build_statistical_hierarchy(self, entities: List[Tuple[str, str, Dict[str, Any]]], **kwargs):
        """构建统计层次结构"""
        # 提取特征
        features = []
        entity_names = []
        
        for entity_name, entity_type, properties in entities:
            feature_vector = self._extract_feature_vector(entity_name, entity_type, properties)
            features.append(feature_vector)
            entity_names.append(entity_name)
        
        if len(features) < self.min_cluster_size:
            return
        
        # 执行聚类
        n_clusters = kwargs.get('n_clusters', min(len(features) // self.min_cluster_size, 10))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(features)
        
        # 构建层次关系
        for i, (entity_name, entity_type, _) in enumerate(entities):
            cluster_id = cluster_labels[i]
            concept_id = f"{entity_type}_{entity_name}"
            
            # 添加聚类中心作为父概念
            parent_concept = f"cluster_{cluster_id}_{entity_type}"
            
            if parent_concept not in self.concepts:
                self.add_concept(
                    concept_id=parent_concept,
                    name=f"{entity_type}类群集{cluster_id}",
                    description=f"包含多个{entity_type}实体的聚类",
                    level=self._get_concept_level(entity_type) - 1
                )
            
            # 添加层次关系
            self.add_hierarchy_relation(
                source_concept=parent_concept,
                target_concept=concept_id,
                relation_type=HierarchyType.TAXONOMY,
                strength=0.7,
                confidence=0.8
            )
    
    def _build_linguistic_hierarchy(self, entities: List[Tuple[str, str, Dict[str, Any]]], **kwargs):
        """构建语言学层次结构"""
        # 基于语言学规则进行分类
        for entity_name, entity_type, properties in entities:
            # 检查是否匹配语言学模式
            for category, patterns in self.classifiers[ClassificationMethod.LINGUISTIC]['patterns'].items():
                for pattern in patterns:
                    if pattern in entity_name:
                        # 添加分类关系
                        category_concept = f"category_{category}"
                        
                        if category_concept not in self.concepts:
                            self.add_concept(
                                concept_id=category_concept,
                                name=category,
                                description=f"{category}类别的概念",
                                level=1
                            )
                        
                        self.add_hierarchy_relation(
                            source_concept=category_concept,
                            target_concept=f"{entity_type}_{entity_name}",
                            relation_type=HierarchyType.TAXONOMY,
                            confidence=0.6
                        )
                        break
    
    def _build_semantic_hierarchy(self, entities: List[Tuple[str, str, Dict[str, Any]]], **kwargs):
        """构建语义层次结构"""
        # 计算语义相似度并构建层次
        entity_ids = [f"{entity_type}_{entity_name}" for entity_name, entity_type, _ in entities]
        
        for i, entity_id1 in enumerate(entity_ids):
            for j, entity_id2 in enumerate(entity_ids[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(entity_id1, entity_id2)
                
                if similarity >= self.similarity_threshold:
                    # 确定父子关系（基于相似度和频率）
                    freq1 = self.concepts.get(entity_id1, {}).frequency
                    freq2 = self.concepts.get(entity_id2, {}).frequency
                    
                    if freq1 > freq2:
                        parent, child = entity_id1, entity_id2
                    else:
                        parent, child = entity_id2, entity_id1
                    
                    self.add_hierarchy_relation(
                        source_concept=parent,
                        target_concept=child,
                        relation_type=HierarchyType.SEMANTIC,
                        strength=similarity,
                        confidence=min(0.9, similarity + 0.2)
                    )
    
    def _build_frequency_hierarchy(self, entities: List[Tuple[str, str, Dict[str, Any]]], **kwargs):
        """构建频率层次结构"""
        # 计算实体频率
        frequency_map = defaultdict(int)
        for entity_name, entity_type, _ in entities:
            frequency_map[f"{entity_type}_{entity_name}"] += 1
        
        # 按频率排序
        sorted_entities = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        
        # 构建频率层次
        for i, (entity_id, freq) in enumerate(sorted_entities):
            if i > 0:
                # 与前一个高频实体建立关系
                prev_entity_id = sorted_entities[i-1][0]
                
                self.add_hierarchy_relation(
                    source_concept=prev_entity_id,
                    target_concept=entity_id,
                    relation_type=HierarchyType.FREQUENCY,
                    strength=freq / sorted_entities[0][1],  # 相对于最高频率的比率
                    confidence=0.7
                )
    
    def _build_context_hierarchy(self, entities: List[Tuple[str, str, Dict[str, Any]]], **kwargs):
        """构建上下文层次结构"""
        # 基于共现关系构建上下文层次
        context_relations = defaultdict(int)
        
        # 分析文本上下文（简化实现）
        for i, (entity_name1, entity_type1, _) in enumerate(entities):
            for j, (entity_name2, entity_type2, _) in enumerate(entities[i+1:], i+1):
                # 检查是否在同一上下文中出现
                if self._are_in_same_context(entity_name1, entity_name2, entities):
                    entity_id1 = f"{entity_type1}_{entity_name1}"
                    entity_id2 = f"{entity_type2}_{entity_name2}"
                    context_relations[(entity_id1, entity_id2)] += 1
        
        # 构建上下文层次关系
        for (entity_id1, entity_id2), frequency in context_relations.items():
            if frequency >= 2:  # 至少共同出现2次
                self.add_hierarchy_relation(
                    source_concept=entity_id1,
                    target_concept=entity_id2,
                    relation_type=HierarchyType.ASSOCIATION,
                    strength=min(1.0, frequency / 5.0),
                    confidence=0.6,
                    evidence=[f"共同出现{frequency}次"]
                )
    
    def _extract_feature_vector(self, name: str, entity_type: str, properties: Dict[str, Any]) -> List[float]:
        """提取特征向量"""
        features = []
        
        # 名称特征
        features.append(len(name))  # 名称长度
        features.append(name.count(' ') + 1)  # 单词数量
        
        # 类型特征
        type_features = [0] * 8  # 支持8种类型
        type_index = {
            'PERSON': 0, 'LOCATION': 1, 'ORGANIZATION': 2, 'EVENT': 3,
            'CONCEPT': 4, 'PRODUCT': 5, 'TIME': 6, 'NUMBER': 7
        }
        type_idx = type_index.get(entity_type.upper(), 7)
        type_features[type_idx] = 1
        features.extend(type_features)
        
        # 属性特征
        features.append(len(properties))  # 属性数量
        
        # 数值属性
        numeric_properties = [v for v in properties.values() if isinstance(v, (int, float))]
        if numeric_properties:
            features.append(np.mean(numeric_properties))
            features.append(np.std(numeric_properties))
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _are_in_same_context(self, entity1: str, entity2: str, entities: List[Tuple[str, str, Dict[str, Any]]]) -> bool:
        """检查两个实体是否在同一上下文中"""
        # 简化实现：假设列表中的相邻实体在同一上下文
        entity_names = [entity[0] for entity in entities]
        
        try:
            idx1 = entity_names.index(entity1)
            idx2 = entity_names.index(entity2)
            
            # 检查是否相邻或在窗口范围内
            return abs(idx1 - idx2) <= 2
        except ValueError:
            return False
    
    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """计算语义相似度"""
        # 简化实现：基于概念属性
        node1 = self.concepts.get(concept1)
        node2 = self.concepts.get(concept2)
        
        if not node1 or not node2:
            return 0.0
        
        # 属性重叠度
        props1 = set(node1.properties.keys()) | set(node1.properties.values())
        props2 = set(node2.properties.keys()) | set(node2.properties.values())
        
        intersection = len(props1 & props2)
        union = len(props1 | props2)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # 名称相似度
        name_similarity = self._string_similarity(node1.name, node2.name)
        
        # 组合相似度
        combined_similarity = 0.6 * jaccard_similarity + 0.4 * name_similarity
        
        return combined_similarity
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        if str1 == str2:
            return 1.0
        
        # 简化的编辑距离相似度
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str2 if len(str1) > len(str2) else str1
        
        if len(longer) == 0:
            return 1.0
        
        # 计算公共子序列长度
        common_length = 0
        for i in range(len(shorter)):
            if shorter[i] in longer:
                common_length += 1
        
        return common_length / len(longer)
    
    def _hierarchical_similarity(self, concept1: str, concept2: str) -> float:
        """计算层次相似度"""
        # 找到最近公共祖先
        ancestors1 = self.get_concept_ancestors(concept1)
        ancestors2 = self.get_concept_ancestors(concept2)
        
        common_ancestors = ancestors1 & ancestors2
        
        if not common_ancestors:
            return 0.0
        
        # 找到最近的公共祖先
        min_depth = float('inf')
        nearest_ancestor = None
        
        for ancestor in common_ancestors:
            depth = self.concepts[ancestor].level
            if depth < min_depth:
                min_depth = depth
                nearest_ancestor = ancestor
        
        # 计算层次距离
        distance1 = len(nx.shortest_path(self.hierarchy_graph, concept1, nearest_ancestor))
        distance2 = len(nx.shortest_path(self.hierarchy_graph, concept2, nearest_ancestor))
        
        total_distance = distance1 + distance2
        
        # 转换为相似度
        similarity = 1.0 / (1.0 + total_distance)
        return similarity
    
    def _path_similarity(self, concept1: str, concept2: str) -> float:
        """计算路径相似度"""
        path = self.find_concept_path(concept1, concept2)
        
        if not path:
            return 0.0
        
        # 路径长度相似度
        path_length = len(path)
        similarity = 1.0 / (1.0 + path_length)
        
        return similarity
    
    def _feature_similarity(self, concept1: str, concept2: str) -> float:
        """计算特征相似度"""
        node1 = self.concepts.get(concept1)
        node2 = self.concepts.get(concept2)
        
        if not node1 or not node2:
            return 0.0
        
        # 属性特征相似度
        props1 = np.array([v for v in node1.properties.values() if isinstance(v, (int, float))])
        props2 = np.array([v for v in node2.properties.values() if isinstance(v, (int, float))])
        
        if len(props1) == 0 or len(props2) == 0:
            return 0.0
        
        # 补齐向量长度
        max_len = max(len(props1), len(props2))
        if len(props1) < max_len:
            props1 = np.pad(props1, (0, max_len - len(props1)))
        if len(props2) < max_len:
            props2 = np.pad(props2, (0, max_len - len(props2)))
        
        try:
            similarity = 1 - cosine(props1, props2)
            return max(0.0, similarity)
        except:
            return 0.0
    
    def _context_similarity(self, concept1: str, concept2: str) -> float:
        """计算上下文相似度"""
        # 基于共同邻居的概念相似度
        neighbors1 = set(self.hierarchy_graph.neighbors(concept1))
        neighbors2 = set(self.hierarchy_graph.neighbors(concept2))
        
        common_neighbors = neighbors1 & neighbors2
        total_neighbors = neighbors1 | neighbors2
        
        if len(total_neighbors) == 0:
            return 0.0
        
        similarity = len(common_neighbors) / len(total_neighbors)
        return similarity
    
    def _statistical_classification(self, concept_data: Dict[str, Any]) -> ClassificationResult:
        """统计分类"""
        # 简化的统计分类器
        name = concept_data.get('name', '')
        properties = concept_data.get('properties', {})
        
        # 基于名称特征进行分类
        if any(char.isdigit() for char in name):
            category = 'number'
        elif any(word in name for word in ['公司', '企业', '组织']):
            category = 'organization'
        elif any(word in name for word in ['北京', '上海', '广州']):
            category = 'location'
        elif any(word in name for word in ['人', '先生', '女士', '老师']):
            category = 'person'
        else:
            category = 'concept'
        
        confidence = 0.7
        
        return ClassificationResult(
            concept_id=concept_data.get('id', ''),
            assigned_category=category,
            confidence=confidence,
            method=ClassificationMethod.STATISTICAL,
            features={'name_length': len(name), 'property_count': len(properties)},
            reasoning=f"基于统计特征分类: {category}"
        )
    
    def _linguistic_classification(self, concept_data: Dict[str, Any]) -> ClassificationResult:
        """语言学分类"""
        name = concept_data.get('name', '')
        
        for category, patterns in self.classifiers[ClassificationMethod.LINGUISTIC]['patterns'].items():
            for pattern in patterns:
                if pattern in name:
                    return ClassificationResult(
                        concept_id=concept_data.get('id', ''),
                        assigned_category=category,
                        confidence=0.8,
                        method=ClassificationMethod.LINGUISTIC,
                        reasoning=f"匹配语言学模式: {pattern}"
                    )
        
        return ClassificationResult(
            concept_id=concept_data.get('id', ''),
            assigned_category='unknown',
            confidence=0.1,
            method=ClassificationMethod.LINGUISTIC
        )
    
    def _semantic_classification(self, concept_data: Dict[str, Any]) -> ClassificationResult:
        """语义分类"""
        # 基于语义特征的分类
        properties = concept_data.get('properties', {})
        
        if 'location' in properties or 'coordinates' in properties:
            category = 'location'
        elif 'population' in properties or 'area' in properties:
            category = 'geographical'
        elif 'revenue' in properties or 'employees' in properties:
            category = 'organization'
        else:
            category = 'concept'
        
        confidence = 0.6
        
        return ClassificationResult(
            concept_id=concept_data.get('id', ''),
            assigned_category=category,
            confidence=confidence,
            method=ClassificationMethod.SEMANTIC,
            reasoning=f"基于语义属性分类: {category}"
        )
    
    def _frequency_classification(self, concept_data: Dict[str, Any]) -> ClassificationResult:
        """频率分类"""
        frequency = concept_data.get('frequency', 0)
        
        if frequency >= 100:
            category = 'high_frequency'
            confidence = 0.9
        elif frequency >= 10:
            category = 'medium_frequency'
            confidence = 0.7
        else:
            category = 'low_frequency'
            confidence = 0.5
        
        return ClassificationResult(
            concept_id=concept_data.get('id', ''),
            assigned_category=category,
            confidence=confidence,
            method=ClassificationMethod.FREQUENCY,
            reasoning=f"基于频率分级: {category}"
        )
    
    def _context_classification(self, concept_data: Dict[str, Any]) -> ClassificationResult:
        """上下文分类"""
        # 基于上下文的分类逻辑
        context = concept_data.get('context', '')
        
        if '工作' in context or '职业' in context:
            category = 'occupation'
        elif '地点' in context or '位置' in context:
            category = 'spatial'
        elif '时间' in context or '日期' in context:
            category = 'temporal'
        else:
            category = 'general'
        
        confidence = 0.6
        
        return ClassificationResult(
            concept_id=concept_data.get('id', ''),
            assigned_category=category,
            confidence=confidence,
            method=ClassificationMethod.CONTEXT,
            reasoning=f"基于上下文分类: {category}"
        )
    
    def _optimize_hierarchy(self):
        """优化层次结构"""
        # 移除孤立节点
        isolated_nodes = list(nx.isolates(self.hierarchy_graph))
        for node in isolated_nodes:
            if node in self.concepts:
                del self.concepts[node]
                self.hierarchy_graph.remove_node(node)
        
        # 合并相似节点
        self._merge_similar_nodes()
        
        # 重新计算层次级别
        self._recalculate_levels()
    
    def _merge_similar_nodes(self):
        """合并相似节点"""
        similar_pairs = []
        
        for concept_id1, concept1 in self.concepts.items():
            for concept_id2, concept2 in self.concepts.items():
                if concept_id1 >= concept_id2:
                    continue
                
                similarity = self.get_concept_similarity(concept_id1, concept_id2)
                
                if similarity > 0.9:  # 高度相似
                    similar_pairs.append((concept_id1, concept_id2, similarity))
        
        # 合并相似节点
        for concept_id1, concept_id2, similarity in similar_pairs:
            if concept_id1 in self.concepts and concept_id2 in self.concepts:
                # 保留较新的概念，合并属性
                if self.concepts[concept_id1].created_at > self.concepts[concept_id2].created_at:
                    primary, secondary = concept_id1, concept_id2
                else:
                    primary, secondary = concept_id2, concept_id1
                
                # 合并属性
                self.concepts[primary].properties.update(self.concepts[secondary].properties)
                self.concepts[primary].aliases.update(self.concepts[secondary].aliases)
                
                # 更新连接
                for parent in self.concepts[secondary].parent_concepts:
                    if parent in self.concepts:
                        self.concepts[parent].child_concepts.remove(secondary)
                        self.concepts[parent].child_concepts.add(primary)
                
                for child in self.concepts[secondary].child_concepts:
                    if child in self.concepts:
                        self.concepts[child].parent_concepts.remove(secondary)
                        self.concepts[child].parent_concepts.add(primary)
                
                # 删除被合并的节点
                del self.concepts[secondary]
                self.hierarchy_graph.remove_node(secondary)
    
    def _recalculate_levels(self):
        """重新计算层次级别"""
        # 重置级别
        for concept in self.concepts.values():
            concept.level = 0
        
        # 使用拓扑排序重新计算级别
        try:
            sorted_concepts = list(nx.topological_sort(self.hierarchy_graph))
            
            for concept_id in sorted_concepts:
                if concept_id in self.concepts:
                    # 计算基于父节点的最大级别
                    max_parent_level = -1
                    for parent_id in self.concepts[concept_id].parent_concepts:
                        if parent_id in self.concepts:
                            max_parent_level = max(max_parent_level, self.concepts[parent_id].level)
                    
                    if max_parent_level >= 0:
                        self.concepts[concept_id].level = max_parent_level + 1
                    
                    # 更新层次索引
                    self.levels[self.concepts[concept_id].level].add(concept_id)
                    
        except nx.NetworkXError:
            # 图中存在环，使用BFS计算级别
            self._calculate_levels_bfs()
    
    def _calculate_levels_bfs(self):
        """使用BFS计算级别"""
        # 找到根节点（无父节点的节点）
        root_nodes = [cid for cid, concept in self.concepts.items() 
                     if not concept.parent_concepts]
        
        # 使用队列进行BFS
        queue = deque()
        for root in root_nodes:
            self.concepts[root].level = 0
            self.levels[0].add(root)
            queue.append(root)
        
        while queue:
            current = queue.popleft()
            
            for child_id in self.concepts[current].child_concepts:
                if child_id in self.concepts:
                    child_level = self.concepts[current].level + 1
                    
                    if self.concepts[child_id].level < child_level:
                        self.concepts[child_id].level = child_level
                        self.levels[child_level].add(child_id)
                        queue.append(child_id)
    
    def _validate_hierarchy_consistency(self):
        """验证层次一致性"""
        issues = []
        
        # 检查环
        try:
            cycles = list(nx.simple_cycles(self.hierarchy_graph.to_directed()))
            if cycles:
                issues.append(f"发现 {len(cycles)} 个环: {cycles}")
        except:
            pass
        
        # 检查孤立节点
        isolated = list(nx.isolates(self.hierarchy_graph))
        if isolated:
            issues.append(f"发现 {len(isolated)} 个孤立节点")
        
        # 检查深度
        max_depth = max((concept.level for concept in self.concepts.values()), default=0)
        if max_depth > self.max_levels:
            issues.append(f"最大深度 {max_depth} 超过限制 {self.max_levels}")
        
        # 检查一致性
        for concept_id, concept in self.concepts.items():
            # 检查父子关系一致性
            for parent_id in concept.parent_concepts:
                if concept_id not in self.concepts[parent_id].child_concepts:
                    issues.append(f"父子关系不一致: {parent_id} -> {concept_id}")
            
            for child_id in concept.child_concepts:
                if concept_id not in self.concepts[child_id].parent_concepts:
                    issues.append(f"父子关系不一致: {concept_id} -> {child_id}")
        
        if issues:
            self.logger.warning(f"层次结构一致性检查发现问题: {issues}")
        else:
            self.logger.info("层次结构一致性检查通过")
    
    def _update_hierarchy_stats(self):
        """更新层次统计信息"""
        self.hierarchy_stats['total_concepts'] = len(self.concepts)
        self.hierarchy_stats['total_relations'] = len(self.hierarchy_relations)
        
        # 计算最大深度
        if self.concepts:
            max_depth = max(concept.level for concept in self.concepts.values())
            self.hierarchy_stats['max_depth'] = max_depth
        
        self.hierarchy_stats['last_update'] = datetime.now()
    
    def _export_json(self) -> Dict[str, Any]:
        """导出JSON格式"""
        concepts_data = {}
        for concept_id, concept in self.concepts.items():
            concepts_data[concept_id] = {
                'concept_id': concept.concept_id,
                'name': concept.name,
                'description': concept.description,
                'level': concept.level,
                'parent_concepts': list(concept.parent_concepts),
                'child_concepts': list(concept.child_concepts),
                'properties': concept.properties,
                'confidence': concept.confidence,
                'created_at': concept.created_at.isoformat(),
                'updated_at': concept.updated_at.isoformat(),
                'sources': list(concept.sources),
                'aliases': list(concept.aliases),
                'frequency': concept.frequency,
                'popularity': concept.popularity
            }
        
        relations_data = []
        for relation in self.hierarchy_relations:
            relations_data.append({
                'source_concept': relation.source_concept,
                'target_concept': relation.target_concept,
                'relation_type': relation.relation_type.value,
                'strength': relation.strength,
                'confidence': relation.confidence,
                'evidence': relation.evidence,
                'created_at': relation.created_at.isoformat(),
                'metadata': relation.metadata
            })
        
        return {
            'concepts': concepts_data,
            'relations': relations_data,
            'statistics': self.get_hierarchy_statistics(),
            'levels': {level: list(concepts) for level, concepts in self.levels.items()}
        }
    
    def _export_tree(self) -> Dict[str, Any]:
        """导出树格式"""
        tree = {}
        
        # 找到根节点
        root_concepts = [cid for cid, concept in self.concepts.items() 
                        if not concept.parent_concepts]
        
        def build_subtree(concept_id: str) -> Dict[str, Any]:
            if concept_id not in self.concepts:
                return {}
            
            concept = self.concepts[concept_id]
            subtree = {
                'name': concept.name,
                'description': concept.description,
                'level': concept.level,
                'properties': concept.properties,
                'children': []
            }
            
            for child_id in concept.child_concepts:
                subtree['children'].append(build_subtree(child_id))
            
            return subtree
        
        for root_id in root_concepts:
            tree[root_id] = build_subtree(root_id)
        
        return tree
    
    def _export_network(self) -> Dict[str, Any]:
        """导出网络格式"""
        nodes = []
        for concept_id, concept in self.concepts.items():
            nodes.append({
                'id': concept_id,
                'name': concept.name,
                'level': concept.level,
                'properties': concept.properties
            })
        
        edges = []
        for relation in self.hierarchy_relations:
            edges.append({
                'source': relation.source_concept,
                'target': relation.target_concept,
                'type': relation.relation_type.value,
                'strength': relation.strength,
                'confidence': relation.confidence
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_hierarchy_statistics()
        }