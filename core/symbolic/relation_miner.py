#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关系挖掘模块

本模块实现知识图谱的关系挖掘器，从实体和文本数据中发现关系模式。
支持多种关系挖掘算法，包括共现分析、关联规则挖掘、路径分析等。

主要功能：
- 基于共现的关系发现
- 关联规则挖掘
- 路径关系分析
- 语义关系推断
- 关系强度评估
- 动态关系演化

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import logging
import itertools
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import re
from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RelationType(Enum):
    """关系类型枚举"""
    CO_OCCUR = "co_occur"           # 共现关系
    SEMANTIC = "semantic"           # 语义关系
    CAUSAL = "causal"              # 因果关系
    TEMPORAL = "temporal"          # 时间关系
    SPATIAL = "spatial"            # 空间关系
    SIMILARITY = "similarity"      # 相似关系
    OPPOSITE = "opposite"          # 对立关系
    PART_OF = "part_of"           # 组成关系
    IS_A = "is_a"                 # 分类关系
    CUSTOM = "custom"             # 自定义关系


@dataclass
class RelationInstance:
    """关系实例"""
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float
    support_count: int
    context: str
    extracted_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RelationPattern:
    """关系模式"""
    pattern_id: str
    relation_type: RelationType
    source_types: Set[str]
    target_types: Set[str]
    confidence: float
    support: float
    lift: float
    contexts: List[str]
    examples: List[Tuple[str, str]]
    
    def __post_init__(self):
        if not self.contexts:
            self.contexts = []
        if not self.examples:
            self.examples = []


@dataclass
class AssociationRule:
    """关联规则"""
    antecedent: Set[str]
    consequent: Set[str]
    confidence: float
    support: float
    lift: float
    conviction: float


class RelationMiner:
    """
    关系挖掘器
    
    实现多种关系发现算法：
    - 基于共现统计的关系挖掘
    - 关联规则挖掘（Apriori算法）
    - 路径分析关系发现
    - 语义关系推断
    - 时间序列关系分析
    
    特性：
    - 支持大规模数据处理
    - 多种相似度度量
    - 动态关系更新
    - 关系强度评估
    - 可配置的关系类型
    """
    
    def __init__(self, 
                 min_support: float = 0.01,
                 min_confidence: float = 0.5,
                 similarity_threshold: float = 0.3,
                 context_window: int = 100,
                 enable_temporal: bool = True,
                 enable_spatial: bool = True):
        """
        初始化关系挖掘器
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
            similarity_threshold: 相似度阈值
            context_window: 上下文窗口大小
            enable_temporal: 是否启用时间关系分析
            enable_spatial: 是否启用空间关系分析
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        self.context_window = context_window
        self.enable_temporal = enable_temporal
        self.enable_spatial = enable_spatial
        
        # 关系存储
        self.relation_instances = []  # 关系实例列表
        self.relation_patterns = {}   # 关系模式字典
        self.entity_cooccurrence = defaultdict(int)  # 实体共现统计
        self.relation_weights = defaultdict(float)   # 关系权重
        
        # 分析结果缓存
        self.pattern_cache = {}
        self.similarity_cache = {}
        
        # 统计信息
        self.mining_stats = {
            'total_entities': 0,
            'total_relations': 0,
            'mining_time': 0.0,
            'patterns_found': 0,
            'rules_generated': 0
        }
        
        # 相似度计算器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english' if isinstance('test', str) else None
        )
        
        self.logger = logging.getLogger("RelationMiner")
        
        self.logger.info(f"关系挖掘器初始化完成，支持度: {min_support}, 置信度: {min_confidence}")
    
    def mine_relations_from_entities(self, 
                                    entities_list: List[List[Tuple[str, str]]],
                                    texts: List[str] = None,
                                    timestamps: List[datetime] = None) -> List[RelationInstance]:
        """
        从实体列表中挖掘关系
        
        Args:
            entities_list: 实体列表，每项包含(实体名, 实体类型)
            texts: 对应的文本列表（可选）
            timestamps: 时间戳列表（可选）
            
        Returns:
            List[RelationInstance]: 挖掘出的关系实例列表
        """
        start_time = datetime.now()
        relations = []
        
        self.logger.info(f"开始从 {len(entities_list)} 个文档中挖掘关系")
        
        # 1. 计算实体共现
        self._calculate_cooccurrence(entities_list)
        
        # 2. 挖掘基于共现的关系
        cooccurrence_relations = self._mine_cooccurrence_relations(entities_list)
        relations.extend(cooccurrence_relations)
        
        # 3. 挖掘语义相似关系
        if texts:
            semantic_relations = self._mine_semantic_relations(entities_list, texts)
            relations.extend(semantic_relations)
        
        # 4. 挖掘时间关系
        if self.enable_temporal and timestamps:
            temporal_relations = self._mine_temporal_relations(entities_list, timestamps)
            relations.extend(temporal_relations)
        
        # 5. 挖掘关联规则
        association_rules = self._mine_association_rules(entities_list)
        
        # 转换关联规则为关系实例
        for rule in association_rules:
            relation = RelationInstance(
                source_entity=str(list(rule.antecedent)[0]) if rule.antecedent else "",
                target_entity=str(list(rule.consequent)[0]) if rule.consequent else "",
                relation_type=RelationType.CO_OCCUR,
                confidence=rule.confidence,
                support_count=int(rule.support * len(entities_list)),
                context="association_rule",
                extracted_at=datetime.now(),
                metadata={
                    'lift': rule.lift,
                    'conviction': rule.conviction,
                    'rule_type': 'association'
                }
            )
            relations.append(relation)
        
        # 存储关系实例
        self.relation_instances.extend(relations)
        
        # 更新统计
        self.mining_stats['total_entities'] += sum(len(entities) for entities in entities_list)
        self.mining_stats['total_relations'] += len(relations)
        self.mining_stats['mining_time'] += (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"关系挖掘完成，发现 {len(relations)} 个关系")
        return relations
    
    def mine_relations_from_text(self, 
                                texts: List[str], 
                                entity_extractor) -> List[RelationInstance]:
        """
        从文本中直接挖掘关系
        
        Args:
            texts: 文本列表
            entity_extractor: 实体抽取器实例
            
        Returns:
            List[RelationInstance]: 挖掘出的关系实例列表
        """
        relations = []
        
        self.logger.info(f"从 {len(texts)} 个文本中挖掘关系")
        
        # 从每个文本中抽取实体
        all_entities = []
        for text in texts:
            entities = entity_extractor.extract_entities(text)
            # 转换为 (实体名, 实体类型) 格式
            entity_tuples = [(entity.text, entity.entity_type.value) for entity in entities]
            all_entities.append(entity_tuples)
        
        # 挖掘关系
        relations = self.mine_relations_from_entities(all_entities, texts)
        
        return relations
    
    def find_relation_patterns(self, 
                              min_frequency: int = 3,
                              max_pattern_length: int = 3) -> List[RelationPattern]:
        """
        发现关系模式
        
        Args:
            min_frequency: 最小出现频率
            max_pattern_length: 最大模式长度
            
        Returns:
            List[RelationPattern]: 关系模式列表
        """
        patterns = []
        
        # 按关系类型分组
        relations_by_type = defaultdict(list)
        for relation in self.relation_instances:
            relations_by_type[relation.relation_type].append(relation)
        
        for relation_type, relations in relations_by_type.items():
            if len(relations) < min_frequency:
                continue
            
            # 构建关系网络
            relation_network = self._build_relation_network(relations)
            
            # 寻找频繁模式
            frequent_patterns = self._find_frequent_patterns(
                relation_network, min_frequency, max_pattern_length
            )
            
            # 转换为关系模式
            for pattern in frequent_patterns:
                pattern_obj = self._create_relation_pattern(pattern, relation_type, relations)
                if pattern_obj:
                    patterns.append(pattern_obj)
        
        # 存储模式
        for pattern in patterns:
            self.relation_patterns[pattern.pattern_id] = pattern
        
        self.mining_stats['patterns_found'] += len(patterns)
        
        self.logger.info(f"发现 {len(patterns)} 个关系模式")
        return patterns
    
    def calculate_similarity(self, 
                           entity1: str, 
                           entity2: str, 
                           method: str = 'cosine') -> float:
        """
        计算两个实体之间的相似度
        
        Args:
            entity1: 实体1
            entity2: 实体2  
            method: 相似度计算方法 ('cosine', 'jaccard', 'pearson')
            
        Returns:
            float: 相似度值 (0-1)
        """
        # 检查缓存
        cache_key = f"{entity1}_{entity2}_{method}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            if method == 'cosine':
                similarity = self._cosine_similarity(entity1, entity2)
            elif method == 'jaccard':
                similarity = self._jaccard_similarity(entity1, entity2)
            elif method == 'pearson':
                similarity = self._pearson_similarity(entity1, entity2)
            else:
                raise ValueError(f"不支持的相似度计算方法: {method}")
            
            # 缓存结果
            self.similarity_cache[cache_key] = similarity
            return similarity
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {entity1} vs {entity2}, 错误: {str(e)}")
            return 0.0
    
    def update_relation_strength(self, 
                               entity1: str, 
                               entity2: str, 
                               relation_type: RelationType,
                               interaction_count: int = 1,
                               temporal_decay: bool = True):
        """
        更新关系强度
        
        Args:
            entity1: 实体1
            entity2: 实体2
            relation_type: 关系类型
            interaction_count: 交互次数
            temporal_decay: 是否应用时间衰减
        """
        relation_key = f"{entity1}::{entity2}::{relation_type.value}"
        
        # 计算当前权重
        current_weight = self.relation_weights.get(relation_key, 0.0)
        
        # 新权重计算
        if temporal_decay:
            # 时间衰减函数（简化实现）
            time_factor = np.exp(-0.1 * (datetime.now() - datetime.now()).days)
            new_weight = current_weight + interaction_count * time_factor
        else:
            new_weight = current_weight + interaction_count
        
        # 更新权重
        self.relation_weights[relation_key] = new_weight
        
        self.logger.debug(f"更新关系强度: {relation_key} -> {new_weight}")
    
    def get_strongest_relations(self, 
                               entity: str, 
                               top_k: int = 10,
                               relation_type: RelationType = None) -> List[Tuple[str, float]]:
        """
        获取实体的最强关系
        
        Args:
            entity: 目标实体
            top_k: 返回前k个最强关系
            relation_type: 关系类型过滤（可选）
            
        Returns:
            List[Tuple[str, float]]: (相关实体, 关系强度) 列表
        """
        relations = []
        
        for relation_key, weight in self.relation_weights.items():
            parts = relation_key.split("::")
            if len(parts) >= 3:
                ent1, ent2, rel_type = parts[0], parts[1], parts[2]
                
                if relation_type and rel_type != relation_type.value:
                    continue
                
                if ent1 == entity:
                    relations.append((ent2, weight))
                elif ent2 == entity:
                    relations.append((ent1, weight))
        
        # 按强度排序并返回前k个
        relations.sort(key=lambda x: x[1], reverse=True)
        return relations[:top_k]
    
    def export_relations(self, format: str = 'json') -> Dict[str, Any]:
        """
        导出关系数据
        
        Args:
            format: 导出格式 ('json', 'csv', 'graph')
            
        Returns:
            Dict[str, Any]: 导出的关系数据
        """
        if format == 'json':
            return self._export_json()
        elif format == 'csv':
            return self._export_csv()
        elif format == 'graph':
            return self._export_graph()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取挖掘统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.mining_stats.copy()
        
        # 关系类型分布
        relation_type_dist = Counter(rel.relation_type for rel in self.relation_instances)
        stats['relation_type_distribution'] = dict(relation_type_dist)
        
        # 实体连接度统计
        entity_degrees = defaultdict(int)
        for rel in self.relation_instances:
            entity_degrees[rel.source_entity] += 1
            entity_degrees[rel.target_entity] += 1
        
        if entity_degrees:
            stats['avg_entity_degree'] = np.mean(list(entity_degrees.values()))
            stats['max_entity_degree'] = max(entity_degrees.values())
        else:
            stats['avg_entity_degree'] = 0
            stats['max_entity_degree'] = 0
        
        return stats
    
    def _calculate_cooccurrence(self, entities_list: List[List[Tuple[str, str]]]):
        """计算实体共现统计"""
        for entities in entities_list:
            entity_names = [entity[0] for entity in entities]
            
            # 计算两两共现
            for i, entity1 in enumerate(entity_names):
                for j, entity2 in enumerate(entity_names[i+1:], i+1):
                    pair = tuple(sorted([entity1, entity2]))
                    self.entity_cooccurrence[pair] += 1
        
        self.logger.debug(f"共现统计完成，共 {len(self.entity_cooccurrence)} 个实体对")
    
    def _mine_cooccurrence_relations(self, entities_list: List[List[Tuple[str, str]]]) -> List[RelationInstance]:
        """挖掘基于共现的关系"""
        relations = []
        total_docs = len(entities_list)
        
        for (entity1, entity2), co_count in self.entity_cooccurrence.items():
            # 计算支持度
            support = co_count / total_docs
            
            if support >= self.min_support:
                # 计算置信度
                conf1_2 = co_count / sum(1 for entities in entities_list if entity1 in [e[0] for e in entities])
                conf2_1 = co_count / sum(1 for entities in entities_list if entity2 in [e[0] for e in entities])
                
                if conf1_2 >= self.min_confidence:
                    relation = RelationInstance(
                        source_entity=entity1,
                        target_entity=entity2,
                        relation_type=RelationType.CO_OCCUR,
                        confidence=conf1_2,
                        support_count=co_count,
                        context=f"co_occurrence_in_{co_count}_documents",
                        extracted_at=datetime.now()
                    )
                    relations.append(relation)
                
                if conf2_1 >= self.min_confidence:
                    relation = RelationInstance(
                        source_entity=entity2,
                        target_entity=entity1,
                        relation_type=RelationType.CO_OCCUR,
                        confidence=conf2_1,
                        support_count=co_count,
                        context=f"co_occurrence_in_{co_count}_documents",
                        extracted_at=datetime.now()
                    )
                    relations.append(relation)
        
        self.logger.debug(f"共现关系挖掘完成，发现 {len(relations)} 个关系")
        return relations
    
    def _mine_semantic_relations(self, 
                                entities_list: List[List[Tuple[str, str]]], 
                                texts: List[str]) -> List[RelationInstance]:
        """挖掘语义相似关系"""
        relations = []
        
        try:
            # 构建文档-实体映射
            entity_documents = defaultdict(list)
            for entities, text in zip(entities_list, texts):
                for entity_name, entity_type in entities:
                    entity_documents[entity_name].append(text)
            
            # 计算实体间的TF-IDF相似度
            if len(entity_documents) > 1:
                documents = [' '.join(docs) for docs in entity_documents.values()]
                
                # 检查是否有足够的数据
                non_empty_docs = [doc for doc in documents if doc.strip()]
                if len(non_empty_docs) > 1:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(non_empty_docs)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # 提取高相似度的实体对
                    entity_names = list(entity_documents.keys())
                    for i, entity1 in enumerate(entity_names):
                        for j, entity2 in enumerate(entity_names[i+1:], i+1):
                            similarity = similarity_matrix[i][j]
                            if similarity >= self.similarity_threshold:
                                relation = RelationInstance(
                                    source_entity=entity1,
                                    target_entity=entity2,
                                    relation_type=RelationType.SIMILARITY,
                                    confidence=similarity,
                                    support_count=int(similarity * 10),  # 简化的支持度计算
                                    context="semantic_similarity",
                                    extracted_at=datetime.now()
                                )
                                relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"语义关系挖掘失败: {str(e)}")
        
        self.logger.debug(f"语义关系挖掘完成，发现 {len(relations)} 个关系")
        return relations
    
    def _mine_temporal_relations(self, 
                               entities_list: List[List[Tuple[str, str]]], 
                               timestamps: List[datetime]) -> List[RelationInstance]:
        """挖掘时间关系"""
        relations = []
        
        try:
            # 按时间排序
            time_entities = list(zip(timestamps, entities_list))
            time_entities.sort(key=lambda x: x[0])
            
            # 寻找时间上的共现模式
            window_size = 3  # 时间窗口大小
            
            for i in range(len(time_entities) - window_size + 1):
                window_entities = []
                for j in range(window_size):
                    window_entities.extend([entity[0] for entity in time_entities[i+j][1]])
                
                # 在时间窗口内寻找频繁共现的实体对
                entity_pairs = list(itertools.combinations(set(window_entities), 2))
                
                for entity1, entity2 in entity_pairs:
                    if entity1 in window_entities and entity2 in window_entities:
                        count = window_entities.count(entity1) + window_entities.count(entity2)
                        if count >= 2:  # 至少出现2次
                            relation = RelationInstance(
                                source_entity=entity1,
                                target_entity=entity2,
                                relation_type=RelationType.TEMPORAL,
                                confidence=min(1.0, count / window_size),
                                support_count=count,
                                context=f"temporal_co_occurrence",
                                extracted_at=datetime.now(),
                                metadata={'time_window': window_size}
                            )
                            relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"时间关系挖掘失败: {str(e)}")
        
        self.logger.debug(f"时间关系挖掘完成，发现 {len(relations)} 个关系")
        return relations
    
    def _mine_association_rules(self, entities_list: List[List[Tuple[str, str]]]) -> List[AssociationRule]:
        """挖掘关联规则（简化实现）"""
        rules = []
        
        # 将实体列表转换为事务
        transactions = []
        for entities in entities_list:
            transaction = set(entity[0] for entity in entities)
            if len(transaction) > 1:  # 至少包含2个实体
                transactions.append(transaction)
        
        if len(transactions) < 2:
            return rules
        
        # 简化的关联规则挖掘：寻找频繁项集
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # 找出频繁项集
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count / len(transactions) >= self.min_support}
        
        # 生成简单的关联规则：A -> B
        frequent_items_list = list(frequent_items.keys())
        
        for i, item_a in enumerate(frequent_items_list):
            for j, item_b in enumerate(frequent_items_list[i+1:], i+1):
                # 计算 A -> B 的置信度
                ab_count = sum(1 for transaction in transactions 
                             if item_a in transaction and item_b in transaction)
                
                if ab_count >= self.min_support * len(transactions):
                    confidence = ab_count / item_counts[item_a]
                    
                    if confidence >= self.min_confidence:
                        # 计算lift
                        support_ab = ab_count / len(transactions)
                        support_a = item_counts[item_a] / len(transactions)
                        support_b = item_counts[item_b] / len(transactions)
                        
                        lift = support_ab / (support_a * support_b) if support_a * support_b > 0 else 0
                        
                        rule = AssociationRule(
                            antecedent={item_a},
                            consequent={item_b},
                            confidence=confidence,
                            support=support_ab,
                            lift=lift,
                            conviction=confidence / (1 - support_b) if support_b < 1 else float('inf')
                        )
                        rules.append(rule)
        
        self.logger.debug(f"关联规则挖掘完成，生成 {len(rules)} 个规则")
        return rules
    
    def _build_relation_network(self, relations: List[RelationInstance]) -> Dict[str, Dict[str, List[RelationInstance]]]:
        """构建关系网络"""
        network = defaultdict(lambda: defaultdict(list))
        
        for relation in relations:
            network[relation.source_entity][relation.target_entity].append(relation)
        
        return network
    
    def _find_frequent_patterns(self, 
                               network: Dict[str, Dict[str, List[RelationInstance]]], 
                               min_frequency: int, 
                               max_length: int) -> List[List[str]]:
        """寻找频繁模式"""
        # 简化的频繁模式挖掘算法
        patterns = []
        
        # 收集所有路径
        all_paths = []
        for source, targets in network.items():
            for target, relations in targets.items():
                if len(relations) >= min_frequency:
                    all_paths.append([source, target])
        
        # 生成更长的模式
        for length in range(3, max_length + 1):
            for i in range(len(all_paths) - length + 1):
                path = all_paths[i:i+length]
                
                # 检查路径中所有边的频率
                min_edge_freq = min(len(network[path[j]][path[j+1]]) 
                                  for j in range(len(path)-1))
                
                if min_edge_freq >= min_frequency:
                    patterns.append(path)
        
        return patterns
    
    def _create_relation_pattern(self, 
                                pattern: List[str], 
                                relation_type: RelationType, 
                                relations: List[RelationInstance]) -> Optional[RelationPattern]:
        """创建关系模式"""
        try:
            # 收集模式示例
            examples = []
            contexts = []
            
            for i in range(len(pattern) - 1):
                source, target = pattern[i], pattern[i+1]
                related_relations = [r for r in relations 
                                   if r.source_entity == source and r.target_entity == target]
                
                if related_relations:
                    examples.append((source, target))
                    contexts.extend([r.context for r in related_relations])
            
            if not examples:
                return None
            
            # 计算模式统计
            avg_confidence = np.mean([r.confidence for r in relations 
                                    if (r.source_entity, r.target_entity) in examples])
            
            support = len(examples) / len(set(r.relation_type for r in relations))
            lift = avg_confidence / self.min_confidence if self.min_confidence > 0 else 1.0
            
            pattern_id = f"{relation_type.value}_" + "_".join(pattern)
            
            return RelationPattern(
                pattern_id=pattern_id,
                relation_type=relation_type,
                source_types=set(),  # 简化实现
                target_types=set(),  # 简化实现
                confidence=avg_confidence,
                support=support,
                lift=lift,
                contexts=contexts[:10],  # 限制上下文数量
                examples=examples[:5]   # 限制示例数量
            )
            
        except Exception as e:
            self.logger.error(f"创建关系模式失败: {str(e)}")
            return None
    
    def _cosine_similarity(self, entity1: str, entity2: str) -> float:
        """计算余弦相似度"""
        # 简化实现：基于共现关系
        if entity1 == entity2:
            return 1.0
        
        pair1 = tuple(sorted([entity1, entity2]))
        cooccurrence_count = self.entity_cooccurrence.get(pair1, 0)
        
        # 获取各实体的总出现次数
        entity1_count = sum(count for pair, count in self.entity_cooccurrence.items() 
                           if entity1 in pair)
        entity2_count = sum(count for pair, count in self.entity_cooccurrence.items() 
                           if entity2 in pair)
        
        if entity1_count == 0 or entity2_count == 0:
            return 0.0
        
        # 简化的余弦相似度计算
        similarity = cooccurrence_count / np.sqrt(entity1_count * entity2_count)
        return min(1.0, similarity)
    
    def _jaccard_similarity(self, entity1: str, entity2: str) -> float:
        """计算Jaccard相似度"""
        # 获取与两个实体共同出现的其他实体
        neighbors1 = set()
        neighbors2 = set()
        
        for pair, count in self.entity_cooccurrence.items():
            if entity1 in pair:
                neighbors1.update(pair)
            if entity2 in pair:
                neighbors2.update(pair)
        
        neighbors1.discard(entity1)
        neighbors2.discard(entity2)
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _pearson_similarity(self, entity1: str, entity2: str) -> float:
        """计算Pearson相似度"""
        # 构建实体共现向量
        all_entities = set()
        for pair in self.entity_cooccurrence.keys():
            all_entities.update(pair)
        
        entity1_vector = []
        entity2_vector = []
        
        for entity in sorted(all_entities):
            # 计算实体在该实体对中的共现强度
            pair1 = tuple(sorted([entity1, entity]))
            pair2 = tuple(sorted([entity2, entity]))
            
            entity1_vector.append(self.entity_cooccurrence.get(pair1, 0))
            entity2_vector.append(self.entity_cooccurrence.get(pair2, 0))
        
        if len(entity1_vector) < 2:
            return 0.0
        
        try:
            correlation, _ = pearsonr(entity1_vector, entity2_vector)
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _export_json(self) -> Dict[str, Any]:
        """导出JSON格式"""
        return {
            'relation_instances': [
                {
                    'source_entity': rel.source_entity,
                    'target_entity': rel.target_entity,
                    'relation_type': rel.relation_type.value,
                    'confidence': rel.confidence,
                    'support_count': rel.support_count,
                    'context': rel.context,
                    'extracted_at': rel.extracted_at.isoformat(),
                    'metadata': rel.metadata
                }
                for rel in self.relation_instances
            ],
            'relation_patterns': {
                pattern_id: {
                    'pattern_id': pattern.pattern_id,
                    'relation_type': pattern.relation_type.value,
                    'confidence': pattern.confidence,
                    'support': pattern.support,
                    'lift': pattern.lift,
                    'contexts': pattern.contexts,
                    'examples': pattern.examples
                }
                for pattern_id, pattern in self.relation_patterns.items()
            },
            'statistics': self.get_statistics()
        }
    
    def _export_csv(self) -> str:
        """导出CSV格式"""
        lines = ['source_entity,target_entity,relation_type,confidence,support_count,context,extracted_at']
        
        for rel in self.relation_instances:
            line = f"{rel.source_entity},{rel.target_entity},{rel.relation_type.value},{rel.confidence},{rel.support_count},{rel.context},{rel.extracted_at.isoformat()}"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _export_graph(self) -> Dict[str, Any]:
        """导出图格式"""
        nodes = set()
        edges = []
        
        for rel in self.relation_instances:
            nodes.add(rel.source_entity)
            nodes.add(rel.target_entity)
            edges.append({
                'source': rel.source_entity,
                'target': rel.target_entity,
                'type': rel.relation_type.value,
                'weight': rel.confidence
            })
        
        return {
            'nodes': list(nodes),
            'edges': edges,
            'statistics': self.get_statistics()
        }