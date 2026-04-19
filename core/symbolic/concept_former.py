"""
概念形成主类 - ConceptFormer
Concept Formation Main Class

这个类实现了从感知经验中形成抽象概念的核心机制，整合了抽象引擎、
相似度计算、继承和组合等多个组件。本实现基于现代认知科学的概念形成理论：

理论基础：
- 概念形成理论（Concept Formation Theory）
- 原型理论（Prototype Theory）
- 特征列表理论（Feature List Theory）
- 范畴化理论（Categorization Theory）
- 典型效应（Typicality Effects）
- 范畴边界模糊性（Category Boundary Fuzziness）
- 概念层次理论（Concept Hierarchical Theory）

技术特点：
- 多层次概念抽象（从具体到抽象）
- 实时概念形成和修改
- 概念相似度和关联度计算
- 概念继承和组合机制
- 支持概念的可视化和解释
- 高性能概念检索和索引

Author: NeuroMinecraft Genesis Team
Date: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
import threading
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import pickle
import weakref
import hashlib

logger = logging.getLogger(__name__)


class ConceptLevel(Enum):
    """概念层次枚举
    
    基于认知科学的概念层次理论：
    - 实例层次：具体的个体感知
    - 基本层次：人类最常用的概念层次
    - 上位层次：更抽象的分类概念
    - 元概念：关于概念的抽象概念
    """
    INSTANCE = "instance"              # 实例层次：具体个体感知
    BASIC = "basic"                    # 基本层次：最常用的概念层次
    SUPERORDINATE = "superordinate"    # 上位层次：抽象分类概念
    METACONCEPT = "metaconcept"        # 元概念：关于概念的抽象概念


class FormationMethod(Enum):
    """概念形成方法枚举
    
    基于不同的认知学习机制：
    """
    INDUCTIVE = "inductive"            # 归纳推理：从具体到一般
    DEDUCTIVE = "deductive"            # 演绎推理：从一般到具体
    ANALOGICAL = "analogical"          # 类比推理：基于相似性
    PROTOTYPICAL = "prototypical"      # 原型形成：基于典型性
    RULE_BASED = "rule_based"          # 规则形成：基于逻辑规则
    CONNECTIVE = "connective"          # 连接形成：基于关系网络
    TEMPORAL = "temporal"              # 时序形成：基于时间序列
    SPATIAL = "spatial"                # 空间形成：基于空间结构


@dataclass
class ConceptFeature:
    """概念特征数据类
    
    表示一个概念的单一特征，包含值、权重和不确定性
    """
    name: str
    value: Any
    weight: float = 1.0
    confidence: float = 1.0
    source: str = "unknown"           # 特征来源
    feature_type: str = "attribute"   # 特征类型
    abstractable: bool = True         # 是否可抽象
    stable: bool = True               # 是否稳定特征
    
    def update_value(self, new_value: Any, new_confidence: float = 1.0):
        """更新特征值"""
        if isinstance(self.value, (int, float)) and isinstance(new_value, (int, float)):
            # 数值特征使用加权平均
            total_weight = self.weight * self.confidence
            new_weight = new_confidence
            combined_weight = total_weight + new_weight
            
            self.value = (self.value * total_weight + new_value * new_weight) / combined_weight
            self.confidence = min(1.0, combined_weight / (self.weight + new_weight))
        else:
            # 非数值特征使用置信度加权
            if new_confidence > self.confidence:
                self.value = new_value
                self.confidence = new_confidence
    
    def compute_similarity(self, other_feature: 'ConceptFeature') -> float:
        """计算特征相似度"""
        if self.feature_type != other_feature.feature_type:
            return 0.0
        
        if self.value == other_feature.value:
            return self.confidence * other_feature.confidence
        
        # 数值特征相似度
        if isinstance(self.value, (int, float)) and isinstance(other_feature.value, (int, float)):
            max_val = max(abs(self.value), abs(other_feature.value))
            if max_val > 0:
                diff = abs(self.value - other_feature.value) / max_val
                return max(0.0, 1.0 - diff) * self.confidence * other_feature.confidence
        
        return 0.0


@dataclass
class Concept:
    """概念数据结构
    
    基于原型理论的完整概念表示，包含特征、实例、关系和认知指标
    """
    concept_id: str
    name: str
    level: ConceptLevel
    
    # 认知特征存储
    core_features: Dict[str, ConceptFeature] = field(default_factory=dict)      # 核心特征
    peripheral_features: Dict[str, ConceptFeature] = field(default_factory=dict)  # 边缘特征
    prototype_features: Dict[str, ConceptFeature] = field(default_factory=dict)  # 原型特征
    
    # 实例管理
    instances: Dict[str, Dict[str, Any]] = field(default_factory=dict)           # 具体实例
    instance_count: int = 0
    active_instances: Set[str] = field(default_factory=set)                      # 活跃实例
    
    # 结构关系（基于关系网络理论）
    parent_concepts: Set[str] = field(default_factory=set)                      # 父概念
    child_concepts: Set[str] = field(default_factory=set)                       # 子概念
    similar_concepts: Dict[str, float] = field(default_factory=dict)            # 相似概念及相似度
    related_concepts: Dict[str, str] = field(default_factory=dict)              # 相关概念及关系类型
    
    # 认知心理学指标
    confidence: float = 0.0                # 概念置信度
    typicality: float = 0.0                # 原型典型性
    clarity: float = 0.0                   # 概念清晰度
    stability: float = 0.0                 # 概念稳定性
    abstractness: float = 0.0              # 抽象程度
    complexity: float = 0.0                # 概念复杂度
    
    # 使用统计
    frequency: int = 0                     # 使用频率
    activation_count: int = 0              # 激活次数
    last_accessed: float = field(default_factory=time.time)
    
    # 形成信息
    formation_method: FormationMethod = FormationMethod.INDUCTIVE
    formation_time: float = field(default_factory=time.time)
    formation_examples: List[str] = field(default_factory=list)                 # 形成依据的例子
    evidence_strength: float = 0.0                                         # 证据强度
    
    # 元数据
    created_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 锁机制（支持并发访问）
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        # 自动生成ID
        if not self.concept_id:
            self.concept_id = str(uuid.uuid4())
        
        # 计算初始认知指标
        self._compute_cognitive_metrics()
    
    def add_instance(self, instance_id: str, features: Dict[str, Any]) -> bool:
        """添加具体实例
        
        Args:
            instance_id: 实例标识符
            features: 实例特征
            
        Returns:
            是否成功添加
        """
        with self._lock:
            self.instances[instance_id] = features.copy()
            self.instance_count += 1
            
            # 更新统计信息
            self.frequency += 1
            self.last_accessed = time.time()
            self.activation_count += 1
            
            # 更新概念指标
            self._update_concept_metrics()
            self.last_updated = time.time()
            
            return True
    
    def remove_instance(self, instance_id: str) -> bool:
        """移除实例"""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                self.instance_count -= 1
                self.active_instances.discard(instance_id)
                self.last_updated = time.time()
                self._update_concept_metrics()
                return True
            return False
    
    def add_feature(self, feature: ConceptFeature, is_core: bool = True) -> None:
        """添加概念特征
        
        Args:
            feature: 概念特征
            is_core: 是否为核心特征
        """
        with self._lock:
            target_dict = self.core_features if is_core else self.peripheral_features
            
            if feature.name in target_dict:
                # 合并特征
                existing = target_dict[feature.name]
                existing.update_value(feature.value, feature.confidence)
            else:
                # 添加新特征
                target_dict[feature.name] = feature
            
            self.last_updated = time.time()
            self._update_concept_metrics()
    
    def compute_prototype(self) -> Dict[str, Any]:
        """计算概念原型
        
        基于原型理论，计算该概念最典型的表示
        """
        if not self.instances:
            return {}
        
        prototype = {}
        
        # 收集所有特征值
        all_features = defaultdict(list)
        for instance in self.instances.values():
            for key, value in instance.items():
                all_features[key].append(value)
        
        # 计算每个特征的原值（最常见的值或平均值）
        for feature_name, values in all_features.items():
            if not values:
                continue
            
            # 数值特征使用平均值
            if all(isinstance(v, (int, float)) for v in values):
                prototype[feature_name] = np.mean(values)
            else:
                # 分类特征使用众数
                from collections import Counter
                value_counts = Counter(values)
                prototype[feature_name] = value_counts.most_common(1)[0][0]
        
        return prototype
    
    def compute_typicality(self, instance_features: Dict[str, Any]) -> float:
        """计算实例的典型性
        
        基于实例与原型的相似度计算其典型性
        """
        if not self.instances:
            return 0.0
        
        prototype = self.compute_prototype()
        if not prototype:
            return 0.0
        
        # 计算特征匹配度
        matched_features = 0
        total_features = len(prototype)
        
        for feature_name, typical_value in prototype.items():
            if feature_name in instance_features:
                instance_value = instance_features[feature_name]
                
                if isinstance(typical_value, (int, float)) and isinstance(instance_value, (int, float)):
                    # 数值特征相似度
                    max_val = max(abs(typical_value), abs(instance_value))
                    if max_val > 0:
                        similarity = 1.0 - abs(typical_value - instance_value) / max_val
                        if similarity > 0.7:  # 相似度阈值
                            matched_features += 1
                elif typical_value == instance_value:
                    # 分类特征匹配
                    matched_features += 1
        
        return matched_features / total_features if total_features > 0 else 0.0
    
    def _compute_cognitive_metrics(self) -> None:
        """计算认知指标"""
        # 计算抽象程度
        self._compute_abstractness()
        
        # 计算复杂度
        self._compute_complexity()
        
        # 计算稳定性
        self._compute_stability()
        
        # 计算清晰度
        self._compute_clarity()
    
    def _compute_abstractness(self) -> None:
        """计算概念抽象程度"""
        # 实例数量因子
        instance_factor = 1.0 / (1.0 + len(self.instances))
        
        # 层次因子
        level_factor = {
            ConceptLevel.INSTANCE: 0.1,
            ConceptLevel.BASIC: 0.5,
            ConceptLevel.SUPERORDINATE: 0.8,
            ConceptLevel.METACONCEPT: 1.0
        }.get(self.level, 0.5)
        
        # 关系复杂度因子
        relation_factor = min(len(self.parent_concepts) + len(self.child_concepts) / 10.0, 1.0)
        
        self.abstractness = (1 - instance_factor) * level_factor * (0.5 + 0.5 * relation_factor)
    
    def _compute_complexity(self) -> None:
        """计算概念复杂度"""
        feature_count = len(self.core_features) + len(self.peripheral_features)
        relation_count = len(self.parent_concepts) + len(self.child_concepts)
        
        # 使用特征提取方法计算实例方差
        instance_variance = 0
        if self.instances:
            instance_features = []
            for instance in self.instances.values():
                features = self._extract_numerical_features(instance)
                if features:
                    # 使用特征向量的平均值作为该实例的代表值
                    instance_features.append(np.mean(features))
            
            if instance_features:
                instance_variance = np.var(instance_features)
        
        # 综合复杂度指标
        self.complexity = min(1.0, (
            feature_count / 10.0 +
            relation_count / 5.0 +
            instance_variance / 100.0
        ))
    
    def _compute_stability(self) -> None:
        """计算概念稳定性"""
        if self.frequency == 0:
            self.stability = 0.0
            return
        
        # 基于更新频率和访问频率的稳定性计算
        update_frequency = self.instance_count / max(1, time.time() - self.created_time)
        access_frequency = self.frequency / max(1, time.time() - self.created_time)
        
        # 稳定性 = 访问频率 / (更新频率 + 访问频率)
        self.stability = access_frequency / (update_frequency + access_frequency + 1e-6)
    
    def _compute_clarity(self) -> None:
        """计算概念清晰度"""
        # 基于特征一致性和实例相似度
        if not self.instances:
            self.clarity = 0.0
            return
        
        # 计算实例间的平均相似度
        similarities = []
        instance_list = list(self.instances.values())
        
        for i in range(len(instance_list)):
            for j in range(i + 1, len(instance_list)):
                similarity = self._compute_instance_similarity(instance_list[i], instance_list[j])
                similarities.append(similarity)
        
        if similarities:
            self.clarity = np.mean(similarities)
        else:
            self.clarity = 1.0
    
    def _extract_numerical_features(self, perception_data: Dict[str, Any]) -> List[float]:
        """从感知数据中提取数值特征向量
        
        将包含字符串、数字等混合类型的感知字典转换为数值特征向量
        用于统计计算和相似度分析
        """
        features = []
        
        for key, value in perception_data.items():
            if isinstance(value, (int, float)):
                # 数值特征直接添加
                features.append(float(value))
            elif isinstance(value, str):
                # 字符串特征转换为数字编码
                # 使用简单哈希编码确保一致性
                hash_val = hash(value) % 1000  # 映射到 0-999
                features.append(hash_val / 1000.0)  # 归一化到 0-1
            elif isinstance(value, bool):
                # 布尔值转换为数值
                features.append(1.0 if value else 0.0)
            elif value is None:
                # None值设为0
                features.append(0.0)
            else:
                # 其他类型尝试转换，失败则设为0
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    hash_val = hash(str(value)) % 1000
                    features.append(hash_val / 1000.0)
        
        return features

    def _compute_instance_similarity(self, inst1: Dict[str, Any], inst2: Dict[str, Any]) -> float:
        """计算实例间相似度"""
        common_features = set(inst1.keys()) & set(inst2.keys())
        if not common_features:
            return 0.0
        
        total_similarity = 0.0
        for feature in common_features:
            val1, val2 = inst1[feature], inst2[feature]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    total_similarity += similarity
            elif val1 == val2:
                total_similarity += 1.0
            else:
                # 对于不同类型的值，使用特征提取后的相似度
                features1 = self._extract_numerical_features({feature: val1})
                features2 = self._extract_numerical_features({feature: val2})
                if features1 and features2:
                    # 计算第一个特征的相似度
                    max_val = max(abs(features1[0]), abs(features2[0]))
                    if max_val > 0:
                        similarity = 1.0 - abs(features1[0] - features2[0]) / max_val
                        total_similarity += similarity
        
        return total_similarity / len(common_features) if common_features else 0.0
    
    def _update_concept_metrics(self) -> None:
        """更新概念指标"""
        self._compute_cognitive_metrics()
        self.version += 1
    
    def activate(self) -> None:
        """激活概念"""
        with self._lock:
            self.activation_count += 1
            self.last_accessed = time.time()
    
    def deactivate(self) -> None:
        """去激活概念"""
        # 可用于清理缓存或执行清理操作
        pass
    
    def get_activation_strength(self, context_features: Dict[str, Any] = None) -> float:
        """计算概念激活强度
        
        Args:
            context_features: 上下文特征
            
        Returns:
            激活强度 [0, 1]
        """
        base_activation = min(1.0, self.frequency / 10.0)
        
        # 如果提供了上下文，计算上下文相关度
        if context_features:
            context_relevance = self._compute_context_relevance(context_features)
            return base_activation * (0.5 + 0.5 * context_relevance)
        
        return base_activation
    
    def _compute_context_relevance(self, context_features: Dict[str, Any]) -> float:
        """计算上下文相关性"""
        if not self.core_features:
            return 0.0
        
        relevant_features = 0
        total_features = len(self.core_features)
        
        for feature_name, concept_feature in self.core_features.items():
            if feature_name in context_features:
                context_value = context_features[feature_name]
                similarity = concept_feature.compute_similarity(
                    ConceptFeature(name=feature_name, value=context_value)
                )
                if similarity > 0.5:
                    relevant_features += 1
        
        return relevant_features / total_features if total_features > 0 else 0.0
    
    def merge_with(self, other_concept: 'Concept') -> 'Concept':
        """合并两个概念
        
        创建一个新概念，包含两个概念的合并信息
        """
        # 创建新的合并概念
        merged_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"{self.name}_&_{other_concept.name}",
            level=max(self.level, other_concept.level, key=lambda x: x.value),
            formation_method=FormationMethod.INDUCTIVE
        )
        
        # 合并核心特征
        all_core_features = set(self.core_features.keys()) | set(other_concept.core_features.keys())
        for feature_name in all_core_features:
            if feature_name in self.core_features and feature_name in other_concept.core_features:
                # 特征冲突，使用置信度更高的
                if self.core_features[feature_name].confidence >= other_concept.core_features[feature_name].confidence:
                    merged_concept.core_features[feature_name] = self.core_features[feature_name]
                else:
                    merged_concept.core_features[feature_name] = other_concept.core_features[feature_name]
            elif feature_name in self.core_features:
                merged_concept.core_features[feature_name] = self.core_features[feature_name]
            else:
                merged_concept.core_features[feature_name] = other_concept.core_features[feature_name]
        
        # 合并实例
        merged_concept.instances.update(self.instances)
        merged_concept.instances.update(other_concept.instances)
        merged_concept.instance_count = len(merged_concept.instances)
        
        # 合并关系
        merged_concept.parent_concepts = self.parent_concepts | other_concept.parent_concepts
        merged_concept.child_concepts = self.child_concepts | other_concept.child_concepts
        
        # 合并相似概念
        for concept_id, similarity in self.similar_concepts.items():
            if concept_id in other_concept.similar_concepts:
                merged_concept.similar_concepts[concept_id] = max(similarity, other_concept.similar_concepts[concept_id])
            else:
                merged_concept.similar_concepts[concept_id] = similarity
        
        merged_concept.similar_concepts.update(other_concept.similar_concepts)
        
        # 计算指标
        merged_concept._update_concept_metrics()
        merged_concept.evidence_strength = (self.evidence_strength + other_concept.evidence_strength) / 2
        
        return merged_concept
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化"""
        return {
            'concept_id': self.concept_id,
            'name': self.name,
            'level': self.level.value,
            'formation_method': self.formation_method.value,
            
            # 特征数据
            'core_features': {name: {
                'name': feat.name,
                'value': feat.value,
                'weight': feat.weight,
                'confidence': feat.confidence,
                'source': feat.source,
                'feature_type': feat.feature_type,
                'abstractable': feat.abstractable,
                'stable': feat.stable
            } for name, feat in self.core_features.items()},
            
            'peripheral_features': {name: {
                'name': feat.name,
                'value': feat.value,
                'weight': feat.weight,
                'confidence': feat.confidence,
                'source': feat.source,
                'feature_type': feat.feature_type,
                'abstractable': feat.abstractable,
                'stable': feat.stable
            } for name, feat in self.peripheral_features.items()},
            
            # 实例数据
            'instances': self.instances,
            'instance_count': self.instance_count,
            
            # 关系
            'parent_concepts': list(self.parent_concepts),
            'child_concepts': list(self.child_concepts),
            'similar_concepts': self.similar_concepts,
            'related_concepts': self.related_concepts,
            
            # 认知指标
            'confidence': self.confidence,
            'typicality': self.typicality,
            'clarity': self.clarity,
            'stability': self.stability,
            'abstractness': self.abstractness,
            'complexity': self.complexity,
            
            # 使用统计
            'frequency': self.frequency,
            'activation_count': self.activation_count,
            'last_accessed': self.last_accessed,
            
            # 形成信息
            'formation_examples': self.formation_examples,
            'evidence_strength': self.evidence_strength,
            
            # 元数据
            'created_time': self.created_time,
            'last_updated': self.last_updated,
            'version': self.version,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """从字典创建概念对象"""
        # 重建核心特征
        core_features = {}
        for name, feat_data in data.get('core_features', {}).items():
            core_features[name] = ConceptFeature(
                name=feat_data['name'],
                value=feat_data['value'],
                weight=feat_data.get('weight', 1.0),
                confidence=feat_data.get('confidence', 1.0),
                source=feat_data.get('source', 'unknown'),
                feature_type=feat_data.get('feature_type', 'attribute'),
                abstractable=feat_data.get('abstractable', True),
                stable=feat_data.get('stable', True)
            )
        
        # 重建边缘特征
        peripheral_features = {}
        for name, feat_data in data.get('peripheral_features', {}).items():
            peripheral_features[name] = ConceptFeature(
                name=feat_data['name'],
                value=feat_data['value'],
                weight=feat_data.get('weight', 1.0),
                confidence=feat_data.get('confidence', 1.0),
                source=feat_data.get('source', 'unknown'),
                feature_type=feat_data.get('feature_type', 'attribute'),
                abstractable=feat_data.get('abstractable', True),
                stable=feat_data.get('stable', True)
            )
        
        # 创建概念对象
        concept = cls(
            concept_id=data['concept_id'],
            name=data['name'],
            level=ConceptLevel(data['level']),
            formation_method=FormationMethod(data.get('formation_method', 'inductive')),
            core_features=core_features,
            peripheral_features=peripheral_features,
            instances=data.get('instances', {}),
            instance_count=data.get('instance_count', 0),
            parent_concepts=set(data.get('parent_concepts', [])),
            child_concepts=set(data.get('child_concepts', [])),
            similar_concepts=data.get('similar_concepts', {}),
            related_concepts=data.get('related_concepts', {}),
            confidence=data.get('confidence', 0.0),
            typicality=data.get('typicality', 0.0),
            clarity=data.get('clarity', 0.0),
            stability=data.get('stability', 0.0),
            abstractness=data.get('abstractness', 0.0),
            complexity=data.get('complexity', 0.0),
            frequency=data.get('frequency', 0),
            activation_count=data.get('activation_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            formation_examples=data.get('formation_examples', []),
            evidence_strength=data.get('evidence_strength', 0.0),
            created_time=data.get('created_time', time.time()),
            last_updated=data.get('last_updated', time.time()),
            version=data.get('version', 1),
            metadata=data.get('metadata', {})
        )
        
        return concept


class ConceptIndex:
    """概念索引系统
    
    提供高性能的概念检索和匹配功能，支持多种索引策略
    """
    
    def __init__(self):
        self.name_index: Dict[str, Set[str]] = defaultdict(set)           # 名称索引
        self.feature_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))  # 特征索引
        self.level_index: Dict[ConceptLevel, Set[str]] = defaultdict(set)  # 层次索引
        self.active_index: Set[str] = set()                               # 活跃概念索引
        self.similarity_index: Dict[str, Dict[str, float]] = defaultdict(dict)  # 相似度索引
    
    def add_concept(self, concept: Concept) -> None:
        """添加概念到索引"""
        # 名称索引
        self.name_index[concept.name.lower()].add(concept.concept_id)
        
        # 特征索引
        for feature_name, feature in concept.core_features.items():
            self.feature_index[feature_name][feature.value].add(concept.concept_id)
        
        # 层次索引
        self.level_index[concept.level].add(concept.concept_id)
        
        # 活跃索引
        if concept.activation_count > 0:
            self.active_index.add(concept.concept_id)
    
    def remove_concept(self, concept_id: str, concept: Concept) -> None:
        """从索引移除概念"""
        # 名称索引
        self.name_index[concept.name.lower()].discard(concept_id)
        
        # 特征索引
        for feature_name, feature in concept.core_features.items():
            if concept_id in self.feature_index[feature_name][feature.value]:
                self.feature_index[feature_name][feature.value].discard(concept_id)
        
        # 层次索引
        self.level_index[concept.level].discard(concept_id)
        
        # 活跃索引
        self.active_index.discard(concept_id)
    
    def search_by_name(self, name: str) -> Set[str]:
        """按名称搜索概念"""
        name_lower = name.lower()
        return self.name_index[name_lower]
    
    def search_by_feature(self, feature_name: str, feature_value: Any) -> Set[str]:
        """按特征搜索概念"""
        return self.feature_index[feature_name].get(feature_value, set())
    
    def search_by_level(self, level: ConceptLevel) -> Set[str]:
        """按层次搜索概念"""
        return self.level_index[level]
    
    def get_active_concepts(self) -> Set[str]:
        """获取活跃概念"""
        return self.active_index.copy()
    
    def update_similarity(self, concept_id: str, similar_id: str, similarity: float) -> None:
        """更新相似度索引"""
        self.similarity_index[concept_id][similar_id] = similarity
        self.similarity_index[similar_id][concept_id] = similarity


class PerformanceMonitor:
    """性能监控器
    
    监控概念形成系统的性能指标
    """
    
    def __init__(self):
        self.formation_times: deque = deque(maxlen=1000)    # 概念形成时间
        self.search_times: deque = deque(maxlen=1000)       # 搜索时间
        self.memory_usage: List[float] = []                 # 内存使用
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.active_operations: int = 0
        
        # 性能阈值
        self.max_formation_time = 1.0  # 秒
        self.max_search_time = 0.1     # 秒
    
    def record_formation_time(self, formation_time: float) -> None:
        """记录概念形成时间"""
        self.formation_times.append(formation_time)
        
        if formation_time > self.max_formation_time:
            logger.warning(f"概念形成时间过长: {formation_time:.3f}s")
    
    def record_search_time(self, search_time: float) -> None:
        """记录搜索时间"""
        self.search_times.append(search_time)
        
        if search_time > self.max_search_time:
            logger.warning(f"搜索时间过长: {search_time:.3f}s")
    
    def get_average_formation_time(self) -> float:
        """获取平均概念形成时间"""
        return np.mean(self.formation_times) if self.formation_times else 0.0
    
    def get_average_search_time(self) -> float:
        """获取平均搜索时间"""
        return np.mean(self.search_times) if self.search_times else 0.0
    
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ConceptFormer:
    """概念形成主类
    
    负责从感知经验中形成、管理和操作概念，整合了认知科学理论
    和高性能计算技术，提供完整的概念形成机制。
    
    主要功能：
    - 多层次概念形成和管理
    - 高性能概念检索和匹配
    - 概念关系的动态更新
    - 概念合并和分化操作
    - 性能监控和优化
    - 并发安全支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化概念形成器
        
        Args:
            config: 配置参数
                - max_concepts: 最大概念数量
                - cache_size: 缓存大小
                - performance_monitoring: 是否启用性能监控
                - concurrent_safety: 是否启用并发安全
                - similarity_threshold: 相似度阈值
        """
        self.config = config or {}
        
        # 核心存储
        self.concepts: Dict[str, Concept] = {}
        self.concept_index = ConceptIndex()
        self.performance_monitor = PerformanceMonitor()
        
        # 配置参数
        self.max_concepts = self.config.get('max_concepts', 10000)
        self.cache_size = self.config.get('cache_size', 1000)
        self.enable_monitoring = self.config.get('performance_monitoring', True)
        self.concurrent_safety = self.config.get('concurrent_safety', True)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # 缓存系统
        self.search_cache: Dict[str, Tuple[Set[str], float]] = {}  # 搜索缓存
        self.similarity_cache: Dict[Tuple[str, str], float] = {}   # 相似度缓存
        
        # 统计信息
        self.stats = {
            'total_concepts': 0,
            'formation_operations': 0,
            'search_operations': 0,
            'merge_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 线程锁（如果启用并发安全）
        self._main_lock = threading.RLock() if self.concurrent_safety else None
        
        # 定时任务
        self.cleanup_interval = self.config.get('cleanup_interval', 3600)  # 秒
        self.last_cleanup = time.time()
        
        logger.info(f"ConceptFormer 初始化完成 - 配置: {self.config}")
    
    def form_concept(self,
                    name: str,
                    perceptions: List[Dict[str, Any]],
                    level: ConceptLevel = ConceptLevel.BASIC,
                    method: FormationMethod = FormationMethod.INDUCTIVE,
                    features: Optional[Dict[str, ConceptFeature]] = None,
                    min_evidence_strength: float = 0.1) -> str:
        """从感知经验中形成概念
        
        这是概念形成的核心方法，基于认知科学的概念形成理论实现：
        
        Args:
            name: 概念名称
            perceptions: 感知经验列表
            level: 概念层次
            method: 形成方法
            features: 预定义的特征
            min_evidence_strength: 最小证据强度阈值
            
        Returns:
            形成的概念ID
            
        Raises:
            ValueError: 如果输入参数无效
            RuntimeError: 如果系统资源不足
        """
        start_time = time.time()
        
        with self._get_lock():
            try:
                # 输入验证
                if not name or not name.strip():
                    raise ValueError("概念名称不能为空")
                
                if not perceptions:
                    raise ValueError("感知经验不能为空")
                
                if len(perceptions) < 2 and method == FormationMethod.INDUCTIVE:
                    logger.warning(f"形成归纳概念 '{name}' 的感知经验过少，建议至少2个")
                
                # 检查概念数量限制
                if len(self.concepts) >= self.max_concepts:
                    raise RuntimeError(f"概念数量已达到上限 {self.max_concepts}")
                
                logger.info(f"开始形成概念: {name} (层次: {level.value}, 方法: {method.value})")
                
                # 提取感知特征
                extracted_features = self._extract_features_from_perceptions(perceptions)
                
                # 创建新概念
                concept_id = self._create_concept(
                    name=name,
                    level=level,
                    method=method,
                    extracted_features=extracted_features,
                    predefined_features=features or {}
                )
                
                # 添加感知实例
                for i, perception in enumerate(perceptions):
                    instance_id = f"{concept_id}_instance_{i}"
                    success = self.concepts[concept_id].add_instance(instance_id, perception)
                    if not success:
                        logger.warning(f"添加实例 {instance_id} 失败")
                
                # 计算证据强度
                evidence_strength = self._compute_evidence_strength(perceptions)
                self.concepts[concept_id].evidence_strength = evidence_strength
                
                # 检查证据强度阈值
                if evidence_strength < min_evidence_strength:
                    logger.warning(f"概念 '{name}' 的证据强度 ({evidence_strength:.3f}) 低于阈值 ({min_evidence_strength})")
                
                # 建立概念关系
                self._establish_concept_relations(concept_id)
                
                # 更新索引
                self.concept_index.add_concept(self.concepts[concept_id])
                
                # 更新统计
                self._update_formation_statistics()
                
                # 性能监控
                if self.enable_monitoring:
                    formation_time = time.time() - start_time
                    self.performance_monitor.record_formation_time(formation_time)
                
                logger.info(f"成功形成概念: {name} ({concept_id}) - 证据强度: {evidence_strength:.3f}")
                return concept_id
                
            except Exception as e:
                logger.error(f"形成概念失败: {name} - {str(e)}")
                raise
    
    def _create_concept(self,
                       name: str,
                       level: ConceptLevel,
                       method: FormationMethod,
                       extracted_features: Dict[str, ConceptFeature],
                       predefined_features: Dict[str, ConceptFeature]) -> str:
        """创建新概念"""
        
        # 生成唯一ID
        concept_id = str(uuid.uuid4())
        
        # 合并特征（预定义特征优先）
        final_features = extracted_features.copy()
        final_features.update(predefined_features)
        
        # 创建概念对象
        concept = Concept(
            concept_id=concept_id,
            name=name,
            level=level,
            formation_method=method
        )
        
        # 添加特征到概念
        for feature_name, feature in final_features.items():
            concept.add_feature(feature, is_core=True)
        
        # 存储概念
        self.concepts[concept_id] = concept
        
        return concept_id
    
    def _extract_features_from_perceptions(self, 
                                         perceptions: List[Dict[str, Any]]) -> Dict[str, ConceptFeature]:
        """从感知经验中提取特征
        
        基于特征列表理论，从具体的感知经验中提取共同特征
        """
        if not perceptions:
            return {}
        
        # 收集所有特征值
        feature_values = defaultdict(list)
        feature_occurrences = defaultdict(int)
        
        for perception in perceptions:
            if not isinstance(perception, dict):
                continue
            
            for feature_name, feature_value in perception.items():
                feature_values[feature_name].append(feature_value)
                feature_occurrences[feature_name] += 1
        
        # 计算特征统计
        extracted_features = {}
        total_perceptions = len(perceptions)
        
        for feature_name, values in feature_values.items():
            occurrence_count = feature_occurrences[feature_name]
            occurrence_rate = occurrence_count / total_perceptions
            
            # 跳过低频特征（基于认知负荷理论）
            if occurrence_rate < 0.1:  # 至少10%的感知中出现
                continue
            
            # 创建概念特征
            if all(isinstance(v, (int, float)) for v in values):
                # 数值特征：计算统计指标
                mean_value = np.mean(values)
                std_value = np.std(values)
                feature_value = {
                    'mean': mean_value,
                    'std': std_value,
                    'min': np.min(values),
                    'max': np.max(values)
                }
                confidence = 1.0 / (1.0 + std_value) if std_value > 0 else 1.0
            else:
                # 分类特征：计算众数和频率
                from collections import Counter
                value_counts = Counter(values)
                mode_value = value_counts.most_common(1)[0][0]
                mode_frequency = value_counts[mode_value] / len(values)
                
                feature_value = mode_value
                confidence = mode_frequency
            
            # 创建概念特征对象
            extracted_features[feature_name] = ConceptFeature(
                name=feature_name,
                value=feature_value,
                weight=occurrence_rate,
                confidence=confidence,
                source="perceptual_extraction",
                feature_type="extracted",
                abstractable=True,
                stable=occurrence_rate > 0.8  # 80%以上出现认为是稳定特征
            )
        
        return extracted_features
    
    def _compute_evidence_strength(self, perceptions: List[Dict[str, Any]]) -> float:
        """计算证据强度
        
        基于统计理论和认知心理学，评估概念形成的证据质量
        """
        if not perceptions:
            return 0.0
        
        # 实例数量因子
        instance_factor = min(1.0, len(perceptions) / 20.0)
        
        # 特征一致性因子
        feature_consistency = self._compute_feature_consistency(perceptions)
        
        # 信息丰富度因子
        information_richness = self._compute_information_richness(perceptions)
        
        # 综合证据强度
        evidence_strength = (
            0.4 * instance_factor +
            0.3 * feature_consistency +
            0.3 * information_richness
        )
        
        return min(1.0, evidence_strength)
    
    def _compute_feature_consistency(self, perceptions: List[Dict[str, Any]]) -> float:
        """计算特征一致性"""
        if len(perceptions) < 2:
            return 1.0
        
        # 收集所有特征名
        all_features = set()
        for perception in perceptions:
            all_features.update(perception.keys())
        
        if not all_features:
            return 1.0
        
        total_consistency = 0.0
        
        for feature_name in all_features:
            values = [p.get(feature_name) for p in perceptions if feature_name in p]
            
            if len(values) < 2:
                continue
            
            if all(isinstance(v, (int, float)) for v in values):
                # 数值特征：计算变异系数
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 1.0
                consistency = 1.0 / (1.0 + cv)
            else:
                # 分类特征：计算多样性指数
                from collections import Counter
                value_counts = Counter(values)
                entropy = -sum((count / len(values)) * np.log2(count / len(values)) 
                              for count in value_counts.values())
                max_entropy = np.log2(len(value_counts))
                consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            total_consistency += consistency
        
        return total_consistency / len(all_features)
    
    def _compute_information_richness(self, perceptions: List[Dict[str, Any]]) -> float:
        """计算信息丰富度"""
        if not perceptions:
            return 0.0
        
        # 计算平均特征数
        avg_feature_count = np.mean([len(p) for p in perceptions])
        max_possible_features = 20  # 假设最大特征数
        
        # 计算特征类型多样性
        feature_types = set()
        for perception in perceptions:
            for value in perception.values():
                feature_types.add(type(value).__name__)
        
        type_diversity = len(feature_types) / 5.0  # 假设最多5种类型
        
        return min(1.0, (avg_feature_count / max_possible_features + type_diversity) / 2)
    
    def _establish_concept_relations(self, concept_id: str) -> None:
        """建立概念关系
        
        基于概念层次理论和相似性理论建立概念间的关系
        """
        new_concept = self.concepts[concept_id]
        
        # 查找相似概念
        for existing_id, existing_concept in self.concepts.items():
            if existing_id == concept_id:
                continue
            
            # 计算相似度
            similarity = self._calculate_concept_similarity(concept_id, existing_id)
            
            if similarity > self.similarity_threshold:
                # 建立相似关系
                new_concept.similar_concepts[existing_id] = similarity
                existing_concept.similar_concepts[concept_id] = similarity
                
                # 更新索引
                self.concept_index.update_similarity(concept_id, existing_id, similarity)
                
                # 建立层次关系
                if new_concept.level == existing_concept.level:
                    # 同级概念，可能需要分化
                    pass
                elif new_concept.level.value > existing_concept.level.value:
                    # 新概念更抽象，可能是父概念
                    new_concept.parent_concepts.add(existing_id)
                    existing_concept.child_concepts.add(concept_id)
                else:
                    # 新概念更具体，可能是子概念
                    new_concept.child_concepts.add(existing_id)
                    existing_concept.parent_concepts.add(concept_id)
    
    def _calculate_concept_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """计算概念间相似度
        
        基于多种相似性度量方法的综合相似度计算
        """
        # 检查缓存
        cache_key = tuple(sorted([concept_id1, concept_id2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        concept1 = self.concepts.get(concept_id1)
        concept2 = self.concepts.get(concept_id2)
        
        if not concept1 or not concept2:
            return 0.0
        
        # 特征相似度
        feature_similarity = self._compute_feature_similarity(concept1, concept2)
        
        # 结构相似度
        structural_similarity = self._compute_structural_similarity(concept1, concept2)
        
        # 层次相似度
        hierarchical_similarity = self._compute_hierarchical_similarity(concept1, concept2)
        
        # 语义相似度（基于名称）
        semantic_similarity = self._compute_semantic_similarity(concept1, concept2)
        
        # 综合相似度
        weights = [0.4, 0.3, 0.2, 0.1]  # 特征、结构、层次、语义权重
        similarities = [feature_similarity, structural_similarity, 
                       hierarchical_similarity, semantic_similarity]
        
        overall_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        # 缓存结果
        self.similarity_cache[cache_key] = overall_similarity
        
        return overall_similarity
    
    def _compute_feature_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算特征相似度"""
        common_features = set(concept1.core_features.keys()) & set(concept2.core_features.keys())
        
        if not common_features:
            return 0.0
        
        total_similarity = 0.0
        for feature_name in common_features:
            feat1 = concept1.core_features[feature_name]
            feat2 = concept2.core_features[feature_name]
            similarity = feat1.compute_similarity(feat2)
            total_similarity += similarity * min(feat1.weight, feat2.weight)
        
        return total_similarity / len(common_features)
    
    def _compute_structural_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算结构相似度"""
        # 关系数量相似度
        rel1_count = len(concept1.parent_concepts) + len(concept1.child_concepts)
        rel2_count = len(concept2.parent_concepts) + len(concept2.child_concepts)
        
        max_relations = max(rel1_count, rel2_count, 1)
        relation_similarity = 1.0 - abs(rel1_count - rel2_count) / max_relations
        
        # 复杂度相似度
        complexity_similarity = 1.0 - abs(concept1.complexity - concept2.complexity)
        
        return (relation_similarity + complexity_similarity) / 2
    
    def _compute_hierarchical_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算层次相似度"""
        # 层次距离
        level_distances = {
            (ConceptLevel.INSTANCE, ConceptLevel.BASIC): 1,
            (ConceptLevel.INSTANCE, ConceptLevel.SUPERORDINATE): 2,
            (ConceptLevel.INSTANCE, ConceptLevel.METACONCEPT): 3,
            (ConceptLevel.BASIC, ConceptLevel.SUPERORDINATE): 1,
            (ConceptLevel.BASIC, ConceptLevel.METACONCEPT): 2,
            (ConceptLevel.SUPERORDINATE, ConceptLevel.METACONCEPT): 1
        }
        
        # 转换为有序对
        level_pair = tuple(sorted([concept1.level.value, concept2.level.value], 
                                 key=lambda x: list(ConceptLevel).index(ConceptLevel(x))))
        level_pair = (ConceptLevel(level_pair[0]), ConceptLevel(level_pair[1]))
        
        max_distance = 3  # 最大层次距离
        distance = level_distances.get(level_pair, 0)
        
        return 1.0 - distance / max_distance
    
    def _compute_semantic_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算语义相似度"""
        # 基于名称的简单相似度
        name1_lower = concept1.name.lower()
        name2_lower = concept2.name.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        
        # 计算编辑距离
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(name1_lower, name2_lower)
        max_len = max(len(name1_lower), len(name2_lower))
        
        return 1.0 - distance / max_len if max_len > 0 else 0.0
    
    def _update_formation_statistics(self) -> None:
        """更新形成统计信息"""
        self.stats['formation_operations'] += 1
        self.stats['total_concepts'] = len(self.concepts)
    
    def _get_lock(self):
        """获取锁（如果启用了并发安全）"""
        return self._main_lock if self._main_lock else self
    
    def update_concept(self, 
                      concept_id: str, 
                      new_features: Optional[Dict[str, ConceptFeature]] = None,
                      new_perceptions: Optional[List[Dict[str, Any]]] = None,
                      evidence_threshold: float = 0.1) -> bool:
        """更新现有概念
        
        基于在线学习理论，更新概念以适应新的感知经验
        """
        with self._get_lock():
            if concept_id not in self.concepts:
                logger.warning(f"概念不存在: {concept_id}")
                return False
            
            concept = self.concepts[concept_id]
            
            try:
                # 激活概念
                concept.activate()
                
                # 更新特征
                if new_features:
                    for feature_name, feature in new_features.items():
                        concept.add_feature(feature, is_core=True)
                
                # 添加新感知实例
                if new_perceptions:
                    for perception in new_perceptions:
                        instance_id = f"{concept_id}_instance_{int(time.time())}"
                        concept.add_instance(instance_id, perception)
                    
                    # 重新计算证据强度
                    new_evidence = self._compute_evidence_strength(
                        list(concept.instances.values())
                    )
                    
                    # 检查是否需要概念分化
                    if concept.evidence_strength > 0 and new_evidence < evidence_threshold:
                        logger.info(f"概念 {concept.name} 证据强度显著下降，考虑分化")
                        return self._split_concept(concept_id, new_perceptions)
                
                # 更新统计
                self.stats['formation_operations'] += 1
                
                logger.info(f"成功更新概念: {concept.name}")
                return True
                
            except Exception as e:
                logger.error(f"更新概念失败: {concept_id} - {str(e)}")
                return False
    
    def _split_concept(self, concept_id: str, new_perceptions: List[Dict[str, Any]]) -> bool:
        """概念分化
        
        当概念变得过于异质时，将其分解为更专门的概念
        """
        logger.info(f"开始概念分化: {concept_id}")
        
        # 基于聚类分析进行概念分化
        existing_instances = list(self.concepts[concept_id].instances.values())
        all_instances = existing_instances + new_perceptions
        
        if len(all_instances) < 4:  # 最少需要4个实例进行有意义的分化
            return False
        
        # 简单的特征聚类分化
        clusters = self._cluster_instances(all_instances)
        
        if len(clusters) <= 1:
            return False
        
        # 为每个聚类创建新概念
        original_concept = self.concepts[concept_id]
        new_concept_ids = []
        
        for i, cluster_instances in enumerate(clusters.values()):
            if len(cluster_instances) < 2:  # 忽略太小的聚类
                continue
            
            # 创建分化概念
            split_name = f"{original_concept.name}_类型{i+1}"
            split_id = self.form_concept(
                name=split_name,
                perceptions=cluster_instances,
                level=original_concept.level,
                method=FormationMethod.INDUCTIVE
            )
            
            new_concept_ids.append(split_id)
        
        # 如果成功创建了新概念，删除原概念
        if new_concept_ids:
            del self.concepts[concept_id]
            logger.info(f"概念分化完成: 原概念 {concept_id} -> {len(new_concept_ids)} 个新概念")
            return True
        
        return False
    
    def _cluster_instances(self, instances: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """简单的实例聚类"""
        if len(instances) < 4:
            return {0: instances}
        
        # 收集所有特征
        all_features = set()
        for instance in instances:
            all_features.update(instance.keys())
        
        if not all_features:
            return {0: instances}
        
        # 构建特征向量
        feature_vectors = []
        for instance in instances:
            vector = []
            for feature in sorted(all_features):  # 排序保证一致性
                value = instance.get(feature, 0)
                if isinstance(value, (int, float)):
                    vector.append(value)
                else:
                    # 分类特征使用哈希
                    vector.append(hash(str(value)) % 100)
            feature_vectors.append(vector)
        
        # 简单的K-means聚类（K=2或3）
        n_clusters = min(3, len(instances) // 2)
        clusters = self._kmeans_clustering(feature_vectors, n_clusters)
        
        # 将聚类结果组织为实例列表
        result = {}
        for cluster_id, instance_indices in clusters.items():
            result[cluster_id] = [instances[i] for i in instance_indices]
        
        return result
    
    def _kmeans_clustering(self, vectors: List[List[float]], k: int, max_iter: int = 10) -> Dict[int, List[int]]:
        """简单的K-means聚类算法"""
        if len(vectors) < k:
            return {i: [i] for i in range(len(vectors))}
        
        # 初始化质心
        centroids = vectors[:k]
        
        for _ in range(max_iter):
            # 分配点到最近的质心
            clusters = {i: [] for i in range(k)}
            assignments = []
            
            for i, vector in enumerate(vectors):
                distances = [np.linalg.norm(np.array(vector) - np.array(centroid)) for centroid in centroids]
                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(i)
                assignments.append(closest_centroid)
            
            # 更新质心
            new_centroids = []
            for cluster_points in clusters.values():
                if cluster_points:
                    cluster_vectors = [vectors[i] for i in cluster_points]
                    new_centroid = np.mean(cluster_vectors, axis=0).tolist()
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[len(new_centroids)])
            
            centroids = new_centroids
        
        # 移除空聚类
        result = {}
        for cluster_id, instance_indices in clusters.items():
            if instance_indices:
                result[len(result)] = instance_indices
        
        return result
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """获取概念对象"""
        with self._get_lock():
            concept = self.concepts.get(concept_id)
            if concept:
                concept.activate()  # 激活概念
            return concept
    
    def get_concepts_by_name(self, name: str) -> List[Concept]:
        """根据名称获取概念列表"""
        concepts = []
        for concept in self.concepts.values():
            if name.lower() in concept.name.lower():
                concepts.append(concept)
                concept.activate()
        return concepts
    
    def search_concepts(self, 
                       query: Union[str, Dict[str, Any]], 
                       max_results: int = 10,
                       similarity_threshold: float = None) -> List[Tuple[Concept, float]]:
        """搜索概念
        
        提供多种搜索方式：关键词搜索、特征搜索、相似度搜索
        """
        start_time = time.time()
        
        with self._get_lock():
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold
            
            results = []
            
            if isinstance(query, str):
                # 关键词搜索
                results = self._search_by_keyword(query, similarity_threshold)
            elif isinstance(query, dict):
                # 特征搜索
                results = self._search_by_features(query, similarity_threshold)
            else:
                raise ValueError("搜索查询必须是字符串或字典")
            
            # 排序和限制结果数量
            results.sort(key=lambda x: x[1], reverse=True)
            limited_results = results[:max_results]
            
            # 激活所有结果概念
            for concept, score in limited_results:
                concept.activate()
            
            # 更新统计
            self.stats['search_operations'] += 1
            if self.enable_monitoring:
                search_time = time.time() - start_time
                self.performance_monitor.record_search_time(search_time)
            
            return limited_results
    
    def _search_by_keyword(self, keyword: str, threshold: float) -> List[Tuple[Concept, float]]:
        """基于关键词搜索概念"""
        results = []
        keyword_lower = keyword.lower()
        
        for concept in self.concepts.values():
            score = 0.0
            
            # 名称匹配（高权重）
            if keyword_lower in concept.name.lower():
                score += 1.0
            
            # 特征名称匹配（中权重）
            for feature_name in concept.core_features.keys():
                if keyword_lower in feature_name.lower():
                    score += 0.5
            
            # 特征值匹配（低权重）
            for feature in concept.core_features.values():
                if isinstance(feature.value, str) and keyword_lower in feature.value.lower():
                    score += 0.2
            
            # 元数据匹配（低权重）
            for metadata_value in concept.metadata.values():
                if isinstance(metadata_value, str) and keyword_lower in metadata_value.lower():
                    score += 0.1
            
            if score >= threshold:
                results.append((concept, score))
        
        return results
    
    def _search_by_features(self, feature_query: Dict[str, Any], threshold: float) -> List[Tuple[Concept, float]]:
        """基于特征搜索概念"""
        results = []
        
        for concept in self.concepts.values():
            score = 0.0
            total_features = len(feature_query)
            matched_features = 0
            
            for query_feature, query_value in feature_query.items():
                # 检查概念特征
                if query_feature in concept.core_features:
                    concept_feature = concept.core_features[query_feature]
                    
                    if isinstance(query_value, (int, float)) and isinstance(concept_feature.value, (int, float)):
                        # 数值特征相似度
                        max_val = max(abs(query_value), abs(concept_feature.value))
                        if max_val > 0:
                            similarity = 1.0 - abs(query_value - concept_feature.value) / max_val
                            if similarity > threshold:
                                matched_features += 1
                                score += similarity * concept_feature.weight
                    elif concept_feature.value == query_value:
                        # 精确匹配
                        matched_features += 1
                        score += 1.0
                
                # 检查实例特征
                for instance_features in concept.instances.values():
                    if query_feature in instance_features:
                        instance_value = instance_features[query_feature]
                        
                        if isinstance(query_value, (int, float)) and isinstance(instance_value, (int, float)):
                            max_val = max(abs(query_value), abs(instance_value))
                            if max_val > 0:
                                similarity = 1.0 - abs(query_value - instance_value) / max_val
                                if similarity > threshold:
                                    matched_features += 1
                                    score += similarity * 0.5  # 实例匹配权重较低
                        elif instance_value == query_value:
                            matched_features += 1
                            score += 0.8  # 实例精确匹配
            
            # 计算最终得分
            if total_features > 0:
                coverage = matched_features / total_features
                score = score * coverage
            
            if score >= threshold:
                results.append((concept, score))
        
        return results
    
    def merge_concepts(self, concept_ids: List[str], 
                      merge_strategy: str = "weighted_average",
                      preserve_relationships: bool = True) -> Optional[str]:
        """合并多个概念
        
        基于概念整合理论，将多个概念合并为新概念
        """
        if len(concept_ids) < 2:
            logger.warning("合并操作至少需要2个概念")
            return None
        
        with self._get_lock():
            # 验证概念存在
            concepts = []
            for concept_id in concept_ids:
                if concept_id not in self.concepts:
                    logger.warning(f"概念不存在: {concept_id}")
                    return None
                concepts.append(self.concepts[concept_id])
            
            # 执行合并
            merged_concept = self._perform_concept_merge(concepts, merge_strategy, preserve_relationships)
            
            if merged_concept:
                # 移除原概念
                for concept_id in concept_ids:
                    del self.concepts[concept_id]
                
                # 添加新概念
                self.concepts[merged_concept.concept_id] = merged_concept
                self.concept_index.add_concept(merged_concept)
                
                # 更新统计
                self.stats['merge_operations'] += 1
                
                logger.info(f"成功合并 {len(concept_ids)} 个概念为: {merged_concept.name}")
                return merged_concept.concept_id
            
            return None
    
    def _perform_concept_merge(self, concepts: List[Concept], 
                             strategy: str,
                             preserve_relationships: bool) -> Optional[Concept]:
        """执行概念合并逻辑"""
        try:
            # 创建合并概念名称
            merge_names = [c.name for c in concepts]
            merged_name = "_".join(merge_names)
            
            # 选择最高层次作为目标层次
            target_level = max(concepts, key=lambda x: x.level.value).level
            
            # 合并概念
            if strategy == "weighted_average":
                merged_concept = self._merge_by_weighted_average(concepts, merged_name, target_level)
            elif strategy == "feature_fusion":
                merged_concept = self._merge_by_feature_fusion(concepts, merged_name, target_level)
            elif strategy == "prototype_based":
                merged_concept = self._merge_by_prototype(concepts, merged_name, target_level)
            else:
                logger.warning(f"未知的合并策略: {strategy}")
                return None
            
            # 保留关系（如果需要）
            if preserve_relationships:
                self._preserve_concept_relationships(concepts, merged_concept)
            
            return merged_concept
            
        except Exception as e:
            logger.error(f"概念合并失败: {str(e)}")
            return None
    
    def _merge_by_weighted_average(self, concepts: List[Concept], 
                                 name: str, level: ConceptLevel) -> Concept:
        """基于加权平均的合并策略"""
        merged_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=name,
            level=level,
            formation_method=FormationMethod.INDUCTIVE
        )
        
        # 合并核心特征
        all_feature_names = set()
        for concept in concepts:
            all_feature_names.update(concept.core_features.keys())
        
        for feature_name in all_feature_names:
            feature_values = []
            feature_weights = []
            feature_confidences = []
            
            for concept in concepts:
                if feature_name in concept.core_features:
                    feature = concept.core_features[feature_name]
                    feature_values.append(feature.value)
                    feature_weights.append(feature.weight * concept.confidence)
                    feature_confidences.append(feature.confidence)
            
            if feature_values:
                # 加权合并
                total_weight = sum(feature_weights)
                if total_weight > 0:
                    if all(isinstance(v, (int, float)) for v in feature_values):
                        # 数值特征加权平均
                        merged_value = sum(v * w for v, w in zip(feature_values, feature_weights)) / total_weight
                    else:
                        # 分类特征使用置信度最高的
                        max_confidence_idx = np.argmax(feature_confidences)
                        merged_value = feature_values[max_confidence_idx]
                    
                    avg_confidence = np.mean(feature_confidences)
                    
                    merged_feature = ConceptFeature(
                        name=feature_name,
                        value=merged_value,
                        weight=total_weight,
                        confidence=avg_confidence,
                        source="merge_operation"
                    )
                    
                    merged_concept.add_feature(merged_feature, is_core=True)
        
        # 合并实例
        for concept in concepts:
            merged_concept.instances.update(concept.instances)
        merged_concept.instance_count = len(merged_concept.instances)
        
        # 计算指标
        merged_concept._update_concept_metrics()
        
        return merged_concept
    
    def _merge_by_feature_fusion(self, concepts: List[Concept], 
                               name: str, level: ConceptLevel) -> Concept:
        """基于特征融合的合并策略"""
        # 特征融合策略保留所有特征，通过关系进行连接
        merged_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=name,
            level=level,
            formation_method=FormationMethod.INDUCTIVE
        )
        
        # 融合所有特征
        for concept in concepts:
            for feature_name, feature in concept.core_features.items():
                # 为融合特征添加来源标识
                fusion_feature = ConceptFeature(
                    name=f"{feature_name}_from_{concept.name}",
                    value=feature.value,
                    weight=feature.weight,
                    confidence=feature.confidence,
                    source=f"fusion_from_{concept.name}",
                    feature_type=feature.feature_type
                )
                merged_concept.add_feature(fusion_feature, is_core=True)
        
        # 合并实例
        for concept in concepts:
            merged_concept.instances.update(concept.instances)
        merged_concept.instance_count = len(merged_concept.instances)
        
        # 建立关系记录
        merged_concept.metadata['merged_from'] = [c.concept_id for c in concepts]
        merged_concept.metadata['merge_strategy'] = 'feature_fusion'
        
        merged_concept._update_concept_metrics()
        return merged_concept
    
    def _merge_by_prototype(self, concepts: List[Concept], 
                          name: str, level: ConceptLevel) -> Concept:
        """基于原型的合并策略"""
        # 找到最具代表性的概念作为基础
        representative_concept = max(concepts, key=lambda c: c.typicality + c.confidence)
        
        # 复制代表性概念
        merged_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=name,
            level=level,
            formation_method=FormationMethod.INDUCTIVE
        )
        
        # 继承代表性概念的特征和实例
        merged_concept.core_features = representative_concept.core_features.copy()
        merged_concept.instances = representative_concept.instances.copy()
        merged_concept.instance_count = representative_concept.instance_count
        
        # 融合其他概念的补充特征
        for concept in concepts:
            if concept.concept_id != representative_concept.concept_id:
                for feature_name, feature in concept.core_features.items():
                    if feature_name not in merged_concept.core_features:
                        merged_concept.core_features[feature_name] = feature
        
        # 合并实例
        for concept in concepts:
            if concept.concept_id != representative_concept.concept_id:
                merged_concept.instances.update(concept.instances)
        merged_concept.instance_count = len(merged_concept.instances)
        
        merged_concept._update_concept_metrics()
        return merged_concept
    
    def _preserve_concept_relationships(self, original_concepts: List[Concept], 
                                      merged_concept: Concept) -> None:
        """保留概念关系"""
        # 收集所有相关概念
        related_concepts = set()
        for concept in original_concepts:
            related_concepts.update(concept.parent_concepts)
            related_concepts.update(concept.child_concepts)
        
        # 建立与相关概念的关系
        for related_id in related_concepts:
            if related_id in self.concepts:
                # 检查是否原概念是该相关概念的子概念
                is_parent = any(related_id in c.parent_concepts for c in original_concepts)
                
                if is_parent:
                    merged_concept.parent_concepts.add(related_id)
                    self.concepts[related_id].child_concepts.add(merged_concept.concept_id)
                else:
                    merged_concept.child_concepts.add(related_id)
                    self.concepts[related_id].parent_concepts.add(merged_concept.concept_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        with self._get_lock():
            stats = self.stats.copy()
            
            # 概念分布统计
            stats['concept_distribution'] = {
                level.value: len([c for c in self.concepts.values() if c.level == level])
                for level in ConceptLevel
            }
            
            # 认知指标统计
            if self.concepts:
                all_concepts = list(self.concepts.values())
                
                stats['cognitive_metrics'] = {
                    'average_confidence': np.mean([c.confidence for c in all_concepts]),
                    'average_typicality': np.mean([c.typicality for c in all_concepts]),
                    'average_clarity': np.mean([c.clarity for c in all_concepts]),
                    'average_stability': np.mean([c.stability for c in all_concepts]),
                    'average_abstractness': np.mean([c.abstractness for c in all_concepts]),
                    'average_complexity': np.mean([c.complexity for c in all_concepts])
                }
                
                # 特征统计
                all_features = []
                for concept in all_concepts:
                    all_features.extend(list(concept.core_features.keys()))
                
                stats['feature_statistics'] = {
                    'total_features': len(all_features),
                    'unique_features': len(set(all_features)),
                    'average_features_per_concept': len(all_features) / len(all_concepts) if all_concepts else 0
                }
            
            # 性能统计
            if self.enable_monitoring:
                stats['performance'] = {
                    'average_formation_time': self.performance_monitor.get_average_formation_time(),
                    'average_search_time': self.performance_monitor.get_average_search_time(),
                    'cache_hit_rate': self.performance_monitor.get_cache_hit_rate()
                }
            
            # 缓存统计
            stats['cache'] = {
                'search_cache_size': len(self.search_cache),
                'similarity_cache_size': len(self.similarity_cache)
            }
            
            return stats
    
    def cleanup_expired_data(self) -> int:
        """清理过期数据"""
        current_time = time.time()
        
        # 清理缓存
        expired_cache_keys = []
        for key, (concepts, timestamp) in self.search_cache.items():
            if current_time - timestamp > 3600:  # 1小时过期
                expired_cache_keys.append(key)
        
        for key in expired_cache_keys:
            del self.search_cache[key]
        
        # 清理不活跃概念（可选，基于内存管理策略）
        inactive_concepts = []
        for concept_id, concept in self.concepts.items():
            if (current_time - concept.last_accessed > 86400 and  # 24小时未访问
                concept.activation_count == 0):  # 从未激活
                inactive_concepts.append(concept_id)
        
        # 移除不活跃概念
        for concept_id in inactive_concepts:
            del self.concepts[concept_id]
        
        logger.info(f"清理完成: 删除 {len(expired_cache_keys)} 个缓存项, {len(inactive_concepts)} 个不活跃概念")
        return len(expired_cache_keys) + len(inactive_concepts)
    
    def save_state(self, filepath: str) -> None:
        """保存概念形成器状态"""
        with self._get_lock():
            try:
                # 准备序列化数据
                state = {
                    'config': self.config,
                    'concepts': {cid: concept.to_dict() for cid, concept in self.concepts.items()},
                    'stats': self.stats,
                    'timestamp': time.time()
                }
                
                # 保存到文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2, default=str)
                
                logger.info(f"概念形成器状态已保存到: {filepath}")
                
            except Exception as e:
                logger.error(f"保存状态失败: {str(e)}")
                raise
    
    def load_state(self, filepath: str) -> None:
        """加载概念形成器状态"""
        with self._get_lock():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # 恢复配置
                if 'config' in state:
                    self.config.update(state['config'])
                
                # 恢复概念
                if 'concepts' in state:
                    self.concepts.clear()
                    for concept_id, concept_data in state['concepts'].items():
                        concept = Concept.from_dict(concept_data)
                        self.concepts[concept_id] = concept
                        
                        # 重建索引
                        self.concept_index.add_concept(concept)
                
                # 恢复统计
                if 'stats' in state:
                    self.stats.update(state['stats'])
                
                logger.info(f"成功加载 {len(self.concepts)} 个概念")
                
            except Exception as e:
                logger.error(f"加载状态失败: {str(e)}")
                raise
    
    def clear_all_concepts(self) -> None:
        """清除所有概念"""
        with self._get_lock():
            self.concepts.clear()
            self.concept_index = ConceptIndex()
            self.search_cache.clear()
            self.similarity_cache.clear()
            
            self.stats = {
                'total_concepts': 0,
                'formation_operations': 0,
                'search_operations': 0,
                'merge_operations': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            
            logger.info("所有概念和缓存已清除")
    
    def export_concepts(self, concept_ids: List[str], filepath: str) -> None:
        """导出特定概念到文件"""
        with self._get_lock():
            export_data = {}
            
            for concept_id in concept_ids:
                if concept_id in self.concepts:
                    export_data[concept_id] = self.concepts[concept_id].to_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"导出了 {len(export_data)} 个概念到: {filepath}")
    
    def import_concepts(self, filepath: str, merge_strategy: str = "keep_existing") -> List[str]:
        """从文件导入概念"""
        with self._get_lock():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                imported_ids = []
                
                for concept_id, concept_data in import_data.items():
                    # 检查是否已存在
                    if concept_id in self.concepts:
                        if merge_strategy == "keep_existing":
                            continue
                        elif merge_strategy == "overwrite":
                            del self.concepts[concept_id]
                        elif merge_strategy == "merge":
                            # 合并现有概念和新概念
                            existing_concept = self.concepts[concept_id]
                            new_concept = Concept.from_dict(concept_data)
                            merged_concept = existing_concept.merge_with(new_concept)
                            self.concepts[concept_id] = merged_concept
                            imported_ids.append(concept_id)
                            continue
                    
                    # 创建新概念
                    concept = Concept.from_dict(concept_data)
                    self.concepts[concept_id] = concept
                    self.concept_index.add_concept(concept)
                    imported_ids.append(concept_id)
                
                # 更新统计
                self.stats['total_concepts'] = len(self.concepts)
                
                logger.info(f"导入了 {len(imported_ids)} 个概念")
                return imported_ids
                
            except Exception as e:
                logger.error(f"导入概念失败: {str(e)}")
                raise
    
    def get_concept_visualization_data(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """获取概念可视化数据
        
        用于支持概念网络的可视化显示
        """
        if concept_id not in self.concepts:
            return None
        
        concept = self.concepts[concept_id]
        
        visualization_data = {
            'concept': {
                'id': concept.concept_id,
                'name': concept.name,
                'level': concept.level.value,
                'confidence': concept.confidence,
                'typicality': concept.typicality,
                'abstractness': concept.abstractness,
                'complexity': concept.complexity
            },
            'features': [
                {
                    'name': name,
                    'value': str(feat.value),
                    'weight': feat.weight,
                    'confidence': feat.confidence,
                    'type': feat.feature_type
                }
                for name, feat in concept.core_features.items()
            ],
            'relationships': {
                'parents': list(concept.parent_concepts),
                'children': list(concept.child_concepts),
                'similar': list(concept.similar_concepts.items())
            },
            'instances': {
                'count': concept.instance_count,
                'examples': list(concept.instances.keys())[:5]  # 只显示前5个实例
            },
            'cognitive_metrics': {
                'clarity': concept.clarity,
                'stability': concept.stability,
                'frequency': concept.frequency,
                'evidence_strength': concept.evidence_strength
            }
        }
        
        return visualization_data