"""
概念继承机制 - ConceptInheritance
Concept Inheritance Mechanism

这个类实现了概念之间的继承关系，支持属性的继承、特化、泛化等操作。
本实现基于现代认知科学和人工智能的继承理论：

理论基础：
- 继承理论（Inheritance Theory）
- 概念层次理论（Concept Hierarchical Theory）
- 类-实例理论（Class-Instance Theory）
- 原型理论（Prototype Theory）
- 概念特化与泛化（Concept Specialization and Generalization）
- 多重继承理论（Multiple Inheritance Theory）
- 认知心理学继承模型（Cognitive Psychology Inheritance Models）

技术特点：
- 多类型继承机制
- 继承强度计算
- 属性冲突解决
- 动态继承更新
- 继承关系验证
- 性能优化缓存
- 支持并发访问

Author: NeuroMinecraft Genesis Team
Date: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import threading
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import json
from copy import deepcopy

from .concept_former import Concept, ConceptLevel, ConceptFeature, ConceptFormer

logger = logging.getLogger(__name__)


class InheritanceType(Enum):
    """继承类型枚举
    
    基于认知心理学的不同继承模式：
    """
    SIMPLE_INHERITANCE = "simple_inheritance"           # 简单继承：单亲继承
    MULTIPLE_INHERITANCE = "multiple_inheritance"       # 多重继承：多个父概念
    MULTI_LEVEL_INHERITANCE = "multi_level_inheritance" # 多级继承：跨越多个层次
    PROTOTYPICAL_INHERITANCE = "prototypical_inheritance" # 原型继承：基于典型特征
    BEHAVIORAL_INHERITANCE = "behavioral_inheritance"   # 行为继承：基于功能行为
    STRUCTURAL_INHERITANCE = "structural_inheritance"   # 结构继承：基于内部结构
    CONTEXTUAL_INHERITANCE = "contextual_inheritance"   # 上下文继承：基于使用场景
    FUZZY_INHERITANCE = "fuzzy_inheritance"             # 模糊继承：概率性继承


class InheritanceStrategy(Enum):
    """继承策略枚举"""
    CONSERVATIVE = "conservative"           # 保守策略：优先保持父概念特征
    PROGRESSIVE = "progressive"             # 激进策略：优先新特征
    BALANCED = "balanced"                   # 平衡策略：加权平均
    DOMINANT = "dominant"                   # 主导策略：最强特征优先
    ADAPTIVE = "adaptive"                   # 自适应策略：根据上下文调整


class ConflictResolution(Enum):
    """冲突解决策略"""
    PRIORITY_BASED = "priority_based"       # 基于优先级的冲突解决
    WEIGHT_BASED = "weight_based"           # 基于权重的冲突解决
    CONFIDENCE_BASED = "confidence_based"   # 基于置信度的冲突解决
    LATEST_FIRST = "latest_first"           # 最新优先策略
    MANUAL_RESOLUTION = "manual_resolution" # 手动解决策略


@dataclass
class InheritanceRule:
    """继承规则定义
    
    定义继承操作的条件、策略和参数
    """
    rule_id: str
    name: str
    description: str
    inheritance_type: InheritanceType
    strategy: InheritanceStrategy
    conflict_resolution: ConflictResolution
    
    # 继承条件
    min_parent_confidence: float = 0.6      # 最小父概念置信度
    min_inheritance_strength: float = 0.5    # 最小继承强度
    feature_coverage_threshold: float = 0.3  # 特征覆盖率阈值
    
    # 继承参数
    priority_weights: Dict[str, float] = field(default_factory=dict)  # 优先级权重
    inheritance_depth_limit: int = 10        # 继承深度限制
    max_parents: int = 5                    # 最大父概念数量
    
    # 约束条件
    allowed_inheritance_types: Set[InheritanceType] = field(default_factory=set)
    forbidden_feature_types: Set[str] = field(default_factory=set)
    
    # 质量控制
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    
    # 性能统计
    usage_count: int = 0
    success_rate: float = 0.0
    average_inheritance_quality: float = 0.0
    
    def validate_inheritance_request(self, 
                                    parent_concepts: List[Concept],
                                    child_concept: Concept) -> Tuple[bool, str]:
        """验证继承请求
        
        Args:
            parent_concepts: 父概念列表
            child_concept: 子概念
            
        Returns:
            (是否验证通过, 失败原因)
        """
        # 检查父概念数量
        if len(parent_concepts) > self.max_parents:
            return False, f"父概念数量超过限制: {len(parent_concepts)} > {self.max_parents}"
        
        # 检查继承类型允许
        if self.inheritance_type not in self.allowed_inheritance_types:
            return False, f"继承类型 {self.inheritance_type.value} 不被允许"
        
        # 检查父概念置信度
        min_confidence = min(c.confidence for c in parent_concepts)
        if min_confidence < self.min_parent_confidence:
            return False, f"父概念置信度不足: {min_confidence:.3f} < {self.min_parent_confidence}"
        
        # 检查继承强度
        for parent in parent_concepts:
            inheritance_strength = self._calculate_inheritance_strength(parent, child_concept)
            if inheritance_strength < self.min_inheritance_strength:
                return False, f"继承强度不足: {inheritance_strength:.3f} < {self.min_inheritance_strength}"
        
        # 检查深度限制
        if self._calculate_inheritance_depth(parent_concepts) > self.inheritance_depth_limit:
            return False, f"继承深度超过限制: {self.inheritance_depth_limit}"
        
        return True, "验证通过"
    
    def _calculate_inheritance_strength(self, parent: Concept, child: Concept) -> float:
        """计算继承强度"""
        # 基于特征覆盖率计算继承强度
        parent_features = set(parent.core_features.keys())
        child_features = set(child.core_features.keys())
        
        if not parent_features:
            return 0.0
        
        # 继承的特征数量
        inherited_features = parent_features & child_features
        inheritance_rate = len(inherited_features) / len(parent_features)
        
        # 特征质量因子
        avg_parent_confidence = np.mean([f.confidence for f in parent.core_features.values()])
        avg_child_confidence = np.mean([f.confidence for f in child.core_features.values()])
        
        # 综合继承强度
        inheritance_strength = inheritance_rate * (avg_parent_confidence + avg_child_confidence) / 2
        
        return min(1.0, inheritance_strength)
    
    def _calculate_inheritance_depth(self, parent_concepts: List[Concept]) -> int:
        """计算继承深度"""
        if not parent_concepts:
            return 0
        
        # 计算父概念的平均层次深度
        level_depths = []
        for concept in parent_concepts:
            depth = self._get_concept_depth(concept)
            level_depths.append(depth)
        
        return max(level_depths)
    
    def _get_concept_depth(self, concept: Concept) -> int:
        """获取概念层次深度"""
        depth_mapping = {
            ConceptLevel.INSTANCE: 0,
            ConceptLevel.BASIC: 1,
            ConceptLevel.SUPERORDINATE: 2,
            ConceptLevel.METACONCEPT: 3
        }
        return depth_mapping.get(concept.level, 0)


@dataclass
class InheritanceResult:
    """继承结果"""
    child_concept_id: str
    parent_concept_ids: List[str]
    inheritance_type: InheritanceType
    success: bool
    
    # 继承的特征和属性
    inherited_features: Dict[str, ConceptFeature]
    modified_features: Dict[str, ConceptFeature]
    new_features: Dict[str, ConceptFeature]
    
    # 继承质量指标
    inheritance_strength: float
    feature_coverage_rate: float
    conflict_resolution_rate: float
    
    # 处理信息
    processing_time: float
    conflicts_resolved: int
    validation_errors: List[str]
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    rule_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InheritanceConflict:
    """继承冲突定义"""
    conflict_id: str
    conflict_type: str
    feature_name: str
    conflicting_values: List[Tuple[Any, float]]  # (值, 置信度)
    resolution_strategy: ConflictResolution
    resolved_value: Optional[Any] = None
    resolution_confidence: float = 0.0


class InheritanceCache:
    """继承关系缓存系统"""
    
    def __init__(self, max_size: int = 2000, ttl: float = 7200):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}  # {child_id: {feature_name: inherited_value}}
        self.access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get_inherited_features(self, child_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的继承特征"""
        with self._lock:
            if child_id in self.cache:
                self.access_times[child_id] = time.time()
                return self.cache[child_id]
            return None
    
    def store_inherited_features(self, child_id: str, features: Dict[str, Any]) -> None:
        """存储继承特征到缓存"""
        with self._lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[child_id] = features
            self.access_times[child_id] = time.time()
    
    def invalidate_cache(self, concept_id: str) -> None:
        """使与指定概念相关的缓存失效"""
        with self._lock:
            keys_to_remove = []
            for key in self.cache.keys():
                if concept_id in key or concept_id in str(self.cache[key]):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """移除最近最少使用的缓存项"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()


class ConceptInheritance:
    """概念继承机制
    
    负责管理概念之间的继承关系，支持多种继承类型和策略。
    基于认知科学和面向对象编程的继承理论实现。
    
    主要功能：
    - 多类型继承机制
    - 继承冲突解决
    - 继承强度计算
    - 继承关系验证
    - 动态继承更新
    - 性能优化缓存
    """
    
    def __init__(self,
                 concept_former: Optional[ConceptFormer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """初始化概念继承机制
        
        Args:
            concept_former: 概念形成器引用
            config: 配置参数
                - cache_size: 缓存大小
                - default_strategy: 默认继承策略
                - conflict_resolution: 默认冲突解决策略
                - enable_cache: 是否启用缓存
        """
        self.config = config or {}
        self.concept_former = concept_former
        
        # 初始化继承规则
        self.inheritance_rules = self._initialize_inheritance_rules()
        
        # 初始化缓存系统
        self.cache = InheritanceCache(
            max_size=self.config.get('cache_size', 2000)
        ) if self.config.get('enable_cache', True) else None
        
        # 默认配置
        self.default_strategy = InheritanceStrategy(
            self.config.get('default_strategy', 'balanced')
        )
        self.default_conflict_resolution = ConflictResolution(
            self.config.get('conflict_resolution', 'confidence_based')
        )
        
        # 继承关系存储
        self.inheritance_relations: Dict[str, InheritanceRelation] = {}
        self.inheritance_graph: Dict[str, Set[str]] = defaultdict(set)  # 继承图
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_inheritance_operations': 0,
            'successful_inheritances': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'average_processing_time': 0.0,
            'inheritance_depth_distribution': defaultdict(int)
        }
        
        logger.info(f"ConceptInheritance 初始化完成 - 配置: {self.config}")
    
    def _initialize_inheritance_rules(self) -> Dict[str, InheritanceRule]:
        """初始化继承规则集合"""
        rules = {}
        
        # 简单继承规则
        rules['simple_inheritance'] = InheritanceRule(
            rule_id='simple_inheritance',
            name='简单继承',
            description='单个父概念向子概念传递属性',
            inheritance_type=InheritanceType.SIMPLE_INHERITANCE,
            strategy=InheritanceStrategy.CONSERVATIVE,
            conflict_resolution=ConflictResolution.PRIORITY_BASED,
            min_parent_confidence=0.6,
            min_inheritance_strength=0.5,
            inheritance_depth_limit=5,
            max_parents=1,
            allowed_inheritance_types={InheritanceType.SIMPLE_INHERITANCE}
        )
        
        # 多重继承规则
        rules['multiple_inheritance'] = InheritanceRule(
            rule_id='multiple_inheritance',
            name='多重继承',
            description='多个父概念向子概念传递属性',
            inheritance_type=InheritanceType.MULTIPLE_INHERITANCE,
            strategy=InheritanceStrategy.BALANCED,
            conflict_resolution=ConflictResolution.WEIGHT_BASED,
            min_parent_confidence=0.5,
            min_inheritance_strength=0.4,
            inheritance_depth_limit=8,
            max_parents=3,
            allowed_inheritance_types={InheritanceType.MULTIPLE_INHERITANCE, InheritanceType.SIMPLE_INHERITANCE}
        )
        
        # 原型继承规则
        rules['prototypical_inheritance'] = InheritanceRule(
            rule_id='prototypical_inheritance',
            name='原型继承',
            description='基于原型特征的继承机制',
            inheritance_type=InheritanceType.PROTOTYPICAL_INHERITANCE,
            strategy=InheritanceStrategy.PROGRESSIVE,
            conflict_resolution=ConflictResolution.CONFIDENCE_BASED,
            min_parent_confidence=0.7,
            min_inheritance_strength=0.6,
            inheritance_depth_limit=3,
            max_parents=1,
            allowed_inheritance_types={InheritanceType.PROTOTYPICAL_INHERITANCE, InheritanceType.SIMPLE_INHERITANCE}
        )
        
        # 结构继承规则
        rules['structural_inheritance'] = InheritanceRule(
            rule_id='structural_inheritance',
            name='结构继承',
            description='基于内部结构的继承机制',
            inheritance_type=InheritanceType.STRUCTURAL_INHERITANCE,
            strategy=InheritanceStrategy.CONSERVATIVE,
            conflict_resolution=ConflictResolution.PRIORITY_BASED,
            min_parent_confidence=0.6,
            min_inheritance_strength=0.5,
            inheritance_depth_limit=6,
            max_parents=2,
            allowed_inheritance_types={InheritanceType.STRUCTURAL_INHERITANCE, InheritanceType.SIMPLE_INHERITANCE}
        )
        
        return rules
    
    def inherit_concept(self,
                       parent_concept_ids: List[str],
                       child_name: str,
                       specialization_features: Optional[Dict[str, ConceptFeature]] = None,
                       inheritance_type: InheritanceType = InheritanceType.SIMPLE_INHERITANCE,
                       strategy: Optional[InheritanceStrategy] = None,
                       rule_id: Optional[str] = None) -> Optional[str]:
        """继承概念
        
        从父概念继承属性创建新的子概念
        
        Args:
            parent_concept_ids: 父概念ID列表
            child_name: 子概念名称
            specialization_features: 特化特征
            inheritance_type: 继承类型
            strategy: 继承策略
            rule_id: 继承规则ID
            
        Returns:
            子概念ID，失败返回None
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # 输入验证
                if not parent_concept_ids:
                    raise ValueError("父概念ID列表不能为空")
                
                if not child_name or not child_name.strip():
                    raise ValueError("子概念名称不能为空")
                
                # 获取父概念对象
                parent_concepts = []
                for concept_id in parent_concept_ids:
                    if self.concept_former:
                        concept = self.concept_former.get_concept(concept_id)
                    else:
                        raise ValueError("需要ConceptFormer来获取概念对象")
                    
                    if not concept:
                        raise ValueError(f"父概念不存在: {concept_id}")
                    
                    parent_concepts.append(concept)
                
                # 使用默认策略（如果未指定）
                if strategy is None:
                    strategy = self.default_strategy
                
                # 选择继承规则
                rule = self._select_inheritance_rule(parent_concepts, inheritance_type, rule_id)
                if not rule:
                    logger.warning(f"未找到适合的继承规则: {inheritance_type.value}")
                    return None
                
                # 验证继承请求
                can_inherit, reason = rule.validate_inheritance_request(parent_concepts, None)
                if not can_inherit:
                    logger.warning(f"继承验证失败: {reason}")
                    return None
                
                logger.info(f"开始继承操作: {len(parent_concepts)} 个父概念 -> {child_name} (类型: {inheritance_type.value})")
                
                # 执行继承
                result = self._perform_inheritance(
                    parent_concepts=parent_concepts,
                    child_name=child_name,
                    specialization_features=specialization_features or {},
                    inheritance_type=inheritance_type,
                    strategy=strategy,
                    rule=rule
                )
                
                if not result or not result.success:
                    logger.error("继承操作执行失败")
                    return None
                
                # 缓存继承结果
                if self.cache:
                    self.cache.store_inherited_features(result.child_concept_id, result.inherited_features)
                
                # 更新统计
                self._update_inheritance_statistics(result, time.time() - start_time)
                
                logger.info(f"继承成功: {result.child_concept_id} (强度: {result.inheritance_strength:.3f})")
                return result.child_concept_id
                
            except Exception as e:
                logger.error(f"继承操作失败: {str(e)}")
                return None
    
    def _select_inheritance_rule(self,
                               parent_concepts: List[Concept],
                               inheritance_type: InheritanceType,
                               rule_id: Optional[str]) -> Optional[InheritanceRule]:
        """选择最适合的继承规则"""
        if rule_id and rule_id in self.inheritance_rules:
            rule = self.inheritance_rules[rule_id]
            if inheritance_type in rule.allowed_inheritance_types:
                rule.usage_count += 1
                return rule
        
        # 寻找匹配的规则
        best_rule = None
        best_score = 0.0
        
        for rule in self.inheritance_rules.values():
            if inheritance_type in rule.allowed_inheritance_types:
                # 计算规则适用性得分
                score = self._calculate_rule_appropriateness(rule, parent_concepts)
                if score > best_score:
                    best_rule = rule
                    best_score = score
        
        if best_rule:
            best_rule.usage_count += 1
            return best_rule
        
        return None
    
    def _calculate_rule_appropriateness(self, rule: InheritanceRule, parent_concepts: List[Concept]) -> float:
        """计算规则适用性得分"""
        score = 0.0
        
        # 基于父概念数量打分
        if rule.max_parents >= len(parent_concepts):
            score += 0.3
        
        # 基于置信度打分
        avg_confidence = np.mean([c.confidence for c in parent_concepts])
        if avg_confidence >= rule.min_parent_confidence:
            score += 0.4
        
        # 基于继承强度打分
        inheritance_strengths = []
        for parent in parent_concepts:
            # 简化的继承强度计算
            strength = len(parent.core_features) / 10.0  # 特征数量作为强度指标
            inheritance_strengths.append(min(1.0, strength))
        
        avg_strength = np.mean(inheritance_strengths)
        if avg_strength >= rule.min_inheritance_strength:
            score += 0.3
        
        return min(1.0, score)
    
    def _perform_inheritance(self,
                           parent_concepts: List[Concept],
                           child_name: str,
                           specialization_features: Dict[str, ConceptFeature],
                           inheritance_type: InheritanceType,
                           strategy: InheritanceStrategy,
                           rule: InheritanceRule) -> Optional[InheritanceResult]:
        """执行具体的继承操作"""
        try:
            # 收集继承的特征
            inherited_features = self._collect_inherited_features(
                parent_concepts, strategy, rule
            )
            
            # 检测和处理冲突
            conflicts = self._detect_and_resolve_conflicts(
                inherited_features, specialization_features, rule
            )
            
            # 合并特征
            final_features = self._merge_features(
                inherited_features, specialization_features, strategy, rule
            )
            
            # 创建子概念
            child_concept_id = self._create_child_concept(
                name=child_name,
                features=final_features,
                parent_concepts=parent_concepts,
                inheritance_type=inheritance_type
            )
            
            if not child_concept_id:
                return None
            
            # 计算继承质量指标
            inheritance_strength = self._calculate_inheritance_strength(parent_concepts, final_features)
            feature_coverage_rate = self._calculate_feature_coverage_rate(parent_concepts, final_features)
            
            # 创建继承结果
            result = InheritanceResult(
                child_concept_id=child_concept_id,
                parent_concept_ids=[c.concept_id for c in parent_concepts],
                inheritance_type=inheritance_type,
                success=True,
                inherited_features={k: v for k, v in final_features.items() 
                                 if k in inherited_features},
                modified_features={k: v for k, v in final_features.items() 
                                 if k in specialization_features},
                new_features={k: v for k, v in final_features.items() 
                            if k not in inherited_features and k not in specialization_features},
                inheritance_strength=inheritance_strength,
                feature_coverage_rate=feature_coverage_rate,
                conflict_resolution_rate=len(conflicts) / max(1, len(inherited_features)),
                processing_time=0.0,  # 将在调用者中设置
                conflicts_resolved=len(conflicts),
                validation_errors=[],
                rule_id=rule.rule_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"继承操作执行失败: {str(e)}")
            return None
    
    def _collect_inherited_features(self,
                                  parent_concepts: List[Concept],
                                  strategy: InheritanceStrategy,
                                  rule: InheritanceRule) -> Dict[str, ConceptFeature]:
        """收集继承的特征"""
        inherited_features = {}
        
        for parent_concept in parent_concepts:
            for feature_name, feature in parent_concept.core_features.items():
                if feature_name in inherited_features:
                    # 特征冲突，需要解决
                    existing_feature = inherited_features[feature_name]
                    resolved_feature = self._resolve_feature_conflict(
                        existing_feature, feature, strategy, rule
                    )
                    inherited_features[feature_name] = resolved_feature
                else:
                    inherited_features[feature_name] = feature
        
        return inherited_features
    
    def _resolve_feature_conflict(self,
                                feature1: ConceptFeature,
                                feature2: ConceptFeature,
                                strategy: InheritanceStrategy,
                                rule: InheritanceRule) -> ConceptFeature:
        """解决特征冲突"""
        if feature1.name != feature2.name:
            return feature1  # 不应该发生，但作为安全措施
        
        conflict = InheritanceConflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type="feature_value_conflict",
            feature_name=feature1.name,
            conflicting_values=[(feature1.value, feature1.confidence), 
                              (feature2.value, feature2.confidence)],
            resolution_strategy=rule.conflict_resolution
        )
        
        # 根据冲突解决策略选择值
        if rule.conflict_resolution == ConflictResolution.CONFIDENCE_BASED:
            # 基于置信度选择
            if feature1.confidence >= feature2.confidence:
                chosen_feature = feature1
                conflict.resolved_value = feature1.value
                conflict.resolution_confidence = feature1.confidence
            else:
                chosen_feature = feature2
                conflict.resolved_value = feature2.value
                conflict.resolution_confidence = feature2.confidence
        
        elif rule.conflict_resolution == ConflictResolution.WEIGHT_BASED:
            # 基于权重选择
            if feature1.weight >= feature2.weight:
                chosen_feature = feature1
                conflict.resolved_value = feature1.value
                conflict.resolution_confidence = feature1.confidence
            else:
                chosen_feature = feature2
                conflict.resolved_value = feature2.value
                conflict.resolution_confidence = feature2.confidence
        
        elif rule.conflict_resolution == ConflictResolution.PRIORITY_BASED:
            # 基于优先级选择（这里简化为第一个特征）
            chosen_feature = feature1
            conflict.resolved_value = feature1.value
            conflict.resolution_confidence = feature1.confidence
        
        else:
            # 默认选择第一个特征
            chosen_feature = feature1
            conflict.resolved_value = feature1.value
            conflict.resolution_confidence = feature1.confidence
        
        self.stats['conflicts_resolved'] += 1
        return chosen_feature
    
    def _detect_and_resolve_conflicts(self,
                                    inherited_features: Dict[str, ConceptFeature],
                                    specialization_features: Dict[str, ConceptFeature],
                                    rule: InheritanceRule) -> List[InheritanceConflict]:
        """检测和解决冲突"""
        conflicts = []
        
        # 检查继承特征与特化特征的冲突
        for feature_name in inherited_features:
            if feature_name in specialization_features:
                inherited_feat = inherited_features[feature_name]
                specialized_feat = specialization_features[feature_name]
                
                # 检查值冲突
                if inherited_feat.value != specialized_feat.value:
                    conflict = InheritanceConflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type="inheritance_specialization_conflict",
                        feature_name=feature_name,
                        conflicting_values=[(inherited_feat.value, inherited_feat.confidence),
                                         (specialized_feat.value, specialized_feat.confidence)],
                        resolution_strategy=rule.conflict_resolution
                    )
                    
                    # 解决冲突（特化特征优先）
                    conflict.resolved_value = specialized_feat.value
                    conflict.resolution_confidence = specialized_feat.confidence
                    
                    conflicts.append(conflict)
        
        self.stats['conflicts_detected'] += len(conflicts)
        return conflicts
    
    def _merge_features(self,
                       inherited_features: Dict[str, ConceptFeature],
                       specialization_features: Dict[str, ConceptFeature],
                       strategy: InheritanceStrategy,
                       rule: InheritanceRule) -> Dict[str, ConceptFeature]:
        """合并特征"""
        merged_features = inherited_features.copy()
        
        # 合并特化特征
        for feature_name, feature in specialization_features.items():
            if strategy == InheritanceStrategy.CONSERVATIVE:
                # 保守策略：特化特征优先
                merged_features[feature_name] = feature
            elif strategy == InheritanceStrategy.PROGRESSIVE:
                # 激进策略：特化特征优先
                merged_features[feature_name] = feature
            elif strategy == InheritanceStrategy.BALANCED:
                # 平衡策略：合并或选择
                if feature_name in merged_features:
                    # 合并数值特征
                    existing_feature = merged_features[feature_name]
                    merged_feature = self._merge_numerical_features(existing_feature, feature)
                    merged_features[feature_name] = merged_feature
                else:
                    merged_features[feature_name] = feature
            else:
                # 其他策略：特化特征优先
                merged_features[feature_name] = feature
        
        return merged_features
    
    def _merge_numerical_features(self,
                                feature1: ConceptFeature,
                                feature2: ConceptFeature) -> ConceptFeature:
        """合并数值特征"""
        if not isinstance(feature1.value, (int, float)) or not isinstance(feature2.value, (int, float)):
            # 非数值特征，保留后一个
            return feature2
        
        # 数值特征加权合并
        total_weight = feature1.weight * feature1.confidence + feature2.weight * feature2.confidence
        if total_weight > 0:
            merged_value = (feature1.value * feature1.weight * feature1.confidence + 
                          feature2.value * feature2.weight * feature2.confidence) / total_weight
        else:
            merged_value = (feature1.value + feature2.value) / 2
        
        merged_confidence = (feature1.confidence + feature2.confidence) / 2
        merged_weight = (feature1.weight + feature2.weight) / 2
        
        return ConceptFeature(
            name=feature1.name,
            value=merged_value,
            weight=merged_weight,
            confidence=merged_confidence,
            source="inheritance_merge",
            feature_type=feature1.feature_type,
            abstractable=feature1.abstractable and feature2.abstractable,
            stable=feature1.stable and feature2.stable
        )
    
    def _create_child_concept(self,
                            name: str,
                            features: Dict[str, ConceptFeature],
                            parent_concepts: List[Concept],
                            inheritance_type: InheritanceType) -> Optional[str]:
        """创建子概念"""
        if not self.concept_former:
            logger.error("需要ConceptFormer来创建子概念")
            return None
        
        try:
            # 确定子概念层次（通常比父概念更具体）
            parent_levels = [c.level for c in parent_concepts]
            child_level = ConceptLevel.INSTANCE if ConceptLevel.INSTANCE in parent_levels else min(parent_levels)
            
            # 创建概念特征字典
            feature_dict = {f.name: f for f in features.values()}
            
            # 形成子概念
            child_concept_id = self.concept_former.form_concept(
                name=name,
                perceptions=[],  # 继承的概念通常没有直接的感知实例
                level=child_level,
                method=FormationMethod.DEDUCTIVE,  # 演绎推理：从一般到特殊
                features=feature_dict
            )
            
            if child_concept_id:
                # 建立继承关系
                child_concept = self.concept_former.get_concept(child_concept_id)
                for parent in parent_concepts:
                    child_concept.parent_concepts.add(parent.concept_id)
                    parent.child_concepts.add(child_concept_id)
                
                # 记录继承关系
                relation = InheritanceRelation(
                    relation_id=str(uuid.uuid4()),
                    parent_concept_ids=[p.concept_id for p in parent_concepts],
                    child_concept_id=child_concept_id,
                    inheritance_type=inheritance_type,
                    inheritance_strength=self._calculate_inheritance_strength(parent_concepts, features),
                    confidence_level=np.mean([c.confidence for c in parent_concepts])
                )
                
                self.inheritance_relations[child_concept_id] = relation
                
                # 更新继承图
                for parent in parent_concepts:
                    self.inheritance_graph[parent.concept_id].add(child_concept_id)
                
                logger.info(f"成功创建子概念: {name} ({child_concept_id})")
            
            return child_concept_id
            
        except Exception as e:
            logger.error(f"创建子概念失败: {str(e)}")
            return None
    
    def _calculate_inheritance_strength(self,
                                      parent_concepts: List[Concept],
                                      child_features: Dict[str, ConceptFeature]) -> float:
        """计算继承强度"""
        if not parent_concepts:
            return 0.0
        
        # 收集所有父特征
        parent_features = set()
        for parent in parent_concepts:
            parent_features.update(parent.core_features.keys())
        
        if not parent_features:
            return 0.0
        
        # 计算继承的特征比例
        child_feature_names = set(child_features.keys())
        inherited_features = parent_features & child_feature_names
        inheritance_rate = len(inherited_features) / len(parent_features)
        
        # 计算特征质量
        inherited_feature_qualities = []
        for feature_name in inherited_features:
            if feature_name in child_features:
                quality = child_features[feature_name].confidence
                inherited_feature_qualities.append(quality)
        
        avg_quality = np.mean(inherited_feature_qualities) if inherited_feature_qualities else 0.0
        
        # 综合继承强度
        inheritance_strength = inheritance_rate * avg_quality
        
        return min(1.0, inheritance_strength)
    
    def _calculate_feature_coverage_rate(self,
                                       parent_concepts: List[Concept],
                                       child_features: Dict[str, ConceptFeature]) -> float:
        """计算特征覆盖率"""
        if not parent_concepts:
            return 0.0
        
        # 收集所有父特征
        all_parent_features = set()
        for parent in parent_concepts:
            all_parent_features.update(parent.core_features.keys())
        
        if not all_parent_features:
            return 1.0
        
        # 计算子概念特征覆盖率
        child_feature_names = set(child_features.keys())
        coverage = len(child_feature_names & all_parent_features) / len(all_parent_features)
        
        return coverage
    
    def _update_inheritance_statistics(self, result: InheritanceResult, processing_time: float) -> None:
        """更新继承统计信息"""
        self.stats['total_inheritance_operations'] += 1
        self.stats['successful_inheritances'] += 1
        
        # 更新平均处理时间
        current_avg = self.stats['average_processing_time']
        total_ops = self.stats['total_inheritance_operations']
        self.stats['average_processing_time'] = (current_avg * (total_ops - 1) + processing_time) / total_ops
        
        # 更新继承深度分布
        depth = len(result.parent_concept_ids)
        self.stats['inheritance_depth_distribution'][depth] += 1
    
    def get_inherited_features(self, child_concept_id: str, 
                             refresh_cache: bool = False) -> Optional[Dict[str, ConceptFeature]]:
        """获取概念的继承特征
        
        Args:
            child_concept_id: 子概念ID
            refresh_cache: 是否刷新缓存
            
        Returns:
            继承的特征字典
        """
        if not self.concept_former:
            return None
        
        child_concept = self.concept_former.get_concept(child_concept_id)
        if not child_concept:
            return None
        
        # 检查缓存
        if self.cache and not refresh_cache:
            cached_features = self.cache.get_inherited_features(child_concept_id)
            if cached_features:
                # 转换为ConceptFeature对象
                return {
                    name: ConceptFeature(
                        name=name,
                        value=data['value'],
                        weight=data.get('weight', 1.0),
                        confidence=data.get('confidence', 1.0),
                        source=data.get('source', 'cache'),
                        feature_type=data.get('feature_type', 'inherited')
                    )
                    for name, data in cached_features.items()
                }
        
        # 从概念中提取继承特征
        inherited_features = {}
        
        for parent_id in child_concept.parent_concepts:
            parent_concept = self.concept_former.get_concept(parent_id)
            if parent_concept:
                for feature_name, feature in parent_concept.core_features.items():
                    if feature_name in child_concept.core_features:
                        # 标记为继承特征
                        inherited_feature = ConceptFeature(
                            name=feature_name,
                            value=feature.value,
                            weight=feature.weight,
                            confidence=feature.confidence,
                            source=f"inherited_from_{parent_id}",
                            feature_type="inherited",
                            abstractable=feature.abstractable,
                            stable=feature.stable
                        )
                        inherited_features[feature_name] = inherited_feature
        
        # 缓存结果
        if self.cache:
            cached_data = {
                name: {
                    'value': feat.value,
                    'weight': feat.weight,
                    'confidence': feat.confidence,
                    'source': feat.source,
                    'feature_type': feat.feature_type
                }
                for name, feat in inherited_features.items()
            }
            self.cache.store_inherited_features(child_concept_id, cached_data)
        
        return inherited_features
    
    def update_inheritance_relations(self, concept_id: str) -> bool:
        """更新与指定概念相关的继承关系
        
        Args:
            concept_id: 概念ID
            
        Returns:
            是否成功更新
        """
        with self._lock:
            try:
                # 使缓存失效
                if self.cache:
                    self.cache.invalidate_cache(concept_id)
                
                # 检查是否有继承关系需要更新
                if concept_id in self.inheritance_relations:
                    # 更新现有的继承关系
                    relation = self.inheritance_relations[concept_id]
                    relation.access_count += 1
                
                # 如果概念有父概念，更新继承图
                if concept_id in self.inheritance_graph:
                    # 检查父概念是否仍然存在
                    if self.concept_former:
                        valid_children = set()
                        for child_id in self.inheritance_graph[concept_id]:
                            child_concept = self.concept_former.get_concept(child_id)
                            if child_concept and concept_id in child_concept.parent_concepts:
                                valid_children.add(child_id)
                        self.inheritance_graph[concept_id] = valid_children
                
                logger.debug(f"继承关系更新完成: {concept_id}")
                return True
                
            except Exception as e:
                logger.error(f"更新继承关系失败: {str(e)}")
                return False
    
    def find_inheritance_paths(self, 
                             from_concept_id: str, 
                             to_concept_id: str) -> List[List[str]]:
        """查找两个概念之间的继承路径
        
        Args:
            from_concept_id: 起始概念ID
            to_concept_id: 目标概念ID
            
        Returns:
            继承路径列表，每条路径是一个概念ID列表
        """
        if not self.concept_former:
            return []
        
        # 使用广度优先搜索查找路径
        paths = []
        queue = [(from_concept_id, [from_concept_id])]
        visited = set()
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == to_concept_id:
                paths.append(path)
                continue
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 探索子概念
            if current_id in self.inheritance_graph:
                for child_id in self.inheritance_graph[current_id]:
                    if child_id not in visited:
                        new_path = path + [child_id]
                        queue.append((child_id, new_path))
        
        return paths
    
    def validate_inheritance_chain(self, chain: List[str]) -> Tuple[bool, List[str]]:
        """验证继承链的有效性
        
        Args:
            chain: 继承链，形式为 [最抽象概念, ..., 最具体概念]
            
        Returns:
            (是否有效, 错误信息列表)
        """
        if not self.concept_former:
            return False, ["缺少ConceptFormer引用"]
        
        errors = []
        
        # 检查概念存在性
        for concept_id in chain:
            concept = self.concept_former.get_concept(concept_id)
            if not concept:
                errors.append(f"概念不存在: {concept_id}")
                return False, errors
        
        # 检查继承关系连续性
        for i in range(len(chain) - 1):
            current_concept = self.concept_former.get_concept(chain[i])
            next_concept = self.concept_former.get_concept(chain[i + 1])
            
            # 检查是否存在继承关系
            if chain[i + 1] not in current_concept.child_concepts:
                errors.append(f"继承关系不连续: {chain[i]} -> {chain[i + 1]}")
            
            # 检查层次关系（应该是从抽象到具体）
            current_depth = self._get_concept_depth(current_concept)
            next_depth = self._get_concept_depth(next_concept)
            
            if current_depth >= next_depth:
                errors.append(f"概念层次不正确: {current_concept.name} (深度{current_depth}) -> "
                            f"{next_concept.name} (深度{next_depth})")
        
        return len(errors) == 0, errors
    
    def _get_concept_depth(self, concept: Concept) -> int:
        """获取概念层次深度"""
        depth_mapping = {
            ConceptLevel.INSTANCE: 0,
            ConceptLevel.BASIC: 1,
            ConceptLevel.SUPERORDINATE: 2,
            ConceptLevel.METACONCEPT: 3
        }
        return depth_mapping.get(concept.level, 0)
    
    def get_inheritance_statistics(self, concept_id: str) -> Dict[str, Any]:
        """获取概念的继承统计信息
        
        Args:
            concept_id: 概念ID
            
        Returns:
            继承统计信息
        """
        if not self.concept_former:
            return {}
        
        concept = self.concept_former.get_concept(concept_id)
        if not concept:
            return {}
        
        stats = {
            'concept_id': concept_id,
            'concept_name': concept.name,
            'parent_count': len(concept.parent_concepts),
            'child_count': len(concept.child_concepts),
            'inheritance_depth': self._get_concept_depth(concept),
            'inherited_features': len(self.get_inherited_features(concept_id) or {}),
            'own_features': len(concept.core_features)
        }
        
        # 添加继承关系统计
        if concept_id in self.inheritance_relations:
            relation = self.inheritance_relations[concept_id]
            stats.update({
                'inheritance_type': relation.inheritance_type.value,
                'inheritance_strength': relation.inheritance_strength,
                'confidence_level': relation.confidence_level,
                'access_count': relation.access_count
            })
        
        return stats
    
    def optimize_inheritance_structure(self, 
                                     concept_ids: List[str],
                                     max_iterations: int = 100) -> Dict[str, Any]:
        """优化继承结构
        
        分析并优化概念间的继承关系结构
        """
        optimization_result = {
            'iterations': 0,
            'conflicts_resolved': 0,
            'redundant_relations_removed': 0,
            'optimization_suggestions': []
        }
        
        if not self.concept_former:
            return optimization_result
        
        # 检测冗余的继承关系
        redundant_relations = self._find_redundant_relations(concept_ids)
        if redundant_relations:
            optimization_result['redundant_relations_removed'] = len(redundant_relations)
            optimization_result['optimization_suggestions'].append(
                f"检测到 {len(redundant_relations)} 个冗余继承关系"
            )
        
        # 检测循环继承
        cycles = self._find_inheritance_cycles(concept_ids)
        if cycles:
            optimization_result['optimization_suggestions'].append(
                f"检测到 {len(cycles)} 个可能的循环继承"
            )
        
        # 优化继承深度
        depth_violations = self._check_inheritance_depth_limits(concept_ids)
        if depth_violations:
            optimization_result['optimization_suggestions'].append(
                f"有 {len(depth_violations)} 个概念超出深度限制"
            )
        
        return optimization_result
    
    def _find_redundant_relations(self, concept_ids: List[str]) -> List[Tuple[str, str]]:
        """查找冗余的继承关系"""
        redundant_relations = []
        
        for concept_id in concept_ids:
            concept = self.concept_former.get_concept(concept_id)
            if not concept:
                continue
            
            # 检查是否有传递性冗余
            for parent_id in concept.parent_concepts:
                parent_concept = self.concept_former.get_concept(parent_id)
                if not parent_concept:
                    continue
                
                # 检查是否存在间接路径
                for grand_parent_id in parent_concept.parent_concepts:
                    if grand_parent_id in concept.parent_concepts:
                        # 存在传递性冗余
                        redundant_relations.append((concept_id, parent_id))
        
        return redundant_relations
    
    def _find_inheritance_cycles(self, concept_ids: List[str]) -> List[List[str]]:
        """查找继承循环"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            concept = self.concept_former.get_concept(node)
            if concept:
                for child_id in concept.child_concepts:
                    if child_id in concept_ids:
                        dfs(child_id, path.copy())
            
            rec_stack.remove(node)
        
        for concept_id in concept_ids:
            if concept_id not in visited:
                dfs(concept_id, [])
        
        return cycles
    
    def _check_inheritance_depth_limits(self, concept_ids: List[str]) -> List[str]:
        """检查继承深度限制"""
        violations = []
        
        for concept_id in concept_ids:
            concept = self.concept_former.get_concept(concept_id)
            if not concept:
                continue
            
            # 计算继承深度
            depth = self._calculate_concept_inheritance_depth(concept_id)
            
            # 检查是否超过默认限制
            if depth > 10:  # 默认最大深度
                violations.append(concept_id)
        
        return violations
    
    def _calculate_concept_inheritance_depth(self, concept_id: str, visited: Optional[Set[str]] = None) -> int:
        """计算概念的继承深度"""
        if visited is None:
            visited = set()
        
        if concept_id in visited:
            return 0  # 避免循环
        
        visited.add(concept_id)
        
        concept = self.concept_former.get_concept(concept_id)
        if not concept:
            return 0
        
        if not concept.parent_concepts:
            return 1
        
        max_depth = 0
        for parent_id in concept.parent_concepts:
            parent_depth = self._calculate_concept_inheritance_depth(parent_id, visited.copy())
            max_depth = max(max_depth, parent_depth)
        
        return max_depth + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取继承机制统计信息"""
        with self._lock:
            stats = self.stats.copy()
            
            # 继承规则使用统计
            stats['rule_usage'] = {
                rule_id: {
                    'usage_count': rule.usage_count,
                    'success_rate': rule.success_rate,
                    'average_quality': rule.average_inheritance_quality
                }
                for rule_id, rule in self.inheritance_rules.items()
            }
            
            # 继承类型分布
            inheritance_type_dist = defaultdict(int)
            for relation in self.inheritance_relations.values():
                inheritance_type_dist[relation.inheritance_type.value] += 1
            stats['inheritance_type_distribution'] = dict(inheritance_type_dist)
            
            # 缓存统计
            if self.cache:
                stats['cache'] = {
                    'size': len(self.cache.cache),
                    'max_size': self.cache.max_size
                }
            
            return stats
    
    def export_inheritance_graph(self, filepath: str) -> None:
        """导出继承图到文件
        
        Args:
            filepath: 输出文件路径
        """
        graph_data = {
            'inheritance_relations': {
                child_id: {
                    'parent_ids': list(relation.parent_concept_ids) if hasattr(relation, 'parent_concept_ids') else [],
                    'inheritance_type': relation.inheritance_type.value,
                    'inheritance_strength': relation.inheritance_strength,
                    'confidence_level': relation.confidence_level
                }
                for child_id, relation in self.inheritance_relations.items()
            },
            'inheritance_graph': {
                parent_id: list(children)
                for parent_id, children in self.inheritance_graph.items()
            },
            'export_time': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"继承图已导出到: {filepath}")
    
    def clear_cache(self) -> None:
        """清空继承缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("继承缓存已清空")
    
    def add_custom_inheritance_rule(self, rule: InheritanceRule) -> None:
        """添加自定义继承规则"""
        self.inheritance_rules[rule.rule_id] = rule
        logger.info(f"已添加自定义继承规则: {rule.name}")
    
    def remove_inheritance_relation(self, child_concept_id: str, 
                                  parent_concept_id: Optional[str] = None) -> bool:
        """移除继承关系
        
        Args:
            child_concept_id: 子概念ID
            parent_concept_id: 父概念ID，如果为None则移除所有父关系
            
        Returns:
            是否成功移除
        """
        with self._lock:
            try:
                if child_concept_id not in self.inheritance_relations:
                    return False
                
                relation = self.inheritance_relations[child_concept_id]
                
                if parent_concept_id is None:
                    # 移除所有继承关系
                    del self.inheritance_relations[child_concept_id]
                    
                    # 清理继承图
                    for parent_id in relation.parent_concept_ids:
                        self.inheritance_graph[parent_id].discard(child_concept_id)
                    
                    # 清理概念关系
                    if self.concept_former:
                        child_concept = self.concept_former.get_concept(child_concept_id)
                        if child_concept:
                            child_concept.parent_concepts.clear()
                            
                            for parent_id in relation.parent_concept_ids:
                                parent_concept = self.concept_former.get_concept(parent_id)
                                if parent_concept:
                                    parent_concept.child_concepts.discard(child_concept_id)
                
                else:
                    # 移除特定父关系
                    if parent_concept_id in relation.parent_concept_ids:
                        relation.parent_concept_ids.remove(parent_concept_id)
                        
                        # 更新继承图
                        self.inheritance_graph[parent_concept_id].discard(child_concept_id)
                        
                        # 更新概念关系
                        if self.concept_former:
                            child_concept = self.concept_former.get_concept(child_concept_id)
                            if child_concept:
                                child_concept.parent_concepts.discard(parent_concept_id)
                            
                            parent_concept = self.concept_former.get_concept(parent_concept_id)
                            if parent_concept:
                                parent_concept.child_concepts.discard(child_concept_id)
                
                # 使缓存失效
                if self.cache:
                    self.cache.invalidate_cache(child_concept_id)
                
                logger.info(f"继承关系已移除: {child_concept_id} -> {parent_concept_id}")
                return True
                
            except Exception as e:
                logger.error(f"移除继承关系失败: {str(e)}")
                return False