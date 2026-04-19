"""
抽象引擎 - AbstractionEngine
Abstraction Engine

这个类实现了概念抽象化的核心机制，支持从具体概念中提取一般性特征。
本实现基于现代认知科学和人工智能的抽象化理论：

理论基础：
- 抽象化理论（Abstraction Theory）
- 概念层次理论（Concept Hierarchical Theory）  
- 归纳推理理论（Inductive Reasoning Theory）
- 原型形成理论（Prototype Formation Theory）
- 概念网络理论（Concept Network Theory）
- 认知负荷理论（Cognitive Load Theory）
- 规则归纳理论（Rule Induction Theory）

技术特点：
- 多层次抽象化（实例→基本→上位→元概念）
- 基于特征的抽象化算法
- 支持概念相似度分析
- 实时抽象化处理
- 抽象质量评估机制
- 支持并发操作

Author: NeuroMinecraft Genesis Team
Date: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import threading
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import itertools
import json

from .concept_former import Concept, ConceptLevel, FormationMethod, ConceptFeature, ConceptFormer

logger = logging.getLogger(__name__)


class AbstractionStrategy(Enum):
    """抽象化策略枚举
    
    基于不同的认知抽象化机制：
    """
    FEATURE_GENERALIZATION = "feature_generalization"   # 特征泛化
    INSTANCE_ABSTRACTION = "instance_abstraction"       # 实例抽象
    RELATIONAL_ABSTRACTION = "relational_abstraction"   # 关系抽象
    STRUCTURAL_ABSTRACTION = "structural_abstraction"   # 结构抽象
    TEMPORAL_ABSTRACTION = "temporal_abstraction"       # 时序抽象
    CAUSAL_ABSTRACTION = "causal_abstraction"           # 因果抽象
    PROTOTYPICAL_ABSTRACTION = "prototypical_abstraction"  # 原型抽象
    RULE_BASED_ABSTRACTION = "rule_based_abstraction"   # 规则抽象
    ANALOGICAL_ABSTRACTION = "analogical_abstraction"   # 类比抽象


class AbstractionType(Enum):
    """抽象化类型枚举"""
    HORIZONTAL = "horizontal"           # 水平抽象：同级概念泛化
    VERTICAL = "vertical"               # 垂直抽象：概念层次上升
    LATERAL = "lateral"                 # 横向抽象：跨类概念抽象
    MULTI_LEVEL = "multi_level"         # 多层次抽象：跨多个层次
    TEMPORAL = "temporal"               # 时序抽象：基于时间演化
    STRUCTURAL = "structural"           # 结构抽象：基于内部结构
    FUNCTIONAL = "functional"           # 功能抽象：基于功能特性


@dataclass
class AbstractionRule:
    """抽象化规则定义
    
    描述抽象化操作的条件、策略和参数
    """
    rule_id: str
    name: str
    description: str
    abstraction_strategy: AbstractionStrategy
    abstraction_type: AbstractionType
    
    # 抽象化条件
    min_concepts_required: int = 2                    # 最少概念数量
    min_confidence_threshold: float = 0.6             # 最小置信度阈值
    similarity_threshold: float = 0.7                 # 相似度阈值
    feature_coverage_threshold: float = 0.5           # 特征覆盖率阈值
    
    # 抽象化参数
    weight_factors: Dict[str, float] = field(default_factory=dict)  # 权重因子
    priority_rules: List[str] = field(default_factory=list)         # 优先级规则
    
    # 质量评估
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # 质量指标
    success_indicators: List[str] = field(default_factory=list)      # 成功指标
    
    # 约束条件
    max_abstraction_depth: int = 5                    # 最大抽象深度
    min_evidence_strength: float = 0.3                # 最小证据强度
    
    # 性能统计
    usage_count: int = 0
    success_rate: float = 0.0
    average_abstraction_quality: float = 0.0
    
    def evaluate_abstraction_conditions(self, concepts: List[Concept]) -> Tuple[bool, float]:
        """评估抽象化条件
        
        Args:
            concepts: 待抽象的概念列表
            
        Returns:
            (是否满足条件, 适用性得分)
        """
        # 检查概念数量
        if len(concepts) < self.min_concepts_required:
            return False, 0.0
        
        # 检查置信度阈值
        avg_confidence = np.mean([c.confidence for c in concepts])
        if avg_confidence < self.min_confidence_threshold:
            return False, avg_confidence / self.min_confidence_threshold
        
        # 检查相似度
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sim = self._compute_concept_similarity(concepts[i], concepts[j])
                similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            if avg_similarity < self.similarity_threshold:
                return False, avg_similarity / self.similarity_threshold
        
        # 检查特征覆盖率
        feature_coverage = self._compute_feature_coverage(concepts)
        if feature_coverage < self.feature_coverage_threshold:
            return False, feature_coverage / self.feature_coverage_threshold
        
        # 计算整体适用性得分
        applicability_score = (
            len(concepts) / 10.0 +  # 概念数量得分
            avg_confidence +        # 置信度得分
            avg_similarity +        # 相似度得分
            feature_coverage        # 覆盖率得分
        ) / 4.0
        
        return True, min(1.0, applicability_score)
    
    def _compute_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算概念相似度"""
        # 特征相似度
        common_features = set(concept1.core_features.keys()) & set(concept2.core_features.keys())
        if not common_features:
            return 0.0
        
        feature_similarity = 0.0
        for feature_name in common_features:
            feat1 = concept1.core_features[feature_name]
            feat2 = concept2.core_features[feature_name]
            similarity = feat1.compute_similarity(feat2)
            feature_similarity += similarity
        
        feature_similarity /= len(common_features)
        
        # 结构相似度
        level_similarity = 1.0 if concept1.level == concept2.level else 0.5
        
        # 综合相似度
        return (feature_similarity + level_similarity) / 2
    
    def _compute_feature_coverage(self, concepts: List[Concept]) -> float:
        """计算特征覆盖率"""
        if not concepts:
            return 0.0
        
        # 收集所有特征
        all_features = set()
        for concept in concepts:
            all_features.update(concept.core_features.keys())
        
        if not all_features:
            return 1.0
        
        # 计算每个特征在概念中的出现频率
        total_feature_occurrences = 0
        for feature_name in all_features:
            occurrences = sum(1 for concept in concepts if feature_name in concept.core_features)
            total_feature_occurrences += occurrences
        
        # 覆盖率 = 实际出现次数 / 理论最大出现次数
        max_occurrences = len(all_features) * len(concepts)
        coverage = total_feature_occurrences / max_occurrences if max_occurrences > 0 else 0.0
        
        return coverage


@dataclass
class AbstractionResult:
    """抽象化结果"""
    abstract_concept_id: str
    source_concept_ids: List[str]
    abstraction_method: AbstractionStrategy
    abstraction_level: ConceptLevel
    confidence: float
    quality_score: float
    evidence_strength: float
    
    # 抽象化过程信息
    processing_time: float
    features_preserved: List[str]
    features_generalized: List[str]
    relationships_preserved: List[str]
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    rule_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractionCache:
    """抽象化缓存系统
    
    缓存抽象化结果以提高性能
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[Tuple[Tuple, ...], AbstractionResult] = {}
        self.access_times: Dict[Tuple, float] = {}
        self.access_count: Dict[Tuple, int] = defaultdict(int)
    
    def get(self, key: Tuple) -> Optional[AbstractionResult]:
        """获取缓存的抽象化结果"""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: Tuple, result: AbstractionResult) -> None:
        """存储抽象化结果到缓存"""
        # 检查缓存大小限制
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = result
        self.access_times[key] = time.time()
        self.access_count[key] = 1
    
    def _evict_lru(self) -> None:
        """移除最近最少使用的缓存项"""
        if not self.cache:
            return
        
        # 找到最少访问的键
        min_access_key = min(self.access_times.keys(), key=lambda k: (self.access_count[k], self.access_times[k]))
        del self.cache[min_access_key]
        del self.access_times[min_access_key]
        del self.access_count[min_access_key]
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_accesses': sum(self.access_count.values()),
            'hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not self.cache:
            return 0.0
        
        total_keys = len(self.cache)
        accessed_keys = sum(1 for count in self.access_count.values() if count > 1)
        return accessed_keys / total_keys if total_keys > 0 else 0.0


class AbstractionEngine:
    """抽象引擎
    
    负责将具体概念抽象为更一般的概念，支持多种抽象化策略和方法。
    基于认知科学的概念抽象化理论实现。
    
    主要功能：
    - 特征泛化抽象
    - 原型形成抽象  
    - 规则归纳抽象
    - 关系抽象
    - 多层次抽象化
    - 抽象质量评估
    """
    
    def __init__(self, 
                 concept_former: Optional[ConceptFormer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """初始化抽象引擎
        
        Args:
            concept_former: 概念形成器引用，用于概念管理
            config: 配置参数
                - cache_size: 缓存大小
                - max_abstraction_depth: 最大抽象深度
                - quality_threshold: 质量阈值
                - enable_cache: 是否启用缓存
        """
        self.config = config or {}
        self.concept_former = concept_former
        
        # 初始化抽象规则
        self.abstraction_rules = self._initialize_abstraction_rules()
        
        # 初始化缓存系统
        self.cache = AbstractionCache(
            max_size=self.config.get('cache_size', 1000)
        ) if self.config.get('enable_cache', True) else None
        
        # 配置参数
        self.max_abstraction_depth = self.config.get('max_abstraction_depth', 5)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.enable_performance_monitoring = self.config.get('performance_monitoring', True)
        
        # 线程锁（支持并发访问）
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_abstractions': 0,
            'successful_abstractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0,
            'abstraction_depth_distribution': defaultdict(int)
        }
        
        logger.info(f"AbstractionEngine 初始化完成 - 配置: {self.config}")
    
    def abstract_concept(self,
                       source_concepts: List[str],
                       target_level: ConceptLevel,
                       strategy: AbstractionStrategy = AbstractionStrategy.FEATURE_GENERALIZATION,
                       min_quality: float = None) -> Optional[str]:
        """抽象概念
        
        将一个或多个具体概念抽象为更高级别的概念
        
        Args:
            source_concepts: 源概念ID列表
            target_level: 目标抽象层次
            strategy: 抽象化策略
            min_quality: 最小质量阈值
            
        Returns:
            抽象概念的ID，失败返回None
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 抽象化过程失败
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # 输入验证
                if not source_concepts:
                    raise ValueError("源概念列表不能为空")
                
                if len(source_concepts) == 0:
                    raise ValueError("至少需要一个源概念")
                
                # 获取概念对象
                concepts = []
                for concept_id in source_concepts:
                    if self.concept_former:
                        concept = self.concept_former.get_concept(concept_id)
                    else:
                        raise ValueError("需要ConceptFormer来获取概念对象")
                    
                    if not concept:
                        raise ValueError(f"概念不存在: {concept_id}")
                    
                    concepts.append(concept)
                
                # 检查抽象深度限制
                current_max_level = max(c.level for c in concepts)
                if self._get_level_depth(target_level) - self._get_level_depth(current_max_level) > self.max_abstraction_depth:
                    raise ValueError(f"抽象深度超过限制: {self.max_abstraction_depth}")
                
                # 设置默认质量阈值
                if min_quality is None:
                    min_quality = self.quality_threshold
                
                logger.info(f"开始抽象化: {len(concepts)} 个概念 -> {target_level.value} (策略: {strategy.value})")
                
                # 检查缓存
                cache_key = self._generate_cache_key(source_concepts, target_level, strategy)
                if self.cache:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        logger.info("使用缓存的抽象化结果")
                        self.stats['cache_hits'] += 1
                        return cached_result.abstract_concept_id
                    self.stats['cache_misses'] += 1
                
                # 选择抽象规则
                rule = self._select_abstraction_rule(concepts, strategy, target_level)
                if not rule:
                    logger.warning(f"未找到适合的抽象规则: {strategy.value}")
                    return None
                
                # 评估抽象条件
                can_abstract, applicability_score = rule.evaluate_abstraction_conditions(concepts)
                if not can_abstract:
                    logger.warning(f"概念不满足抽象条件: {applicability_score:.3f}")
                    return None
                
                # 执行抽象化
                result = self._perform_abstraction(concepts, target_level, strategy, rule)
                
                if not result:
                    logger.error("抽象化执行失败")
                    return None
                
                # 质量检查
                if result.quality_score < min_quality:
                    logger.warning(f"抽象质量不达标: {result.quality_score:.3f} < {min_quality}")
                    return None
                
                # 缓存结果
                if self.cache:
                    self.cache.put(cache_key, result)
                
                # 更新统计
                self._update_abstraction_statistics(result, time.time() - start_time)
                
                logger.info(f"抽象化成功: {result.abstract_concept_id} (质量: {result.quality_score:.3f})")
                return result.abstract_concept_id
                
            except Exception as e:
                logger.error(f"抽象化失败: {str(e)}")
                return None
    
    def _initialize_abstraction_rules(self) -> Dict[str, AbstractionRule]:
        """初始化抽象化规则集合"""
        rules = {}
        
        # 特征泛化规则
        rules['feature_generalization'] = AbstractionRule(
            rule_id='feature_generalization',
            name='特征泛化抽象',
            description='通过识别共同特征并泛化具体值来实现抽象',
            abstraction_strategy=AbstractionStrategy.FEATURE_GENERALIZATION,
            abstraction_type=AbstractionType.VERTICAL,
            min_concepts_required=2,
            min_confidence_threshold=0.6,
            similarity_threshold=0.5,
            feature_coverage_threshold=0.3,
            weight_factors={
                'feature_importance': 0.4,
                'frequency_weight': 0.3,
                'confidence_weight': 0.3
            },
            max_abstraction_depth=3
        )
        
        # 原型形成规则
        rules['prototypical_abstraction'] = AbstractionRule(
            rule_id='prototypical_abstraction',
            name='原型形成抽象',
            description='基于典型特征形成原型概念',
            abstraction_strategy=AbstractionStrategy.PROTOTYPICAL_ABSTRACTION,
            abstraction_type=AbstractionType.HORIZONTAL,
            min_concepts_required=3,
            min_confidence_threshold=0.7,
            similarity_threshold=0.6,
            feature_coverage_threshold=0.4,
            weight_factors={
                'typicality_weight': 0.5,
                'frequency_weight': 0.3,
                'clarity_weight': 0.2
            },
            max_abstraction_depth=2
        )
        
        # 关系抽象规则
        rules['relational_abstraction'] = AbstractionRule(
            rule_id='relational_abstraction',
            name='关系抽象',
            description='通过分析概念间的关系模式进行抽象',
            abstraction_strategy=AbstractionStrategy.RELATIONAL_ABSTRACTION,
            abstraction_type=AbstractionType.STRUCTURAL,
            min_concepts_required=2,
            min_confidence_threshold=0.5,
            similarity_threshold=0.4,
            feature_coverage_threshold=0.2,
            weight_factors={
                'relationship_strength': 0.6,
                'structural_importance': 0.4
            },
            max_abstraction_depth=4
        )
        
        # 规则归纳规则
        rules['rule_based_abstraction'] = AbstractionRule(
            rule_id='rule_based_abstraction',
            name='规则归纳抽象',
            description='基于规则和模式识别进行抽象',
            abstraction_strategy=AbstractionStrategy.RULE_BASED_ABSTRACTION,
            abstraction_type=AbstractionType.HORIZONTAL,
            min_concepts_required=3,
            min_confidence_threshold=0.8,
            similarity_threshold=0.7,
            feature_coverage_threshold=0.5,
            weight_factors={
                'rule_confidence': 0.5,
                'pattern_strength': 0.3,
                'generalizability': 0.2
            },
            max_abstraction_depth=3
        )
        
        return rules
    
    def _select_abstraction_rule(self, 
                                concepts: List[Concept],
                                strategy: AbstractionStrategy,
                                target_level: ConceptLevel) -> Optional[AbstractionRule]:
        """选择最适合的抽象规则"""
        # 首先尝试匹配指定策略的规则
        for rule in self.abstraction_rules.values():
            if rule.abstraction_strategy == strategy:
                if rule.evaluate_abstraction_conditions(concepts)[0]:
                    rule.usage_count += 1
                    return rule
        
        # 如果指定策略不适用，选择其他可用规则
        best_rule = None
        best_score = 0.0
        
        for rule in self.abstraction_rules.values():
            can_use, score = rule.evaluate_abstraction_conditions(concepts)
            if can_use and score > best_score:
                best_rule = rule
                best_score = score
        
        if best_rule:
            best_rule.usage_count += 1
            return best_rule
        
        return None
    
    def _perform_abstraction(self,
                           concepts: List[Concept],
                           target_level: ConceptLevel,
                           strategy: AbstractionStrategy,
                           rule: AbstractionRule) -> Optional[AbstractionResult]:
        """执行具体的抽象化操作"""
        try:
            # 根据策略选择抽象方法
            if strategy == AbstractionStrategy.FEATURE_GENERALIZATION:
                return self._abstract_by_feature_generalization(concepts, target_level, rule)
            elif strategy == AbstractionStrategy.PROTOTYPICAL_ABSTRACTION:
                return self._abstract_by_prototype_formation(concepts, target_level, rule)
            elif strategy == AbstractionStrategy.RELATIONAL_ABSTRACTION:
                return self._abstract_by_relational_analysis(concepts, target_level, rule)
            elif strategy == AbstractionStrategy.RULE_BASED_ABSTRACTION:
                return self._abstract_by_rule_induction(concepts, target_level, rule)
            else:
                # 默认使用特征泛化
                return self._abstract_by_feature_generalization(concepts, target_level, rule)
                
        except Exception as e:
            logger.error(f"抽象化方法执行失败: {str(e)}")
            return None
    
    def _abstract_by_feature_generalization(self,
                                          concepts: List[Concept],
                                          target_level: ConceptLevel,
                                          rule: AbstractionRule) -> Optional[AbstractionResult]:
        """基于特征泛化的抽象化"""
        # 收集所有特征
        all_features = {}
        feature_frequency = defaultdict(int)
        
        for concept in concepts:
            for feature_name, feature in concept.core_features.items():
                if feature.abstractable:  # 只处理可抽象的特征
                    all_features[feature_name] = all_features.get(feature_name, [])
                    all_features[feature_name].append(feature)
                    feature_frequency[feature_name] += 1
        
        # 识别共同特征
        min_frequency = max(2, len(concepts) // 2)  # 至少一半概念包含
        common_features = {
            name: features for name, features in all_features.items()
            if feature_frequency[name] >= min_frequency
        }
        
        if not common_features:
            logger.warning("未找到足够的共同特征进行抽象")
            return None
        
        # 创建抽象特征
        abstract_features = []
        for feature_name, features in common_features.items():
            abstract_feature = self._generalize_feature(features, rule)
            if abstract_feature:
                abstract_features.append(abstract_feature)
        
        if not abstract_features:
            logger.warning("特征泛化失败")
            return None
        
        # 创建抽象概念
        abstract_name = self._generate_abstract_name(concepts, target_level)
        abstract_concept_id = self._create_abstract_concept(
            name=abstract_name,
            features=abstract_features,
            level=target_level,
            source_concepts=[c.concept_id for c in concepts]
        )
        
        if not abstract_concept_id:
            return None
        
        # 计算抽象质量
        quality_score = self._calculate_abstraction_quality(
            concepts, abstract_concept_id, rule, strategy
        )
        
        # 创建抽象结果
        result = AbstractionResult(
            abstract_concept_id=abstract_concept_id,
            source_concept_ids=[c.concept_id for c in concepts],
            abstraction_method=strategy,
            abstraction_level=target_level,
            confidence=np.mean([c.confidence for c in concepts]),
            quality_score=quality_score,
            evidence_strength=self._calculate_evidence_strength(concepts),
            processing_time=0.0,  # 将在调用者中设置
            features_preserved=list(common_features.keys()),
            features_generalized=[f.name for f in abstract_features],
            relationships_preserved=self._preserve_relationships(concepts, abstract_concept_id),
            rule_id=rule.rule_id
        )
        
        return result
    
    def _abstract_by_prototype_formation(self,
                                       concepts: List[Concept],
                                       target_level: ConceptLevel,
                                       rule: AbstractionRule) -> Optional[AbstractionResult]:
        """基于原型形成的抽象化"""
        # 计算每个概念的原型性得分
        prototype_scores = {}
        for concept in concepts:
            # 基于置信度、清晰度和稳定性计算原型性
            score = (
                concept.confidence * 0.4 +
                concept.clarity * 0.3 +
                concept.stability * 0.3
            )
            prototype_scores[concept.concept_id] = score
        
        # 选择最具代表性的概念作为原型基础
        prototype_concept = max(concepts, key=lambda c: prototype_scores[c.concept_id])
        
        # 收集其他概念的补充特征
        supplementary_features = {}
        for concept in concepts:
            if concept.concept_id != prototype_concept.concept_id:
                for feature_name, feature in concept.core_features.items():
                    if feature_name not in prototype_concept.core_features:
                        supplementary_features[feature_name] = feature
        
        # 创建原型概念
        prototype_name = f"原型_{self._generate_abstract_name(concepts, target_level)}"
        abstract_concept_id = self._create_prototype_concept(
            name=prototype_name,
            prototype_features=prototype_concept.core_features,
            supplementary_features=supplementary_features,
            level=target_level,
            source_concepts=[c.concept_id for c in concepts]
        )
        
        if not abstract_concept_id:
            return None
        
        # 计算质量得分
        quality_score = self._calculate_abstraction_quality(
            concepts, abstract_concept_id, rule, AbstractionStrategy.PROTOTYPICAL_ABSTRACTION
        )
        
        result = AbstractionResult(
            abstract_concept_id=abstract_concept_id,
            source_concept_ids=[c.concept_id for c in concepts],
            abstraction_method=AbstractionStrategy.PROTOTYPICAL_ABSTRACTION,
            abstraction_level=target_level,
            confidence=max([c.confidence for c in concepts]),  # 使用最高置信度
            quality_score=quality_score,
            evidence_strength=self._calculate_evidence_strength(concepts),
            processing_time=0.0,
            features_preserved=list(prototype_concept.core_features.keys()),
            features_generalized=list(supplementary_features.keys()),
            relationships_preserved=self._preserve_relationships(concepts, abstract_concept_id),
            rule_id=rule.rule_id
        )
        
        return result
    
    def _abstract_by_relational_analysis(self,
                                       concepts: List[Concept],
                                       target_level: ConceptLevel,
                                       rule: AbstractionRule) -> Optional[AbstractionResult]:
        """基于关系分析的抽象化"""
        # 分析概念间的关系模式
        relationship_patterns = self._analyze_relationship_patterns(concepts)
        
        if not relationship_patterns:
            logger.warning("未找到有意义的关系模式")
            return None
        
        # 创建关系抽象特征
        relational_features = []
        for pattern_name, pattern_data in relationship_patterns.items():
            # 创建描述关系模式的特征
            pattern_feature = ConceptFeature(
                name=f"relation_pattern_{pattern_name}",
                value=pattern_data,
                weight=pattern_data.get('strength', 0.5),
                confidence=pattern_data.get('confidence', 0.5),
                source="relational_abstraction",
                feature_type="relational",
                abstractable=False,  # 关系特征通常不可进一步抽象
                stable=True
            )
            relational_features.append(pattern_feature)
        
        # 合并原始概念的共享特征
        shared_features = self._extract_shared_features(concepts)
        
        # 创建关系抽象概念
        abstract_name = f"关系_{self._generate_abstract_name(concepts, target_level)}"
        abstract_concept_id = self._create_relational_concept(
            name=abstract_name,
            shared_features=shared_features,
            relational_features=relational_features,
            level=target_level,
            source_concepts=[c.concept_id for c in concepts]
        )
        
        if not abstract_concept_id:
            return None
        
        quality_score = self._calculate_abstraction_quality(
            concepts, abstract_concept_id, rule, AbstractionStrategy.RELATIONAL_ABSTRACTION
        )
        
        result = AbstractionResult(
            abstract_concept_id=abstract_concept_id,
            source_concept_ids=[c.concept_id for c in concepts],
            abstraction_method=AbstractionStrategy.RELATIONAL_ABSTRACTION,
            abstraction_level=target_level,
            confidence=np.mean([c.confidence for c in concepts]),
            quality_score=quality_score,
            evidence_strength=self._calculate_evidence_strength(concepts),
            processing_time=0.0,
            features_preserved=list(shared_features.keys()),
            features_generalized=list(relationship_patterns.keys()),
            relationships_preserved=list(relationship_patterns.keys()),
            rule_id=rule.rule_id
        )
        
        return result
    
    def _abstract_by_rule_induction(self,
                                  concepts: List[Concept],
                                  target_level: ConceptLevel,
                                  rule: AbstractionRule) -> Optional[AbstractionResult]:
        """基于规则归纳的抽象化"""
        # 从概念实例中归纳规则
        inductive_rules = self._induce_rules_from_concepts(concepts)
        
        if not inductive_rules:
            logger.warning("无法从概念中归纳出有效规则")
            return None
        
        # 将规则转换为概念特征
        rule_features = []
        for rule_name, rule_data in inductive_rules.items():
            rule_feature = ConceptFeature(
                name=f"rule_{rule_name}",
                value=rule_data,
                weight=rule_data.get('confidence', 0.5),
                confidence=rule_data.get('confidence', 0.5),
                source="rule_induction",
                feature_type="rule",
                abstractable=False,  # 规则通常不可进一步抽象
                stable=True
            )
            rule_features.append(rule_feature)
        
        # 创建规则概念
        abstract_name = f"规则_{self._generate_abstract_name(concepts, target_level)}"
        abstract_concept_id = self._create_rule_based_concept(
            name=abstract_name,
            rule_features=rule_features,
            level=target_level,
            source_concepts=[c.concept_id for c in concepts]
        )
        
        if not abstract_concept_id:
            return None
        
        quality_score = self._calculate_abstraction_quality(
            concepts, abstract_concept_id, rule, AbstractionStrategy.RULE_BASED_ABSTRACTION
        )
        
        result = AbstractionResult(
            abstract_concept_id=abstract_concept_id,
            source_concept_ids=[c.concept_id for c in concepts],
            abstraction_method=AbstractionStrategy.RULE_BASED_ABSTRACTION,
            abstraction_level=target_level,
            confidence=max([c.confidence for c in concepts]),  # 使用最高置信度
            quality_score=quality_score,
            evidence_strength=self._calculate_evidence_strength(concepts),
            processing_time=0.0,
            features_preserved=[],
            features_generalized=[f.name for f in rule_features],
            relationships_preserved=[],
            rule_id=rule.rule_id
        )
        
        return result
    
    def _generalize_feature(self, features: List[ConceptFeature], rule: AbstractionRule) -> Optional[ConceptFeature]:
        """泛化特征列表为抽象特征"""
        if not features:
            return None
        
        # 计算加权平均特征值
        total_weight = sum(f.weight * f.confidence for f in features)
        if total_weight == 0:
            return None
        
        # 泛化特征值
        if all(isinstance(f.value, (int, float)) for f in features):
            # 数值特征：计算统计摘要
            values = [f.value for f in features]
            weights = [f.weight * f.confidence for f in features]
            
            generalized_value = {
                'mean': np.average(values, weights=weights),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
            
            generalized_confidence = np.mean([f.confidence for f in features])
        else:
            # 分类特征：使用众数
            value_counts = Counter([f.value for f in features])
            mode_value = value_counts.most_common(1)[0][0]
            mode_frequency = value_counts[mode_value] / len(features)
            
            generalized_value = {
                'mode': mode_value,
                'frequency': mode_frequency,
                'alternatives': dict(value_counts.most_common(5))
            }
            
            generalized_confidence = mode_frequency
        
        # 创建抽象特征
        feature_name = features[0].name
        abstracted_feature = ConceptFeature(
            name=feature_name,
            value=generalized_value,
            weight=total_weight / len(features),
            confidence=generalized_confidence,
            source="abstraction_engine",
            feature_type="abstracted",
            abstractable=True,  # 抽象特征可以进一步抽象
            stable=np.mean([f.stable for f in features]) > 0.5
        )
        
        return abstracted_feature
    
    def _generate_abstract_name(self, concepts: List[Concept], target_level: ConceptLevel) -> str:
        """生成抽象概念名称"""
        # 基于概念名称模式生成抽象名称
        names = [c.name for c in concepts]
        
        if len(names) == 1:
            return f"抽象_{names[0]}"
        
        # 寻找共同前缀或后缀
        common_prefix = self._find_common_prefix(names)
        common_suffix = self._find_common_suffix(names)
        
        if common_prefix and len(common_prefix) > 2:
            return f"{common_prefix}类"
        elif common_suffix and len(common_suffix) > 2:
            return f"{common_suffix}类"
        else:
            # 生成描述性名称
            level_descriptors = {
                ConceptLevel.INSTANCE: "实例",
                ConceptLevel.BASIC: "基础",
                ConceptLevel.SUPERORDINATE: "上位",
                ConceptLevel.METACONCEPT: "元概念"
            }
            return f"{level_descriptors.get(target_level, '抽象')}概念"
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """找到字符串列表的公共前缀"""
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    def _find_common_suffix(self, strings: List[str]) -> str:
        """找到字符串列表的公共后缀"""
        if not strings:
            return ""
        
        suffix = strings[0]
        for s in strings[1:]:
            while not s.endswith(suffix):
                suffix = suffix[1:]
                if not suffix:
                    return ""
        return suffix
    
    def _create_abstract_concept(self,
                               name: str,
                               features: List[ConceptFeature],
                               level: ConceptLevel,
                               source_concepts: List[str]) -> Optional[str]:
        """创建抽象概念"""
        if not self.concept_former:
            logger.error("需要ConceptFormer来创建抽象概念")
            return None
        
        try:
            # 创建空的感知列表（抽象概念通常不直接对应感知）
            empty_perceptions = []
            
            # 使用特征作为预定义特征
            feature_dict = {f.name: f for f in features}
            
            # 形成抽象概念
            abstract_concept_id = self.concept_former.form_concept(
                name=name,
                perceptions=empty_perceptions,
                level=level,
                method=FormationMethod.INDUCTIVE,
                features=feature_dict
            )
            
            if abstract_concept_id:
                # 建立与源概念的关系
                abstract_concept = self.concept_former.get_concept(abstract_concept_id)
                for source_id in source_concepts:
                    abstract_concept.parent_concepts.add(source_id)
                    
                    # 也要在源概念中建立反向关系
                    source_concept = self.concept_former.get_concept(source_id)
                    if source_concept:
                        source_concept.child_concepts.add(abstract_concept_id)
                
                logger.info(f"成功创建抽象概念: {name} ({abstract_concept_id})")
            
            return abstract_concept_id
            
        except Exception as e:
            logger.error(f"创建抽象概念失败: {str(e)}")
            return None
    
    def _create_prototype_concept(self,
                                name: str,
                                prototype_features: Dict[str, ConceptFeature],
                                supplementary_features: Dict[str, ConceptFeature],
                                level: ConceptLevel,
                                source_concepts: List[str]) -> Optional[str]:
        """创建原型概念"""
        if not self.concept_former:
            return None
        
        try:
            # 合并原型特征和补充特征
            all_features = {}
            all_features.update(prototype_features)
            
            # 标记补充特征
            for feature_name, feature in supplementary_features.items():
                feature.source = "prototypical_supplement"
                all_features[feature_name] = feature
            
            # 创建概念
            abstract_concept_id = self.concept_former.form_concept(
                name=name,
                perceptions=[],
                level=level,
                method=FormationMethod.PROTOTYPICAL,
                features=all_features
            )
            
            if abstract_concept_id:
                # 建立关系
                abstract_concept = self.concept_former.get_concept(abstract_concept_id)
                for source_id in source_concepts:
                    abstract_concept.parent_concepts.add(source_id)
                    source_concept = self.concept_former.get_concept(source_id)
                    if source_concept:
                        source_concept.child_concepts.add(abstract_concept_id)
            
            return abstract_concept_id
            
        except Exception as e:
            logger.error(f"创建原型概念失败: {str(e)}")
            return None
    
    def _create_relational_concept(self,
                                 name: str,
                                 shared_features: Dict[str, ConceptFeature],
                                 relational_features: List[ConceptFeature],
                                 level: ConceptLevel,
                                 source_concepts: List[str]) -> Optional[str]:
        """创建关系概念"""
        if not self.concept_former:
            return None
        
        try:
            all_features = shared_features.copy()
            for feature in relational_features:
                all_features[feature.name] = feature
            
            abstract_concept_id = self.concept_former.form_concept(
                name=name,
                perceptions=[],
                level=level,
                method=FormationMethod.INDUCTIVE,
                features=all_features
            )
            
            if abstract_concept_id:
                abstract_concept = self.concept_former.get_concept(abstract_concept_id)
                for source_id in source_concepts:
                    abstract_concept.parent_concepts.add(source_id)
                    source_concept = self.concept_former.get_concept(source_id)
                    if source_concept:
                        source_concept.child_concepts.add(abstract_concept_id)
            
            return abstract_concept_id
            
        except Exception as e:
            logger.error(f"创建关系概念失败: {str(e)}")
            return None
    
    def _create_rule_based_concept(self,
                                 name: str,
                                 rule_features: List[ConceptFeature],
                                 level: ConceptLevel,
                                 source_concepts: List[str]) -> Optional[str]:
        """创建基于规则的抽象概念"""
        if not self.concept_former:
            return None
        
        try:
            feature_dict = {f.name: f for f in rule_features}
            
            abstract_concept_id = self.concept_former.form_concept(
                name=name,
                perceptions=[],
                level=level,
                method=FormationMethod.RULE_BASED,
                features=feature_dict
            )
            
            if abstract_concept_id:
                abstract_concept = self.concept_former.get_concept(abstract_concept_id)
                for source_id in source_concepts:
                    abstract_concept.parent_concepts.add(source_id)
                    source_concept = self.concept_former.get_concept(source_id)
                    if source_concept:
                        source_concept.child_concepts.add(abstract_concept_id)
            
            return abstract_concept_id
            
        except Exception as e:
            logger.error(f"创建规则概念失败: {str(e)}")
            return None
    
    def _analyze_relationship_patterns(self, concepts: List[Concept]) -> Dict[str, Dict[str, Any]]:
        """分析概念间的关系模式"""
        patterns = {}
        
        # 分析层次关系模式
        level_counts = Counter(c.level for c in concepts)
        if len(level_counts) > 1:
            patterns['hierarchical_distribution'] = {
                'levels': dict(level_counts),
                'diversity': len(level_counts),
                'entropy': self._calculate_entropy(level_counts.values())
            }
        
        # 分析相似性模式
        similarity_scores = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if self.concept_former:
                    similarity = self.concept_former._calculate_concept_similarity(
                        concepts[i].concept_id, concepts[j].concept_id
                    )
                    similarity_scores.append(similarity)
        
        if similarity_scores:
            patterns['similarity_pattern'] = {
                'mean_similarity': np.mean(similarity_scores),
                'similarity_variance': np.var(similarity_scores),
                'cohesion_score': 1.0 - np.var(similarity_scores)
            }
        
        # 分析特征共现模式
        feature_cooccurrence = self._analyze_feature_cooccurrence(concepts)
        if feature_cooccurrence:
            patterns['feature_cooccurrence'] = feature_cooccurrence
        
        return patterns
    
    def _analyze_feature_cooccurrence(self, concepts: List[Concept]) -> Dict[str, Any]:
        """分析特征共现模式"""
        # 收集所有特征对
        feature_pairs = defaultdict(int)
        total_pairs = 0
        
        for concept in concepts:
            features = list(concept.core_features.keys())
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pair = tuple(sorted([features[i], features[j]]))
                    feature_pairs[pair] += 1
                    total_pairs += 1
        
        if not feature_pairs:
            return {}
        
        # 找出高频共现特征对
        high_cooccurrence = {
            pair: count for pair, count in feature_pairs.items()
            if count >= len(concepts) // 2  # 至少一半概念中有此特征对
        }
        
        return {
            'total_pairs': total_pairs,
            'high_cooccurrence_pairs': len(high_cooccurrence),
            'cooccurrence_rate': len(high_cooccurrence) / len(feature_pairs) if feature_pairs else 0.0,
            'top_pairs': dict(sorted(high_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _extract_shared_features(self, concepts: List[Concept]) -> Dict[str, ConceptFeature]:
        """提取共享特征"""
        if not concepts:
            return {}
        
        # 收集所有特征
        all_features = defaultdict(list)
        for concept in concepts:
            for feature_name, feature in concept.core_features.items():
                all_features[feature_name].append(feature)
        
        # 识别共享特征（出现在至少一半概念中）
        min_occurrence = max(1, len(concepts) // 2)
        shared_features = {}
        
        for feature_name, feature_list in all_features.items():
            if len(feature_list) >= min_occurrence:
                # 合并共享特征
                if all(isinstance(f.value, (int, float)) for f in feature_list):
                    # 数值特征：计算统计值
                    values = [f.value for f in feature_list]
                    weights = [f.weight * f.confidence for f in feature_list]
                    total_weight = sum(weights)
                    
                    shared_value = np.average(values, weights=weights) if total_weight > 0 else np.mean(values)
                    shared_confidence = np.mean([f.confidence for f in feature_list])
                else:
                    # 分类特征：使用众数
                    value_counts = Counter([f.value for f in feature_list])
                    shared_value = value_counts.most_common(1)[0][0]
                    shared_confidence = value_counts.most_common(1)[0][1] / len(feature_list)
                
                shared_features[feature_name] = ConceptFeature(
                    name=feature_name,
                    value=shared_value,
                    weight=np.mean([f.weight for f in feature_list]),
                    confidence=shared_confidence,
                    source="shared_extraction",
                    feature_type="shared",
                    abstractable=True,
                    stable=True
                )
        
        return shared_features
    
    def _induce_rules_from_concepts(self, concepts: List[Concept]) -> Dict[str, Dict[str, Any]]:
        """从概念中归纳规则"""
        rules = {}
        
        # 分析实例特征模式
        all_instances = []
        for concept in concepts:
            all_instances.extend(concept.instances.values())
        
        if len(all_instances) < 3:
            return rules
        
        # 收集所有特征值
        feature_patterns = defaultdict(list)
        for instance in all_instances:
            for feature_name, value in instance.items():
                feature_patterns[feature_name].append(value)
        
        # 识别统计规则
        for feature_name, values in feature_patterns.items():
            if len(values) >= 3:
                if all(isinstance(v, (int, float)) for v in values):
                    # 数值特征规则
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    rules[f"{feature_name}_statistical"] = {
                        'type': 'statistical',
                        'mean': mean_val,
                        'std': std_val,
                        'confidence': min(1.0, 1.0 / (1.0 + std_val)) if std_val > 0 else 1.0,
                        'support': len(values)
                    }
                else:
                    # 分类特征规则
                    value_counts = Counter(values)
                    mode_value = value_counts.most_common(1)[0][0]
                    mode_freq = value_counts[mode_value] / len(values)
                    
                    if mode_freq > 0.6:  # 如果某个值出现频率超过60%
                        rules[f"{feature_name}_dominant"] = {
                            'type': 'dominant_value',
                            'value': mode_value,
                            'frequency': mode_freq,
                            'confidence': mode_freq,
                            'support': len(values)
                        }
        
        return rules
    
    def _calculate_abstraction_quality(self,
                                     source_concepts: List[Concept],
                                     abstract_concept_id: str,
                                     rule: AbstractionRule,
                                     strategy: AbstractionStrategy) -> float:
        """计算抽象化质量"""
        if not self.concept_former:
            return 0.0
        
        abstract_concept = self.concept_former.get_concept(abstract_concept_id)
        if not abstract_concept:
            return 0.0
        
        # 特征覆盖率
        source_features = set()
        for concept in source_concepts:
            source_features.update(concept.core_features.keys())
        
        abstract_features = set(abstract_concept.core_features.keys())
        coverage = len(abstract_features & source_features) / len(source_features) if source_features else 0.0
        
        # 概念凝聚性
        cohesion_score = 1.0
        if len(source_concepts) > 1:
            similarities = []
            for i in range(len(source_concepts)):
                for j in range(i + 1, len(source_concepts)):
                    sim = self.concept_former._calculate_concept_similarity(
                        source_concepts[i].concept_id, source_concepts[j].concept_id
                    )
                    similarities.append(sim)
            cohesion_score = np.mean(similarities) if similarities else 0.0
        
        # 抽象层次适配性
        max_source_level = max(c.level for c in source_concepts)
        level_advancement = self._get_level_depth(abstract_concept.level) - self._get_level_depth(max_source_level)
        level_score = min(1.0, level_advancement / 2.0)  # 最多提升2层得满分
        
        # 置信度保持
        avg_source_confidence = np.mean([c.confidence for c in source_concepts])
        confidence_preservation = abstract_concept.confidence / avg_source_confidence if avg_source_confidence > 0 else 0.0
        
        # 综合质量得分
        quality_weights = {
            'coverage': 0.3,
            'cohesion': 0.25,
            'level_advancement': 0.2,
            'confidence_preservation': 0.25
        }
        
        quality_score = (
            quality_weights['coverage'] * coverage +
            quality_weights['cohesion'] * cohesion_score +
            quality_weights['level_advancement'] * level_score +
            quality_weights['confidence_preservation'] * confidence_preservation
        )
        
        return min(1.0, quality_score)
    
    def _calculate_evidence_strength(self, concepts: List[Concept]) -> float:
        """计算证据强度"""
        if not concepts:
            return 0.0
        
        # 基于概念数量、置信度和稳定性的证据强度
        concept_count_factor = min(1.0, len(concepts) / 10.0)
        avg_confidence = np.mean([c.confidence for c in concepts])
        avg_stability = np.mean([c.stability for c in concepts])
        
        evidence_strength = (
            0.4 * concept_count_factor +
            0.4 * avg_confidence +
            0.2 * avg_stability
        )
        
        return min(1.0, evidence_strength)
    
    def _preserve_relationships(self, source_concepts: List[Concept], abstract_concept_id: str) -> List[str]:
        """保留关系信息"""
        preserved_relationships = []
        
        # 检查源概念之间的相似关系
        for i, concept1 in enumerate(source_concepts):
            for j, concept2 in enumerate(source_concepts[i+1:], i+1):
                # 查找相似概念
                if concept2.concept_id in concept1.similar_concepts:
                    preserved_relationships.append(f"similar_to_{concept2.concept_id}")
        
        return preserved_relationships
    
    def _get_level_depth(self, level: ConceptLevel) -> int:
        """获取概念层次深度"""
        depth_mapping = {
            ConceptLevel.INSTANCE: 0,
            ConceptLevel.BASIC: 1,
            ConceptLevel.SUPERORDINATE: 2,
            ConceptLevel.METACONCEPT: 3
        }
        return depth_mapping.get(level, 0)
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """计算熵值"""
        total = sum(values)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in values:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _generate_cache_key(self, source_concepts: List[str], target_level: ConceptLevel, strategy: AbstractionStrategy) -> Tuple:
        """生成缓存键"""
        return (tuple(sorted(source_concepts)), target_level.value, strategy.value)
    
    def _update_abstraction_statistics(self, result: AbstractionResult, processing_time: float) -> None:
        """更新抽象化统计信息"""
        self.stats['total_abstractions'] += 1
        self.stats['successful_abstractions'] += 1
        
        # 更新平均处理时间
        current_avg = self.stats['average_processing_time']
        total_ops = self.stats['total_abstractions']
        self.stats['average_processing_time'] = (current_avg * (total_ops - 1) + processing_time) / total_ops
        
        # 更新抽象深度分布
        depth = self._get_level_depth(result.abstraction_level)
        self.stats['abstraction_depth_distribution'][depth] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取抽象引擎统计信息"""
        with self._lock:
            stats = self.stats.copy()
            
            # 规则使用统计
            stats['rule_usage'] = {
                rule_id: {
                    'usage_count': rule.usage_count,
                    'success_rate': rule.success_rate,
                    'average_quality': rule.average_abstraction_quality
                }
                for rule_id, rule in self.abstraction_rules.items()
            }
            
            # 策略统计
            strategy_stats = defaultdict(int)
            for rule in self.abstraction_rules.values():
                strategy_stats[rule.abstraction_strategy.value] += rule.usage_count
            stats['strategy_usage'] = dict(strategy_stats)
            
            # 缓存统计
            if self.cache:
                stats['cache'] = self.cache.get_stats()
            
            return stats
    
    def get_abstraction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取抽象化历史记录"""
        # 这里可以实现抽象化历史记录功能
        # 目前返回缓存统计作为示例
        return [{
            'type': 'cache_stats',
            'data': self.cache.get_stats() if self.cache else {}
        }][:limit]
    
    def clear_cache(self) -> None:
        """清空抽象化缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("抽象化缓存已清空")
    
    def update_abstraction_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新抽象化规则"""
        if rule_id not in self.abstraction_rules:
            return False
        
        rule = self.abstraction_rules[rule_id]
        
        # 更新允许的参数
        updatable_params = {
            'min_concepts_required', 'min_confidence_threshold', 'similarity_threshold',
            'feature_coverage_threshold', 'weight_factors', 'max_abstraction_depth',
            'min_evidence_strength'
        }
        
        for key, value in updates.items():
            if key in updatable_params:
                setattr(rule, key, value)
        
        logger.info(f"抽象化规则 {rule_id} 已更新")
        return True
    
    def add_custom_abstraction_rule(self, rule: AbstractionRule) -> None:
        """添加自定义抽象化规则"""
        self.abstraction_rules[rule.rule_id] = rule
        logger.info(f"已添加自定义抽象化规则: {rule.name}")
    
    def validate_abstraction_conditions(self,
                                      concept_ids: List[str],
                                      target_level: ConceptLevel,
                                      strategy: AbstractionStrategy) -> Dict[str, Any]:
        """验证抽象化条件"""
        if not self.concept_former:
            return {'can_abstraction': False, 'reason': '缺少ConceptFormer引用'}
        
        # 获取概念
        concepts = []
        for concept_id in concept_ids:
            concept = self.concept_former.get_concept(concept_id)
            if not concept:
                return {'can_abstraction': False, 'reason': f'概念不存在: {concept_id}'}
            concepts.append(concept)
        
        # 评估条件
        rule = self._select_abstraction_rule(concepts, strategy, target_level)
        if not rule:
            return {'can_abstraction': False, 'reason': '未找到合适的抽象规则'}
        
        can_abstact, score = rule.evaluate_abstraction_conditions(concepts)
        
        return {
            'can_abstraction': can_abstact,
            'applicability_score': score,
            'selected_rule': rule.rule_id,
            'source_concepts': len(concepts),
            'target_level': target_level.value,
            'strategy': strategy.value
        }