"""
概念相似度计算引擎 - ConceptSimilarity
Concept Similarity Calculation Engine

这个类实现了多种概念相似度计算方法，支持不同类型的概念比较。
本实现基于现代认知科学和计算心理学的相似度理论：

理论基础：
- 相似度理论（Similarity Theory）
- 原型理论（Prototype Theory）
- 特征匹配理论（Feature Matching Theory）
- 认知心理学相似度模型（Cognitive Psychology Similarity Models）
- 语义相似度计算（Semantic Similarity Computation）
- 结构相似度理论（Structural Similarity Theory）
- 关联网络理论（Associative Network Theory）

技术特点：
- 多种相似度度量方法
- 基于特征的相似度计算
- 结构相似度分析
- 层次相似度评估
- 上下文敏感的相似度
- 高性能缓存机制
- 支持权重调整和参数优化

Author: NeuroMinecraft Genesis Team
Date: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import time
import math
import threading
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import itertools
import json

from .concept_former import Concept, ConceptLevel, ConceptFeature

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """相似度度量枚举
    
    基于认知心理学和计算科学的相似度度量方法：
    """
    # 基于特征的相似度
    FEATURE_OVERLAP = "feature_overlap"               # 特征重叠度
    FEATURE_CORRELATION = "feature_correlation"       # 特征相关性
    FEATURE_DISTANCE = "feature_distance"             # 特征距离
    
    # 结构相似度
    STRUCTURAL_ISOMORPHISM = "structural_isomorphism"  # 结构同构
    TOPOLOGICAL_SIMILARITY = "topological_similarity" # 拓扑相似度
    HIERARCHICAL_SIMILARITY = "hierarchical_similarity" # 层次相似度
    
    # 语义相似度
    SEMANTIC_DISTANCE = "semantic_distance"           # 语义距离
    CONCEPTUAL_DISTANCE = "conceptual_distance"       # 概念距离
    TAXONOMIC_SIMILARITY = "taxonomic_similarity"     # 分类相似度
    
    # 统计相似度
    COSINE_SIMILARITY = "cosine_similarity"           # 余弦相似度
    JACCARD_SIMILARITY = "jaccard_similarity"         # 杰卡德相似度
    PEARSON_CORRELATION = "pearson_correlation"       # 皮尔逊相关系数
    EUCLIDEAN_DISTANCE = "euclidean_distance"         # 欧几里得距离
    
    # 综合相似度
    WEIGHTED_COMPOSITE = "weighted_composite"         # 加权综合
    FUZZY_MATCHING = "fuzzy_matching"                 # 模糊匹配
    CONTEXT_SENSITIVE = "context_sensitive"           # 上下文敏感


class SimilarityContext(Enum):
    """相似度上下文枚举"""
    GENERAL = "general"               # 一般相似度
    SEMANTIC = "semantic"             # 语义相似度
    STRUCTURAL = "structural"         # 结构相似度
    FUNCTIONAL = "functional"         # 功能相似度
    TAXONOMIC = "taxonomic"           # 分类相似度
    TEMPORAL = "temporal"             # 时序相似度
    SPATIAL = "spatial"               # 空间相似度


@dataclass
class SimilarityWeights:
    """相似度权重配置"""
    # 特征权重
    feature_importance: float = 0.4         # 特征重要性权重
    feature_frequency: float = 0.2          # 特征频率权重
    feature_confidence: float = 0.2         # 特征置信度权重
    
    # 结构权重
    structural_weight: float = 0.1          # 结构权重
    hierarchical_weight: float = 0.1        # 层次权重
    
    # 语义权重
    semantic_weight: float = 0.1            # 语义权重
    
    # 归一化检查
    def normalize(self) -> 'SimilarityWeights':
        """归一化权重"""
        total = (self.feature_importance + self.feature_frequency + 
                self.feature_confidence + self.structural_weight + 
                self.hierarchical_weight + self.semantic_weight)
        
        if total > 0:
            self.feature_importance /= total
            self.feature_frequency /= total
            self.feature_confidence /= total
            self.structural_weight /= total
            self.hierarchical_weight /= total
            self.semantic_weight /= total
        
        return self


@dataclass
class SimilarityCalculationResult:
    """相似度计算结果"""
    concept1_id: str
    concept2_id: str
    overall_similarity: float
    metric_scores: Dict[str, float]
    confidence: float
    computation_time: float
    context: SimilarityContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SimilarityCache:
    """相似度缓存系统"""
    
    def __init__(self, max_size: int = 5000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
        self.cache: Dict[Tuple[str, str, SimilarityContext, SimilarityMetric], SimilarityCalculationResult] = {}
        self.access_times: Dict[Tuple, float] = {}
        self.access_count: Dict[Tuple, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def get(self, key: Tuple) -> Optional[SimilarityCalculationResult]:
        """获取缓存的相似度结果"""
        with self._lock:
            if key in self.cache:
                result = self.cache[key]
                # 检查是否过期
                if time.time() - result.timestamp < self.ttl:
                    self.access_times[key] = time.time()
                    self.access_count[key] += 1
                    return result
                else:
                    # 移除过期结果
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_count[key]
            return None
    
    def put(self, key: Tuple, result: SimilarityCalculationResult) -> None:
        """存储相似度结果到缓存"""
        with self._lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.max_size:
                self._evict_expired()
            
            self.cache[key] = result
            self.access_times[key] = time.time()
            self.access_count[key] = 1
    
    def _evict_expired(self) -> None:
        """移除过期或最少使用的缓存项"""
        current_time = time.time()
        
        # 移除过期项
        expired_keys = []
        for key, result in self.cache.items():
            if current_time - result.timestamp >= self.ttl:
                expired_keys.append(key)
        
        # 如果没有足够过期项，移除LRU项
        if len(self.cache) + len(expired_keys) - len(self.cache) >= self.max_size:
            lru_keys = sorted(self.access_times.keys(), 
                            key=lambda k: (self.access_count[k], self.access_times[k]))[:100]
            expired_keys.extend(lru_keys)
        
        # 移除选中的键
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_count:
                del self.access_count[key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_accesses = sum(self.access_count.values())
            unique_keys = len(self.cache)
            
            return {
                'size': unique_keys,
                'max_size': self.max_size,
                'total_accesses': total_accesses,
                'hit_rate': total_accesses / unique_keys if unique_keys > 0 else 0.0,
                'average_accesses_per_entry': total_accesses / unique_keys if unique_keys > 0 else 0.0
            }


class SimilarityCalculator(ABC):
    """相似度计算器基类"""
    
    @abstractmethod
    def calculate(self, 
                  concept1: Concept, 
                  concept2: Concept, 
                  weights: SimilarityWeights) -> float:
        """计算相似度"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取计算器名称"""
        pass


class FeatureBasedSimilarityCalculator(SimilarityCalculator):
    """基于特征的相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算基于特征的相似度"""
        # 获取共同特征
        common_features = set(concept1.core_features.keys()) & set(concept2.core_features.keys())
        
        if not common_features:
            return 0.0
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature_name in common_features:
            feature1 = concept1.core_features[feature_name]
            feature2 = concept2.core_features[feature_name]
            
            # 计算特征相似度
            feature_similarity = feature1.compute_similarity(feature2)
            
            # 计算特征权重
            importance_weight = feature1.weight * weights.feature_importance
            frequency_weight = min(feature1.confidence, feature2.confidence) * weights.feature_frequency
            confidence_weight = (feature1.confidence + feature2.confidence) / 2 * weights.feature_confidence
            
            feature_weight = importance_weight + frequency_weight + confidence_weight
            
            total_similarity += feature_similarity * feature_weight
            total_weight += feature_weight
        
        return total_similarity / total_weight if total_weight > 0 else 0.0
    
    def get_name(self) -> str:
        return "FeatureBasedSimilarity"


class StructuralSimilarityCalculator(SimilarityCalculator):
    """结构相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算结构相似度"""
        # 概念复杂度相似度
        complexity_similarity = 1.0 - abs(concept1.complexity - concept2.complexity)
        
        # 关系数量相似度
        rel1_count = len(concept1.parent_concepts) + len(concept1.child_concepts)
        rel2_count = len(concept2.parent_concepts) + len(concept2.child_concepts)
        
        max_relations = max(rel1_count, rel2_count, 1)
        relation_similarity = 1.0 - abs(rel1_count - rel2_count) / max_relations
        
        # 实例数量相似度
        inst1_count = len(concept1.instances)
        inst2_count = len(concept2.instances)
        
        max_instances = max(inst1_count, inst2_count, 1)
        instance_similarity = 1.0 - abs(inst1_count - inst2_count) / max_instances
        
        # 综合结构相似度
        structural_similarity = (
            0.4 * complexity_similarity +
            0.3 * relation_similarity +
            0.3 * instance_similarity
        )
        
        return structural_similarity * weights.structural_weight
    
    def get_name(self) -> str:
        return "StructuralSimilarity"


class HierarchicalSimilarityCalculator(SimilarityCalculator):
    """层次相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算层次相似度"""
        # 层次距离计算
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
        
        # 层次相似度
        hierarchical_similarity = 1.0 - distance / max_distance
        
        return hierarchical_similarity * weights.hierarchical_weight
    
    def get_name(self) -> str:
        return "HierarchicalSimilarity"


class SemanticSimilarityCalculator(SimilarityCalculator):
    """语义相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算语义相似度"""
        # 基于名称的语义相似度
        name_similarity = self._calculate_name_similarity(concept1.name, concept2.name)
        
        # 基于概念描述的相似度（如果有）
        description1 = concept1.metadata.get('description', '')
        description2 = concept2.metadata.get('description', '')
        
        if description1 and description2:
            description_similarity = self._calculate_text_similarity(description1, description2)
        else:
            description_similarity = 0.0
        
        # 基于标签的相似度
        tags1 = set(concept1.metadata.get('tags', []))
        tags2 = set(concept2.metadata.get('tags', []))
        
        if tags1 and tags2:
            tag_similarity = len(tags1 & tags2) / len(tags1 | tags2)
        else:
            tag_similarity = 0.0
        
        # 综合语义相似度
        semantic_similarity = (
            0.5 * name_similarity +
            0.3 * description_similarity +
            0.2 * tag_similarity
        )
        
        return semantic_similarity * weights.semantic_weight
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        
        # 计算编辑距离
        distance = self._levenshtein_distance(name1_lower, name2_lower)
        max_len = max(len(name1_lower), len(name2_lower))
        
        return 1.0 - distance / max_len if max_len > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的词袋模型相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_name(self) -> str:
        return "SemanticSimilarity"


class StatisticalSimilarityCalculator(SimilarityCalculator):
    """统计相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算统计相似度"""
        # 收集数值特征
        numerical_features1 = []
        numerical_features2 = []
        
        for feature_name, feature in concept1.core_features.items():
            if isinstance(feature.value, (int, float)):
                numerical_features1.append(feature.value)
        
        for feature_name, feature in concept2.core_features.items():
            if isinstance(feature.value, (int, float)):
                numerical_features2.append(feature.value)
        
        if not numerical_features1 or not numerical_features2:
            return 0.0
        
        # 确保长度一致
        min_length = min(len(numerical_features1), len(numerical_features2))
        if min_length == 0:
            return 0.0
        
        vec1 = np.array(numerical_features1[:min_length])
        vec2 = np.array(numerical_features2[:min_length])
        
        # 余弦相似度
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # 欧几里得距离相似度
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # 皮尔逊相关系数
        if len(vec1) > 1 and np.std(vec1) > 0 and np.std(vec2) > 0:
            pearson_corr = np.corrcoef(vec1, vec2)[0, 1]
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        else:
            pearson_corr = 0.0
        
        # 综合统计相似度
        statistical_similarity = (
            0.4 * max(0.0, cosine_sim) +
            0.3 * euclidean_sim +
            0.3 * max(0.0, pearson_corr)
        )
        
        return statistical_similarity
    
    def get_name(self) -> str:
        return "StatisticalSimilarity"


class FuzzyMatchingCalculator(SimilarityCalculator):
    """模糊匹配相似度计算器"""
    
    def calculate(self, concept1: Concept, concept2: Concept, weights: SimilarityWeights) -> float:
        """计算模糊匹配相似度"""
        # 基于模糊逻辑的特征匹配
        fuzzy_similarities = []
        
        all_features = set(concept1.core_features.keys()) | set(concept2.core_features.keys())
        
        for feature_name in all_features:
            similarity = 0.0
            
            if (feature_name in concept1.core_features and 
                feature_name in concept2.core_features):
                # 精确匹配
                feat1 = concept1.core_features[feature_name]
                feat2 = concept2.core_features[feature_name]
                similarity = feat1.compute_similarity(feat2)
            elif feature_name in concept1.core_features:
                # 概念1独有特征 - 基于特征名称的模糊匹配
                similarity = self._fuzzy_feature_match(concept1.core_features[feature_name], concept2)
            elif feature_name in concept2.core_features:
                # 概念2独有特征
                similarity = self._fuzzy_feature_match(concept2.core_features[feature_name], concept1)
            
            fuzzy_similarities.append(similarity)
        
        return np.mean(fuzzy_similarities) if fuzzy_similarities else 0.0
    
    def _fuzzy_feature_match(self, feature: ConceptFeature, other_concept: Concept) -> float:
        """模糊特征匹配"""
        # 基于特征值的模糊匹配
        max_similarity = 0.0
        
        for other_feature in other_concept.core_features.values():
            # 类型相似性
            type_similarity = 1.0 if type(feature.value) == type(other_feature.value) else 0.5
            
            # 值相似性
            if isinstance(feature.value, str) and isinstance(other_feature.value, str):
                value_similarity = self._string_similarity(feature.value, other_feature.value)
            elif isinstance(feature.value, (int, float)) and isinstance(other_feature.value, (int, float)):
                max_val = max(abs(feature.value), abs(other_feature.value))
                value_similarity = 1.0 - abs(feature.value - other_feature.value) / max_val if max_val > 0 else 1.0
            else:
                value_similarity = 0.0
            
            # 置信度权重
            confidence_weight = (feature.confidence + other_feature.confidence) / 2
            
            total_similarity = type_similarity * value_similarity * confidence_weight
            max_similarity = max(max_similarity, total_similarity)
        
        return max_similarity
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """字符串相似度"""
        # 基于编辑距离的字符串相似度
        distance = self._levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        
        return 1.0 - distance / max_len if max_len > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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
    
    def get_name(self) -> str:
        return "FuzzyMatching"


class ConceptSimilarity:
    """概念相似度计算引擎
    
    负责计算概念之间的相似度，支持多种相似度度量方法和上下文。
    基于认知心理学的相似度理论实现。
    
    主要功能：
    - 多维度相似度计算
    - 可配置相似度权重
    - 上下文敏感的相似度评估
    - 高性能缓存机制
    - 相似度质量评估
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化相似度计算引擎
        
        Args:
            config: 配置参数
                - cache_size: 缓存大小
                - cache_ttl: 缓存生存时间
                - default_weights: 默认权重
                - enable_cache: 是否启用缓存
                - precision_threshold: 精度阈值
        """
        self.config = config or {}
        
        # 初始化相似度计算器
        self.calculators = self._initialize_calculators()
        
        # 初始化缓存系统
        self.cache = SimilarityCache(
            max_size=self.config.get('cache_size', 5000),
            ttl=self.config.get('cache_ttl', 3600)
        ) if self.config.get('enable_cache', True) else None
        
        # 默认权重
        self.default_weights = SimilarityWeights(
            **self.config.get('default_weights', {})
        ).normalize()
        
        # 配置参数
        self.precision_threshold = self.config.get('precision_threshold', 0.001)
        self.enable_performance_monitoring = self.config.get('performance_monitoring', True)
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_calculation_time': 0.0,
            'metric_usage': defaultdict(int)
        }
        
        logger.info(f"ConceptSimilarity 初始化完成 - 配置: {self.config}")
    
    def _initialize_calculators(self) -> Dict[str, SimilarityCalculator]:
        """初始化相似度计算器"""
        calculators = {
            'feature_based': FeatureBasedSimilarityCalculator(),
            'structural': StructuralSimilarityCalculator(),
            'hierarchical': HierarchicalSimilarityCalculator(),
            'semantic': SemanticSimilarityCalculator(),
            'statistical': StatisticalSimilarityCalculator(),
            'fuzzy_matching': FuzzyMatchingCalculator()
        }
        return calculators
    
    def calculate_similarity(self,
                           concept1: Union[Concept, str],
                           concept2: Union[Concept, str],
                           context: SimilarityContext = SimilarityContext.GENERAL,
                           weights: Optional[SimilarityWeights] = None,
                           metrics: Optional[List[str]] = None) -> SimilarityCalculationResult:
        """计算概念相似度
        
        Args:
            concept1: 概念对象或概念ID
            concept2: 概念对象或概念ID
            context: 相似度上下文
            weights: 相似度权重配置
            metrics: 指定使用的度量方法
            
        Returns:
            相似度计算结果
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 计算过程失败
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # 获取概念对象
                if isinstance(concept1, str):
                    raise ValueError("需要ConceptFormer来获取概念对象")
                if isinstance(concept2, str):
                    raise ValueError("需要ConceptFormer来获取概念对象")
                
                # 输入验证
                if not concept1 or not concept2:
                    raise ValueError("概念对象不能为空")
                
                # 使用默认权重（如果未指定）
                if weights is None:
                    weights = self.default_weights
                
                # 生成缓存键
                cache_key = (concept1.concept_id, concept2.concept_id, context, tuple(metrics or []))
                
                # 检查缓存
                if self.cache:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        self.stats['cache_hits'] += 1
                        return cached_result
                    self.stats['cache_misses'] += 1
                
                logger.debug(f"计算相似度: {concept1.name} vs {concept2.name} (上下文: {context.value})")
                
                # 计算各种相似度指标
                metric_scores = {}
                
                if metrics:
                    # 使用指定的度量方法
                    for metric_name in metrics:
                        if metric_name in self.calculators:
                            calculator = self.calculators[metric_name]
                            score = calculator.calculate(concept1, concept2, weights)
                            metric_scores[metric_name] = score
                            self.stats['metric_usage'][metric_name] += 1
                else:
                    # 使用所有可用的度量方法
                    for metric_name, calculator in self.calculators.items():
                        # 根据上下文过滤合适的度量方法
                        if self._is_metric_suitable_for_context(metric_name, context):
                            score = calculator.calculate(concept1, concept2, weights)
                            metric_scores[metric_name] = score
                            self.stats['metric_usage'][metric_name] += 1
                
                # 计算综合相似度
                overall_similarity = self._compute_overall_similarity(metric_scores, context, weights)
                
                # 计算置信度
                confidence = self._calculate_confidence(metric_scores, concept1, concept2)
                
                # 创建结果
                result = SimilarityCalculationResult(
                    concept1_id=concept1.concept_id,
                    concept2_id=concept2.concept_id,
                    overall_similarity=overall_similarity,
                    metric_scores=metric_scores,
                    confidence=confidence,
                    computation_time=time.time() - start_time,
                    context=context,
                    metadata={
                        'concept1_name': concept1.name,
                        'concept2_name': concept2.name,
                        'weights_used': weights.__dict__,
                        'calculators_used': list(metric_scores.keys())
                    }
                )
                
                # 缓存结果
                if self.cache:
                    self.cache.put(cache_key, result)
                
                # 更新统计
                self._update_calculation_statistics(result.computation_time)
                
                return result
                
            except Exception as e:
                logger.error(f"相似度计算失败: {str(e)}")
                raise
    
    def _is_metric_suitable_for_context(self, metric_name: str, context: SimilarityContext) -> bool:
        """检查度量方法是否适合指定上下文"""
        context_suitability = {
            SimilarityContext.GENERAL: ['feature_based', 'structural', 'hierarchical', 'semantic'],
            SimilarityContext.SEMANTIC: ['semantic', 'feature_based', 'fuzzy_matching'],
            SimilarityContext.STRUCTURAL: ['structural', 'hierarchical', 'feature_based'],
            SimilarityContext.FUNCTIONAL: ['feature_based', 'statistical'],
            SimilarityContext.TAXONOMIC: ['hierarchical', 'semantic', 'feature_based'],
            SimilarityContext.TEMPORAL: ['statistical', 'feature_based'],
            SimilarityContext.SPATIAL: ['statistical', 'feature_based']
        }
        
        suitable_metrics = context_suitability.get(context, [])
        return metric_name in suitable_metrics
    
    def _compute_overall_similarity(self,
                                  metric_scores: Dict[str, float],
                                  context: SimilarityContext,
                                  weights: SimilarityWeights) -> float:
        """计算综合相似度"""
        if not metric_scores:
            return 0.0
        
        # 根据上下文调整权重
        context_weights = self._get_context_weights(context)
        
        # 计算加权平均
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, score in metric_scores.items():
            metric_weight = context_weights.get(metric_name, 0.1)
            total_weighted_score += score * metric_weight
            total_weight += metric_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _get_context_weights(self, context: SimilarityContext) -> Dict[str, float]:
        """根据上下文获取权重配置"""
        context_weight_configs = {
            SimilarityContext.GENERAL: {
                'feature_based': 0.3,
                'structural': 0.2,
                'hierarchical': 0.2,
                'semantic': 0.2,
                'statistical': 0.1,
                'fuzzy_matching': 0.0
            },
            SimilarityContext.SEMANTIC: {
                'feature_based': 0.2,
                'structural': 0.1,
                'hierarchical': 0.1,
                'semantic': 0.5,
                'statistical': 0.0,
                'fuzzy_matching': 0.1
            },
            SimilarityContext.STRUCTURAL: {
                'feature_based': 0.1,
                'structural': 0.5,
                'hierarchical': 0.3,
                'semantic': 0.1,
                'statistical': 0.0,
                'fuzzy_matching': 0.0
            },
            SimilarityContext.TAXONOMIC: {
                'feature_based': 0.2,
                'structural': 0.1,
                'hierarchical': 0.6,
                'semantic': 0.1,
                'statistical': 0.0,
                'fuzzy_matching': 0.0
            },
            SimilarityContext.FUNCTIONAL: {
                'feature_based': 0.4,
                'structural': 0.1,
                'hierarchical': 0.1,
                'semantic': 0.2,
                'statistical': 0.2,
                'fuzzy_matching': 0.0
            }
        }
        
        return context_weight_configs.get(context, context_weight_configs[SimilarityContext.GENERAL])
    
    def _calculate_confidence(self,
                            metric_scores: Dict[str, float],
                            concept1: Concept,
                            concept2: Concept) -> float:
        """计算相似度置信度"""
        # 基于指标数量和一致性的置信度
        num_metrics = len(metric_scores)
        if num_metrics == 0:
            return 0.0
        
        # 指标分数方差（一致性）
        scores = list(metric_scores.values())
        score_variance = np.var(scores)
        consistency = 1.0 - min(1.0, score_variance)
        
        # 概念置信度因子
        concept_confidence = (concept1.confidence + concept2.confidence) / 2
        
        # 数据充分性因子
        data_adequacy = min(1.0, (
            len(concept1.core_features) / 10.0 +
            len(concept2.core_features) / 10.0 +
            len(concept1.instances) / 5.0 +
            len(concept2.instances) / 5.0
        ) / 4.0)
        
        # 综合置信度
        confidence = (
            0.4 * consistency +
            0.3 * concept_confidence +
            0.3 * data_adequacy
        )
        
        return min(1.0, confidence)
    
    def _update_calculation_statistics(self, computation_time: float) -> None:
        """更新计算统计信息"""
        self.stats['total_calculations'] += 1
        
        # 更新平均计算时间
        current_avg = self.stats['average_calculation_time']
        total_ops = self.stats['total_calculations']
        self.stats['average_calculation_time'] = (current_avg * (total_ops - 1) + computation_time) / total_ops
    
    def batch_calculate_similarity(self,
                                 concept_pairs: List[Tuple[Union[Concept, str], Union[Concept, str]]],
                                 context: SimilarityContext = SimilarityContext.GENERAL,
                                 weights: Optional[SimilarityWeights] = None,
                                 progress_callback: Optional[Callable[[int, int], None]] = None) -> List[SimilarityCalculationResult]:
        """批量计算概念相似度"""
        results = []
        total_pairs = len(concept_pairs)
        
        for i, (concept1, concept2) in enumerate(concept_pairs):
            try:
                result = self.calculate_similarity(concept1, concept2, context, weights)
                results.append(result)
                
                # 进度回调
                if progress_callback and i % 10 == 0:
                    progress_callback(i + 1, total_pairs)
                    
            except Exception as e:
                logger.error(f"批量计算中第 {i+1} 对概念相似度失败: {str(e)}")
                # 创建错误结果
                error_result = SimilarityCalculationResult(
                    concept1_id=str(concept1) if isinstance(concept1, str) else getattr(concept1, 'concept_id', 'unknown'),
                    concept2_id=str(concept2) if isinstance(concept2, str) else getattr(concept2, 'concept_id', 'unknown'),
                    overall_similarity=0.0,
                    metric_scores={},
                    confidence=0.0,
                    computation_time=0.0,
                    context=context,
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def find_most_similar_concepts(self,
                                 target_concept: Union[Concept, str],
                                 candidate_concepts: List[Union[Concept, str]],
                                 top_k: int = 5,
                                 context: SimilarityContext = SimilarityContext.GENERAL,
                                 weights: Optional[SimilarityWeights] = None,
                                 threshold: float = 0.0) -> List[Tuple[Union[Concept, str], float]]:
        """查找最相似的概念
        
        Args:
            target_concept: 目标概念
            candidate_concepts: 候选概念列表
            top_k: 返回前k个最相似的概念
            context: 相似度上下文
            weights: 权重配置
            threshold: 相似度阈值
            
        Returns:
            (概念, 相似度) 的列表，按相似度降序排列
        """
        if isinstance(target_concept, str):
            raise ValueError("需要ConceptFormer来获取概念对象")
        
        similarities = []
        
        for candidate in candidate_concepts:
            if isinstance(candidate, str):
                raise ValueError("需要ConceptFormer来获取概念对象")
            
            result = self.calculate_similarity(target_concept, candidate, context, weights)
            
            if result.overall_similarity >= threshold:
                similarities.append((candidate, result.overall_similarity))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_concepts_by_similarity(self,
                                     concepts: List[Union[Concept, str]],
                                     similarity_threshold: float = 0.7,
                                     context: SimilarityContext = SimilarityContext.GENERAL,
                                     weights: Optional[SimilarityWeights] = None) -> List[List[Union[Concept, str]]]:
        """基于相似度对概念进行聚类
        
        Args:
            concepts: 概念列表
            similarity_threshold: 相似度阈值
            context: 相似度上下文
            weights: 权重配置
            
        Returns:
            聚类结果列表
        """
        if len(concepts) < 2:
            return [concepts]
        
        # 构建相似度矩阵
        similarity_matrix = {}
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                if isinstance(concept1, str) or isinstance(concept2, str):
                    raise ValueError("需要ConceptFormer来获取概念对象")
                
                result = self.calculate_similarity(concept1, concept2, context, weights)
                similarity_matrix[(i, j)] = result.overall_similarity
                similarity_matrix[(j, i)] = result.overall_similarity
        
        # 简单的层次聚类
        clusters = self._hierarchical_clustering(concepts, similarity_matrix, similarity_threshold)
        
        return clusters
    
    def _hierarchical_clustering(self,
                               concepts: List[Union[Concept, str]],
                               similarity_matrix: Dict[Tuple[int, int], float],
                               threshold: float) -> List[List[Union[Concept, str]]]:
        """层次聚类算法"""
        # 初始化：每个概念为一个簇
        clusters = [[i] for i in range(len(concepts))]
        
        while len(clusters) > 1:
            # 找到最相似的两个簇
            max_similarity = -1
            merge_indices = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算两个簇间的平均相似度
                    cluster_similarity = 0.0
                    count = 0
                    
                    for idx1 in clusters[i]:
                        for idx2 in clusters[j]:
                            sim_key = (min(idx1, idx2), max(idx1, idx2))
                            if sim_key in similarity_matrix:
                                cluster_similarity += similarity_matrix[sim_key]
                                count += 1
                    
                    if count > 0:
                        avg_similarity = cluster_similarity / count
                        if avg_similarity > max_similarity:
                            max_similarity = avg_similarity
                            merge_indices = (i, j)
            
            # 检查是否需要合并
            if max_similarity < threshold:
                break
            
            # 合并最相似的两个簇
            i, j = merge_indices
            merged_cluster = clusters[i] + clusters[j]
            
            # 移除旧簇，添加新簇
            clusters.pop(j)
            clusters.pop(i)
            clusters.append(merged_cluster)
        
        # 转换为原始概念列表
        result_clusters = []
        for cluster in clusters:
            concept_cluster = [concepts[idx] for idx in cluster]
            result_clusters.append(concept_cluster)
        
        return result_clusters
    
    def get_similarity_network(self,
                             concepts: List[Union[Concept, str]],
                             similarity_threshold: float = 0.5,
                             context: SimilarityContext = SimilarityContext.GENERAL,
                             weights: Optional[SimilarityWeights] = None) -> Dict[str, Dict[str, float]]:
        """构建概念相似度网络
        
        Args:
            concepts: 概念列表
            similarity_threshold: 相似度阈值
            context: 相似度上下文
            weights: 权重配置
            
        Returns:
            相似度网络（邻接表表示）
        """
        network = {}
        
        for concept in concepts:
            if isinstance(concept, str):
                raise ValueError("需要ConceptFormer来获取概念对象")
            network[concept.concept_id] = {}
        
        # 计算所有概念对的相似度
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                result = self.calculate_similarity(concept1, concept2, context, weights)
                
                if result.overall_similarity >= similarity_threshold:
                    network[concept1.concept_id][concept2.concept_id] = result.overall_similarity
                    network[concept2.concept_id][concept1.concept_id] = result.overall_similarity
        
        return network
    
    def analyze_similarity_patterns(self,
                                  concepts: List[Union[Concept, str]],
                                  context: SimilarityContext = SimilarityContext.GENERAL,
                                  weights: Optional[SimilarityWeights] = None) -> Dict[str, Any]:
        """分析概念间的相似度模式
        
        Args:
            concepts: 概念列表
            context: 相似度上下文
            weights: 权重配置
            
        Returns:
            相似度模式分析结果
        """
        if len(concepts) < 2:
            return {'error': '需要至少2个概念进行分析'}
        
        # 计算所有概念对的相似度
        similarities = []
        metric_distributions = defaultdict(list)
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                result = self.calculate_similarity(concept1, concept2, context, weights)
                similarities.append(result.overall_similarity)
                
                # 收集各指标分数分布
                for metric_name, score in result.metric_scores.items():
                    metric_distributions[metric_name].append(score)
        
        # 分析相似度分布
        similarities = np.array(similarities)
        
        analysis_result = {
            'overall_statistics': {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'median_similarity': float(np.median(similarities))
            },
            'similarity_distribution': {
                'high_similarity_pairs': int(np.sum(similarities > 0.8)),
                'medium_similarity_pairs': int(np.sum((similarities > 0.5) & (similarities <= 0.8))),
                'low_similarity_pairs': int(np.sum(similarities <= 0.5))
            },
            'metric_analysis': {}
        }
        
        # 分析各指标的分布
        for metric_name, scores in metric_distributions.items():
            if scores:
                analysis_result['metric_analysis'][metric_name] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'contribution_ratio': float(np.mean(scores) / np.mean(similarities)) if np.mean(similarities) > 0 else 0.0
                }
        
        return analysis_result
    
    def optimize_weights(self,
                       training_pairs: List[Tuple[Union[Concept, str], Union[Concept, str], float]],
                       validation_pairs: Optional[List[Tuple[Union[Concept, str], Union[Concept, str], float]]] = None,
                       iterations: int = 100,
                       learning_rate: float = 0.01) -> SimilarityWeights:
        """优化相似度权重
        
        基于训练数据优化权重配置
        
        Args:
            training_pairs: 训练数据 [(概念1, 概念2, 真实相似度), ...]
            validation_pairs: 验证数据
            iterations: 迭代次数
            learning_rate: 学习率
            
        Returns:
            优化后的权重配置
        """
        logger.info("开始权重优化...")
        
        # 初始化权重
        current_weights = SimilarityWeights()
        
        # 梯度下降优化
        for iteration in range(iterations):
            total_loss = 0.0
            gradients = {
                'feature_importance': 0.0,
                'feature_frequency': 0.0,
                'feature_confidence': 0.0,
                'structural_weight': 0.0,
                'hierarchical_weight': 0.0,
                'semantic_weight': 0.0
            }
            
            # 前向传播和损失计算
            for concept1, concept2, true_similarity in training_pairs:
                if isinstance(concept1, str) or isinstance(concept2, str):
                    continue
                
                result = self.calculate_similarity(concept1, concept2, SimilarityContext.GENERAL, current_weights)
                predicted_similarity = result.overall_similarity
                
                # 计算损失（MSE）
                loss = (predicted_similarity - true_similarity) ** 2
                total_loss += loss
                
                # 简化的梯度计算（实际中需要更复杂的反向传播）
                error = predicted_similarity - true_similarity
                
                # 更新梯度（这里使用简化的方法）
                for metric_name, score in result.metric_scores.items():
                    gradient_key = self._get_weight_key_for_metric(metric_name)
                    if gradient_key in gradients:
                        gradients[gradient_key] += error * score / len(result.metric_scores)
            
            # 计算平均梯度
            num_samples = len(training_pairs)
            for key in gradients:
                gradients[key] /= num_samples
            
            # 更新权重
            current_weights.feature_importance -= learning_rate * gradients['feature_importance']
            current_weights.feature_frequency -= learning_rate * gradients['feature_frequency']
            current_weights.feature_confidence -= learning_rate * gradients['feature_confidence']
            current_weights.structural_weight -= learning_rate * gradients['structural_weight']
            current_weights.hierarchical_weight -= learning_rate * gradients['hierarchical_weight']
            current_weights.semantic_weight -= learning_rate * gradients['semantic_weight']
            
            # 确保权重非负
            current_weights.feature_importance = max(0.0, current_weights.feature_importance)
            current_weights.feature_frequency = max(0.0, current_weights.feature_frequency)
            current_weights.feature_confidence = max(0.0, current_weights.feature_confidence)
            current_weights.structural_weight = max(0.0, current_weights.structural_weight)
            current_weights.hierarchical_weight = max(0.0, current_weights.hierarchical_weight)
            current_weights.semantic_weight = max(0.0, current_weights.semantic_weight)
            
            # 归一化权重
            current_weights.normalize()
            
            # 记录进度
            if iteration % 10 == 0:
                avg_loss = total_loss / num_samples
                logger.info(f"迭代 {iteration}/{iterations}, 平均损失: {avg_loss:.4f}")
        
        # 验证优化结果
        if validation_pairs:
            validation_loss = 0.0
            for concept1, concept2, true_similarity in validation_pairs:
                if isinstance(concept1, str) or isinstance(concept2, str):
                    continue
                
                result = self.calculate_similarity(concept1, concept2, SimilarityContext.GENERAL, current_weights)
                predicted_similarity = result.overall_similarity
                validation_loss += (predicted_similarity - true_similarity) ** 2
            
            avg_validation_loss = validation_loss / len(validation_pairs)
            logger.info(f"验证损失: {avg_validation_loss:.4f}")
        
        logger.info("权重优化完成")
        return current_weights
    
    def _get_weight_key_for_metric(self, metric_name: str) -> str:
        """获取指标对应的权重键"""
        metric_weight_mapping = {
            'feature_based': 'feature_importance',
            'structural': 'structural_weight',
            'hierarchical': 'hierarchical_weight',
            'semantic': 'semantic_weight',
            'statistical': 'feature_confidence',
            'fuzzy_matching': 'feature_frequency'
        }
        return metric_weight_mapping.get(metric_name, 'feature_importance')
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取相似度引擎统计信息"""
        with self._lock:
            stats = self.stats.copy()
            
            # 缓存统计
            if self.cache:
                stats['cache'] = self.cache.get_stats()
            
            # 性能统计
            if self.enable_performance_monitoring:
                stats['performance'] = {
                    'average_calculation_time': self.stats['average_calculation_time'],
                    'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                }
            
            # 计算器使用统计
            stats['calculator_usage'] = dict(self.stats['metric_usage'])
            
            return stats
    
    def clear_cache(self) -> None:
        """清空相似度缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("相似度缓存已清空")
    
    def set_weights(self, weights: SimilarityWeights) -> None:
        """设置默认相似度权重"""
        self.default_weights = weights.normalize()
        logger.info("相似度权重已更新")
    
    def export_similarity_matrix(self,
                               concepts: List[Union[Concept, str]],
                               filepath: str,
                               context: SimilarityContext = SimilarityContext.GENERAL,
                               weights: Optional[SimilarityWeights] = None) -> None:
        """导出相似度矩阵到文件
        
        Args:
            concepts: 概念列表
            filepath: 输出文件路径
            context: 相似度上下文
            weights: 权重配置
        """
        if isinstance(concepts[0], str):
            raise ValueError("需要ConceptFormer来获取概念对象")
        
        # 计算相似度矩阵
        similarity_matrix = []
        concept_ids = []
        
        for i, concept1 in enumerate(concepts):
            if isinstance(concept1, str):
                raise ValueError("需要ConceptFormer来获取概念对象")
            
            row = [concept1.concept_id]
            concept_ids.append(concept1.concept_id)
            
            for j, concept2 in enumerate(concepts):
                if isinstance(concept2, str):
                    raise ValueError("需要ConceptFormer来获取概念对象")
                
                if i == j:
                    similarity = 1.0
                else:
                    result = self.calculate_similarity(concept1, concept2, context, weights)
                    similarity = result.overall_similarity
                
                row.append(similarity)
            
            similarity_matrix.append(row)
        
        # 添加表头
        matrix_data = {
            'concept_ids': concept_ids,
            'concept_names': [c.name for c in concepts],
            'similarity_matrix': similarity_matrix,
            'context': context.value,
            'export_time': time.time()
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"相似度矩阵已导出到: {filepath}")
    
    def load_similarity_matrix(self, filepath: str) -> Dict[str, Any]:
        """从文件加载相似度矩阵
        
        Args:
            filepath: 输入文件路径
            
        Returns:
            相似度矩阵数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            matrix_data = json.load(f)
        
        logger.info(f"相似度矩阵已从 {filepath} 加载")
        return matrix_data