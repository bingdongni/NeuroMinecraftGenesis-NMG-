"""
概念组合机制 - ConceptComposition
Concept Composition Mechanism

这个类实现了概念之间的组合操作，支持将多个概念组合成新概念，
包括特征融合、属性合并、关系构建等操作。

认知科学理论基础：
- 概念整合理论（Conceptual Integration Theory）
- 组合语义学（Combinatory Semantics）
- 框架理论（Frame Theory）
- 脚本理论（Script Theory）
- 复合概念形成（Compound Concept Formation）

Author: NeuroMinecraft Genesis Team
Date: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid
import time
import itertools

from .concept_former import Concept, ConceptLevel

logger = logging.getLogger(__name__)


class CompositionType(Enum):
    """组合类型枚举"""
    FEATURE_FUSION = "feature_fusion"         # 特征融合
    PROPERTY_MERGE = "property_merge"         # 属性合并
    RELATION_BUILDING = "relation_building"   # 关系构建
    STRUCTURAL_COMPOSITION = "structural_composition"  # 结构组合
    BEHAVIORAL_COMPOSITION = "behavioral_composition"  # 行为组合
    TEMPORAL_COMPOSITION = "temporal_composition"      # 时序组合
    SPATIAL_COMPOSITION = "spatial_composition"        # 空间组合
    CAUSAL_COMPOSITION = "causal_composition"          # 因果组合


@dataclass
class CompositionRule:
    """组合规则
    
    定义概念组合的条件、策略和参数
    """
    rule_id: str
    name: str
    description: str
    composition_type: CompositionType
    
    # 组合条件
    min_concepts_required: int = 2                    # 最少概念数量
    min_similarity_threshold: float = 0.3            # 最小相似度阈值
    max_concepts_limit: int = 10                     # 最大组合概念数量
    
    # 组合策略
    fusion_strategy: str = "weighted_average"        # 融合策略
    conflict_resolution: str = "priority_based"      # 冲突解决策略
    
    # 权重参数
    component_weights: Dict[str, float] = field(default_factory=dict)  # 组件权重
    
    # 约束条件
    compatibility_rules: List[str] = field(default_factory=list)  # 兼容性规则
    exclusion_rules: List[str] = field(default_factory=list)      # 互斥规则
    
    # 性能统计
    usage_count: int = 0
    success_rate: float = 0.0
    average_composition_quality: float = 0.0
    
    def evaluate_composition_conditions(self, concepts: List[Concept]) -> Tuple[bool, float]:
        """评估组合条件
        
        Args:
            concepts: 待组合的概念列表
            
        Returns:
            (是否满足条件, 适用性得分)
        """
        # 检查概念数量
        if len(concepts) < self.min_concepts_required:
            return False, 0.0
        
        if len(concepts) > self.max_concepts_limit:
            return False, 0.0
        
        # 检查相似度阈值
        avg_similarity = self._calculate_average_similarity(concepts)
        if avg_similarity < self.min_similarity_threshold:
            return False, 0.0
        
        # 检查兼容性
        if not self._check_compatibility(concepts):
            return False, 0.0
        
        # 检查互斥性
        if not self._check_exclusion(concepts):
            return False, 0.0
        
        # 计算适用性得分
        applicability_score = self._calculate_applicability_score(concepts, avg_similarity)
        
        return True, applicability_score
    
    def _calculate_average_similarity(self, concepts: List[Concept]) -> float:
        """计算概念间平均相似度
        
        Args:
            concepts: 概念列表
            
        Returns:
            平均相似度
        """
        if len(concepts) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = self._calculate_concept_similarity(concepts[i], concepts[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _check_compatibility(self, concepts: List[Concept]) -> bool:
        """检查概念兼容性
        
        Args:
            concepts: 概念列表
            
        Returns:
            是否兼容
        """
        # 简化的兼容性检查
        # 实际实现中应该检查具体的兼容性规则
        
        for rule in self.compatibility_rules:
            if not self._evaluate_compatibility_rule(rule, concepts):
                return False
        
        return True
    
    def _check_exclusion(self, concepts: List[Concept]) -> bool:
        """检查概念互斥性
        
        Args:
            concepts: 概念列表
            
        Returns:
            是否满足互斥条件
        """
        # 简化的互斥检查
        for rule in self.exclusion_rules:
            if self._evaluate_exclusion_rule(rule, concepts):
                return False
        
        return True
    
    def _evaluate_compatibility_rule(self, rule: str, concepts: List[Concept]) -> bool:
        """评估兼容性规则
        
        Args:
            rule: 兼容性规则
            concepts: 概念列表
            
        Returns:
            是否满足规则
        """
        # 简化的规则评估
        # 实际实现中应该解析和评估具体的规则
        
        if "level_compatible" in rule:
            # 检查概念层次兼容性
            levels = [c.level for c in concepts]
            return len(set(levels)) <= 2  # 最多2种不同层次
        
        return True
    
    def _evaluate_exclusion_rule(self, rule: str, concepts: List[Concept]) -> bool:
        """评估互斥规则
        
        Args:
            rule: 互斥规则
            concepts: 概念列表
            
        Returns:
            是否违反互斥规则
        """
        # 简化的互斥规则评估
        return False
    
    def _calculate_applicability_score(self, concepts: List[Concept], avg_similarity: float) -> float:
        """计算适用性得分
        
        Args:
            concepts: 概念列表
            avg_similarity: 平均相似度
            
        Returns:
            适用性得分
        """
        score = 0.0
        
        # 相似度得分
        score += avg_similarity * 0.4
        
        # 置信度得分
        avg_confidence = np.mean([c.confidence for c in concepts])
        score += avg_confidence * 0.3
        
        # 概念数量得分（适中的数量得分较高）
        optimal_count = (self.min_concepts_required + self.max_concepts_limit) / 2
        count_score = 1.0 - abs(len(concepts) - optimal_count) / (self.max_concepts_limit - self.min_concepts_required)
        score += max(0.0, count_score) * 0.3
        
        return score
    
    def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算概念相似度"""
        # 简化的概念相似度计算
        common_features = set(concept1.features.keys()) & set(concept2.features.keys())
        feature_similarity = len(common_features) / max(len(concept1.features), len(concept2.features), 1)
        
        common_properties = set(concept1.properties.keys()) & set(concept2.properties.keys())
        property_similarity = len(common_properties) / max(len(concept1.properties), len(concept2.properties), 1)
        
        return (feature_similarity + property_similarity) / 2


@dataclass
class CompositionResult:
    """组合结果
    
    存储概念组合操作的结果信息
    """
    result_id: str
    composition_type: CompositionType
    
    # 组合结果
    composed_concept: Optional[Concept] = None
    
    # 组件信息
    component_concept_ids: List[str] = field(default_factory=list)
    component_weights: Dict[str, float] = field(default_factory=dict)
    
    # 组合统计
    composition_quality: float = 0.0           # 组合质量
    coherence_score: float = 0.0               # 连贯性得分
    novelty_score: float = 0.0                 # 新颖性得分
    
    # 时间和性能
    composition_time: float = 0.0
    created_time: float = field(default_factory=time.time)
    
    # 元数据
    composition_rule_id: Optional[str] = None
    success: bool = False
    error_message: str = ""


class ConceptComposition:
    """概念组合机制
    
    负责管理概念之间的组合操作，支持：
    1. 多概念融合
    2. 特征和属性组合
    3. 关系构建
    4. 结构化组合
    5. 约束检查和验证
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化概念组合系统
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        
        # 组合规则
        self.composition_rules = self._initialize_composition_rules()
        
        # 组合结果存储
        self.composition_results: Dict[str, CompositionResult] = {}
        
        # 组合历史
        self.composition_history: List[CompositionResult] = []
        
        # 性能优化
        self.composition_cache: Dict[Tuple[str, ...], str] = {}  # 组合键 -> 结果ID
        
        # 统计信息
        self.stats = {
            'total_composition_operations': 0,
            'successful_compositions': 0,
            'failed_compositions': 0,
            'composition_types_used': defaultdict(int),
            'cache_hits': 0,
            'average_composition_time': 0.0,
            'average_quality_score': 0.0
        }
        
        logger.info("ConceptComposition 初始化完成")
    
    def _initialize_composition_rules(self) -> Dict[str, CompositionRule]:
        """初始化组合规则
        
        Returns:
            组合规则字典
        """
        rules = {}
        
        # 规则1：特征融合
        rules['feature_fusion'] = CompositionRule(
            rule_id='feature_fusion',
            name='特征融合',
            description='将多个概念的特征进行融合',
            composition_type=CompositionType.FEATURE_FUSION,
            min_concepts_required=2,
            min_similarity_threshold=0.2,
            max_concepts_limit=5,
            fusion_strategy="weighted_average",
            conflict_resolution="merge_based",
            component_weights={'default': 1.0}
        )
        
        # 规则2：属性合并
        rules['property_merge'] = CompositionRule(
            rule_id='property_merge',
            name='属性合并',
            description='合并多个概念的核心属性',
            composition_type=CompositionType.PROPERTY_MERGE,
            min_concepts_required=2,
            min_similarity_threshold=0.3,
            max_concepts_limit=8,
            fusion_strategy="union_based",
            conflict_resolution="priority_based",
            component_weights={'default': 1.0}
        )
        
        # 规则3：关系构建
        rules['relation_building'] = CompositionRule(
            rule_id='relation_building',
            name='关系构建',
            description='基于概念间关系构建组合概念',
            composition_type=CompositionType.RELATION_BUILDING,
            min_concepts_required=2,
            min_similarity_threshold=0.4,
            max_concepts_limit=3,
            fusion_strategy="relational",
            conflict_resolution="context_based",
            component_weights={'primary': 0.6, 'secondary': 0.4}
        )
        
        # 规则4：结构化组合
        rules['structural_composition'] = CompositionRule(
            rule_id='structural_composition',
            name='结构化组合',
            description='基于概念结构进行组合',
            composition_type=CompositionType.STRUCTURAL_COMPOSITION,
            min_concepts_required=3,
            min_similarity_threshold=0.5,
            max_concepts_limit=6,
            fusion_strategy="structural_merge",
            conflict_resolution="hierarchy_based",
            component_weights={'default': 1.0}
        )
        
        # 规则5：行为组合
        rules['behavioral_composition'] = CompositionRule(
            rule_id='behavioral_composition',
            name='行为组合',
            description='组合概念的行为特征',
            composition_type=CompositionType.BEHAVIORAL_COMPOSITION,
            min_concepts_required=2,
            min_similarity_threshold=0.3,
            max_concepts_limit=4,
            fusion_strategy="behavioral_sequence",
            conflict_resolution="temporal_based",
            component_weights={'default': 1.0}
        )
        
        return rules
    
    def compose_concepts(self, 
                        concepts: List[Concept], 
                        composition_type: CompositionType = CompositionType.FEATURE_FUSION,
                        rule_id: Optional[str] = None) -> Optional[str]:
        """组合多个概念
        
        这是概念组合的主要接口方法
        
        Args:
            concepts: 待组合的概念列表
            composition_type: 组合类型
            rule_id: 指定的组合规则ID
            
        Returns:
            组合结果ID，如果组合失败则返回None
        """
        if len(concepts) < 2:
            logger.warning("概念组合需要至少2个概念")
            return None
        
        logger.info(f"开始概念组合: {len(concepts)} 个概念, 类型: {composition_type.value}")
        
        # 选择组合规则
        composition_rule = self._select_composition_rule(concepts, composition_type, rule_id)
        
        if not composition_rule:
            logger.warning("未找到适用的组合规则")
            return None
        
        # 检查缓存
        cache_key = self._generate_composition_cache_key(concepts, composition_type)
        if cache_key in self.composition_cache:
            self.stats['cache_hits'] += 1
            result_id = self.composition_cache[cache_key]
            logger.info(f"使用缓存的组合结果: {result_id}")
            return result_id
        
        # 执行组合
        start_time = time.time()
        result = self._apply_composition_rule(composition_rule, concepts, composition_type)
        composition_time = time.time() - start_time
        
        # 存储结果
        if result.success:
            result.composition_time = composition_time
            self.composition_results[result.result_id] = result
            self.composition_history.append(result)
            
            # 缓存结果
            self.composition_cache[cache_key] = result.result_id
            
            # 更新统计信息
            self.stats['successful_compositions'] += 1
            self.stats['composition_types_used'][composition_type.value] += 1
            logger.info(f"概念组合成功: {result.result_id}")
        else:
            self.stats['failed_compositions'] += 1
            logger.warning(f"概念组合失败: {result.error_message}")
        
        self.stats['total_composition_operations'] += 1
        
        # 更新性能统计
        self._update_performance_stats(composition_time, result.composition_quality)
        
        return result.result_id if result.success else None
    
    def _select_composition_rule(self, 
                               concepts: List[Concept], 
                               composition_type: CompositionType,
                               rule_id: Optional[str]) -> Optional[CompositionRule]:
        """选择最佳的组合规则
        
        Args:
            concepts: 待组合的概念列表
            composition_type: 组合类型
            rule_id: 指定的规则ID
            
        Returns:
            最佳组合规则
        """
        # 如果指定了规则ID，直接返回
        if rule_id and rule_id in self.composition_rules:
            rule = self.composition_rules[rule_id]
            # 检查规则是否适用
            compatible, score = rule.evaluate_composition_conditions(concepts)
            if compatible:
                return rule
        
        # 选择最适合的规则
        best_rule = None
        best_score = 0.0
        
        for rule in self.composition_rules.values():
            if rule.composition_type == composition_type:
                compatible, score = rule.evaluate_composition_conditions(concepts)
                if compatible and score > best_score:
                    best_score = score
                    best_rule = rule
        
        return best_rule
    
    def _apply_composition_rule(self, 
                              rule: CompositionRule, 
                              concepts: List[Concept],
                              composition_type: CompositionType) -> CompositionResult:
        """应用组合规则
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            composition_type: 组合类型
            
        Returns:
            组合结果
        """
        result = CompositionResult(
            result_id=str(uuid.uuid4()),
            composition_type=composition_type,
            component_concept_ids=[c.concept_id for c in concepts],
            composition_rule_id=rule.rule_id
        )
        
        try:
            # 根据组合类型执行相应的组合操作
            if composition_type == CompositionType.FEATURE_FUSION:
                composed_concept = self._fuse_features(rule, concepts)
            elif composition_type == CompositionType.PROPERTY_MERGE:
                composed_concept = self._merge_properties(rule, concepts)
            elif composition_type == CompositionType.RELATION_BUILDING:
                composed_concept = self._build_relations(rule, concepts)
            elif composition_type == CompositionType.STRUCTURAL_COMPOSITION:
                composed_concept = self._compose_structure(rule, concepts)
            elif composition_type == CompositionType.BEHAVIORAL_COMPOSITION:
                composed_concept = self._compose_behaviors(rule, concepts)
            elif composition_type == CompositionType.TEMPORAL_COMPOSITION:
                composed_concept = self._compose_temporally(rule, concepts)
            elif composition_type == CompositionType.SPATIAL_COMPOSITION:
                composed_concept = self._compose_spatially(rule, concepts)
            elif composition_type == CompositionType.CAUSAL_COMPOSITION:
                composed_concept = self._compose_causally(rule, concepts)
            else:
                raise ValueError(f"未知的组合类型: {composition_type}")
            
            if composed_concept:
                result.composed_concept = composed_concept
                result.success = True
                
                # 计算组合质量
                result.composition_quality = self._calculate_composition_quality(concepts, composed_concept)
                result.coherence_score = self._calculate_coherence_score(concepts, composed_concept)
                result.novelty_score = self._calculate_novelty_score(concepts, composed_concept)
                
                # 更新规则统计
                rule.usage_count += 1
                rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + 1.0) / rule.usage_count
                rule.average_composition_quality = (
                    rule.average_composition_quality * (rule.usage_count - 1) + result.composition_quality
                ) / rule.usage_count
                
            else:
                result.success = False
                result.error_message = "组合操作未能生成有效概念"
                rule.success_rate = (rule.success_rate * rule.usage_count) / (rule.usage_count + 1)
                
        except Exception as e:
            result.success = False
            result.error_message = f"组合过程发生错误: {str(e)}"
            logger.error(f"应用组合规则失败: {e}")
        
        return result
    
    def _fuse_features(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行特征融合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            融合后的概念
        """
        # 收集所有特征
        all_features = defaultdict(list)
        feature_weights = {}
        
        for i, concept in enumerate(concepts):
            weight = rule.component_weights.get(concept.concept_id, rule.component_weights.get('default', 1.0))
            
            for feature_name, feature_value in concept.features.items():
                all_features[feature_name].append((feature_value, weight, i))
                feature_weights[feature_name] = weight
        
        # 融合特征
        fused_features = {}
        for feature_name, value_weight_pairs in all_features.items():
            if len(value_weight_pairs) == 1:
                # 单一特征直接使用
                fused_features[feature_name] = value_weight_pairs[0][0]
            else:
                # 多值融合
                fused_features[feature_name] = self._fuse_feature_values(
                    feature_name, value_weight_pairs, rule.fusion_strategy
                )
        
        # 创建融合概念
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Fused_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=fused_features.copy()
        )
        
        # 设置组合属性
        composed_concept.properties = {
            'composition_type': 'feature_fusion',
            'component_count': len(concepts),
            'fused_feature_count': len(fused_features)
        }
        
        # 计算置信度和抽象程度
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.8
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _merge_properties(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行属性合并
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            合并后的概念
        """
        merged_properties = {}
        property_sources = {}
        
        # 收集所有属性
        for concept in concepts:
            for prop_name, prop_value in concept.properties.items():
                if prop_name not in merged_properties:
                    merged_properties[prop_name] = prop_value
                    property_sources[prop_name] = [concept.concept_id]
                else:
                    # 处理属性冲突
                    resolved_value = self._resolve_property_conflict(
                        prop_name, merged_properties[prop_value], prop_value, rule
                    )
                    merged_properties[prop_name] = resolved_value
                    property_sources[prop_name].append(concept.concept_id)
        
        # 创建合并概念
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Merged_{len(concepts)}_concepts",
            level=ConceptLevel.CATEGORICAL,
            features={},
            properties=merged_properties.copy()
        )
        
        # 设置合并属性
        composed_concept.properties.update({
            'composition_type': 'property_merge',
            'component_count': len(concepts),
            'property_sources': property_sources
        })
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.9
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _build_relations(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行关系构建
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            关系构建后的概念
        """
        if len(concepts) < 2:
            return None
        
        # 识别概念间的关系
        relations = []
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                relation = self._identify_concept_relation(concept1, concept2)
                if relation:
                    relations.append(relation)
        
        # 创建关系概念
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Related_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features={},
            properties={
                'composition_type': 'relation_building',
                'component_count': len(concepts),
                'relations': relations
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.7
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _compose_structure(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行结构化组合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            结构化组合概念
        """
        # 提取结构特征
        structural_features = self._extract_structural_features(concepts)
        
        # 创建结构化概念
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Structured_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=structural_features.copy(),
            properties={
                'composition_type': 'structural_composition',
                'component_count': len(concepts)
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.8
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _compose_behaviors(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行行为组合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            行为组合概念
        """
        # 提取行为特征
        behavioral_features = self._extract_behavioral_features(concepts)
        
        # 创建行为概念
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Behavioral_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=behavioral_features.copy(),
            properties={
                'composition_type': 'behavioral_composition',
                'component_count': len(concepts)
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.75
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _compose_temporally(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行时序组合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            时序组合概念
        """
        # 简化的时序组合
        temporal_features = {
            'temporal_sequence': [concept.name for concept in concepts],
            'temporal_order': len(concepts)
        }
        
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Temporal_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=temporal_features.copy(),
            properties={
                'composition_type': 'temporal_composition',
                'component_count': len(concepts)
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.6
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _compose_spatially(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行空间组合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            空间组合概念
        """
        # 简化的空间组合
        spatial_features = {
            'spatial_layout': [concept.name for concept in concepts],
            'spatial_count': len(concepts)
        }
        
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Spatial_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=spatial_features.copy(),
            properties={
                'composition_type': 'spatial_composition',
                'component_count': len(concepts)
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.7
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _compose_causally(self, rule: CompositionRule, concepts: List[Concept]) -> Optional[Concept]:
        """执行因果组合
        
        Args:
            rule: 组合规则
            concepts: 待组合的概念列表
            
        Returns:
            因果组合概念
        """
        # 简化的因果组合
        causal_features = {
            'causal_chain': [concept.name for concept in concepts],
            'causal_count': len(concepts)
        }
        
        composed_concept = Concept(
            concept_id=str(uuid.uuid4()),
            name=f"Causal_{len(concepts)}_concepts",
            level=ConceptLevel.ABSTRACT,
            features=causal_features.copy(),
            properties={
                'composition_type': 'causal_composition',
                'component_count': len(concepts)
            }
        )
        
        composed_concept.confidence = np.mean([c.confidence for c in concepts]) * 0.65
        composed_concept.compute_abstractness()
        
        return composed_concept
    
    def _fuse_feature_values(self, 
                           feature_name: str, 
                           value_weight_pairs: List[Tuple[Any, float, int]], 
                           fusion_strategy: str) -> Any:
        """融合特征值
        
        Args:
            feature_name: 特征名称
            value_weight_pairs: (值, 权重, 概念索引) 列表
            fusion_strategy: 融合策略
            
        Returns:
            融合后的特征值
        """
        values = [pair[0] for pair in value_weight_pairs]
        weights = [pair[1] for pair in value_weight_pairs]
        
        if fusion_strategy == "weighted_average":
            # 加权平均（适用于数值特征）
            if all(isinstance(v, (int, float)) for v in values):
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    return weighted_sum / total_weight
            return values[0]  # 回退策略
        
        elif fusion_strategy == "union_based":
            # 集合并（适用于集合特征）
            if isinstance(values[0], list):
                return list(set(itertools.chain.from_iterable(values)))
            return list(set(values))
        
        elif fusion_strategy == "priority_based":
            # 优先级融合
            return values[0]  # 优先级最高的值
        
        elif fusion_strategy == "merge_based":
            # 合并策略
            if isinstance(values[0], dict):
                merged = {}
                for value in values:
                    if isinstance(value, dict):
                        merged.update(value)
                return merged
            return values[0]
        
        else:
            return values[0]  # 默认回退
    
    def _resolve_property_conflict(self, 
                                 prop_name: str, 
                                 existing_value: Any, 
                                 new_value: Any,
                                 rule: CompositionRule) -> Any:
        """解决属性冲突
        
        Args:
            prop_name: 属性名称
            existing_value: 现有值
            new_value: 新值
            rule: 组合规则
            
        Returns:
            解决后的属性值
        """
        if rule.conflict_resolution == "priority_based":
            return existing_value  # 优先保留现有值
        elif rule.conflict_resolution == "merge_based":
            if isinstance(existing_value, list) and isinstance(new_value, list):
                return list(set(existing_value + new_value))
            elif isinstance(existing_value, dict) and isinstance(new_value, dict):
                merged = existing_value.copy()
                merged.update(new_value)
                return merged
            else:
                return [existing_value, new_value]
        elif rule.conflict_resolution == "context_based":
            # 基于上下文的冲突解决
            return existing_value  # 简化策略
        elif rule.conflict_resolution == "hierarchy_based":
            # 基于层次的冲突解决
            return existing_value  # 简化策略
        else:
            return existing_value
    
    def _identify_concept_relation(self, concept1: Concept, concept2: Concept) -> Optional[Dict[str, Any]]:
        """识别概念间关系
        
        Args:
            concept1: 第一个概念
            concept2: 第二个概念
            
        Returns:
            关系描述字典
        """
        # 简化的关系识别
        common_features = set(concept1.features.keys()) & set(concept2.features.keys())
        common_properties = set(concept1.properties.keys()) & set(concept2.properties.keys())
        
        if common_features:
            relation_type = "feature_based"
        elif common_properties:
            relation_type = "property_based"
        else:
            relation_type = "weakly_related"
        
        return {
            'type': relation_type,
            'source_concept': concept1.concept_id,
            'target_concept': concept2.concept_id,
            'strength': min(len(common_features) / 10.0, len(common_properties) / 10.0, 1.0)
        }
    
    def _extract_structural_features(self, concepts: List[Concept]) -> Dict[str, Any]:
        """提取结构特征
        
        Args:
            concepts: 概念列表
            
        Returns:
            结构特征字典
        """
        structural_features = {
            'component_count': len(concepts),
            'level_distribution': {},
            'feature_count_stats': {},
            'property_count_stats': {}
        }
        
        # 层次分布
        for concept in concepts:
            level = concept.level.value
            structural_features['level_distribution'][level] = \
                structural_features['level_distribution'].get(level, 0) + 1
        
        # 特征统计
        feature_counts = [len(c.features) for c in concepts]
        property_counts = [len(c.properties) for c in concepts]
        
        structural_features['feature_count_stats'] = {
            'mean': np.mean(feature_counts),
            'std': np.std(feature_counts),
            'min': min(feature_counts),
            'max': max(feature_counts)
        }
        
        structural_features['property_count_stats'] = {
            'mean': np.mean(property_counts),
            'std': np.std(property_counts),
            'min': min(property_counts),
            'max': max(property_counts)
        }
        
        return structural_features
    
    def _extract_behavioral_features(self, concepts: List[Concept]) -> Dict[str, Any]:
        """提取行为特征
        
        Args:
            concepts: 概念列表
            
        Returns:
            行为特征字典
        """
        behavioral_features = {
            'behavioral_patterns': [],
            'interaction_count': 0,
            'behavioral_complexity': 0.0
        }
        
        # 识别行为相关特征
        for concept in concepts:
            behavioral_aspects = []
            for feature_name in concept.features.keys():
                if any(keyword in feature_name.lower() for keyword in 
                      ['action', 'behavior', 'function', 'capability', 'activity']):
                    behavioral_aspects.append(feature_name)
            
            if behavioral_aspects:
                behavioral_features['behavioral_patterns'].append({
                    'concept_id': concept.concept_id,
                    'behaviors': behavioral_aspects
                })
        
        # 计算交互复杂度
        behavioral_features['behavioral_complexity'] = len(behavioral_features['behavioral_patterns']) / len(concepts)
        
        return behavioral_features
    
    def _calculate_composition_quality(self, original_concepts: List[Concept], composed_concept: Concept) -> float:
        """计算组合质量
        
        Args:
            original_concepts: 原始概念列表
            composed_concept: 组合后的概念
            
        Returns:
            组合质量分数 [0,1]
        """
        quality_scores = []
        
        # 1. 特征覆盖度
        original_feature_count = sum(len(c.features) for c in original_concepts)
        composed_feature_count = len(composed_concept.features)
        
        if original_feature_count > 0:
            coverage_score = min(1.0, composed_feature_count / original_feature_count)
            quality_scores.append(coverage_score)
        
        # 2. 属性保留度
        original_property_count = sum(len(c.properties) for c in original_concepts)
        composed_property_count = len(composed_concept.properties)
        
        if original_property_count > 0:
            preservation_score = min(1.0, composed_property_count / original_property_count)
            quality_scores.append(preservation_score)
        
        # 3. 语义一致性
        avg_original_confidence = np.mean([c.confidence for c in original_concepts])
        confidence_score = min(1.0, composed_concept.confidence / max(avg_original_confidence, 0.1))
        quality_scores.append(confidence_score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _calculate_coherence_score(self, original_concepts: List[Concept], composed_concept: Concept) -> float:
        """计算连贯性得分
        
        Args:
            original_concepts: 原始概念列表
            composed_concept: 组合后的概念
            
        Returns:
            连贯性得分 [0,1]
        """
        # 基于概念间相似度计算连贯性
        if len(original_concepts) < 2:
            return 1.0
        
        similarities = []
        for concept in original_concepts:
            # 计算与组合概念的相似度
            similarity = self._calculate_concept_similarity(concept, composed_concept)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_novelty_score(self, original_concepts: List[Concept], composed_concept: Concept) -> float:
        """计算新颖性得分
        
        Args:
            original_concepts: 原始概念列表
            composed_concept: 组合后的概念
            
        Returns:
            新颖性得分 [0,1]
        """
        # 简化的新颖性计算
        # 实际实现中应该与历史组合结果比较
        
        # 基于特征数量和属性的新颖性
        original_diversity = len(set().union(*[set(c.features.keys()) for c in original_concepts]))
        composed_uniqueness = len(set(composed_concept.features.keys()) - 
                                set().union(*[set(c.features.keys()) for c in original_concepts]))
        
        if original_diversity > 0:
            novelty_score = composed_uniqueness / original_diversity
        else:
            novelty_score = 0.0
        
        return min(1.0, novelty_score)
    
    def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算概念相似度"""
        # 简化的相似度计算
        common_features = set(concept1.features.keys()) & set(concept2.features.keys())
        feature_similarity = len(common_features) / max(len(concept1.features), len(concept2.features), 1)
        
        common_properties = set(concept1.properties.keys()) & set(concept2.properties.keys())
        property_similarity = len(common_properties) / max(len(concept1.properties), len(concept2.properties), 1)
        
        return (feature_similarity + property_similarity) / 2
    
    def _generate_composition_cache_key(self, concepts: List[Concept], composition_type: CompositionType) -> Tuple[str, ...]:
        """生成组合缓存键
        
        Args:
            concepts: 概念列表
            composition_type: 组合类型
            
        Returns:
            缓存键元组
        """
        concept_ids = tuple(sorted([c.concept_id for c in concepts]))
        return (composition_type.value,) + concept_ids
    
    def _update_performance_stats(self, composition_time: float, quality_score: float):
        """更新性能统计信息
        
        Args:
            composition_time: 组合时间
            quality_score: 质量分数
        """
        # 更新平均组合时间
        old_avg_time = self.stats['average_composition_time']
        count = self.stats['total_composition_operations']
        self.stats['average_composition_time'] = (old_avg_time * (count - 1) + composition_time) / count
        
        # 更新平均质量分数
        old_avg_quality = self.stats['average_quality_score']
        self.stats['average_quality_score'] = (old_avg_quality * (count - 1) + quality_score) / count
    
    def get_composition_result(self, result_id: str) -> Optional[CompositionResult]:
        """获取组合结果
        
        Args:
            result_id: 结果ID
            
        Returns:
            组合结果对象
        """
        return self.composition_results.get(result_id)
    
    def get_composition_history(self, limit: int = 10) -> List[CompositionResult]:
        """获取组合历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            组合历史记录列表
        """
        return self.composition_history[-limit:]
    
    def find_similar_compositions(self, 
                                target_concepts: List[Concept], 
                                composition_type: CompositionType,
                                threshold: float = 0.8,
                                max_results: int = 5) -> List[Tuple[CompositionResult, float]]:
        """查找相似的组合结果
        
        Args:
            target_concepts: 目标概念列表
            composition_type: 组合类型
            threshold: 相似度阈值
            max_results: 最大结果数
            
        Returns:
            (组合结果, 相似度) 列表
        """
        target_concept_ids = set(c.concept_id for c in target_concepts)
        similar_compositions = []
        
        for result in self.composition_history:
            if result.composition_type == composition_type:
                result_concept_ids = set(result.component_concept_ids)
                
                # 计算概念集合相似度
                intersection = len(target_concept_ids & result_concept_ids)
                union = len(target_concept_ids | result_concept_ids)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        similar_compositions.append((result, similarity))
        
        # 按相似度排序
        similar_compositions.sort(key=lambda x: x[1], reverse=True)
        return similar_compositions[:max_results]
    
    def get_composition_statistics(self) -> Dict[str, Any]:
        """获取组合系统统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        stats['composition_types_used'] = dict(stats['composition_types_used'])
        
        # 组合结果统计
        stats['total_composition_results'] = len(self.composition_results)
        
        if self.composition_history:
            avg_quality = np.mean([r.composition_quality for r in self.composition_history])
            avg_coherence = np.mean([r.coherence_score for r in self.composition_history])
            avg_novelty = np.mean([r.novelty_score for r in self.composition_history])
            
            stats['average_composition_quality'] = avg_quality
            stats['average_coherence_score'] = avg_coherence
            stats['average_novelty_score'] = avg_novelty
            
            # 成功率
            successful_results = [r for r in self.composition_history if r.success]
            stats['composition_success_rate'] = len(successful_results) / len(self.composition_history)
        else:
            stats['average_composition_quality'] = 0.0
            stats['average_coherence_score'] = 0.0
            stats['average_novelty_score'] = 0.0
            stats['composition_success_rate'] = 0.0
        
        # 规则性能统计
        rule_performance = {}
        for rule_id, rule in self.composition_rules.items():
            rule_performance[rule_id] = {
                'usage_count': rule.usage_count,
                'success_rate': rule.success_rate,
                'average_quality': rule.average_composition_quality,
                'description': rule.description
            }
        stats['rule_performance'] = rule_performance
        
        # 缓存统计
        stats['cache_size'] = len(self.composition_cache)
        stats['cache_hit_rate'] = stats['cache_hits'] / max(stats['total_composition_operations'], 1)
        
        return stats
    
    def clear_cache(self):
        """清除组合缓存"""
        self.composition_cache.clear()
        logger.info("组合缓存已清除")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_composition_operations': 0,
            'successful_compositions': 0,
            'failed_compositions': 0,
            'composition_types_used': defaultdict(int),
            'cache_hits': 0,
            'average_composition_time': 0.0,
            'average_quality_score': 0.0
        }
        
        for rule in self.composition_rules.values():
            rule.usage_count = 0
            rule.success_rate = 0.0
            rule.average_composition_quality = 0.0
        
        logger.info("组合系统统计信息已重置")