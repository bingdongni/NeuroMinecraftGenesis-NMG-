#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识归纳模块

本模块实现知识图谱的知识归纳器，从观察数据中归纳出一般性规律和知识。
支持归纳推理、演绎推理、类比推理等多种推理方式。

主要功能：
- 从实例归纳一般规律
- 演绎推理验证
- 类比推理发现
- 因果关系推断
- 知识演化优化
- 推理链构建

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from itertools import combinations, chain
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


class InductionMethod(Enum):
    """归纳方法枚举"""
    FREQUENT_PATTERN = "frequent_pattern"      # 频繁模式归纳
    RULE_INDUCTION = "rule_induction"          # 规则归纳
    CAUSAL_INFERENCE = "causal_inference"      # 因果推断
    ANALOGICAL_REASONING = "analogical_reasoning"  # 类比推理
    CASE_BASED = "case_based"                  # 基于案例的推理
    STATISTICAL = "statistical"                # 统计归纳


class ReasoningType(Enum):
    """推理类型枚举"""
    INDUCTIVE = "inductive"       # 归纳推理
    DEDUCTIVE = "deductive"       # 演绎推理
    ABDUCTIVE = "abductive"       # 溯因推理
    ANALOGICAL = "analogical"     # 类比推理
    CAUSAL = "causal"            # 因果推理


@dataclass
class InducedRule:
    """归纳出的规则"""
    rule_id: str
    premise: Set[str]            # 前提条件
    conclusion: Set[str]         # 结论
    confidence: float            # 置信度
    support: float              # 支持度
    lift: float                 # 提升度
    method: InductionMethod      # 归纳方法
    examples: List[Dict[str, Any]] = field(default_factory=list)
    counter_examples: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.examples:
            self.examples = []
        if not self.counter_examples:
            self.counter_examples = []
        if not self.reasoning_chain:
            self.reasoning_chain = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class InferenceResult:
    """推理结果"""
    conclusion: str
    confidence: float
    reasoning_type: ReasoningType
    evidence: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgePattern:
    """知识模式"""
    pattern_id: str
    pattern_type: str
    elements: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (elem1, relation, elem2)
    frequency: int = 0
    stability: float = 0.0
    context: str = ""
    confidence: float = 0.0


class KnowledgeInduction:
    """
    知识归纳器
    
    实现多种知识归纳和推理功能：
    - 从实例数据归纳一般规律
    - 因果关系推断
    - 类比推理发现
    - 规则挖掘和验证
    - 知识一致性检查
    
    特性：
    - 支持大规模数据归纳
    - 多层次推理链
    - 动态知识更新
    - 冲突检测和解决
    - 可解释推理过程
    """
    
    def __init__(self,
                 min_support: float = 0.05,
                 min_confidence: float = 0.6,
                 induction_threshold: float = 0.7,
                 reasoning_depth: int = 3,
                 enable_analogical: bool = True,
                 enable_causal: bool = True):
        """
        初始化知识归纳器
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
            induction_threshold: 归纳阈值
            reasoning_depth: 推理深度
            enable_analogical: 是否启用类比推理
            enable_causal: 是否启用因果推断
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.induction_threshold = induction_threshold
        self.reasoning_depth = reasoning_depth
        self.enable_analogical = enable_analogical
        self.enable_causal = enable_causal
        
        # 知识存储
        self.induced_rules = {}     # rule_id -> InducedRule
        self.knowledge_patterns = {}  # pattern_id -> KnowledgePattern
        self.reasoning_cache = {}   # 推理缓存
        
        # 观察数据
        self.observations = []      # 观察实例列表
        self.feature_matrix = None  # 特征矩阵
        self.feature_names = []     # 特征名称
        
        # 统计信息
        self.induction_stats = {
            'total_observations': 0,
            'induced_rules': 0,
            'patterns_found': 0,
            'inferences_made': 0,
            'accuracy_rate': 0.0,
            'processing_time': 0.0
        }
        
        self.logger = logging.getLogger("KnowledgeInduction")
        
        self.logger.info(f"知识归纳器初始化完成，支持度: {min_support}, 置信度: {min_confidence}")
    
    def add_observation(self, observation: Dict[str, Any]):
        """
        添加观察实例
        
        Args:
            observation: 观察数据字典
        """
        try:
            # 验证观察数据
            if not isinstance(observation, dict):
                raise ValueError("观察数据必须是字典类型")
            
            # 添加时间戳
            if 'timestamp' not in observation:
                observation['timestamp'] = datetime.now()
            
            # 存储观察
            self.observations.append(observation)
            
            # 更新特征矩阵
            self._update_feature_matrix(observation)
            
            # 更新统计
            self.induction_stats['total_observations'] = len(self.observations)
            
            self.logger.debug(f"添加观察实例，当前总数: {len(self.observations)}")
            
        except Exception as e:
            self.logger.error(f"添加观察失败: {str(e)}")
    
    def add_observations_batch(self, observations: List[Dict[str, Any]]):
        """
        批量添加观察实例
        
        Args:
            observations: 观察实例列表
        """
        for observation in observations:
            self.add_observation(observation)
        
        self.logger.info(f"批量添加 {len(observations)} 个观察实例")
    
    def induce_knowledge(self, 
                        method: InductionMethod = InductionMethod.FREQUENT_PATTERN,
                        **kwargs) -> List[InducedRule]:
        """
        从观察数据中归纳知识
        
        Args:
            method: 归纳方法
            **kwargs: 方法特定参数
            
        Returns:
            List[InducedRule]: 归纳出的规则列表
        """
        start_time = datetime.now()
        
        try:
            if len(self.observations) < 2:
                self.logger.warning("观察数据不足，无法进行归纳")
                return []
            
            rules = []
            
            if method == InductionMethod.FREQUENT_PATTERN:
                rules = self._induce_frequent_patterns(**kwargs)
            elif method == InductionMethod.RULE_INDUCTION:
                rules = self._induce_rules(**kwargs)
            elif method == InductionMethod.CAUSAL_INFERENCE:
                rules = self._infer_causal_relations(**kwargs)
            elif method == InductionMethod.ANALOGICAL_REASONING:
                rules = self._analogical_reasoning(**kwargs)
            elif method == InductionMethod.CASE_BASED:
                rules = self._case_based_reasoning(**kwargs)
            elif method == InductionMethod.STATISTICAL:
                rules = self._statistical_induction(**kwargs)
            else:
                raise ValueError(f"不支持的归纳方法: {method}")
            
            # 验证和过滤规则
            validated_rules = self._validate_rules(rules)
            
            # 存储规则
            for rule in validated_rules:
                self.induced_rules[rule.rule_id] = rule
            
            # 更新统计
            self.induction_stats['induced_rules'] += len(validated_rules)
            self.induction_stats['processing_time'] += (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"知识归纳完成，使用方法: {method}, 产生规则: {len(validated_rules)}")
            return validated_rules
            
        except Exception as e:
            self.logger.error(f"知识归纳失败: {str(e)}")
            return []
    
    def deductive_reasoning(self, 
                          premise: Set[str],
                          target_conclusion: str = None) -> List[InferenceResult]:
        """
        演绎推理
        
        Args:
            premise: 前提条件集合
            target_conclusion: 目标结论（可选）
            
        Returns:
            List[InferenceResult]: 推理结果列表
        """
        results = []
        
        try:
            # 应用归纳的规则进行演绎
            applicable_rules = self._find_applicable_rules(premise)
            
            for rule in applicable_rules:
                # 检查规则前提是否被满足
                if rule.premise.issubset(premise):
                    conclusion = list(rule.conclusion)[0] if rule.conclusion else ""
                    
                    # 计算演绎置信度
                    deductive_confidence = rule.confidence * self._calculate_premise_satisfaction(premise, rule.premise)
                    
                    result = InferenceResult(
                        conclusion=conclusion,
                        confidence=deductive_confidence,
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"规则: {rule.rule_id}", f"前提: {list(rule.premise)}"],
                        reasoning_steps=[
                            f"应用规则: {rule.rule_id}",
                            f"前提条件满足: {rule.premise}",
                            f"得出结论: {conclusion}"
                        ]
                    )
                    results.append(result)
            
            # 如果有目标结论，进行搜索
            if target_conclusion:
                results = [r for r in results if target_conclusion in r.conclusion]
            
            # 按置信度排序
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            self.induction_stats['inferences_made'] += len(results)
            
            self.logger.debug(f"演绎推理完成，产出 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"演绎推理失败: {str(e)}")
            return []
    
    def abductive_reasoning(self, 
                          observation: str,
                          max_explanations: int = 5) -> List[InferenceResult]:
        """
        溯因推理（寻找最佳解释）
        
        Args:
            observation: 观察现象
            max_explanations: 最大解释数量
            
        Returns:
            List[InferenceResult]: 解释结果列表
        """
        results = []
        
        try:
            # 查找能解释观察的规则
            explaining_rules = []
            
            for rule in self.induced_rules.values():
                for conclusion in rule.conclusion:
                    if observation.lower() in conclusion.lower() or conclusion.lower() in observation.lower():
                        explaining_rules.append((rule, conclusion))
            
            # 评估每个解释
            for rule, conclusion in explaining_rules:
                # 计算解释的质量
                explanation_quality = self._evaluate_explanation_quality(rule, observation)
                
                # 构建解释链
                explanation_steps = [
                    f"观察到: {observation}",
                    f"可能解释: {conclusion}",
                    f"支持证据: {len(rule.examples)} 个正例",
                    f"规则: {rule.rule_id}"
                ]
                
                # 如果规则有前提，添加中间步骤
                if rule.premise:
                    explanation_steps.insert(-1, f"需要满足前提: {rule.premise}")
                
                result = InferenceResult(
                    conclusion=conclusion,
                    confidence=explanation_quality,
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    evidence=[f"规则支持度: {rule.support}", f"规则置信度: {rule.confidence}"],
                    reasoning_steps=explanation_steps
                )
                results.append(result)
            
            # 按质量排序并限制数量
            results.sort(key=lambda x: x.confidence, reverse=True)
            results = results[:max_explanations]
            
            self.induction_stats['inferences_made'] += len(results)
            
            self.logger.debug(f"溯因推理完成，找到 {len(results)} 个解释")
            return results
            
        except Exception as e:
            self.logger.error(f"溯因推理失败: {str(e)}")
            return []
    
    def analogical_reasoning(self, 
                           source_domain: Dict[str, Any],
                           target_domain: Dict[str, Any]) -> List[InferenceResult]:
        """
        类比推理
        
        Args:
            source_domain: 源域数据
            target_domain: 目标域数据
            
        Returns:
            List[InferenceResult]: 类比推理结果
        """
        if not self.enable_analogical:
            self.logger.warning("类比推理未启用")
            return []
        
        results = []
        
        try:
            # 寻找相似的情况
            similar_cases = self._find_similar_cases(source_domain, target_domain)
            
            for similar_case in similar_cases:
                # 提取类比映射
                analogy_mapping = self._extract_analogy_mapping(source_domain, similar_case, target_domain)
                
                if analogy_mapping:
                    # 基于类比进行推理
                    inferred_properties = self._infer_by_analogy(similar_case, target_domain, analogy_mapping)
                    
                    for property_name, property_value in inferred_properties.items():
                        confidence = self._calculate_analogy_confidence(similar_case, target_domain, property_name)
                        
                        result = InferenceResult(
                            conclusion=f"{property_name} = {property_value}",
                            confidence=confidence,
                            reasoning_type=ReasoningType.ANALOGICAL,
                            evidence=[
                                f"源域案例: {similar_case.get('id', 'unknown')}",
                                f"类比映射: {analogy_mapping}",
                                f"相似度: {confidence}"
                            ],
                            reasoning_steps=[
                                f"发现相似案例: {similar_case.get('description', 'N/A')}",
                                f"建立类比映射: {analogy_mapping}",
                                f"推理属性: {property_name} -> {property_value}",
                                f"置信度: {confidence:.3f}"
                            ]
                        )
                        results.append(result)
            
            # 按置信度排序
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            self.induction_stats['inferences_made'] += len(results)
            
            self.logger.debug(f"类比推理完成，产生 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"类比推理失败: {str(e)}")
            return []
    
    def causal_inference(self, 
                        potential_causes: List[str],
                        effect: str,
                        observations: List[Dict[str, Any]] = None) -> List[InferenceResult]:
        """
        因果推断
        
        Args:
            potential_causes: 潜在原因列表
            effect: 效果
            observations: 观察数据（可选）
            
        Returns:
            List[InferenceResult]: 因果推断结果
        """
        if not self.enable_causal:
            self.logger.warning("因果推断未启用")
            return []
        
        results = []
        
        try:
            # 使用默认观察数据或提供的观察数据
            data = observations if observations else self.observations
            
            if len(data) < 10:
                self.logger.warning("观察数据不足，无法进行可靠的因果推断")
                return []
            
            for cause in potential_causes:
                # 计算因果强度
                causal_strength = self._calculate_causal_strength(cause, effect, data)
                
                if causal_strength > self.induction_threshold:
                    # 进行因果验证
                    causal_validation = self._validate_causal_relationship(cause, effect, data)
                    
                    result = InferenceResult(
                        conclusion=f"{cause} -> {effect}",
                        confidence=causal_strength * causal_validation,
                        reasoning_type=ReasoningType.CAUSAL,
                        evidence=[
                            f"因果强度: {causal_strength:.3f}",
                            f"验证分数: {causal_validation:.3f}",
                            f"观察数据量: {len(data)}"
                        ],
                        reasoning_steps=[
                            f"假设因果关系: {cause} 导致 {effect}",
                            f"计算因果强度: {causal_strength:.3f}",
                            f"验证因果关系: {causal_validation:.3f}",
                            f"综合置信度: {causal_strength * causal_validation:.3f}"
                        ]
                    )
                    results.append(result)
            
            # 按置信度排序
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            self.induction_stats['inferences_made'] += len(results)
            
            self.logger.debug(f"因果推断完成，产生 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"因果推断失败: {str(e)}")
            return []
    
    def build_reasoning_chain(self, 
                            initial_facts: Set[str],
                            target_conclusion: str,
                            max_length: int = 5) -> List[List[str]]:
        """
        构建推理链
        
        Args:
            initial_facts: 初始事实集合
            target_conclusion: 目标结论
            max_length: 最大推理链长度
            
        Returns:
            List[List[str]]: 推理链列表
        """
        chains = []
        
        try:
            # 使用BFS搜索推理链
            from collections import deque
            
            queue = deque([(list(initial_facts), [])])  # (current_facts, reasoning_chain)
            visited = set()
            
            while queue and len(chains) < 10:  # 限制结果数量
                current_facts, chain = queue.popleft()
                
                # 构建状态键
                state_key = tuple(sorted(current_facts))
                if state_key in visited:
                    continue
                visited.add(state_key)
                
                # 检查是否达到目标
                if target_conclusion.lower() in [fact.lower() for fact in current_facts]:
                    chains.append(chain)
                    continue
                
                # 限制推理链长度
                if len(chain) >= max_length:
                    continue
                
                # 应用规则生成新事实
                for rule in self.induced_rules.values():
                    if rule.premise.issubset(set(current_facts)):
                        new_facts = current_facts + list(rule.conclusion)
                        
                        # 检查是否产生新事实
                        if set(new_facts) != set(current_facts):
                            new_chain = chain + [f"应用规则 {rule.rule_id}: {rule.premise} -> {rule.conclusion}"]
                            queue.append((new_facts, new_chain))
            
            self.logger.debug(f"构建推理链完成，找到 {len(chains)} 条链")
            return chains
            
        except Exception as e:
            self.logger.error(f"构建推理链失败: {str(e)}")
            return []
    
    def get_knowledge_quality(self) -> Dict[str, float]:
        """
        获取知识质量评估
        
        Returns:
            Dict[str, float]: 质量指标
        """
        try:
            quality_metrics = {}
            
            # 规则一致性
            consistency_score = self._calculate_consistency_score()
            quality_metrics['consistency'] = consistency_score
            
            # 规则覆盖度
            coverage_score = self._calculate_coverage_score()
            quality_metrics['coverage'] = coverage_score
            
            # 规则精度
            accuracy_score = self._calculate_accuracy_score()
            quality_metrics['accuracy'] = accuracy_score
            
            # 知识完整性
            completeness_score = self._calculate_completeness_score()
            quality_metrics['completeness'] = completeness_score
            
            # 综合质量分数
            overall_score = np.mean(list(quality_metrics.values()))
            quality_metrics['overall'] = overall_score
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"知识质量评估失败: {str(e)}")
            return {'overall': 0.0}
    
    def export_knowledge(self, format: str = 'json') -> Dict[str, Any]:
        """
        导出知识
        
        Args:
            format: 导出格式
            
        Returns:
            Dict[str, Any]: 导出的知识数据
        """
        if format == 'json':
            return self._export_json()
        elif format == 'rules':
            return self._export_rules()
        elif format == 'patterns':
            return self._export_patterns()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _update_feature_matrix(self, observation: Dict[str, Any]):
        """更新特征矩阵"""
        try:
            if self.feature_matrix is None:
                self.feature_names = list(observation.keys())
                self.feature_matrix = np.array([[observation.get(name, 0) for name in self.feature_names]])
            else:
                # 添加新行
                new_row = [observation.get(name, 0) for name in self.feature_names]
                self.feature_matrix = np.vstack([self.feature_matrix, new_row])
                
        except Exception as e:
            self.logger.error(f"更新特征矩阵失败: {str(e)}")
    
    def _induce_frequent_patterns(self, **kwargs) -> List[InducedRule]:
        """归纳频繁模式"""
        rules = []
        
        try:
            # 转换观察数据为事务格式
            transactions = []
            for obs in self.observations:
                transaction = set()
                for key, value in obs.items():
                    if isinstance(value, str):
                        transaction.add(f"{key}:{value}")
                    elif isinstance(value, (int, float)) and value != 0:
                        transaction.add(f"{key}:{value}")
                transactions.append(transaction)
            
            # 挖掘频繁项集（简化实现）
            item_counts = Counter()
            for transaction in transactions:
                for item in transaction:
                    item_counts[item] += 1
            
            # 找出频繁项集
            frequent_items = {item: count for item, count in item_counts.items() 
                            if count / len(transactions) >= self.min_support}
            
            # 生成关联规则
            frequent_items_list = list(frequent_items.keys())
            
            for i, item_a in enumerate(frequent_items_list):
                for j, item_b in enumerate(frequent_items_list[i+1:], i+1):
                    # 计算 A -> B 的置信度
                    ab_count = sum(1 for transaction in transactions 
                                 if item_a in transaction and item_b in transaction)
                    a_count = frequent_items[item_a]
                    
                    if a_count > 0:
                        confidence = ab_count / a_count
                        
                        if confidence >= self.min_confidence:
                            # 创建规则
                            rule_id = f"freq_pattern_{len(rules)}"
                            
                            rule = InducedRule(
                                rule_id=rule_id,
                                premise={item_a},
                                conclusion={item_b},
                                confidence=confidence,
                                support=ab_count / len(transactions),
                                lift=confidence / (item_counts.get(item_b, 0) / len(transactions)),
                                method=InductionMethod.FREQUENT_PATTERN,
                                examples=[obs for obs in self.observations 
                                        if item_a in str(obs) and item_b in str(obs)]
                            )
                            rules.append(rule)
            
            self.logger.debug(f"频繁模式归纳完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"频繁模式归纳失败: {str(e)}")
            return []
    
    def _induce_rules(self, **kwargs) -> List[InducedRule]:
        """规则归纳"""
        rules = []
        
        try:
            # 简化的规则归纳：基于属性共现
            attribute_combinations = defaultdict(int)
            attribute_contexts = defaultdict(list)
            
            for obs in self.observations:
                attributes = list(obs.keys())
                
                # 记录属性组合
                for combo_size in range(2, min(len(attributes), 4)):  # 最多3个属性的组合
                    for combo in combinations(attributes, combo_size):
                        combo_key = tuple(sorted(combo))
                        attribute_combinations[combo_key] += 1
                        attribute_contexts[combo_key].append(obs)
            
            # 生成规则
            for combo, count in attribute_combinations.items():
                if count >= self.min_support * len(self.observations):
                    # 计算条件概率
                    context_data = attribute_contexts[combo]
                    
                    # 简化的规则生成：假设最后一个属性是结果
                    if len(combo) >= 2:
                        premise_attrs = combo[:-1]
                        conclusion_attr = combo[-1]
                        
                        premise_values = []
                        for obs in context_data:
                            premise_val = tuple(obs[attr] for attr in premise_attrs)
                            premise_values.append(premise_val)
                        
                        # 计算最频繁的结论值
                        conclusion_values = [obs[conclusion_attr] for obs in context_data]
                        most_common_conclusion = Counter(conclusion_values).most_common(1)[0]
                        
                        if most_common_conclusion[1] / len(conclusion_values) >= self.min_confidence:
                            rule_id = f"rule_induction_{len(rules)}"
                            
                            premise = {f"{attr}:{premise_val}" for attr, premise_val in 
                                     zip(premise_attrs, premise_values[0]) if attr != 'timestamp'}
                            conclusion = {f"{conclusion_attr}:{most_common_conclusion[0]}"}
                            
                            rule = InducedRule(
                                rule_id=rule_id,
                                premise=premise,
                                conclusion=conclusion,
                                confidence=most_common_conclusion[1] / len(conclusion_values),
                                support=count / len(self.observations),
                                lift=1.0,  # 简化计算
                                method=InductionMethod.RULE_INDUCTION,
                                examples=context_data[:10]  # 限制示例数量
                            )
                            rules.append(rule)
            
            self.logger.debug(f"规则归纳完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"规则归纳失败: {str(e)}")
            return []
    
    def _infer_causal_relations(self, **kwargs) -> List[InducedRule]:
        """因果关系推断"""
        rules = []
        
        try:
            if not self.enable_causal:
                return rules
            
            # 简化的因果推断：基于时间序列和相关性
            numeric_columns = []
            for col in self.feature_names:
                if col != 'timestamp':
                    values = self.feature_matrix[:, self.feature_names.index(col)]
                    if np.any(np.isfinite(values)):
                        numeric_columns.append(col)
            
            # 计算相关性矩阵
            if len(numeric_columns) >= 2:
                data_matrix = self.feature_matrix[:, [self.feature_names.index(col) for col in numeric_columns]]
                
                # 移除包含NaN的行
                valid_rows = ~np.any(np.isnan(data_matrix), axis=1)
                clean_data = data_matrix[valid_rows]
                
                if len(clean_data) > 10:
                    correlation_matrix = np.corrcoef(clean_data.T)
                    
                    # 寻找强相关关系作为潜在的因果关系
                    for i, cause in enumerate(numeric_columns):
                        for j, effect in enumerate(numeric_columns[i+1:], i+1):
                            correlation = abs(correlation_matrix[i][j])
                            
                            if correlation >= self.induction_threshold:
                                rule_id = f"causal_{len(rules)}"
                                
                                # 确定因果方向（简化：数值变化的方向）
                                cause_changes = np.diff(clean_data[:, i])
                                effect_changes = np.diff(clean_data[:, j])
                                
                                # 计算因果强度
                                if np.any(cause_changes != 0):
                                    correlation_coeff, p_value = pearsonr(cause_changes, effect_changes)
                                    causal_strength = abs(correlation_coeff)
                                    
                                    if causal_strength >= self.induction_threshold:
                                        rule = InducedRule(
                                            rule_id=rule_id,
                                            premise={f"cause:{cause}"},
                                            conclusion={f"effect:{effect}"},
                                            confidence=causal_strength,
                                            support=len(clean_data) / len(self.observations),
                                            lift=causal_strength,
                                            method=InductionMethod.CAUSAL_INFERENCE,
                                            reasoning_chain=[
                                                f"观察到 {cause} 和 {effect} 的强相关性: {correlation:.3f}",
                                                f"计算时间序列因果性: {causal_strength:.3f}",
                                                "基于统计显著性推断因果关系"
                                            ],
                                            metadata={
                                                'correlation': correlation,
                                                'p_value': p_value,
                                                'causal_direction': 'positive' if correlation > 0 else 'negative'
                                            }
                                        )
                                        rules.append(rule)
            
            self.logger.debug(f"因果推断完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"因果推断失败: {str(e)}")
            return []
    
    def _analogical_reasoning(self, **kwargs) -> List[InducedRule]:
        """类比推理"""
        rules = []
        
        if not self.enable_analogical:
            return rules
        
        try:
            # 基于相似性的类比推理
            if len(self.observations) < 3:
                return rules
            
            # 使用聚类找到相似的观察
            numeric_data = []
            valid_indices = []
            
            for i, obs in enumerate(self.observations):
                numeric_values = []
                for col in self.feature_names:
                    if col != 'timestamp':
                        val = obs.get(col, 0)
                        if isinstance(val, (int, float)):
                            numeric_values.append(val)
                        else:
                            # 尝试编码分类变量
                            try:
                                numeric_values.append(hash(str(val)) % 1000 / 1000.0)
                            except:
                                numeric_values.append(0.0)
                
                if len(numeric_values) > 0:
                    numeric_data.append(numeric_values)
                    valid_indices.append(i)
            
            if len(numeric_data) >= 3:
                # 执行聚类
                numeric_array = np.array(numeric_data)
                
                # 标准化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_array)
                
                # DBSCAN聚类
                clustering = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = clustering.fit_predict(scaled_data)
                
                # 为每个聚类生成类比规则
                for cluster_id in set(cluster_labels):
                    if cluster_id == -1:  # 噪声点
                        continue
                    
                    cluster_indices = [valid_indices[i] for i, label in enumerate(cluster_labels) 
                                     if label == cluster_id]
                    
                    if len(cluster_indices) >= 2:
                        cluster_obs = [self.observations[i] for i in cluster_indices]
                        
                        # 寻找共同模式
                        common_patterns = self._find_common_patterns(cluster_obs)
                        
                        for pattern in common_patterns:
                            rule_id = f"analogy_{len(rules)}"
                            
                            rule = InducedRule(
                                rule_id=rule_id,
                                premise=set(pattern['premise']),
                                conclusion=set(pattern['conclusion']),
                                confidence=pattern['confidence'],
                                support=len(cluster_indices) / len(self.observations),
                                lift=1.0,
                                method=InductionMethod.ANALOGICAL_REASONING,
                                examples=cluster_obs[:5],
                                reasoning_chain=[
                                    f"发现 {len(cluster_indices)} 个相似案例",
                                    f"识别共同模式: {pattern}",
                                    "基于类比推断新知识"
                                ],
                                metadata={
                                    'cluster_size': len(cluster_indices),
                                    'cluster_id': cluster_id,
                                    'similarity_threshold': 0.5
                                }
                            )
                            rules.append(rule)
            
            self.logger.debug(f"类比推理完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"类比推理失败: {str(e)}")
            return []
    
    def _case_based_reasoning(self, **kwargs) -> List[InducedRule]:
        """基于案例的推理"""
        rules = []
        
        try:
            if len(self.observations) < 2:
                return rules
            
            # 找到相似的案例并归纳通用模式
            for i, case1 in enumerate(self.observations):
                similar_cases = []
                
                # 寻找相似案例
                for j, case2 in enumerate(self.observations):
                    if i != j:
                        similarity = self._calculate_case_similarity(case1, case2)
                        if similarity > 0.7:  # 相似度阈值
                            similar_cases.append(case2)
                
                if len(similar_cases) >= 2:  # 至少2个相似案例
                    # 归纳共同特征
                    common_features = self._find_common_features(case1, similar_cases)
                    
                    if common_features:
                        rule_id = f"case_based_{len(rules)}"
                        
                        premise_features = {k: v for k, v in common_features.items() 
                                          if k not in ['outcome', 'result', 'conclusion']}
                        conclusion_features = {k: v for k, v in common_features.items() 
                                             if k in ['outcome', 'result', 'conclusion']}
                        
                        if premise_features and conclusion_features:
                            rule = InducedRule(
                                rule_id=rule_id,
                                premise=set(f"{k}:{v}" for k, v in premise_features.items()),
                                conclusion=set(f"{k}:{v}" for k, v in conclusion_features.items()),
                                confidence=len(similar_cases) / len(self.observations),
                                support=1.0,  # 基于案例的规则支持度为1
                                lift=1.0,
                                method=InductionMethod.CASE_BASED,
                                examples=[case1] + similar_cases[:3],
                                reasoning_chain=[
                                    f"找到 {len(similar_cases)} 个相似案例",
                                    f"提取共同特征: {common_features}",
                                    "基于案例归纳规则"
                                ]
                            )
                            rules.append(rule)
            
            self.logger.debug(f"基于案例的推理完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"基于案例的推理失败: {str(e)}")
            return []
    
    def _statistical_induction(self, **kwargs) -> List[InducedRule]:
        """统计归纳"""
        rules = []
        
        try:
            if len(self.observations) < 10:
                return rules
            
            # 基于统计显著性的归纳
            numeric_columns = []
            for col in self.feature_names:
                if col != 'timestamp':
                    try:
                        values = self.feature_matrix[:, self.feature_names.index(col)]
                        if np.any(np.isfinite(values)) and len(set(values)) > 1:
                            numeric_columns.append(col)
                    except:
                        continue
            
            if len(numeric_columns) >= 2:
                # 分析变量间的统计关系
                for i, var1 in enumerate(numeric_columns):
                    for var2 in numeric_columns[i+1:]:
                        # 计算统计指标
                        data1 = self.feature_matrix[:, self.feature_names.index(var1)]
                        data2 = self.feature_matrix[:, self.feature_names.index(var2)]
                        
                        # 移除NaN值
                        valid_mask = np.isfinite(data1) & np.isfinite(data2)
                        clean_data1 = data1[valid_mask]
                        clean_data2 = data2[valid_mask]
                        
                        if len(clean_data1) >= 10:
                            # 计算相关性
                            correlation, p_value = pearsonr(clean_data1, clean_data2)
                            
                            # 如果相关性显著且强
                            if abs(correlation) >= self.induction_threshold and p_value < 0.05:
                                rule_id = f"statistical_{len(rules)}"
                                
                                # 确定关系类型
                                relation_type = "positive" if correlation > 0 else "negative"
                                strength = abs(correlation)
                                
                                rule = InducedRule(
                                    rule_id=rule_id,
                                    premise={f"variable:{var1}"},
                                    conclusion={f"variable:{var2},{relation_type}"},
                                    confidence=strength,
                                    support=np.sum(valid_mask) / len(data1),
                                    lift=strength,
                                    method=InductionMethod.STATISTICAL,
                                    reasoning_chain=[
                                        f"统计检验: {var1} vs {var2}",
                                        f"相关系数: {correlation:.3f} (p={p_value:.4f})",
                                        "统计显著性检验通过，归纳关系"
                                    ],
                                    metadata={
                                        'correlation': correlation,
                                        'p_value': p_value,
                                        'sample_size': len(clean_data1),
                                        'significance_level': 0.05
                                    }
                                )
                                rules.append(rule)
            
            self.logger.debug(f"统计归纳完成，生成 {len(rules)} 个规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"统计归纳失败: {str(e)}")
            return []
    
    def _validate_rules(self, rules: List[InducedRule]) -> List[InducedRule]:
        """验证规则"""
        validated_rules = []
        
        try:
            for rule in rules:
                # 检查基本条件
                if (rule.confidence >= self.min_confidence and 
                    rule.support >= self.min_support and
                    rule.premise and rule.conclusion):
                    
                    # 检查示例支持
                    if rule.examples:
                        # 验证示例是否真的支持规则
                        supported_examples = self._verify_rule_examples(rule)
                        
                        if len(supported_examples) >= len(rule.examples) * 0.5:  # 至少50%的示例支持
                            rule.examples = supported_examples[:10]  # 保留验证后的示例
                            validated_rules.append(rule)
                
            self.logger.debug(f"规则验证完成，保留 {len(validated_rules)} / {len(rules)} 个规则")
            return validated_rules
            
        except Exception as e:
            self.logger.error(f"规则验证失败: {str(e)}")
            return rules
    
    def _verify_rule_examples(self, rule: InducedRule) -> List[Dict[str, Any]]:
        """验证规则示例"""
        supported_examples = []
        
        try:
            for example in rule.examples:
                # 检查示例是否满足规则的前提和结论
                premise_satisfied = True
                conclusion_satisfied = True
                
                # 检查前提
                for premise in rule.premise:
                    if ':' in premise:
                        attr, value = premise.split(':', 1)
                        if example.get(attr) != value:
                            premise_satisfied = False
                            break
                
                # 检查结论
                for conclusion in rule.conclusion:
                    if ':' in conclusion:
                        attr, value = conclusion.split(':', 1)
                        if example.get(attr) != value:
                            conclusion_satisfied = False
                            break
                
                if premise_satisfied and conclusion_satisfied:
                    supported_examples.append(example)
            
            return supported_examples
            
        except Exception as e:
            self.logger.error(f"验证规则示例失败: {str(e)}")
            return rule.examples
    
    def _find_applicable_rules(self, premise: Set[str]) -> List[InducedRule]:
        """查找适用的规则"""
        applicable = []
        
        for rule in self.induced_rules.values():
            # 检查规则前提是否被当前事实满足
            if rule.premise.issubset(premise):
                applicable.append(rule)
        
        return applicable
    
    def _calculate_premise_satisfaction(self, current_facts: Set[str], rule_premise: Set[str]) -> float:
        """计算前提满足度"""
        satisfied = len(rule_premise.intersection(current_facts))
        return satisfied / len(rule_premise) if rule_premise else 0.0
    
    def _evaluate_explanation_quality(self, rule: InducedRule, observation: str) -> float:
        """评估解释质量"""
        quality = rule.confidence
        
        # 考虑规则的简单性
        simplicity = 1.0 / (1.0 + len(rule.premise))
        quality *= simplicity
        
        # 考虑规则的支持证据数量
        evidence_factor = min(1.0, len(rule.examples) / 10.0)
        quality *= evidence_factor
        
        return min(1.0, quality)
    
    def _find_similar_cases(self, source_domain: Dict[str, Any], target_domain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """寻找相似案例"""
        similar_cases = []
        
        try:
            # 计算与目标域相似的源域案例
            for obs in self.observations:
                similarity = self._calculate_domain_similarity(obs, target_domain)
                
                if similarity > 0.6:  # 相似度阈值
                    similar_cases.append(obs)
            
            return similar_cases
            
        except Exception as e:
            self.logger.error(f"寻找相似案例失败: {str(e)}")
            return []
    
    def _extract_analogy_mapping(self, source_case: Dict[str, Any], similar_case: Dict[str, Any], 
                               target_case: Dict[str, Any]) -> Dict[str, str]:
        """提取类比映射"""
        mapping = {}
        
        try:
            # 简化的属性映射：基于属性名称相似性
            source_attrs = set(source_case.keys())
            target_attrs = set(target_case.keys())
            
            for s_attr in source_attrs:
                for t_attr in target_attrs:
                    # 计算属性名相似度
                    if self._string_similarity(s_attr, t_attr) > 0.5:
                        mapping[s_attr] = t_attr
                        break
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"提取类比映射失败: {str(e)}")
            return {}
    
    def _infer_by_analogy(self, similar_case: Dict[str, Any], target_case: Dict[str, Any], 
                        mapping: Dict[str, str]) -> Dict[str, Any]:
        """基于类比进行推理"""
        inferred = {}
        
        try:
            # 将源域案例的属性通过映射转换到目标域
            for source_attr, target_attr in mapping.items():
                if source_attr in similar_case and target_attr not in target_case:
                    inferred[target_attr] = similar_case[source_attr]
            
            return inferred
            
        except Exception as e:
            self.logger.error(f"基于类比推理失败: {str(e)}")
            return {}
    
    def _calculate_analogy_confidence(self, source_case: Dict[str, Any], target_case: Dict[str, Any], 
                                    property_name: str) -> float:
        """计算类比置信度"""
        try:
            # 基于源案例和目标案例的相似度计算置信度
            similarity = self._calculate_domain_similarity(source_case, target_case)
            
            # 考虑属性的重要性（简化）
            importance_factor = 0.8  # 默认重要性
            
            return similarity * importance_factor
            
        except Exception as e:
            self.logger.error(f"计算类比置信度失败: {str(e)}")
            return 0.0
    
    def _calculate_causal_strength(self, cause: str, effect: str, data: List[Dict[str, Any]]) -> float:
        """计算因果强度"""
        try:
            if len(data) < 10:
                return 0.0
            
            # 简化因果强度计算：基于时间先后和相关性
            cause_values = []
            effect_values = []
            
            for obs in data:
                if cause in obs and effect in obs:
                    cause_val = obs[cause]
                    effect_val = obs[effect]
                    
                    if isinstance(cause_val, (int, float)) and isinstance(effect_val, (int, float)):
                        cause_values.append(cause_val)
                        effect_values.append(effect_val)
            
            if len(cause_values) < 5:
                return 0.0
            
            # 计算相关性作为因果强度的代理
            correlation, p_value = pearsonr(cause_values, effect_values)
            
            if p_value < 0.05:
                return abs(correlation)
            else:
                return abs(correlation) * 0.5  # 非显著相关时降低权重
            
        except Exception as e:
            self.logger.error(f"计算因果强度失败: {str(e)}")
            return 0.0
    
    def _validate_causal_relationship(self, cause: str, effect: str, data: List[Dict[str, Any]]) -> float:
        """验证因果关系"""
        try:
            # 简化的因果验证：基于一致性和特异性
            consistency_score = 0.0
            specificity_score = 0.0
            
            # 一致性：原因变化时，效果是否也变化
            cause_effect_pairs = []
            for obs in data:
                if cause in obs and effect in obs:
                    cause_effect_pairs.append((obs[cause], obs[effect]))
            
            if len(cause_effect_pairs) >= 3:
                # 计算变化方向的一致性
                cause_changes = [pair[0] for pair in cause_effect_pairs]
                effect_changes = [pair[1] for pair in cause_effect_pairs]
                
                # 简单的相关性验证
                if len(set(cause_changes)) > 1 and len(set(effect_changes)) > 1:
                    try:
                        correlation, _ = pearsonr(cause_changes, effect_changes)
                        consistency_score = abs(correlation)
                    except:
                        consistency_score = 0.5
            
            # 特异性：效果主要由该原因引起
            # 这里简化为基于该原因解释效果变异的比例
            specificity_score = min(1.0, len(cause_effect_pairs) / len(data))
            
            return (consistency_score + specificity_score) / 2.0
            
        except Exception as e:
            self.logger.error(f"验证因果关系失败: {str(e)}")
            return 0.0
    
    def _find_common_patterns(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """寻找共同模式"""
        patterns = []
        
        try:
            if len(cases) < 2:
                return patterns
            
            # 分析所有案例的共同属性
            all_attributes = set()
            for case in cases:
                all_attributes.update(case.keys())
            
            # 找出频繁出现的属性值组合
            value_combinations = defaultdict(int)
            
            for case in cases:
                for attr in all_attributes:
                    if attr in case:
                        value_combinations[(attr, case[attr])] += 1
            
            # 找出频繁组合
            frequent_combinations = {combo: count for combo, count in value_combinations.items() 
                                   if count >= len(cases) * 0.6}
            
            if frequent_combinations:
                # 生成模式
                premise_attrs = []
                conclusion_attrs = []
                
                for (attr, value), count in frequent_combinations.items():
                    if count >= len(cases) * 0.8:  # 高度一致的属性作为结论
                        conclusion_attrs.append(f"{attr}:{value}")
                    else:  # 作为前提
                        premise_attrs.append(f"{attr}:{value}")
                
                if premise_attrs and conclusion_attrs:
                    patterns.append({
                        'premise': premise_attrs,
                        'conclusion': conclusion_attrs,
                        'confidence': len(frequent_combinations) / len(all_attributes)
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"寻找共同模式失败: {str(e)}")
            return []
    
    def _calculate_case_similarity(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> float:
        """计算案例相似度"""
        try:
            common_attributes = set(case1.keys()) & set(case2.keys())
            
            if not common_attributes:
                return 0.0
            
            # 去除时间戳
            common_attributes.discard('timestamp')
            
            if not common_attributes:
                return 0.0
            
            # 计算属性匹配度
            matches = 0
            for attr in common_attributes:
                if case1[attr] == case2[attr]:
                    matches += 1
            
            return matches / len(common_attributes)
            
        except Exception as e:
            self.logger.error(f"计算案例相似度失败: {str(e)}")
            return 0.0
    
    def _find_common_features(self, case: Dict[str, Any], similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """寻找共同特征"""
        common_features = {}
        
        try:
            # 收集所有相似案例的属性
            all_attributes = set(case.keys())
            for similar_case in similar_cases:
                all_attributes.update(similar_case.keys())
            
            # 去除时间戳
            all_attributes.discard('timestamp')
            
            # 找出在所有案例中都出现的属性及其值
            for attr in all_attributes:
                values = [case.get(attr) for case in [case] + similar_cases if attr in case]
                
                # 检查是否为共同值
                if len(values) == len([case] + similar_cases):
                    value_counts = Counter(values)
                    most_common_value, count = value_counts.most_common(1)[0]
                    
                    if count >= len(values) * 0.6:  # 至少60%的案例有这个值
                        common_features[attr] = most_common_value
            
            return common_features
            
        except Exception as e:
            self.logger.error(f"寻找共同特征失败: {str(e)}")
            return {}
    
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
    
    def _calculate_domain_similarity(self, domain1: Dict[str, Any], domain2: Dict[str, Any]) -> float:
        """计算域相似度"""
        try:
            common_keys = set(domain1.keys()) & set(domain2.keys())
            common_keys.discard('timestamp')
            
            if not common_keys:
                return 0.0
            
            # 计算属性匹配度
            matches = 0
            for key in common_keys:
                if domain1[key] == domain2[key]:
                    matches += 1
            
            return matches / len(common_keys)
            
        except Exception as e:
            self.logger.error(f"计算域相似度失败: {str(e)}")
            return 0.0
    
    def _calculate_consistency_score(self) -> float:
        """计算一致性分数"""
        if len(self.induced_rules) < 2:
            return 1.0
        
        # 检查规则间是否存在冲突
        conflicts = 0
        total_pairs = 0
        
        rule_list = list(self.induced_rules.values())
        for i, rule1 in enumerate(rule_list):
            for rule2 in rule_list[i+1:]:
                total_pairs += 1
                
                # 检查是否有相同前提但不同结论
                if rule1.premise == rule2.premise and rule1.conclusion != rule2.conclusion:
                    conflicts += 1
        
        if total_pairs == 0:
            return 1.0
        
        consistency = 1.0 - (conflicts / total_pairs)
        return max(0.0, consistency)
    
    def _calculate_coverage_score(self) -> float:
        """计算覆盖度分数"""
        if not self.observations:
            return 0.0
        
        # 检查有多少观察实例能被规则解释
        covered_observations = 0
        
        for obs in self.observations:
            obs_facts = set()
            for key, value in obs.items():
                if key != 'timestamp':
                    obs_facts.add(f"{key}:{value}")
            
            # 检查是否有规则能解释这个观察
            has_explanation = False
            for rule in self.induced_rules.values():
                # 检查规则的前提是否在观察中满足
                if rule.premise.issubset(obs_facts):
                    has_explanation = True
                    break
            
            if has_explanation:
                covered_observations += 1
        
        return covered_observations / len(self.observations)
    
    def _calculate_accuracy_score(self) -> float:
        """计算准确度分数"""
        if not self.induced_rules:
            return 1.0
        
        # 基于规则置信度的加权平均
        total_confidence = sum(rule.confidence for rule in self.induced_rules.values())
        return total_confidence / len(self.induced_rules) if self.induced_rules else 0.0
    
    def _calculate_completeness_score(self) -> float:
        """计算完整性分数"""
        if not self.observations:
            return 1.0
        
        # 检查规则是否覆盖了所有重要的观察模式
        important_attributes = set()
        for obs in self.observations:
            important_attributes.update(obs.keys())
        important_attributes.discard('timestamp')
        
        if not important_attributes:
            return 1.0
        
        # 检查每个重要属性是否在某个规则中被使用
        covered_attributes = set()
        for rule in self.induced_rules:
            for premise in self.induced_rules[rule].premise:
                if ':' in premise:
                    attr = premise.split(':')[0]
                    covered_attributes.add(attr)
            for conclusion in self.induced_rules[rule].conclusion:
                if ':' in conclusion:
                    attr = conclusion.split(':')[0]
                    covered_attributes.add(attr)
        
        return len(covered_attributes) / len(important_attributes)
    
    def _export_json(self) -> Dict[str, Any]:
        """导出JSON格式"""
        rules_data = {}
        for rule_id, rule in self.induced_rules.items():
            rules_data[rule_id] = {
                'rule_id': rule.rule_id,
                'premise': list(rule.premise),
                'conclusion': list(rule.conclusion),
                'confidence': rule.confidence,
                'support': rule.support,
                'lift': rule.lift,
                'method': rule.method.value,
                'examples': rule.examples[:5],  # 限制示例数量
                'reasoning_chain': rule.reasoning_chain,
                'created_at': rule.created_at.isoformat(),
                'metadata': rule.metadata
            }
        
        patterns_data = {}
        for pattern_id, pattern in self.knowledge_patterns.items():
            patterns_data[pattern_id] = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'elements': pattern.elements,
                'relationships': pattern.relationships,
                'frequency': pattern.frequency,
                'stability': pattern.stability,
                'context': pattern.context,
                'confidence': pattern.confidence
            }
        
        return {
            'induced_rules': rules_data,
            'knowledge_patterns': patterns_data,
            'statistics': self.induction_stats,
            'observations_count': len(self.observations)
        }
    
    def _export_rules(self) -> List[Dict[str, Any]]:
        """导出规则格式"""
        rules_list = []
        
        for rule in self.induced_rules.values():
            rules_list.append({
                'id': rule.rule_id,
                'premise': list(rule.premise),
                'conclusion': list(rule.conclusion),
                'confidence': rule.confidence,
                'support': rule.support,
                'method': rule.method.value,
                'description': ' -> '.join(rule.premise) + ' => ' + ' -> '.join(rule.conclusion)
            })
        
        return rules_list
    
    def _export_patterns(self) -> List[Dict[str, Any]]:
        """导出模式格式"""
        patterns_list = []
        
        for pattern in self.knowledge_patterns.values():
            patterns_list.append({
                'id': pattern.pattern_id,
                'type': pattern.pattern_type,
                'elements': pattern.elements,
                'relationships': pattern.relationships,
                'frequency': pattern.frequency,
                'stability': pattern.stability,
                'confidence': pattern.confidence
            })
        
        return patterns_list