# -*- coding: utf-8 -*-
"""
迁移分析器
Transfer Analyzer

该模块负责分析和评估知识在不同领域间的迁移效果。
通过多维度的指标计算和统计分析，全面评估跨域学习的
迁移效率、质量和可行性。

主要功能：
- 迁移效率测量
- 知识损失评估
- 迁移质量分析
- 最佳迁移路径规划
- 迁移瓶颈识别
- 迁移策略优化

作者: AI系统
日期: 2025-11-13
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import json
import asyncio
import math
from collections import defaultdict

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TransferMetrics:
    """迁移指标数据结构"""
    source_domain: str
    target_domain: str
    transfer_efficiency: float              # 迁移效率
    knowledge_retention: float              # 知识保持率
    transfer_quality: float                 # 迁移质量
    semantic_preservation: float           # 语义保真度
    structural_integrity: float            # 结构完整性
    functional_preservation: float         # 功能保真度
    transfer_time: float                   # 迁移时间
    memory_usage: float                    # 内存使用
    success_rate: float                    # 成功率
    failure_modes: List[str]               # 失败模式
    optimization_potential: float          # 优化潜力


@dataclass
class TransferPath:
    """迁移路径数据结构"""
    path_id: str
    sequence: List[str]                    # 迁移序列
    total_efficiency: float                # 总体效率
    quality_score: float                   # 质量评分
    estimated_time: float                  # 估计时间
    complexity_score: float                # 复杂度评分
    confidence_level: float               # 置信水平


class TransferAnalyzer:
    """
    迁移分析器
    
    负责分析跨域知识迁移的效果，识别迁移过程中的关键因素和瓶颈，
    为跨域学习提供科学的评估和优化建议。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('TransferAnalyzer')
        
        # 迁移参数配置
        self.transfer_params = {
            'efficiency_threshold': config.get('efficiency_threshold', 0.7),
            'quality_threshold': config.get('quality_threshold', 0.6),
            'max_transfer_hops': config.get('max_transfer_hops', 3),
            'similarity_weight': config.get('similarity_weight', 0.4),
            'complexity_weight': config.get('complexity_weight', 0.3),
            'capacity_weight': config.get('capacity_weight', 0.3),
            'enable_detailed_analysis': config.get('enable_detailed_analysis', True)
        }
        
        # 领域特征矩阵
        self.domain_features = {
            'game': {
                'abstractness': 0.7,
                'complexity': 0.6,
                'predictability': 0.8,
                'structure_rigidity': 0.5,
                'semantic_density': 0.6,
                'functional_scope': 0.7
            },
            'physics': {
                'abstractness': 0.9,
                'complexity': 0.9,
                'predictability': 0.95,
                'structure_rigidity': 0.8,
                'semantic_density': 0.8,
                'functional_scope': 0.8
            },
            'social': {
                'abstractness': 0.6,
                'complexity': 0.8,
                'predictability': 0.4,
                'structure_rigidity': 0.3,
                'semantic_density': 0.9,
                'functional_scope': 0.9
            },
            'language': {
                'abstractness': 0.8,
                'complexity': 0.7,
                'predictability': 0.6,
                'structure_rigidity': 0.7,
                'semantic_density': 0.95,
                'functional_scope': 0.8
            },
            'spatial': {
                'abstractness': 0.6,
                'complexity': 0.6,
                'predictability': 0.8,
                'structure_rigidity': 0.6,
                'semantic_density': 0.6,
                'functional_scope': 0.6
            }
        }
        
        # 迁移历史记录
        self.transfer_history = []
        
        self.logger.info("迁移分析器初始化完成")
    
    async def measure_transfer_efficiency(self,
                                        source_domains: List[str],
                                        target_domains: List[str],
                                        knowledge_base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        测量迁移效率
        
        这是系统的核心方法，用于评估知识在不同领域间的迁移效率。
        通过多维度指标计算，全面分析迁移效果。
        
        Args:
            source_domains: 源领域列表
            target_domains: 目标领域列表
            knowledge_base: 知识库
            
        Returns:
            Dict: 迁移效率测量结果
        """
        self.logger.info(f"开始迁移效率测量: {source_domains} -> {target_domains}")
        
        transfer_results = {}
        
        for target_domain in target_domains:
            self.logger.info(f"分析迁移到领域: {target_domain}")
            
            # 为每个源领域计算迁移效率
            domain_transfer_metrics = {}
            
            for source_domain in source_domains:
                # 计算领域相似性
                similarity_score = await self._calculate_domain_similarity(source_domain, target_domain)
                
                # 计算知识保持率
                retention_rate = await self._calculate_knowledge_retention(
                    source_domain, target_domain, knowledge_base
                )
                
                # 计算迁移质量
                quality_score = await self._calculate_transfer_quality(
                    source_domain, target_domain, knowledge_base
                )
                
                # 计算语义保真度
                semantic_score = await self._calculate_semantic_preservation(
                    source_domain, target_domain, knowledge_base
                )
                
                # 计算结构完整性
                structural_score = await self._calculate_structural_integrity(
                    source_domain, target_domain, knowledge_base
                )
                
                # 计算功能保真度
                functional_score = await self._calculate_functional_preservation(
                    source_domain, target_domain, knowledge_base
                )
                
                # 计算综合迁移效率
                transfer_efficiency = await self._calculate_transfer_efficiency(
                    similarity_score, retention_rate, quality_score,
                    semantic_score, structural_score, functional_score
                )
                
                # 创建迁移指标对象
                transfer_metrics = TransferMetrics(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    transfer_efficiency=transfer_efficiency,
                    knowledge_retention=retention_rate,
                    transfer_quality=quality_score,
                    semantic_preservation=semantic_score,
                    structural_integrity=structural_score,
                    functional_preservation=functional_score,
                    transfer_time=0.0,  # 将在后续计算
                    memory_usage=0.0,   # 将在后续计算
                    success_rate=1.0,   # 将在后续计算
                    failure_modes=[],   # 将在后续识别
                    optimization_potential=self._calculate_optimization_potential(transfer_efficiency)
                )
                
                domain_transfer_metrics[source_domain] = transfer_metrics
            
            # 为目标领域选择最佳源领域
            best_source = max(domain_transfer_metrics.items(), 
                            key=lambda x: x[1].transfer_efficiency)
            
            transfer_results[target_domain] = {
                'best_source': best_source[0],
                'best_efficiency': best_source[1].transfer_efficiency,
                'all_sources': domain_transfer_metrics,
                'optimization_suggestions': await self._generate_optimization_suggestions(
                    source_domain, target_domain, best_source[1]
                )
            }
        
        self.logger.info("迁移效率测量完成")
        return transfer_results
    
    async def _calculate_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """计算领域相似性"""
        
        if source_domain not in self.domain_features or target_domain not in self.domain_features:
            return 0.5  # 默认中等相似性
        
        source_features = self.domain_features[source_domain]
        target_features = self.domain_features[target_domain]
        
        # 计算特征向量的余弦相似性
        dot_product = sum(source_features[feat] * target_features[feat] 
                         for feat in source_features.keys())
        
        source_magnitude = math.sqrt(sum(source_features[feat] ** 2 for feat in source_features.keys()))
        target_magnitude = math.sqrt(sum(target_features[feat] ** 2 for feat in target_features.keys()))
        
        if source_magnitude == 0 or target_magnitude == 0:
            return 0.0
        
        cosine_similarity = dot_product / (source_magnitude * target_magnitude)
        
        # 应用权重调整
        weights = {
            'abstractness': 0.2,
            'complexity': 0.2,
            'predictability': 0.15,
            'structure_rigidity': 0.15,
            'semantic_density': 0.15,
            'functional_scope': 0.15
        }
        
        weighted_similarity = 0.0
        for feature in source_features.keys():
            feature_weight = weights.get(feature, 0.1)
            feature_similarity = 1.0 - abs(source_features[feature] - target_features[feature])
            weighted_similarity += feature_weight * feature_similarity
        
        # 综合相似性评分
        final_similarity = (cosine_similarity * 0.6 + weighted_similarity * 0.4)
        
        return max(0.0, min(1.0, final_similarity))
    
    async def _calculate_knowledge_retention(self, source_domain: str, 
                                           target_domain: str,
                                           knowledge_base: Dict[str, Any]) -> float:
        """计算知识保持率"""
        
        # 基于领域复杂度差异计算保持率
        if source_domain in self.domain_features and target_domain in self.domain_features:
            source_complexity = self.domain_features[source_domain]['complexity']
            target_complexity = self.domain_features[target_domain]['complexity']
            
            # 复杂度差异越大，保持率越低
            complexity_diff = abs(source_complexity - target_complexity)
            complexity_factor = 1.0 - (complexity_diff * 0.5)
        else:
            complexity_factor = 0.8  # 默认因子
        
        # 基于相似性计算保持率
        similarity = await self._calculate_domain_similarity(source_domain, target_domain)
        similarity_factor = similarity
        
        # 基于知识容量计算保持率
        knowledge_size = len(knowledge_base.get('concepts', []))
        size_factor = min(1.0, knowledge_size / 10.0) if knowledge_size > 0 else 0.5
        
        # 综合保持率
        retention_rate = (
            complexity_factor * 0.4 +
            similarity_factor * 0.5 +
            size_factor * 0.1
        )
        
        return max(0.0, min(1.0, retention_rate))
    
    async def _calculate_transfer_quality(self, source_domain: str,
                                        target_domain: str,
                                        knowledge_base: Dict[str, Any]) -> float:
        """计算迁移质量"""
        
        # 语义一致性评估
        semantic_consistency = await self._evaluate_semantic_consistency(
            source_domain, target_domain, knowledge_base
        )
        
        # 结构保持性评估
        structural_preservation = await self._evaluate_structural_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 功能完整性评估
        functional_completeness = await self._evaluate_functional_completeness(
            source_domain, target_domain, knowledge_base
        )
        
        # 错误容忍度评估
        error_tolerance = await self._evaluate_error_tolerance(
            source_domain, target_domain, knowledge_base
        )
        
        # 综合质量评分
        transfer_quality = (
            semantic_consistency * 0.3 +
            structural_preservation * 0.25 +
            functional_completeness * 0.3 +
            error_tolerance * 0.15
        )
        
        return max(0.0, min(1.0, transfer_quality))
    
    async def _evaluate_semantic_consistency(self, source_domain: str,
                                           target_domain: str,
                                           knowledge_base: Dict[str, Any]) -> float:
        """评估语义一致性"""
        
        # 模拟语义一致性评估
        concepts = knowledge_base.get('concepts', [])
        
        if not concepts:
            return 0.5  # 默认中等一致性
        
        # 基于概念映射的一致性评估
        consistency_scores = []
        for concept in concepts:
            # 模拟概念映射质量
            mapping_quality = np.random.uniform(0.6, 0.9)
            consistency_scores.append(mapping_quality)
        
        return np.mean(consistency_scores)
    
    async def _evaluate_structural_preservation(self, source_domain: str,
                                              target_domain: str,
                                              knowledge_base: Dict[str, Any]) -> float:
        """评估结构保持性"""
        
        # 基于关系映射的结构保持性评估
        relationships = knowledge_base.get('relationships', {})
        
        if not relationships:
            return 0.5  # 默认中等保持性
        
        # 计算关系保持率
        total_relations = sum(len(targets) for targets in relationships.values())
        preserved_relations = total_relations * 0.8  # 模拟80%保持率
        
        return preserved_relations / max(total_relations, 1)
    
    async def _evaluate_functional_completeness(self, source_domain: str,
                                              target_domain: str,
                                              knowledge_base: Dict[str, Any]) -> float:
        """评估功能完整性"""
        
        # 模拟功能完整性评估
        source_functions = knowledge_base.get('source_functions', [])
        adapted_functions = knowledge_base.get('adapted_functions', [])
        
        if not source_functions:
            return 0.5  # 默认中等完整性
        
        # 计算功能覆盖率
        coverage_rate = len(adapted_functions) / max(len(source_functions), 1)
        
        return min(1.0, coverage_rate)
    
    async def _evaluate_error_tolerance(self, source_domain: str,
                                      target_domain: str,
                                      knowledge_base: Dict[str, Any]) -> float:
        """评估错误容忍度"""
        
        # 基于领域特性评估错误容忍度
        target_complexity = self.domain_features.get(target_domain, {}).get('complexity', 0.5)
        
        # 复杂度越高，错误容忍度越低
        error_tolerance = 1.0 - (target_complexity * 0.3)
        
        return max(0.3, min(1.0, error_tolerance))
    
    async def _calculate_semantic_preservation(self, source_domain: str,
                                             target_domain: str,
                                             knowledge_base: Dict[str, Any]) -> float:
        """计算语义保真度"""
        
        # 概念语义保真度
        concept_preservation = await self._calculate_concept_semantic_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 关系语义保真度
        relation_preservation = await self._calculate_relation_semantic_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 上下文语义保真度
        context_preservation = await self._calculate_context_semantic_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 综合语义保真度
        semantic_preservation = (
            concept_preservation * 0.4 +
            relation_preservation * 0.35 +
            context_preservation * 0.25
        )
        
        return max(0.0, min(1.0, semantic_preservation))
    
    async def _calculate_concept_semantic_preservation(self, source_domain: str,
                                                     target_domain: str,
                                                     knowledge_base: Dict[str, Any]) -> float:
        """计算概念语义保真度"""
        
        concepts = knowledge_base.get('concepts', [])
        
        if not concepts:
            return 0.5  # 默认中等保真度
        
        # 模拟概念映射质量
        mapping_scores = []
        for concept in concepts:
            # 基于领域相似性调整映射质量
            similarity = await self._calculate_domain_similarity(source_domain, target_domain)
            base_quality = 0.7
            adjusted_quality = base_quality + (similarity - 0.5) * 0.4
            mapping_scores.append(max(0.3, min(1.0, adjusted_quality)))
        
        return np.mean(mapping_scores)
    
    async def _calculate_relation_semantic_preservation(self, source_domain: str,
                                                      target_domain: str,
                                                      knowledge_base: Dict[str, Any]) -> float:
        """计算关系语义保真度"""
        
        relationships = knowledge_base.get('relationships', {})
        
        if not relationships:
            return 0.5  # 默认中等保真度
        
        # 基于关系类型和映射质量计算保真度
        relation_scores = []
        
        for relation_type, targets in relationships.items():
            # 模拟关系映射质量
            type_preservation = 0.8  # 类型保持度
            target_mapping_quality = np.random.uniform(0.6, 0.9)  # 目标映射质量
            
            relation_score = type_preservation * target_mapping_quality
            relation_scores.append(relation_score)
        
        return np.mean(relation_scores) if relation_scores else 0.5
    
    async def _calculate_context_semantic_preservation(self, source_domain: str,
                                                     target_domain: str,
                                                     knowledge_base: Dict[str, Any]) -> float:
        """计算上下文语义保真度"""
        
        # 基于领域上下文特征计算保真度
        source_context = self.domain_features.get(source_domain, {})
        target_context = self.domain_features.get(target_domain, {})
        
        if not source_context or not target_context:
            return 0.5  # 默认中等保真度
        
        # 计算上下文特征相似性
        context_features = ['abstractness', 'semantic_density', 'functional_scope']
        similarities = []
        
        for feature in context_features:
            source_value = source_context.get(feature, 0.5)
            target_value = target_context.get(feature, 0.5)
            similarity = 1.0 - abs(source_value - target_value)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    async def _calculate_structural_integrity(self, source_domain: str,
                                            target_domain: str,
                                            knowledge_base: Dict[str, Any]) -> float:
        """计算结构完整性"""
        
        # 层次结构完整性
        hierarchy_integrity = await self._evaluate_hierarchy_integrity(
            source_domain, target_domain, knowledge_base
        )
        
        # 连接性完整性
        connectivity_integrity = await self._evaluate_connectivity_integrity(
            source_domain, target_domain, knowledge_base
        )
        
        # 约束完整性
        constraint_integrity = await self._evaluate_constraint_integrity(
            source_domain, target_domain, knowledge_base
        )
        
        # 综合结构完整性
        structural_integrity = (
            hierarchy_integrity * 0.4 +
            connectivity_integrity * 0.35 +
            constraint_integrity * 0.25
        )
        
        return max(0.0, min(1.0, structural_integrity))
    
    async def _evaluate_hierarchy_integrity(self, source_domain: str,
                                          target_domain: str,
                                          knowledge_base: Dict[str, Any]) -> float:
        """评估层次结构完整性"""
        
        # 模拟层次结构完整性评估
        hierarchy_levels = knowledge_base.get('hierarchy_levels', 3)
        preserved_levels = hierarchy_levels * 0.8  # 模拟80%层次保持
        
        return preserved_levels / max(hierarchy_levels, 1)
    
    async def _evaluate_connectivity_integrity(self, source_domain: str,
                                             target_domain: str,
                                             knowledge_base: Dict[str, Any]) -> float:
        """评估连接性完整性"""
        
        # 基于关系密度计算连接完整性
        relationships = knowledge_base.get('relationships', {})
        
        if not relationships:
            return 0.5  # 默认中等连接性
        
        total_connections = sum(len(targets) for targets in relationships.values())
        preserved_connections = total_connections * 0.75  # 模拟75%连接保持
        
        return preserved_connections / max(total_connections, 1)
    
    async def _evaluate_constraint_integrity(self, source_domain: str,
                                           target_domain: str,
                                           knowledge_base: Dict[str, Any]) -> float:
        """评估约束完整性"""
        
        # 基于规则保持度计算约束完整性
        rules = knowledge_base.get('rules', [])
        
        if not rules:
            return 0.5  # 默认中等约束完整性
        
        # 模拟约束保持度
        constraint_preservation = len(rules) * 0.7 / max(len(rules), 1)
        
        return constraint_preservation
    
    async def _calculate_functional_preservation(self, source_domain: str,
                                               target_domain: str,
                                               knowledge_base: Dict[str, Any]) -> float:
        """计算功能保真度"""
        
        # 核心功能保真度
        core_function_preservation = await self._evaluate_core_function_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 辅助功能保真度
        auxiliary_function_preservation = await self._evaluate_auxiliary_function_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 性能保真度
        performance_preservation = await self._evaluate_performance_preservation(
            source_domain, target_domain, knowledge_base
        )
        
        # 综合功能保真度
        functional_preservation = (
            core_function_preservation * 0.5 +
            auxiliary_function_preservation * 0.3 +
            performance_preservation * 0.2
        )
        
        return max(0.0, min(1.0, functional_preservation))
    
    async def _evaluate_core_function_preservation(self, source_domain: str,
                                                 target_domain: str,
                                                 knowledge_base: Dict[str, Any]) -> float:
        """评估核心功能保真度"""
        
        # 模拟核心功能评估
        core_functions = knowledge_base.get('core_functions', [])
        
        if not core_functions:
            return 0.5  # 默认中等保真度
        
        # 基于领域特性调整核心功能保真度
        domain_similarity = await self._calculate_domain_similarity(source_domain, target_domain)
        
        base_preservation = 0.8
        similarity_adjustment = (domain_similarity - 0.5) * 0.3
        
        return max(0.4, min(1.0, base_preservation + similarity_adjustment))
    
    async def _evaluate_auxiliary_function_preservation(self, source_domain: str,
                                                      target_domain: str,
                                                      knowledge_base: Dict[str, Any]) -> float:
        """评估辅助功能保真度"""
        
        # 辅助功能通常比核心功能更容易丢失
        auxiliary_functions = knowledge_base.get('auxiliary_functions', [])
        
        if not auxiliary_functions:
            return 0.5  # 默认中等保真度
        
        # 模拟较低的辅助功能保真度
        return len(auxiliary_functions) * 0.6 / max(len(auxiliary_functions), 1)
    
    async def _evaluate_performance_preservation(self, source_domain: str,
                                               target_domain: str,
                                               knowledge_base: Dict[str, Any]) -> float:
        """评估性能保真度"""
        
        # 基于性能指标计算保真度
        performance_metrics = knowledge_base.get('performance_metrics', {})
        
        if not performance_metrics:
            return 0.5  # 默认中等保真度
        
        # 模拟性能指标
        metrics = ['accuracy', 'speed', 'efficiency', 'reliability']
        preservation_scores = []
        
        for metric in metrics:
            # 基于领域相似性调整性能保真度
            domain_similarity = await self._calculate_domain_similarity(source_domain, target_domain)
            base_performance = 0.75
            similarity_adjustment = (domain_similarity - 0.5) * 0.2
            
            score = max(0.3, min(1.0, base_performance + similarity_adjustment))
            preservation_scores.append(score)
        
        return np.mean(preservation_scores)
    
    async def _calculate_transfer_efficiency(self, similarity_score: float,
                                           retention_rate: float,
                                           quality_score: float,
                                           semantic_score: float,
                                           structural_score: float,
                                           functional_score: float) -> float:
        """计算综合迁移效率"""
        
        # 应用权重
        weights = {
            'similarity': self.transfer_params['similarity_weight'],
            'retention': 0.2,
            'quality': self.transfer_params['quality_weight'],
            'semantic': 0.15,
            'structural': 0.15,
            'functional': 0.1
        }
        
        # 计算加权效率
        efficiency = (
            similarity_score * weights['similarity'] +
            retention_rate * weights['retention'] +
            quality_score * weights['quality'] +
            semantic_score * weights['semantic'] +
            structural_score * weights['structural'] +
            functional_score * weights['functional']
        )
        
        # 应用阈值修正
        if efficiency < self.transfer_params['efficiency_threshold']:
            efficiency *= 0.9  # 低于阈值的效率进一步惩罚
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_optimization_potential(self, current_efficiency: float) -> float:
        """计算优化潜力"""
        
        # 效率越低，优化潜力越大
        base_potential = 1.0 - current_efficiency
        
        # 基于效率范围调整潜力
        if current_efficiency < 0.3:
            potential_multiplier = 1.5  # 低效率有较大优化空间
        elif current_efficiency < 0.6:
            potential_multiplier = 1.2  # 中等效率有中等优化空间
        else:
            potential_multiplier = 0.8  # 高效率优化空间有限
        
        return min(1.0, base_potential * potential_multiplier)
    
    async def _generate_optimization_suggestions(self, source_domain: str,
                                               target_domain: str,
                                               metrics: TransferMetrics) -> List[str]:
        """生成优化建议"""
        
        suggestions = []
        
        # 基于迁移效率的建议
        if metrics.transfer_efficiency < 0.6:
            suggestions.append("考虑选择相似性更高的中间领域进行分步迁移")
            suggestions.append("增加源领域的预训练样本数量")
        
        # 基于知识保持率的建议
        if metrics.knowledge_retention < 0.7:
            suggestions.append("优化知识表示方法以提高保持率")
            suggestions.append("使用增量学习策略减少知识遗忘")
        
        # 基于语义保真度的建议
        if metrics.semantic_preservation < 0.6:
            suggestions.append("增强语义映射算法的精度")
            suggestions.append("引入领域特定的语义词典")
        
        # 基于结构完整性的建议
        if metrics.structural_integrity < 0.6:
            suggestions.append("设计更稳定的结构转换机制")
            suggestions.append("增加结构约束验证步骤")
        
        # 基于功能保真度的建议
        if metrics.functional_preservation < 0.6:
            suggestions.append("优化功能映射算法")
            suggestions.append("增加功能验证测试")
        
        # 通用优化建议
        if metrics.optimization_potential > 0.7:
            suggestions.append("系统性地重新设计迁移策略")
            suggestions.append("考虑使用深度迁移学习方法")
        
        return suggestions
    
    async def analyze_transfer_bottlenecks(self, transfer_metrics: TransferMetrics) -> Dict[str, Any]:
        """分析迁移瓶颈"""
        
        bottlenecks = {
            'identified_bottlenecks': [],
            'bottleneck_severity': {},
            'resolution_strategies': [],
            'priority_areas': []
        }
        
        # 识别瓶颈
        metric_scores = {
            'transfer_efficiency': transfer_metrics.transfer_efficiency,
            'knowledge_retention': transfer_metrics.knowledge_retention,
            'transfer_quality': transfer_metrics.transfer_quality,
            'semantic_preservation': transfer_metrics.semantic_preservation,
            'structural_integrity': transfer_metrics.structural_integrity,
            'functional_preservation': transfer_metrics.functional_preservation
        }
        
        for metric, score in metric_scores.items():
            if score < 0.5:
                bottlenecks['identified_bottlenecks'].append(metric)
                bottlenecks['bottleneck_severity'][metric] = 1.0 - score
                
                # 生成解决策略
                if metric == 'transfer_efficiency':
                    bottlenecks['resolution_strategies'].extend([
                        "提高领域相似性计算精度",
                        "优化迁移算法参数",
                        "增加中间领域辅助"
                    ])
                elif metric == 'knowledge_retention':
                    bottlenecks['resolution_strategies'].extend([
                        "改进知识编码方式",
                        "使用记忆增强机制",
                        "优化学习率调度"
                    ])
                elif metric == 'semantic_preservation':
                    bottlenecks['resolution_strategies'].extend([
                        "增强语义向量表示",
                        "引入领域特定的语义模型",
                        "使用上下文感知的映射"
                    ])
                # 可以继续添加其他指标的处理策略
        
        # 确定优先级领域
        sorted_bottlenecks = sorted(
            bottlenecks['bottleneck_severity'].items(),
            key=lambda x: x[1], reverse=True
        )
        
        bottlenecks['priority_areas'] = [
            area for area, severity in sorted_bottlenecks[:3]
        ]
        
        return bottlenecks
    
    async def plan_transfer_path(self, source_domains: List[str],
                               target_domain: str,
                               max_hops: int = None) -> List[TransferPath]:
        """规划迁移路径"""
        
        max_hops = max_hops or self.transfer_params['max_transfer_hops']
        all_domains = list(self.domain_features.keys())
        
        # 生成可能的迁移路径
        paths = []
        
        for hop_count in range(1, max_hops + 1):
            # 简化的路径生成算法
            path_sequence = [source_domains[0]] + [target_domain]
            
            # 计算路径质量
            efficiency_scores = []
            for i in range(len(path_sequence) - 1):
                efficiency = await self._calculate_domain_similarity(
                    path_sequence[i], path_sequence[i + 1]
                )
                efficiency_scores.append(efficiency)
            
            total_efficiency = np.mean(efficiency_scores)
            quality_score = total_efficiency * 0.8  # 质量评分为效率的80%
            
            # 计算估计时间
            estimated_time = hop_count * 2.0  # 每跳估计2个时间单位
            
            # 计算复杂度
            complexity_score = hop_count / max_hops
            
            path = TransferPath(
                path_id=f"path_{len(paths)}",
                sequence=path_sequence,
                total_efficiency=total_efficiency,
                quality_score=quality_score,
                estimated_time=estimated_time,
                complexity_score=complexity_score,
                confidence_level=min(0.9, total_efficiency + 0.1)
            )
            
            paths.append(path)
        
        # 按效率排序
        paths.sort(key=lambda x: x.total_efficiency, reverse=True)
        
        return paths
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """获取迁移统计信息"""
        
        if not self.transfer_history:
            return {
                'total_transfers': 0,
                'average_efficiency': 0.0,
                'success_rate': 0.0,
                'common_bottlenecks': []
            }
        
        # 计算统计指标
        total_transfers = len(self.transfer_history)
        
        efficiencies = [metrics.transfer_efficiency for metrics in self.transfer_history]
        average_efficiency = np.mean(efficiencies)
        
        successful_transfers = len([e for e in efficiencies if e > 0.7])
        success_rate = successful_transfers / total_transfers
        
        # 识别常见瓶颈
        all_bottlenecks = []
        for metrics in self.transfer_history:
            if metrics.transfer_efficiency < 0.6:
                all_bottlenecks.append(metrics.target_domain)
        
        from collections import Counter
        bottleneck_counts = Counter(all_bottlenecks)
        common_bottlenecks = [domain for domain, count in bottleneck_counts.most_common(3)]
        
        return {
            'total_transfers': total_transfers,
            'average_efficiency': average_efficiency,
            'success_rate': success_rate,
            'common_bottlenecks': common_bottlenecks,
            'efficiency_distribution': {
                'high': len([e for e in efficiencies if e >= 0.8]),
                'medium': len([e for e in efficiencies if 0.5 <= e < 0.8]),
                'low': len([e for e in efficiencies if e < 0.5])
            }
        }
    
    def record_transfer_metrics(self, metrics: TransferMetrics) -> None:
        """记录迁移指标到历史"""
        self.transfer_history.append(metrics)
        self.logger.info(f"已记录迁移指标: {metrics.source_domain} -> {metrics.target_domain}")


def create_transfer_analyzer(config: Optional[Dict[str, Any]] = None) -> TransferAnalyzer:
    """创建迁移分析器实例的便捷函数"""
    return TransferAnalyzer(config or {})


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建迁移分析器
        analyzer = create_transfer_analyzer({
            'efficiency_threshold': 0.7,
            'quality_threshold': 0.6,
            'max_transfer_hops': 3
        })
        
        # 测量迁移效率
        result = await analyzer.measure_transfer_efficiency(
            source_domains=['game'],
            target_domains=['physics', 'social'],
            knowledge_base={'concepts': ['strategy', 'movement'], 'relationships': {'has': ['rule']}}
        )
        
        print(f"迁移效率分析完成: {result}")
    
    # 运行示例
    # asyncio.run(main())