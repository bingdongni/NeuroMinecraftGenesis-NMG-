"""
符号知识提取模块

该模块实现了从神经网络激活中提取符号知识的功能。
通过分析神经网络的激活模式、权重结构和特征表示，
提取出可解释的符号规则、概念和关系。

作者: NeuroMinecraftGenesis Team
日期: 2025-11-13
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict, Counter
import json
import time
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class SymbolExtraction:
    """
    符号知识提取器类
    
    该类专门负责从神经网络中提取符号知识，包括：
    1. 从激活模式中提取概念
    2. 从权重结构中提取关系
    3. 生成符号规则
    4. 构建知识图谱
    5. 知识质量评估
    """
    
    def __init__(self, 
                 network_config: Dict[str, Any],
                 symbolic_config: Dict[str, Any]):
        """
        初始化符号提取器
        
        Args:
            network_config: 神经网络配置
            symbolic_config: 符号推理配置
        """
        self.network_config = network_config
        self.symbolic_config = symbolic_config
        
        # 设置日志
        self.logger = logging.getLogger("SymbolExtraction")
        
        # 提取参数配置
        self.extraction_config = {
            "activation_threshold": symbolic_config.get("activation_threshold", 0.5),
            "concept_clustering_method": symbolic_config.get("concept_clustering", "dbscan"),
            "relation_threshold": symbolic_config.get("relation_threshold", 0.7),
            "rule_complexity_limit": symbolic_config.get("rule_complexity_limit", 5),
            "knowledge_confidence_threshold": symbolic_config.get("knowledge_confidence_threshold", 0.6),
            "temporal_smoothing": symbolic_config.get("temporal_smoothing", True)
        }
        
        # 知识存储
        self.knowledge_base = {}
        self.extracted_concepts = {}
        self.extracted_relations = {}
        self.extracted_rules = {}
        
        # 提取状态跟踪
        self.extraction_history = []
        self.concept_activation_patterns = {}
        self.relation_cooccurrence_matrix = {}
        
        # 性能统计
        self.extraction_stats = {
            "total_extractions": 0,
            "concepts_extracted": 0,
            "relations_extracted": 0,
            "rules_generated": 0,
            "average_extraction_time": 0.0,
            "knowledge_quality_scores": []
        }
        
        self.logger.info("符号知识提取器初始化完成")
    
    def set_knowledge_base(self, knowledge_base: Dict[str, Any]) -> None:
        """
        设置知识库
        
        Args:
            knowledge_base: 知识库数据
        """
        self.knowledge_base = knowledge_base.copy()
        self.logger.info("知识库设置完成")
    
    def extract_symbolic_knowledge(self, 
                                 symbolic_representation: Dict[str, Any],
                                 neural_activations: torch.Tensor,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        从符号表示和神经激活中提取符号知识
        
        Args:
            symbolic_representation: 符号表示
            neural_activations: 神经网络激活
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 提取的符号知识
        """
        try:
            start_time = time.time()
            
            # 1. 提取概念
            concepts = self._extract_concepts(neural_activations, symbolic_representation, context)
            
            # 2. 提取关系
            relations = self._extract_relations(concepts, neural_activations, symbolic_representation)
            
            # 3. 生成规则
            rules = self._generate_rules(concepts, relations, symbolic_representation)
            
            # 4. 构建知识图谱
            knowledge_graph = self._build_knowledge_graph(concepts, relations, rules)
            
            # 5. 评估知识质量
            quality_assessment = self._assess_knowledge_quality(concepts, relations, rules)
            
            # 构建提取结果
            extraction_result = {
                "concepts": concepts,
                "relations": relations,
                "rules": rules,
                "knowledge_graph": knowledge_graph,
                "quality_assessment": quality_assessment,
                "extraction_metadata": {
                    "timestamp": time.time(),
                    "extraction_time": time.time() - start_time,
                    "context": context,
                    "neural_activation_shape": list(neural_activations.shape)
                }
            }
            
            # 更新历史记录
            self._update_extraction_history(extraction_result)
            
            # 更新统计信息
            self._update_extraction_stats(extraction_result, time.time() - start_time)
            
            self.logger.info(f"符号知识提取完成，耗时: {time.time() - start_time:.2f}秒")
            self.logger.info(f"提取概念: {len(concepts)}, 关系: {len(relations)}, 规则: {len(rules)}")
            
            return extraction_result
            
        except Exception as e:
            self.logger.error(f"符号知识提取失败: {str(e)}")
            return {"error": str(e)}
    
    def _extract_concepts(self, 
                        neural_activations: torch.Tensor,
                        symbolic_representation: Dict[str, Any],
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        从神经激活中提取概念
        
        Args:
            neural_activations: 神经激活
            symbolic_representation: 符号表示
            context: 上下文
            
        Returns:
            Dict[str, Any]: 提取的概念
        """
        try:
            concepts = {}
            
            # 1. 识别激活模式
            activation_patterns = self._identify_activation_patterns(neural_activations)
            
            # 2. 聚类相似激活模式
            clustered_patterns = self._cluster_activation_patterns(activation_patterns)
            
            # 3. 将聚类转换为概念
            for cluster_id, pattern_indices in clustered_patterns.items():
                concept_info = self._cluster_to_concept(cluster_id, pattern_indices, neural_activations)
                if concept_info["confidence"] > self.extraction_config["knowledge_confidence_threshold"]:
                    concepts[concept_info["name"]] = concept_info
            
            # 4. 基于符号表示增强概念
            enhanced_concepts = self._enhance_concepts_with_symbolic_info(
                concepts, symbolic_representation
            )
            
            self.extraction_stats["concepts_extracted"] += len(enhanced_concepts)
            
            return enhanced_concepts
            
        except Exception as e:
            self.logger.error(f"概念提取失败: {str(e)}")
            return {}
    
    def _identify_activation_patterns(self, 
                                    activations: torch.Tensor) -> Dict[str, Any]:
        """
        识别神经激活模式
        
        Args:
            activations: 神经激活张量
            
        Returns:
            Dict[str, Any]: 激活模式
        """
        try:
            # 转换为numpy数组
            if isinstance(activations, torch.Tensor):
                activations_np = activations.detach().cpu().numpy()
            else:
                activations_np = np.array(activations)
            
            patterns = {
                "activated_neurons": [],
                "activation_groups": [],
                "sparsity_pattern": None,
                "activation_distribution": {}
            }
            
            # 识别激活的神经元
            threshold = self.extraction_config["activation_threshold"]
            activated_mask = activations_np > threshold
            activated_indices = np.where(activated_mask)[0]
            patterns["activated_neurons"] = activated_indices.tolist()
            
            # 分析激活强度分布
            activation_strengths = activations_np[activated_mask]
            patterns["activation_distribution"] = {
                "mean": float(np.mean(activation_strengths)),
                "std": float(np.std(activation_strengths)),
                "max": float(np.max(activation_strengths)),
                "min": float(np.min(activation_strengths))
            }
            
            # 识别激活组（连续的激活区域）
            if len(activated_indices) > 0:
                groups = self._identify_activation_groups(activated_indices)
                patterns["activation_groups"] = groups
            
            # 分析稀疏性模式
            patterns["sparsity_pattern"] = self._analyze_sparsity_pattern(activations_np)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"激活模式识别失败: {str(e)}")
            return {}
    
    def _identify_activation_groups(self, activated_indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        识别激活组
        
        Args:
            activated_indices: 激活的神经元索引
            
        Returns:
            List[Dict[str, Any]]: 激活组列表
        """
        groups = []
        
        if len(activated_indices) == 0:
            return groups
        
        # 找到连续的激活区域
        current_group = [activated_indices[0]]
        
        for i in range(1, len(activated_indices)):
            if activated_indices[i] - activated_indices[i-1] <= 2:  # 允许小间隔
                current_group.append(activated_indices[i])
            else:
                groups.append({
                    "indices": current_group.copy(),
                    "size": len(current_group),
                    "range": [current_group[0], current_group[-1]]
                })
                current_group = [activated_indices[i]]
        
        # 添加最后一个组
        if current_group:
            groups.append({
                "indices": current_group.copy(),
                "size": len(current_group),
                "range": [current_group[0], current_group[-1]]
            })
        
        return groups
    
    def _analyze_sparsity_pattern(self, activations: np.ndarray) -> Dict[str, Any]:
        """
        分析稀疏性模式
        
        Args:
            activations: 激活值数组
            
        Returns:
            Dict[str, Any]: 稀疏性分析结果
        """
        threshold = self.extraction_config["activation_threshold"]
        inactive_neurons = np.sum(activations <= threshold)
        total_neurons = len(activations)
        sparsity_ratio = inactive_neurons / total_neurons
        
        return {
            "sparsity_ratio": float(sparsity_ratio),
            "active_neurons": int(total_neurons - inactive_neurons),
            "inactive_neurons": int(inactive_neurons),
            "sparsity_type": "high" if sparsity_ratio > 0.8 else "medium" if sparsity_ratio > 0.5 else "low"
        }
    
    def _cluster_activation_patterns(self, activation_patterns: Dict[str, Any]) -> Dict[int, List[int]]:
        """
        聚类激活模式
        
        Args:
            activation_patterns: 激活模式
            
        Returns:
            Dict[int, List[int]]: 聚类结果
        """
        try:
            activated_neurons = activation_patterns["activated_neurons"]
            
            if len(activated_neurons) < 2:
                return {0: activated_neurons}
            
            # 使用DBSCAN进行聚类
            if self.extraction_config["concept_clustering_method"] == "dbscan":
                # 将激活神经元索引转换为特征向量
                features = self._neuron_indices_to_features(activated_neurons)
                
                clustering = DBSCAN(eps=3, min_samples=2)
                cluster_labels = clustering.fit_predict(features)
                
                clusters = defaultdict(list)
                for neuron_idx, cluster_label in zip(activated_neurons, cluster_labels):
                    clusters[cluster_label].append(neuron_idx)
                
                # 过滤噪声点（label=-1）
                filtered_clusters = {
                    k: v for k, v in clusters.items() 
                    if k != -1 and len(v) >= 1
                }
                
                return dict(filtered_clusters)
            
            elif self.extraction_config["concept_clustering_method"] == "kmeans":
                # 使用K-means聚类
                features = self._neuron_indices_to_features(activated_neurons)
                n_clusters = min(len(activated_neurons) // 3 + 1, 5)  # 自适应聚类数
                
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(features)
                
                clusters = defaultdict(list)
                for neuron_idx, cluster_label in zip(activated_neurons, cluster_labels):
                    clusters[cluster_label].append(neuron_idx)
                
                return dict(clusters)
            
            else:
                # 简单分组策略
                group_size = max(1, len(activated_neurons) // 3)
                groups = {}
                for i in range(0, len(activated_neurons), group_size):
                    group_indices = activated_neurons[i:i+group_size]
                    groups[i//group_size] = group_indices
                
                return groups
                
        except Exception as e:
            self.logger.error(f"激活模式聚类失败: {str(e)}")
            return {0: activation_patterns["activated_neurons"]}
    
    def _neuron_indices_to_features(self, neuron_indices: List[int]) -> np.ndarray:
        """
        将神经元索引转换为特征向量
        
        Args:
            neuron_indices: 神经元索引列表
            
        Returns:
            np.ndarray: 特征矩阵
        """
        # 创建基于位置的编码特征
        features = np.array(neuron_indices).reshape(-1, 1)
        return features
    
    def _cluster_to_concept(self, 
                          cluster_id: int, 
                          pattern_indices: List[int],
                          activations: torch.Tensor) -> Dict[str, Any]:
        """
        将激活聚类转换为概念
        
        Args:
            cluster_id: 聚类ID
            pattern_indices: 模式索引
            activations: 激活值
            
        Returns:
            Dict[str, Any]: 概念信息
        """
        try:
            # 计算激活强度统计
            if isinstance(activations, torch.Tensor):
                activation_values = activations[pattern_indices].detach().cpu().numpy()
            else:
                activation_values = np.array(activations)[pattern_indices]
            
            # 构建概念信息
            concept_info = {
                "id": f"concept_{cluster_id}",
                "name": f"Concept_{cluster_id}",
                "neuron_indices": pattern_indices,
                "activation_stats": {
                    "mean": float(np.mean(activation_values)),
                    "std": float(np.std(activation_values)),
                    "max": float(np.max(activation_values)),
                    "min": float(np.min(activation_values))
                },
                "confidence": float(np.mean(activation_values)),
                "cluster_size": len(pattern_indices),
                "extraction_method": "activation_clustering"
            }
            
            # 计算概念特征向量
            concept_info["feature_vector"] = self._compute_concept_feature_vector(
                pattern_indices, activations
            )
            
            return concept_info
            
        except Exception as e:
            self.logger.error(f"聚类转概念失败: {str(e)}")
            return {
                "id": f"concept_{cluster_id}",
                "name": f"Concept_{cluster_id}",
                "confidence": 0.0
            }
    
    def _compute_concept_feature_vector(self, 
                                      pattern_indices: List[int],
                                      activations: torch.Tensor) -> List[float]:
        """
        计算概念特征向量
        
        Args:
            pattern_indices: 模式索引
            activations: 激活值
            
        Returns:
            List[float]: 特征向量
        """
        try:
            if isinstance(activations, torch.Tensor):
                activations_np = activations.detach().cpu().numpy()
            else:
                activations_np = np.array(activations)
            
            # 计算统计特征
            selected_activations = activations_np[pattern_indices]
            
            features = [
                float(np.mean(selected_activations)),
                float(np.std(selected_activations)),
                float(np.max(selected_activations)),
                float(np.min(selected_activations)),
                float(len(pattern_indices) / len(activations_np))  # 激活比例
            ]
            
            return features
            
        except Exception as e:
            self.logger.error(f"概念特征向量计算失败: {str(e)}")
            return [0.0] * 5
    
    def _enhance_concepts_with_symbolic_info(self, 
                                           concepts: Dict[str, Any],
                                           symbolic_representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用符号表示增强概念
        
        Args:
            concepts: 基础概念
            symbolic_representation: 符号表示
            
        Returns:
            Dict[str, Any]: 增强后的概念
        """
        try:
            enhanced_concepts = concepts.copy()
            
            # 检查符号表示中的概念信息
            if "concept_hierarchy" in symbolic_representation:
                hierarchy = symbolic_representation["concept_hierarchy"]
                active_symbolic_concepts = symbolic_representation.get("active_concepts", [])
                
                # 为每个提取的概念匹配符号概念
                for concept_id, concept_info in enhanced_concepts.items():
                    # 找到最相似的符号概念
                    best_match = self._find_best_symbolic_match(
                        concept_info, active_symbolic_concepts
                    )
                    
                    if best_match:
                        # 增强概念信息
                        concept_info["symbolic_link"] = best_match["concept"]
                        concept_info["symbolic_confidence"] = best_match["confidence"]
                        concept_info["enhanced_attributes"] = best_match.get("attributes", {})
            
            return enhanced_concepts
            
        except Exception as e:
            self.logger.error(f"概念增强失败: {str(e)}")
            return concepts
    
    def _find_best_symbolic_match(self, 
                                concept_info: Dict[str, Any],
                                symbolic_concepts: List[str]) -> Optional[Dict[str, Any]]:
        """
        找到最佳符号概念匹配
        
        Args:
            concept_info: 概念信息
            symbolic_concepts: 符号概念列表
            
        Returns:
            Optional[Dict[str, Any]]: 最佳匹配
        """
        try:
            if not symbolic_concepts:
                return None
            
            # 基于置信度选择最佳匹配
            concept_confidence = concept_info.get("confidence", 0.0)
            
            # 简化的匹配策略：选择第一个活跃符号概念
            best_symbolic = symbolic_concepts[0]
            
            return {
                "concept": best_symbolic,
                "confidence": min(concept_confidence, 0.8),  # 限制最大置信度
                "attributes": {}  # 可以从知识库中获取
            }
            
        except Exception as e:
            self.logger.error(f"符号概念匹配失败: {str(e)}")
            return None
    
    def _extract_relations(self, 
                         concepts: Dict[str, Any],
                         neural_activations: torch.Tensor,
                         symbolic_representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取概念间关系
        
        Args:
            concepts: 概念
            neural_activations: 神经激活
            symbolic_representation: 符号表示
            
        Returns:
            Dict[str, Any]: 提取的关系
        """
        try:
            relations = {}
            
            # 1. 从符号表示中提取关系
            if "relational_network" in symbolic_representation:
                network_relations = self._extract_from_relational_network(
                    symbolic_representation["relational_network"]
                )
                relations.update(network_relations)
            
            # 2. 基于概念共现分析关系
            cooccurrence_relations = self._analyze_concept_cooccurrence(
                concepts, neural_activations
            )
            relations.update(cooccurrence_relations)
            
            # 3. 基于神经激活相关性分析关系
            correlation_relations = self._analyze_activation_correlation(
                concepts, neural_activations
            )
            relations.update(correlation_relations)
            
            self.extraction_stats["relations_extracted"] += len(relations)
            
            return relations
            
        except Exception as e:
            self.logger.error(f"关系提取失败: {str(e)}")
            return {}
    
    def _extract_from_relational_network(self, 
                                       network: Dict[str, Any]) -> Dict[str, Any]:
        """
        从关系网络中提取关系
        
        Args:
            network: 关系网络
            
        Returns:
            Dict[str, Any]: 关系字典
        """
        relations = {}
        
        try:
            edges = network.get("edges", [])
            
            for edge in edges:
                relation_id = f"rel_{edge['source']}_{edge['target']}"
                relations[relation_id] = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "strength": edge.get("strength", 0.5),
                    "type": "symbolic_relation",
                    "confidence": edge.get("strength", 0.5)
                }
            
            return relations
            
        except Exception as e:
            self.logger.error(f"从关系网络提取失败: {str(e)}")
            return {}
    
    def _analyze_concept_cooccurrence(self, 
                                    concepts: Dict[str, Any],
                                    neural_activations: torch.Tensor) -> Dict[str, Any]:
        """
        分析概念共现关系
        
        Args:
            concepts: 概念
            neural_activations: 神经激活
            
        Returns:
            Dict[str, Any]: 共现关系
        """
        relations = {}
        
        try:
            concept_ids = list(concepts.keys())
            
            # 计算概念间的共现强度
            for i, concept1_id in enumerate(concept_ids):
                for concept2_id in concept_ids[i+1:]:
                    cooccurrence_strength = self._calculate_concept_cooccurrence(
                        concepts[concept1_id], concepts[concept2_id]
                    )
                    
                    if cooccurrence_strength > self.extraction_config["relation_threshold"]:
                        relation_id = f"cooc_{concept1_id}_{concept2_id}"
                        relations[relation_id] = {
                            "source": concept1_id,
                            "target": concept2_id,
                            "strength": cooccurrence_strength,
                            "type": "cooccurrence",
                            "confidence": cooccurrence_strength
                        }
            
            return relations
            
        except Exception as e:
            self.logger.error(f"概念共现分析失败: {str(e)}")
            return {}
    
    def _calculate_concept_cooccurrence(self, 
                                      concept1: Dict[str, Any],
                                      concept2: Dict[str, Any]) -> float:
        """
        计算概念共现强度
        
        Args:
            concept1: 概念1
            concept2: 概念2
            
        Returns:
            float: 共现强度
        """
        try:
            # 计算神经元索引的重叠
            indices1 = set(concept1.get("neuron_indices", []))
            indices2 = set(concept2.get("neuron_indices", []))
            
            if not indices1 or not indices2:
                return 0.0
            
            overlap = len(indices1.intersection(indices2))
            union = len(indices1.union(indices2))
            
            # Jaccard相似度
            jaccard_similarity = overlap / union if union > 0 else 0.0
            
            # 结合置信度
            confidence1 = concept1.get("confidence", 0.0)
            confidence2 = concept2.get("confidence", 0.0)
            avg_confidence = (confidence1 + confidence2) / 2.0
            
            # 综合共现强度
            cooccurrence_strength = jaccard_similarity * avg_confidence
            
            return cooccurrence_strength
            
        except Exception as e:
            self.logger.error(f"概念共现强度计算失败: {str(e)}")
            return 0.0
    
    def _analyze_activation_correlation(self, 
                                      concepts: Dict[str, Any],
                                      neural_activations: torch.Tensor) -> Dict[str, Any]:
        """
        分析激活相关性关系
        
        Args:
            concepts: 概念
            neural_activations: 神经激活
            
        Returns:
            Dict[str, Any]: 相关性关系
        """
        relations = {}
        
        try:
            if not isinstance(neural_activations, torch.Tensor):
                neural_activations = torch.tensor(neural_activations)
            
            # 为每个概念构建激活向量
            concept_activation_vectors = {}
            for concept_id, concept_info in concepts.items():
                neuron_indices = concept_info.get("neuron_indices", [])
                if neuron_indices:
                    activation_vector = neural_activations[neuron_indices]
                    concept_activation_vectors[concept_id] = activation_vector
            
            # 计算概念间的激活相关性
            concept_ids = list(concept_activation_vectors.keys())
            
            for i, concept1_id in enumerate(concept_ids):
                for concept2_id in concept_ids[i+1:]:
                    correlation = torch.nn.functional.cosine_similarity(
                        concept_activation_vectors[concept1_id].unsqueeze(0),
                        concept_activation_vectors[concept2_id].unsqueeze(0)
                    ).item()
                    
                    if abs(correlation) > self.extraction_config["relation_threshold"]:
                        relation_id = f"corr_{concept1_id}_{concept2_id}"
                        relations[relation_id] = {
                            "source": concept1_id,
                            "target": concept2_id,
                            "strength": abs(correlation),
                            "type": "activation_correlation",
                            "correlation_value": correlation,
                            "confidence": abs(correlation)
                        }
            
            return relations
            
        except Exception as e:
            self.logger.error(f"激活相关性分析失败: {str(e)}")
            return {}
    
    def _generate_rules(self, 
                      concepts: Dict[str, Any],
                      relations: Dict[str, Any],
                      symbolic_representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成符号规则
        
        Args:
            concepts: 概念
            relations: 关系
            symbolic_representation: 符号表示
            
        Returns:
            Dict[str, Any]: 生成的规则
        """
        try:
            rules = {}
            
            # 1. 从符号表示中提取规则
            if "symbolic_formulas" in symbolic_representation:
                formula_rules = self._extract_rules_from_formulas(
                    symbolic_representation["symbolic_formulas"]
                )
                rules.update(formula_rules)
            
            # 2. 基于关系生成规则
            relation_rules = self._generate_rules_from_relations(relations)
            rules.update(relation_rules)
            
            # 3. 生成概念激活规则
            activation_rules = self._generate_activation_rules(concepts, relations)
            rules.update(activation_rules)
            
            # 4. 生成组合规则
            combination_rules = self._generate_combination_rules(concepts, relations)
            rules.update(combination_rules)
            
            self.extraction_stats["rules_generated"] += len(rules)
            
            return rules
            
        except Exception as e:
            self.logger.error(f"规则生成失败: {str(e)}")
            return {}
    
    def _extract_rules_from_formulas(self, 
                                   formulas: List[str]) -> Dict[str, Any]:
        """
        从符号公式中提取规则
        
        Args:
            formulas: 符号公式列表
            
        Returns:
            Dict[str, Any]: 提取的规则
        """
        rules = {}
        
        try:
            for i, formula in enumerate(formulas):
                rule_id = f"formula_rule_{i}"
                rules[rule_id] = {
                    "premise": formula,
                    "conclusion": f"Derived from formula: {formula}",
                    "type": "formula_based",
                    "confidence": 0.7,  # 默认置信度
                    "complexity": self._calculate_rule_complexity(formula)
                }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"公式规则提取失败: {str(e)}")
            return {}
    
    def _generate_rules_from_relations(self, 
                                     relations: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于关系生成规则
        
        Args:
            relations: 关系
            
        Returns:
            Dict[str, Any]: 基于关系的规则
        """
        rules = {}
        
        try:
            for relation_id, relation_info in relations.items():
                source = relation_info.get("source")
                target = relation_info.get("target")
                strength = relation_info.get("strength", 0.0)
                
                if source and target and strength > 0.5:
                    rule_id = f"rel_rule_{relation_id}"
                    rules[rule_id] = {
                        "premise": f"IF {source} AND {relation_info.get('type', 'related_to')}",
                        "conclusion": f"THEN {target} (strength: {strength:.2f})",
                        "type": "relation_based",
                        "strength": strength,
                        "confidence": min(strength, 0.9),
                        "complexity": 2
                    }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"关系规则生成失败: {str(e)}")
            return {}
    
    def _generate_activation_rules(self, 
                                 concepts: Dict[str, Any],
                                 relations: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成概念激活规则
        
        Args:
            concepts: 概念
            relations: 关系
            
        Returns:
            Dict[str, Any]: 激活规则
        """
        rules = {}
        
        try:
            for concept_id, concept_info in concepts.items():
                confidence = concept_info.get("confidence", 0.0)
                
                if confidence > 0.6:
                    rule_id = f"activation_rule_{concept_id}"
                    rules[rule_id] = {
                        "premise": f"Activation level above threshold in neurons {concept_info.get('neuron_indices', [])}",
                        "conclusion": f"Concept {concept_id} is active (confidence: {confidence:.2f})",
                        "type": "activation_based",
                        "confidence": confidence,
                        "complexity": 1
                    }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"激活规则生成失败: {str(e)}")
            return {}
    
    def _generate_combination_rules(self, 
                                  concepts: Dict[str, Any],
                                  relations: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成组合规则
        
        Args:
            concepts: 概念
            relations: 关系
            
        Returns:
            Dict[str, Any]: 组合规则
        """
        rules = {}
        
        try:
            # 找到有强关系的概念对
            strong_relations = {
                rid: rinfo for rid, rinfo in relations.items()
                if rinfo.get("strength", 0.0) > 0.7
            }
            
            if len(strong_relations) >= 2:
                # 生成多概念组合规则
                relation_list = list(strong_relations.values())
                premises = []
                conclusions = []
                
                for relation in relation_list[:3]:  # 最多3个关系
                    premises.append(relation["source"])
                    conclusions.append(relation["target"])
                
                if premises and conclusions:
                    rule_id = "combination_rule"
                    rules[rule_id] = {
                        "premise": f"IF {' AND '.join(premises)}",
                        "conclusion": f"THEN {' AND '.join(conclusions)}",
                        "type": "combination_based",
                        "confidence": 0.6,
                        "complexity": min(len(premises), self.extraction_config["rule_complexity_limit"])
                    }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"组合规则生成失败: {str(e)}")
            return {}
    
    def _build_knowledge_graph(self, 
                             concepts: Dict[str, Any],
                             relations: Dict[str, Any],
                             rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建知识图谱
        
        Args:
            concepts: 概念
            relations: 关系
            rules: 规则
            
        Returns:
            Dict[str, Any]: 知识图谱
        """
        try:
            # 使用NetworkX构建图谱
            graph = nx.DiGraph()
            
            # 添加概念节点
            for concept_id, concept_info in concepts.items():
                graph.add_node(concept_id, 
                             type="concept",
                             confidence=concept_info.get("confidence", 0.0),
                             attributes=concept_info)
            
            # 添加关系边
            for relation_id, relation_info in relations.items():
                source = relation_info.get("source")
                target = relation_info.get("target")
                if source and target:
                    graph.add_edge(source, target,
                                 type=relation_info.get("type", "relation"),
                                 strength=relation_info.get("strength", 0.0),
                                 relation_id=relation_id)
            
            # 添加规则节点
            for rule_id, rule_info in rules.items():
                graph.add_node(rule_id,
                             type="rule",
                             confidence=rule_info.get("confidence", 0.0),
                             rule_info=rule_info)
                
                # 连接规则到前提和结论概念
                if "premise" in rule_info:
                    # 这里可以解析前提并连接到相应概念
                    pass
                if "conclusion" in rule_info:
                    # 这里可以解析结论并连接到相应概念
                    pass
            
            # 计算图谱统计
            graph_stats = {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
                "connected_components": nx.number_weakly_connected_components(graph),
                "average_clustering": nx.average_clustering(graph.to_undirected())
            }
            
            # 转换为可序列化的格式
            graph_data = {
                "nodes": [
                    {"id": node, **graph.nodes[node]} 
                    for node in graph.nodes()
                ],
                "edges": [
                    {"source": edge[0], "target": edge[1], **graph.edges[edge]}
                    for edge in graph.edges()
                ],
                "statistics": graph_stats
            }
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"知识图谱构建失败: {str(e)}")
            return {"nodes": [], "edges": [], "statistics": {}}
    
    def _assess_knowledge_quality(self, 
                                concepts: Dict[str, Any],
                                relations: Dict[str, Any],
                                rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估提取知识的质量
        
        Args:
            concepts: 概念
            relations: 关系
            rules: 规则
            
        Returns:
            Dict[str, Any]: 质量评估结果
        """
        quality_metrics = {}
        
        try:
            # 1. 概念质量评估
            concept_quality = self._assess_concept_quality(concepts)
            quality_metrics["concept_quality"] = concept_quality
            
            # 2. 关系质量评估
            relation_quality = self._assess_relation_quality(relations)
            quality_metrics["relation_quality"] = relation_quality
            
            # 3. 规则质量评估
            rule_quality = self._assess_rule_quality(rules)
            quality_metrics["rule_quality"] = rule_quality
            
            # 4. 整体质量评估
            overall_quality = self._calculate_overall_quality(
                concept_quality, relation_quality, rule_quality
            )
            quality_metrics["overall_quality"] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"知识质量评估失败: {str(e)}")
            return {"overall_quality": 0.0}
    
    def _assess_concept_quality(self, concepts: Dict[str, Any]) -> Dict[str, float]:
        """评估概念质量"""
        if not concepts:
            return {"average_confidence": 0.0, "coverage_score": 0.0, "quality_score": 0.0}
        
        confidences = [info.get("confidence", 0.0) for info in concepts.values()]
        
        return {
            "average_confidence": float(np.mean(confidences)),
            "coverage_score": min(len(concepts) / 10.0, 1.0),  # 标准化到0-1
            "quality_score": float(np.mean(confidences))
        }
    
    def _assess_relation_quality(self, relations: Dict[str, Any]) -> Dict[str, float]:
        """评估关系质量"""
        if not relations:
            return {"average_strength": 0.0, "connectivity_score": 0.0, "quality_score": 0.0}
        
        strengths = [info.get("strength", 0.0) for info in relations.values()]
        
        return {
            "average_strength": float(np.mean(strengths)),
            "connectivity_score": min(len(relations) / 5.0, 1.0),  # 标准化到0-1
            "quality_score": float(np.mean(strengths))
        }
    
    def _assess_rule_quality(self, rules: Dict[str, Any]) -> Dict[str, float]:
        """评估规则质量"""
        if not rules:
            return {"average_confidence": 0.0, "complexity_score": 0.0, "quality_score": 0.0}
        
        confidences = [info.get("confidence", 0.0) for info in rules.values()]
        complexities = [info.get("complexity", 1) for info in rules.values()]
        
        # 理想复杂度为中等水平
        ideal_complexity = 3.0
        complexity_scores = [1.0 - abs(comp - ideal_complexity) / ideal_complexity 
                           for comp in complexities]
        
        return {
            "average_confidence": float(np.mean(confidences)),
            "complexity_score": float(np.mean(complexity_scores)),
            "quality_score": float(np.mean(confidences)) * 0.7 + float(np.mean(complexity_scores)) * 0.3
        }
    
    def _calculate_overall_quality(self, 
                                 concept_quality: Dict[str, float],
                                 relation_quality: Dict[str, float],
                                 rule_quality: Dict[str, float]) -> float:
        """计算整体质量分数"""
        weights = {"concepts": 0.4, "relations": 0.3, "rules": 0.3}
        
        overall = (
            concept_quality.get("quality_score", 0.0) * weights["concepts"] +
            relation_quality.get("quality_score", 0.0) * weights["relations"] +
            rule_quality.get("quality_score", 0.0) * weights["rules"]
        )
        
        return float(overall)
    
    def _update_extraction_history(self, extraction_result: Dict[str, Any]) -> None:
        """更新提取历史"""
        self.extraction_history.append({
            "timestamp": time.time(),
            "concepts_count": len(extraction_result.get("concepts", {})),
            "relations_count": len(extraction_result.get("relations", {})),
            "rules_count": len(extraction_result.get("rules", {})),
            "quality_score": extraction_result.get("quality_assessment", {}).get("overall_quality", 0.0)
        })
        
        # 保持历史记录在合理范围内
        if len(self.extraction_history) > 100:
            self.extraction_history = self.extraction_history[-100:]
    
    def _update_extraction_stats(self, extraction_result: Dict[str, Any], extraction_time: float) -> None:
        """更新提取统计信息"""
        self.extraction_stats["total_extractions"] += 1
        
        # 更新平均提取时间
        current_avg = self.extraction_stats["average_extraction_time"]
        total_count = self.extraction_stats["total_extractions"]
        self.extraction_stats["average_extraction_time"] = (
            (current_avg * (total_count - 1) + extraction_time) / total_count
        )
        
        # 更新质量分数
        quality_score = extraction_result.get("quality_assessment", {}).get("overall_quality", 0.0)
        self.extraction_stats["knowledge_quality_scores"].append(quality_score)
        
        # 保持分数历史在合理范围内
        if len(self.extraction_stats["knowledge_quality_scores"]) > 50:
            self.extraction_stats["knowledge_quality_scores"] = \
                self.extraction_stats["knowledge_quality_scores"][-50:]
    
    def _calculate_rule_complexity(self, rule_text: str) -> int:
        """计算规则复杂度"""
        # 简化的复杂度计算：基于逻辑操作符和条件的数量
        complexity_indicators = ["IF", "THEN", "AND", "OR", "NOT", "(", ")", ","]
        complexity = sum(1 for indicator in complexity_indicators if indicator in rule_text)
        return min(complexity, self.extraction_config["rule_complexity_limit"])
    
    def add_knowledge(self, new_knowledge: Dict[str, Any]) -> None:
        """
        添加新知识到知识库
        
        Args:
            new_knowledge: 新知识
        """
        try:
            # 合并知识
            if "concepts" in new_knowledge:
                self.knowledge_base.setdefault("concepts", {}).update(new_knowledge["concepts"])
            
            if "relations" in new_knowledge:
                self.knowledge_base.setdefault("relations", {}).update(new_knowledge["relations"])
            
            if "rules" in new_knowledge:
                self.knowledge_base.setdefault("rules", []).extend(new_knowledge.get("rules", []))
            
            self.logger.info("新知识添加到知识库")
            
        except Exception as e:
            self.logger.error(f"添加知识失败: {str(e)}")
    
    def validate_extraction(self) -> float:
        """
        验证提取结果的有效性
        
        Returns:
            float: 验证分数
        """
        try:
            # 检查提取历史
            if not self.extraction_history:
                return 0.0
            
            # 计算历史质量分数
            quality_scores = [record["quality_score"] for record in self.extraction_history[-10:]]
            avg_quality = np.mean(quality_scores)
            
            # 检查一致性
            consistency_score = 1.0 - np.std(quality_scores) if len(quality_scores) > 1 else 1.0
            
            # 组合验证分数
            validation_score = (avg_quality + consistency_score) / 2.0
            
            self.logger.info(f"提取验证完成，分数: {validation_score:.3f}")
            return float(validation_score)
            
        except Exception as e:
            self.logger.error(f"提取验证失败: {str(e)}")
            return 0.0
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        获取提取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        recent_quality = self.extraction_stats["knowledge_quality_scores"][-10:] if self.extraction_stats["knowledge_quality_scores"] else []
        
        return {
            "extraction_stats": self.extraction_stats.copy(),
            "knowledge_base_size": {
                "concepts": len(self.knowledge_base.get("concepts", {})),
                "relations": len(self.knowledge_base.get("relations", {})),
                "rules": len(self.knowledge_base.get("rules", []))
            },
            "recent_quality_trend": recent_quality,
            "extraction_history_length": len(self.extraction_history)
        }