"""
海马体记忆系统 - 记忆处理核心模块
负责概念形成、知识蒸馏、语义网络、记忆提取和长期巩固
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import math
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """记忆单元结构"""
    memory_id: str
    content: Any
    vector_embedding: Optional[np.ndarray]  # 向量表示
    concept_level: int  # 概念抽象层级 (0-5)
    semantic_tags: Set[str]  # 语义标签
    associations: Set[str]  # 关联记忆ID
    timestamp: float
    strength: float  # 记忆强度 (0-1)
    access_count: int  # 访问次数
    consolidation_level: int  # 巩固层级 (0-5)
    memory_type: str = "episodic"  # 记忆类型
    reward_value: float = 0.0  # 奖励值
    emotional_valence: float = 0.0  # 情感极性
    creativity_flag: bool = False  # 创造力标记


@dataclass
class Concept:
    """概念结构"""
    concept_id: str
    name: str
    definition: str
    attributes: Set[str]
    examples: List[str]
    abstraction_level: int  # 抽象层级
    prototype_embedding: np.ndarray  # 原型向量
    constituent_memories: List[str]  # 组成记忆ID
    related_concepts: Set[str]  # 相关概念ID
    formation_time: float
    confidence_score: float = 1.0


@dataclass
class DistilledKnowledge:
    """蒸馏知识结构"""
    knowledge_id: str
    original_memory_ids: List[str]
    compressed_embedding: np.ndarray  # 压缩后的向量
    key_features: Dict[str, float]  # 关键特征
    compression_ratio: float  # 压缩比例
    fidelity_score: float  # 保真度分数
    formation_time: float
    quality_score: float  # 质量分数


class SemanticNetwork:
    """语义记忆网络"""
    
    def __init__(self, embedding_dim: int = 256):
        self.nodes: Dict[str, Concept] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.categories: Dict[str, Set[str]] = defaultdict(set)
        self.embedding_dim = embedding_dim
        self.hierarchy_levels = 6
        
    def add_concept(self, concept: Concept):
        """添加概念到语义网络"""
        self.nodes[concept.concept_id] = concept
        
    def add_association(self, concept1: str, concept2: str, strength: float = 1.0):
        """添加概念间关联"""
        self.edges[concept1][concept2] = strength
        self.edges[concept2][concept1] = strength
        
    def get_related_concepts(self, concept_id: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """获取相关概念"""
        if concept_id not in self.edges:
            return []
        
        related = [(other, strength) for other, strength in self.edges[concept_id].items() 
                  if strength >= threshold]
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def build_semantic_hierarchy(self) -> Dict[int, Set[str]]:
        """构建语义层次结构"""
        hierarchy = defaultdict(set)
        
        for concept in self.nodes.values():
            hierarchy[concept.abstraction_level].add(concept.concept_id)
        
        return dict(hierarchy)
    
    def compute_concept_similarity(self, concept1_id: str, concept2_id: str) -> float:
        """计算概念相似度"""
        if concept1_id not in self.nodes or concept2_id not in self.nodes:
            return 0.0
        
        concept1 = self.nodes[concept1_id]
        concept2 = self.nodes[concept2_id]
        
        # 属性重叠度
        attr_overlap = len(concept1.attributes & concept2.attributes)
        attr_union = len(concept1.attributes | concept2.attributes)
        attr_similarity = attr_overlap / max(attr_union, 1)
        
        # 抽象层级相似度
        level_diff = abs(concept1.abstraction_level - concept2.abstraction_level)
        level_similarity = 1.0 / (1.0 + level_diff)
        
        # 嵌入向量相似度
        if concept1.prototype_embedding is not None and concept2.prototype_embedding is not None:
            vec_similarity = np.dot(concept1.prototype_embedding, concept2.prototype_embedding) / (
                np.linalg.norm(concept1.prototype_embedding) * np.linalg.norm(concept2.prototype_embedding) + 1e-8
            )
        else:
            vec_similarity = 0.0
        
        # 综合相似度
        total_similarity = (attr_similarity * 0.4 + level_similarity * 0.3 + vec_similarity * 0.3)
        return max(0.0, total_similarity)


class ConceptFormationNetwork(nn.Module):
    """概念形成网络"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 概念编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # 抽象化网络
        self.abstraction_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6个抽象层级
        )
        
        # 概念分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 10种概念类型
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 概念编码
        encoded = self.encoder(embeddings)
        
        # 抽象级别预测
        abstraction_probs = F.softmax(self.abstraction_net(encoded), dim=-1)
        
        # 概念类型预测
        type_probs = F.softmax(self.classifier(encoded), dim=-1)
        
        return {
            'encoded_concepts': encoded,
            'abstraction_probs': abstraction_probs,
            'type_probs': type_probs
        }
    
    def extract_prototype(self, embeddings: torch.Tensor) -> torch.Tensor:
        """提取原型向量"""
        return torch.mean(embeddings, dim=0)


class KnowledgeDistiller(nn.Module):
    """知识蒸馏器"""
    
    def __init__(self, input_dim: int = 256, compression_ratio: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = int(input_dim * compression_ratio)
        
        # 注意力聚合器
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 压缩器
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, self.compressed_dim),
            nn.ReLU(),
            nn.Linear(self.compressed_dim, input_dim)
        )
        
        # 质量评估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # 保真度 + 泛化能力
        )
    
    def forward(self, memory_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """知识蒸馏过程"""
        batch_size = memory_embeddings.size(0)
        
        # 注意力聚合
        attended, attention_weights = self.attention(
            memory_embeddings, memory_embeddings, memory_embeddings
        )
        
        # 全局表示
        global_repr = attended.mean(dim=0)
        
        # 特征提取
        features = self.feature_extractor(global_repr)
        
        # 压缩表示
        compressed = self.compressor(global_repr)
        
        # 质量评估
        quality_scores = self.quality_assessor(global_repr)
        
        return {
            'compressed_embedding': compressed,
            'attention_weights': attention_weights,
            'fidelity_score': quality_scores[0],
            'generalization_score': quality_scores[1],
            'features': features
        }


class HippocampusMemorySystem:
    """海马体记忆系统核心类"""
    
    def __init__(self, 
                 max_memory_size: int = 10000,
                 embedding_dim: int = 256,
                 consolidation_hour: int = 22):
        # 核心参数
        self.max_memory_size = max_memory_size
        self.embedding_dim = embedding_dim
        self.consolidation_hour = consolidation_hour
        
        # 记忆存储
        self.memories: Dict[str, Memory] = {}
        self.semantic_network = SemanticNetwork(embedding_dim)
        
        # 概念和知识存储
        self.concepts: Dict[str, Concept] = {}
        self.distilled_knowledge: Dict[str, DistilledKnowledge] = {}
        
        # 神经网络组件
        self.concept_network = ConceptFormationNetwork(embedding_dim)
        self.knowledge_distiller = KnowledgeDistiller(embedding_dim)
        
        # 工作记忆缓冲区
        self.working_memory: deque = deque(maxlen=7)
        self.memory_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 记忆处理参数
        self.attention_threshold = 0.7
        self.consolidation_threshold = 0.8
        self.forgetting_threshold = 0.1
        
        # 统计信息
        self.stats = {
            'total_memories': 0,
            'consolidated_memories': 0,
            'concepts_formed': 0,
            'knowledge_distilled': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0
        }
        
        # 并发处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        logger.info(f"海马体记忆系统初始化完成 - 容量: {max_memory_size}, 维度: {embedding_dim}")
    
    # ==================== 概念形成和抽象化机制 ====================
    
    def form_concepts_from_memories(self, memory_ids: List[str]) -> List[str]:
        """从记忆中形成概念"""
        if len(memory_ids) < 3:
            return []
        
        with self.lock:
            try:
                # 收集记忆数据
                memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
                if len(memories) < 3:
                    return []
                
                # 准备嵌入向量
                embeddings = []
                for memory in memories:
                    if memory.vector_embedding is not None:
                        embeddings.append(torch.tensor(memory.vector_embedding, dtype=torch.float32))
                    else:
                        # 生成随机嵌入向量
                        embeddings.append(torch.randn(self.embedding_dim, dtype=torch.float32))
                
                if not embeddings:
                    return []
                
                # 概念网络处理
                embedding_tensor = torch.stack(embeddings)
                concept_result = self.concept_network(embedding_tensor)
                
                # 创建概念
                concept_ids = []
                for i, memory in enumerate(memories):
                    if i >= len(concept_result['abstraction_probs']):
                        break
                    
                    # 确定抽象级别
                    abstraction_probs = concept_result['abstraction_probs'][i]
                    abstraction_level = torch.argmax(abstraction_probs).item()
                    
                    # 确定概念类型
                    type_probs = concept_result['type_probs'][i]
                    concept_type = torch.argmax(type_probs).item()
                    
                    # 创建概念
                    concept_id = str(uuid.uuid4())
                    prototype = concept_result['encoded_concepts'][i].detach().float().numpy()
                    
                    concept = Concept(
                        concept_id=concept_id,
                        name=f"概念_{concept_type}_{len(self.concepts)}",
                        definition=f"基于{len(memories)}个记忆形成的抽象概念",
                        attributes=self._extract_concept_attributes(memory),
                        examples=[memory.content for memory in memories[:5]],
                        abstraction_level=abstraction_level,
                        prototype_embedding=prototype,
                        constituent_memories=[m.memory_id for m in memories],
                        related_concepts=set(),
                        formation_time=time.time(),
                        confidence_score=float(torch.max(abstraction_probs))
                    )
                    
                    self.concepts[concept_id] = concept
                    self.semantic_network.add_concept(concept)
                    
                    # 建立记忆与概念的关联
                    memory.concept_level = abstraction_level
                    memory.associations.add(concept_id)
                    
                    concept_ids.append(concept_id)
                
                self.stats['concepts_formed'] += len(concept_ids)
                logger.info(f"形成{len(concept_ids)}个新概念")
                
                return concept_ids
                
            except Exception as e:
                logger.error(f"概念形成失败: {str(e)}")
                return []
    
    def _extract_concept_attributes(self, memory: Memory) -> Set[str]:
        """提取概念属性"""
        attributes = set()
        
        # 从内容提取属性
        if isinstance(memory.content, str):
            content_lower = memory.content.lower()
            
            # 关键词提取
            keywords = ['学习', '工作', '生活', '朋友', '家庭', '时间', '地点', '建造', '创造']
            for keyword in keywords:
                if keyword in content_lower:
                    attributes.add(keyword)
            
            # 情感属性
            positive_words = ['好', '棒', '喜欢', '快乐', '成功', '满意']
            negative_words = ['坏', '差', '讨厌', '痛苦', '失败', '不满']
            
            for word in positive_words:
                if word in content_lower:
                    attributes.add('positive')
                    break
            
            for word in negative_words:
                if word in content_lower:
                    attributes.add('negative')
                    break
        
        # 添加记忆类型属性
        attributes.add(memory.memory_type)
        
        # 添加奖励属性
        if memory.reward_value > 0.5:
            attributes.add('high_reward')
        elif memory.reward_value < -0.5:
            attributes.add('low_reward')
        
        return attributes
    
    # ==================== 知识蒸馏和压缩存储 ====================
    
    def distill_knowledge(self, memory_ids: List[str]) -> Optional[str]:
        """知识蒸馏 - 压缩存储记忆"""
        if len(memory_ids) < 5:
            return None
        
        with self.lock:
            try:
                # 收集记忆
                memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
                if len(memories) < 5:
                    return None
                
                # 准备嵌入向量
                embeddings = []
                valid_memories = []
                
                for memory in memories:
                    if memory.vector_embedding is not None:
                        embeddings.append(torch.tensor(memory.vector_embedding, dtype=torch.float32))
                        valid_memories.append(memory)
                
                if len(embeddings) < 5:
                    return None
                
                # 执行知识蒸馏
                embedding_tensor = torch.stack(embeddings)
                distillation_result = self.knowledge_distiller(embedding_tensor)
                
                # 创建蒸馏知识对象
                knowledge_id = str(uuid.uuid4())
                compressed_embedding = distillation_result['compressed_embedding'].detach().float().numpy()
                
                # 提取关键特征
                key_features = self._extract_key_features(valid_memories)
                
                # 计算压缩比例
                original_size = len(embeddings) * self.embedding_dim
                compressed_size = compressed_embedding.size
                compression_ratio = original_size / max(compressed_size, 1)
                
                # 计算质量分数
                fidelity_score = float(distillation_result['fidelity_score'])
                generalization_score = float(distillation_result['generalization_score'])
                quality_score = (fidelity_score + generalization_score) / 2
                
                distilled_knowledge = DistilledKnowledge(
                    knowledge_id=knowledge_id,
                    original_memory_ids=[m.memory_id for m in valid_memories],
                    compressed_embedding=compressed_embedding,
                    key_features=key_features,
                    compression_ratio=compression_ratio,
                    fidelity_score=fidelity_score,
                    formation_time=time.time(),
                    quality_score=quality_score
                )
                
                self.distilled_knowledge[knowledge_id] = distilled_knowledge
                self.stats['knowledge_distilled'] += 1
                
                # 标记原始记忆为已蒸馏
                for memory in valid_memories:
                    memory.consolidation_level += 1
                
                logger.info(f"蒸馏知识 {knowledge_id[:8]}... - 压缩比: {compression_ratio:.2f}, 质量: {quality_score:.3f}")
                
                return knowledge_id
                
            except Exception as e:
                logger.error(f"知识蒸馏失败: {str(e)}")
                return None
    
    def _extract_key_features(self, memories: List[Memory]) -> Dict[str, float]:
        """提取关键特征"""
        features = {}
        
        # 统计特征
        rewards = [m.reward_value for m in memories]
        features['avg_reward'] = np.mean(rewards)
        features['reward_variance'] = np.var(rewards)
        features['max_reward'] = np.max(rewards)
        features['min_reward'] = np.min(rewards)
        
        # 情感特征
        emotional_vals = [m.emotional_valence for m in memories]
        features['avg_emotional_valence'] = np.mean(emotional_vals)
        features['emotional_variance'] = np.var(emotional_vals)
        
        # 记忆类型分布
        type_counts = defaultdict(int)
        for memory in memories:
            type_counts[memory.memory_type] += 1
        
        for mem_type, count in type_counts.items():
            features[f'type_{mem_type}_ratio'] = count / len(memories)
        
        # 创造力比例
        creative_count = sum(1 for m in memories if m.creativity_flag)
        features['creativity_ratio'] = creative_count / len(memories)
        
        # 概念层级分布
        concept_levels = [m.concept_level for m in memories]
        features['avg_concept_level'] = np.mean(concept_levels)
        
        return features
    
    # ==================== 语义记忆网络 ====================
    
    def build_semantic_network(self):
        """构建语义记忆网络"""
        with self.lock:
            # 添加所有概念到语义网络
            for concept in self.concepts.values():
                self.semantic_network.add_concept(concept)
            
            # 计算概念间关联
            concept_ids = list(self.concepts.keys())
            for i, concept1_id in enumerate(concept_ids):
                for concept2_id in concept_ids[i+1:]:
                    similarity = self.semantic_network.compute_concept_similarity(
                        concept1_id, concept2_id
                    )
                    
                    if similarity > 0.5:
                        self.semantic_network.add_association(
                            concept1_id, concept2_id, similarity
                        )
                        
                        # 更新概念的关联集合
                        self.concepts[concept1_id].related_concepts.add(concept2_id)
                        self.concepts[concept2_id].related_concepts.add(concept1_id)
            
            logger.info(f"语义网络构建完成 - {len(self.concepts)}个概念, "
                       f"{sum(len(connections) for connections in self.semantic_network.edges.values()) // 2}个关联")
    
    def find_semantic_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """查找概念的语义关系"""
        if concept_id not in self.semantic_network.nodes:
            return []
        
        relationships = []
        
        # 直接关联
        direct_relations = self.semantic_network.get_related_concepts(concept_id, threshold=0.3)
        for related_id, strength in direct_relations:
            relationships.append({
                'type': 'direct',
                'target': related_id,
                'target_name': self.concepts[related_id].name,
                'strength': strength,
                'description': f"直接关联 - 强度: {strength:.3f}"
            })
        
        # 语义相似
        for other_id in self.concepts:
            if other_id != concept_id:
                similarity = self.semantic_network.compute_concept_similarity(concept_id, other_id)
                if similarity > 0.6:
                    relationships.append({
                        'type': 'similarity',
                        'target': other_id,
                        'target_name': self.concepts[other_id].name,
                        'strength': similarity,
                        'description': f"语义相似 - 相似度: {similarity:.3f}"
                    })
        
        # 按强度排序
        relationships.sort(key=lambda x: x['strength'], reverse=True)
        return relationships[:10]  # 返回前10个关系
    
    # ==================== 记忆提取和关联 ====================
    
    def retrieve_memories(self, 
                         query: Any, 
                         top_k: int = 10,
                         similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """记忆提取和关联检索"""
        start_time = time.time()
        
        with self.lock:
            try:
                # 生成查询嵌入
                query_embedding = self._create_embedding(query)
                
                # 计算与所有记忆的相似度
                similarities = []
                
                for memory_id, memory in self.memories.items():
                    similarity = self._compute_similarity(query_embedding, memory)
                    
                    if similarity >= similarity_threshold:
                        similarities.append((memory, similarity))
                
                # 按相似度排序
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 获取top_k结果
                results = []
                for memory, similarity in similarities[:top_k]:
                    # 获取关联记忆
                    related_memories = self._find_associated_memories(memory)
                    
                    # 获取相关概念
                    related_concepts = self._find_related_concepts(memory)
                    
                    result = {
                        'memory': memory,
                        'similarity_score': similarity,
                        'related_memories': related_memories,
                        'related_concepts': related_concepts,
                        'association_strength': self._calculate_association_strength(memory, query_embedding)
                    }
                    
                    results.append(result)
                
                # 更新检索统计
                retrieval_time = time.time() - start_time
                if results:
                    self.stats['successful_retrievals'] += 1
                else:
                    self.stats['failed_retrievals'] += 1
                
                logger.info(f"记忆检索完成 - 找到{len(results)}个结果, 耗时: {retrieval_time*1000:.1f}ms")
                
                return results
                
            except Exception as e:
                logger.error(f"记忆检索失败: {str(e)}")
                self.stats['failed_retrievals'] += 1
                return []
    
    def _create_embedding(self, content: Any) -> np.ndarray:
        """创建向量嵌入"""
        if isinstance(content, np.ndarray):
            return content
        
        if isinstance(content, str):
            # 简单的文本嵌入 (实际应用中应使用更好的方法)
            hash_value = hash(content)
            np.random.seed(hash_value % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        # 其他类型的处理
        content_str = str(content)
        hash_value = hash(content_str)
        np.random.seed(hash_value % (2**32))
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def _compute_similarity(self, query_embedding: np.ndarray, memory: Memory) -> float:
        """计算相似度"""
        if memory.vector_embedding is None:
            # 为记忆生成向量表示
            memory.vector_embedding = self._create_embedding(memory.content)
        
        # 余弦相似度
        query_norm = np.linalg.norm(query_embedding)
        memory_norm = np.linalg.norm(memory.vector_embedding)
        
        if query_norm == 0 or memory_norm == 0:
            return 0.0
        
        similarity = np.dot(query_embedding, memory.vector_embedding) / (query_norm * memory_norm)
        
        # 应用记忆强度权重
        weighted_similarity = similarity * (0.5 + memory.strength * 0.5)
        
        return max(0.0, weighted_similarity)
    
    def _find_associated_memories(self, memory: Memory) -> List[Memory]:
        """查找关联记忆"""
        associated = []
        
        for assoc_id in memory.associations:
            if assoc_id in self.memories:
                associated.append(self.memories[assoc_id])
        
        # 通过语义网络查找相关概念的记忆
        for concept_id in memory.associations:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                for mem_id in concept.constituent_memories:
                    if mem_id in self.memories and mem_id != memory.memory_id:
                        associated.append(self.memories[mem_id])
        
        # 去重
        unique_associated = []
        seen_ids = set()
        for assoc_memory in associated:
            if assoc_memory.memory_id not in seen_ids:
                seen_ids.add(assoc_memory.memory_id)
                unique_associated.append(assoc_memory)
        
        return unique_associated[:5]  # 返回前5个关联记忆
    
    def _find_related_concepts(self, memory: Memory) -> List[Concept]:
        """查找相关概念"""
        related_concepts = []
        
        for concept_id in memory.associations:
            if concept_id in self.concepts:
                related_concepts.append(self.concepts[concept_id])
        
        # 通过语义网络查找相似概念
        for concept_id in memory.associations:
            if concept_id in self.semantic_network.edges:
                for related_id in self.semantic_network.edges[concept_id]:
                    if related_id in self.concepts and related_id not in [c.concept_id for c in related_concepts]:
                        related_concepts.append(self.concepts[related_id])
        
        return related_concepts[:3]  # 返回前3个相关概念
    
    def _calculate_association_strength(self, memory: Memory, query_embedding: np.ndarray) -> float:
        """计算关联强度"""
        # 基于记忆强度的关联强度
        base_strength = memory.strength
        
        # 基于访问频率的增强
        access_boost = 1.0 + math.log(memory.access_count + 1) * 0.1
        
        # 基于情感极性的增强
        emotion_boost = 1.0 + abs(memory.emotional_valence) * 0.2
        
        # 基于创造力的增强
        creativity_boost = 1.1 if memory.creativity_flag else 1.0
        
        total_strength = base_strength * access_boost * emotion_boost * creativity_boost
        return min(1.0, total_strength)
    
    # ==================== 长期记忆巩固 ====================
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """长期记忆巩固"""
        start_time = time.time()
        
        with self.lock:
            try:
                current_time = time.time()
                
                # 检查巩固条件
                if not force and not self._should_consolidate():
                    return {
                        'status': 'skipped',
                        'reason': '未达到巩固条件',
                        'consolidation_time': None
                    }
                
                logger.info("开始记忆巩固过程...")
                
                # 1. 重新计算记忆强度
                self._recalculate_memory_strengths()
                
                # 2. 应用遗忘机制
                forgotten_count = self._apply_forgetting_mechanism()
                
                # 3. 概念形成
                new_concepts = self._trigger_concept_formation()
                
                # 4. 知识蒸馏
                new_distilled_knowledge = self._trigger_knowledge_distillation()
                
                # 5. 语义网络更新
                self.build_semantic_network()
                
                # 6. 记忆巩固
                consolidated_count = 0
                for memory in list(self.memories.values()):
                    if (memory.strength > self.consolidation_threshold and 
                        memory.consolidation_level < 5):
                        memory.consolidation_level += 1
                        consolidated_count += 1
                
                # 更新统计
                self.stats['consolidated_memories'] += consolidated_count
                
                consolidation_time = time.time() - start_time
                
                result = {
                    'status': 'success',
                    'consolidation_time': current_time,
                    'processing_time': consolidation_time,
                    'consolidated_memories': consolidated_count,
                    'forgotten_memories': forgotten_count,
                    'new_concepts': len(new_concepts),
                    'new_distilled_knowledge': len(new_distilled_knowledge),
                    'total_memories': len(self.memories),
                    'total_concepts': len(self.concepts),
                    'total_distilled_knowledge': len(self.distilled_knowledge)
                }
                
                logger.info(f"记忆巩固完成 - 巩固{consolidated_count}个记忆, "
                           f"遗忘{forgotten_count}个记忆, 耗时{consolidation_time:.2f}秒")
                
                return result
                
            except Exception as e:
                logger.error(f"记忆巩固失败: {str(e)}")
                return {'status': 'error', 'error': str(e)}
    
    def _should_consolidate(self) -> bool:
        """检查是否应该进行巩固"""
        # 检查是否到达巩固时间
        current_hour = int(time.time() / 3600) % 24
        if current_hour == self.consolidation_hour:
            return True
        
        # 或者记忆数量达到阈值
        if len(self.memories) > self.max_memory_size * 0.8:
            return True
        
        # 或者有足够的弱记忆需要巩固
        weak_memories = sum(1 for m in self.memories.values() if m.strength < 0.3)
        if weak_memories > len(self.memories) * 0.3:
            return True
        
        return False
    
    def _recalculate_memory_strengths(self):
        """重新计算记忆强度"""
        current_time = time.time()
        
        for memory in self.memories.values():
            # 时间衰减
            hours_elapsed = (current_time - memory.timestamp) / 3600
            time_decay = math.exp(-0.1 * hours_elapsed)
            
            # 访问频率增强
            access_boost = 1.0 + math.log(memory.access_count + 1) * 0.1
            
            # 情感权重
            emotional_weight = 1.0 + abs(memory.emotional_valence) * 0.3
            
            # 奖励权重
            reward_weight = 1.0 + max(0, memory.reward_value) * 0.5
            
            # 创造力权重
            creativity_weight = 1.1 if memory.creativity_flag else 1.0
            
            # 记忆类型权重
            type_weights = {
                'episodic': 1.0,
                'semantic': 1.2,
                'procedural': 1.1,
                'creative': 1.3
            }
            type_weight = type_weights.get(memory.memory_type, 1.0)
            
            # 综合强度计算
            memory.strength = (time_decay * access_boost * emotional_weight * 
                             reward_weight * creativity_weight * type_weight)
            
            # 确保强度在合理范围内
            memory.strength = max(0.001, min(1.0, memory.strength))
    
    def _apply_forgetting_mechanism(self) -> int:
        """应用遗忘机制"""
        forgotten_count = 0
        memories_to_remove = []
        
        for memory_id, memory in self.memories.items():
            # 遗忘条件
            should_forget = (
                memory.reward_value < self.forgetting_threshold or
                memory.strength < 0.005 or
                (memory.access_count == 0 and memory.strength < 0.01)
            )
            
            if should_forget:
                memories_to_remove.append(memory_id)
        
        # 移除遗忘的记忆
        for memory_id in memories_to_remove:
            del self.memories[memory_id]
            if memory_id in self.memory_activity:
                del self.memory_activity[memory_id]
            forgotten_count += 1
        
        return forgotten_count
    
    def _trigger_concept_formation(self) -> List[str]:
        """触发概念形成"""
        # 按相似度分组记忆
        memory_groups = self._group_memories_by_similarity()
        
        new_concepts = []
        for group in memory_groups:
            if len(group) >= 3:
                concept_ids = self.form_concepts_from_memories(group)
                new_concepts.extend(concept_ids)
        
        return new_concepts
    
    def _trigger_knowledge_distillation(self) -> List[str]:
        """触发知识蒸馏"""
        # 按类型分组记忆
        memory_groups = self._group_memories_by_type()
        
        new_distilled_knowledge = []
        for group in memory_groups:
            if len(group) >= 5:
                knowledge_id = self.distill_knowledge(group)
                if knowledge_id:
                    new_distilled_knowledge.append(knowledge_id)
        
        return new_distilled_knowledge
    
    def _group_memories_by_similarity(self, threshold: float = 0.7) -> List[List[str]]:
        """按相似度分组记忆"""
        memory_ids = list(self.memories.keys())
        groups = []
        used_memories = set()
        
        for i, memory_id in enumerate(memory_ids):
            if memory_id in used_memories:
                continue
            
            group = [memory_id]
            used_memories.add(memory_id)
            
            for j, other_id in enumerate(memory_ids[i+1:], i+1):
                if other_id in used_memories:
                    continue
                
                similarity = self._compute_memory_similarity(memory_id, other_id)
                if similarity > threshold:
                    group.append(other_id)
                    used_memories.add(other_id)
            
            groups.append(group)
        
        return groups
    
    def _group_memories_by_type(self) -> List[List[str]]:
        """按类型分组记忆"""
        type_groups = defaultdict(list)
        
        for memory_id, memory in self.memories.items():
            type_groups[memory.memory_type].append(memory_id)
        
        return [group for group in type_groups.values() if len(group) >= 5]
    
    def _compute_memory_similarity(self, memory_id1: str, memory_id2: str) -> float:
        """计算两个记忆的相似度"""
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return 0.0
        
        memory1 = self.memories[memory_id1]
        memory2 = self.memories[memory_id2]
        
        # 确保记忆有向量表示
        if memory1.vector_embedding is None:
            memory1.vector_embedding = self._create_embedding(memory1.content).astype(np.float32)
        if memory2.vector_embedding is None:
            memory2.vector_embedding = self._create_embedding(memory2.content).astype(np.float32)
        
        # 计算余弦相似度
        similarity = np.dot(memory1.vector_embedding, memory2.vector_embedding) / (
            np.linalg.norm(memory1.vector_embedding) * np.linalg.norm(memory2.vector_embedding) + 1e-8
        )
        
        return max(0.0, similarity)
    
    # ==================== 主要接口方法 ====================
    
    def store_memory(self, 
                    content: Any,
                    memory_type: str = "episodic",
                    reward_value: float = 0.0,
                    emotional_valence: float = 0.0,
                    creativity_flag: bool = False) -> str:
        """存储新记忆"""
        with self.lock:
            memory_id = str(uuid.uuid4())
            
            # 创建记忆
            memory = Memory(
                memory_id=memory_id,
                content=content,
                vector_embedding=self._create_embedding(content),
                concept_level=0,
                semantic_tags=self._extract_semantic_tags(content),
                associations=set(),
                timestamp=time.time(),
                strength=1.0,
                access_count=0,
                consolidation_level=0,
                memory_type=memory_type,
                reward_value=reward_value,
                emotional_valence=emotional_valence,
                creativity_flag=creativity_flag
            )
            
            self.memories[memory_id] = memory
            self.working_memory.append(memory_id)
            
            # 启动异步概念形成
            self.executor.submit(self._async_concept_detection, memory_id)
            
            self.stats['total_memories'] += 1
            logger.info(f"存储记忆: {memory_id[:8]}... ({memory_type})")
            
            return memory_id
    
    def _extract_semantic_tags(self, content: Any) -> Set[str]:
        """提取语义标签"""
        tags = set()
        
        if isinstance(content, str):
            content_lower = content.lower()
            
            # 关键词标签
            keywords = ['是', '有', '被', '进行', '完成', '开始', '结束', '学习', 
                       '工作', '生活', '朋友', '家庭', '时间', '地点', '原因', '结果',
                       '建造', '创造', '发现', '理解', '实现', '改进', '优化']
            
            for keyword in keywords:
                if keyword in content_lower:
                    tags.add(keyword)
            
            # 情感标签
            positive_words = ['好', '棒', '喜欢', '快乐', '成功', '满意', '兴奋']
            negative_words = ['坏', '差', '讨厌', '痛苦', '失败', '不满', '失望']
            
            for word in positive_words:
                if word in content_lower:
                    tags.add('positive')
                    break
            
            for word in negative_words:
                if word in content_lower:
                    tags.add('negative')
                    break
            
            # 时间标签
            time_indicators = ['今天', '昨天', '明天', '现在', '以前', '以后', '将来', '过去']
            for indicator in time_indicators:
                if indicator in content_lower:
                    tags.add('temporal')
                    break
            
            # 行动标签
            action_words = ['建造', '创建', '修复', '改进', '分析', '设计', '测试', '验证']
            for action in action_words:
                if action in content_lower:
                    tags.add('action')
                    break
        
        return tags
    
    def _async_concept_detection(self, memory_id: str):
        """异步概念检测"""
        try:
            # 等待一段时间让更多记忆积累
            time.sleep(1)
            
            # 检查是否有足够的相似记忆
            memory = self.memories.get(memory_id)
            if not memory:
                return
            
            similar_memories = []
            for other_id, other_memory in self.memories.items():
                if other_id != memory_id:
                    similarity = self._compute_memory_similarity(memory_id, other_id)
                    if similarity > 0.6:
                        similar_memories.append(other_id)
            
            if len(similar_memories) >= 2:
                # 触发概念形成
                self.form_concepts_from_memories([memory_id] + similar_memories[:2])
                
        except Exception as e:
            logger.error(f"异步概念检测失败: {str(e)}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        with self.lock:
            # 记忆分布统计
            memory_type_counts = defaultdict(int)
            consolidation_state_counts = defaultdict(int)
            concept_level_counts = defaultdict(int)
            
            for memory in self.memories.values():
                memory_type_counts[memory.memory_type] += 1
                consolidation_state_counts[memory.consolidation_level] += 1
                concept_level_counts[memory.concept_level] += 1
            
            # 记忆强度分布
            strength_distribution = {
                'weak': sum(1 for m in self.memories.values() if m.strength < 0.3),
                'medium': sum(1 for m in self.memories.values() if 0.3 <= m.strength < 0.7),
                'strong': sum(1 for m in self.memories.values() if m.strength >= 0.7)
            }
            
            # 语义网络统计
            semantic_stats = {
                'total_concepts': len(self.concepts),
                'total_associations': sum(len(connections) for connections in self.semantic_network.edges.values()) // 2,
                'avg_concept_connections': np.mean([len(connections) for connections in self.semantic_network.edges.values()]) if self.semantic_network.edges else 0
            }
            
            # 检索性能
            total_retrievals = self.stats['successful_retrievals'] + self.stats['failed_retrievals']
            retrieval_accuracy = self.stats['successful_retrievals'] / max(total_retrievals, 1)
            
            return {
                'memory_overview': {
                    'total_memories': len(self.memories),
                    'memory_capacity_usage': len(self.memories) / self.max_memory_size,
                    'working_memory_size': len(self.working_memory)
                },
                'memory_distribution': {
                    'by_type': dict(memory_type_counts),
                    'by_consolidation_level': dict(consolidation_state_counts),
                    'by_concept_level': dict(concept_level_counts),
                    'by_strength': strength_distribution
                },
                'conceptual_stats': {
                    'total_concepts': len(self.concepts),
                    'concepts_formed': self.stats['concepts_formed'],
                    'semantic_network': semantic_stats
                },
                'knowledge_stats': {
                    'total_distilled_knowledge': len(self.distilled_knowledge),
                    'knowledge_distilled': self.stats['knowledge_distilled'],
                    'avg_compression_ratio': np.mean([dk.compression_ratio for dk in self.distilled_knowledge.values()]) if self.distilled_knowledge else 0,
                    'avg_quality_score': np.mean([dk.quality_score for dk in self.distilled_knowledge.values()]) if self.distilled_knowledge else 0
                },
                'performance_stats': {
                    'retrieval_accuracy': retrieval_accuracy,
                    'successful_retrievals': self.stats['successful_retrievals'],
                    'failed_retrievals': self.stats['failed_retrievals'],
                    'consolidation_cycles': self.stats['consolidated_memories']
                },
                'system_stats': self.stats.copy()
            }
    
    def export_memory_state(self, filepath: str):
        """导出记忆状态"""
        with self.lock:
            export_data = {
                'memories': {
                    mid: {
                        'memory_id': m.memory_id,
                        'content': str(m.content) if not isinstance(m.content, (str, int, float)) else m.content,
                        'concept_level': m.concept_level,
                        'semantic_tags': list(m.semantic_tags),
                        'associations': list(m.associations),
                        'timestamp': m.timestamp,
                        'strength': m.strength,
                        'access_count': m.access_count,
                        'consolidation_level': m.consolidation_level,
                        'memory_type': m.memory_type,
                        'reward_value': m.reward_value,
                        'emotional_valence': m.emotional_valence,
                        'creativity_flag': m.creativity_flag,
                        'vector_embedding': m.vector_embedding.tolist() if m.vector_embedding is not None else None
                    }
                    for mid, m in self.memories.items()
                },
                'concepts': {
                    cid: {
                        'concept_id': c.concept_id,
                        'name': c.name,
                        'definition': c.definition,
                        'attributes': list(c.attributes),
                        'examples': c.examples,
                        'abstraction_level': c.abstraction_level,
                        'prototype_embedding': c.prototype_embedding.tolist() if c.prototype_embedding is not None else None,
                        'constituent_memories': c.constituent_memories,
                        'related_concepts': list(c.related_concepts),
                        'formation_time': c.formation_time,
                        'confidence_score': c.confidence_score
                    }
                    for cid, c in self.concepts.items()
                },
                'distilled_knowledge': {
                    kid: {
                        'knowledge_id': dk.knowledge_id,
                        'original_memory_ids': dk.original_memory_ids,
                        'compressed_embedding': dk.compressed_embedding.tolist(),
                        'key_features': dk.key_features,
                        'compression_ratio': dk.compression_ratio,
                        'fidelity_score': dk.fidelity_score,
                        'formation_time': dk.formation_time,
                        'quality_score': dk.quality_score
                    }
                    for kid, dk in self.distilled_knowledge.items()
                },
                'stats': self.stats,
                'export_time': time.time()
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"记忆状态已导出到: {filepath}")
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)


# 使用示例
if __name__ == "__main__":
    print("=" * 80)
    print("🧠 海马体记忆系统测试")
    print("   概念形成 + 知识蒸馏 + 语义网络 + 记忆提取 + 长期巩固")
    print("=" * 80)
    
    # 创建记忆系统
    memory_system = HippocampusMemorySystem(max_memory_size=1000, embedding_dim=128)
    
    # 存储测试记忆
    test_memories = [
        ("学会了新的编程技术", "semantic", 0.8, 0.5, False),
        ("完成了一个重要项目", "episodic", 0.9, 0.7, True),
        ("和朋友们一起度过了愉快的时光", "episodic", 0.6, 0.8, False),
        ("发现了解决问题的新方法", "creative", 0.7, 0.6, True),
        ("工作中遇到了困难", "episodic", -0.2, -0.3, False),
        ("学会了高效的编程技巧", "semantic", 0.8, 0.5, False),
        ("创造了一个有趣的应用", "creative", 0.9, 0.8, True),
        ("团队合作取得了成功", "episodic", 0.8, 0.7, False)
    ]
    
    print("\n1. 存储测试记忆...")
    memory_ids = []
    for i, (content, mem_type, reward, emotion, creativity) in enumerate(test_memories):
        memory_id = memory_system.store_memory(
            content=content,
            memory_type=mem_type,
            reward_value=reward,
            emotional_valence=emotion,
            creativity_flag=creativity
        )
        memory_ids.append(memory_id)
        print(f"   记忆 {i+1}: {content[:30]}...")
    
    # 等待异步处理
    time.sleep(2)
    
    # 检索测试
    print("\n2. 记忆检索测试...")
    query_results = memory_system.retrieve_memories("编程学习", top_k=5)
    print(f"   查询'编程学习'找到 {len(query_results)} 个结果:")
    for i, result in enumerate(query_results):
        memory = result['memory']
        print(f"     {i+1}. {memory.content} (相似度: {result['similarity_score']:.3f})")
    
    # 概念形成测试
    print("\n3. 概念形成测试...")
    if len(memory_ids) >= 3:
        # 选择几个相似的记忆进行概念形成
        similar_memory_ids = memory_ids[:3]
        concepts = memory_system.form_concepts_from_memories(similar_memory_ids)
        print(f"   形成了 {len(concepts)} 个概念")
        for concept_id in concepts[:2]:  # 显示前2个概念
            concept = memory_system.concepts[concept_id]
            print(f"     - {concept.name}: {concept.definition}")
    
    # 知识蒸馏测试
    print("\n4. 知识蒸馏测试...")
    if len(memory_ids) >= 5:
        # 选择多个记忆进行蒸馏
        distillation_ids = memory_ids[:5]
        knowledge_id = memory_system.distill_knowledge(distillation_ids)
        if knowledge_id:
            knowledge = memory_system.distilled_knowledge[knowledge_id]
            print(f"   蒸馏知识 ID: {knowledge_id[:8]}...")
            print(f"     压缩比: {knowledge.compression_ratio:.2f}")
            print(f"     质量分数: {knowledge.quality_score:.3f}")
        else:
            print("   知识蒸馏失败")
    
    # 语义网络测试
    print("\n5. 语义网络测试...")
    memory_system.build_semantic_network()
    if memory_system.concepts:
        # 找一个概念查看其关系
        first_concept_id = list(memory_system.concepts.keys())[0]
        relationships = memory_system.find_semantic_relationships(first_concept_id)
        concept = memory_system.concepts[first_concept_id]
        print(f"   概念 '{concept.name}' 有 {len(relationships)} 个语义关系")
        for rel in relationships[:2]:  # 显示前2个关系
            print(f"     - {rel['description']}")
    
    # 记忆巩固测试
    print("\n6. 记忆巩固测试...")
    consolidation_result = memory_system.consolidate_memories(force=True)
    print(f"   巩固结果: {consolidation_result['status']}")
    print(f"   巩固记忆数: {consolidation_result.get('consolidated_memories', 0)}")
    print(f"   遗忘记忆数: {consolidation_result.get('forgotten_memories', 0)}")
    print(f"   新概念数: {consolidation_result.get('new_concepts', 0)}")
    print(f"   新蒸馏知识数: {consolidation_result.get('new_distilled_knowledge', 0)}")
    
    # 系统统计
    print("\n7. 系统统计...")
    stats = memory_system.get_memory_statistics()
    print(f"   总记忆数: {stats['memory_overview']['total_memories']}")
    print(f"   记忆使用率: {stats['memory_overview']['memory_capacity_usage']:.1%}")
    print(f"   概念数: {stats['conceptual_stats']['total_concepts']}")
    print(f"   检索准确率: {stats['performance_stats']['retrieval_accuracy']:.1%}")
    
    print("\n" + "=" * 80)
    print("✅ 海马体记忆系统测试完成！")
    print("   系统已具备完整的记忆处理能力")
    print("=" * 80)
    
    # 清理资源
    memory_system.cleanup()