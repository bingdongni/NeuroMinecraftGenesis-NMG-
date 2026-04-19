"""
大规模多智能体协同进化系统 (Mass Multi-Agent Cooperative Evolution System)

该系统实现了数千个AI智能体的协作网络，包含：
1. 大规模智能体网络管理
2. 社会智能和群体智慧
3. 分层网络和动态组织
4. 集体决策和投票系统
5. 社会学习和文化传递
"""

import asyncio
import numpy as np
import random
import json
import time
import threading
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum
import uuid
from copy import deepcopy
import pickle
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    ACTIVE = "active"
    LEARNING = "learning"
    COLLABORATING = "collaborating"
    DECIDING = "deciding"
    RESTING = "resting"

class NetworkLayer(Enum):
    """网络层级枚举"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    COMMUNITY = "community"
    SOCIETY = "society"
    META_SOCIETY = "meta_society"

@dataclass
class KnowledgeNode:
    """知识节点"""
    id: str
    content: Any
    confidence: float
    creator_agent: str
    timestamp: float
    usage_count: int = 0
    influence_score: float = 0.0
    connections: Set[str] = field(default_factory=set)

@dataclass
class CulturalArtifact:
    """文化产物"""
    id: str
    type: str  # skill, norm, belief, protocol
    content: Any
    creators: List[str]
    adoption_count: int
    effectiveness_score: float
    created_time: float
    last_update_time: float

@dataclass
class SocialConnection:
    """社交连接"""
    agent_id: str
    connection_type: str  # collaboration, influence, mentorship
    strength: float
    trust_level: float
    interaction_history: List[Dict] = field(default_factory=list)

class SocialAgent:
    """社会智能体"""
    
    def __init__(self, agent_id: str, capabilities: Dict[str, float], personality: Dict[str, float]):
        self.id = agent_id
        self.capabilities = capabilities  # 能力向量
        self.personality = personality   # 性格特征
        self.state = AgentState.IDLE
        self.social_connections: Dict[str, SocialConnection] = {}
        self.knowledge_base: Dict[str, KnowledgeNode] = {}
        self.learned_cultural_artifacts: Set[str] = set()
        self.experience_level = 0.0
        self.reputation_score = 0.5
        self.influence_network: Set[str] = set()
        self.collaboration_history = []
        self.decision_history = []
        self.last_activity = time.time()
        self.generation = 0
        
    def update_capabilities(self, improvements: Dict[str, float]):
        """更新能力"""
        for skill, improvement in improvements.items():
            if skill in self.capabilities:
                self.capabilities[skill] = min(1.0, self.capabilities[skill] + improvement)
                self.experience_level += improvement * 0.1
        
    def add_social_connection(self, other_agent: 'SocialAgent', connection_type: str, strength: float):
        """添加社交连接"""
        connection = SocialConnection(
            agent_id=other_agent.id,
            connection_type=connection_type,
            strength=strength,
            trust_level=0.5
        )
        self.social_connections[other_agent.id] = connection
        self.influence_network.add(other_agent.id)
        
    def get_influence_score(self) -> float:
        """计算影响力分数"""
        base_score = len(self.influence_network) * 0.1
        trust_bonus = sum(conn.trust_level for conn in self.social_connections.values()) * 0.2
        capability_bonus = sum(self.capabilities.values()) * 0.3
        return min(1.0, base_score + trust_bonus + capability_bonus)

class DecisionMaker:
    """集体决策器"""
    
    def __init__(self, decision_history_size: int = 1000):
        self.decision_history = deque(maxlen=decision_history_size)
        self.consensus_threshold = 0.67  # 共识阈值
        self.voting_weight_factors = {}
        
    def calculate_voting_weight(self, agent: SocialAgent, context: Dict[str, Any]) -> float:
        """计算投票权重"""
        base_weight = agent.get_influence_score()
        expertise_weight = 0.0
        
        # 基于领域专业知识的权重
        if 'domain' in context and context['domain'] in agent.capabilities:
            expertise_weight = agent.capabilities[context['domain']] * 0.5
            
        # 基于历史决策表现的权重
        performance_weight = self._get_decision_performance(agent.id, context.get('type', 'general')) * 0.3
        
        return min(1.0, base_weight + expertise_weight + performance_weight)
    
    def _get_decision_performance(self, agent_id: str, decision_type: str) -> float:
        """获取决策表现分数"""
        # 在 individual_votes 中查找该智能体的决策记录
        agent_votes = []
        for decision_record in self.decision_history:
            if decision_record.get('type') == decision_type:
                # 在 individual_votes 中查找该智能体
                for vote_record in decision_record.get('individual_votes', []):
                    if vote_record.get('agent_id') == agent_id:
                        agent_votes.append(vote_record)
        
        if not agent_votes:
            return 0.5
            
        success_rate = np.mean([vote.get('success', False) for vote in agent_votes])
        return success_rate
    
    def collective_decision(self, agents: List[SocialAgent], proposal: Any, 
                          decision_type: str = 'general', domain: str = 'general') -> Dict[str, Any]:
        """集体决策"""
        votes = {}
        vote_details = []
        
        # 计算每个智能体的投票权重
        context = {'type': decision_type, 'domain': domain}
        
        for agent in agents:
            weight = self.calculate_voting_weight(agent, context)
            vote = self._get_agent_vote(agent, proposal, decision_type)
            votes[agent.id] = {
                'vote': vote,
                'weight': weight,
                'confidence': self._get_confidence(agent, proposal)
            }
            vote_details.append({
                'agent_id': agent.id,
                'vote': vote,
                'weight': weight,
                'confidence': votes[agent.id]['confidence']
            })
        
        # 权重投票
        weighted_votes = []
        for agent_id, vote_info in votes.items():
            if vote_info['vote'] == 'yes':
                weighted_votes.append(vote_info['weight'])
            elif vote_info['vote'] == 'no':
                weighted_votes.append(-vote_info['weight'])
            else:  # abstain
                weighted_votes.append(0)
        
        total_weight = sum(votes[v]['weight'] for v in votes)
        yes_weight = sum(w for i, w in enumerate(weighted_votes) if votes[list(votes.keys())[i]]['vote'] == 'yes')
        no_weight = abs(sum(w for i, w in enumerate(weighted_votes) if votes[list(votes.keys())[i]]['vote'] == 'no'))
        
        # 计算结果
        if total_weight == 0:
            decision = 'rejected'
            confidence = 0.0
        else:
            consensus_score = (yes_weight - no_weight) / total_weight
            if abs(consensus_score) >= self.consensus_threshold:
                decision = 'approved' if consensus_score > 0 else 'rejected'
                confidence = min(1.0, abs(consensus_score))
            else:
                decision = 'no_consensus'
                confidence = abs(consensus_score)
        
        # 创建个人决策记录
        individual_votes = []
        for agent in agents:
            agent_vote_info = votes.get(agent.id, {'vote': 'abstain', 'weight': 0.0})
            individual_votes.append({
                'agent_id': agent.id,
                'vote': agent_vote_info['vote'],
                'weight': agent_vote_info['weight'],
                'success': decision == 'approved'  # 简化的成功定义
            })
        
        decision_record = {
            'timestamp': time.time(),
            'proposal': proposal,
            'type': decision_type,
            'domain': domain,
            'decision': decision,
            'confidence': confidence,
            'vote_details': vote_details,
            'individual_votes': individual_votes,
            'consensus_score': consensus_score if 'consensus_score' in locals() else 0,
            'participants': len(agents)
        }
        
        self.decision_history.append(decision_record)
        return decision_record
    
    def _get_agent_vote(self, agent: SocialAgent, proposal: Any, decision_type: str) -> str:
        """获取智能体投票"""
        # 简化的投票逻辑，基于能力和提案的匹配度
        proposal_requirements = getattr(proposal, 'requirements', {})
        agent_fit_score = 0.0
        
        for req_skill, req_level in proposal_requirements.items():
            if req_skill in agent.capabilities:
                agent_fit_score += min(agent.capabilities[req_skill], req_level)
        
        if agent_fit_score >= len(proposal_requirements):
            return 'yes'
        elif agent_fit_score >= len(proposal_requirements) * 0.5:
            return 'abstain'
        else:
            return 'no'
    
    def _get_confidence(self, agent: SocialAgent, proposal: Any) -> float:
        """获取置信度"""
        return min(1.0, agent.get_influence_score() + random.uniform(0, 0.3))

class CulturalEvolution:
    """文化进化管理器"""
    
    def __init__(self):
        self.cultural_artifacts: Dict[str, CulturalArtifact] = {}
        self.diffusion_network: Dict[str, Set[str]] = defaultdict(set)  # 知识扩散网络
        self.cultural_clusters: Dict[str, Set[str]] = defaultdict(set)  # 文化集群
        self.adoption_tracking: Dict[str, List[Dict]] = defaultdict(list)  # 采纳追踪
        
    def create_cultural_artifact(self, creators: List[str], artifact_type: str, 
                               content: Any, context: Dict[str, Any]) -> CulturalArtifact:
        """创建文化产物"""
        artifact_id = str(uuid.uuid4())
        artifact = CulturalArtifact(
            id=artifact_id,
            type=artifact_type,
            content=content,
            creators=creators,
            adoption_count=0,
            effectiveness_score=0.0,
            created_time=time.time(),
            last_update_time=time.time()
        )
        
        self.cultural_artifacts[artifact_id] = artifact
        
        # 更新扩散网络
        for creator in creators:
            self.diffusion_network[creator].add(artifact_id)
            
        # 计算初始有效性分数
        self._calculate_effectiveness(artifact, context)
        
        logger.info(f"创建新的文化产物: {artifact_id}, 类型: {artifact_type}")
        return artifact
    
    def diffuse_cultural_knowledge(self, knowledge_holder: str, target_agent: str, 
                                 artifact_id: str) -> bool:
        """文化知识扩散"""
        if artifact_id not in self.cultural_artifacts:
            return False
            
        artifact = self.cultural_artifacts[artifact_id]
        
        # 基于社交连接强度决定传播概率
        diffusion_probability = self._calculate_diffusion_probability(knowledge_holder, target_agent, artifact)
        
        if random.random() < diffusion_probability:
            # 记录采纳事件
            adoption_event = {
                'artifact_id': artifact_id,
                'from_agent': knowledge_holder,
                'to_agent': target_agent,
                'timestamp': time.time(),
                'success': True
            }
            self.adoption_tracking[artifact_id].append(adoption_event)
            
            # 更新文化产物
            artifact.adoption_count += 1
            artifact.last_update_time = time.time()
            
            # 更新扩散网络
            self.diffusion_network[target_agent].add(artifact_id)
            
            # 更新文化集群
            self._update_cultural_clusters(artifact_id)
            
            logger.debug(f"文化知识扩散成功: {knowledge_holder} -> {target_agent}, 产物: {artifact_id}")
            return True
        
        return False
    
    def evaluate_cultural_fitness(self, artifact_id: str, outcomes: List[Dict]) -> float:
        """评估文化适应性"""
        artifact = self.cultural_artifacts.get(artifact_id)
        if not artifact:
            return 0.0
        
        # 基于结果计算适应性
        success_rate = np.mean([1 if outcome.get('success', False) else 0 for outcome in outcomes])
        efficiency_score = np.mean([outcome.get('efficiency', 0.5) for outcome in outcomes])
        innovation_score = np.mean([outcome.get('innovation', 0.5) for outcome in outcomes])
        
        # 加权综合评分
        fitness = (success_rate * 0.5 + efficiency_score * 0.3 + innovation_score * 0.2)
        
        artifact.effectiveness_score = fitness
        
        # 基于适应性决定是否保留或淘汰
        if fitness < 0.3:  # 低适应性文化产物
            self._eliminate_cultural_artifact(artifact_id)
        
        return fitness
    
    def _calculate_diffusion_probability(self, knowledge_holder: str, target_agent: str, 
                                       artifact: CulturalArtifact) -> float:
        """计算扩散概率"""
        base_probability = 0.1
        
        # 基于连接强度
        # 这里需要根据具体的社交网络信息调整
        connection_strength = 0.5  # 默认连接强度
        
        # 基于文化产物类型
        type_bonus = {
            'skill': 0.3,
            'norm': 0.2,
            'belief': 0.15,
            'protocol': 0.25
        }.get(artifact.type, 0.1)
        
        # 基于时间因子（越新扩散概率越高）
        time_factor = min(1.0, (time.time() - artifact.created_time) / (24 * 3600))  # 24小时衰减
        time_bonus = 0.2 * (1 - time_factor)
        
        probability = min(0.8, base_probability + connection_strength * 0.4 + type_bonus + time_bonus)
        return probability
    
    def _calculate_effectiveness(self, artifact: CulturalArtifact, context: Dict[str, Any]):
        """计算有效性分数"""
        # 基于上下文和产物类型计算初始有效性
        base_effectiveness = {
            'skill': 0.7,
            'norm': 0.6,
            'belief': 0.5,
            'protocol': 0.65
        }.get(artifact.type, 0.5)
        
        # 根据创建者能力调整
        creator_bonus = min(0.3, len(artifact.creators) * 0.1)
        
        artifact.effectiveness_score = base_effectiveness + creator_bonus
    
    def _update_cultural_clusters(self, artifact_id: str):
        """更新文化集群"""
        # 基于相似的文化产物形成集群
        artifact = self.cultural_artifacts[artifact_id]
        cluster_key = f"{artifact.type}_{len(artifact.content) % 10}"  # 简化的聚类
        
        self.cultural_clusters[cluster_key].add(artifact_id)
    
    def _eliminate_cultural_artifact(self, artifact_id: str):
        """淘汰低效文化产物"""
        if artifact_id in self.cultural_artifacts:
            del self.cultural_artifacts[artifact_id]
            logger.info(f"淘汰低效文化产物: {artifact_id}")

class SocialLearningSystem:
    """社会学习系统"""
    
    def __init__(self):
        self.learning_strategies = {
            'imitation': self._imitation_learning,
            'innovation': self._innovation_learning,
            'collaboration': self._collaborative_learning,
            'competition': self._competitive_learning
        }
        self.learning_history = []
        self.performance_tracking = defaultdict(list)
        
    def social_learn(self, learner: SocialAgent, teacher: SocialAgent, 
                    knowledge_node: KnowledgeNode, strategy: str = 'imitation') -> Dict[str, Any]:
        """社会学习"""
        if strategy not in self.learning_strategies:
            strategy = 'imitation'
        
        learning_result = self.learning_strategies[strategy](learner, teacher, knowledge_node)
        
        # 记录学习过程
        learning_record = {
            'learner_id': learner.id,
            'teacher_id': teacher.id,
            'knowledge_id': knowledge_node.id,
            'strategy': strategy,
            'success': learning_result['success'],
            'improvement': learning_result['improvement'],
            'timestamp': time.time()
        }
        
        self.learning_history.append(learning_record)
        
        # 更新表现追踪
        self.performance_tracking[learner.id].append({
            'strategy': strategy,
            'performance_gain': learning_result['improvement'],
            'timestamp': time.time()
        })
        
        return learning_result
    
    def _imitation_learning(self, learner: SocialAgent, teacher: SocialAgent, 
                          knowledge_node: KnowledgeNode) -> Dict[str, Any]:
        """模仿学习"""
        # 基于信任度和相似性计算学习成功率
        trust_level = 0.5  # 需要从社交连接中获取
        similarity = self._calculate_agent_similarity(learner, teacher)
        
        success_probability = (trust_level + similarity) / 2
        
        if random.random() < success_probability:
            # 成功学习
            improvement = knowledge_node.confidence * teacher.get_influence_score() * 0.1
            learner.update_capabilities({skill: improvement * 0.1 for skill in learner.capabilities.keys()})
            
            # 学习文化产物
            learner.learned_cultural_artifacts.add(knowledge_node.id)
            
            return {'success': True, 'improvement': improvement}
        
        return {'success': False, 'improvement': 0.0}
    
    def _innovation_learning(self, learner: SocialAgent, teacher: SocialAgent, 
                           knowledge_node: KnowledgeNode) -> Dict[str, Any]:
        """创新学习"""
        # 创新学习基于现有知识的组合和改进
        base_improvement = 0.05
        innovation_factor = learner.personality.get('creativity', 0.5)
        
        # 随机创新
        if random.random() < innovation_factor:
            improvement = base_improvement * (1 + innovation_factor)
            learner.update_capabilities({skill: improvement * random.uniform(0.5, 1.5) 
                                       for skill in learner.capabilities.keys()})
            return {'success': True, 'improvement': improvement}
        
        return {'success': False, 'improvement': 0.0}
    
    def _collaborative_learning(self, learner: SocialAgent, teacher: SocialAgent, 
                              knowledge_node: KnowledgeNode) -> Dict[str, Any]:
        """协作学习"""
        # 协作学习通过共享和讨论增强理解
        collaboration_bonus = 0.1
        mutual_understanding = self._calculate_agent_similarity(learner, teacher)
        
        if mutual_understanding > 0.6:
            improvement = collaboration_bonus + mutual_understanding * 0.05
            learner.update_capabilities({skill: improvement for skill in learner.capabilities.keys()})
            return {'success': True, 'improvement': improvement}
        
        return {'success': False, 'improvement': 0.0}
    
    def _competitive_learning(self, learner: SocialAgent, teacher: SocialAgent, 
                            knowledge_node: KnowledgeNode) -> Dict[str, Any]:
        """竞争学习"""
        # 竞争学习通过挑战现有知识促进进步
        competition_factor = 0.08
        challenge_acceptance = learner.personality.get('competitiveness', 0.5)
        
        if random.random() < challenge_acceptance:
            improvement = competition_factor * challenge_acceptance
            learner.update_capabilities({skill: improvement * random.uniform(0.8, 1.2) 
                                       for skill in learner.capabilities.keys()})
            return {'success': True, 'improvement': improvement}
        
        return {'success': False, 'improvement': 0.0}
    
    def _calculate_agent_similarity(self, agent1: SocialAgent, agent2: SocialAgent) -> float:
        """计算智能体相似性"""
        # 基于能力和性格的相似性
        capability_similarity = np.mean([abs(agent1.capabilities.get(k, 0) - agent2.capabilities.get(k, 0))
                                       for k in set(agent1.capabilities.keys()) | set(agent2.capabilities.keys())])
        
        personality_similarity = np.mean([abs(agent1.personality.get(k, 0) - agent2.personality.get(k, 0))
                                        for k in set(agent1.personality.keys()) | set(agent2.personality.keys())])
        
        # 转换为相似度（0-1之间）
        similarity = 1.0 - (capability_similarity + personality_similarity) / 2
        return max(0.0, min(1.0, similarity))

class NetworkLayerManager:
    """网络层级管理器"""
    
    def __init__(self):
        self.layers = {
            NetworkLayer.INDIVIDUAL: [],
            NetworkLayer.GROUP: [],
            NetworkLayer.COMMUNITY: [],
            NetworkLayer.SOCIETY: [],
            NetworkLayer.META_SOCIETY: []
        }
        self.layer_connections = defaultdict(list)
        self.hierarchy_mapping = {}
        
    def organize_hierarchy(self, agents: List[SocialAgent]) -> Dict[str, str]:
        """组织层次结构"""
        # 按能力和影响力对智能体进行排序和分层
        sorted_agents = sorted(agents, key=lambda a: (a.get_influence_score(), a.experience_level), reverse=True)
        
        total_agents = len(sorted_agents)
        
        # 分层分配
        self.layers[NetworkLayer.META_SOCIETY] = sorted_agents[:max(1, total_agents // 100)]
        self.layers[NetworkLayer.SOCIETY] = sorted_agents[max(1, total_agents // 100):total_agents // 20]
        self.layers[NetworkLayer.COMMUNITY] = sorted_agents[total_agents // 20:total_agents // 4]
        self.layers[NetworkLayer.GROUP] = sorted_agents[total_agents // 4:total_agents // 2]
        self.layers[NetworkLayer.INDIVIDUAL] = sorted_agents[total_agents // 2:]
        
        # 创建层级映射
        hierarchy_mapping = {}
        for layer, layer_agents in self.layers.items():
            for agent in layer_agents:
                hierarchy_mapping[agent.id] = layer.value
        
        self.hierarchy_mapping = hierarchy_mapping
        
        # 建立层级间的连接
        self._establish_layer_connections()
        
        return hierarchy_mapping
    
    def _establish_layer_connections(self):
        """建立层级间连接"""
        self.layer_connections = {}
        
        # 高层级与低层级的连接
        connections = [
            (NetworkLayer.META_SOCIETY, NetworkLayer.SOCIETY),
            (NetworkLayer.SOCIETY, NetworkLayer.COMMUNITY),
            (NetworkLayer.COMMUNITY, NetworkLayer.GROUP),
            (NetworkLayer.GROUP, NetworkLayer.INDIVIDUAL)
        ]
        
        for higher_layer, lower_layer in connections:
            if self.layers[higher_layer] and self.layers[lower_layer]:
                # 每个高层智能体连接到多个低层智能体
                for high_agent in self.layers[higher_layer]:
                    connections_made = 0
                    max_connections = min(10, len(self.layers[lower_layer]))
                    
                    for low_agent in random.sample(self.layers[lower_layer], max_connections):
                        if connections_made < max_connections:
                            high_agent.add_social_connection(low_agent, 'mentorship', random.uniform(0.6, 0.9))
                            connections_made += 1
    
    def get_agents_by_layer(self, layer: NetworkLayer) -> List[SocialAgent]:
        """获取指定层级的智能体"""
        return self.layers.get(layer, [])
    
    def reorganize_layers(self, agents: List[SocialAgent], criteria: str = 'performance'):
        """重新组织层级"""
        if criteria == 'performance':
            sorted_agents = sorted(agents, key=lambda a: a.get_influence_score(), reverse=True)
        elif criteria == 'experience':
            sorted_agents = sorted(agents, key=lambda a: a.experience_level, reverse=True)
        else:  # random
            sorted_agents = agents.copy()
            random.shuffle(sorted_agents)
        
        self.organize_hierarchy(sorted_agents)
        logger.info(f"重新组织层级，标准: {criteria}")

class MassEvolutionSystem:
    """大规模多智能体协同进化系统主类"""
    
    def __init__(self, num_agents: int = 5000, initial_capabilities: Dict[str, List[float]] = None):
        self.num_agents = num_agents
        self.agents: Dict[str, SocialAgent] = {}
        self.decision_maker = DecisionMaker()
        self.cultural_evolution = CulturalEvolution()
        self.social_learning = SocialLearningSystem()
        self.network_manager = NetworkLayerManager()
        
        # 系统统计
        self.evolution_generation = 0
        self.system_metrics = {
            'average_fitness': 0.0,
            'diversity_index': 0.0,
            'collaboration_rate': 0.0,
            'innovation_rate': 0.0,
            'cultural_diffusion_speed': 0.0
        }
        
        # 线程池用于并发处理
        self.executor = ThreadPoolExecutor(max_workers=16)
        
        # 初始化智能体
        self._initialize_agents(initial_capabilities)
        
        logger.info(f"初始化大规模多智能体系统，智能体数量: {num_agents}")
    
    def _initialize_agents(self, initial_capabilities: Dict[str, List[float]] = None):
        """初始化智能体"""
        if initial_capabilities is None:
            initial_capabilities = {
                'reasoning': [0.3, 0.9],
                'creativity': [0.2, 0.8],
                'collaboration': [0.4, 0.9],
                'learning': [0.3, 0.8],
                'adaptability': [0.3, 0.7],
                'communication': [0.4, 0.8]
            }
        
        agent_types = ['specialist', 'generalist', 'innovator', 'collaborator']
        
        for i in range(self.num_agents):
            agent_id = f"agent_{i:05d}"
            
            # 随机选择能力范围
            capabilities = {}
            for skill, (min_val, max_val) in initial_capabilities.items():
                capabilities[skill] = random.uniform(min_val, max_val)
            
            # 调整能力以符合智能体类型
            agent_type = random.choice(agent_types)
            if agent_type == 'specialist':
                # 专家型：某项能力特别强
                main_skill = random.choice(list(capabilities.keys()))
                capabilities[main_skill] = random.uniform(0.8, 1.0)
                for skill in capabilities:
                    if skill != main_skill:
                        capabilities[skill] *= random.uniform(0.3, 0.6)
            
            elif agent_type == 'innovator':
                # 创新型：创造力和适应性较强
                capabilities['creativity'] = random.uniform(0.7, 1.0)
                capabilities['adaptability'] = random.uniform(0.6, 0.9)
            
            elif agent_type == 'collaborator':
                # 协作型：协作和沟通能力较强
                capabilities['collaboration'] = random.uniform(0.7, 1.0)
                capabilities['communication'] = random.uniform(0.7, 1.0)
            
            # 生成性格特征
            personality = {
                'openness': random.uniform(0.3, 0.9),
                'conscientiousness': random.uniform(0.4, 0.8),
                'extraversion': random.uniform(0.3, 0.8),
                'agreeableness': random.uniform(0.5, 0.9),
                'neuroticism': random.uniform(0.1, 0.6),
                'creativity': capabilities['creativity'],
                'competitiveness': random.uniform(0.3, 0.8)
            }
            
            agent = SocialAgent(agent_id, capabilities, personality)
            self.agents[agent_id] = agent
        
        # 建立初始社交网络
        self._establish_initial_network()
        
        # 组织层级结构
        self.network_manager.organize_hierarchy(list(self.agents.values()))
    
    def _establish_initial_network(self):
        """建立初始社交网络"""
        agent_list = list(self.agents.values())
        
        # 每个智能体连接几个其他智能体
        for agent in agent_list:
            num_connections = random.randint(3, 12)
            potential_connections = [a for a in agent_list if a.id != agent.id]
            
            connected = 0
            for other_agent in random.sample(potential_connections, min(num_connections, len(potential_connections))):
                # 基于相似性和能力互补性建立连接
                similarity = self.social_learning._calculate_agent_similarity(agent, other_agent)
                complementarity = 1 - similarity  # 互补性
                
                if similarity > 0.3 or complementarity > 0.6:
                    connection_type = 'collaboration' if complementarity > 0.6 else 'influence'
                    strength = max(similarity, complementarity) * random.uniform(0.4, 0.8)
                    agent.add_social_connection(other_agent, connection_type, strength)
                    connected += 1
                    
                    # 相互连接
                    if connected < num_connections * 0.5:
                        other_agent.add_social_connection(agent, connection_type, strength)
    
    def run_evolution_cycle(self, num_cycles: int = 100) -> Dict[str, Any]:
        """运行进化周期"""
        cycle_results = []
        
        for cycle in range(num_cycles):
            cycle_start_time = time.time()
            
            # 1. 社会学习和知识传播
            learning_results = self._social_learning_phase()
            
            # 2. 协作和集体决策
            decision_results = self._collaboration_phase()
            
            # 3. 文化进化和传播
            cultural_results = self._cultural_evolution_phase()
            
            # 4. 性能评估和选择
            fitness_results = self._fitness_evaluation_phase()
            
            # 5. 网络重组和优化
            network_results = self._network_reorganization_phase()
            
            cycle_time = time.time() - cycle_start_time
            
            cycle_result = {
                'cycle': cycle,
                'timestamp': cycle_start_time,
                'duration': cycle_time,
                'learning_results': learning_results,
                'decision_results': decision_results,
                'cultural_results': cultural_results,
                'fitness_results': fitness_results,
                'network_results': network_results,
                'system_metrics': self._calculate_system_metrics()
            }
            
            cycle_results.append(cycle_result)
            
            if cycle % 10 == 0:
                logger.info(f"完成进化周期 {cycle}/{num_cycles}, 系统指标: {self.system_metrics}")
        
        # 运行完整的进化过程
        self._complete_evolution_process(num_cycles)
        
        return {
            'cycles': cycle_results,
            'final_system_state': self._get_system_state(),
            'evolution_summary': self._generate_evolution_summary(cycle_results)
        }
    
    def _social_learning_phase(self) -> Dict[str, Any]:
        """社会学习阶段"""
        learning_events = []
        
        # 随机选择学习对
        agents_list = list(self.agents.values())
        num_learning_pairs = min(1000, len(agents_list) // 2)  # 限制学习对数量
        
        for _ in range(num_learning_pairs):
            teacher, learner = random.sample(agents_list, 2)
            
            # 选择学习的知识节点
            if teacher.knowledge_base:
                knowledge_node = random.choice(list(teacher.knowledge_base.values()))
                strategy = random.choice(list(self.social_learning.learning_strategies.keys()))
                
                result = self.social_learning.social_learn(learner, teacher, knowledge_node, strategy)
                if result['success']:
                    learning_events.append(result)
        
        return {
            'learning_events': learning_events,
            'success_rate': len(learning_events) / max(1, num_learning_pairs),
            'average_improvement': np.mean([event['improvement'] for event in learning_events]) if learning_events else 0
        }
    
    def _collaboration_phase(self) -> Dict[str, Any]:
        """协作和集体决策阶段"""
        collaboration_events = []
        decision_events = []
        
        # 创建协作项目
        num_projects = min(500, len(self.agents) // 10)
        
        for _ in range(num_projects):
            # 选择协作团队
            team_size = random.randint(3, 15)
            team_agents = random.sample(list(self.agents.values()), min(team_size, len(self.agents)))
            
            # 模拟项目提案
            project_requirements = random.choice([
                {'reasoning': 0.7, 'collaboration': 0.6},
                {'creativity': 0.8, 'communication': 0.7},
                {'learning': 0.6, 'adaptability': 0.8}
            ])
            
            # 集体决策
            proposal = type('Project', (), {'requirements': project_requirements})()
            decision_result = self.decision_maker.collective_decision(
                team_agents, proposal, 'project_approval', 'collaboration'
            )
            
            if decision_result['decision'] == 'approved':
                collaboration_events.append({
                    'team_size': len(team_agents),
                    'project_type': 'collaborative_project',
                    'success': True,
                    'participants': [agent.id for agent in team_agents]
                })
            
            decision_events.append(decision_result)
        
        return {
            'collaboration_events': collaboration_events,
            'decision_events': decision_events,
            'approval_rate': len([e for e in decision_events if e['decision'] == 'approved']) / len(decision_events)
        }
    
    def _cultural_evolution_phase(self) -> Dict[str, Any]:
        """文化进化阶段"""
        cultural_events = []
        
        # 创建新的文化产物
        num_artifacts = min(200, len(self.agents) // 25)
        
        for _ in range(num_artifacts):
            # 选择创建者（通常是表现好的智能体）
            creators = random.sample(list(self.agents.values()), random.randint(1, 3))
            creator_ids = [agent.id for agent in creators]
            
            artifact_type = random.choice(['skill', 'norm', 'belief', 'protocol'])
            content = self._generate_artifact_content(artifact_type)
            
            artifact = self.cultural_evolution.create_cultural_artifact(
                creator_ids, artifact_type, content, {}
            )
            
            cultural_events.append({
                'artifact_id': artifact.id,
                'type': artifact_type,
                'creators': creator_ids,
                'initial_effectiveness': artifact.effectiveness_score
            })
        
        # 文化传播
        diffusion_events = []
        for artifact_id, artifact in list(self.cultural_evolution.cultural_artifacts.items())[:100]:
            for creator_id in artifact.creators:
                # 选择传播目标
                connected_agents = [aid for aid, conn in self.agents[creator_id].social_connections.items()]
                if connected_agents:
                    target_id = random.choice(connected_agents)
                    success = self.cultural_evolution.diffuse_cultural_knowledge(
                        creator_id, target_id, artifact_id
                    )
                    if success:
                        diffusion_events.append({
                            'artifact_id': artifact_id,
                            'from': creator_id,
                            'to': target_id
                        })
        
        return {
            'cultural_events': cultural_events,
            'diffusion_events': diffusion_events,
            'total_artifacts': len(self.cultural_evolution.cultural_artifacts),
            'diffusion_success_rate': len(diffusion_events) / max(1, len(self.cultural_evolution.cultural_artifacts) * 2)
        }
    
    def _fitness_evaluation_phase(self) -> Dict[str, Any]:
        """性能评估阶段"""
        fitness_scores = {}
        
        for agent_id, agent in self.agents.items():
            # 综合评估智能体性能
            capability_score = np.mean(list(agent.capabilities.values()))
            influence_score = agent.get_influence_score()
            experience_score = agent.experience_level
            
            # 综合适应度
            fitness = (capability_score * 0.4 + influence_score * 0.3 + experience_score * 0.3)
            fitness_scores[agent_id] = fitness
        
        # 记录适应度分数
        self.system_metrics['average_fitness'] = np.mean(list(fitness_scores.values()))
        
        # 基于适应度进行选择
        selected_agents = self._selection_process(fitness_scores)
        
        return {
            'fitness_scores': fitness_scores,
            'selected_agents': selected_agents,
            'average_fitness': self.system_metrics['average_fitness'],
            'diversity': self._calculate_diversity()
        }
    
    def _network_reorganization_phase(self) -> Dict[str, Any]:
        """网络重组阶段"""
        # 定期重组网络
        if self.evolution_generation % 10 == 0:
            self.network_manager.reorganize_layers(list(self.agents.values()), 'performance')
        
        # 更新社交连接强度
        connection_updates = 0
        for agent in self.agents.values():
            for connection in list(agent.social_connections.values()):
                # 基于互动历史更新连接强度
                if connection.interaction_history:
                    recent_success_rate = np.mean([h.get('success', 0.5) for h in connection.interaction_history[-5:]])
                    connection.strength = (connection.strength * 0.8 + recent_success_rate * 0.2)
                    connection_updates += 1
        
        return {
            'connection_updates': connection_updates,
            'hierarchy_layers': {layer.value: len(agents) for layer, agents in self.network_manager.layers.items()},
            'total_connections': sum(len(agent.social_connections) for agent in self.agents.values())
        }
    
    def _selection_process(self, fitness_scores: Dict[str, float]) -> Set[str]:
        """选择过程"""
        # 选择前20%的智能体作为优秀个体
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        selection_rate = 0.2
        num_selected = int(len(sorted_agents) * selection_rate)
        
        selected_ids = {agent_id for agent_id, _ in sorted_agents[:num_selected]}
        
        return selected_ids
    
    def _calculate_diversity(self) -> float:
        """计算多样性指数"""
        capabilities_matrix = np.array([list(agent.capabilities.values()) for agent in self.agents.values()])
        
        # 使用方差作为多样性指标
        diversity = np.mean(np.var(capabilities_matrix, axis=0))
        self.system_metrics['diversity_index'] = diversity
        return diversity
    
    def _calculate_system_metrics(self) -> Dict[str, float]:
        """计算系统指标"""
        # 协作率
        total_connections = sum(len(agent.social_connections) for agent in self.agents.values())
        possible_connections = len(self.agents) * (len(self.agents) - 1) / 2
        collaboration_rate = min(1.0, total_connections / possible_connections) if possible_connections > 0 else 0
        
        # 创新率
        innovation_rate = len(self.cultural_evolution.cultural_artifacts) / len(self.agents)
        
        # 文化传播速度
        total_adoptions = sum(len(adoptions) for adoptions in self.cultural_evolution.adoption_tracking.values())
        cultural_diffusion_speed = total_adoptions / max(1, self.evolution_generation)
        
        self.system_metrics.update({
            'collaboration_rate': collaboration_rate,
            'innovation_rate': innovation_rate,
            'cultural_diffusion_speed': cultural_diffusion_speed
        })
        
        return self.system_metrics.copy()
    
    def _generate_artifact_content(self, artifact_type: str) -> Any:
        """生成文化产物内容"""
        if artifact_type == 'skill':
            return {
                'name': f"Skill_{random.randint(1000, 9999)}",
                'type': 'procedure',
                'steps': random.randint(3, 10),
                'complexity': random.uniform(0.3, 0.9)
            }
        elif artifact_type == 'norm':
            return {
                'rule': f"Social_Norm_{random.randint(1000, 9999)}",
                'scope': random.choice(['local', 'group', 'community']),
                'enforcement': random.uniform(0.4, 0.9)
            }
        elif artifact_type == 'belief':
            return {
                'statement': f"Shared_Belief_{random.randint(1000, 9999)}",
                'confidence': random.uniform(0.5, 0.9),
                'evidence': random.randint(1, 5)
            }
        elif artifact_type == 'protocol':
            return {
                'name': f"Protocol_{random.randint(1000, 9999)}",
                'participants': random.randint(2, 8),
                'steps': random.randint(4, 12),
                'success_rate': random.uniform(0.6, 0.95)
            }
        else:
            return {"content": f"Artifact_{random.randint(1000, 9999)}"}
    
    def _complete_evolution_process(self, num_cycles: int):
        """完整的进化过程"""
        logger.info("开始完整的进化过程...")
        
        # 模拟多代进化
        for generation in range(5):  # 5个主要进化周期
            self.evolution_generation = generation
            
            # 创建后代
            self._reproduce_agents(generation)
            
            # 变异
            self._mutate_agents()
            
            # 环境适应
            self._adapt_to_environment()
            
            logger.info(f"完成第 {generation} 代进化，剩余智能体数量: {len(self.agents)}")
        
        # 最终评估
        final_fitness = self._final_evaluation()
        logger.info(f"进化过程完成，最终平均适应度: {final_fitness}")
    
    def _reproduce_agents(self, generation: int):
        """繁殖智能体"""
        # 基于适应度选择父母
        fitness_scores = {agent.id: agent.get_influence_score() for agent in self.agents.values()}
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前30%作为父母池
        parent_pool_size = max(10, len(sorted_agents) // 3)
        parent_pool = [agent_id for agent_id, _ in sorted_agents[:parent_pool_size]]
        
        # 生成新智能体
        num_offspring = min(len(self.agents) // 4, 1000)  # 最多生成1000个后代
        
        for i in range(num_offspring):
            # 选择父母
            parent1_id = random.choice(parent_pool)
            parent2_id = random.choice(parent_pool)
            
            parent1 = self.agents[parent1_id]
            parent2 = self.agents[parent2_id]
            
            # 创建后代
            offspring_id = f"gen_{generation}_agent_{i:05d}"
            offspring = self._create_offspring(parent1, parent2, offspring_id)
            
            # 添加到种群
            self.agents[offspring_id] = offspring
        
        logger.info(f"繁殖了 {num_offspring} 个新智能体")
    
    def _create_offspring(self, parent1: SocialAgent, parent2: SocialAgent, offspring_id: str) -> SocialAgent:
        """创建后代"""
        # 能力遗传（交叉）
        capabilities = {}
        for skill in parent1.capabilities.keys():
            if skill in parent2.capabilities:
                # 中值遗传
                capabilities[skill] = (parent1.capabilities[skill] + parent2.capabilities[skill]) / 2
            else:
                capabilities[skill] = parent1.capabilities[skill]
        
        # 性格遗传
        personality = {}
        for trait in parent1.personality.keys():
            if trait in parent2.personality:
                personality[trait] = (parent1.personality[trait] + parent2.personality[trait]) / 2
            else:
                personality[trait] = parent1.personality[trait]
        
        offspring = SocialAgent(offspring_id, capabilities, personality)
        offspring.generation = max(parent1.generation, parent2.generation) + 1
        
        return offspring
    
    def _mutate_agents(self):
        """智能体变异"""
        mutation_rate = 0.1
        
        for agent in self.agents.values():
            # 能力变异
            for skill in agent.capabilities:
                if random.random() < mutation_rate:
                    mutation_strength = random.uniform(-0.1, 0.1)
                    agent.capabilities[skill] = max(0.0, min(1.0, agent.capabilities[skill] + mutation_strength))
            
            # 性格变异
            for trait in agent.personality:
                if random.random() < mutation_rate:
                    mutation_strength = random.uniform(-0.05, 0.05)
                    agent.personality[trait] = max(0.0, min(1.0, agent.personality[trait] + mutation_strength))
    
    def _adapt_to_environment(self):
        """环境适应"""
        # 模拟环境变化对智能体的影响
        environmental_pressure = random.uniform(0.8, 1.2)
        
        for agent in self.agents.values():
            # 适应性强的智能体表现更好
            adaptability = agent.capabilities.get('adaptability', 0.5)
            performance_multiplier = adaptability * environmental_pressure
            
            # 更新经验和声誉
            agent.experience_level *= performance_multiplier
            agent.reputation_score = min(1.0, agent.reputation_score + performance_multiplier * 0.01)
    
    def _final_evaluation(self) -> float:
        """最终评估"""
        total_fitness = 0.0
        
        for agent in self.agents.values():
            agent_fitness = agent.get_influence_score()
            total_fitness += agent_fitness
        
        average_fitness = total_fitness / len(self.agents) if self.agents else 0.0
        return average_fitness
    
    def _get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'num_agents': len(self.agents),
            'evolution_generation': self.evolution_generation,
            'system_metrics': self.system_metrics.copy(),
            'network_layers': {layer.value: len(agents) for layer, agents in self.network_manager.layers.items()},
            'cultural_artifacts_count': len(self.cultural_evolution.cultural_artifacts),
            'total_learning_events': len(self.social_learning.learning_history),
            'total_decisions': len(self.decision_maker.decision_history)
        }
    
    def _generate_evolution_summary(self, cycle_results: List[Dict]) -> Dict[str, Any]:
        """生成进化摘要"""
        if not cycle_results:
            return {}
        
        # 计算趋势
        fitness_trend = [result['system_metrics']['average_fitness'] for result in cycle_results]
        diversity_trend = [result['system_metrics']['diversity_index'] for result in cycle_results]
        
        summary = {
            'total_cycles': len(cycle_results),
            'average_fitness_improvement': fitness_trend[-1] - fitness_trend[0] if len(fitness_trend) > 1 else 0,
            'diversity_change': diversity_trend[-1] - diversity_trend[0] if len(diversity_trend) > 1 else 0,
            'final_system_state': cycle_results[-1]['system_metrics'],
            'learning_efficiency': np.mean([result['learning_results']['success_rate'] for result in cycle_results]),
            'collaboration_effectiveness': np.mean([result['decision_results']['approval_rate'] for result in cycle_results]),
            'cultural_evolution_rate': np.mean([result['cultural_results']['diffusion_success_rate'] for result in cycle_results]),
            'total_evolution_time': sum(result['duration'] for result in cycle_results)
        }
        
        return summary
    
    def save_system_state(self, filepath: str):
        """保存系统状态"""
        system_state = {
            'agents': {aid: {
                'id': agent.id,
                'capabilities': agent.capabilities,
                'personality': agent.personality,
                'state': agent.state.value,
                'experience_level': agent.experience_level,
                'reputation_score': agent.reputation_score,
                'generation': agent.generation,
                'knowledge_base_size': len(agent.knowledge_base),
                'learned_cultural_artifacts_count': len(agent.learned_cultural_artifacts)
            } for aid, agent in self.agents.items()},
            'system_metrics': self.system_metrics,
            'evolution_generation': self.evolution_generation,
            'cultural_artifacts': {aid: {
                'type': artifact.type,
                'adoption_count': artifact.adoption_count,
                'effectiveness_score': artifact.effectiveness_score
            } for aid, artifact in self.cultural_evolution.cultural_artifacts.items()},
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"系统状态已保存到: {filepath}")
    
    def load_system_state(self, filepath: str):
        """加载系统状态"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                system_state = json.load(f)
            
            # 恢复系统状态（这里简化实现，实际需要更复杂的反序列化）
            self.system_metrics = system_state.get('system_metrics', {})
            self.evolution_generation = system_state.get('evolution_generation', 0)
            
            logger.info(f"系统状态已从 {filepath} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载系统状态失败: {e}")
            return False

def run_mass_evolution_demo():
    """运行大规模多智能体进化演示"""
    print("=" * 60)
    print("大规模多智能体协同进化系统演示")
    print("=" * 60)
    
    # 创建系统
    print("正在初始化大规模多智能体系统...")
    num_agents = 1000  # 可调整数量
    
    evolution_system = MassEvolutionSystem(num_agents=num_agents)
    
    print(f"成功创建 {num_agents} 个智能体")
    print(f"初始系统指标: {evolution_system.system_metrics}")
    
    # 运行进化周期
    print("\n开始运行进化周期...")
    evolution_results = evolution_system.run_evolution_cycle(num_cycles=50)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("进化结果摘要")
    print("=" * 60)
    
    summary = evolution_results['evolution_summary']
    print(f"总进化周期: {summary['total_cycles']}")
    print(f"适应度改善: {summary['average_fitness_improvement']:.4f}")
    print(f"多样性变化: {summary['diversity_change']:.4f}")
    print(f"学习效率: {summary['learning_efficiency']:.2%}")
    print(f"协作效果: {summary['collaboration_effectiveness']:.2%}")
    print(f"文化进化率: {summary['cultural_evolution_rate']:.2%}")
    print(f"总进化时间: {summary['total_evolution_time']:.2f} 秒")
    
    final_state = evolution_results['final_system_state']
    system_metrics = final_state['system_metrics']
    print(f"\n最终系统状态:")
    print(f"  平均适应度: {system_metrics.get('average_fitness', 0.0):.4f}")
    print(f"  协作率: {system_metrics.get('collaboration_rate', 0.0):.2%}")
    print(f"  创新率: {system_metrics.get('innovation_rate', 0.0):.4f}")
    print(f"  文化传播速度: {system_metrics.get('cultural_diffusion_speed', 0.0):.2f}")
    
    # 保存系统状态
    output_file = "/workspace/agents/mass_evolution/evolution_results.json"
    evolution_system.save_system_state(output_file)
    print(f"\n系统状态已保存到: {output_file}")
    
    return evolution_results

if __name__ == "__main__":
    # 运行演示
    results = run_mass_evolution_demo()
    
    print("\n" + "=" * 60)
    print("大规模多智能体协同进化系统演示完成！")
    print("=" * 60)