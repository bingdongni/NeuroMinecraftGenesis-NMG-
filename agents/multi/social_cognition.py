"""
社会认知能力系统 (Social Cognition System)
实现智能体间的意图识别、信任建立、社会学习和领导力选举

功能特性：
- 意图识别：理解其他智能体的目标和计划
- 信任建立：通过合作历史建立相互信任
- 社会学习：从其他智能体学习新技能
- 领导力选举：动态选举部落领导者
- 社交网络分析
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import math
import logging

# 配置日志
logger = logging.getLogger(__name__)

class IntentionType(Enum):
    """意图类型枚举"""
    EXPLORATION = "exploration"  # 探索
    CONSTRUCTION = "construction"  # 建造
    RESOURCE_COLLECTION = "resource_collection"  # 资源收集
    DEFENSE = "defense"  # 防御
    RESCUE = "rescue"  # 救援
    LEARNING = "learning"  # 学习
    LEADERSHIP = "leadership"  # 领导
    COLLABORATION = "collaboration"  # 合作

class TrustLevel(Enum):
    """信任等级枚举"""
    STRONG_MISTRUST = -3  # 强烈不信任
    MISTRUST = -2  # 不信任
    CAUTIOUS = -1  # 谨慎
    NEUTRAL = 0  # 中性
    MODERATE_TRUST = 1  # 中等信任
    HIGH_TRUST = 2  # 高度信任
    COMPLETE_TRUST = 3  # 完全信任

@dataclass
class SocialAction:
    """社会行为记录"""
    actor_id: str
    action_type: str
    target_id: str
    timestamp: datetime
    success: bool
    impact_score: float  # 对他人的影响评分
    description: str
    context: Dict[str, Any]

@dataclass
class Intention:
    """意图对象"""
    agent_id: str
    intention_type: IntentionType
    confidence: float  # 0.0-1.0
    target: Any
    timeline: datetime
    priority: int  # 1-10
    resources_required: List[str]
    description: str

@dataclass
class SocialLearningEvent:
    """社会学习事件"""
    learner_id: str
    teacher_id: str
    skill_type: str
    learning_method: str  # "observation", "direct_teaching", "imitation"
    success: bool
    timestamp: datetime
    skill_progress: float  # 0.0-1.0

class SocialCognitionSystem:
    """
    社会认知能力系统核心类
    
    管理智能体间的社会交互、意图识别、信任建立和学习
    """
    
    def __init__(self, agent_count: int = 16):
        self.agent_count = agent_count
        self.social_actions: List[SocialAction] = []
        self.intentions: Dict[str, List[Intention]] = defaultdict(list)
        self.learning_events: List[SocialLearningEvent] = []
        
        # 信任网络: agent_id -> {other_agent_id: trust_score}
        self.trust_network: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 社交关系强度
        self.social_bonds: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 技能分布
        self.skills: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # 领导力指标
        self.leadership_scores: Dict[str, float] = defaultdict(float)
        
        # 交互历史
        self.interaction_history: Dict[Tuple[str, str], List[SocialAction]] = defaultdict(list)
        
        # 意图预测准确率跟踪
        self.prediction_accuracy: Dict[str, float] = defaultdict(float)
        
        # 并发控制
        self.lock = threading.RLock()
        
        # 社交网络拓扑结构
        self.social_network_graph = defaultdict(set)
        
        logger = logging.getLogger(__name__)
        logger.info("社会认知系统初始化完成")
    
    def record_social_action(self, action: SocialAction):
        """记录社会行为"""
        with self.lock:
            self.social_actions.append(action)
            
            # 更新交互历史
            pair = tuple(sorted([action.actor_id, action.target_id]))
            self.interaction_history[pair].append(action)
            
            # 更新社会关系强度
            self._update_social_bond(action.actor_id, action.target_id, action.success, action.impact_score)
            
            # 更新信任网络
            self._update_trust_from_action(action)
            
            logger.debug(f"记录社会行为: {action.actor_id} -> {action.target_id} ({action.action_type})")
    
    def _update_social_bond(self, actor_id: str, target_id: str, success: bool, impact: float):
        """更新社会关系强度"""
        if actor_id == target_id:
            return
        
        # 基础影响
        impact_delta = (impact * 0.1 if success else -impact * 0.05)
        
        # 衰减因子
        time_decay = 0.95  # 每次交互衰减5%
        
        # 更新关系强度
        current_bond = self.social_bonds[actor_id].get(target_id, 0.0) * time_decay + impact_delta
        
        # 限制范围
        self.social_bonds[actor_id][target_id] = max(-1.0, min(1.0, current_bond))
    
    def _update_trust_from_action(self, action: SocialAction):
        """从行为更新信任度"""
        actor_id = action.actor_id
        target_id = action.target_id
        
        if actor_id == target_id:
            return
        
        # 信任变化量
        trust_delta = 0.0
        
        # 根据行为类型调整信任
        if action.action_type in ["help", "cooperate", "share_resources", "defend"]:
            trust_delta = 0.1 if action.success else -0.05
        elif action.action_type in ["compete", "block", "ignore"]:
            trust_delta = -0.1 if action.success else 0.05
        elif action.action_type == "trust_shown":
            # 直接信任行为
            trust_delta = 0.2
        
        # 考虑影响力
        trust_delta *= (1 + action.impact_score * 0.5)
        
        # 更新信任网络
        current_trust = self.trust_network[actor_id].get(target_id, 0.0)
        new_trust = max(-3.0, min(3.0, current_trust + trust_delta))
        self.trust_network[actor_id][target_id] = new_trust
        
        logger.debug(f"信任更新: {actor_id} -> {target_id}: {current_trust:.2f} -> {new_trust:.2f}")
    
    def analyze_intentions(self, agent_id: str, observation_window: int = 50) -> Dict[IntentionType, float]:
        """
        分析智能体的意图概率
        
        Args:
            agent_id: 智能体ID
            observation_window: 观察窗口大小
            
        Returns:
            Dict[IntentionType, float]: 各种意图的概率分布
        """
        with self.lock:
            # 获取该智能体的最近行为
            agent_actions = [action for action in self.social_actions[-observation_window:] 
                           if action.actor_id == agent_id]
            
            if not agent_actions:
                return {intent_type: 1.0 / len(IntentionType) for intent_type in IntentionType}
            
            # 行为类型到意图类型的映射
            behavior_to_intention = {
                "explore": [IntentionType.EXPLORATION],
                "build": [IntentionType.CONSTRUCTION],
                "collect": [IntentionType.RESOURCE_COLLECTION],
                "defend": [IntentionType.DEFENSE],
                "help": [IntentionType.COLLABORATION, IntentionType.RESCUE],
                "teach": [IntentionType.LEADERSHIP],
                "study": [IntentionType.LEARNING]
            }
            
            # 计算意图概率
            intention_scores = defaultdict(float)
            total_weight = 0
            
            for action in agent_actions:
                # 行为权重（基于时间衰减和影响）
                weight = 1.0
                if action.success:
                    weight *= (1 + action.impact_score * 0.5)
                
                # 时间衰减
                age_hours = (datetime.now() - action.timestamp).total_seconds() / 3600
                weight *= max(0.1, math.exp(-age_hours / 24))  # 24小时半衰期
                
                # 匹配意图类型
                for behavior, intentions in behavior_to_intention.items():
                    if behavior in action.action_type.lower():
                        for intention in intentions:
                            intention_scores[intention] += weight
                            total_weight += weight
                        break
            
            # 归一化概率
            if total_weight > 0:
                probabilities = {intent: score / total_weight 
                               for intent, score in intention_scores.items()}
                
                # 确保所有意图类型都有概率
                for intent_type in IntentionType:
                    if intent_type not in probabilities:
                        probabilities[intent_type] = 0.1 / len(IntentionType)
            else:
                probabilities = {intent_type: 1.0 / len(IntentionType) 
                               for intent_type in IntentionType}
            
            logger.debug(f"分析意图 {agent_id}: {dict(probabilities)}")
            return probabilities
    
    def build_trust_model(self, agent_id: str) -> Dict[str, float]:
        """
        为智能体构建信任模型
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Dict[str, float]: 对其他智能体的信任度评分
        """
        with self.lock:
            # 获取信任网络中的数据
            trust_scores = self.trust_network.get(agent_id, {})
            
            # 考虑交互历史的影响
            adjusted_scores = {}
            
            for other_agent, base_trust in trust_scores.items():
                # 获取交互历史
                pair = tuple(sorted([agent_id, other_agent]))
                interactions = self.interaction_history.get(pair, [])
                
                if interactions:
                    # 计算正面交互比例
                    positive_interactions = sum(1 for action in interactions if action.success)
                    interaction_frequency = len(interactions)
                    
                    # 调整信任分数
                    adjustment = (
                        (positive_interactions / interaction_frequency - 0.5) * 0.2 +
                        min(interaction_frequency / 10, 0.1)
                    )
                    
                    adjusted_scores[other_agent] = max(-3.0, min(3.0, base_trust + adjustment))
                else:
                    adjusted_scores[other_agent] = base_trust * 0.8  # 没有历史数据时降低信任
            
            # 考虑社会关系强度
            social_bonds = self.social_bonds.get(agent_id, {})
            for other_agent, bond_strength in social_bonds.items():
                if other_agent in adjusted_scores:
                    adjusted_scores[other_agent] += bond_strength * 0.5
            
            logger.debug(f"构建信任模型 {agent_id}: {len(adjusted_scores)} 个信任关系")
            return adjusted_scores
    
    def detect_trustworthy_agents(self, agent_id: str, 
                                threshold: float = 1.0,
                                limit: int = 5) -> List[Tuple[str, float]]:
        """
        检测值得信任的智能体
        
        Args:
            agent_id: 智能体ID
            threshold: 信任阈值
            limit: 返回数量限制
            
        Returns:
            List[Tuple[str, float]]: (智能体ID, 信任度) 列表，按信任度排序
        """
        trust_model = self.build_trust_model(agent_id)
        
        # 筛选高于阈值的智能体
        trustworthy = [(agent_id, score) for agent_id, score in trust_model.items() 
                      if score >= threshold]
        
        # 按信任度排序
        trustworthy.sort(key=lambda x: x[1], reverse=True)
        
        return trustworthy[:limit]
    
    def initiate_social_learning(self, learner_id: str, teacher_id: str, 
                               skill_type: str, learning_method: str) -> bool:
        """
        发起社会学习
        
        Args:
            learner_id: 学习者ID
            teacher_id: 导师ID
            skill_type: 技能类型
            learning_method: 学习方法
            
        Returns:
            bool: 学习是否成功
        """
        with self.lock:
            # 检查信任度
            trust_scores = self.build_trust_model(learner_id)
            teacher_trust = trust_scores.get(teacher_id, 0.0)
            
            # 如果信任度不足，尝试建立关系
            if teacher_trust < 0.0:
                # 尝试合作行为建立信任
                cooperation_action = SocialAction(
                    actor_id=learner_id,
                    action_type="cooperate",
                    target_id=teacher_id,
                    timestamp=datetime.now(),
                    success=True,
                    impact_score=0.3,
                    description=f"请求学习 {skill_type}",
                    context={"skill_type": skill_type, "learning_method": learning_method}
                )
                self.record_social_action(cooperation_action)
                
                # 重新检查信任
                trust_scores = self.build_trust_model(learner_id)
                teacher_trust = trust_scores.get(teacher_id, 0.0)
            
            # 判断学习成功概率
            if teacher_trust >= -0.5:  # 最低容忍度
                # 学习成功率基于多种因素
                base_success_rate = 0.7
                trust_bonus = max(0, teacher_trust) * 0.2
                skill_complexity_factor = 1.0 if skill_type in ["basic", "intermediate"] else 0.8
                method_bonus = {"observation": 0.1, "direct_teaching": 0.2, "imitation": 0.05}[learning_method]
                
                success_probability = (base_success_rate + trust_bonus + method_bonus) * skill_complexity_factor
                success = random.random() < success_probability
                
                # 计算技能进步
                skill_progress = random.uniform(0.1, 0.3) if success else random.uniform(0.0, 0.1)
                
                # 记录学习事件
                learning_event = SocialLearningEvent(
                    learner_id=learner_id,
                    teacher_id=teacher_id,
                    skill_type=skill_type,
                    learning_method=learning_method,
                    success=success,
                    timestamp=datetime.now(),
                    skill_progress=skill_progress
                )
                self.learning_events.append(learning_event)
                
                # 更新技能水平
                if success:
                    current_skill = self.skills[learner_id][skill_type]
                    self.skills[learner_id][skill_type] = min(1.0, current_skill + skill_progress)
                
                # 记录导师的教学行为
                teaching_action = SocialAction(
                    actor_id=teacher_id,
                    action_type="teach",
                    target_id=learner_id,
                    timestamp=datetime.now(),
                    success=success,
                    impact_score=skill_progress,
                    description=f"教学 {skill_type}",
                    context={"student_id": learner_id, "skill_progress": skill_progress}
                )
                self.record_social_action(teaching_action)
                
                logger.info(f"社会学习: {learner_id} <- {teacher_id} ({skill_type}) - {'成功' if success else '失败'}")
                return success
            else:
                logger.info(f"社会学习被拒绝: {learner_id} 对 {teacher_id} 信任度不足")
                return False
    
    def get_skill_distribution(self) -> Dict[str, float]:
        """
        获取部落技能分布
        
        Returns:
            Dict[str, float]: 平均技能水平
        """
        with self.lock:
            skill_totals = defaultdict(float)
            skill_counts = defaultdict(int)
            
            for agent_skills in self.skills.values():
                for skill, level in agent_skills.items():
                    skill_totals[skill] += level
                    skill_counts[skill] += 1
            
            # 计算平均技能水平
            avg_skills = {
                skill: total / max(count, 1) 
                for skill, (total, count) in zip(
                    skill_totals.keys(), 
                    zip(skill_totals.values(), skill_counts.values())
                )
            }
            
            return avg_skills
    
    def elect_leader(self, election_criteria: str = "balanced") -> str:
        """
        选举部落领导者
        
        Args:
            election_criteria: 选举标准 ("popularity", "competence", "balanced")
            
        Returns:
            str: 选出的领导者ID
        """
        with self.lock:
            candidate_scores = {}
            
            for agent_id in range(self.agent_count):
                agent_id_str = f"agent_{agent_id}"
                
                if election_criteria == "popularity":
                    # 基于受欢迎程度（社交关系强度）
                    popularity_score = sum(self.social_bonds[agent_id_str].values())
                    candidate_scores[agent_id_str] = popularity_score
                
                elif election_criteria == "competence":
                    # 基于能力（技能水平）
                    competence_score = sum(self.skills[agent_id_str].values())
                    candidate_scores[agent_id_str] = competence_score
                
                elif election_criteria == "balanced":
                    # 平衡考虑多个因素
                    # 1. 社交关系
                    social_score = sum(self.social_bonds[agent_id_str].values())
                    
                    # 2. 能力水平
                    competence_score = sum(self.skills[agent_id_str].values())
                    
                    # 3. 信任网络
                    trust_influence = sum(self.trust_network.get(agent_id_str, {}).values())
                    
                    # 4. 领导历史
                    leadership_history = self.leadership_scores.get(agent_id_str, 0.0)
                    
                    # 加权组合
                    candidate_scores[agent_id_str] = (
                        social_score * 0.3 +
                        competence_score * 0.3 +
                        trust_influence * 0.2 +
                        leadership_history * 0.2
                    )
            
            # 选择得分最高的候选者
            leader = max(candidate_scores, key=candidate_scores.get)
            
            # 更新领导历史
            self.leadership_scores[leader] += 1
            
            logger.info(f"选举领导者: {leader} (标准: {election_criteria})")
            return leader
    
    def update_leadership_effectiveness(self, leader_id: str, effectiveness_score: float):
        """
        更新领导者效能
        
        Args:
            leader_id: 领导者ID
            effectiveness_score: 效能评分 (0.0-1.0)
        """
        current_effectiveness = self.leadership_scores[leader_id]
        # 使用移动平均
        alpha = 0.1
        self.leadership_scores[leader_id] = (1 - alpha) * current_effectiveness + alpha * effectiveness_score
        
        logger.debug(f"更新领导者效能: {leader_id} -> {self.leadership_scores[leader_id]:.3f}")
    
    def analyze_social_network(self) -> Dict[str, Any]:
        """
        分析社交网络结构
        
        Returns:
            Dict[str, Any]: 网络分析结果
        """
        with self.lock:
            # 计算网络度中心性
            centrality_scores = {}
            for agent_id in range(self.agent_count):
                agent_id_str = f"agent_{agent_id}"
                # 度中心性（连接数）
                connections = len(self.social_bonds[agent_id_str])
                trust_connections = len([score for score in self.trust_network.get(agent_id_str, {}).values() if score > 0])
                centrality_scores[agent_id_str] = (connections + trust_connections) / (self.agent_count - 1)
            
            # 计算聚类系数
            clustering_coeffs = {}
            for agent_id_str in [f"agent_{i}" for i in range(self.agent_count)]:
                neighbors = set(self.social_bonds[agent_id_str].keys()) | set(self.trust_network.get(agent_id_str, {}).keys())
                if len(neighbors) < 2:
                    clustering_coeffs[agent_id_str] = 0.0
                    continue
                
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = 0
                
                for neighbor1 in neighbors:
                    for neighbor2 in neighbors:
                        if neighbor1 != neighbor2:
                            # 检查neighbor1和neighbor2是否直接连接
                            if neighbor2 in self.social_bonds[neighbor1] or neighbor1 in self.social_bonds[neighbor2]:
                                actual_edges += 1
                
                clustering_coeffs[agent_id_str] = actual_edges / (possible_edges * 2) if possible_edges > 0 else 0.0
            
            # 计算网络密度
            total_possible_connections = self.agent_count * (self.agent_count - 1) / 2
            actual_connections = sum(len(bonds) for bonds in self.social_bonds.values())
            network_density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
            
            return {
                "centrality_scores": centrality_scores,
                "clustering_coefficients": clustering_coeffs,
                "network_density": network_density,
                "total_interactions": len(self.social_actions),
                "average_trust_level": np.mean([
                    score for trust_dict in self.trust_network.values() 
                    for score in trust_dict.values()
                ]) if self.trust_network else 0.0
            }
    
    def get_social_recommendations(self, agent_id: str, 
                                 recommendation_type: str = "collaboration") -> List[Tuple[str, float]]:
        """
        获取社交推荐
        
        Args:
            agent_id: 智能体ID
            recommendation_type: 推荐类型 ("collaboration", "learning", "trust")
            
        Returns:
            List[Tuple[str, float]]: (智能体ID, 推荐分数) 列表
        """
        with self.lock:
            recommendations = []
            
            if recommendation_type == "collaboration":
                # 推荐潜在合作伙伴
                trust_model = self.build_trust_model(agent_id)
                
                for other_agent, trust_score in trust_model.items():
                    if other_agent != agent_id and trust_score > 0.0:
                        # 计算协作适配度
                        compatibility_score = trust_score
                        
                        # 考虑技能互补性
                        agent_skills = set(self.skills[agent_id].keys())
                        other_skills = set(self.skills[other_agent].keys())
                        complementary_skills = len(agent_skills ^ other_skills)
                        compatibility_score += complementary_skills * 0.1
                        
                        recommendations.append((other_agent, compatibility_score))
            
            elif recommendation_type == "learning":
                # 推荐学习对象
                for other_agent in [f"agent_{i}" for i in range(self.agent_count)]:
                    if other_agent != agent_id:
                        # 基于技能优势推荐
                        agent_skills = self.skills[agent_id]
                        other_skills = self.skills[other_agent]
                        
                        # 找出其他智能体更擅长的技能
                        skill_advantages = []
                        for skill, level in other_skills.items():
                            agent_level = agent_skills.get(skill, 0.0)
                            if level > agent_level + 0.2:  # 显著优势
                                skill_advantages.append(level - agent_level)
                        
                        if skill_advantages:
                            learning_potential = sum(skill_advantages) / len(skill_advantages)
                            trust_bonus = self.build_trust_model(agent_id).get(other_agent, 0.0) * 0.1
                            recommendations.append((other_agent, learning_potential + trust_bonus))
            
            elif recommendation_type == "trust":
                # 推荐值得信任的智能体
                potential_trust = []
                trust_model = self.build_trust_model(agent_id)
                
                for other_agent in [f"agent_{i}" for i in range(self.agent_count)]:
                    if other_agent != agent_id and other_agent not in trust_model:
                        # 基于社会关系推荐
                        social_bonds = self.social_bonds.get(agent_id, {})
                        mutual_connections = 0
                        
                        # 检查是否有共同朋友
                        agent_friends = set(social_bonds.keys())
                        other_friends = set(self.social_bonds.get(other_agent, {}).keys())
                        mutual_connections = len(agent_friends & other_friends)
                        
                        # 基于共同朋友数量推荐
                        if mutual_connections > 0:
                            trust_potential = mutual_connections * 0.2
                            potential_trust.append((other_agent, trust_potential))
                
                recommendations.extend(potential_trust)
            
            # 按推荐分数排序
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:5]  # 返回前5个推荐
    
    def get_agent_social_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        获取智能体的社交档案
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Dict[str, Any]: 社交档案信息
        """
        with self.lock:
            # 获取行为历史
            agent_actions = [action for action in self.social_actions if action.actor_id == agent_id]
            target_actions = [action for action in self.social_actions if action.target_id == agent_id]
            
            # 计算社交活跃度
            activity_score = min(1.0, len(agent_actions) / 20)  # 归一化到20次
            
            # 成功率
            success_rate = np.mean([action.success for action in agent_actions]) if agent_actions else 0.0
            
            # 平均影响力
            avg_impact = np.mean([action.impact_score for action in agent_actions]) if agent_actions else 0.0
            
            # 信任网络
            trust_model = self.build_trust_model(agent_id)
            trusted_count = len([score for score in trust_model.values() if score > 0])
            distrust_count = len([score for score in trust_model.values() if score < 0])
            
            # 技能水平
            skills = self.skills.get(agent_id, {})
            avg_skill_level = np.mean(list(skills.values())) if skills else 0.0
            
            # 社交地位
            social_status = self.leadership_scores.get(agent_id, 0.0)
            
            return {
                "agent_id": agent_id,
                "social_activity": {
                    "total_actions": len(agent_actions),
                    "targeted_actions": len(target_actions),
                    "activity_score": activity_score,
                    "success_rate": success_rate,
                    "average_impact": avg_impact
                },
                "trust_network": {
                    "trusts_others": trusted_count,
                    "distrusted_by": distrust_count,
                    "average_trust_given": np.mean(list(trust_model.values())) if trust_model else 0.0
                },
                "skills": {
                    "skill_count": len(skills),
                    "average_level": avg_skill_level,
                    "top_skills": sorted(skills.items(), key=lambda x: x[1], reverse=True)[:3]
                },
                "social_status": {
                    "leadership_score": social_status,
                    "popularity": sum(self.social_bonds.get(agent_id, {}).values())
                }
            }
    
    def simulate_interaction(self, agent1_id: str, agent2_id: str, 
                           interaction_type: str, context: Dict[str, Any] = None) -> SocialAction:
        """
        模拟智能体交互
        
        Args:
            agent1_id: 智能体1 ID
            agent2_id: 智能体2 ID
            interaction_type: 交互类型
            context: 交互上下文
            
        Returns:
            SocialAction: 产生的社交行为
        """
        with self.lock:
            # 基于信任度和历史计算成功概率
            trust_scores = self.build_trust_model(agent1_id)
            trust_level = trust_scores.get(agent2_id, 0.0)
            
            # 交互成功率
            base_success_rates = {
                "help": 0.8,
                "cooperate": 0.75,
                "compete": 0.6,
                "share": 0.7,
                "teach": 0.85,
                "learn": 0.7
            }
            
            base_rate = base_success_rates.get(interaction_type, 0.5)
            trust_bonus = max(0, trust_level) * 0.2
            success_probability = min(1.0, base_rate + trust_bonus)
            
            success = random.random() < success_probability
            
            # 计算影响力
            impact_score = random.uniform(0.2, 0.8)
            if success:
                impact_score *= 1.5
            
            # 创建行为记录
            action = SocialAction(
                actor_id=agent1_id,
                action_type=interaction_type,
                target_id=agent2_id,
                timestamp=datetime.now(),
                success=success,
                impact_score=impact_score,
                description=f"{agent1_id} 与 {agent2_id} 进行 {interaction_type} 交互",
                context=context or {}
            )
            
            self.record_social_action(action)
            return action
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        with self.lock:
            total_actions = len(self.social_actions)
            successful_actions = sum(1 for action in self.social_actions if action.success)
            success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
            
            # 信任网络统计
            trust_relationships = sum(len(trusts) for trusts in self.trust_network.values())
            avg_trust = np.mean([
                score for trust_dict in self.trust_network.values() 
                for score in trust_dict.values()
            ]) if self.trust_network else 0.0
            
            # 学习事件统计
            learning_success_rate = np.mean([
                event.success for event in self.learning_events
            ]) if self.learning_events else 0.0
            
            # 社交网络分析
            network_analysis = self.analyze_social_network()
            
            return {
                "total_social_actions": total_actions,
                "action_success_rate": success_rate,
                "trust_relationships": trust_relationships,
                "average_trust_level": avg_trust,
                "learning_events": len(self.learning_events),
                "learning_success_rate": learning_success_rate,
                "network_analysis": network_analysis,
                "skill_distribution": self.get_skill_distribution(),
                "leadership_distribution": dict(self.leadership_scores)
            }


# 工具函数
def create_intention(agent_id: str, intention_type: IntentionType, 
                    confidence: float, target: Any, priority: int = 5,
                    resources_required: List[str] = None,
                    description: str = "") -> Intention:
    """创建意图对象"""
    return Intention(
        agent_id=agent_id,
        intention_type=intention_type,
        confidence=confidence,
        target=target,
        timeline=datetime.now(),
        priority=priority,
        resources_required=resources_required or [],
        description=description
    )