"""
多智能体社会系统 (Multi-Agent Tribal Society System)
实现16个智能体构成的部落，包含完整的协作和社会机制

功能特性：
- 16个智能体构成的部落架构
- 共享集体记忆库和私有目标管理
- 智能体个体特征和能力建模
- 社会网络形成和演化
- 集体智能涌现机制
- 完整的模拟和测试框架
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import logging
from pathlib import Path

# 导入自定义模块
from collective_memory import CollectiveMemory, MemoryEntry, create_danger_zone_memory, create_resource_hotspot_memory, create_blueprint_memory
from social_cognition import SocialCognitionSystem, IntentionType, TrustLevel, SocialAction, create_intention
from collaboration_protocol import (
    CollaborationProtocol, Task, TaskType, TaskPriority, Resource, ResourceType,
    DecisionType, DecisionProposal
)

logger = logging.getLogger(__name__)

class AgentPersonality(Enum):
    """智能体性格类型"""
    LEADER = "leader"           # 领导者型
    COLLABORATOR = "collaborator"  # 协作者型
    EXPLORER = "explorer"       # 探索者型
    BUILDER = "builder"         # 建造者型
    RESEARCHER = "researcher"   # 研究者型
    PROTECTOR = "protector"     # 保护者型
    CREATOR = "creator"         # 创造者型
    ANALYST = "analyst"         # 分析者型

class AgentState(Enum):
    """智能体状态"""
    IDLE = "idle"               # 空闲
    WORKING = "working"         # 工作中
    EXPLORING = "exploring"     # 探索中
    COLLABORATING = "collaborating"  # 协作中
    LEARNING = "learning"       # 学习中
    DECIDING = "deciding"       # 决策中
    RESTING = "resting"         # 休息中

@dataclass
class AgentCharacteristics:
    """智能体特征"""
    agent_id: str
    personality: AgentPersonality
    energy_level: float  # 0.0-1.0
    motivation: float   # 0.0-1.0
    social_tendency: float  # 0.0-1.0 (社交倾向)
    risk_tolerance: float  # 0.0-1.0 (风险容忍度)
    learning_speed: float  # 0.0-1.0 (学习速度)
    creativity: float  # 0.0-1.0 (创造力)
    leadership_potential: float  # 0.0-1.0 (领导力潜力)
    
    # 私有目标
    personal_goals: List[Dict[str, Any]]
    preferred_exploration_direction: Tuple[float, float, float]  # 探索偏好方向
    building_preferences: List[str]  # 建筑偏好类型
    
    # 当前状态
    current_state: AgentState = AgentState.IDLE
    current_task_id: Optional[str] = None
    current_activity: Optional[str] = None
    location: Optional[Tuple[float, float, float]] = None
    skills_learned: Set[str] = None
    
    def __post_init__(self):
        if self.skills_learned is None:
            self.skills_learned = set()

class TribalSociety:
    """
    部落社会系统核心类
    
    管理16个智能体的部落，包括集体记忆、协作协议和社会认知
    """
    
    def __init__(self, agent_count: int = 16):
        self.agent_count = agent_count
        self.agents: Dict[str, AgentCharacteristics] = {}
        self.world_state: Dict[str, Any] = {}
        
        # 核心系统
        self.collective_memory = CollectiveMemory(memory_capacity=5000)
        self.social_cognition = SocialCognitionSystem(agent_count=agent_count)
        self.collaboration_protocol = CollaborationProtocol(agent_count=agent_count)
        
        # 模拟状态
        self.simulation_time = datetime.now()
        self.simulation_step = 0
        self.is_running = False
        self.simulation_speed = 1.0  # 1.0 = 实时，2.0 = 2倍速
        
        # 社会网络
        self.social_network: Dict[str, Set[str]] = defaultdict(set)
        self.influence_network: Dict[str, float] = defaultdict(float)
        
        # 集体智能指标
        self.collective_intelligence_metrics: Dict[str, float] = {}
        self.emergence_indicators: Dict[str, float] = {}
        
        # 事件日志
        self.event_log: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # 并发控制
        self.lock = threading.RLock()
        
        # 初始化部落
        self._initialize_tribe()
        
        logger.info(f"部落社会系统初始化完成: {agent_count} 个智能体")
    
    def _initialize_tribe(self):
        """初始化部落和智能体"""
        personalities = list(AgentPersonality)
        
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            
            # 分配性格类型
            personality = personalities[i % len(personalities)]
            
            # 根据性格调整特征
            characteristics = self._create_personality_based_characteristics(agent_id, personality)
            
            self.agents[agent_id] = characteristics
            logger.debug(f"创建智能体: {agent_id} ({personality.value})")
        
        # 创建初始资源
        self._initialize_resources()
        
        # 创建初始记忆
        self._initialize_collective_memory()
        
        # 建立初始社交网络
        self._initialize_social_network()
    
    def _create_personality_based_characteristics(self, agent_id: str, personality: AgentPersonality) -> AgentCharacteristics:
        """基于性格创建智能体特征"""
        # 基础特征
        base_characteristics = {
            "energy_level": random.uniform(0.7, 1.0),
            "motivation": random.uniform(0.6, 0.9),
            "social_tendency": random.uniform(0.3, 0.8),
            "risk_tolerance": random.uniform(0.2, 0.7),
            "learning_speed": random.uniform(0.4, 0.8),
            "creativity": random.uniform(0.3, 0.8),
            "leadership_potential": random.uniform(0.2, 0.7)
        }
        
        # 根据性格调整特征
        if personality == AgentPersonality.LEADER:
            base_characteristics.update({
                "social_tendency": min(1.0, base_characteristics["social_tendency"] + 0.2),
                "leadership_potential": min(1.0, base_characteristics["leadership_potential"] + 0.3),
                "motivation": min(1.0, base_characteristics["motivation"] + 0.1)
            })
        elif personality == AgentPersonality.COLLABORATOR:
            base_characteristics.update({
                "social_tendency": min(1.0, base_characteristics["social_tendency"] + 0.3),
                "energy_level": min(1.0, base_characteristics["energy_level"] + 0.1)
            })
        elif personality == AgentPersonality.EXPLORER:
            base_characteristics.update({
                "risk_tolerance": min(1.0, base_characteristics["risk_tolerance"] + 0.3),
                "creativity": min(1.0, base_characteristics["creativity"] + 0.2),
                "learning_speed": min(1.0, base_characteristics["learning_speed"] + 0.1)
            })
        elif personality == AgentPersonality.BUILDER:
            base_characteristics.update({
                "energy_level": min(1.0, base_characteristics["energy_level"] + 0.2),
                "creativity": min(1.0, base_characteristics["creativity"] + 0.2)
            })
        elif personality == AgentPersonality.RESEARCHER:
            base_characteristics.update({
                "learning_speed": min(1.0, base_characteristics["learning_speed"] + 0.3),
                "risk_tolerance": max(0.0, base_characteristics["risk_tolerance"] - 0.1),
                "motivation": max(0.0, base_characteristics["motivation"] - 0.1)
            })
        
        # 生成私有目标
        personal_goals = self._generate_personal_goals(personality)
        
        # 生成探索偏好方向
        preferred_direction = (
            random.uniform(-1, 1),
            random.uniform(-1, 1), 
            random.uniform(-1, 1)
        )
        
        # 生成建筑偏好
        building_preferences = self._generate_building_preferences(personality)
        
        return AgentCharacteristics(
            agent_id=agent_id,
            personality=personality,
            energy_level=base_characteristics["energy_level"],
            motivation=base_characteristics["motivation"],
            social_tendency=base_characteristics["social_tendency"],
            risk_tolerance=base_characteristics["risk_tolerance"],
            learning_speed=base_characteristics["learning_speed"],
            creativity=base_characteristics["creativity"],
            leadership_potential=base_characteristics["leadership_potential"],
            personal_goals=personal_goals,
            preferred_exploration_direction=preferred_direction,
            building_preferences=building_preferences
        )
    
    def _generate_personal_goals(self, personality: AgentPersonality) -> List[Dict[str, Any]]:
        """根据性格生成私人目标"""
        goals = []
        
        goal_templates = {
            AgentPersonality.LEADER: [
                {"type": "leadership", "description": "成为部落领导者", "priority": 9, "progress": 0.0},
                {"type": "organization", "description": "建立高效协作机制", "priority": 8, "progress": 0.0}
            ],
            AgentPersonality.COLLABORATOR: [
                {"type": "collaboration", "description": "与所有智能体建立良好关系", "priority": 8, "progress": 0.0},
                {"type": "support", "description": "帮助其他智能体完成任务", "priority": 7, "progress": 0.0}
            ],
            AgentPersonality.EXPLORER: [
                {"type": "exploration", "description": "探索整个世界区域", "priority": 9, "progress": 0.0},
                {"type": "discovery", "description": "发现新的资源和地点", "priority": 8, "progress": 0.0}
            ],
            AgentPersonality.BUILDER: [
                {"type": "construction", "description": "建造宏伟的建筑", "priority": 8, "progress": 0.0},
                {"type": "innovation", "description": "发明新的建造技术", "priority": 7, "progress": 0.0}
            ],
            AgentPersonality.RESEARCHER: [
                {"type": "research", "description": "深入研究世界规律", "priority": 9, "progress": 0.0},
                {"type": "knowledge", "description": "积累全面知识", "priority": 8, "progress": 0.0}
            ],
            AgentPersonality.PROTECTOR: [
                {"type": "defense", "description": "保护部落安全", "priority": 9, "progress": 0.0},
                {"type": "preparedness", "description": "准备应对各种威胁", "priority": 7, "progress": 0.0}
            ],
            AgentPersonality.CREATOR: [
                {"type": "creation", "description": "创造独特的艺术品", "priority": 8, "progress": 0.0},
                {"type": "innovation", "description": "发明新的技术和方法", "priority": 7, "progress": 0.0}
            ],
            AgentPersonality.ANALYST: [
                {"type": "analysis", "description": "分析部落运行数据", "priority": 7, "progress": 0.0},
                {"type": "optimization", "description": "优化部落效率", "priority": 8, "progress": 0.0}
            ]
        }
        
        return goal_templates.get(personality, [])
    
    def _generate_building_preferences(self, personality: AgentPersonality) -> List[str]:
        """根据性格生成建筑偏好"""
        preferences = {
            AgentPersonality.LEADER: ["headquarters", "meeting_rooms", "command_center"],
            AgentPersonality.COLLABORATOR: ["workshop", "common_areas", "communication_towers"],
            AgentPersonality.EXPLORER: ["outposts", "observation_towers", "storage_facilities"],
            AgentPersonality.BUILDER: ["construction_yards", "workshops", "architectural_showcase"],
            AgentPersonality.RESEARCHER: ["laboratories", "libraries", "observatories"],
            AgentPersonality.PROTECTOR: ["fortifications", "defense_towers", "bunkers"],
            AgentPersonality.CREATOR: ["art_studios", "exhibition_halls", "innovation_centers"],
            AgentPersonality.ANALYST: ["data_centers", "monitoring_stations", "analysis_labs"]
        }
        
        return preferences.get(personality, ["residential", "storage"])
    
    def _initialize_resources(self):
        """初始化基础资源"""
        # 基础材料
        basic_resources = [
            ("stone", 50, ResourceType.MATERIAL),
            ("wood", 30, ResourceType.MATERIAL),
            ("iron", 20, ResourceType.MATERIAL),
            ("tools", 10, ResourceType.TOOL),
            ("food", 25, ResourceType.FOOD),
            ("energy", 100, ResourceType.ENERGY)
        ]
        
        for name, quantity, resource_type in basic_resources:
            resource = Resource(
                id=f"resource_{name}",
                resource_type=resource_type,
                name=name,
                quantity=quantity,
                quality=0.8,
                shared=True,
                accessibility="easy"
            )
            self.collaboration_protocol.create_resource(resource)
    
    def _initialize_collective_memory(self):
        """初始化集体记忆"""
        # 创建一些初始记忆
        
        # 危险区域示例
        danger_memories = [
            create_danger_zone_memory(100, 0, 50, "creeper_spawn", "苦力怕刷新点", "system"),
            create_danger_zone_memory(-50, 0, -30, "lava_pit", "岩浆坑", "system"),
            create_danger_zone_memory(20, -10, 80, "deep_water", "深水区", "system")
        ]
        
        for memory in danger_memories:
            self.collective_memory.store_memory(memory)
        
        # 资源热点示例
        resource_memories = [
            create_resource_hotspot_memory(0, 5, 0, "iron", "moderate", 0.8, "system"),
            create_resource_hotspot_memory(40, 3, -20, "coal", "abundant", 0.7, "system"),
            create_resource_hotspot_memory(-30, 8, 60, "diamond", "scarce", 0.9, "system")
        ]
        
        for memory in resource_memories:
            self.collective_memory.store_memory(memory)
        
        # 建筑蓝图示例
        blueprint_memories = [
            create_blueprint_memory(
                "basic_shelter", 
                {
                    "materials": ["wood", "stone"],
                    "steps": ["foundation", "walls", "roof"],
                    "time": 4
                },
                "simple",
                "system"
            ),
            create_blueprint_memory(
                "defense_tower",
                {
                    "materials": ["stone", "iron"],
                    "steps": ["base", "walls", "observation_deck"],
                    "time": 12
                },
                "complex",
                "system"
            )
        ]
        
        for memory in blueprint_memories:
            self.collective_memory.store_memory(memory)
    
    def _initialize_social_network(self):
        """初始化社交网络"""
        # 随机创建初始社交连接
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            
            # 每个智能体建立2-4个初始连接
            connection_count = random.randint(2, 4)
            potential_connections = [f"agent_{j}" for j in range(self.agent_count) if j != i]
            
            for _ in range(connection_count):
                if potential_connections:
                    connected_agent = random.choice(potential_connections)
                    self.social_network[agent_id].add(connected_agent)
                    potential_connections.remove(connected_agent)
    
    def start_simulation(self):
        """启动模拟"""
        self.is_running = True
        self.simulation_time = datetime.now()
        self.simulation_step = 0
        
        logger.info("开始部落模拟")
        
        # 启动主要模拟循环
        self._run_simulation_loop()
    
    def _run_simulation_loop(self):
        """运行主要模拟循环"""
        while self.is_running:
            try:
                # 执行一个模拟步骤
                self._execute_simulation_step()
                
                # 时间推进
                self.simulation_time += timedelta(hours=1)
                self.simulation_step += 1
                
                # 记录性能数据
                if self.simulation_step % 24 == 0:  # 每天记录一次
                    self._record_performance_metrics()
                
                # 模拟世界事件
                if self.simulation_step % 12 == 0:  # 每12步触发世界事件
                    self._trigger_world_events()
                
                time.sleep(0.1)  # 控制模拟速度
                
            except Exception as e:
                logger.error(f"模拟执行出错: {e}")
                break
    
    def _execute_simulation_step(self):
        """执行一个模拟步骤"""
        with self.lock:
            # 1. 智能体决策和行动
            self._update_agent_decisions()
            
            # 2. 社会交互
            self._process_social_interactions()
            
            # 3. 记忆更新
            self._update_collective_memory()
            
            # 4. 任务执行
            self._process_task_execution()
            
            # 5. 集体智能计算
            self._calculate_collective_intelligence()
            
            # 6. 记录事件
            self._log_current_state()
    
    def _update_agent_decisions(self):
        """更新智能体决策"""
        for agent_id, agent in self.agents.items():
            # 基于性格和当前状态决定行动
            action = self._decide_agent_action(agent)
            
            if action:
                # 执行行动
                self._execute_agent_action(agent_id, agent, action)
                
                # 更新智能体状态
                self._update_agent_state(agent_id, agent, action)
    
    def _decide_agent_action(self, agent: AgentCharacteristics) -> Optional[str]:
        """决定智能体行动"""
        # 基于性格和目标决定行动
        personality = agent.personality
        
        # 检查当前状态
        if agent.current_state == AgentState.WORKING and agent.current_task_id:
            return "continue_task"
        
        # 随机决策（简化版本）
        action_weights = {
            AgentPersonality.LEADER: {
                "organize": 0.3, "communicate": 0.2, "lead": 0.25, "analyze": 0.15, "rest": 0.1
            },
            AgentPersonality.COLLABORATOR: {
                "collaborate": 0.4, "help": 0.3, "communicate": 0.2, "rest": 0.1
            },
            AgentPersonality.EXPLORER: {
                "explore": 0.4, "discover": 0.3, "document": 0.2, "rest": 0.1
            },
            AgentPersonality.BUILDER: {
                "construct": 0.4, "plan": 0.3, "innovate": 0.2, "rest": 0.1
            },
            AgentPersonality.RESEARCHER: {
                "research": 0.4, "analyze": 0.3, "study": 0.2, "rest": 0.1
            },
            AgentPersonality.PROTECTOR: {
                "defend": 0.3, "patrol": 0.3, "prepare": 0.25, "rest": 0.15
            },
            AgentPersonality.CREATOR: {
                "create": 0.4, "design": 0.3, "innovate": 0.2, "rest": 0.1
            },
            AgentPersonality.ANALYST: {
                "analyze": 0.4, "optimize": 0.3, "report": 0.2, "rest": 0.1
            }
        }
        
        weights = action_weights.get(personality, {"explore": 0.5, "rest": 0.5})
        
        # 根据权重选择行动
        actions = list(weights.keys())
        probabilities = list(weights.values())
        
        return np.random.choice(actions, p=probabilities)
    
    def _execute_agent_action(self, agent_id: str, agent: AgentCharacteristics, action: str):
        """执行智能体行动"""
        if action == "explore":
            self._execute_exploration(agent_id, agent)
        elif action == "construct":
            self._execute_construction(agent_id, agent)
        elif action == "collaborate":
            self._execute_collaboration(agent_id, agent)
        elif action == "research":
            self._execute_research(agent_id, agent)
        elif action == "communicate":
            self._execute_communication(agent_id, agent)
        elif action == "help":
            self._execute_helping(agent_id, agent)
        elif action == "organize":
            self._execute_organizing(agent_id, agent)
        elif action == "analyze":
            self._execute_analysis(agent_id, agent)
    
    def _execute_exploration(self, agent_id: str, agent: AgentCharacteristics):
        """执行探索行动"""
        # 获取探索偏好方向
        direction = agent.preferred_exploration_direction
        
        # 计算新位置（简化）
        new_location = (
            (agent.location[0] if agent.location else 0) + direction[0] * 10,
            (agent.location[1] if agent.location else 0) + direction[1] * 5,
            (agent.location[2] if agent.location else 0) + direction[2] * 10
        )
        
        # 更新位置
        agent.location = new_location
        
        # 记录探索行为
        if random.random() < 0.3:  # 30%概率发现新事物
            discovery_type = random.choice(["resource", "danger", "landmark"])
            self._record_discovery(agent_id, new_location, discovery_type)
        
        agent.current_state = AgentState.EXPLORING
        agent.current_activity = "exploration"
    
    def _execute_construction(self, agent_id: str, agent: AgentCharacteristics):
        """执行建造行动"""
        # 选择建筑项目
        building_type = random.choice(agent.building_preferences) if agent.building_preferences else "shelter"
        
        # 创建建造任务
        task = create_simple_task(
            TaskType.CONSTRUCTION,
            f"建造{building_type}",
            TaskPriority.MEDIUM,
            random.randint(4, 12)
        )
        
        task.required_resources = {
            ResourceType.MATERIAL: random.randint(5, 15),
            ResourceType.TOOL: 1
        }
        task.created_by = agent_id
        
        task_id = self.collaboration_protocol.create_task(task)
        
        # 尝试分配给自己
        if self.collaboration_protocol.assign_task(task_id, agent_id):
            agent.current_state = AgentState.WORKING
            agent.current_task_id = task_id
            agent.current_activity = "construction"
    
    def _execute_collaboration(self, agent_id: str, agent: AgentCharacteristics):
        """执行协作行动"""
        # 寻找协作伙伴
        potential_partners = []
        for other_id in self.social_network.get(agent_id, []):
            other_agent = self.agents.get(other_id)
            if other_agent and other_agent.current_state in [AgentState.IDLE, AgentState.RESTING]:
                potential_partners.append(other_id)
        
        if potential_partners:
            partner_id = random.choice(potential_partners)
            
            # 创建协作任务
            collaboration_type = random.choice(["resource_sharing", "knowledge_transfer", "joint_project"])
            task = create_simple_task(
                TaskType.CONSTRUCTION if collaboration_type == "joint_project" else TaskType.RESEARCH,
                f"{collaboration_type}与{partner_id}",
                TaskPriority.MEDIUM,
                random.randint(2, 6)
            )
            
            task_id = self.collaboration_protocol.create_task(task)
            task.assigned_to = agent_id  # 发起者负责
            
            agent.current_state = AgentState.COLLABORATING
            agent.current_activity = f"collaboration_with_{partner_id}"
            agent.current_task_id = task_id
            
            # 记录社交行为
            social_action = SocialAction(
                actor_id=agent_id,
                action_type="cooperate",
                target_id=partner_id,
                timestamp=self.simulation_time,
                success=True,
                impact_score=0.5,
                description=f"发起{collaboration_type}协作",
                context={"task_id": task_id, "collaboration_type": collaboration_type}
            )
            self.social_cognition.record_social_action(social_action)
    
    def _execute_research(self, agent_id: str, agent: AgentCharacteristics):
        """执行研究行动"""
        # 研究主题选择
        research_topics = ["world_mechanics", "resource_optimization", "social_dynamics", "construction_techniques"]
        topic = random.choice(research_topics)
        
        # 创建研究任务
        task = create_simple_task(
            TaskType.RESEARCH,
            f"研究{topic}",
            TaskPriority.MEDIUM,
            random.randint(3, 8)
        )
        
        task.required_resources = {
            ResourceType.TIME: random.randint(2, 5)
        }
        task.created_by = agent_id
        
        task_id = self.collaboration_protocol.create_task(task)
        
        if self.collaboration_protocol.assign_task(task_id, agent_id):
            agent.current_state = AgentState.WORKING
            agent.current_task_id = task_id
            agent.current_activity = "research"
            
            # 记录学习事件
            learning_event = {
                "agent_id": agent_id,
                "topic": topic,
                "timestamp": self.simulation_time,
                "success": random.random() < 0.8,  # 80%成功率
                "progress": random.uniform(0.1, 0.3)
            }
            
            # 将研究成果存入集体记忆
            self._store_research_memory(agent_id, topic, learning_event)
    
    def _execute_communication(self, agent_id: str, agent: AgentCharacteristics):
        """执行沟通行动"""
        # 与随机连接的智能体交流
        connections = list(self.social_network.get(agent_id, []))
        if connections:
            target_id = random.choice(connections)
            
            # 模拟交流内容
            communication_type = random.choice(["information_share", "strategy_discussion", "emotional_support"])
            
            # 记录沟通行为
            social_action = SocialAction(
                actor_id=agent_id,
                action_type="communicate",
                target_id=target_id,
                timestamp=self.simulation_time,
                success=True,
                impact_score=0.3,
                description=f"{communication_type}",
                context={"communication_type": communication_type}
            )
            self.social_cognition.record_social_action(social_action)
            
            # 随机发现或分享信息
            if random.random() < 0.4:  # 40%概率发现新信息
                info_type = random.choice(["resource_location", "danger_warning", "building_technique"])
                self._share_information(agent_id, target_id, info_type)
    
    def _execute_helping(self, agent_id: str, agent: AgentCharacteristics):
        """执行帮助行动"""
        # 寻找需要帮助的智能体
        agents_needing_help = []
        for other_id, other_agent in self.agents.items():
            if (other_id != agent_id and 
                other_agent.current_state == AgentState.WORKING and 
                other_agent.energy_level < 0.3):  # 能量低的智能体
            
                agents_needing_help.append(other_id)
        
        if agents_needing_help:
            target_id = random.choice(agents_needing_help)
            
            # 提供帮助
            help_action = SocialAction(
                actor_id=agent_id,
                action_type="help",
                target_id=target_id,
                timestamp=self.simulation_time,
                success=True,
                impact_score=0.6,
                description="提供帮助",
                context={"help_type": "energy_support"}
            )
            self.social_cognition.record_social_action(help_action)
            
            # 恢复目标智能体能量
            self.agents[target_id].energy_level = min(1.0, self.agents[target_id].energy_level + 0.3)
    
    def _execute_organizing(self, agent_id: str, agent: AgentCharacteristics):
        """执行组织行动"""
        # 领导者特征明显时执行组织行动
        if agent.leadership_potential > 0.7:
            # 组织会议或活动
            meeting_type = random.choice(["strategy_meeting", "resource_planning", "conflict_resolution"])
            
            # 邀请其他智能体
            invitees = []
            for other_id in self.agents.keys():
                if other_id != agent_id:
                    invitees.append(other_id)
            
            invited_count = random.randint(3, min(8, len(invitees)))
            selected_invitees = random.sample(invitees, invited_count)
            
            # 记录组织行为
            for invitee in selected_invitees:
                social_action = SocialAction(
                    actor_id=agent_id,
                    action_type="organize",
                    target_id=invitee,
                    timestamp=self.simulation_time,
                    success=True,
                    impact_score=0.7,
                    description=f"组织{meeting_type}",
                    context={"meeting_type": meeting_type}
                )
                self.social_cognition.record_social_action(social_action)
            
            # 创建决策提案
            if meeting_type == "strategy_meeting":
                self._create_strategic_decision(agent_id, selected_invitees)
    
    def _execute_analysis(self, agent_id: str, agent: AgentCharacteristics):
        """执行分析行动"""
        # 分析当前部落状态
        analysis_type = random.choice([
            "performance_analysis", "resource_analysis", "social_analysis", "efficiency_analysis"
        ])
        
        # 生成分析报告
        analysis_result = self._perform_tribal_analysis(analysis_type)
        
        # 将分析结果存入集体记忆
        analysis_memory = MemoryEntry(
            id="",
            content={
                "analysis_type": analysis_type,
                "results": analysis_result,
                "recommendations": self._generate_analysis_recommendations(analysis_type, analysis_result)
            },
            memory_type="knowledge",
            timestamp=self.simulation_time,
            reliability_score=0.8,
            contributor_id=agent_id,
            verification_count=0,
            access_count=0,
            last_accessed=self.simulation_time,
            tags={"analysis", analysis_type}
        )
        
        self.collective_memory.store_memory(analysis_memory)
        
        agent.current_state = AgentState.DECIDING
        agent.current_activity = "analysis"
    
    def _record_discovery(self, agent_id: str, location: Tuple[float, float, float], discovery_type: str):
        """记录发现"""
        if discovery_type == "resource":
            resource_type = random.choice(["iron", "coal", "gold", "diamond", "wood"])
            memory = create_resource_hotspot_memory(
                location[0], location[1], location[2],
                resource_type, "moderate", 0.7, agent_id
            )
        elif discovery_type == "danger":
            danger_type = random.choice(["creeper_spawn", "lava_pool", "deep_pit", "monster_nest"])
            memory = create_danger_zone_memory(
                location[0], location[1], location[2],
                danger_type, f"发现的危险区域", agent_id
            )
        else:  # landmark
            memory = MemoryEntry(
                id="",
                content={
                    "type": "landmark",
                    "description": "探索发现的标志性地标",
                    "significance": "navigation_reference"
                },
                memory_type="knowledge",
                timestamp=self.simulation_time,
                reliability_score=0.6,
                contributor_id=agent_id,
                verification_count=0,
                access_count=0,
                last_accessed=self.simulation_time,
                tags={"landmark", "exploration"},
                spatial_coords=location
            )
        
        self.collective_memory.store_memory(memory)
    
    def _store_research_memory(self, agent_id: str, topic: str, learning_event: Dict[str, Any]):
        """存储研究记忆"""
        memory = MemoryEntry(
            id="",
            content={
                "topic": topic,
                "findings": f"关于{topic}的研究成果",
                "learning_event": learning_event,
                "applications": self._generate_applications(topic)
            },
            memory_type="knowledge",
            timestamp=self.simulation_time,
            reliability_score=0.8,
            contributor_id=agent_id,
            verification_count=0,
            access_count=0,
            last_accessed=self.simulation_time,
            tags={"research", topic}
        )
        
        self.collective_memory.store_memory(memory)
    
    def _share_information(self, sharer_id: str, recipient_id: str, info_type: str):
        """分享信息"""
        # 创建信息分享记忆
        memory = MemoryEntry(
            id="",
            content={
                "info_type": info_type,
                "sharer_id": sharer_id,
                "shared_at": self.simulation_time.isoformat(),
                "information": f"分享的{info_type}信息"
            },
            memory_type="knowledge",
            timestamp=self.simulation_time,
            reliability_score=0.7,
            contributor_id=sharer_id,
            verification_count=1,  # 已验证
            access_count=1,
            last_accessed=self.simulation_time,
            tags={"information_share", info_type}
        )
        
        memory_id = self.collective_memory.store_memory(memory)
        
        # 记录分享行为
        share_action = SocialAction(
            actor_id=sharer_id,
            action_type="share_information",
            target_id=recipient_id,
            timestamp=self.simulation_time,
            success=True,
            impact_score=0.4,
            description=f"分享{info_type}信息",
            context={"memory_id": memory_id, "info_type": info_type}
        )
        self.social_cognition.record_social_action(share_action)
    
    def _create_strategic_decision(self, proposer_id: str, invitees: List[str]):
        """创建战略决策"""
        decision = DecisionProposal(
            id="",
            decision_type=DecisionType.STRATEGIC_PLANNING,
            title="部落发展战略讨论",
            description="制定部落未来发展方向和策略",
            proposer_id=proposer_id,
            timestamp=self.simulation_time,
            arguments={proposer_id: ["促进部落发展", "提高整体效率"]},
            voting_deadline=self.simulation_time + timedelta(hours=6),
            required_quorum=max(3, len(invitees) // 2),
            decision_threshold=0.6
        )
        
        decision_id = self.collaboration_protocol.propose_decision(decision)
        
        # 模拟投票
        for agent_id in invitees:
            if agent_id != proposer_id:
                vote = 1 if random.random() < 0.8 else 0  # 80%支持率
                self.collaboration_protocol.cast_vote(decision_id, agent_id, vote)
        
        return decision_id
    
    def _perform_tribal_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """执行部落分析"""
        if analysis_type == "performance_analysis":
            metrics = self.collaboration_protocol.get_collaboration_metrics()
            return {
                "task_completion_rate": metrics.get("task_completion_rate", 0.0),
                "collaboration_efficiency": metrics.get("overall_collaboration_efficiency", 0.0),
                "trend": "improving" if metrics.get("task_completion_rate", 0.0) > 0.7 else "stable"
            }
        elif analysis_type == "resource_analysis":
            resource_stats = self.collaboration_protocol.get_system_status()["resources"]
            return {
                "total_resources": resource_stats["total"],
                "utilization_rate": len(self.collaboration_protocol.resource_allocation) / max(resource_stats["total"], 1),
                "shortages": random.randint(0, 3)  # 模拟短缺
            }
        elif analysis_type == "social_analysis":
            social_stats = self.social_cognition.get_system_statistics()
            return {
                "trust_level": social_stats.get("average_trust_level", 0.0),
                "cooperation_rate": social_stats.get("action_success_rate", 0.0),
                "social_cohesion": social_stats.get("network_analysis", {}).get("network_density", 0.0)
            }
        else:  # efficiency_analysis
            return {
                "overall_efficiency": random.uniform(0.6, 0.9),
                "bottlenecks": ["resource_allocation", "task_coordination"],
                "optimization_potential": 0.2
            }
    
    def _generate_analysis_recommendations(self, analysis_type: str, results: Dict[str, Any]) -> List[str]:
        """生成分析建议"""
        recommendations = {
            "performance_analysis": [
                "增加团队协作训练",
                "优化任务分配算法",
                "建立更好的激励机制"
            ],
            "resource_analysis": [
                "加强资源管理",
                "建立储备机制",
                "优化资源分配流程"
            ],
            "social_analysis": [
                "增进成员间沟通",
                "建立信任机制",
                "促进知识分享"
            ],
            "efficiency_analysis": [
                "简化工作流程",
                "提高决策速度",
                "减少重复工作"
            ]
        }
        
        return recommendations.get(analysis_type, ["需要更多数据分析"])
    
    def _generate_applications(self, topic: str) -> List[str]:
        """生成应用建议"""
        applications = {
            "world_mechanics": ["改善环境适应", "优化资源利用", "预测环境变化"],
            "resource_optimization": ["减少浪费", "提高效率", "发现新资源"],
            "social_dynamics": ["改善协作", "解决冲突", "增进团结"],
            "construction_techniques": ["建造更快", "质量更好", "成本更低"]
        }
        
        return applications.get(topic, ["进一步研究"])
    
    def _update_agent_state(self, agent_id: str, agent: AgentCharacteristics, action: str):
        """更新智能体状态"""
        # 能量消耗
        energy_costs = {
            "explore": 0.1, "construct": 0.15, "collaborate": 0.08, "research": 0.12,
            "communicate": 0.03, "help": 0.05, "organize": 0.1, "analyze": 0.08, "rest": -0.2
        }
        
        energy_cost = energy_costs.get(action, 0.05)
        agent.energy_level = max(0.0, min(1.0, agent.energy_level - energy_cost))
        
        # 如果能量过低，自动休息
        if agent.energy_level < 0.2:
            agent.current_state = AgentState.RESTING
            agent.current_activity = "rest"
    
    def _process_social_interactions(self):
        """处理社会交互"""
        # 模拟随机社交交互
        if random.random() < 0.3:  # 30%概率发生社交事件
            agents = list(self.agents.keys())
            if len(agents) >= 2:
                agent1, agent2 = random.sample(agents, 2)
                
                # 创建随机社交事件
                interaction_type = random.choice(["greeting", "resource_request", "information_share"])
                
                social_action = SocialAction(
                    actor_id=agent1,
                    action_type=interaction_type,
                    target_id=agent2,
                    timestamp=self.simulation_time,
                    success=random.random() < 0.8,  # 80%成功率
                    impact_score=random.uniform(0.2, 0.6),
                    description=f"随机社交交互",
                    context={"interaction_type": interaction_type}
                )
                
                self.social_cognition.record_social_action(social_action)
    
    def _update_collective_memory(self):
        """更新集体记忆"""
        # 定期清理过期记忆
        if self.simulation_step % 48 == 0:  # 每48步清理一次
            self.collective_memory._cleanup_memories()
        
        # 验证记忆可靠性
        if self.simulation_step % 24 == 0:  # 每24步验证一次
            for memory in list(self.collective_memory.memory_store.values()):
                if random.random() < 0.1:  # 10%概率被验证
                    verification_score = random.uniform(0.5, 1.0)
                    self.collective_memory.verify_memory(memory.id, "system", verification_score)
    
    def _process_task_execution(self):
        """处理任务执行"""
        # 更新任务进度
        for task_id, task in list(self.collaboration_protocol.tasks.items()):
            if task.status == "in_progress":
                # 随机推进任务进度
                progress_increment = random.uniform(0.05, 0.15)
                new_progress = min(1.0, task.progress + progress_increment)
                
                self.collaboration_protocol.update_task_progress(task_id, new_progress)
                
                # 检查是否完成
                if task.progress >= 1.0:
                    # 任务完成后的处理
                    self._handle_task_completion(task)
    
    def _handle_task_completion(self, task: Task):
        """处理任务完成"""
        if task.assigned_to:
            agent = self.agents.get(task.assigned_to)
            if agent:
                # 完成任务奖励
                agent.energy_level = min(1.0, agent.energy_level + 0.1)
                agent.motivation = min(1.0, agent.motivation + 0.05)
                
                # 根据任务类型更新技能
                skill_mapping = {
                    TaskType.EXPLORATION: "exploration",
                    TaskType.CONSTRUCTION: "construction",
                    TaskType.RESEARCH: "research",
                    TaskType.DEFENSE: "combat"
                }
                
                skill = skill_mapping.get(task.task_type)
                if skill:
                    agent.skills_learned.add(skill)
    
    def _calculate_collective_intelligence(self):
        """计算集体智能指标"""
        # 1. 社会网络密度
        social_analysis = self.social_cognition.analyze_social_network()
        network_density = social_analysis.get("network_density", 0.0)
        
        # 2. 知识共享程度
        knowledge_memories = self.collective_memory.retrieve_memories(memory_type="knowledge", limit=1000)
        knowledge_sharing_score = len(knowledge_memories) / max(len(self.agents), 1)
        
        # 3. 协作效率
        collaboration_metrics = self.collaboration_protocol.get_collaboration_metrics()
        collaboration_efficiency = collaboration_metrics.get("overall_collaboration_efficiency", 0.0)
        
        # 4. 适应性评分
        adaptability_score = self._calculate_adaptability()
        
        # 5. 创新能力
        innovation_score = self._calculate_innovation_score()
        
        # 综合集体智能评分
        collective_intelligence = (
            network_density * 0.2 +
            knowledge_sharing_score * 0.2 +
            collaboration_efficiency * 0.3 +
            adaptability_score * 0.15 +
            innovation_score * 0.15
        )
        
        self.collective_intelligence_metrics = {
            "collective_intelligence": collective_intelligence,
            "network_density": network_density,
            "knowledge_sharing": knowledge_sharing_score,
            "collaboration_efficiency": collaboration_efficiency,
            "adaptability": adaptability_score,
            "innovation": innovation_score,
            "timestamp": self.simulation_time.isoformat()
        }
    
    def _calculate_adaptability(self) -> float:
        """计算适应性评分"""
        # 基于智能体对新环境的反应能力
        adaptability_scores = []
        
        for agent in self.agents.values():
            # 基于学习速度和风险容忍度计算适应性
            adaptability = (
                agent.learning_speed * 0.6 +
                agent.risk_tolerance * 0.4
            )
            adaptability_scores.append(adaptability)
        
        return np.mean(adaptability_scores) if adaptability_scores else 0.5
    
    def _calculate_innovation_score(self) -> float:
        """计算创新能力评分"""
        # 基于创造力、协作倾向和知识基础
        creativity_scores = []
        
        for agent in self.agents.values():
            innovation_base = agent.creativity
            
            # 考虑协作影响
            social_bonus = agent.social_tendency * 0.2
            knowledge_bonus = len(agent.skills_learned) * 0.1
            
            innovation = min(1.0, innovation_base + social_bonus + knowledge_bonus)
            creativity_scores.append(innovation)
        
        return np.mean(creativity_scores) if creativity_scores else 0.5
    
    def _trigger_world_events(self):
        """触发世界事件"""
        event_types = [
            "resource_discovery", "danger_emergence", "weather_change", 
            "population_growth", "technology_breakthrough"
        ]
        
        event_type = random.choice(event_types)
        
        if event_type == "resource_discovery":
            self._trigger_resource_discovery_event()
        elif event_type == "danger_emergence":
            self._trigger_danger_emergence_event()
        elif event_type == "technology_breakthrough":
            self._trigger_technology_breakthrough_event()
    
    def _trigger_resource_discovery_event(self):
        """触发资源发现事件"""
        # 随机位置发现新资源
        new_location = (
            random.uniform(-200, 200),
            random.uniform(-50, 100),
            random.uniform(-200, 200)
        )
        
        resource_type = random.choice(["diamond", "ancient_artifacts", "mystical_crystals", "rare_elements"])
        
        memory = create_resource_hotspot_memory(
            new_location[0], new_location[1], new_location[2],
            resource_type, "abundant", 0.9, "world_event"
        )
        
        self.collective_memory.store_memory(memory)
        
        # 记录事件
        event = {
            "type": "resource_discovery",
            "description": f"发现新的{resource_type}资源",
            "location": new_location,
            "timestamp": self.simulation_time.isoformat(),
            "impact": "positive"
        }
        self._log_event(event)
    
    def _trigger_danger_emergence_event(self):
        """触发危险出现事件"""
        # 随机位置出现新危险
        danger_location = (
            random.uniform(-100, 100),
            random.uniform(-20, 50),
            random.uniform(-100, 100)
        )
        
        danger_type = random.choice(["toxic_spores", "hostile_creatures", "environmental_hazard"])
        
        memory = create_danger_zone_memory(
            danger_location[0], danger_location[1], danger_location[2],
            danger_type, "新出现的危险", "world_event"
        )
        
        self.collective_memory.store_memory(memory)
        
        # 记录事件
        event = {
            "type": "danger_emergence",
            "description": f"新危险{danger_type}出现",
            "location": danger_location,
            "timestamp": self.simulation_time.isoformat(),
            "impact": "negative"
        }
        self._log_event(event)
    
    def _trigger_technology_breakthrough_event(self):
        """触发技术突破事件"""
        # 随机技术突破
        breakthrough_tech = random.choice([
            "advanced_architecture", "efficient_resource_extraction",
            "defensive_systems", "communication_networks"
        ])
        
        memory = MemoryEntry(
            id="",
            content={
                "technology": breakthrough_tech,
                "description": f"部落实现了{breakthrough_tech}技术突破",
                "benefits": self._generate_technology_benefits(breakthrough_tech)
            },
            memory_type="knowledge",
            timestamp=self.simulation_time,
            reliability_score=0.9,
            contributor_id="technology_breakthrough",
            verification_count=1,
            access_count=0,
            last_accessed=self.simulation_time,
            tags={"technology", "breakthrough", breakthrough_tech}
        )
        
        self.collective_memory.store_memory(memory)
        
        # 记录事件
        event = {
            "type": "technology_breakthrough",
            "description": f"技术突破: {breakthrough_tech}",
            "timestamp": self.simulation_time.isoformat(),
            "impact": "positive"
        }
        self._log_event(event)
    
    def _generate_technology_benefits(self, technology: str) -> List[str]:
        """生成技术益处"""
        benefits = {
            "advanced_architecture": ["建造速度提升50%", "结构更稳固", "设计更美观"],
            "efficient_resource_extraction": ["资源利用率提高30%", "减少浪费", "发现新资源"],
            "defensive_systems": ["安全系数提升", "早期预警", "自动防护"],
            "communication_networks": ["信息传递更快速", "协作更高效", "知识共享更便利"]
        }
        
        return benefits.get(technology, ["效率提升"])
    
    def _log_current_state(self):
        """记录当前状态"""
        current_state = {
            "simulation_step": self.simulation_step,
            "simulation_time": self.simulation_time.isoformat(),
            "agent_states": {
                agent_id: {
                    "state": agent.current_state.value,
                    "energy": agent.energy_level,
                    "activity": agent.current_activity,
                    "location": agent.location
                }
                for agent_id, agent in self.agents.items()
            },
            "collective_metrics": self.collective_intelligence_metrics,
            "system_status": self.collaboration_protocol.get_system_status(),
            "memory_stats": self.collective_memory.get_memory_statistics()
        }
        
        self.event_log.append(current_state)
    
    def _log_event(self, event: Dict[str, Any]):
        """记录特定事件"""
        event_log_entry = {
            "event": event,
            "timestamp": self.simulation_time.isoformat(),
            "step": self.simulation_step
        }
        self.event_log.append(event_log_entry)
    
    def _record_performance_metrics(self):
        """记录性能指标"""
        metrics = {
            "step": self.simulation_step,
            "timestamp": self.simulation_time.isoformat(),
            "collective_intelligence": self.collective_intelligence_metrics.get("collective_intelligence", 0.0),
            "collaboration_metrics": self.collaboration_protocol.get_collaboration_metrics(),
            "social_metrics": self.social_cognition.get_system_statistics(),
            "memory_utilization": self.collective_memory.get_memory_statistics(),
            "agent_health": {
                "avg_energy": np.mean([agent.energy_level for agent in self.agents.values()]),
                "avg_motivation": np.mean([agent.motivation for agent in self.agents.values()]),
                "active_agents": len([agent for agent in self.agents.values() if agent.current_state != AgentState.RESTING])
            }
        }
        
        self.performance_history.append(metrics)
        
        logger.info(f"步骤 {self.simulation_step}: 集体智能评分 {metrics['collective_intelligence']:.3f}")
    
    def get_tribal_status(self) -> Dict[str, Any]:
        """获取部落整体状态"""
        with self.lock:
            return {
                "basic_info": {
                    "agent_count": len(self.agents),
                    "simulation_step": self.simulation_step,
                    "simulation_time": self.simulation_time.isoformat(),
                    "running": self.is_running
                },
                "collective_metrics": self.collective_intelligence_metrics,
                "system_status": self.collaboration_protocol.get_system_status(),
                "social_analysis": self.social_cognition.get_system_statistics(),
                "memory_analysis": self.collective_memory.get_memory_statistics(),
                "agent_overview": {
                    agent_id: {
                        "personality": agent.personality.value,
                        "state": agent.current_state.value,
                        "energy": agent.energy_level,
                        "motivation": agent.motivation,
                        "skills": list(agent.skills_learned),
                        "location": agent.location
                    }
                    for agent_id, agent in self.agents.items()
                },
                "performance_trend": self.performance_history[-10:] if self.performance_history else []
            }
    
    def export_simulation_data(self, filepath: str):
        """导出模拟数据"""
        export_data = {
            "tribal_society": {
                "basic_info": {
                    "agent_count": len(self.agents),
                    "simulation_step": self.simulation_step,
                    "simulation_time": self.simulation_time.isoformat(),
                    "creation_time": datetime.now().isoformat()
                },
                "agents": {
                    agent_id: {
                        **asdict(agent),
                        "skills_learned": list(agent.skills_learned),
                        "personality": agent.personality.value,
                        "current_state": agent.current_state.value
                    }
                    for agent_id, agent in self.agents.items()
                },
                "collective_memory": self.collective_memory.get_memory_statistics(),
                "social_network": self.social_cognition.get_system_statistics(),
                "collaboration_metrics": self.collaboration_protocol.get_collaboration_metrics(),
                "collective_intelligence": self.collective_intelligence_metrics,
                "performance_history": self.performance_history,
                "event_log": self.event_log
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"模拟数据已导出到: {filepath}")
    
    def stop_simulation(self):
        """停止模拟"""
        self.is_running = False
        logger.info("停止部落模拟")
    
    def run_demo_simulation(self, duration_hours: int = 72):
        """运行演示模拟"""
        logger.info(f"开始{duration_hours}小时演示模拟")
        
        # 启动模拟
        self.start_simulation()
        
        # 运行指定时间
        time.sleep(duration_hours * 0.1)  # 快速运行演示
        
        # 停止模拟
        self.stop_simulation()
        
        # 导出结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"/workspace/NeuroMinecraftGenesis/agents/multi/simulation_results_{timestamp}.json"
        self.export_simulation_data(export_path)
        
        # 生成报告
        self.generate_analysis_report(export_path)
        
        return export_path
    
    def generate_analysis_report(self, data_file: str):
        """生成分析报告"""
        # 读取模拟数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 生成分析报告
        report = f"""# 多智能体部落系统分析报告

## 1. 模拟概况
- 智能体数量: {data['tribal_society']['basic_info']['agent_count']}
- 模拟步数: {data['tribal_society']['basic_info']['simulation_step']}
- 模拟时间: {data['tribal_society']['basic_info']['simulation_time']}

## 2. 集体智能分析
"""
        
        if data['tribal_society']['collective_intelligence']:
            metrics = data['tribal_society']['collective_intelligence']
            report += f"""### 核心指标
- 集体智能评分: {metrics.get('collective_intelligence', 0):.3f}
- 社会网络密度: {metrics.get('network_density', 0):.3f}
- 知识共享程度: {metrics.get('knowledge_sharing', 0):.3f}
- 协作效率: {metrics.get('collaboration_efficiency', 0):.3f}
- 适应性: {metrics.get('adaptability', 0):.3f}
- 创新能力: {metrics.get('innovation', 0):.3f}

"""
        
        if data['tribal_society']['collaboration_metrics']:
            collab = data['tribal_society']['collaboration_metrics']
            report += f"""## 3. 协作性能
- 任务完成率: {collab.get('task_completion_rate', 0):.1%}
- 资源利用率: {collab.get('resource_utilization_rate', 0):.1%}
- 冲突解决率: {collab.get('conflict_resolution_rate', 0):.1%}
- 决策成功率: {collab.get('decision_success_rate', 0):.1%}
- 综合协作效率: {collab.get('overall_collaboration_efficiency', 0):.3f}

"""
        
        if data['tribal_society']['social_network']:
            social = data['tribal_society']['social_network']
            report += f"""## 4. 社会网络分析
- 总交互次数: {social.get('total_social_actions', 0)}
- 交互成功率: {social.get('action_success_rate', 0):.1%}
- 平均信任度: {social.get('average_trust_level', 0):.3f}
- 学习事件: {social.get('learning_events', 0)}

"""
        
        # 保存报告
        report_path = data_file.replace('.json', '_analysis.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析报告已保存到: {report_path}")
        return report_path


# 便捷函数
def create_tribal_society_with_config(config: Dict[str, Any] = None) -> TribalSociety:
    """根据配置创建部落社会"""
    agent_count = config.get('agent_count', 16) if config else 16
    return TribalSociety(agent_count=agent_count)

def run_comprehensive_demo() -> str:
    """运行综合演示"""
    logger.info("开始多智能体社会系统综合演示")
    
    # 创建部落
    tribe = TribalSociety(agent_count=16)
    
    # 运行24小时演示
    return tribe.run_demo_simulation(duration_hours=24)

# 导入便捷函数
try:
    from collaboration_protocol import create_simple_task
except ImportError:
    # 如果导入失败，定义一个本地版本
    def create_simple_task(task_type, title, priority=TaskPriority.MEDIUM, duration=8):
        """创建简单任务"""
        return Task(
            id="",
            task_type=task_type,
            title=title,
            description=f"{title}任务",
            priority=priority,
            estimated_duration=duration,
            required_skills=[],
            required_resources={},
            created_by="tribal_system"
        )