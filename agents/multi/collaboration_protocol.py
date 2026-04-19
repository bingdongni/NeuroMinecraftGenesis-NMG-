"""
协作协议系统 (Collaboration Protocol System)
实现任务分配、资源分享、冲突解决和集体决策机制

功能特性：
- 任务分配：根据能力分配不同任务
- 资源分享：部落内资源动态分配
- 冲突解决：智能体间冲突的智能解决机制
- 集体决策：重要决策的集体投票机制
- 协作效率优化
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

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    EXPLORATION = "exploration"  # 探索
    RESOURCE_COLLECTION = "resource_collection"  # 资源收集
    CONSTRUCTION = "construction"  # 建造
    DEFENSE = "defense"  # 防御
    RESEARCH = "research"  # 研究
    RESCUE = "rescue"  # 救援
    MAINTENANCE = "maintenance"  # 维护
    LEADERSHIP = "leadership"  # 领导

class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 5  # 紧急
    HIGH = 4      # 高
    MEDIUM = 3    # 中
    LOW = 2       # 低
    MINIMAL = 1   # 最低

class ResourceType(Enum):
    """资源类型枚举"""
    MATERIAL = "material"  # 材料
    TOOL = "tool"  # 工具
    FOOD = "food"  # 食物
    ENERGY = "energy"  # 能源
    TIME = "time"  # 时间
    INFORMATION = "information"  # 信息
    SPACE = "space"  # 空间

class ConflictType(Enum):
    """冲突类型枚举"""
    RESOURCE_COMPETITION = "resource_competition"
    TASK_ASSIGNMENT = "task_assignment"
    TERRITORY = "territory"
    LEADERSHIP = "leadership"
    KNOWLEDGE_DISPUTE = "knowledge_dispute"
    PERSONAL_DIFFERENCES = "personal_differences"

class DecisionType(Enum):
    """决策类型枚举"""
    EMERGENCY_RESPONSE = "emergency_response"
    RESOURCE_ALLOCATION = "resource_allocation"
    STRATEGIC_PLANNING = "strategic_planning"
    RULE_ESTABLISHMENT = "rule_establishment"
    LEADERSHIP_ELECTION = "leadership_election"
    PROJECT_INITIATION = "project_initiation"

@dataclass
class Task:
    """任务对象"""
    id: str
    task_type: TaskType
    title: str
    description: str
    priority: TaskPriority
    estimated_duration: int  # 小时
    required_skills: List[str]
    required_resources: Dict[ResourceType, int]
    location: Optional[Tuple[float, float, float]] = None
    deadline: Optional[datetime] = None
    created_by: str = ""
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, cancelled
    progress: float = 0.0  # 0.0-1.0
    dependencies: List[str] = None  # 依赖的任务ID
    reward: float = 0.0
    
    def __post_init__(self):
        if not self.dependencies:
            self.dependencies = []

@dataclass
class Resource:
    """资源对象"""
    id: str
    resource_type: ResourceType
    name: str
    quantity: int
    quality: float  # 0.0-1.0
    location: Optional[Tuple[float, float, float]] = None
    owner: Optional[str] = None
    shared: bool = False
    accessibility: str = "medium"  # easy, medium, difficult
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

@dataclass
class Conflict:
    """冲突对象"""
    id: str
    conflict_type: ConflictType
    description: str
    involved_agents: List[str]
    timestamp: datetime
    severity: int  # 1-5
    resolved: bool = False
    resolution_method: Optional[str] = None
    resolution_details: Dict[str, Any] = None
    resolution_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.resolution_details:
            self.resolution_details = {}

@dataclass
class DecisionProposal:
    """决策提案对象"""
    id: str
    decision_type: DecisionType
    title: str
    description: str
    proposer_id: str
    timestamp: datetime
    arguments: Dict[str, List[str]]  # 智能体ID -> 支持论据
    voting_deadline: datetime
    required_quorum: int  # 最小投票人数
    decision_threshold: float  # 决策通过阈值 (0.0-1.0)
    status: str = "active"  # active, voted, implemented, cancelled
    votes: Dict[str, int] = None  # 智能体ID -> 投票 (1=支持, -1=反对, 0=弃权)
    
    def __post_init__(self):
        if not self.votes:
            self.votes = {}

class CollaborationProtocol:
    """
    协作协议系统核心类
    
    管理多智能体间的协作流程、任务分配和冲突解决
    """
    
    def __init__(self, agent_count: int = 16):
        self.agent_count = agent_count
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self.conflicts: Dict[str, Conflict] = {}
        self.decisions: Dict[str, DecisionProposal] = {}
        
        # 任务分配历史
        self.task_history: List[Task] = []
        self.assignment_success_rate: Dict[str, float] = defaultdict(float)
        
        # 资源分配状态
        self.resource_allocation: Dict[str, Dict[str, int]] = defaultdict(dict)  # resource_id -> agent_id -> quantity
        self.resource_sharing_networks: Dict[str, Set[str]] = defaultdict(set)
        
        # 冲突解决历史
        self.conflict_resolution_methods: Dict[ConflictType, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.resolution_effectiveness: Dict[str, float] = {}
        
        # 决策历史
        self.decision_history: List[DecisionProposal] = []
        self.decision_outcomes: Dict[str, Dict[str, Any]] = {}
        
        # 能力矩阵：agent_id -> skill_level
        self.agent_capabilities: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # 协作效率统计
        self.collaboration_metrics: Dict[str, float] = defaultdict(float)
        
        # 并发控制
        self.lock = threading.RLock()
        
        # 初始化能力矩阵
        self._initialize_capabilities()
        
        logger.info("协作协议系统初始化完成")
    
    def _initialize_capabilities(self):
        """初始化智能体能力"""
        skill_names = ["exploration", "construction", "combat", "research", "leadership", "coordination", "innovation"]
        
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            # 为每个智能体随机分配基础技能水平
            for skill in skill_names:
                # 技能水平遵循正态分布，平均0.6，标准差0.2
                skill_level = max(0.1, min(1.0, random.gauss(0.6, 0.2)))
                self.agent_capabilities[agent_id][skill] = skill_level
    
    def create_task(self, task: Task) -> str:
        """
        创建新任务
        
        Args:
            task: 任务对象
            
        Returns:
            str: 任务ID
        """
        with self.lock:
            if not task.id:
                task.id = f"task_{len(self.tasks)}_{int(time.time() * 1000)}"
            
            self.tasks[task.id] = task
            logger.info(f"创建任务: {task.title} (ID: {task.id})")
            
            return task.id
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """
        分配任务给智能体
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            
        Returns:
            bool: 分配是否成功
        """
        with self.lock:
            if task_id not in self.tasks or agent_id not in [f"agent_{i}" for i in range(self.agent_count)]:
                return False
            
            task = self.tasks[task_id]
            
            # 检查任务状态
            if task.status != "pending":
                return False
            
            # 评估智能体适合性
            suitability_score = self._evaluate_agent_suitability(agent_id, task)
            
            if suitability_score < 0.3:  # 最低适合度阈值
                logger.info(f"智能体 {agent_id} 不适合任务 {task_id} (适合度: {suitability_score:.2f})")
                return False
            
            # 分配任务
            task.assigned_to = agent_id
            task.status = "assigned"
            
            # 分配所需资源
            success = self._allocate_resources_for_task(task, agent_id)
            
            if success:
                logger.info(f"任务分配成功: {agent_id} <- {task.title}")
                
                # 更新成功率统计
                self._update_assignment_success_rate(agent_id, True)
                return True
            else:
                logger.warning(f"资源分配失败，任务分配回滚")
                task.assigned_to = None
                task.status = "pending"
                self._update_assignment_success_rate(agent_id, False)
                return False
    
    def _evaluate_agent_suitability(self, agent_id: str, task: Task) -> float:
        """评估智能体对任务的适合度"""
        if agent_id not in self.agent_capabilities:
            return 0.0
        
        capabilities = self.agent_capabilities[agent_id]
        
        # 技能匹配度
        skill_match = 0.0
        for skill in task.required_skills:
            skill_level = capabilities.get(skill, 0.0)
            skill_match += skill_level
        
        if task.required_skills:
            skill_match /= len(task.required_skills)
        
        # 工作负载考虑
        current_workload = self._get_agent_workload(agent_id)
        workload_factor = max(0.1, 1.0 - current_workload * 0.5)
        
        # 历史表现
        historical_performance = self.assignment_success_rate.get(agent_id, 0.5)
        
        # 综合评分
        suitability = (
            skill_match * 0.5 +
            workload_factor * 0.3 +
            historical_performance * 0.2
        )
        
        return suitability
    
    def _get_agent_workload(self, agent_id: str) -> float:
        """获取智能体当前工作负载"""
        active_tasks = [task for task in self.tasks.values() 
                       if task.assigned_to == agent_id and task.status in ["assigned", "in_progress"]]
        
        # 计算工作量（基于任务优先级和持续时间）
        total_workload = 0.0
        for task in active_tasks:
            priority_weight = task.priority.value / 5.0  # 归一化优先级
            duration_weight = min(task.estimated_duration / 48.0, 1.0)  # 48小时为满负载
            total_workload += priority_weight * duration_weight
        
        return min(1.0, total_workload)
    
    def _allocate_resources_for_task(self, task: Task, agent_id: str) -> bool:
        """为任务分配所需资源"""
        for resource_type, required_quantity in task.required_resources.items():
            # 查找可用资源
            available_resources = self._find_available_resources(resource_type, required_quantity)
            
            if not available_resources:
                logger.warning(f"资源不足: {resource_type} 需要 {required_quantity}")
                return False
            
            # 分配资源
            allocated = 0
            for resource_id, available_quantity in available_resources:
                if allocated >= required_quantity:
                    break
                
                resource = self.resources.get(resource_id)
                if not resource:
                    continue
                
                # 计算分配数量
                allocation_quantity = min(available_quantity, required_quantity - allocated)
                
                # 更新资源分配状态
                if resource_id not in self.resource_allocation:
                    self.resource_allocation[resource_id] = {}
                
                self.resource_allocation[resource_id][agent_id] = (
                    self.resource_allocation[resource_id].get(agent_id, 0) + allocation_quantity
                )
                
                # 更新资源持有量
                resource.quantity -= allocation_quantity
                if resource.quantity <= 0:
                    resource.shared = False  # 如果耗尽，不再共享
                
                allocated += allocation_quantity
                logger.debug(f"分配资源: {resource_id} -> {agent_id} ({allocation_quantity})")
        
        return True
    
    def _find_available_resources(self, resource_type: ResourceType, required_quantity: int) -> List[Tuple[str, int]]:
        """查找可用资源"""
        available = []
        
        for resource_id, resource in self.resources.items():
            if (resource.resource_type == resource_type and 
                resource.quantity > 0 and 
                (resource.owner is None or resource.owner == "shared")):
                available.append((resource_id, resource.quantity))
        
        # 按质量和数量排序
        available.sort(key=lambda x: (self.resources[x[0]].quality, x[1]), reverse=True)
        
        return available
    
    def _update_assignment_success_rate(self, agent_id: str, success: bool):
        """更新任务分配成功率"""
        current_rate = self.assignment_success_rate.get(agent_id, 0.5)
        alpha = 0.1  # 学习率
        
        if success:
            self.assignment_success_rate[agent_id] = (1 - alpha) * current_rate + alpha * 1.0
        else:
            self.assignment_success_rate[agent_id] = (1 - alpha) * current_rate + alpha * 0.0
    
    def get_task_recommendations(self, agent_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        为智能体推荐合适的任务
        
        Args:
            agent_id: 智能体ID
            limit: 推荐数量限制
            
        Returns:
            List[Tuple[str, float]]: (任务ID, 适合度评分) 列表
        """
        with self.lock:
            recommendations = []
            
            for task_id, task in self.tasks.items():
                if (task.status == "pending" and 
                    task.assigned_to is None):
                    
                    suitability_score = self._evaluate_agent_suitability(agent_id, task)
                    if suitability_score >= 0.3:
                        recommendations.append((task_id, suitability_score))
            
            # 按适合度排序
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:limit]
    
    def update_task_progress(self, task_id: str, progress: float) -> bool:
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 新进度 (0.0-1.0)
            
        Returns:
            bool: 更新是否成功
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.progress = max(0.0, min(1.0, progress))
            
            if task.progress >= 1.0:
                task.status = "completed"
                self.task_history.append(task)
                logger.info(f"任务完成: {task.title} (ID: {task_id})")
            elif task.progress > 0.0 and task.status == "assigned":
                task.status = "in_progress"
            
            return True
    
    def share_resource(self, resource_id: str, sharer_id: str, 
                      recipient_id: str, quantity: int) -> bool:
        """
        分享资源
        
        Args:
            resource_id: 资源ID
            sharer_id: 分享者ID
            recipient_id: 接受者ID
            quantity: 分享数量
            
        Returns:
            bool: 分享是否成功
        """
        with self.lock:
            if resource_id not in self.resources or sharer_id == recipient_id:
                return False
            
            resource = self.resources[resource_id]
            
            # 检查分享权限
            if not (resource.owner == sharer_id or resource.shared):
                return False
            
            # 检查资源充足性
            allocated_quantity = self.resource_allocation.get(resource_id, {}).get(sharer_id, 0)
            available_quantity = resource.quantity - allocated_quantity
            
            if available_quantity < quantity:
                logger.warning(f"资源不足分享: {resource_id} 需要 {quantity}, 可用 {available_quantity}")
                return False
            
            # 执行分享
            if resource_id not in self.resource_allocation:
                self.resource_allocation[resource_id] = {}
            
            # 减少分享者的持有
            self.resource_allocation[resource_id][sharer_id] -= quantity
            
            # 增加接受者的持有
            if recipient_id not in self.resource_allocation[resource_id]:
                self.resource_allocation[resource_id][recipient_id] = 0
            self.resource_allocation[resource_id][recipient_id] += quantity
            
            # 建立共享网络
            self.resource_sharing_networks[resource_id].add(sharer_id)
            self.resource_sharing_networks[resource_id].add(recipient_id)
            
            logger.info(f"资源分享: {sharer_id} -> {recipient_id} ({quantity}x {resource.name})")
            return True
    
    def create_resource(self, resource: Resource) -> str:
        """创建资源"""
        with self.lock:
            if not resource.id:
                resource.id = f"resource_{len(self.resources)}_{int(time.time() * 1000)}"
            
            self.resources[resource.id] = resource
            logger.info(f"创建资源: {resource.name} (ID: {resource.id})")
            
            return resource.id
    
    def resolve_conflict(self, conflict: Conflict) -> str:
        """
        解决冲突
        
        Args:
            conflict: 冲突对象
            
        Returns:
            str: 冲突解决ID
        """
        with self.lock:
            if not conflict.id:
                conflict.id = f"conflict_{len(self.conflicts)}_{int(time.time() * 1000)}"
            
            self.conflicts[conflict.id] = conflict
            logger.info(f"记录冲突: {conflict.description} (ID: {conflict.id})")
            
            # 选择解决策略
            resolution_method = self._select_resolution_strategy(conflict)
            
            # 执行解决
            resolution_success = self._execute_resolution(conflict, resolution_method)
            
            # 记录结果
            conflict.resolved = resolution_success
            conflict.resolution_method = resolution_method
            conflict.resolution_timestamp = datetime.now()
            
            if resolution_success:
                logger.info(f"冲突解决成功: {conflict.id} (方法: {resolution_method})")
            else:
                logger.warning(f"冲突解决失败: {conflict.id}")
            
            # 更新统计
            self.conflict_resolution_methods[conflict.conflict_type][resolution_method] += 1
            
            return conflict.id
    
    def _select_resolution_strategy(self, conflict: Conflict) -> str:
        """选择冲突解决策略"""
        strategies = {
            ConflictType.RESOURCE_COMPETITION: [
                "resource_redistribution", "time_sharing", "alternative_sourcing"
            ],
            ConflictType.TASK_ASSIGNMENT: [
                "skill_based_reassignment", "rotation_system", "collaboration_requirement"
            ],
            ConflictType.TERRITORY: [
                "territory_demarcation", "time_division", "joint_usage"
            ],
            ConflictType.LEADERSHIP: [
                "democratic_vote", "performance_based_selection", "rotation_system"
            ],
            ConflictType.KNOWLEDGE_DISPUTE: [
                "evidence_based_arbitration", "expert_consultation", "experimental_validation"
            ],
            ConflictType.PERSONAL_DIFFERENCES: [
                "mediation_session", "communication_training", "role_separation"
            ]
        }
        
        available_strategies = strategies.get(conflict.conflict_type, ["arbitration"])
        
        # 根据冲突严重程度和历史效果选择策略
        if conflict.severity >= 4:  # 严重冲突
            preferred = ["democratic_vote", "evidence_based_arbitration", "mediation_session"]
            for strategy in preferred:
                if strategy in available_strategies:
                    return strategy
        
        # 默认返回第一个可用策略
        return available_strategies[0]
    
    def _execute_resolution(self, conflict: Conflict, method: str) -> bool:
        """执行冲突解决"""
        try:
            if method == "resource_redistribution":
                return self._resolve_resource_redistribution(conflict)
            elif method == "skill_based_reassignment":
                return self._resolve_skill_reassignment(conflict)
            elif method == "democratic_vote":
                return self._resolve_democratic_vote(conflict)
            elif method == "evidence_based_arbitration":
                return self._resolve_evidence_arbitration(conflict)
            elif method == "communication_training":
                return self._resolve_communication_training(conflict)
            else:
                return self._resolve_arbitration(conflict)
        except Exception as e:
            logger.error(f"冲突解决执行失败: {e}")
            return False
    
    def _resolve_resource_redistribution(self, conflict: Conflict) -> bool:
        """资源重新分配解决"""
        # 简化实现：重新分配资源
        return True
    
    def _resolve_skill_reassignment(self, conflict: Conflict) -> bool:
        """基于技能的重新分配"""
        # 基于技能匹配重新分配任务
        return True
    
    def _resolve_democratic_vote(self, conflict: Conflict) -> bool:
        """民主投票解决"""
        # 创建投票决策
        decision = DecisionProposal(
            id=f"conflict_resolution_{conflict.id}",
            decision_type=DecisionType.STRATEGIC_PLANNING,
            title=f"解决冲突: {conflict.description}",
            description=f"通过投票解决冲突: {conflict.description}",
            proposer_id=conflict.involved_agents[0] if conflict.involved_agents else "system",
            timestamp=datetime.now(),
            arguments={},
            voting_deadline=datetime.now() + timedelta(hours=1),
            required_quorum=max(3, len(conflict.involved_agents) // 2),
            decision_threshold=0.6
        )
        
        self.decisions[decision.id] = decision
        return True
    
    def _resolve_evidence_arbitration(self, conflict: Conflict) -> bool:
        """基于证据的仲裁"""
        # 基于事实和证据进行仲裁
        # 简化实现
        return True
    
    def _resolve_communication_training(self, conflict: Conflict) -> bool:
        """沟通训练解决"""
        # 为冲突双方提供沟通训练
        return True
    
    def _resolve_arbitration(self, conflict: Conflict) -> bool:
        """默认仲裁解决"""
        # 系统仲裁
        return True
    
    def propose_decision(self, proposal: DecisionProposal) -> str:
        """
        提出决策提案
        
        Args:
            proposal: 决策提案对象
            
        Returns:
            str: 提案ID
        """
        with self.lock:
            if not proposal.id:
                proposal.id = f"decision_{len(self.decisions)}_{int(time.time() * 1000)}"
            
            self.decisions[proposal.id] = proposal
            logger.info(f"提出决策: {proposal.title} (ID: {proposal.id})")
            
            return proposal.id
    
    def cast_vote(self, decision_id: str, agent_id: str, vote: int) -> bool:
        """
        投票
        
        Args:
            decision_id: 决策ID
            agent_id: 智能体ID
            vote: 投票 (-1=反对, 0=弃权, 1=支持)
            
        Returns:
            bool: 投票是否成功
        """
        with self.lock:
            if decision_id not in self.decisions or agent_id not in [f"agent_{i}" for i in range(self.agent_count)]:
                return False
            
            decision = self.decisions[decision_id]
            
            if datetime.now() > decision.voting_deadline:
                return False
            
            decision.votes[agent_id] = vote
            logger.debug(f"投票: {agent_id} -> {decision_id} ({vote})")
            
            return True
    
    def finalize_decision(self, decision_id: str) -> Dict[str, Any]:
        """
        结束决策并计算结果
        
        Args:
            decision_id: 决策ID
            
        Returns:
            Dict[str, Any]: 决策结果
        """
        with self.lock:
            if decision_id not in self.decisions:
                return {"success": False, "reason": "Decision not found"}
            
            decision = self.decisions[decision_id]
            
            if decision.status != "active":
                return {"success": False, "reason": "Decision not active"}
            
            # 检查投票人数
            total_votes = len(decision.votes)
            if total_votes < decision.required_quorum:
                return {"success": False, "reason": "Insufficient quorum"}
            
            # 计算投票结果
            total_support = sum(1 for vote in decision.votes.values() if vote == 1)
            total_oppose = sum(1 for vote in decision.votes.values() if vote == -1)
            total_abstain = sum(1 for vote in decision.votes.values() if vote == 0)
            
            support_ratio = total_support / total_votes if total_votes > 0 else 0
            
            # 判断决策是否通过
            decision_passed = support_ratio >= decision.decision_threshold
            
            # 更新决策状态
            decision.status = "voted" if not decision_passed else "implemented"
            
            # 记录决策结果
            result = {
                "success": decision_passed,
                "total_votes": total_votes,
                "support": total_support,
                "oppose": total_oppose,
                "abstain": total_abstain,
                "support_ratio": support_ratio,
                "threshold": decision.decision_threshold,
                "timestamp": datetime.now()
            }
            
            self.decision_history.append(decision)
            self.decision_outcomes[decision_id] = result
            
            logger.info(f"决策结果: {decision.title} - {'通过' if decision_passed else '否决'}")
            
            return result
    
    def get_collaboration_metrics(self) -> Dict[str, float]:
        """
        获取协作指标
        
        Returns:
            Dict[str, float]: 协作性能指标
        """
        with self.lock:
            metrics = {}
            
            # 任务完成率
            completed_tasks = sum(1 for task in self.tasks.values() if task.status == "completed")
            total_tasks = len(self.tasks)
            metrics["task_completion_rate"] = completed_tasks / total_tasks if total_tasks > 0 else 0.0
            
            # 资源利用率
            total_resources = sum(resource.quantity for resource in self.resources.values())
            allocated_resources = sum(
                sum(allocations.values()) for allocations in self.resource_allocation.values()
            )
            metrics["resource_utilization_rate"] = allocated_resources / total_resources if total_resources > 0 else 0.0
            
            # 冲突解决率
            resolved_conflicts = sum(1 for conflict in self.conflicts.values() if conflict.resolved)
            total_conflicts = len(self.conflicts)
            metrics["conflict_resolution_rate"] = resolved_conflicts / total_conflicts if total_conflicts > 0 else 0.0
            
            # 决策成功率
            successful_decisions = sum(
                1 for outcome in self.decision_outcomes.values() if outcome.get("success", False)
            )
            total_decisions = len(self.decision_outcomes)
            metrics["decision_success_rate"] = successful_decisions / total_decisions if total_decisions > 0 else 0.0
            
            # 协作效率（综合指标）
            metrics["overall_collaboration_efficiency"] = (
                metrics["task_completion_rate"] * 0.3 +
                metrics["resource_utilization_rate"] * 0.3 +
                metrics["conflict_resolution_rate"] * 0.2 +
                metrics["decision_success_rate"] * 0.2
            )
            
            return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            return {
                "tasks": {
                    "total": len(self.tasks),
                    "pending": len([t for t in self.tasks.values() if t.status == "pending"]),
                    "assigned": len([t for t in self.tasks.values() if t.status in ["assigned", "in_progress"]]),
                    "completed": len([t for t in self.tasks.values() if t.status == "completed"])
                },
                "resources": {
                    "total": len(self.resources),
                    "shared": len([r for r in self.resources.values() if r.shared]),
                    "sharing_networks": len(self.resource_sharing_networks)
                },
                "conflicts": {
                    "total": len(self.conflicts),
                    "resolved": len([c for c in self.conflicts.values() if c.resolved]),
                    "active": len([c for c in self.conflicts.values() if not c.resolved])
                },
                "decisions": {
                    "total": len(self.decisions),
                    "implemented": len([d for d in self.decisions.values() if d.status == "implemented"]),
                    "voting": len([d for d in self.decisions.values() if d.status == "active"])
                },
                "collaboration_metrics": self.get_collaboration_metrics()
            }


# 工具函数
def create_simple_task(task_type: TaskType, title: str, 
                      priority: TaskPriority = TaskPriority.MEDIUM,
                      duration: int = 8) -> Task:
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
        created_by="system"
    )

def create_basic_resource(resource_type: ResourceType, name: str, 
                         quantity: int = 1) -> Resource:
    """创建基础资源"""
    return Resource(
        id="",
        resource_type=resource_type,
        name=name,
        quantity=quantity,
        quality=1.0,
        shared=True,
        accessibility="medium"
    )