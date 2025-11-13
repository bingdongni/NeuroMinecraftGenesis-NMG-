"""
推理引擎核心模块
实现多种推理算法，包括前向链式推理、后向链式推理、模糊推理和不确定性推理
"""

from typing import Dict, List, Set, Union, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import time
import uuid
from datetime import datetime
import math
import heapq
import networkx as nx


class InferenceType(Enum):
    """推理类型枚举"""
    FORWARD_CHAINING = "forward_chaining"     # 前向链式推理
    BACKWARD_CHAINING = "backward_chain"      # 后向链式推理
    BIDIRECTIONAL = "bidirectional"           # 双向推理
    FUZZY_REASONING = "fuzzy_reasoning"       # 模糊推理
    PROBABILISTIC = "probabilistic"           # 概率推理
    TEMPORAL = "temporal"                     # 时态推理
    ABDUCTIVE = "abductive"                   # 溯因推理
    ANALOGICAL = "analogical"                 # 类比推理


class ConflictResolutionStrategy(Enum):
    """冲突解决策略枚举"""
    PRIORITY_BASED = "priority_based"         # 基于优先级
    SPECIFICITY_BASED = "specificity_based"   # 基于特异性
    CERTAINTY_BASED = "certainty_based"       # 基于确定性
    RECENCY_BASED = "recency_based"           # 基于时间
    FIRE_ALL = "fire_all"                     # 全部执行


class ReasoningState(Enum):
    """推理状态枚举"""
    IDLE = "idle"                             # 空闲
    RUNNING = "running"                       # 运行中
    COMPLETED = "completed"                   # 完成
    FAILED = "failed"                         # 失败
    TIMEOUT = "timeout"                       # 超时


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    step_type: InferenceType
    premise: str
    conclusion: str
    rule_id: Optional[str]
    certainty: float
    confidence: float
    timestamp: datetime
    variables: Dict[str, str] = field(default_factory=dict)
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.certainty <= 1:
            raise ValueError("确定性值必须在0-1之间")
        if not 0 <= self.confidence <= 1:
            raise ValueError("置信度值必须在0-1之间")


@dataclass
class ReasoningPath:
    """推理路径"""
    path_id: str
    goal: str
    steps: List[ReasoningStep]
    total_certainty: float
    total_confidence: float
    length: int
    cost: float
    
    def __post_init__(self):
        if not self.steps:
            raise ValueError("推理路径不能为空")
    
    def get_step_at(self, index: int) -> Optional[ReasoningStep]:
        """获取指定步骤"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def add_step(self, step: ReasoningStep):
        """添加推理步骤"""
        self.steps.append(step)
        self.length += 1
        # 重新计算总确定性和置信度
        if self.steps:
            self.total_certainty = sum(step.certainty for step in self.steps) / len(self.steps)
            self.total_confidence = sum(step.confidence for step in self.steps) / len(self.steps)
        self.cost += 1.0


@dataclass
class ReasoningResult:
    """推理结果"""
    success: bool
    result_type: InferenceType
    conclusion: Optional[str] = None
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    execution_time: float = 0.0
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_path(self) -> Optional[ReasoningPath]:
        """获取最佳推理路径"""
        if not self.reasoning_paths:
            return None
        
        # 按总确定性排序
        return max(self.reasoning_paths, key=lambda p: p.total_certainty * p.total_confidence)
    
    def get_all_conclusions(self) -> List[str]:
        """获取所有结论"""
        conclusions = []
        for path in self.reasoning_paths:
            if path.goal not in conclusions:
                conclusions.append(path.goal)
        return conclusions


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, strategy: ConflictResolutionStrategy):
        """初始化冲突解决器"""
        self.strategy = strategy
    
    def resolve_conflicts(self, conflicting_paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """解决冲突"""
        if self.strategy == ConflictResolutionStrategy.PRIORITY_BASED:
            return self._priority_based_resolution(conflicting_paths)
        elif self.strategy == ConflictResolutionStrategy.SPECIFICITY_BASED:
            return self._specificity_based_resolution(conflicting_paths)
        elif self.strategy == ConflictResolutionStrategy.CERTAINTY_BASED:
            return self._certainty_based_resolution(conflicting_paths)
        elif self.strategy == ConflictResolutionStrategy.RECENCY_BASED:
            return self._recency_based_resolution(conflicting_paths)
        elif self.strategy == ConflictResolutionStrategy.FIRE_ALL:
            return conflicting_paths
        else:
            return conflicting_paths[:1]  # 默认返回第一个
    
    def _priority_based_resolution(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """基于优先级解决冲突"""
        # 按路径中规则的平均优先级排序
        return sorted(paths, key=lambda p: self._calculate_path_priority(p), reverse=True)
    
    def _specificity_based_resolution(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """基于特异性解决冲突"""
        # 特异性用路径长度和条件数量衡量
        return sorted(paths, key=lambda p: (p.length, len(p.steps)), reverse=True)
    
    def _certainty_based_resolution(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """基于确定性解决冲突"""
        return sorted(paths, key=lambda p: p.total_certainty, reverse=True)
    
    def _recency_based_resolution(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """基于时间解决冲突"""
        return sorted(paths, key=lambda p: p.steps[-1].timestamp if p.steps else datetime.min, reverse=True)
    
    def _calculate_path_priority(self, path: ReasoningPath) -> float:
        """计算路径优先级"""
        if not path.steps:
            return 0.0
        
        total_priority = 0.0
        for step in path.steps:
            # 这里需要从规则库获取规则优先级
            # 简化实现，使用确定性作为优先级
            total_priority += step.certainty
        
        return total_priority / len(path.steps)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self._start_times = {}
        self._end_times = {}
        self.metrics = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float('inf'),
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_usage": 0.0,
            "last_updated": datetime.now()
        }
    
    def start_timing(self) -> str:
        """开始计时"""
        timing_id = str(uuid.uuid4())
        self._start_times[timing_id] = time.time()
        return timing_id
    
    def end_timing(self, timing_id: str, success: bool = True) -> float:
        """结束计时"""
        if timing_id not in self._start_times:
            return 0.0
        
        elapsed_time = time.time() - self._start_times[timing_id]
        del self._start_times[timing_id]
        
        self._update_metrics(elapsed_time, success)
        return elapsed_time
    
    def _update_metrics(self, execution_time: float, success: bool):
        """更新性能指标"""
        self.metrics["total_inferences"] += 1
        
        if success:
            self.metrics["successful_inferences"] += 1
        else:
            self.metrics["failed_inferences"] += 1
        
        # 更新执行时间统计
        self.metrics["average_execution_time"] += execution_time
        self.metrics["max_execution_time"] = max(self.metrics["max_execution_time"], execution_time)
        self.metrics["min_execution_time"] = min(self.metrics["min_execution_time"], execution_time)
        self.metrics["last_updated"] = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.metrics["total_inferences"] > 0:
            self.metrics["average_execution_time"] /= self.metrics["total_inferences"]
        
        return self.metrics.copy()


class InferenceEngine:
    """推理引擎核心类"""
    
    def __init__(self, max_iterations: int = 1000, timeout: float = 30.0):
        """初始化推理引擎"""
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.state = ReasoningState.IDLE
        
        # 推理历史和缓存
        self.inference_history: List[ReasoningStep] = []
        self.reasoning_cache: Dict[str, Any] = {}
        self.dependency_graph = nx.DiGraph()
        
        # 冲突解决器
        self.conflict_resolver = ConflictResolver(ConflictResolutionStrategy.CERTAINTY_BASED)
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor()
        
        # 推理配置
        self.config = {
            "enable_caching": True,
            "enable_optimization": True,
            "max_reasoning_depth": 10,
            "min_confidence_threshold": 0.1,
            "parallel_inference": False,
            "fuzzy_threshold": 0.5
        }
    
    def forward_chain(self, facts: List[str], knowledge_base, rule_base, 
                     target_goals: Optional[List[str]] = None) -> ReasoningResult:
        """前向链式推理"""
        timing_id = self.performance_monitor.start_timing()
        
        try:
            self.state = ReasoningState.RUNNING
            start_time = time.time()
            
            # 初始化推理结果
            result = ReasoningResult(
                success=False,
                result_type=InferenceType.FORWARD_CHAINING
            )
            
            # 工作记忆：当前所有已知事实
            working_memory = set(facts)
            new_facts = set()
            reasoning_paths = []
            
            # 迭代推理过程
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                
                if time.time() - start_time > self.timeout:
                    self.state = ReasoningState.TIMEOUT
                    break
                
                # 获取可应用的规则
                applicable_rules = rule_base.get_active_rules()
                
                # 检查每个规则
                new_facts_this_iteration = []
                
                for rule in applicable_rules:
                    if self._check_rule_applicability(rule, working_memory):
                        # 应用规则
                        rule_execution_result = self._execute_rule(rule, working_memory)
                        
                        if rule_execution_result.success:
                            for new_fact in rule_execution_result.new_facts:
                                if new_fact not in working_memory:
                                    new_facts_this_iteration.append(new_fact)
                                    
                                    # 创建推理路径
                                    path = ReasoningPath(
                                        path_id=str(uuid.uuid4()),
                                        goal=new_fact,
                                        steps=[],
                                        total_certainty=rule.certainty,
                                        total_confidence=rule_execution_result.confidence,
                                        length=0,
                                        cost=0.0
                                    )
                                    
                                    # 添加推理步骤
                                    step = ReasoningStep(
                                        step_id=str(uuid.uuid4()),
                                        step_type=InferenceType.FORWARD_CHAINING,
                                        premise=" & ".join(rule.antecedent),
                                        conclusion=new_fact,
                                        rule_id=rule.id,
                                        certainty=rule.certainty,
                                        confidence=rule_execution_result.confidence,
                                        timestamp=datetime.now(),
                                        variables=rule_execution_result.variables
                                    )
                                    
                                    path.add_step(step)
                                    reasoning_paths.append(path)
                                    result.reasoning_steps.append(step)
                
                # 更新工作记忆
                if new_facts_this_iteration:
                    new_facts.update(new_facts_this_iteration)
                    working_memory.update(new_facts_this_iteration)
                else:
                    # 没有新事实产生，推理完成
                    break
            
            # 检查是否达到目标
            if target_goals:
                reached_goals = []
                for goal in target_goals:
                    for fact in working_memory:
                        if goal.lower() in fact.lower() or fact.lower() in goal.lower():
                            reached_goals.append(goal)
                            break
                
                result.success = len(reached_goals) > 0
                result.conclusion = " & ".join(reached_goals) if reached_goals else None
            else:
                # 没有特定目标，检查是否产生了新知识
                result.success = len(new_facts) > 0
                result.conclusion = "推理完成" if result.success else "未能产生新知识"
            
            result.reasoning_paths = reasoning_paths
            result.execution_time = self.performance_monitor.end_timing(timing_id, result.success)
            result.meta_data = {
                "iterations": iteration + 1 if 'iteration' in locals() else 0,
                "new_facts_count": len(new_facts),
                "total_facts": len(working_memory),
                "reasoning_paths_count": len(reasoning_paths)
            }
            
            self.state = ReasoningState.COMPLETED
            return result
            
        except Exception as e:
            self.state = ReasoningState.FAILED
            result.success = False
            result.execution_time = self.performance_monitor.end_timing(timing_id, False)
            result.meta_data = {"error": str(e)}
            return result
    
    def backward_chain(self, goal: str, facts: List[str], knowledge_base, rule_base,
                      max_depth: int = None) -> ReasoningResult:
        """后向链式推理"""
        timing_id = self.performance_monitor.start_timing()
        
        if max_depth is None:
            max_depth = self.config["max_reasoning_depth"]
        
        try:
            self.state = ReasoningState.RUNNING
            start_time = time.time()
            
            result = ReasoningResult(
                success=False,
                result_type=InferenceType.BACKWARD_CHAINING
            )
            
            visited_goals = set()
            reasoning_paths = []
            
            def prove_goal(goal: str, current_facts: Set[str], depth: int, 
                          path: ReasoningPath) -> Tuple[bool, ReasoningPath]:
                """递归证明目标"""
                if time.time() - start_time > self.timeout:
                    return False, path
                
                if depth > max_depth:
                    return False, path
                
                if goal in current_facts:
                    return True, path
                
                goal_key = f"{goal}_{depth}"
                if goal_key in visited_goals:
                    return False, path
                
                visited_goals.add(goal_key)
                
                # 查找可以证明目标的规则
                applicable_rules = []
                for rule in rule_base.get_active_rules():
                    if (goal.lower() in rule.conclusion.lower() or 
                        rule.conclusion.lower() in goal.lower()):
                        applicable_rules.append(rule)
                
                for rule in applicable_rules:
                    # 尝试证明规则的所有条件
                    all_conditions_proved = True
                    proof_path = ReasoningPath(
                        path_id=str(uuid.uuid4()),
                        goal=goal,
                        steps=[],
                        total_certainty=rule.certainty,
                        total_confidence=1.0,
                        length=0,
                        cost=0.0
                    )
                    
                    for condition in rule.antecedent:
                        condition_proved, _ = prove_goal(
                            condition, current_facts, depth + 1, proof_path
                        )
                        if not condition_proved:
                            all_conditions_proved = False
                            break
                    
                    if all_conditions_proved:
                        # 添加推理步骤
                        step = ReasoningStep(
                            step_id=str(uuid.uuid4()),
                            step_type=InferenceType.BACKWARD_CHAINING,
                            premise=" & ".join(rule.antecedent),
                            conclusion=goal,
                            rule_id=rule.id,
                            certainty=rule.certainty,
                            confidence=1.0,
                            timestamp=datetime.now()
                        )
                        
                        proof_path.add_step(step)
                        result.reasoning_steps.append(step)
                        reasoning_paths.append(proof_path)
                        return True, proof_path
                
                return False, path
            
            # 开始证明
            working_facts = set(facts)
            success, final_path = prove_goal(goal, working_facts, 0, ReasoningPath(
                path_id=str(uuid.uuid4()),
                goal=goal,
                steps=[],
                total_certainty=0.0,
                total_confidence=0.0,
                length=0,
                cost=0.0
            ))
            
            result.success = success
            result.conclusion = goal if success else None
            result.reasoning_paths = reasoning_paths
            result.execution_time = self.performance_monitor.end_timing(timing_id, success)
            result.meta_data = {
                "max_depth": max_depth,
                "visited_goals": len(visited_goals),
                "reasoning_paths_count": len(reasoning_paths)
            }
            
            self.state = ReasoningState.COMPLETED
            return result
            
        except Exception as e:
            self.state = ReasoningState.FAILED
            result.success = False
            result.execution_time = self.performance_monitor.end_timing(timing_id, False)
            result.meta_data = {"error": str(e)}
            return result
    
    def fuzzy_reasoning(self, fuzzy_facts: List[Dict[str, Any]], 
                       fuzzy_rules: List[Dict[str, Any]]) -> ReasoningResult:
        """模糊逻辑推理"""
        timing_id = self.performance_monitor.start_timing()
        
        try:
            self.state = ReasoningState.RUNNING
            
            result = ReasoningResult(
                success=False,
                result_type=InferenceType.FUZZY_REASONING
            )
            
            # 模糊工作记忆：变量->隶属度映射
            fuzzy_memory = {}
            for fact in fuzzy_facts:
                if isinstance(fact, dict):
                    variable = fact.get("variable", "")
                    membership = fact.get("membership_degree", 0.0)
                    if variable:
                        fuzzy_memory[variable] = membership
            
            reasoning_paths = []
            new_fuzzy_facts = []
            
            # 应用模糊规则
            for fuzzy_rule in fuzzy_rules:
                if isinstance(fuzzy_rule, dict):
                    antecedent = fuzzy_rule.get("antecedent", [])
                    consequent = fuzzy_rule.get("consequent", "")
                    rule_strength = fuzzy_rule.get("strength", 1.0)
                    
                    # 计算 antecedent 的模糊强度
                    antecedent_strength = 1.0
                    for var in antecedent:
                        if var in fuzzy_memory:
                            antecedent_strength = min(antecedent_strength, fuzzy_memory[var])
                        else:
                            antecedent_strength = 0.0
                            break
                    
                    # 如果 antecedent 满足条件，产生新事实
                    if antecedent_strength > self.config["fuzzy_threshold"]:
                        new_membership = antecedent_strength * rule_strength
                        
                        if consequent not in fuzzy_memory or new_membership > fuzzy_memory[consequent]:
                            fuzzy_memory[consequent] = new_membership
                            new_fuzzy_facts.append({
                                "variable": consequent,
                                "membership_degree": new_membership
                            })
                            
                            # 创建推理路径
                            path = ReasoningPath(
                                path_id=str(uuid.uuid4()),
                                goal=consequent,
                                steps=[],
                                total_certainty=new_membership,
                                total_confidence=antecedent_strength,
                                length=1,
                                cost=1.0
                            )
                            
                            # 添加推理步骤
                            step = ReasoningStep(
                                step_id=str(uuid.uuid4()),
                                step_type=InferenceType.FUZZY_REASONING,
                                premise=f" & ".join(antecedent),
                                conclusion=consequent,
                                rule_id=fuzzy_rule.get("id", ""),
                                certainty=new_membership,
                                confidence=antecedent_strength,
                                timestamp=datetime.now()
                            )
                            
                            path.add_step(step)
                            reasoning_paths.append(path)
                            result.reasoning_steps.append(step)
            
            result.success = len(new_fuzzy_facts) > 0
            result.conclusion = f"产生了 {len(new_fuzzy_facts)} 个新的模糊事实" if result.success else "没有产生新事实"
            result.reasoning_paths = reasoning_paths
            result.execution_time = self.performance_monitor.end_timing(timing_id, result.success)
            result.meta_data = {
                "fuzzy_facts_count": len(fuzzy_facts),
                "new_fuzzy_facts_count": len(new_fuzzy_facts),
                "fuzzy_rules_count": len(fuzzy_rules)
            }
            
            self.state = ReasoningState.COMPLETED
            return result
            
        except Exception as e:
            self.state = ReasoningState.FAILED
            result.success = False
            result.execution_time = self.performance_monitor.end_timing(timing_id, False)
            result.meta_data = {"error": str(e)}
            return result
    
    def uncertain_reasoning(self, uncertain_facts: List[Dict[str, Any]], 
                           uncertain_rules: List[Dict[str, Any]]) -> ReasoningResult:
        """不确定性推理"""
        timing_id = self.performance_monitor.start_timing()
        
        try:
            self.state = ReasoningState.RUNNING
            
            result = ReasoningResult(
                success=False,
                result_type=InferenceType.PROBABILISTIC
            )
            
            # 不确定性工作记忆：命题->确定性值映射
            uncertain_memory = {}
            for fact in uncertain_facts:
                if isinstance(fact, dict):
                    proposition = fact.get("proposition", "")
                    certainty = fact.get("certainty", 0.0)
                    if proposition:
                        uncertain_memory[proposition] = certainty
            
            reasoning_paths = []
            new_uncertain_facts = []
            
            # 应用不确定性规则
            for uncertain_rule in uncertain_rules:
                if isinstance(uncertain_rule, dict):
                    antecedent = uncertain_rule.get("antecedent", [])
                    consequent = uncertain_rule.get("consequent", "")
                    rule_certainty = uncertain_rule.get("certainty", 1.0)
                    
                    # 计算 antecedent 的不确定性
                    antecedent_certainty = 1.0
                    for prop in antecedent:
                        if prop in uncertain_memory:
                            antecedent_certainty = min(antecedent_certainty, uncertain_memory[prop])
                        else:
                            antecedent_certainty = 0.0
                            break
                    
                    # 应用规则产生新结论
                    if antecedent_certainty > 0:
                        new_certainty = antecedent_certainty * rule_certainty
                        
                        if consequent not in uncertain_memory or new_certainty > uncertain_memory[consequent]:
                            uncertain_memory[consequent] = new_certainty
                            new_uncertain_facts.append({
                                "proposition": consequent,
                                "certainty": new_certainty
                            })
                            
                            # 创建推理路径
                            path = ReasoningPath(
                                path_id=str(uuid.uuid4()),
                                goal=consequent,
                                steps=[],
                                total_certainty=new_certainty,
                                total_confidence=antecedent_certainty,
                                length=1,
                                cost=1.0
                            )
                            
                            # 添加推理步骤
                            step = ReasoningStep(
                                step_id=str(uuid.uuid4()),
                                step_type=InferenceType.PROBABILISTIC,
                                premise=f" & ".join(antecedent),
                                conclusion=consequent,
                                rule_id=uncertain_rule.get("id", ""),
                                certainty=new_certainty,
                                confidence=antecedent_certainty,
                                timestamp=datetime.now()
                            )
                            
                            path.add_step(step)
                            reasoning_paths.append(path)
                            result.reasoning_steps.append(step)
            
            result.success = len(new_uncertain_facts) > 0
            result.conclusion = f"产生了 {len(new_uncertain_facts)} 个新的不确定性事实" if result.success else "没有产生新事实"
            result.reasoning_paths = reasoning_paths
            result.execution_time = self.performance_monitor.end_timing(timing_id, result.success)
            result.meta_data = {
                "uncertain_facts_count": len(uncertain_facts),
                "new_uncertain_facts_count": len(new_uncertain_facts),
                "uncertain_rules_count": len(uncertain_rules)
            }
            
            self.state = ReasoningState.COMPLETED
            return result
            
        except Exception as e:
            self.state = ReasoningState.FAILED
            result.success = False
            result.execution_time = self.performance_monitor.end_timing(timing_id, False)
            result.meta_data = {"error": str(e)}
            return result
    
    def bidirectional_reasoning(self, goal: str, facts: List[str], 
                               knowledge_base, rule_base) -> ReasoningResult:
        """双向推理"""
        timing_id = self.performance_monitor.start_timing()
        
        try:
            self.state = ReasoningState.RUNNING
            
            result = ReasoningResult(
                success=False,
                result_type=InferenceType.BIDIRECTIONAL
            )
            
            # 前向推理：从已知事实推导
            forward_result = self.forward_chain(facts, knowledge_base, rule_base, [goal])
            
            # 后向推理：从目标反向证明
            backward_result = self.backward_chain(goal, facts, knowledge_base, rule_base)
            
            # 合并结果
            combined_paths = forward_result.reasoning_paths + backward_result.reasoning_paths
            combined_steps = forward_result.reasoning_steps + backward_result.reasoning_steps
            
            # 检查是否有成功的推理路径
            success = forward_result.success or backward_result.success
            
            result.success = success
            result.conclusion = goal if success else None
            result.reasoning_paths = combined_paths
            result.reasoning_steps = combined_steps
            result.execution_time = self.performance_monitor.end_timing(timing_id, success)
            result.meta_data = {
                "forward_success": forward_result.success,
                "backward_success": backward_result.success,
                "forward_paths": len(forward_result.reasoning_paths),
                "backward_paths": len(backward_result.reasoning_paths)
            }
            
            self.state = ReasoningState.COMPLETED
            return result
            
        except Exception as e:
            self.state = ReasoningState.FAILED
            result.success = False
            result.execution_time = self.performance_monitor.end_timing(timing_id, False)
            result.meta_data = {"error": str(e)}
            return result
    
    def _check_rule_applicability(self, rule, facts: Set[str]) -> bool:
        """检查规则是否可应用"""
        # 检查规则的所有条件是否满足
        for condition in rule.conditions:
            condition_satisfied = False
            for fact in facts:
                if self._pattern_match(condition.pattern, fact):
                    condition_satisfied = True
                    break
            if not condition_satisfied:
                return False
        return True
    
    def _pattern_match(self, pattern: str, fact: str) -> bool:
        """模式匹配"""
        # 简单的模式匹配实现
        # 实际实现需要更复杂的变量绑定和合一算法
        return pattern == fact or pattern.lower() in fact.lower() or fact.lower() in pattern.lower()
    
    def _execute_rule(self, rule, facts: Set[str]) -> Any:
        """执行规则（简化实现）"""
        # 这里应该实现具体的规则执行逻辑
        # 简化返回
        return type('Result', (), {
            'success': True,
            'new_facts': [rule.conclusion],
            'confidence': rule.certainty,
            'variables': {}
        })()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_monitor.get_metrics()
    
    def get_inference_history(self, limit: int = 100) -> List[ReasoningStep]:
        """获取推理历史"""
        return self.inference_history[-limit:]
    
    def clear_cache(self):
        """清空缓存"""
        self.reasoning_cache.clear()
    
    def export_engine_state(self) -> Dict[str, Any]:
        """导出引擎状态"""
        return {
            "state": self.state.value,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "config": self.config,
            "cache_size": len(self.reasoning_cache),
            "history_size": len(self.inference_history),
            "performance_metrics": self.performance_monitor.get_metrics()
        }
    
    def reset(self):
        """重置推理引擎"""
        self.state = ReasoningState.IDLE
        self.inference_history.clear()
        self.reasoning_cache.clear()
        self.dependency_graph.clear()
        
        # 重置性能监控器
        self.performance_monitor = PerformanceMonitor()