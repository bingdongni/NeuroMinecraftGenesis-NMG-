"""
规则库管理模块
负责管理推理规则，支持规则匹配、执行和优化
"""

from typing import Dict, List, Set, Union, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
from datetime import datetime
import uuid
from copy import deepcopy


class RuleType(Enum):
    """规则类型枚举"""
    MODUS_PONENS = "modus_ponens"           # 肯定前件
    MODUS_TOLLENS = "modus_tollens"         # 否定后件
    HYPOTHETICAL_SYLLOGISM = "hyp_syllogism"  # 假言三段论
    DISJUNCTIVE_SYLLOGISM = "disj_syllogism"  # 析取三段论
    RESOLUTION = "resolution"               # 归结规则
    FUZZY_RULE = "fuzzy_rule"              # 模糊规则
    PROBABILISTIC_RULE = "probabilistic_rule"  # 概率规则
    TEMPORAL_RULE = "temporal_rule"        # 时态规则
    CUSTOM_RULE = "custom_rule"            # 自定义规则


class RuleStatus(Enum):
    """规则状态枚举"""
    ACTIVE = "active"          # 活跃状态
    INACTIVE = "inactive"      # 非活跃状态
    DEPRECATED = "deprecated"  # 已废弃
    TESTING = "testing"        # 测试中


class MatchType(Enum):
    """匹配类型枚举"""
    EXACT = "exact"           # 精确匹配
    SUBSUME = "subsume"       # 包含匹配
    UNIFY = "unify"          # 合一匹配
    FUZZY_MATCH = "fuzzy_match"  # 模糊匹配


@dataclass
class RuleCondition:
    """规则条件"""
    pattern: str
    variable_bindings: Dict[str, str] = field(default_factory=dict)
    match_type: MatchType = MatchType.EXACT
    weight: float = 1.0
    certainty: float = 1.0
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError("权重值必须在0-1之间")
        if not 0 <= self.certainty <= 1:
            raise ValueError("确定性值必须在0-1之间")


@dataclass
class Rule:
    """规则定义"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    status: RuleStatus
    conditions: List[RuleCondition]
    conclusion: str
    certainty: float
    priority: int
    weight: float
    source: str
    timestamp: datetime
    usage_count: int = 0
    success_count: int = 0
    variables: Set[str] = field(default_factory=set)
    meta_data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not 0 <= self.certainty <= 1:
            raise ValueError("规则确定性值必须在0-1之间")
        if not 0 <= self.weight <= 1:
            raise ValueError("规则权重值必须在0-1之间")
    
    def get_success_rate(self) -> float:
        """获取规则成功率"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def add_usage(self, success: bool = True):
        """记录规则使用情况"""
        self.usage_count += 1
        if success:
            self.success_count += 1


@dataclass
class RuleMatch:
    """规则匹配结果"""
    rule_id: str
    match_quality: float
    variable_bindings: Dict[str, str]
    confidence_score: float
    matched_conditions: List[int]
    execution_cost: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.match_quality <= 1:
            raise ValueError("匹配质量值必须在0-1之间")
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("置信度分数必须在0-1之间")


class RuleMatcher:
    """规则匹配器"""
    
    def __init__(self):
        """初始化规则匹配器"""
        self.variable_pattern = re.compile(r'_[A-Za-z_][A-Za-z0-9_]*')
        self.constant_pattern = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
    
    def match_condition(self, condition: RuleCondition, facts: List[str]) -> Tuple[bool, Dict[str, str]]:
        """匹配单个条件"""
        bindings = {}
        pattern = condition.pattern
        
        # 处理变量
        variables = self.variable_pattern.findall(pattern)
        
        if condition.match_type == MatchType.EXACT:
            # 精确匹配
            for fact in facts:
                if pattern == fact:
                    return True, {}
                # 变量绑定匹配
                if self._variable_match(pattern, fact, bindings):
                    return True, bindings
        
        elif condition.match_type == MatchType.SUBSUME:
            # 包含匹配：模式可以包含事实
            for fact in facts:
                if self._subsume_match(pattern, fact, bindings):
                    return True, bindings
        
        elif condition.match_type == MatchType.UNIFY:
            # 合一匹配
            for fact in facts:
                if self._unify_match(pattern, fact, bindings):
                    return True, bindings
        
        elif condition.match_type == MatchType.FUZZY_MATCH:
            # 模糊匹配（简化实现）
            for fact in facts:
                if self._fuzzy_match(pattern, fact, bindings):
                    return True, bindings
        
        return False, {}
    
    def _variable_match(self, pattern: str, fact: str, bindings: Dict[str, str]) -> bool:
        """变量匹配"""
        pattern_tokens = pattern.split()
        fact_tokens = fact.split()
        
        if len(pattern_tokens) != len(fact_tokens):
            return False
        
        for p_token, f_token in zip(pattern_tokens, fact_tokens):
            if p_token.startswith('_'):  # 变量
                var_name = p_token[1:]
                if var_name in bindings:
                    if bindings[var_name] != f_token:
                        return False
                else:
                    bindings[var_name] = f_token
            else:  # 常量
                if p_token != f_token:
                    return False
        
        return True
    
    def _subsume_match(self, pattern: str, fact: str, bindings: Dict[str, str]) -> bool:
        """包含匹配"""
        # 简化的包含匹配实现
        # 实际实现需要更复杂的模式匹配算法
        return pattern in fact or fact in pattern
    
    def _unify_match(self, pattern: str, fact: str, bindings: Dict[str, str]) -> bool:
        """合一匹配"""
        # 简化的合一匹配实现
        # 实际实现需要处理变量替换和约束
        return self._variable_match(pattern, fact, bindings)
    
    def _fuzzy_match(self, pattern: str, fact: str, bindings: Dict[str, str]) -> float:
        """模糊匹配（返回匹配度）"""
        # 计算字符串相似度
        if len(pattern) == 0 and len(fact) == 0:
            return 1.0
        
        if len(pattern) == 0 or len(fact) == 0:
            return 0.0
        
        # 简单的编辑距离相似度
        max_len = max(len(pattern), len(fact))
        distance = abs(len(pattern) - len(fact))
        
        for i in range(min(len(pattern), len(fact))):
            if pattern[i] != fact[i]:
                distance += 1
        
        similarity = 1.0 - (distance / max_len)
        return similarity > 0.7  # 阈值判断


class RuleOptimizer:
    """规则优化器"""
    
    def __init__(self):
        """初始化规则优化器"""
        self.optimization_cache: Dict[str, Any] = {}
    
    def optimize_rule_set(self, rules: List[Rule]) -> List[Rule]:
        """优化规则集"""
        optimized_rules = deepcopy(rules)
        
        # 移除重复规则
        optimized_rules = self._remove_duplicate_rules(optimized_rules)
        
        # 按优先级和效率排序
        optimized_rules = self._sort_rules(optimized_rules)
        
        # 合并相似规则
        optimized_rules = self._merge_similar_rules(optimized_rules)
        
        return optimized_rules
    
    def _remove_duplicate_rules(self, rules: List[Rule]) -> List[Rule]:
        """移除重复规则"""
        seen_conclusions = set()
        unique_rules = []
        
        for rule in rules:
            conclusion_key = (rule.rule_type.value, rule.conclusion, frozenset(rule.conditions))
            if conclusion_key not in seen_conclusions:
                seen_conclusions.add(conclusion_key)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _sort_rules(self, rules: List[Rule]) -> List[Rule]:
        """按优先级和效率排序规则"""
        return sorted(rules, key=lambda r: (-r.priority, -r.get_success_rate(), r.weight))
    
    def _merge_similar_rules(self, rules: List[Rule]) -> List[Rule]:
        """合并相似规则"""
        merged_rules = []
        rule_groups = defaultdict(list)
        
        # 按结论分组
        for rule in rules:
            rule_groups[rule.conclusion].append(rule)
        
        # 合并每组中的规则
        for conclusion, rule_group in rule_groups.items():
            if len(rule_group) == 1:
                merged_rules.extend(rule_group)
            else:
                # 合并相似规则（简化实现）
                merged_rule = self._merge_rules(rule_group)
                merged_rules.append(merged_rule)
        
        return merged_rules
    
    def _merge_rules(self, rules: List[Rule]) -> Rule:
        """合并多个规则"""
        if not rules:
            return None
        
        base_rule = rules[0]
        merged_conditions = []
        max_certainty = 0.0
        total_priority = 0
        
        for rule in rules:
            merged_conditions.extend(rule.conditions)
            max_certainty = max(max_certainty, rule.certainty)
            total_priority += rule.priority
        
        # 创建合并后的规则
        merged_rule = Rule(
            id=str(uuid.uuid4()),
            name=f"Merged_{base_rule.name}",
            description=f"合并规则: {len(rules)}个相似规则",
            rule_type=base_rule.rule_type,
            status=base_rule.status,
            conditions=merged_conditions,
            conclusion=base_rule.conclusion,
            certainty=max_certainty,
            priority=total_priority // len(rules),
            weight=base_rule.weight,
            source="optimizer",
            timestamp=datetime.now()
        )
        
        return merged_rule


class RuleBase:
    """规则库管理系统"""
    
    def __init__(self, name: str = "default"):
        """初始化规则库"""
        self.name = name
        self.rules: Dict[str, Rule] = {}
        self.rule_index: Dict[str, Set[str]] = {}  # 标签->规则ID索引
        self.type_index: Dict[RuleType, Set[str]] = {}  # 类型->规则ID索引
        self.rule_matcher = RuleMatcher()
        self.rule_optimizer = RuleOptimizer()
        self.inference_count = 0
        self.cache: Dict[str, Any] = {}  # 推理缓存
        
        # 统计信息
        self.statistics = {
            "total_rules": 0,
            "active_rules": 0,
            "most_used_rule": None,
            "avg_success_rate": 0.0,
            "last_updated": datetime.now()
        }
    
    def add_rule(self, name: str, rule_type: RuleType, conditions: List[str],
                 conclusion: str, certainty: float = 1.0, priority: int = 0,
                 weight: float = 1.0, source: str = "user",
                 description: str = "", tags: Set[str] = None) -> str:
        """添加规则"""
        if tags is None:
            tags = set()
        
        rule_id = str(uuid.uuid4())
        
        # 构建规则条件
        rule_conditions = []
        for condition_str in conditions:
            rule_conditions.append(RuleCondition(
                pattern=condition_str,
                match_type=MatchType.EXACT
            ))
        
        # 提取变量
        variables = set()
        for condition in rule_conditions:
            variables.update(self.rule_matcher.variable_pattern.findall(condition.pattern))
        
        rule = Rule(
            id=rule_id,
            name=name,
            description=description,
            rule_type=rule_type,
            status=RuleStatus.ACTIVE,
            conditions=rule_conditions,
            conclusion=conclusion,
            certainty=certainty,
            priority=priority,
            weight=weight,
            source=source,
            timestamp=datetime.now(),
            variables=variables,
            tags=tags
        )
        
        self._add_rule(rule)
        return rule_id
    
    def _add_rule(self, rule: Rule):
        """内部方法：添加规则并更新索引"""
        self.rules[rule.id] = rule
        
        # 更新标签索引
        for tag in rule.tags:
            if tag not in self.rule_index:
                self.rule_index[tag] = set()
            self.rule_index[tag].add(rule.id)
        
        # 更新类型索引
        if rule.rule_type not in self.type_index:
            self.type_index[rule.rule_type] = set()
        self.type_index[rule.rule_type].add(rule.id)
        
        self._update_statistics()
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        # 从索引中移除
        for tag in rule.tags:
            if tag in self.rule_index:
                self.rule_index[tag].discard(rule_id)
                if not self.rule_index[tag]:
                    del self.rule_index[tag]
        
        if rule.rule_type in self.type_index:
            self.type_index[rule.rule_type].discard(rule_id)
            if not self.type_index[rule.rule_type]:
                del self.type_index[rule.rule_type]
        
        # 从规则库中移除
        del self.rules[rule_id]
        
        self._update_statistics()
        return True
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        """根据类型获取规则"""
        rule_ids = self.type_index.get(rule_type, set())
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """根据标签获取规则"""
        rule_ids = self.rule_index.get(tag, set())
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    def get_all_rules(self) -> List[Rule]:
        """获取所有规则"""
        return list(self.rules.values())
    
    def get_active_rules(self) -> List[Rule]:
        """获取活跃规则"""
        return [rule for rule in self.rules.values() if rule.status == RuleStatus.ACTIVE]
    
    def activate_rule(self, rule_id: str) -> bool:
        """激活规则"""
        if rule_id in self.rules:
            self.rules[rule_id].status = RuleStatus.ACTIVE
            self._update_statistics()
            return True
        return False
    
    def deactivate_rule(self, rule_id: str) -> bool:
        """停用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].status = RuleStatus.INACTIVE
            self._update_statistics()
            return True
        return False
    
    def match_rules(self, facts: List[str], rule_types: Optional[List[RuleType]] = None) -> List[RuleMatch]:
        """匹配规则"""
        if rule_types is None:
            rule_types = list(RuleType)
        
        matches = []
        
        for rule_type in rule_types:
            for rule in self.get_rules_by_type(rule_type):
                if rule.status != RuleStatus.ACTIVE:
                    continue
                
                match_result = self._match_rule(rule, facts)
                if match_result:
                    matches.append(match_result)
        
        # 按匹配质量排序
        matches.sort(key=lambda m: m.match_quality, reverse=True)
        return matches
    
    def _match_rule(self, rule: Rule, facts: List[str]) -> Optional[RuleMatch]:
        """匹配单个规则"""
        matched_conditions = []
        total_confidence = 0.0
        variable_bindings = {}
        
        for i, condition in enumerate(rule.conditions):
            condition_match, bindings = self.rule_matcher.match_condition(condition, facts)
            if condition_match:
                matched_conditions.append(i)
                total_confidence += condition.certainty
                
                # 合并变量绑定
                for var, value in bindings.items():
                    if var not in variable_bindings:
                        variable_bindings[var] = value
                    elif variable_bindings[var] != value:
                        # 冲突的变量绑定
                        return None
        
        if not matched_conditions:
            return None
        
        # 计算匹配质量
        match_quality = len(matched_conditions) / len(rule.conditions)
        confidence_score = (total_confidence / len(matched_conditions)) if matched_conditions else 0.0
        
        # 计算执行成本（简化实现）
        execution_cost = len(rule.conditions) * rule.weight
        
        return RuleMatch(
            rule_id=rule.id,
            match_quality=match_quality,
            variable_bindings=variable_bindings,
            confidence_score=confidence_score,
            matched_conditions=matched_conditions,
            execution_cost=execution_cost
        )
    
    def execute_rule(self, rule_match: RuleMatch, facts: List[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """执行匹配的规则"""
        rule = self.rules.get(rule_match.rule_id)
        if not rule:
            return False, [], {}
        
        try:
            # 应用变量绑定到结论
            conclusion = rule.conclusion
            for var, value in rule_match.variable_bindings.items():
                conclusion = conclusion.replace(f"_{var}", value)
            
            # 生成新事实
            new_facts = [conclusion]
            
            # 记录规则使用
            rule.add_usage(True)
            self.inference_count += 1
            
            return True, new_facts, {
                "rule_id": rule.id,
                "original_conclusion": rule.conclusion,
                "bound_conclusion": conclusion,
                "variable_bindings": rule_match.variable_bindings,
                "certainty": rule.certainty,
                "confidence": rule_match.confidence_score
            }
            
        except Exception as e:
            rule.add_usage(False)
            return False, [], {"error": str(e)}
    
    def forward_chain(self, facts: List[str], max_iterations: int = 100) -> Tuple[List[str], List[Dict[str, Any]]]:
        """前向链式推理"""
        current_facts = set(facts)
        inference_history = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            new_facts_found = False
            
            # 匹配规则
            rule_matches = self.match_rules(list(current_facts))
            
            for rule_match in rule_matches:
                success, new_facts, execution_info = self.execute_rule(rule_match, list(current_facts))
                
                if success and new_facts:
                    inference_history.append({
                        "iteration": iteration,
                        "rule_match": rule_match,
                        "execution_info": execution_info,
                        "new_facts": new_facts
                    })
                    
                    # 添加新事实
                    for new_fact in new_facts:
                        if new_fact not in current_facts:
                            current_facts.add(new_fact)
                            new_facts_found = True
            
            # 如果没有新事实产生，停止推理
            if not new_facts_found:
                break
        
        return list(current_facts), inference_history
    
    def backward_chain(self, goal: str, facts: List[str], max_depth: int = 10) -> Tuple[bool, List[str]]:
        """后向链式推理"""
        visited_goals = set()
        
        def prove_goal(goal: str, current_facts: List[str], depth: int) -> Tuple[bool, List[str]]:
            if depth > max_depth:
                return False, []
            
            if goal in current_facts:
                return True, [goal]
            
            if goal in visited_goals:
                return False, []
            
            visited_goals.add(goal)
            
            # 查找可以证明目标的规则
            for rule in self.get_active_rules():
                if rule.conclusion.lower() in goal.lower() or goal.lower() in rule.conclusion.lower():
                    # 尝试证明规则的所有条件
                    all_conditions_proved = True
                    proof_chain = []
                    
                    for condition in rule.conditions:
                        condition_proved, condition_proof = prove_goal(
                            condition.pattern, current_facts, depth + 1
                        )
                        if condition_proved:
                            proof_chain.extend(condition_proof)
                        else:
                            all_conditions_proved = False
                            break
                    
                    if all_conditions_proved:
                        proof_chain.append(goal)
                        return True, proof_chain
            
            return False, []
        
        return prove_goal(goal, facts, 0)
    
    def optimize_rules(self) -> int:
        """优化规则库"""
        original_count = len(self.rules)
        
        # 获取优化后的规则集
        active_rules = self.get_active_rules()
        optimized_rules = self.rule_optimizer.optimize_rule_set(active_rules)
        
        # 移除旧规则，添加优化后的规则
        old_rule_ids = set(rule.id for rule in active_rules)
        self.rules.clear()
        self.rule_index.clear()
        self.type_index.clear()
        
        for rule in optimized_rules:
            self._add_rule(rule)
        
        return original_count - len(self.rules)
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        active_rules = self.get_active_rules()
        total_usage = sum(rule.usage_count for rule in self.rules.values())
        
        most_used_rule = None
        max_usage = 0
        for rule in self.rules.values():
            if rule.usage_count > max_usage:
                max_usage = rule.usage_count
                most_used_rule = rule
        
        avg_success_rate = 0.0
        if self.rules:
            avg_success_rate = sum(rule.get_success_rate() for rule in self.rules.values()) / len(self.rules)
        
        self.statistics = {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "inactive_rules": len(self.rules) - len(active_rules),
            "total_usage": total_usage,
            "most_used_rule": most_used_rule.name if most_used_rule else None,
            "avg_success_rate": avg_success_rate,
            "inference_count": self.inference_count,
            "last_updated": datetime.now()
        }
        
        return self.statistics.copy()
    
    def export_rules(self) -> Dict[str, Any]:
        """导出规则库"""
        return {
            "name": self.name,
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "type": rule.rule_type.value,
                    "status": rule.status.value,
                    "conditions": [cond.pattern for cond in rule.conditions],
                    "conclusion": rule.conclusion,
                    "certainty": rule.certainty,
                    "priority": rule.priority,
                    "weight": rule.weight,
                    "source": rule.source,
                    "timestamp": rule.timestamp.isoformat(),
                    "usage_count": rule.usage_count,
                    "success_count": rule.success_count,
                    "variables": list(rule.variables),
                    "meta_data": rule.meta_data,
                    "tags": list(rule.tags)
                }
                for rule in self.rules.values()
            ],
            "statistics": self.statistics
        }
    
    def import_rules(self, data: Dict[str, Any]) -> bool:
        """导入规则库"""
        try:
            self.name = data.get("name", self.name)
            
            # 清空现有规则
            self.rules.clear()
            self.rule_index.clear()
            self.type_index.clear()
            
            # 导入规则
            for rule_data in data.get("rules", []):
                conditions = rule_data.get("conditions", [])
                rule_conditions = [
                    RuleCondition(pattern=cond) for cond in conditions
                ]
                
                rule = Rule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data.get("description", ""),
                    rule_type=RuleType(rule_data.get("type", "custom_rule")),
                    status=RuleStatus(rule_data.get("status", "active")),
                    conditions=rule_conditions,
                    conclusion=rule_data.get("conclusion", ""),
                    certainty=rule_data.get("certainty", 1.0),
                    priority=rule_data.get("priority", 0),
                    weight=rule_data.get("weight", 1.0),
                    source=rule_data.get("source", ""),
                    timestamp=datetime.fromisoformat(rule_data.get("timestamp", datetime.now().isoformat())),
                    usage_count=rule_data.get("usage_count", 0),
                    success_count=rule_data.get("success_count", 0),
                    variables=set(rule_data.get("variables", [])),
                    meta_data=rule_data.get("meta_data", {}),
                    tags=set(rule_data.get("tags", []))
                )
                
                self._add_rule(rule)
            
            return True
        except Exception as e:
            print(f"导入规则库失败: {e}")
            return False
    
    def clear(self):
        """清空规则库"""
        self.rules.clear()
        self.rule_index.clear()
        self.type_index.clear()
        self.cache.clear()
        self._update_statistics()
    
    def _update_statistics(self):
        """更新统计信息"""
        active_rules = self.get_active_rules()
        avg_success_rate = 0.0
        
        if self.rules:
            avg_success_rate = sum(rule.get_success_rate() for rule in self.rules.values()) / len(self.rules)
        
        self.statistics = {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "avg_success_rate": avg_success_rate,
            "last_updated": datetime.now()
        }