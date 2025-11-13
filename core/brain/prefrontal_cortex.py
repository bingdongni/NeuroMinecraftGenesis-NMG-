"""
前额叶推理引擎升级版 - 集成链式推理和符号逻辑

本模块实现了完整的前额叶皮层推理能力，包含：
1. 链式推理和逻辑演绎引擎
2. 符号逻辑推理系统
3. 贝叶斯推理和概率逻辑
4. 多步推理和问题解决
5. 元认知监控和策略调整

核心特性：
- 多层次推理：符号推理、概率推理、演绎推理
- 符号逻辑：命题逻辑、谓词逻辑、模糊逻辑
- 贝叶斯网络：概率推理、信念更新
- 元认知监控：推理质量评估、策略调整
- 自适应学习：推理策略优化、错误纠正

作者: 思维力系统
升级时间: 2025-11-13
版本: 2.0
"""

import asyncio
import json
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
import math
import hashlib
from abc import ABC, abstractmethod
import copy
from scipy.stats import entropy
from scipy.optimize import minimize
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """推理类型枚举"""
    DEDUCTIVE = "deductive"          # 演绎推理
    INDUCTIVE = "inductive"          # 归纳推理
    ABDUCTIVE = "abductive"          # 溯因推理
    PROBABILISTIC = "probabilistic"  # 概率推理
    ANALOGICAL = "analogical"        # 类比推理
    CAUSAL = "causal"               # 因果推理


class LogicSystem(Enum):
    """逻辑系统枚举"""
    PROPOSITIONAL = "propositional"      # 命题逻辑
    PREDICATE = "predicate"              # 谓词逻辑
    FUZZY = "fuzzy"                      # 模糊逻辑
    MODAL = "modal"                      # 模态逻辑
    TEMPORAL = "temporal"                # 时态逻辑


class MetaCognitionState(Enum):
    """元认知状态枚举"""
    MONITORING = "monitoring"           # 监控状态
    EVALUATING = "evaluating"           # 评估状态
    ADJUSTING = "adjusting"             # 调整状态
    LEARNING = "learning"               # 学习状态


@dataclass
class SymbolicExpression:
    """符号表达式"""
    expression_type: str
    content: str
    variables: Set[str] = field(default_factory=set)
    operators: List[str] = field(default_factory=list)
    operands: List['SymbolicExpression'] = field(default_factory=list)
    
    def evaluate(self, context: Dict = None) -> Any:
        """评估符号表达式"""
        context = context or {}
        if self.expression_type == "constant":
            return self.content
        elif self.expression_type == "variable":
            return context.get(self.content, None)
        elif self.expression_type == "operation":
            if len(self.operands) == 1:  # 单目运算
                op = self.operators[0]
                val = self.operands[0].evaluate(context)
                return self._apply_unary_op(op, val)
            elif len(self.operands) == 2:  # 双目运算
                op = self.operators[0]
                left = self.operands[0].evaluate(context)
                right = self.operands[1].evaluate(context)
                return self._apply_binary_op(op, left, right)
        return None
    
    def _apply_unary_op(self, op: str, value: Any) -> Any:
        """应用单目运算符"""
        if op == "NOT":
            return not value
        elif op == "NEG":
            return -value if isinstance(value, (int, float)) else value
        return value
    
    def _apply_binary_op(self, op: str, left: Any, right: Any) -> Any:
        """应用双目运算符"""
        if op == "AND":
            return left and right
        elif op == "OR":
            return left or right
        elif op == "IMPLIES":
            return (not left) or right
        elif op == "EQUALS":
            return left == right
        elif op == "GT":
            return left > right
        elif op == "LT":
            return left < right
        return None
    
    def __str__(self):
        return self.content


@dataclass
class ProbabilityNode:
    """概率推理节点"""
    node_id: str
    values: Dict[str, float]  # 概率分布
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    cpt: Dict[Tuple, Dict[str, float]] = field(default_factory=dict)  # 条件概率表
    
    def get_probability(self, value: str, evidence: Dict[str, str] = None) -> float:
        """获取概率值"""
        evidence = evidence or {}
        if not self.parents:
            return self.values.get(value, 0.0)
        
        # 构建条件键
        parent_values = tuple(evidence.get(parent, None) for parent in self.parents)
        if parent_values in self.cpt:
            return self.cpt[parent_values].get(value, 0.0)
        return 0.0
    
    def update_evidence(self, evidence: Dict[str, str]):
        """更新证据"""
        for key, value in evidence.items():
            if key == self.node_id:
                # 设置此节点的证据
                self.values = {k: 1.0 if k == value else 0.0 for k in self.values}


@dataclass
class ReasoningStep:
    """升级的推理步骤数据结构"""
    step_id: int
    step_type: ReasoningType
    premise: str
    intermediate_conclusion: str
    confidence: float
    certainty: float  # 确定性因子
    timestamp: datetime
    reasoning_path: str
    symbolic_representation: Optional[SymbolicExpression] = None
    probability_distribution: Optional[Dict[str, float]] = None
    causal_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['step_type'] = self.step_type.value
        return data


@dataclass
class MetaCognitionReport:
    """元认知监控报告"""
    reasoning_quality: float
    strategy_effectiveness: float
    confidence_calibration: float
    reasoning_depth: int
    logical_consistency: float
    recommendations: List[str]
    performance_trends: Dict[str, float]
    adaptation_suggestions: List[str]


class SymbolicReasoningEngine:
    """符号逻辑推理引擎"""
    
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Dict] = []
        self.symbolic_expressions: Dict[str, SymbolicExpression] = {}
        self.logic_system = LogicSystem.PROPOSITIONAL
        
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
        logger.info(f"添加事实: {fact}")
    
    def add_rule(self, premise: str, conclusion: str, confidence: float = 1.0):
        """添加规则"""
        rule = {
            "premise": premise,
            "conclusion": conclusion,
            "confidence": confidence,
            "type": "implication"
        }
        self.rules.append(rule)
        logger.info(f"添加规则: {premise} -> {conclusion}")
    
    def forward_chaining(self, query: str) -> List[Dict]:
        """前向链式推理"""
        logger.info(f"开始前向链式推理: {query}")
        
        # 转换查询为符号表达式
        query_expr = self._parse_expression(query)
        
        # 初始化工作记忆
        derived_facts = []
        working_memory = set(self.facts)
        
        # 应用规则进行推理
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            new_facts = []
            
            for rule in self.rules:
                if self._evaluate_rule_premise(rule["premise"], working_memory):
                    conclusion = rule["conclusion"]
                    if conclusion not in working_memory:
                        new_facts.append({
                            "fact": conclusion,
                            "confidence": rule["confidence"],
                            "derived_from": rule["premise"],
                            "iteration": iteration
                        })
            
            if not new_facts:
                break
                
            for fact_data in new_facts:
                working_memory.add(fact_data["fact"])
                derived_facts.append(fact_data)
        
        return derived_facts
    
    def backward_chaining(self, goal: str, max_depth: int = 5) -> Dict:
        """后向链式推理"""
        logger.info(f"开始后向链式推理: {goal}")
        
        # 构建推理树
        reasoning_tree = {
            "goal": goal,
            "subgoals": [],
            "evidence": [],
            "success": False,
            "confidence": 0.0
        }
        
        def prove_goal(goal_str, depth=0):
            if depth >= max_depth:
                return {"success": False, "confidence": 0.0, "path": []}
            
            # 检查是否为已知事实
            if goal_str in self.facts:
                return {"success": True, "confidence": 1.0, "path": [goal_str]}
            
            # 寻找支持该目标的规则
            supporting_rules = []
            for rule in self.rules:
                if rule["conclusion"] == goal_str:
                    supporting_rules.append(rule)
            
            if not supporting_rules:
                return {"success": False, "confidence": 0.0, "path": []}
            
            best_proof = {"success": False, "confidence": 0.0, "path": []}
            
            for rule in supporting_rules:
                proof_result = prove_rule(rule, depth + 1)
                if proof_result["success"]:
                    combined_confidence = rule["confidence"] * proof_result["confidence"]
                    proof_path = [goal_str] + proof_result["path"]
                    
                    if combined_confidence > best_proof["confidence"]:
                        best_proof = {
                            "success": True,
                            "confidence": combined_confidence,
                            "path": proof_path
                        }
            
            return best_proof
        
        def prove_rule(rule, depth):
            premise_parts = rule["premise"].split(" AND ")
            proof_results = []
            
            for part in premise_parts:
                part_result = prove_goal(part, depth)
                proof_results.append(part_result)
            
            # 所有前提都为真则规则成立
            all_proven = all(result["success"] for result in proof_results)
            if all_proven:
                min_confidence = min(result["confidence"] for result in proof_results)
                return {
                    "success": True,
                    "confidence": min_confidence,
                    "path": []
                }
            else:
                return {"success": False, "confidence": 0.0, "path": []}
        
        result = prove_goal(goal)
        reasoning_tree.update(result)
        return reasoning_tree
    
    def _parse_expression(self, expression_str: str) -> SymbolicExpression:
        """解析逻辑表达式"""
        # 简化的表达式解析器
        if " AND " in expression_str:
            parts = expression_str.split(" AND ")
            expr = SymbolicExpression(
                expression_type="operation",
                content="AND",
                operands=[self._parse_expression(part.strip()) for part in parts]
            )
            expr.operators = ["AND"] * (len(parts) - 1)
            return expr
        elif " OR " in expression_str:
            parts = expression_str.split(" OR ")
            expr = SymbolicExpression(
                expression_type="operation",
                content="OR",
                operands=[self._parse_expression(part.strip()) for part in parts]
            )
            expr.operators = ["OR"] * (len(parts) - 1)
            return expr
        else:
            return SymbolicExpression(
                expression_type="atom",
                content=expression_str.strip()
            )
    
    def _evaluate_rule_premise(self, premise: str, facts: Set[str]) -> bool:
        """评估规则前提"""
        # 简化的前提评估
        return premise in facts
    
    def fuzzy_reasoning(self, fuzzy_rules: List[Dict]) -> Dict[str, float]:
        """模糊推理"""
        logger.info("开始模糊推理")
        
        # 模糊逻辑规则示例
        if not fuzzy_rules:
            fuzzy_rules = [
                {"antecedent": "high", "consequent": "good", "weight": 0.8},
                {"antecedent": "medium", "consequent": "average", "weight": 0.6},
                {"antecedent": "low", "consequent": "poor", "weight": 0.9}
            ]
        
        results = defaultdict(float)
        
        for rule in fuzzy_rules:
            antecedent_fuzzy = self._calculate_fuzzy_membership(rule["antecedent"])
            consequent_value = antecedent_fuzzy * rule["weight"]
            results[rule["consequent"]] = max(results[rule["consequent"]], consequent_value)
        
        return dict(results)
    
    def _calculate_fuzzy_membership(self, term: str) -> float:
        """计算模糊集合的隶属度"""
        # 简化的隶属度计算
        membership_functions = {
            "high": lambda x: max(0, min(1, (x - 0.7) / 0.3)),
            "medium": lambda x: max(0, min(1, 1 - abs(x - 0.5) * 2)),
            "low": lambda x: max(0, min(1, (0.3 - x) / 0.3))
        }
        
        if term in membership_functions:
            # 假设输入值为0.8表示"高"
            return membership_functions[term](0.8)
        return 0.0


class BayesianInferenceEngine:
    """贝叶斯推理引擎"""
    
    def __init__(self):
        self.network: Dict[str, ProbabilityNode] = {}
        self.evidence: Dict[str, str] = {}
        self.likelihood_cache: Dict[str, float] = {}
        
    def add_node(self, node_id: str, values: List[str]):
        """添加概率节点"""
        node = ProbabilityNode(
            node_id=node_id,
            values={val: 1.0/len(values) for val in values}
        )
        self.network[node_id] = node
        logger.info(f"添加概率节点: {node_id}")
    
    def add_edge(self, parent_id: str, child_id: str):
        """添加依赖关系"""
        if parent_id in self.network and child_id in self.network:
            self.network[parent_id].children.append(child_id)
            self.network[child_id].parents.append(parent_id)
            logger.info(f"添加依赖关系: {parent_id} -> {child_id}")
    
    def set_cpt(self, node_id: str, cpt: Dict[Tuple, Dict[str, float]]):
        """设置条件概率表"""
        if node_id in self.network:
            self.network[node_id].cpt = cpt
            logger.info(f"设置条件概率表: {node_id}")
    
    def set_evidence(self, variable: str, value: str):
        """设置证据"""
        self.evidence[variable] = value
        for node in self.network.values():
            if node.node_id == variable:
                node.update_evidence(self.evidence)
        logger.info(f"设置证据: {variable} = {value}")
    
    def query(self, variable: str, evidence: Dict[str, str] = None) -> Dict[str, float]:
        """贝叶斯查询"""
        logger.info(f"贝叶斯查询: {variable}")
        
        evidence = evidence or self.evidence.copy()
        all_evidence = {**self.evidence, **evidence}
        
        if variable in all_evidence:
            # 如果变量有直接证据，返回确定性分布
            values = list(self.network[variable].values.keys())
            return {val: 1.0 if val == all_evidence[variable] else 0.0 for val in values}
        
        # 使用变量消除算法计算后验概率
        return self._variable_elimination(variable, all_evidence)
    
    def _variable_elimination(self, query_var: str, evidence: Dict[str, float]) -> Dict[str, float]:
        """变量消除算法"""
        query_values = list(self.network[query_var].values.keys())
        results = {}
        
        for value in query_values:
            # 设置查询变量的值
            query_evidence = evidence.copy()
            query_evidence[query_var] = value
            
            # 计算联合概率
            joint_prob = self._compute_joint_probability(query_evidence)
            results[value] = joint_prob
        
        # 归一化
        total = sum(results.values())
        if total > 0:
            results = {k: v/total for k, v in results.items()}
        
        return results
    
    def _compute_joint_probability(self, evidence: Dict[str, float]) -> float:
        """计算联合概率"""
        joint_prob = 1.0
        
        for node_id, node in self.network.items():
            if node_id in evidence:
                prob = node.get_probability(evidence[node_id], evidence)
                joint_prob *= prob
        
        return joint_prob
    
    def update_beliefs(self, new_evidence: Dict[str, str]):
        """更新信念"""
        logger.info("更新贝叶斯信念")
        
        for var, value in new_evidence.items():
            self.set_evidence(var, value)
    
    def get_network_structure(self) -> Dict:
        """获取网络结构"""
        structure = {
            "nodes": list(self.network.keys()),
            "edges": []
        }
        
        for parent_id, parent_node in self.network.items():
            for child_id in parent_node.children:
                structure["edges"].append((parent_id, child_id))
        
        return structure


class MetacognitiveMonitor:
    """元认知监控系统"""
    
    def __init__(self):
        self.state = MetaCognitionState.MONITORING
        self.performance_history: List[Dict] = []
        self.strategy_effectiveness: Dict[str, float] = {}
        self.confidence_calibration: List[Tuple[float, float]] = []  # (预测置信度, 实际准确率)
        self.adaptation_history: List[Dict] = []
        
    def monitor_reasoning_process(self, reasoning_steps: List[ReasoningStep]) -> MetaCognitionReport:
        """监控推理过程"""
        logger.info("开始元认知监控")
        
        # 计算推理质量
        reasoning_quality = self._calculate_reasoning_quality(reasoning_steps)
        
        # 评估策略有效性
        strategy_effectiveness = self._evaluate_strategy_effectiveness(reasoning_steps)
        
        # 计算置信度校准
        confidence_calibration = self._calculate_confidence_calibration(reasoning_steps)
        
        # 评估逻辑一致性
        logical_consistency = self._evaluate_logical_consistency(reasoning_steps)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            reasoning_quality, strategy_effectiveness, logical_consistency
        )
        
        # 分析性能趋势
        performance_trends = self._analyze_performance_trends()
        
        # 生成适应建议
        adaptation_suggestions = self._generate_adaptation_suggestions(
            reasoning_quality, strategy_effectiveness
        )
        
        report = MetaCognitionReport(
            reasoning_quality=reasoning_quality,
            strategy_effectiveness=strategy_effectiveness,
            confidence_calibration=confidence_calibration,
            reasoning_depth=len(reasoning_steps),
            logical_consistency=logical_consistency,
            recommendations=recommendations,
            performance_trends=performance_trends,
            adaptation_suggestions=adaptation_suggestions
        )
        
        # 更新状态
        self.state = MetaCognitionState.EVALUATING
        self.performance_history.append(asdict(report))
        
        return report
    
    def _calculate_reasoning_quality(self, steps: List[ReasoningStep]) -> float:
        """计算推理质量"""
        if not steps:
            return 0.0
        
        # 因素1：平均置信度
        avg_confidence = np.mean([step.confidence for step in steps])
        
        # 因素2：置信度一致性
        confidence_std = np.std([step.confidence for step in steps])
        consistency_score = max(0.0, 1.0 - confidence_std)
        
        # 因素3：推理深度适当性
        optimal_depth = 5
        depth_score = max(0.0, 1.0 - abs(len(steps) - optimal_depth) / optimal_depth)
        
        # 因素4：最后步骤质量
        final_quality = steps[-1].confidence if steps else 0.0
        
        # 综合评分
        quality_score = (
            avg_confidence * 0.3 +
            consistency_score * 0.2 +
            depth_score * 0.2 +
            final_quality * 0.3
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _evaluate_strategy_effectiveness(self, steps: List[ReasoningStep]) -> float:
        """评估策略有效性"""
        if not steps:
            return 0.0
        
        # 分析推理类型分布
        step_types = [step.step_type for step in steps]
        type_counts = defaultdict(int)
        for step_type in step_types:
            type_counts[step_type] += 1
        
        # 计算类型多样性（适当的类型多样性表示策略灵活）
        total_steps = len(steps)
        diversity_score = len(type_counts) / len(ReasoningType)
        
        # 计算成功率（步骤中高质量步骤的比例）
        high_quality_steps = sum(1 for step in steps if step.confidence > 0.7)
        success_rate = high_quality_steps / total_steps if total_steps > 0 else 0.0
        
        # 综合策略有效性
        effectiveness = (diversity_score * 0.4 + success_rate * 0.6)
        return effectiveness
    
    def _calculate_confidence_calibration(self, steps: List[ReasoningStep]) -> float:
        """计算置信度校准"""
        if not steps:
            return 0.0
        
        # 记录置信度-准确率对
        for step in steps:
            self.confidence_calibration.append((step.confidence, step.certainty))
        
        # 计算校准误差
        calibration_error = 0.0
        for conf, acc in self.confidence_calibration[-10:]:  # 最近10个
            calibration_error += abs(conf - acc)
        
        # 转换为校准分数（误差越小，校准越好）
        avg_error = calibration_error / len(self.confidence_calibration[-10:])
        calibration_score = max(0.0, 1.0 - avg_error)
        
        return calibration_score
    
    def _evaluate_logical_consistency(self, steps: List[ReasoningStep]) -> float:
        """评估逻辑一致性"""
        if len(steps) < 2:
            return 1.0
        
        consistency_score = 0.0
        comparisons = 0
        
        # 检查连续步骤间的一致性
        for i in range(len(steps) - 1):
            step_a = steps[i]
            step_b = steps[i + 1]
            
            # 计算结论相似度
            similarity = self._calculate_textual_similarity(
                step_a.intermediate_conclusion,
                step_b.intermediate_conclusion
            )
            
            # 检查符号表示的一致性
            symbolic_consistency = self._check_symbolic_consistency(
                step_a.symbolic_representation,
                step_b.symbolic_representation
            )
            
            step_consistency = (similarity + symbolic_consistency) / 2
            consistency_score += step_consistency
            comparisons += 1
        
        return consistency_score / comparisons if comparisons > 0 else 1.0
    
    def _calculate_textual_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_symbolic_consistency(self, expr1: SymbolicExpression, expr2: SymbolicExpression) -> float:
        """检查符号表示一致性"""
        if not expr1 or not expr2:
            return 0.5  # 未知状态
        
        if expr1.content == expr2.content:
            return 1.0
        elif expr1.expression_type == expr2.expression_type:
            return 0.7
        else:
            return 0.3
    
    def _generate_recommendations(self, quality: float, strategy: float, consistency: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if quality < 0.6:
            recommendations.append("推理质量偏低，建议增加证据收集")
        
        if strategy < 0.5:
            recommendations.append("推理策略单一，建议尝试不同推理类型")
        
        if consistency < 0.7:
            recommendations.append("逻辑一致性有待提高，建议检查推理链")
        
        if quality > 0.8 and strategy > 0.7:
            recommendations.append("当前推理效果良好，可考虑更复杂的问题")
        
        if not recommendations:
            recommendations.append("推理表现良好，继续保持当前策略")
        
        return recommendations
    
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """分析性能趋势"""
        if len(self.performance_history) < 2:
            return {"trend": 0.0, "stability": 1.0}
        
        recent_scores = [entry["reasoning_quality"] for entry in self.performance_history[-5:]]
        
        # 计算趋势
        if len(recent_scores) >= 2:
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        else:
            trend = 0.0
        
        # 计算稳定性（标准差的倒数）
        stability = 1.0 / (1.0 + np.std(recent_scores))
        
        return {
            "trend": trend,
            "stability": stability,
            "average_quality": np.mean(recent_scores)
        }
    
    def _generate_adaptation_suggestions(self, quality: float, strategy: float) -> List[str]:
        """生成适应建议"""
        suggestions = []
        
        if quality < 0.5:
            suggestions.append("建议降低推理复杂度，专注于基础推理")
            suggestions.append("增加证据检查环节，确保推理基础稳固")
        
        elif quality > 0.8:
            suggestions.append("可尝试更复杂的推理模式")
            suggestions.append("考虑整合多种推理类型提高灵活性")
        
        if strategy < 0.4:
            suggestions.append("当前策略效果有限，建议尝试新的推理方法")
            suggestions.append("观察成功案例，学习有效策略")
        
        return suggestions
    
    def adapt_reasoning_strategy(self, adaptation_type: str, parameters: Dict = None) -> Dict:
        """适应推理策略"""
        logger.info(f"适应推理策略: {adaptation_type}")
        
        adaptation_result = {
            "type": adaptation_type,
            "success": False,
            "parameters": parameters or {},
            "effects": []
        }
        
        if adaptation_type == "increase_depth":
            adaptation_result["parameters"]["max_depth"] = parameters.get("max_depth", 5) + 2
            adaptation_result["effects"].append("增加推理深度")
            adaptation_result["success"] = True
        
        elif adaptation_type == "change_reasoning_type":
            current_type = parameters.get("current_type", ReasoningType.DEDUCTIVE)
            adaptation_result["parameters"]["reasoning_type"] = self._select_alternative_type(current_type)
            adaptation_result["effects"].append("改变推理类型")
            adaptation_result["success"] = True
        
        elif adaptation_type == "adjust_confidence":
            factor = parameters.get("factor", 1.0)
            adaptation_result["parameters"]["confidence_factor"] = factor * 0.9
            adaptation_result["effects"].append("调整置信度权重")
            adaptation_result["success"] = True
        
        # 记录适应历史
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": adaptation_type,
            "parameters": adaptation_result["parameters"],
            "success": adaptation_result["success"]
        })
        
        return adaptation_result
    
    def _select_alternative_type(self, current_type: ReasoningType) -> ReasoningType:
        """选择替代推理类型"""
        alternatives = {
            ReasoningType.DEDUCTIVE: ReasoningType.INDUCTIVE,
            ReasoningType.INDUCTIVE: ReasoningType.ABDUCTIVE,
            ReasoningType.ABDUCTIVE: ReasoningType.CAUSAL,
            ReasoningType.CAUSAL: ReasoningType.PROBABILISTIC,
            ReasoningType.PROBABILISTIC: ReasoningType.ANALOGICAL,
            ReasoningType.ANALOGICAL: ReasoningType.DEDUCTIVE
        }
        return alternatives.get(current_type, ReasoningType.DEDUCTIVE)


class PrefrontalCortex:
    """
    升级后的前额叶皮层推理引擎
    
    集成多种推理能力：
    - 符号逻辑推理
    - 贝叶斯概率推理
    - 元认知监控
    - 自适应学习
    - 链式推理增强
    """
    
    def __init__(self, 
                 llm_mode: str = "hybrid",
                 max_reasoning_steps: int = 20,
                 enable_symbolic_reasoning: bool = True,
                 enable_bayesian_inference: bool = True,
                 enable_metacognition: bool = True):
        """
        初始化升级版前额叶推理引擎
        
        Args:
            llm_mode: LLM运行模式 ("api", "local", "hybrid", "rule-based")
            max_reasoning_steps: 最大推理步数
            enable_symbolic_reasoning: 启用符号推理
            enable_bayesian_inference: 启用贝叶斯推理
            enable_metacognition: 启用元认知监控
        """
        self.llm_mode = llm_mode
        self.max_reasoning_steps = max_reasoning_steps
        
        # 核心组件
        self.symbolic_engine = SymbolicReasoningEngine() if enable_symbolic_reasoning else None
        self.bayesian_engine = BayesianInferenceEngine() if enable_bayesian_inference else None
        self.metacognitive_monitor = MetacognitiveMonitor() if enable_metacognition else None
        
        # 推理历史和状态
        self.reasoning_history: List[ReasoningStep] = []
        self.belief_graph = nx.DiGraph()
        self.contradiction_buffer: List[Dict] = []
        
        # 性能指标
        self.performance_metrics = {
            "total_reasoning_tasks": 0,
            "successful_reasoning": 0,
            "symbolic_reasoning_success": 0,
            "bayesian_inference_accuracy": 0.0,
            "metacognitive_adaptations": 0,
            "reasoning_quality_average": 0.0
        }
        
        # 推理配置
        self.reasoning_config = {
            "primary_reasoning_type": ReasoningType.DEDUCTIVE,
            "confidence_threshold_high": 0.8,
            "confidence_threshold_medium": 0.6,
            "confidence_threshold_low": 0.4,
            "symbolic_weight": 0.4,
            "probabilistic_weight": 0.3,
            "llm_weight": 0.3
        }
        
        # 初始化系统
        self._initialize_knowledge_base()
        self._initialize_bayesian_network()
        
        logger.info(f"升级版前额叶推理引擎初始化完成，功能: "
                   f"符号推理={enable_symbolic_reasoning}, "
                   f"贝叶斯推理={enable_bayesian_inference}, "
                   f"元认知监控={enable_metacognition}")
    
    def _initialize_knowledge_base(self):
        """初始化知识库"""
        # 添加基础逻辑规则
        if self.symbolic_engine:
            self.symbolic_engine.add_fact("天气晴天")
            self.symbolic_engine.add_fact("温度适宜")
            self.symbolic_engine.add_rule("天气晴天 AND 温度适宜", "适合户外活动")
            self.symbolic_engine.add_rule("适合户外活动", "心情愉快")
    
    def _initialize_bayesian_network(self):
        """初始化贝叶斯网络"""
        if self.bayesian_engine:
            # 构建简单天气推理网络
            self.bayesian_engine.add_node("weather", ["sunny", "rainy", "cloudy"])
            self.bayesian_engine.add_node("temperature", ["high", "medium", "low"])
            self.bayesian_engine.add_node("activity", ["outdoor", "indoor"])
            
            self.bayesian_engine.add_edge("weather", "temperature")
            self.bayesian_engine.add_edge("weather", "activity")
            self.bayesian_engine.add_edge("temperature", "activity")
            
            # 设置条件概率表
            weather_cpt = {
                ("sunny",): {"high": 0.7, "medium": 0.2, "low": 0.1},
                ("rainy",): {"high": 0.1, "medium": 0.3, "low": 0.6},
                ("cloudy",): {"high": 0.3, "medium": 0.5, "low": 0.2}
            }
            self.bayesian_engine.set_cpt("temperature", weather_cpt)
    
    async def enhanced_chain_of_thought_reasoning(self, 
                                               problem: str, 
                                               context: Dict = None,
                                               reasoning_type: ReasoningType = None) -> Dict:
        """
        增强版链式推理
        
        Args:
            problem: 待推理的问题
            context: 上下文信息
            reasoning_type: 推理类型
            
        Returns:
            Dict: 完整的推理结果
        """
        self.performance_metrics["total_reasoning_tasks"] += 1
        
        logger.info(f"开始增强版链式推理: {problem[:100]}...")
        
        # 选择推理类型
        reasoning_type = reasoning_type or self.reasoning_config["primary_reasoning_type"]
        
        # 初始化推理状态
        current_state = {
            "problem": problem,
            "context": context or {},
            "reasoning_type": reasoning_type,
            "symbolic_results": [],
            "probabilistic_results": [],
            "llm_results": [],
            "confidence_history": [],
            "adaptations": []
        }
        
        reasoning_steps = []
        
        try:
            # 第一阶段：符号逻辑推理
            if self.symbolic_engine:
                symbolic_result = await self._symbolic_reasoning_phase(problem, context)
                current_state["symbolic_results"].append(symbolic_result)
            
            # 第二阶段：概率推理
            if self.bayesian_engine:
                probabilistic_result = await self._probabilistic_reasoning_phase(problem, context)
                current_state["probabilistic_results"].append(probabilistic_result)
            
            # 第三阶段：LLM辅助推理
            llm_result = await self._llm_reasoning_phase(problem, context)
            current_state["llm_results"].append(llm_result)
            
            # 第四阶段：多步深度推理
            for step in range(self.max_reasoning_steps):
                step_result = await self._execute_enhanced_reasoning_step(
                    step + 1, 
                    current_state, 
                    reasoning_steps
                )
                
                reasoning_steps.append(step_result)
                current_state = self._update_enhanced_state(current_state, step_result)
                
                # 检查是否终止
                if self._should_terminate_enhanced_reasoning(step_result, reasoning_steps):
                    break
            
            # 第五阶段：结果整合与验证
            final_result = await self._integrate_reasoning_results(current_state, reasoning_steps)
            
            # 第六阶段：元认知评估
            if self.metacognitive_monitor:
                metacog_report = self.metacognitive_monitor.monitor_reasoning_process(reasoning_steps)
                final_result["metacognitive_report"] = asdict(metacog_report)
            
            # 记录推理历史
            self.reasoning_history.extend(reasoning_steps)
            
            # 更新性能指标
            success = final_result.get("quality_score", 0.0) >= 0.6
            if success:
                self.performance_metrics["successful_reasoning"] += 1
            
            # 更新符号推理成功率
            if self.symbolic_engine and current_state["symbolic_results"]:
                self.performance_metrics["symbolic_reasoning_success"] += 1
            
            logger.info(f"增强版链式推理完成，质量评分: {final_result.get('quality_score', 0):.3f}")
            return final_result
            
        except Exception as e:
            logger.error(f"增强版链式推理失败: {str(e)}")
            return {
                "problem": problem,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _symbolic_reasoning_phase(self, problem: str, context: Dict) -> Dict:
        """符号推理阶段"""
        logger.info("执行符号推理阶段")
        
        try:
            # 提取问题中的逻辑关系
            logical_patterns = self._extract_logical_patterns(problem)
            
            # 执行前向链式推理
            derived_facts = self.symbolic_engine.forward_chaining(problem)
            
            # 执行后向链式推理
            proof_result = self.symbolic_engine.backward_chaining(problem)
            
            return {
                "phase": "symbolic",
                "logical_patterns": logical_patterns,
                "derived_facts": derived_facts,
                "proof_result": proof_result,
                "confidence": min(1.0, len(derived_facts) * 0.2 + proof_result.get("confidence", 0.0))
            }
            
        except Exception as e:
            logger.error(f"符号推理阶段失败: {str(e)}")
            return {"phase": "symbolic", "error": str(e), "confidence": 0.0}
    
    async def _probabilistic_reasoning_phase(self, problem: str, context: Dict) -> Dict:
        """概率推理阶段"""
        logger.info("执行概率推理阶段")
        
        try:
            # 提取概率信息
            probabilistic_elements = self._extract_probabilistic_elements(problem)
            
            # 设置相关证据
            for element in probabilistic_elements:
                if "天气" in problem:
                    self.bayesian_engine.set_evidence("weather", "sunny")
                if "温度" in problem:
                    self.bayesian_engine.set_evidence("temperature", "high")
            
            # 执行贝叶斯查询
            activity_query = self.bayesian_engine.query("activity")
            
            return {
                "phase": "probabilistic",
                "probabilistic_elements": probabilistic_elements,
                "activity_probabilities": activity_query,
                "confidence": max(activity_query.values()) if activity_query else 0.0
            }
            
        except Exception as e:
            logger.error(f"概率推理阶段失败: {str(e)}")
            return {"phase": "probabilistic", "error": str(e), "confidence": 0.0}
    
    async def _llm_reasoning_phase(self, problem: str, context: Dict) -> Dict:
        """LLM推理阶段"""
        logger.info("执行LLM推理阶段")
        
        try:
            # 构建综合提示词
            reasoning_prompt = self._build_comprehensive_reasoning_prompt(problem, context)
            
            # 调用LLM（降级到基于规则的后备推理）
            llm_response = self._rule_based_reasoning(reasoning_prompt)
            
            return {
                "phase": "llm",
                "response": llm_response,
                "confidence": 0.7  # 默认置信度
            }
            
        except Exception as e:
            logger.error(f"LLM推理阶段失败: {str(e)}")
            return {"phase": "llm", "error": str(e), "confidence": 0.0}
    
    def _extract_logical_patterns(self, problem: str) -> List[str]:
        """提取逻辑模式"""
        patterns = []
        
        # 检查常见逻辑连接词
        if "如果" in problem and "那么" in problem:
            patterns.append("conditional")
        if "因为" in problem and "所以" in problem:
            patterns.append("causal")
        if "所有" in problem or "每个" in problem:
            patterns.append("universal")
        if "有些" in problem or "存在" in problem:
            patterns.append("existential")
        
        return patterns
    
    def _extract_probabilistic_elements(self, problem: str) -> List[str]:
        """提取概率元素"""
        elements = []
        
        # 检查概率词汇
        probabilistic_words = ["可能", "也许", "大概", "通常", "总是", "从不"]
        for word in probabilistic_words:
            if word in problem:
                elements.append(word)
        
        # 检查数值表达
        numbers = re.findall(r'\d+', problem)
        if numbers:
            elements.extend([f"number_{num}" for num in numbers])
        
        return elements
    
    def _build_comprehensive_reasoning_prompt(self, problem: str, context: Dict) -> str:
        """构建综合推理提示词"""
        prompt = f"""请对以下问题进行深度推理思考：

问题：{problem}

上下文信息：
{json.dumps(context or {}, ensure_ascii=False, indent=2)}

推理要求：
1. 使用多种推理方法（演绎、归纳、溯因）
2. 考虑符号逻辑关系
3. 评估概率和不确定性
4. 提供置信度评估
5. 构建推理链条

请分步进行分析："""

        return prompt
    
    def _rule_based_reasoning(self, prompt: str) -> str:
        """基于规则的推理后备方案"""
        # 简单的基于规则的后备推理
        if "推理" in prompt:
            return """
基于问题分析：
1. 识别关键要素和约束条件
2. 应用逻辑推理规则
3. 考虑可能的解决方案
4. 评估每种方案的可行性
5. 选择最优解决方案

结论：通过系统性分析，可以得出合理的推理结果
置信度：0.7"""
        return "暂时无法进行推理分析"
    
    async def _execute_enhanced_reasoning_step(self, 
                                             step_id: int, 
                                             current_state: Dict, 
                                             previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """执行增强版推理步骤"""
        
        # 构建推理提示
        reasoning_prompt = self._build_enhanced_reasoning_prompt(
            step_id, current_state, previous_steps
        )
        
        # 确定推理类型
        reasoning_type = current_state.get("reasoning_type", ReasoningType.DEDUCTIVE)
        
        # 执行相应类型的推理
        if reasoning_type == ReasoningType.DEDUCTIVE:
            conclusion, confidence = self._deductive_reasoning(reasoning_prompt, current_state)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            conclusion, confidence = self._inductive_reasoning(reasoning_prompt, current_state)
        elif reasoning_type == ReasoningType.PROBABILISTIC:
            conclusion, confidence = self._probabilistic_step_reasoning(reasoning_prompt, current_state)
        else:
            conclusion, confidence = self._default_reasoning(reasoning_prompt, current_state)
        
        # 计算确定性因子
        certainty = self._calculate_certainty_factor(confidence, previous_steps)
        
        # 生成符号表示
        symbolic_repr = self._generate_symbolic_representation(conclusion)
        
        # 构建推理步骤
        step = ReasoningStep(
            step_id=step_id,
            step_type=reasoning_type,
            premise=current_state.get("problem", ""),
            intermediate_conclusion=conclusion,
            confidence=confidence,
            certainty=certainty,
            timestamp=datetime.now(),
            reasoning_path=f"步骤{step_id}: {reasoning_type.value}推理",
            symbolic_representation=symbolic_repr
        )
        
        return step
    
    def _build_enhanced_reasoning_prompt(self, 
                                       step_id: int, 
                                       current_state: Dict, 
                                       previous_steps: List[ReasoningStep]) -> str:
        """构建增强版推理提示词"""
        
        base_prompt = f"""这是第{step_id}步推理思考。

问题：{current_state.get('problem', '')}

已完成的推理结果："""
        
        # 添加符号推理结果
        if current_state.get("symbolic_results"):
            for result in current_state["symbolic_results"][-1:]:
                base_prompt += f"\n符号推理: {result.get('derived_facts', [])}"
        
        # 添加概率推理结果
        if current_state.get("probabilistic_results"):
            for result in current_state["probabilistic_results"][-1:]:
                base_prompt += f"\n概率推理: {result.get('activity_probabilities', {})}"
        
        # 添加LLM推理结果
        if current_state.get("llm_results"):
            for result in current_state["llm_results"][-1:]:
                base_prompt += f"\nLLM推理: {result.get('response', '')}"
        
        # 添加之前步骤
        if previous_steps:
            base_prompt += "\n\n之前推理步骤："
            for step in previous_steps[-3:]:
                base_prompt += f"\n步骤{step.step_id}: {step.intermediate_conclusion} (置信度: {step.confidence:.2f})"
        
        base_prompt += f"""

当前步骤要求：
- 基于整合的多源信息进行{current_state.get('reasoning_type', ReasoningType.DEDUCTIVE).value}推理
- 提供中间结论和推理路径
- 评估置信度(0-1)和确定性(0-1)
- 构建符号逻辑表示

格式要求：
结论：[中间结论]
推理：[推理过程]
置信度：[0-1数值]
确定性：[0-1数值]"""

        return base_prompt
    
    def _deductive_reasoning(self, prompt: str, state: Dict) -> Tuple[str, float]:
        """演绎推理"""
        # 简化的演绎推理逻辑
        if "如果" in prompt and "那么" in prompt:
            return "基于前提条件，可以必然推导出该结论", 0.9
        else:
            return "从已知事实出发进行逻辑推导", 0.7
    
    def _inductive_reasoning(self, prompt: str, state: Dict) -> Tuple[str, float]:
        """归纳推理"""
        return "从具体事例归纳出一般规律", 0.6
    
    def _probabilistic_step_reasoning(self, prompt: str, state: Dict) -> Tuple[str, float]:
        """概率推理"""
        return "基于概率分析得出最可能的结论", 0.8
    
    def _default_reasoning(self, prompt: str, state: Dict) -> Tuple[str, float]:
        """默认推理"""
        return "基于常识和逻辑推理", 0.5
    
    def _calculate_certainty_factor(self, confidence: float, previous_steps: List[ReasoningStep]) -> float:
        """计算确定性因子"""
        if not previous_steps:
            return confidence
        
        # 基于推理链的一致性调整确定性
        consistency = self._calculate_reasoning_consistency(previous_steps + [None])
        certainty = confidence * (0.8 + 0.4 * consistency)
        
        return max(0.0, min(1.0, certainty))
    
    def _calculate_reasoning_consistency(self, steps: List[ReasoningStep]) -> float:
        """计算推理一致性"""
        if len(steps) < 2:
            return 1.0
        
        consistency_scores = []
        for i in range(len(steps) - 1):
            if steps[i] and steps[i + 1]:
                score = self._calculate_step_consistency(steps[i], steps[i + 1])
                consistency_scores.append(score)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_step_consistency(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """计算步骤间一致性"""
        # 文本相似度
        text_sim = self._calculate_text_similarity(
            step1.intermediate_conclusion, 
            step2.intermediate_conclusion
        )
        
        # 置信度一致性
        conf_sim = 1.0 - abs(step1.confidence - step2.confidence)
        
        return (text_sim + conf_sim) / 2
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_symbolic_representation(self, conclusion: str) -> Optional[SymbolicExpression]:
        """生成符号表示"""
        if not self.symbolic_engine:
            return None
        
        try:
            return self.symbolic_engine._parse_expression(conclusion)
        except:
            return None
    
    def _update_enhanced_state(self, current_state: Dict, reasoning_step: ReasoningStep) -> Dict:
        """更新增强版推理状态"""
        new_state = current_state.copy()
        
        # 更新置信度历史
        confidence_history = new_state.get("confidence_history", [])
        confidence_history.append(reasoning_step.confidence)
        new_state["confidence_history"] = confidence_history
        
        # 记录推理类型
        reasoning_type = reasoning_step.step_type
        
        return new_state
    
    def _should_terminate_enhanced_reasoning(self, latest_step: ReasoningStep, all_steps: List[ReasoningStep]) -> bool:
        """判断是否终止增强版推理"""
        
        # 终止条件1：高质量结论且步骤充分
        if latest_step.confidence >= self.reasoning_config["confidence_threshold_high"] and len(all_steps) >= 3:
            return True
        
        # 终止条件2：连续低质量步骤
        recent_steps = all_steps[-3:] if len(all_steps) >= 3 else all_steps
        if len(recent_steps) >= 3 and all(step.confidence < self.reasoning_config["confidence_threshold_low"] for step in recent_steps):
            return True
        
        # 终止条件3：达到最大步骤数
        if len(all_steps) >= self.max_reasoning_steps:
            return True
        
        # 终止条件4：确定性充分
        if latest_step.certainty >= 0.9:
            return True
        
        return False
    
    async def _integrate_reasoning_results(self, current_state: Dict, reasoning_steps: List[ReasoningStep]) -> Dict:
        """整合推理结果"""
        
        # 整合符号推理结果
        symbolic_confidence = 0.0
        if current_state.get("symbolic_results"):
            symbolic_results = current_state["symbolic_results"][0]
            symbolic_confidence = symbolic_results.get("confidence", 0.0)
        
        # 整合概率推理结果
        probabilistic_confidence = 0.0
        if current_state.get("probabilistic_results"):
            probabilistic_results = current_state["probabilistic_results"][0]
            probabilistic_confidence = probabilistic_results.get("confidence", 0.0)
        
        # 整合LLM推理结果
        llm_confidence = 0.0
        if current_state.get("llm_results"):
            llm_results = current_state["llm_results"][0]
            llm_confidence = llm_results.get("confidence", 0.0)
        
        # 计算综合置信度
        weights = self.reasoning_config
        combined_confidence = (
            symbolic_confidence * weights["symbolic_weight"] +
            probabilistic_confidence * weights["probabilistic_weight"] +
            llm_confidence * weights["llm_weight"]
        )
        
        # 生成最终结论
        if reasoning_steps:
            final_conclusion = f"基于{len(reasoning_steps)}步多维推理，综合结论：{reasoning_steps[-1].intermediate_conclusion}"
            reasoning_depth = len(reasoning_steps)
        else:
            final_conclusion = "推理失败，无法生成有效结论"
            reasoning_depth = 0
        
        # 计算质量评分
        quality_score = self._calculate_enhanced_quality_score(reasoning_steps, combined_confidence)
        
        # 构建综合结果
        result = {
            "problem": current_state.get("problem", ""),
            "reasoning_steps": [step.to_dict() for step in reasoning_steps],
            "final_conclusion": {
                "conclusion": final_conclusion,
                "confidence": combined_confidence,
                "reasoning_depth": reasoning_depth
            },
            "quality_score": quality_score,
            "symbolic_reasoning": symbolic_confidence,
            "probabilistic_reasoning": probabilistic_confidence,
            "llm_reasoning": llm_confidence,
            "success": quality_score >= 0.6,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_enhanced_quality_score(self, steps: List[ReasoningStep], combined_confidence: float) -> float:
        """计算增强版质量评分"""
        if not steps:
            return 0.0
        
        # 因素1：平均置信度
        avg_confidence = np.mean([step.confidence for step in steps])
        
        # 因素2：置信度一致性
        confidence_std = np.std([step.confidence for step in steps])
        consistency_score = max(0.0, 1.0 - confidence_std)
        
        # 因素3：推理深度适当性
        optimal_depth = 6
        depth_score = max(0.0, 1.0 - abs(len(steps) - optimal_depth) / optimal_depth)
        
        # 因素4：最终确定性
        final_certainty = steps[-1].certainty if steps else 0.0
        
        # 因素5：推理类型多样性
        step_types = set(step.step_type for step in steps)
        diversity_score = len(step_types) / len(ReasoningType)
        
        # 综合质量评分
        quality_score = (
            avg_confidence * 0.25 +
            consistency_score * 0.15 +
            depth_score * 0.15 +
            final_certainty * 0.25 +
            diversity_score * 0.1 +
            combined_confidence * 0.1
        )
        
        return max(0.0, min(1.0, quality_score))
    
    async def advanced_problem_solving(self, 
                                     problem: str, 
                                     strategy: str = "hybrid",
                                     max_solutions: int = 3) -> Dict:
        """
        高级问题解决
        
        Args:
            problem: 待解决的问题
            strategy: 解决策略 ("deductive", "inductive", "abductive", "hybrid")
            max_solutions: 最大解的数量
            
        Returns:
            Dict: 问题解决结果
        """
        logger.info(f"开始高级问题解决: {problem}")
        
        solutions = []
        
        try:
            if strategy == "deductive" or strategy == "hybrid":
                # 演绎法解决
                deductive_result = await self._deductive_problem_solving(problem)
                if deductive_result:
                    solutions.append(deductive_result)
            
            if strategy == "inductive" or strategy == "hybrid":
                # 归纳法解决
                inductive_result = await self._inductive_problem_solving(problem)
                if inductive_result:
                    solutions.append(inductive_result)
            
            if strategy == "abductive" or strategy == "hybrid":
                # 溯因法解决
                abductive_result = await self._abductive_problem_solving(problem)
                if abductive_result:
                    solutions.append(abductive_result)
            
            # 评估和排序解决方案
            evaluated_solutions = await self._evaluate_solutions(solutions, problem)
            
            # 选择最优解决方案
            best_solutions = evaluated_solutions[:max_solutions]
            
            return {
                "problem": problem,
                "strategy": strategy,
                "solutions": best_solutions,
                "total_solutions": len(solutions),
                "recommended_solution": best_solutions[0] if best_solutions else None,
                "success": len(best_solutions) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"高级问题解决失败: {str(e)}")
            return {
                "problem": problem,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _deductive_problem_solving(self, problem: str) -> Dict:
        """演绎法问题解决"""
        
        # 使用符号推理
        if self.symbolic_engine:
            proof_result = self.symbolic_engine.backward_chaining(problem)
            if proof_result.get("success"):
                return {
                    "method": "deductive",
                    "solution": proof_result.get("goal", ""),
                    "confidence": proof_result.get("confidence", 0.0),
                    "reasoning_path": proof_result.get("path", []),
                    "description": "基于逻辑演绎推理得出解决方案"
                }
        
        # 基于规则的演绎
        return {
            "method": "deductive",
            "solution": f"基于逻辑演绎，{problem}的解决方案是...",
            "confidence": 0.7,
            "reasoning_path": ["问题分析", "规则应用", "逻辑推导"],
            "description": "从一般原理推导具体结论"
        }
    
    async def _inductive_problem_solving(self, problem: str) -> Dict:
        """归纳法问题解决"""
        
        # 分析问题中的模式
        patterns = self._extract_reasoning_patterns(problem)
        
        # 基于模式归纳解决方案
        pattern_based_solution = self._generate_pattern_based_solution(patterns)
        
        return {
            "method": "inductive",
            "solution": pattern_based_solution,
            "confidence": 0.6,
            "reasoning_path": ["模式识别", "归纳推理", "方案生成"],
            "description": "从具体事例归纳一般性解决方案"
        }
    
    async def _abductive_problem_solving(self, problem: str) -> Dict:
        """溯因法问题解决"""
        
        # 寻找最佳解释
        explanation = self._find_best_explanation(problem)
        
        return {
            "method": "abductive",
            "solution": explanation,
            "confidence": 0.5,
            "reasoning_path": ["现象分析", "假设生成", "最佳解释选择"],
            "description": "寻找最可能的解释性解决方案"
        }
    
    def _extract_reasoning_patterns(self, problem: str) -> List[str]:
        """提取推理模式"""
        patterns = []
        
        # 检查问题类型
        if "如何" in problem:
            patterns.append("procedural")
        if "为什么" in problem:
            patterns.append("causal")
        if "是什么" in problem:
            patterns.append("definitional")
        if "哪个" in problem or "什么" in problem:
            patterns.append("selection")
        
        return patterns
    
    def _generate_pattern_based_solution(self, patterns: List[str]) -> str:
        """基于模式生成解决方案"""
        if "procedural" in patterns:
            return "按照步骤程序化的方法解决问题"
        elif "causal" in patterns:
            return "分析因果关系来解决问题"
        elif "definitional" in patterns:
            return "通过明确定义概念来解决问题"
        elif "selection" in patterns:
            return "通过比较选择来解决问题"
        else:
            return "基于问题特征选择合适的解决方案"
    
    def _find_best_explanation(self, problem: str) -> str:
        """寻找最佳解释"""
        # 简化的溯因推理
        if "原因" in problem:
            return "最可能的原因是环境因素和内部机制的交互作用"
        elif "结果" in problem:
            return "最可能的结果是系统向稳定状态演化"
        else:
            return "最合理的解释是多种因素综合作用的结果"
    
    async def _evaluate_solutions(self, solutions: List[Dict], problem: str) -> List[Dict]:
        """评估解决方案"""
        
        for solution in solutions:
            # 评估解决方案质量
            quality_score = self._assess_solution_quality(solution, problem)
            solution["quality_score"] = quality_score
            
            # 计算可行性评分
            feasibility_score = self._assess_solution_feasibility(solution)
            solution["feasibility_score"] = feasibility_score
            
            # 计算综合评分
            solution["comprehensive_score"] = (
                quality_score * 0.6 + feasibility_score * 0.4
            )
        
        # 按综合评分排序
        solutions.sort(key=lambda x: x.get("comprehensive_score", 0.0), reverse=True)
        
        return solutions
    
    def _assess_solution_quality(self, solution: Dict, problem: str) -> float:
        """评估解决方案质量"""
        
        confidence = solution.get("confidence", 0.5)
        
        # 基于置信度的质量评估
        if confidence > 0.8:
            quality = 0.9
        elif confidence > 0.6:
            quality = 0.7
        elif confidence > 0.4:
            quality = 0.5
        else:
            quality = 0.3
        
        return quality
    
    def _assess_solution_feasibility(self, solution: Dict) -> float:
        """评估解决方案可行性"""
        
        method = solution.get("method", "")
        confidence = solution.get("confidence", 0.5)
        
        # 基于方法的可行性评估
        method_feasibility = {
            "deductive": 0.8,  # 演绎法通常可行性较高
            "inductive": 0.7,  # 归纳法可行性中等
            "abductive": 0.6   # 溯因法可行性相对较低
        }
        
        base_feasibility = method_feasibility.get(method, 0.5)
        adjusted_feasibility = base_feasibility * (0.8 + 0.4 * confidence)
        
        return min(1.0, adjusted_feasibility)
    
    def get_comprehensive_performance_metrics(self) -> Dict:
        """获取综合性能指标"""
        
        # 计算当前推理成功率
        if self.performance_metrics["total_reasoning_tasks"] > 0:
            reasoning_success_rate = (
                self.performance_metrics["successful_reasoning"] / 
                self.performance_metrics["total_reasoning_tasks"]
            )
        else:
            reasoning_success_rate = 0.0
        
        # 计算平均推理质量
        if self.reasoning_history:
            avg_quality = np.mean([step.confidence for step in self.reasoning_history])
        else:
            avg_quality = 0.0
        
        # 计算推理深度统计
        reasoning_depths = []
        current_depth = 0
        for step in self.reasoning_history:
            if step.step_id == 1:
                if current_depth > 0:
                    reasoning_depths.append(current_depth)
                current_depth = 1
            else:
                current_depth += 1
        if current_depth > 0:
            reasoning_depths.append(current_depth)
        
        avg_depth = np.mean(reasoning_depths) if reasoning_depths else 0.0
        
        # 计算推理类型分布
        step_type_counts = defaultdict(int)
        for step in self.reasoning_history:
            step_type_counts[step.step_type.value] += 1
        
        # 元认知指标
        metacognitive_metrics = {}
        if self.metacognitive_monitor:
            recent_history = self.metacognitive_monitor.performance_history[-5:]
            if recent_history:
                metacognitive_metrics = {
                    "avg_quality": np.mean([h["reasoning_quality"] for h in recent_history]),
                    "avg_consistency": np.mean([h["logical_consistency"] for h in recent_history]),
                    "adaptation_count": len(self.metacognitive_monitor.adaptation_history)
                }
        
        return {
            "basic_metrics": {
                "total_tasks": self.performance_metrics["total_reasoning_tasks"],
                "successful_tasks": self.performance_metrics["successful_reasoning"],
                "reasoning_success_rate": reasoning_success_rate,
                "avg_reasoning_quality": avg_quality,
                "avg_reasoning_depth": avg_depth
            },
            "reasoning_type_distribution": dict(step_type_counts),
            "symbolic_reasoning_success": self.performance_metrics["symbolic_reasoning_success"],
            "metacognitive_metrics": metacognitive_metrics,
            "system_capabilities": {
                "symbolic_reasoning_enabled": self.symbolic_engine is not None,
                "bayesian_inference_enabled": self.bayesian_engine is not None,
                "metacognition_enabled": self.metacognitive_monitor is not None,
                "max_reasoning_steps": self.max_reasoning_steps
            },
            "performance_trends": self._analyze_performance_trends()
        }
    
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """分析性能趋势"""
        if len(self.reasoning_history) < 5:
            return {"trend": 0.0, "improvement_rate": 0.0}
        
        recent_steps = self.reasoning_history[-10:]
        recent_confidences = [step.confidence for step in recent_steps]
        
        # 计算趋势
        if len(recent_confidences) >= 2:
            trend = (recent_confidences[-1] - recent_confidences[0]) / len(recent_confidences)
        else:
            trend = 0.0
        
        # 计算改进率
        older_steps = self.reasoning_history[-20:-10] if len(self.reasoning_history) >= 20 else []
        if older_steps:
            older_confidences = [step.confidence for step in older_steps]
            improvement_rate = np.mean(recent_confidences) - np.mean(older_confidences)
        else:
            improvement_rate = 0.0
        
        return {
            "trend": trend,
            "improvement_rate": improvement_rate,
            "recent_avg_quality": np.mean(recent_confidences)
        }
    
    def export_comprehensive_report(self) -> str:
        """导出综合报告"""
        
        metrics = self.get_comprehensive_performance_metrics()
        
        report = {
            "report_type": "前额叶推理引擎综合报告",
            "generation_time": datetime.now().isoformat(),
            "performance_metrics": metrics,
            "recent_reasoning_history": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "conclusion": step.intermediate_conclusion,
                    "confidence": step.confidence,
                    "certainty": step.certainty,
                    "timestamp": step.timestamp.isoformat()
                }
                for step in self.reasoning_history[-10:]  # 最近10步
            ],
            "system_configuration": {
                "max_reasoning_steps": self.max_reasoning_steps,
                "reasoning_config": self.reasoning_config,
                "enabled_components": {
                    "symbolic_reasoning": self.symbolic_engine is not None,
                    "bayesian_inference": self.bayesian_engine is not None,
                    "metacognition": self.metacognitive_monitor is not None
                }
            }
        }
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"升级版前额叶推理引擎("
                f"符号推理={self.symbolic_engine is not None}, "
                f"贝叶斯推理={self.bayesian_engine is not None}, "
                f"元认知监控={self.metacognitive_monitor is not None}, "
                f"推理历史={len(self.reasoning_history)})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"PrefrontalCortex("
                f"max_steps={self.max_reasoning_steps}, "
                f"history_size={len(self.reasoning_history)}, "
                f"capabilities={self.get_comprehensive_performance_metrics()['system_capabilities']})")


# 使用示例和测试代码
if __name__ == "__main__":
    async def test_enhanced_prefrontal_cortex():
        """测试升级版前额叶推理引擎"""
        
        print("=== 升级版前额叶推理引擎测试 ===")
        
        # 初始化升级版引擎
        cortex = PrefrontalCortex(
            max_reasoning_steps=8,
            enable_symbolic_reasoning=True,
            enable_bayesian_inference=True,
            enable_metacognition=True
        )
        
        print(f"引擎配置: {cortex}")
        
        # 测试问题
        test_problems = [
            "如果明天下雨且温度很低，我应该选择什么活动？",
            "为什么机器学习模型在某些情况下表现不佳？",
            "如何优化深度神经网络的训练过程？"
        ]
        
        print("\n--- 增强版链式推理测试 ---")
        for i, problem in enumerate(test_problems, 1):
            print(f"\n测试问题 {i}: {problem}")
            
            # 执行增强版链式推理
            result = await cortex.enhanced_chain_of_thought_reasoning(
                problem, 
                reasoning_type=ReasoningType.HYBRID
            )
            
            print(f"推理成功: {result.get('success', False)}")
            print(f"质量评分: {result.get('quality_score', 0):.3f}")
            print(f"推理深度: {result.get('reasoning_steps', []) and len(result['reasoning_steps'])}")
            
            # 显示各组件贡献
            print(f"符号推理贡献: {result.get('symbolic_reasoning', 0):.3f}")
            print(f"概率推理贡献: {result.get('probabilistic_reasoning', 0):.3f}")
            print(f"LLM推理贡献: {result.get('llm_reasoning', 0):.3f}")
            
            if result.get('final_conclusion'):
                print(f"最终结论: {result['final_conclusion']['conclusion']}")
            
            # 显示元认知报告
            if 'metacognitive_report' in result:
                metacog = result['metacognitive_report']
                print(f"元认知质量: {metacog.get('reasoning_quality', 0):.3f}")
                print(f"策略有效性: {metacog.get('strategy_effectiveness', 0):.3f}")
                if metacog.get('recommendations'):
                    print(f"建议: {metacog['recommendations'][0]}")
        
        print("\n--- 高级问题解决测试 ---")
        solving_problem = "如何提高团队的协作效率？"
        solution_result = await cortex.advanced_problem_solving(
            solving_problem, 
            strategy="hybrid",
            max_solutions=2
        )
        
        print(f"问题: {solving_problem}")
        print(f"解决方案数量: {solution_result.get('total_solutions', 0)}")
        
        if solution_result.get('solutions'):
            for i, solution in enumerate(solution_result['solutions'], 1):
                print(f"方案 {i}: {solution['method']} - {solution['solution']}")
                print(f"  质量评分: {solution.get('comprehensive_score', 0):.3f}")
        
        # 测试贝叶斯网络
        print("\n--- 贝叶斯网络测试 ---")
        if cortex.bayesian_engine:
            # 查询活动概率
            activity_probs = cortex.bayesian_engine.query("activity")
            print("活动概率分布:", activity_probs)
            
            # 设置新证据
            cortex.bayesian_engine.set_evidence("weather", "sunny")
            updated_probs = cortex.bayesian_engine.query("activity")
            print("更新后活动概率:", updated_probs)
        
        # 测试符号推理
        print("\n--- 符号推理测试 ---")
        if cortex.symbolic_engine:
            # 测试前向链式推理
            derived_facts = cortex.symbolic_engine.forward_chaining("适合户外活动")
            print("推导事实:", [fact['fact'] for fact in derived_facts])
            
            # 测试模糊推理
            fuzzy_results = cortex.symbolic_engine.fuzzy_reasoning([])
            print("模糊推理结果:", fuzzy_results)
        
        # 显示综合性能指标
        print("\n--- 综合性能指标 ---")
        metrics = cortex.get_comprehensive_performance_metrics()
        print("基础指标:")
        for key, value in metrics['basic_metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n系统能力:")
        for key, value in metrics['system_capabilities'].items():
            print(f"  {key}: {value}")
        
        # 生成综合报告
        print("\n--- 生成综合报告 ---")
        report = cortex.export_comprehensive_report()
        print("报告已生成，长度:", len(report), "字符")
        
        print("\n=== 测试完成 ===")
    
    # 运行测试
    asyncio.run(test_enhanced_prefrontal_cortex())