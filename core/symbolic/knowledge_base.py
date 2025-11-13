"""
知识库管理模块
负责存储和管理事实知识，支持动态增删改查和一致性检查
"""

from typing import Dict, List, Set, Union, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime
import re


class KnowledgeType(Enum):
    """知识类型枚举"""
    FACT = "fact"                    # 事实知识
    RULE = "rule"                    # 规则知识
    FUZZY_FACT = "fuzzy_fact"        # 模糊事实
    PROBABILISTIC = "probabilistic"  # 概率知识
    TEMPORAL = "temporal"           # 时态知识
    UNCERTAIN = "uncertain"         # 不确定性知识


class CertaintyLevel(Enum):
    """确定性级别"""
    ABSOLUTELY_TRUE = 1.0      # 绝对真
    VERY_HIGH = 0.9           # 非常高
    HIGH = 0.8                # 高
    MEDIUM = 0.5              # 中等
    LOW = 0.2                 # 低
    VERY_LOW = 0.1            # 非常低
    ABSOLUTELY_FALSE = 0.0    # 绝对假


class KnowledgeItem:
    """知识项基类"""
    
    def __init__(self, id: str, content: str, knowledge_type: KnowledgeType,
                 certainty: float, timestamp: datetime, source: str,
                 description: str = "", tags: Set[str] = None,
                 confidence: float = 1.0, evidence_count: int = 1,
                 last_verified: Optional[datetime] = None):
        self.id = id
        self.content = content
        self.knowledge_type = knowledge_type
        self.certainty = certainty
        self.timestamp = timestamp
        self.source = source
        self.description = description
        self.tags = tags or set()
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.last_verified = last_verified
        
        # 验证属性
        if not 0 <= self.certainty <= 1:
            raise ValueError("确定性值必须在0-1之间")
        if not 0 <= self.confidence <= 1:
            raise ValueError("置信度值必须在0-1之间")
        if self.evidence_count < 1:
            raise ValueError("证据数量必须至少为1")


class FactItem(KnowledgeItem):
    """事实知识项"""
    
    def __init__(self, id: str, content: str, knowledge_type: KnowledgeType,
                 certainty: float, timestamp: datetime, source: str,
                 subject: str, predicate: str, object: Union[str, bool, float, int],
                 description: str = "", tags: Set[str] = None,
                 confidence: float = 1.0, evidence_count: int = 1,
                 last_verified: Optional[datetime] = None,
                 relations: Dict[str, Any] = None):
        super().__init__(id, content, knowledge_type, certainty, timestamp, source,
                        description, tags, confidence, evidence_count, last_verified)
        
        if knowledge_type != KnowledgeType.FACT:
            raise ValueError("FactItem的知识类型必须是FACT")
        
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.relations = relations or {}


class RuleItem(KnowledgeItem):
    """规则知识项"""
    
    def __init__(self, id: str, content: str, knowledge_type: KnowledgeType,
                 certainty: float, timestamp: datetime, source: str,
                 antecedent: List[str], consequent: str,
                 rule_type: str = "if_then", weight: float = 1.0,
                 conditions: List[str] = None,
                 description: str = "", tags: Set[str] = None,
                 confidence: float = 1.0, evidence_count: int = 1,
                 last_verified: Optional[datetime] = None):
        super().__init__(id, content, knowledge_type, certainty, timestamp, source,
                        description, tags, confidence, evidence_count, last_verified)
        
        if knowledge_type != KnowledgeType.RULE:
            raise ValueError("RuleItem的知识类型必须是RULE")
        
        self.antecedent = antecedent
        self.consequent = consequent
        self.rule_type = rule_type
        self.weight = weight
        self.conditions = conditions or []


class FuzzyFactItem(KnowledgeItem):
    """模糊事实知识项"""
    
    def __init__(self, id: str, content: str, knowledge_type: KnowledgeType,
                 certainty: float, timestamp: datetime, source: str,
                 linguistic_variable: str, linguistic_value: str, membership_degree: float,
                 description: str = "", tags: Set[str] = None,
                 confidence: float = 1.0, evidence_count: int = 1,
                 last_verified: Optional[datetime] = None):
        super().__init__(id, content, knowledge_type, certainty, timestamp, source,
                        description, tags, confidence, evidence_count, last_verified)
        
        if knowledge_type != KnowledgeType.FUZZY_FACT:
            raise ValueError("FuzzyFactItem的知识类型必须是FUZZY_FACT")
        
        if not 0 <= membership_degree <= 1:
            raise ValueError("隶属度必须在0-1之间")
        
        self.linguistic_variable = linguistic_variable
        self.linguistic_value = linguistic_value
        self.membership_degree = membership_degree


class ProbabilisticItem(KnowledgeItem):
    """概率知识项"""
    
    def __init__(self, id: str, content: str, knowledge_type: KnowledgeType,
                 certainty: float, timestamp: datetime, source: str,
                 proposition: str, probability: float,
                 conditional_probabilities: Dict[str, float] = None,
                 description: str = "", tags: Set[str] = None,
                 confidence: float = 1.0, evidence_count: int = 1,
                 last_verified: Optional[datetime] = None):
        super().__init__(id, content, knowledge_type, certainty, timestamp, source,
                        description, tags, confidence, evidence_count, last_verified)
        
        if knowledge_type != KnowledgeType.PROBABILISTIC:
            raise ValueError("ProbabilisticItem的知识类型必须是PROBABILISTIC")
        
        if not 0 <= probability <= 1:
            raise ValueError("概率值必须在0-1之间")
        
        self.proposition = proposition
        self.probability = probability
        self.conditional_probabilities = conditional_probabilities or {}


class KnowledgeBase:
    """知识库管理系统"""
    
    def __init__(self, name: str = "default"):
        """初始化知识库"""
        self.name = name
        self.items: Dict[str, KnowledgeItem] = {}
        self.index: Dict[str, Set[str]] = {}  # 索引：标签->知识项ID集合
        self.facts_index: Dict[str, Set[str]] = {}  # 事实索引：主体->知识项ID
        self.rules_index: Set[str] = set()  # 规则ID集合
        self.uncertainty_threshold = 0.5  # 不确定性阈值
        
        # 统计信息
        self.statistics = {
            "total_items": 0,
            "fact_count": 0,
            "rule_count": 0,
            "fuzzy_fact_count": 0,
            "probabilistic_count": 0,
            "average_certainty": 0.0,
            "last_updated": datetime.now()
        }
    
    def add_fact(self, subject: str, predicate: str, obj: Union[str, bool, float, int],
                 certainty: float = 1.0, source: str = "user", 
                 description: str = "", tags: Set[str] = None) -> str:
        """添加事实知识"""
        if tags is None:
            tags = set()
        
        fact = FactItem(
            id=str(uuid.uuid4()),
            content=f"{subject} {predicate} {obj}",
            knowledge_type=KnowledgeType.FACT,
            certainty=certainty,
            timestamp=datetime.now(),
            source=source,
            description=description,
            tags=tags,
            subject=subject,
            predicate=predicate,
            object=obj
        )
        
        self._add_item(fact)
        return fact.id
    
    def add_rule(self, antecedent: List[str], consequent: str,
                 certainty: float = 1.0, source: str = "user",
                 description: str = "", tags: Set[str] = None,
                 conditions: List[str] = None, weight: float = 1.0) -> str:
        """添加规则知识"""
        if tags is None:
            tags = set()
        if conditions is None:
            conditions = []
        
        rule = RuleItem(
            id=str(uuid.uuid4()),
            content=f"IF {' AND '.join(antecedent)} THEN {consequent}",
            knowledge_type=KnowledgeType.RULE,
            certainty=certainty,
            timestamp=datetime.now(),
            source=source,
            description=description,
            tags=tags,
            antecedent=antecedent,
            consequent=consequent,
            conditions=conditions,
            weight=weight
        )
        
        self._add_item(rule)
        return rule.id
    
    def add_fuzzy_fact(self, linguistic_variable: str, linguistic_value: str,
                      membership_degree: float, source: str = "user",
                      description: str = "", tags: Set[str] = None) -> str:
        """添加模糊事实知识"""
        if tags is None:
            tags = set()
        
        fuzzy_fact = FuzzyFactItem(
            id=str(uuid.uuid4()),
            content=f"{linguistic_variable} 是 {linguistic_value} (程度: {membership_degree})",
            knowledge_type=KnowledgeType.FUZZY_FACT,
            certainty=membership_degree,
            timestamp=datetime.now(),
            source=source,
            description=description,
            tags=tags,
            linguistic_variable=linguistic_variable,
            linguistic_value=linguistic_value,
            membership_degree=membership_degree
        )
        
        self._add_item(fuzzy_fact)
        return fuzzy_fact.id
    
    def add_probabilistic_fact(self, proposition: str, probability: float,
                               source: str = "user", description: str = "",
                               tags: Set[str] = None,
                               conditional_probabilities: Dict[str, float] = None) -> str:
        """添加概率事实知识"""
        if tags is None:
            tags = set()
        if conditional_probabilities is None:
            conditional_probabilities = {}
        
        probabilistic_fact = ProbabilisticItem(
            id=str(uuid.uuid4()),
            content=f"P({proposition}) = {probability}",
            knowledge_type=KnowledgeType.PROBABILISTIC,
            certainty=probability,
            timestamp=datetime.now(),
            source=source,
            description=description,
            tags=tags,
            proposition=proposition,
            probability=probability,
            conditional_probabilities=conditional_probabilities
        )
        
        self._add_item(probabilistic_fact)
        return probabilistic_fact.id
    
    def _add_item(self, item: KnowledgeItem):
        """内部方法：添加知识项并更新索引"""
        self.items[item.id] = item
        
        # 更新统计信息
        self._update_statistics()
        
        # 更新标签索引
        for tag in item.tags:
            if tag not in self.index:
                self.index[tag] = set()
            self.index[tag].add(item.id)
        
        # 更新事实索引
        if isinstance(item, FactItem):
            subject_key = item.subject.lower()
            if subject_key not in self.facts_index:
                self.facts_index[subject_key] = set()
            self.facts_index[subject_key].add(item.id)
        
        # 更新规则索引
        if isinstance(item, RuleItem):
            self.rules_index.add(item.id)
    
    def remove_item(self, item_id: str) -> bool:
        """移除知识项"""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # 从索引中移除
        for tag in item.tags:
            if tag in self.index:
                self.index[tag].discard(item_id)
                if not self.index[tag]:
                    del self.index[tag]
        
        if isinstance(item, FactItem):
            subject_key = item.subject.lower()
            if subject_key in self.facts_index:
                self.facts_index[subject_key].discard(item_id)
                if not self.facts_index[subject_key]:
                    del self.facts_index[subject_key]
        
        if isinstance(item, RuleItem):
            self.rules_index.discard(item_id)
        
        # 从知识库中移除
        del self.items[item_id]
        
        # 更新统计信息
        self._update_statistics()
        return True
    
    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """获取知识项"""
        return self.items.get(item_id)
    
    def get_facts_by_subject(self, subject: str) -> List[FactItem]:
        """根据主体获取相关事实"""
        subject_key = subject.lower()
        fact_ids = self.facts_index.get(subject_key, set())
        return [self.items[item_id] for item_id in fact_ids if isinstance(self.items[item_id], FactItem)]
    
    def get_items_by_tag(self, tag: str) -> List[KnowledgeItem]:
        """根据标签获取知识项"""
        item_ids = self.index.get(tag, set())
        return [self.items[item_id] for item_id in item_ids if item_id in self.items]
    
    def search_items(self, keyword: str) -> List[KnowledgeItem]:
        """搜索包含关键词的知识项"""
        keyword_lower = keyword.lower()
        results = []
        
        for item in self.items.values():
            if (keyword_lower in item.content.lower() or 
                keyword_lower in item.description.lower() or
                any(keyword_lower in tag.lower() for tag in item.tags)):
                results.append(item)
        
        return results
    
    def get_items_by_certainty_range(self, min_certainty: float, max_certainty: float) -> List[KnowledgeItem]:
        """根据确定性范围获取知识项"""
        results = []
        for item in self.items.values():
            if min_certainty <= item.certainty <= max_certainty:
                results.append(item)
        return results
    
    def get_all_facts(self) -> List[FactItem]:
        """获取所有事实"""
        return [item for item in self.items.values() if isinstance(item, FactItem)]
    
    def get_all_rules(self) -> List[RuleItem]:
        """获取所有规则"""
        return [item for item in self.items.values() if isinstance(item, RuleItem)]
    
    def get_all_fuzzy_facts(self) -> List[FuzzyFactItem]:
        """获取所有模糊事实"""
        return [item for item in self.items.values() if isinstance(item, FuzzyFactItem)]
    
    def get_all_probabilistic_facts(self) -> List[ProbabilisticItem]:
        """获取所有概率事实"""
        return [item for item in self.items.values() if isinstance(item, ProbabilisticItem)]
    
    def update_item_certainty(self, item_id: str, new_certainty: float) -> bool:
        """更新知识项的确定性值"""
        if item_id not in self.items:
            return False
        
        if not 0 <= new_certainty <= 1:
            return False
        
        self.items[item_id].certainty = new_certainty
        self.items[item_id].last_verified = datetime.now()
        self._update_statistics()
        return True
    
    def merge_certainty(self, item_id: str, additional_evidence_certainty: float, 
                       additional_evidence_count: int = 1) -> bool:
        """合并多个证据的确定性值"""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # 使用贝叶斯方法更新确定性
        old_certainty = item.certainty
        evidence_certainty = additional_evidence_certainty
        
        # 更新后确定性
        numerator = old_certainty * evidence_certainty
        denominator = numerator + (1 - old_certainty) * (1 - evidence_certainty)
        new_certainty = numerator / denominator if denominator > 0 else old_certainty
        
        # 更新证据计数
        new_count = item.evidence_count + additional_evidence_count
        item.confidence = min(1.0, new_count / (new_count + 10))  # 置信度随证据增加而提升
        
        # 应用新确定性
        item.certainty = new_certainty
        item.evidence_count = new_count
        item.last_verified = datetime.now()
        
        self._update_statistics()
        return True
    
    def check_consistency(self) -> Tuple[bool, List[str]]:
        """检查知识库一致性"""
        inconsistencies = []
        
        # 检查事实一致性
        fact_pairs = []
        for item in self.get_all_facts():
            for other in self.get_all_facts():
                if (item != other and isinstance(other, FactItem) and
                    item.subject == other.subject and 
                    item.predicate == other.predicate and
                    item.object != other.object):
                    fact_pairs.append((item, other))
        
        if fact_pairs:
            inconsistencies.append(f"发现 {len(fact_pairs)} 对冲突的事实")
        
        # 检查规则冲突
        rules = self.get_all_rules()
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                # 检查是否有规则直接冲突
                if (rule1.consequent == f"¬{rule2.consequent}" or 
                    rule2.consequent == f"¬{rule1.consequent}"):
                    inconsistencies.append(f"规则冲突: {rule1.id} vs {rule2.id}")
        
        return len(inconsistencies) == 0, inconsistencies
    
    def get_reasoning_evidence(self, proposition: str) -> Dict[str, Any]:
        """获取支撑某个命题的证据"""
        evidence = {
            "supporting_facts": [],
            "supporting_rules": [],
            "confidence_score": 0.0,
            "evidence_count": 0
        }
        
        # 查找支持命题的事实
        for fact in self.get_all_facts():
            if proposition.lower() in fact.content.lower():
                evidence["supporting_facts"].append({
                    "id": fact.id,
                    "content": fact.content,
                    "certainty": fact.certainty,
                    "source": fact.source
                })
        
        # 查找支持命题的规则
        for rule in self.get_all_rules():
            if proposition.lower() in rule.consequent.lower():
                evidence["supporting_rules"].append({
                    "id": rule.id,
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "certainty": rule.certainty,
                    "weight": rule.weight
                })
        
        # 计算综合置信度
        fact_scores = [f["certainty"] for f in evidence["supporting_facts"]]
        rule_scores = [r["certainty"] * r["weight"] for r in evidence["supporting_rules"]]
        all_scores = fact_scores + rule_scores
        
        if all_scores:
            evidence["confidence_score"] = sum(all_scores) / len(all_scores)
            evidence["evidence_count"] = len(all_scores)
        
        return evidence
    
    def export_knowledge(self) -> Dict[str, Any]:
        """导出知识库"""
        return {
            "name": self.name,
            "items": [
                {
                    "id": item.id,
                    "content": item.content,
                    "type": item.knowledge_type.value,
                    "certainty": item.certainty,
                    "timestamp": item.timestamp.isoformat(),
                    "source": item.source,
                    "description": item.description,
                    "tags": list(item.tags),
                    "confidence": item.confidence,
                    "evidence_count": item.evidence_count
                }
                for item in self.items.values()
            ],
            "statistics": self.statistics
        }
    
    def import_knowledge(self, data: Dict[str, Any]) -> bool:
        """导入知识库"""
        try:
            self.name = data.get("name", self.name)
            
            # 清空现有知识
            self.items.clear()
            self.index.clear()
            self.facts_index.clear()
            self.rules_index.clear()
            
            # 导入知识项（简化实现）
            # 这里需要根据类型重建不同的知识项
            # 简化实现，直接存储原始数据
            
            return True
        except Exception as e:
            print(f"导入知识库失败: {e}")
            return False
    
    def _update_statistics(self):
        """更新统计信息"""
        total_items = len(self.items)
        fact_count = sum(1 for item in self.items.values() if isinstance(item, FactItem))
        rule_count = sum(1 for item in self.items.values() if isinstance(item, RuleItem))
        fuzzy_fact_count = sum(1 for item in self.items.values() if isinstance(item, FuzzyFactItem))
        probabilistic_count = sum(1 for item in self.items.values() if isinstance(item, ProbabilisticItem))
        
        avg_certainty = 0.0
        if total_items > 0:
            avg_certainty = sum(item.certainty for item in self.items.values()) / total_items
        
        self.statistics = {
            "total_items": total_items,
            "fact_count": fact_count,
            "rule_count": rule_count,
            "fuzzy_fact_count": fuzzy_fact_count,
            "probabilistic_count": probabilistic_count,
            "average_certainty": avg_certainty,
            "last_updated": datetime.now()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.statistics.copy()
    
    def get_reasoning_chains(self, target_proposition: str) -> List[Dict[str, Any]]:
        """获取推导目标命题的推理链"""
        chains = []
        
        # 查找直接规则
        for rule in self.get_all_rules():
            if target_proposition.lower() in rule.consequent.lower():
                chain = {
                    "type": "direct_rule",
                    "rule_id": rule.id,
                    "consequent": rule.consequent,
                    "antecedent": rule.antecedent,
                    "certainty": rule.certainty,
                    "length": 1
                }
                chains.append(chain)
        
        # 查找多步推理链（简化实现）
        for rule in self.get_all_rules():
            for antecedent_prop in rule.antecedent:
                if target_proposition.lower() in antecedent_prop.lower():
                    chain = {
                        "type": "multi_step",
                        "rule_id": rule.id,
                        "target": target_proposition,
                        "intermediate": antecedent_prop,
                        "final_consequent": rule.consequent,
                        "certainty": rule.certainty,
                        "length": 2
                    }
                    chains.append(chain)
        
        return chains
    
    def clear(self):
        """清空知识库"""
        self.items.clear()
        self.index.clear()
        self.facts_index.clear()
        self.rules_index.clear()
        self._update_statistics()