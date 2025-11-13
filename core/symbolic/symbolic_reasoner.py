"""
符号逻辑推理引擎主类 - Symbolic Reasoning Engine Main Class
符号逻辑推理引擎的顶层接口和集成组件

功能特性：
- 集成知识库、规则库、推理引擎、逻辑解析器
- 支持多种推理模式（前向、后向、模糊、概率等）
- LLM集成作为高级推理后端
- 实时性能监控和优化
- 推理结果解释和可视化
- 多层次推理路径管理
- 动态知识更新和验证

Author: AI Assistant
Date: 2025-11-13
"""

import time
import json
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from abc import ABC, abstractmethod

# 导入组件
from .knowledge_base import KnowledgeBase, KnowledgeType, CertaintyLevel
from .rule_base import RuleBase, RuleType
from .inference_engine import InferenceEngine, InferenceType
from .logic_parser import LogicParser, LogicType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """推理模式枚举"""
    AUTOMATIC = "automatic"        # 自动模式
    FORWARD_ONLY = "forward_only"  # 仅前向推理
    BACKWARD_ONLY = "backward_only"  # 仅后向推理
    HYBRID = "hybrid"             # 混合推理
    FUZZY_ONLY = "fuzzy_only"     # 仅模糊推理
    PROBABILISTIC_ONLY = "probabilistic_only"  # 仅概率推理
    LLM_ASSISTED = "llm_assisted"  # LLM辅助推理


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ReasoningSession:
    """推理会话"""
    session_id: str
    start_time: datetime
    mode: ReasoningMode
    initial_facts: List[str]
    target_goals: Optional[List[str]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    end_time: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """检查会话是否活跃"""
        return self.status == "active"
    
    def is_completed(self) -> bool:
        """检查会话是否完成"""
        return self.status in ["completed", "failed", "cancelled"]
    
    def add_result(self, result: Dict[str, Any]):
        """添加推理结果"""
        result["timestamp"] = datetime.now().isoformat()
        self.results.append(result)
    
    def complete(self, status: str = "completed"):
        """完成会话"""
        self.status = status
        self.end_time = datetime.now()


@dataclass
class ReasoningConfig:
    """推理配置"""
    mode: ReasoningMode = ReasoningMode.AUTOMATIC
    max_iterations: int = 100
    max_depth: int = 10
    timeout: float = 30.0
    certainty_threshold: float = 0.1
    confidence_threshold: float = 0.1
    enable_caching: bool = True
    enable_optimization: bool = True
    enable_explanation: bool = True
    llm_enabled: bool = False
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-3.5-turbo"
    parallel_reasoning: bool = False
    max_reasoning_paths: int = 10
    fuzzy_threshold: float = 0.5
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []
        
        if self.max_iterations <= 0:
            errors.append("最大迭代次数必须大于0")
        if self.max_depth <= 0:
            errors.append("最大深度必须大于0")
        if self.timeout <= 0:
            errors.append("超时时间必须大于0")
        if not 0.0 <= self.certainty_threshold <= 1.0:
            errors.append("确信度阈值必须在[0,1]范围内")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("置信度阈值必须在[0,1]范围内")
        if not 0.0 <= self.fuzzy_threshold <= 1.0:
            errors.append("模糊阈值必须在[0,1]范围内")
        
        return len(errors) == 0, errors


class LLMCoreInterface(ABC):
    """LLM核心接口"""
    
    @abstractmethod
    async def generate_reasoning_step(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成推理步骤"""
        pass
    
    @abstractmethod
    async def validate_reasoning(self, reasoning_chain: List[str]) -> Tuple[bool, float]:
        """验证推理链"""
        pass
    
    @abstractmethod
    async def explain_inference(self, inference_step: Dict[str, Any]) -> str:
        """解释推理过程"""
        pass


class OpenAIReasoningAdapter(LLMCoreInterface):
    """OpenAI推理适配器"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
    
    async def generate_reasoning_step(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成推理步骤"""
        # 这里实现OpenAI API调用逻辑
        # 简化返回
        return f"基于提示 '{prompt}' 的推理结果"
    
    async def validate_reasoning(self, reasoning_chain: List[str]) -> Tuple[bool, float]:
        """验证推理链"""
        # 这里实现推理验证逻辑
        # 简化返回
        return True, 0.8
    
    async def explain_inference(self, inference_step: Dict[str, Any]) -> str:
        """解释推理过程"""
        # 这里实现推理解释逻辑
        # 简化返回
        return "这个推理步骤基于逻辑规则和应用条件"


class SymbolicReasoner:
    """
    符号逻辑推理引擎主类
    
    集成了知识库、规则库、推理引擎和逻辑解析器的完整推理系统
    支持多种推理模式，包括传统逻辑推理和现代AI推理技术
    """
    
    def __init__(self, name: str = "symbolic_reasoner", config: Optional[ReasoningConfig] = None):
        """初始化符号逻辑推理引擎"""
        self.name = name
        self.config = config or ReasoningConfig()
        
        # 验证配置
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"推理配置无效: {errors}")
        
        # 初始化核心组件
        self.knowledge_base = KnowledgeBase(f"{name}_kb")
        self.rule_base = RuleBase(f"{name}_rb")
        self.inference_engine = InferenceEngine(f"{name}_engine")
        self.logic_parser = LogicParser()
        
        # LLM集成
        self.llm_enabled = self.config.llm_enabled
        if self.llm_enabled:
            self.llm_adapter = self._create_llm_adapter()
        else:
            self.llm_adapter = None
        
        # 推理会话管理
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.session_history: List[ReasoningSession] = []
        
        # 性能监控
        self.performance_metrics = {
            "total_reasoning_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0,
            "average_session_time": 0.0,
            "cache_hit_rate": 0.0,
            "knowledge_items": 0,
            "rules_count": 0,
            "inferences_performed": 0
        }
        
        # 推理统计
        self.reasoning_statistics = {
            "forward_reasoning_count": 0,
            "backward_reasoning_count": 0,
            "fuzzy_reasoning_count": 0,
            "probabilistic_reasoning_count": 0,
            "llm_assisted_count": 0,
            "reasoning_paths_found": 0
        }
        
        # 添加缺失的属性
        self.statistics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "active_sessions": 0,
            "total_sessions": 0,
            "last_updated": datetime.now()
        }
        
        self.logger = logger
        self.max_workers = 4
        self.llm_integration = {"enabled": False}
        self.sessions = {}
        self.current_session = None
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"符号逻辑推理引擎 '{self.name}' 初始化完成")
    
    def reason(self, query: Union[str, Dict[str, Any]], 
              mode: Optional[ReasoningMode] = None,
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行符号逻辑推理
        
        Args:
            query: 推理查询（字符串或字典）
            mode: 推理模式
            context: 推理上下文
            
        Returns:
            Dict[str, Any]: 推理结果
        """
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 创建推理会话
            session = ReasoningSession(
                session_id=session_id,
                start_time=datetime.now(),
                mode=mode or self.config.mode,
                initial_facts=[],
                context=context or {},
                metadata={"query": query, "config": self.config.__dict__}
            )
            
            self.active_sessions[session_id] = session
            
            # 解析查询
            parsed_query = self._parse_query(query)
            session.initial_facts = parsed_query.get("facts", [])
            session.target_goals = parsed_query.get("goals", [])
            
            # 选择推理策略
            reasoning_mode = self._select_reasoning_strategy(parsed_query, session.mode)
            
            # 执行推理
            result = self._execute_reasoning(parsed_query, reasoning_mode, session)
            
            # 后处理结果
            final_result = self._post_process_result(result, session)
            
            # 完成会话
            session.complete("completed")
            session.add_result(final_result)
            
            # 更新统计信息
            self._update_statistics(True, time.time() - start_time, reasoning_mode)
            
            self.session_history.append(session)
            if len(self.session_history) > 100:  # 限制历史记录数量
                self.session_history.pop(0)
            
            logger.info(f"推理完成，会话ID: {session_id}, 模式: {reasoning_mode}")
            return final_result
            
        except Exception as e:
            logger.error(f"推理失败，会话ID: {session_id}, 错误: {str(e)}")
            
            # 失败处理
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.complete("failed")
                session.add_result({"error": str(e)})
                self.session_history.append(session)
            
            self._update_statistics(False, time.time() - start_time, reasoning_mode if 'reasoning_mode' in locals() else ReasoningMode.AUTOMATIC)
            
            raise
    
    def parse_logic_expression(self, expression: str, logic_type: str = "propositional") -> Any:
        """解析逻辑表达式"""
        try:
            from .logic_parser import LogicType
            logic_enum = LogicType(logic_type.lower())
            ast = self.logic_parser.parse_logic_expression(expression, logic_enum)
            return {
                "success": True,
                "ast": ast,
                "formatted": self.logic_parser.format_expression(ast),
                "simplified": self.logic_parser.format_expression(
                    self.logic_parser.simplify_expression(ast)
                )
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def forward_chain(self, query: str, facts: List[str] = None, 
                     reasoning_path: bool = True) -> Dict[str, Any]:
        """前向链式推理"""
        if facts is None:
            facts = []
        
        try:
            # 获取知识库中的事实
            kb_facts = [fact.content for fact in self.knowledge_base.get_all_facts()]
            all_facts = facts + kb_facts
            
            # 执行推理
            result = self.inference_engine.forward_chain(
                facts=all_facts,
                knowledge_base=self.knowledge_base,
                rule_base=self.rule_base,
                target_goals=[query] if query else None
            )
            
            # 更新统计信息
            self._update_query_statistics(True, result.execution_time)
            
            response = {
                "success": result.success,
                "query": query,
                "result": result.conclusion,
                "execution_time": result.execution_time,
                "reasoning_steps": len(result.reasoning_steps),
                "new_facts_generated": result.meta_data.get("new_facts_count", 0)
            }
            
            if reasoning_path:
                response["reasoning_paths"] = [
                    {
                        "path_id": path.path_id,
                        "goal": path.goal,
                        "steps": [
                            {
                                "step_id": step.step_id,
                                "premise": step.premise,
                                "conclusion": step.conclusion,
                                "certainty": step.certainty,
                                "confidence": step.confidence,
                                "timestamp": step.timestamp.isoformat()
                            }
                            for step in path.steps
                        ],
                        "total_certainty": path.total_certainty,
                        "total_confidence": path.total_confidence,
                        "length": path.length
                    }
                    for path in result.reasoning_paths
                ]
            
            return response
            
        except Exception as e:
            self._update_query_statistics(False, 0)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "traceback": traceback.format_exc()
            }
    
    def backward_chain(self, goal: str, facts: List[str] = None,
                      reasoning_path: bool = True) -> Dict[str, Any]:
        """后向链式推理"""
        if facts is None:
            facts = []
        
        try:
            # 获取知识库中的事实
            kb_facts = [fact.content for fact in self.knowledge_base.get_all_facts()]
            all_facts = facts + kb_facts
            
            # 执行推理
            result = self.inference_engine.backward_chain(
                goal=goal,
                facts=all_facts,
                knowledge_base=self.knowledge_base,
                rule_base=self.rule_base
            )
            
            # 更新统计信息
            self._update_query_statistics(True, result.execution_time)
            
            response = {
                "success": result.success,
                "goal": goal,
                "result": result.conclusion,
                "execution_time": result.execution_time,
                "reasoning_steps": len(result.reasoning_steps),
                "proof_found": result.success
            }
            
            if reasoning_path:
                response["reasoning_paths"] = [
                    {
                        "path_id": path.path_id,
                        "goal": path.goal,
                        "steps": [
                            {
                                "step_id": step.step_id,
                                "premise": step.premise,
                                "conclusion": step.conclusion,
                                "certainty": step.certainty,
                                "confidence": step.confidence,
                                "timestamp": step.timestamp.isoformat()
                            }
                            for step in path.steps
                        ],
                        "total_certainty": path.total_certainty,
                        "total_confidence": path.total_confidence,
                        "length": path.length
                    }
                    for path in result.reasoning_paths
                ]
            
            return response
            
        except Exception as e:
            self._update_query_statistics(False, 0)
            return {
                "success": False,
                "error": str(e),
                "goal": goal,
                "traceback": traceback.format_exc()
            }
    
    def fuzzy_reasoning(self, fuzzy_facts: List[Dict[str, Any]], 
                       fuzzy_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """模糊逻辑推理"""
        if fuzzy_rules is None:
            # 从规则库获取模糊规则
            fuzzy_rules = []
            for rule in self.rule_base.get_active_rules():
                fuzzy_rules.append({
                    "id": rule.id,
                    "antecedent": [cond.pattern for cond in rule.conditions],
                    "consequent": rule.conclusion,
                    "strength": rule.certainty
                })
        
        try:
            result = self.inference_engine.fuzzy_reasoning(fuzzy_facts, fuzzy_rules)
            
            self._update_query_statistics(True, result.execution_time)
            
            return {
                "success": result.success,
                "result": result.conclusion,
                "execution_time": result.execution_time,
                "fuzzy_facts_count": len(fuzzy_facts),
                "new_fuzzy_facts": [
                    {
                        "variable": path.goal,
                        "membership_degree": path.total_certainty
                    }
                    for path in result.reasoning_paths
                ],
                "reasoning_steps": len(result.reasoning_steps)
            }
            
        except Exception as e:
            self._update_query_statistics(False, 0)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def uncertain_reasoning(self, uncertain_facts: List[Dict[str, Any]],
                           uncertain_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """不确定性推理"""
        if uncertain_rules is None:
            # 从规则库获取不确定性规则
            uncertain_rules = []
            for rule in self.rule_base.get_active_rules():
                uncertain_rules.append({
                    "id": rule.id,
                    "antecedent": [cond.pattern for cond in rule.conditions],
                    "consequent": rule.conclusion,
                    "certainty": rule.certainty
                })
        
        try:
            result = self.inference_engine.uncertain_reasoning(uncertain_facts, uncertain_rules)
            
            self._update_query_statistics(True, result.execution_time)
            
            return {
                "success": result.success,
                "result": result.conclusion,
                "execution_time": result.execution_time,
                "uncertain_facts_count": len(uncertain_facts),
                "new_uncertain_facts": [
                    {
                        "proposition": path.goal,
                        "certainty": path.total_certainty
                    }
                    for path in result.reasoning_paths
                ],
                "reasoning_steps": len(result.reasoning_steps)
            }
            
        except Exception as e:
            self._update_query_statistics(False, 0)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def bidirectional_reasoning(self, goal: str, facts: List[str] = None) -> Dict[str, Any]:
        """双向推理"""
        if facts is None:
            facts = []
        
        try:
            # 获取知识库中的事实
            kb_facts = [fact.content for fact in self.knowledge_base.get_all_facts()]
            all_facts = facts + kb_facts
            
            result = self.inference_engine.bidirectional_reasoning(
                goal=goal,
                facts=all_facts,
                knowledge_base=self.knowledge_base,
                rule_base=self.rule_base
            )
            
            self._update_query_statistics(True, result.execution_time)
            
            return {
                "success": result.success,
                "goal": goal,
                "result": result.conclusion,
                "execution_time": result.execution_time,
                "forward_paths": result.meta_data.get("forward_paths", 0),
                "backward_paths": result.meta_data.get("backward_paths", 0),
                "total_reasoning_paths": len(result.reasoning_paths),
                "reasoning_steps": len(result.reasoning_steps)
            }
            
        except Exception as e:
            self._update_query_statistics(False, 0)
            return {
                "success": False,
                "error": str(e),
                "goal": goal,
                "traceback": traceback.format_exc()
            }
    
    def reason(self, query: str, reasoning_type: str = "forward_chain",
              facts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """统一推理接口"""
        self.logger.info(f"开始推理查询: {query}, 类型: {reasoning_type}")
        
        start_time = time.time()
        
        # 根据推理类型选择相应方法
        if reasoning_type == "forward_chain":
            result = self.forward_chain(query, facts, **kwargs)
        elif reasoning_type == "backward_chain":
            result = self.backward_chain(query, facts, **kwargs)
        elif reasoning_type == "bidirectional":
            result = self.bidirectional_reasoning(query, facts)
        elif reasoning_type == "fuzzy":
            fuzzy_facts = kwargs.get("fuzzy_facts", [])
            result = self.fuzzy_reasoning(fuzzy_facts)
        elif reasoning_type == "uncertain":
            uncertain_facts = kwargs.get("uncertain_facts", [])
            result = self.uncertain_reasoning(uncertain_facts)
        else:
            return {
                "success": False,
                "error": f"不支持的推理类型: {reasoning_type}",
                "query": query
            }
        
        result["total_time"] = time.time() - start_time
        result["reasoning_type"] = reasoning_type
        
        return result
    
    def add_knowledge(self, knowledge_type: str, data: Dict[str, Any]) -> bool:
        """添加知识"""
        try:
            if knowledge_type == "fact":
                return self.knowledge_base.add_fact(
                    subject=data["subject"],
                    predicate=data["predicate"],
                    obj=data["object"],
                    certainty=data.get("certainty", 1.0),
                    source=data.get("source", "user"),
                    description=data.get("description", ""),
                    tags=data.get("tags", set())
                )
            
            elif knowledge_type == "rule":
                return self.rule_base.add_rule(
                    name=data["name"],
                    rule_type=data["rule_type"],
                    conditions=data["conditions"],
                    conclusion=data["conclusion"],
                    certainty=data.get("certainty", 1.0),
                    priority=data.get("priority", 0),
                    weight=data.get("weight", 1.0),
                    source=data.get("source", "user"),
                    description=data.get("description", ""),
                    tags=data.get("tags", set())
                )
            
            elif knowledge_type == "fuzzy_fact":
                return self.knowledge_base.add_fuzzy_fact(
                    linguistic_variable=data["variable"],
                    linguistic_value=data["value"],
                    membership_degree=data["membership_degree"],
                    source=data.get("source", "user"),
                    description=data.get("description", ""),
                    tags=data.get("tags", set())
                )
            
            elif knowledge_type == "probabilistic_fact":
                return self.knowledge_base.add_probabilistic_fact(
                    proposition=data["proposition"],
                    probability=data["probability"],
                    source=data.get("source", "user"),
                    description=data.get("description", ""),
                    tags=data.get("tags", set())
                )
            
            else:
                self.logger.error(f"不支持的知识类型: {knowledge_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"添加知识失败: {e}")
            return False
    
    def create_session(self, user_id: str, preferences: Dict[str, Any] = None) -> str:
        """创建推理会话"""
        session_id = str(uuid.uuid4())
        session = ReasoningSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            end_time=None,
            queries=[],
            results={},
            knowledge_context={},
            preferences=preferences or {}
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        self.statistics["active_sessions"] += 1
        self.statistics["total_sessions"] += 1
        
        return session_id
    
    def end_session(self, session_id: str) -> bool:
        """结束推理会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.end_time = datetime.now()
            
            if session == self.current_session:
                self.current_session = None
            
            self.statistics["active_sessions"] -= 1
            return True
        
        return False
    
    def batch_reason(self, queries: List[Dict[str, Any]], 
                    max_concurrent: int = None) -> List[Dict[str, Any]]:
        """批量推理"""
        if max_concurrent is None:
            max_concurrent = min(len(queries), self.max_workers)
        
        results = []
        
        # 使用线程池并行执行推理任务
        if self.config["enable_parallel_reasoning"]:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {}
                
                for i, query_data in enumerate(queries):
                    future = executor.submit(
                        self.reason,
                        query_data["query"],
                        query_data.get("type", "forward_chain"),
                        query_data.get("facts", []),
                        **query_data.get("kwargs", {})
                    )
                    futures[future] = i
                
                for future in futures:
                    try:
                        result = future.result(timeout=query_data.get("timeout", 30.0))
                        result["batch_index"] = futures[future]
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "batch_index": futures[future]
                        })
        else:
            # 顺序执行
            for i, query_data in enumerate(queries):
                try:
                    result = self.reason(
                        query_data["query"],
                        query_data.get("type", "forward_chain"),
                        query_data.get("facts", []),
                        **query_data.get("kwargs", {})
                    )
                    result["batch_index"] = i
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "batch_index": i
                    })
        
        # 按原始顺序排序结果
        results.sort(key=lambda r: r["batch_index"])
        
        return results
    
    def explain_reasoning(self, result: Dict[str, Any]) -> str:
        """解释推理过程"""
        if not result.get("success"):
            return f"推理失败: {result.get('error', '未知错误')}"
        
        explanation_parts = []
        
        # 推理类型
        explanation_parts.append(f"推理类型: {result.get('reasoning_type', 'unknown')}")
        
        # 推理步骤
        reasoning_steps = result.get("reasoning_steps", 0)
        explanation_parts.append(f"推理步骤数: {reasoning_steps}")
        
        # 执行时间
        execution_time = result.get("execution_time", 0)
        explanation_parts.append(f"执行时间: {execution_time:.3f}秒")
        
        # 推理路径
        if "reasoning_paths" in result:
            path_count = len(result["reasoning_paths"])
            explanation_parts.append(f"推理路径数: {path_count}")
            
            # 如果有推理路径，显示最佳路径
            if path_count > 0:
                best_path = max(result["reasoning_paths"], 
                              key=lambda p: p.get("total_certainty", 0) * p.get("total_confidence", 0))
                explanation_parts.append(f"最佳路径确定性: {best_path.get('total_certainty', 0):.3f}")
                explanation_parts.append(f"最佳路径置信度: {best_path.get('total_confidence', 0):.3f}")
        
        return "\n".join(explanation_parts)
    
    def optimize_performance(self) -> Dict[str, Any]:
        """优化推理性能"""
        optimization_results = {}
        
        # 优化规则库
        if self.config["auto_optimize"]:
            removed_rules = self.rule_base.optimize_rules()
            optimization_results["rules_optimized"] = removed_rules
        
        # 清空缓存
        if self.config["cache_results"]:
            old_cache_size = len(self.inference_engine.reasoning_cache)
            self.inference_engine.clear_cache()
            optimization_results["cache_cleared"] = old_cache_size
        
        # 获取性能指标
        perf_metrics = self.inference_engine.get_performance_metrics()
        optimization_results["performance_metrics"] = perf_metrics
        
        return optimization_results
    
    def integrate_llm(self, provider: str, api_key: str, model_name: str = None) -> bool:
        """集成LLM作为符号引擎后端"""
        try:
            self.llm_integration.update({
                "enabled": True,
                "provider": provider,
                "api_key": api_key,
                "model_name": model_name
            })
            
            self.logger.info(f"LLM集成成功: {provider}")
            return True
        
        except Exception as e:
            self.logger.error(f"LLM集成失败: {e}")
            return False
    
    def llm_assisted_reasoning(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """LLM辅助推理"""
        if not self.llm_integration["enabled"]:
            return {
                "success": False,
                "error": "LLM未集成或未启用"
            }
        
        try:
            # 这里应该调用具体的LLM API
            # 简化实现，返回模拟结果
            return {
                "success": True,
                "query": query,
                "llm_response": "模拟的LLM响应",
                "confidence": 0.8,
                "reasoning_suggestion": "建议的推理策略"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """导出知识库"""
        return {
            "knowledge_base": self.knowledge_base.export_knowledge(),
            "rule_base": self.rule_base.export_rules(),
            "config": self.config,
            "statistics": self.statistics,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_knowledge_base(self, data: Dict[str, Any]) -> bool:
        """导入知识库"""
        try:
            # 导入知识库
            if "knowledge_base" in data:
                self.knowledge_base.import_knowledge(data["knowledge_base"])
            
            # 导入规则库
            if "rule_base" in data:
                self.rule_base.import_rules(data["rule_base"])
            
            # 导入配置
            if "config" in data:
                self.config.update(data["config"])
            
            self.logger.info("知识库导入成功")
            return True
        
        except Exception as e:
            self.logger.error(f"知识库导入失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 更新知识库和规则库统计
        kb_stats = self.knowledge_base.get_statistics()
        rb_stats = self.rule_base.get_rule_statistics()
        perf_stats = self.inference_engine.get_performance_metrics()
        
        return {
            "reasoner": self.statistics,
            "knowledge_base": kb_stats,
            "rule_base": rb_stats,
            "performance": perf_stats,
            "config": self.config,
            "llm_integration": self.llm_integration
        }
    
    def _update_query_statistics(self, success: bool, execution_time: float):
        """更新查询统计信息"""
        self.statistics["total_queries"] += 1
        
        if success:
            self.statistics["successful_queries"] += 1
        else:
            self.statistics["failed_queries"] += 1
        
        # 更新平均响应时间
        total_queries = self.statistics["total_queries"]
        current_avg = self.statistics["average_response_time"]
        self.statistics["average_response_time"] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        self.statistics["last_updated"] = datetime.now()
    
    def shutdown(self):
        """关闭推理引擎"""
        self.logger.info("正在关闭符号推理引擎...")
        
        # 结束当前会话
        if self.current_session:
            self.end_session(self.current_session.session_id)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 清理缓存
        self.inference_engine.clear_cache()
        
        self.logger.info("符号推理引擎已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()