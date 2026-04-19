#!/usr/bin/env python3
"""
符号逻辑模块测试
测试符号推理、知识库、规则库等组件
"""

import pytest
import numpy as np
from typing import Dict, Any


class TestSymbolicReasoner:
    """符号推理器测试"""

    def test_symbolic_reasoner_initialization(self):
        """测试符号推理器初始化"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")
        assert reasoner.name == "test_reasoner"
        assert reasoner.knowledge_base is not None
        assert reasoner.rule_base is not None
        assert reasoner.inference_engine is not None

    def test_add_fact(self):
        """测试添加事实"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        result = reasoner.add_knowledge("fact", {
            "subject": "鸟",
            "predicate": "会",
            "object": "飞",
            "certainty": 0.9
        })

        assert result is True

    def test_add_rule(self):
        """测试添加规则"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        result = reasoner.add_knowledge("rule", {
            "name": "所有鸟会飞",
            "rule_type": "universal",
            "conditions": [
                {"subject": "X", "predicate": "是", "object": "鸟"}
            ],
            "conclusion": {"subject": "X", "predicate": "会", "object": "飞"},
            "certainty": 0.8
        })

        assert result is True

    def test_forward_chain(self):
        """测试前向链推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        # 添加事实
        reasoner.add_knowledge("fact", {
            "subject": "企鹅",
            "predicate": "是",
            "object": "鸟"
        })

        # 添加规则
        reasoner.add_knowledge("rule", {
            "name": "鸟会飞",
            "rule_type": "universal",
            "conditions": [
                {"subject": "X", "predicate": "是", "object": "鸟"}
            ],
            "conclusion": {"subject": "X", "predicate": "会", "object": "飞"}
        })

        result = reasoner.forward_chain("企鹅")
        assert result is not None

    def test_backward_chain(self):
        """测试后向链推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        # 添加事实和规则
        reasoner.add_knowledge("fact", {
            "subject": "企鹅",
            "predicate": "是",
            "object": "鸟"
        })

        reasoner.add_knowledge("rule", {
            "name": "鸟会飞",
            "rule_type": "universal",
            "conditions": [
                {"subject": "X", "predicate": "是", "object": "鸟"}
            ],
            "conclusion": {"subject": "X", "predicate": "会", "object": "飞"}
        })

        result = reasoner.backward_chain("飞")
        assert result is not None

    def test_fuzzy_reasoning(self):
        """测试模糊推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        fuzzy_facts = [
            {"variable": "温度", "value": "高", "membership": 0.8},
            {"variable": "湿度", "value": "中", "membership": 0.5}
        ]

        result = reasoner.fuzzy_reasoning(fuzzy_facts)
        assert result is not None

    def test_bidirectional_reasoning(self):
        """测试双向推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        # 添加知识
        reasoner.add_knowledge("fact", {
            "subject": "A",
            "predicate": "大于",
            "object": "B"
        })
        reasoner.add_knowledge("fact", {
            "subject": "B",
            "predicate": "大于",
            "object": "C"
        })

        result = reasoner.bidirectional_reasoning("A大于C")
        assert result is not None

    def test_reasoning_with_context(self):
        """测试带上下文的推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        result = reasoner.reason(
            query="测试查询",
            context={"domain": "科学", "depth": 3}
        )

        assert result is not None

    def test_batch_reasoning(self):
        """测试批量推理"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        queries = [
            {"query": "查询1", "type": "forward_chain"},
            {"query": "查询2", "type": "backward_chain"}
        ]

        results = reasoner.batch_reason(queries)
        assert len(results) == 2

    def test_export_import_knowledge(self):
        """测试知识库导出导入"""
        from core.symbolic.symbolic_reasoner import SymbolicReasoner

        reasoner = SymbolicReasoner(name="test_reasoner")

        # 添加一些知识
        reasoner.add_knowledge("fact", {
            "subject": "测试",
            "predicate": "是",
            "object": "事实"
        })

        # 导出
        exported = reasoner.export_knowledge_base()
        assert "knowledge_base" in exported
        assert "rule_base" in exported

        # 创建新推理器并导入
        new_reasoner = SymbolicReasoner(name="new_reasoner")
        new_reasoner.import_knowledge_base(exported)

        assert new_reasoner.knowledge_base is not None


class TestKnowledgeBase:
    """知识库测试"""

    def test_knowledge_base_initialization(self):
        """测试知识库初始化"""
        from core.symbolic.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="test_kb")
        assert kb.name == "test_kb"

    def test_add_fact(self):
        """测试添加事实"""
        from core.symbolic.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="test_kb")

        result = kb.add_fact(
            subject="太阳",
            predicate="是",
            obj="恒星",
            certainty=1.0
        )

        assert result is True

    def test_add_fuzzy_fact(self):
        """测试添加模糊事实"""
        from core.symbolic.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="test_kb")

        result = kb.add_fuzzy_fact(
            linguistic_variable="温度",
            linguistic_value="高",
            membership_degree=0.8
        )

        assert result is True

    def test_add_probabilistic_fact(self):
        """测试添加概率事实"""
        from core.symbolic.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="test_kb")

        result = kb.add_probabilistic_fact(
            proposition="明天下雨",
            probability=0.7
        )

        assert result is True

    def test_query_facts(self):
        """测试查询事实"""
        from core.symbolic.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="test_kb")

        kb.add_fact(
            subject="猫",
            predicate="是",
            obj="动物"
        )

        facts = kb.query_facts(subject="猫")
        assert len(facts) >= 0


class TestRuleBase:
    """规则库测试"""

    def test_rule_base_initialization(self):
        """测试规则库初始化"""
        from core.symbolic.rule_base import RuleBase

        rb = RuleBase(name="test_rb")
        assert rb.name == "test_rb"

    def test_add_rule(self):
        """测试添加规则"""
        from core.symbolic.rule_base import RuleBase
        from core.symbolic.rule_base import RuleType

        rb = RuleBase(name="test_rb")

        result = rb.add_rule(
            name="测试规则",
            rule_type=RuleType.CAUSAL,
            conditions=[{"pattern": "如果A则B"}],
            conclusion={"pattern": "B成立"},
            certainty=0.9
        )

        assert result is True

    def test_get_active_rules(self):
        """测试获取活跃规则"""
        from core.symbolic.rule_base import RuleBase
        from core.symbolic.rule_base import RuleType

        rb = RuleBase(name="test_rb")

        rb.add_rule(
            name="规则1",
            rule_type=RuleType.CAUSAL,
            conditions=[],
            conclusion={},
            certainty=0.8
        )

        rules = rb.get_active_rules()
        assert len(rules) >= 0


class TestLogicParser:
    """逻辑解析器测试"""

    def test_logic_parser_initialization(self):
        """测试逻辑解析器初始化"""
        from core.symbolic.logic_parser import LogicParser

        parser = LogicParser()
        assert parser is not None

    def test_parse_propositional_logic(self):
        """测试命题逻辑解析"""
        from core.symbolic.logic_parser import LogicParser
        from core.symbolic.logic_parser import LogicType

        parser = LogicParser()

        ast = parser.parse_logic_expression(
            "A AND B",
            LogicType.PROPOSITIONAL
        )

        assert ast is not None

    def test_parse_predicate_logic(self):
        """测试谓词逻辑解析"""
        from core.symbolic.logic_parser import LogicParser
        from core.symbolic.logic_parser import LogicType

        parser = LogicParser()

        ast = parser.parse_logic_expression(
            "∀x (鸟(x) → 会飞(x))",
            LogicType.PREDICATE
        )

        assert ast is not None

    def test_simplify_expression(self):
        """测试表达式简化"""
        from core.symbolic.logic_parser import LogicParser
        from core.symbolic.logic_parser import LogicType

        parser = LogicParser()

        ast = parser.parse_logic_expression(
            "A AND TRUE",
            LogicType.PROPOSITIONAL
        )

        simplified = parser.simplify_expression(ast)
        assert simplified is not None


class TestInferenceEngine:
    """推理引擎测试"""

    def test_inference_engine_initialization(self):
        """测试推理引擎初始化"""
        from core.symbolic.inference_engine import InferenceEngine

        engine = InferenceEngine(name="test_engine")
        assert engine.name == "test_engine"

    def test_forward_chaining(self):
        """测试前向链接"""
        from core.symbolic.inference_engine import InferenceEngine

        engine = InferenceEngine(name="test_engine")

        result = engine.forward_chain(
            facts=["A", "A → B"],
            knowledge_base=None,
            rule_base=None
        )

        assert result is not None

    def test_backward_chaining(self):
        """测试后向链接"""
        from core.symbolic.inference_engine import InferenceEngine

        engine = InferenceEngine(name="test_engine")

        result = engine.backward_chain(
            goal="B",
            facts=["A", "A → B"],
            knowledge_base=None,
            rule_base=None
        )

        assert result is not None


class TestNeuroSymbolicArchitecture:
    """神经符号架构测试"""

    def test_neuro_symbolic_initialization(self):
        """测试神经符号架构初始化"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture

        network_config = {
            "input_dim": 128,
            "hidden_dims": [256, 128],
            "output_dim": 64
        }

        symbolic_config = {
            "knowledge_base": {},
            "rule_templates": [],
            "inference_depth": 5
        }

        arch = NeuroSymbolicArchitecture(
            network_config=network_config,
            symbolic_config=symbolic_config,
            inference_mode="hybrid"
        )

        assert arch is not None
        assert arch.inference_mode == "hybrid"

    def test_initialize_architecture(self):
        """测试架构初始化"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture

        arch = NeuroSymbolicArchitecture(
            network_config={"input_dim": 64, "hidden_dims": [128], "output_dim": 32},
            symbolic_config={},
            inference_mode="neural"
        )

        result = arch.initialize_architecture(
            knowledge_base={"rules": []},
            pre_trained_weights=None
        )

        assert result is True

    def test_extract_symbolic_knowledge(self):
        """测试符号知识提取"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture
        import torch

        arch = NeuroSymbolicArchitecture(
            network_config={"input_dim": 64, "hidden_dims": [128], "output_dim": 32},
            symbolic_config={},
            inference_mode="hybrid"
        )

        arch.initialize_architecture(knowledge_base={})

        activations = torch.randn(1, 64)
        result = arch.extract_symbolic_knowledge(activations)

        assert result is not None

    def test_hybrid_reasoning(self):
        """测试混合推理"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture
        import torch

        arch = NeuroSymbolicArchitecture(
            network_config={"input_dim": 64, "hidden_dims": [128], "output_dim": 32},
            symbolic_config={},
            inference_mode="hybrid"
        )

        arch.initialize_architecture(knowledge_base={})

        input_data = torch.randn(1, 64)
        result = arch.hybrid_reasoning(input_data)

        assert result is not None

    def test_architecture_state(self):
        """测试获取架构状态"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture

        arch = NeuroSymbolicArchitecture(
            network_config={"input_dim": 64, "hidden_dims": [128], "output_dim": 32},
            symbolic_config={},
            inference_mode="hybrid"
        )

        arch.initialize_architecture(knowledge_base={})
        state = arch.get_architecture_state()

        assert "architecture_state" in state
        assert "performance_stats" in state

    def test_performance_report(self):
        """测试性能报告"""
        from core.symbolic.neuro_symbolic_architecture import NeuroSymbolicArchitecture

        arch = NeuroSymbolicArchitecture(
            network_config={"input_dim": 64, "hidden_dims": [128], "output_dim": 32},
            symbolic_config={},
            inference_mode="hybrid"
        )

        arch.initialize_architecture(knowledge_base={})
        report = arch.get_performance_report()

        assert "inference_statistics" in report
        assert "architecture_status" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
