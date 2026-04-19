"""
符号逻辑推理引擎使用示例
演示各种推理功能和知识管理操作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.symbolic.symbolic_reasoner import SymbolicReasoner, ReasoningMode, KnowledgeIntegration
from core.symbolic.rule_base import RuleType
import json
from datetime import datetime


def demo_basic_reasoning():
    """演示基本推理功能"""
    print("=" * 60)
    print("符号逻辑推理引擎使用示例")
    print("=" * 60)
    
    # 初始化推理引擎
    reasoner = SymbolicReasoner(name="demo_reasoner")
    
    try:
        print("\n1. 逻辑表达式解析示例")
        print("-" * 40)
        
        # 解析逻辑表达式
        expressions = [
            "A ∧ B",
            "¬(A ∨ B)",
            "∀x (P(x) → Q(x))",
            "□P → ◇Q"
        ]
        
        for expr in expressions:
            result = reasoner.parse_logic_expression(expr)
            print(f"表达式: {expr}")
            if result["success"]:
                print(f"  解析成功: {result['formatted']}")
                print(f"  简化: {result['simplified']}")
            else:
                print(f"  解析失败: {result['error']}")
            print()
        
        print("\n2. 添加知识示例")
        print("-" * 40)
        
        # 添加事实知识
        facts = [
            {
                "subject": "鸟类",
                "predicate": "会飞",
                "object": "不是所有",
                "certainty": 0.8,
                "description": "大部分鸟类会飞，但有例外如企鹅、鸵鸟"
            },
            {
                "subject": "企鹅",
                "predicate": "是",
                "object": "鸟类",
                "certainty": 1.0,
                "description": "企鹅是鸟类的生物分类"
            },
            {
                "subject": "蝙蝠",
                "predicate": "会飞",
                "object": "是",
                "certainty": 1.0,
                "description": "蝙蝠会飞行但不是鸟类，是哺乳动物"
            }
        ]
        
        fact_ids = []
        for fact_data in facts:
            fact_id = reasoner.add_knowledge("fact", fact_data)
            fact_ids.append(fact_id)
            print(f"添加事实: {fact_data['subject']} {fact_data['predicate']} {fact_data['object']}")
        
        # 添加规则知识
        rules = [
            {
                "name": "鸟类飞行规则",
                "rule_type": "modus_ponens",
                "conditions": ["_x 是 鸟类", "_x 不是 企鹅", "_x 不是 鸵鸟"],
                "conclusion": "_x 会飞",
                "certainty": 0.8,
                "description": "大多数鸟类会飞"
            },
            {
                "name": "哺乳动物规则",
                "rule_type": "custom_rule",
                "conditions": ["_x 是 蝙蝠"],
                "conclusion": "_x 是 哺乳动物",
                "certainty": 1.0,
                "description": "蝙蝠是哺乳动物"
            }
        ]
        
        rule_ids = []
        for rule_data in rules:
            rule_id = reasoner.add_knowledge("rule", rule_data)
            rule_ids.append(rule_id)
            print(f"添加规则: {rule_data['name']}")
        
        print("\n3. 前向链式推理示例")
        print("-" * 40)
        
        # 前向推理
        result = reasoner.forward_chain("会飞", facts=["企鹅 是 鸟类", "蝙蝠 是 鸟类"])
        print(f"推理结果: {result}")
        
        if result["success"]:
            print("推理成功!")
            print(f"生成的新事实数: {result['new_facts_generated']}")
            if "reasoning_paths" in result:
                print("推理路径:")
                for i, path in enumerate(result["reasoning_paths"][:3]):  # 只显示前3条
                    print(f"  路径 {i+1}: {path['goal']} (确定性: {path['total_certainty']:.2f})")
        
        print("\n4. 后向链式推理示例")
        print("-" * 40)
        
        # 后向推理
        result = reasoner.backward_chain("会飞", facts=["企鹅 是 鸟类"])
        print(f"推理结果: {result}")
        
        if result["success"]:
            print("证明成功!")
            if "reasoning_paths" in result:
                print("证明路径:")
                for i, path in enumerate(result["reasoning_paths"][:3]):
                    print(f"  路径 {i+1}: 步骤数 {path['length']}")
        
        print("\n5. 模糊逻辑推理示例")
        print("-" * 40)
        
        # 模糊推理
        fuzzy_facts = [
            {"variable": "温度", "membership_degree": 0.8},
            {"variable": "湿度", "membership_degree": 0.7}
        ]
        
        fuzzy_rules = [
            {
                "antecedent": ["温度", "湿度"],
                "consequent": "舒适度",
                "strength": 0.9
            }
        ]
        
        result = reasoner.fuzzy_reasoning(fuzzy_facts, fuzzy_rules)
        print(f"模糊推理结果: {result}")
        
        if result["success"] and "new_fuzzy_facts" in result:
            print("生成的模糊事实:")
            for fact in result["new_fuzzy_facts"]:
                print(f"  {fact['variable']}: {fact['membership_degree']:.2f}")
        
        print("\n6. 不确定性推理示例")
        print("-" * 40)
        
        # 不确定性推理
        uncertain_facts = [
            {"proposition": "今天下雨", "certainty": 0.7},
            {"proposition": "地面湿滑", "certainty": 0.6}
        ]
        
        uncertain_rules = [
            {
                "antecedent": ["今天下雨"],
                "consequent": "地面湿滑",
                "certainty": 0.8
            }
        ]
        
        result = reasoner.uncertain_reasoning(uncertain_facts, uncertain_rules)
        print(f"不确定性推理结果: {result}")
        
        if result["success"] and "new_uncertain_facts" in result:
            print("生成新的不确定性事实:")
            for fact in result["new_uncertain_facts"]:
                print(f"  {fact['proposition']}: 确定性 {fact['certainty']:.2f}")
        
        print("\n7. 双向推理示例")
        print("-" * 40)
        
        result = reasoner.bidirectional_reasoning("会飞", facts=["燕子 是 鸟类"])
        print(f"双向推理结果: {result}")
        
        print("\n8. 知识库状态查看")
        print("-" * 40)
        
        stats = reasoner.get_statistics()
        print("推理引擎统计信息:")
        print(f"  总查询数: {stats['reasoner']['total_queries']}")
        print(f"  成功查询数: {stats['reasoner']['successful_queries']}")
        print(f"  知识库事实数: {stats['knowledge_base']['fact_count']}")
        print(f"  规则库规则数: {stats['rule_base']['total_rules']}")
        print(f"  平均响应时间: {stats['reasoner']['average_response_time']:.3f}秒")
        
        print("\n9. 推理解释示例")
        print("-" * 40)
        
        # 获取一次推理结果并解释
        result = reasoner.forward_chain("是 哺乳动物")
        if result["success"]:
            explanation = reasoner.explain_reasoning(result)
            print("推理解释:")
            print(explanation)
        
        print("\n10. 批量推理示例")
        print("-" * 40)
        
        queries = [
            {"query": "会飞", "type": "forward_chain"},
            {"query": "是 哺乳动物", "type": "backward_chain"},
            {"query": "舒适度", "type": "fuzzy"}
        ]
        
        batch_results = reasoner.batch_reason(queries)
        print("批量推理结果:")
        for i, batch_result in enumerate(batch_results):
            print(f"  查询 {i+1}: {'成功' if batch_result['success'] else '失败'}")
        
        print("\n11. 性能优化示例")
        print("-" * 40)
        
        optimization_results = reasoner.optimize_performance()
        print("性能优化结果:")
        for key, value in optimization_results.items():
            print(f"  {key}: {value}")
        
        print("\n12. 知识库导出/导入示例")
        print("-" * 40)
        
        # 导出知识库
        exported_data = reasoner.export_knowledge_base()
        print("知识库导出成功")
        print(f"导出数据大小: {len(str(exported_data))} 字符")
        
        # 清理并重新创建推理引擎
        reasoner.shutdown()
        new_reasoner = SymbolicReasoner(name="imported_reasoner")
        
        # 导入知识库
        import_success = new_reasoner.import_knowledge_base(exported_data)
        print(f"知识库导入结果: {'成功' if import_success else '失败'}")
        
        if import_success:
            # 验证导入是否成功
            verification_result = new_reasoner.forward_chain("会飞")
            print(f"导入后的推理验证: {'成功' if verification_result['success'] else '失败'}")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭推理引擎
        reasoner.shutdown()
    
    print("\n" + "=" * 60)
    print("示例演示完成")
    print("=" * 60)


def demo_advanced_features():
    """演示高级功能"""
    print("\n高级功能演示")
    print("=" * 40)
    
    # 创建推理引擎并配置高级选项
    reasoner = SymbolicReasoner(name="advanced_demo", max_workers=2)
    
    try:
        print("\n1. 推理会话管理")
        print("-" * 30)
        
        # 创建会话
        session_id = reasoner.create_session(
            user_id="demo_user",
            preferences={"reasoning_style": "detailed", "max_results": 5}
        )
        print(f"创建会话: {session_id}")
        
        # 在会话中执行推理
        result1 = reasoner.forward_chain("是 鸟类")
        result2 = reasoner.backward_chain("会飞")
        
        print(f"会话中的推理完成")
        
        # 结束会话
        reasoner.end_session(session_id)
        print("会话已结束")
        
        print("\n2. LLM集成模拟")
        print("-" * 30)
        
        # 模拟LLM集成
        llm_success = reasoner.integrate_llm(
            provider="demo_provider",
            api_key="demo_key",
            model_name="demo_model"
        )
        print(f"LLM集成设置: {'成功' if llm_success else '失败'}")
        
        if llm_success:
            # LLM辅助推理
            llm_result = reasoner.llm_assisted_reasoning(
                "解释为什么企鹅不会飞",
                {"context": "企鹅是鸟类但不会飞的特殊例子"}
            )
            print("LLM辅助推理结果:")
            print(llm_result)
        
        print("\n3. 复杂逻辑表达式处理")
        print("-" * 30)
        
        complex_expressions = [
            "∀x (鸟类(x) → (会飞(x) ∨ ¬会飞(x)))",
            "□P → ◇Q ∧ □R",
            "∃x (学生(x) ∧ 喜欢(x, 数学))"
        ]
        
        for expr in complex_expressions:
            result = reasoner.parse_logic_expression(expr)
            print(f"复杂表达式: {expr}")
            if result["success"]:
                print(f"  成功解析: {result['formatted']}")
            else:
                print(f"  解析失败: {result['error']}")
            print()
        
    except Exception as e:
        print(f"高级功能演示出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        reasoner.shutdown()


if __name__ == "__main__":
    print("开始符号逻辑推理引擎演示")
    
    # 运行基本功能演示
    demo_basic_reasoning()
    
    # 运行高级功能演示
    demo_advanced_features()
    
    print("\n所有演示完成!")