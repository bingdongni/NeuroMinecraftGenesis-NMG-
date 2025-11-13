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


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("符号逻辑推理引擎基础功能测试")
    print("=" * 60)
    
    # 创建推理引擎
    reasoner = SymbolicReasoner(name="test_reasoner")
    
    try:
        print("\n1. 添加知识测试")
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
            }
        ]
        
        fact_ids = []
        for fact_data in facts:
            fact_id = reasoner.add_knowledge("fact", fact_data)
            fact_ids.append(fact_id)
            print(f"✓ 添加事实: {fact_data['subject']} {fact_data['predicate']} {fact_data['object']}")
        
        # 添加规则知识
        rules = [
            {
                "name": "鸟类飞行规则",
                "rule_type": "modus_ponens",
                "conditions": ["_x 是 鸟类", "_x 不是 企鹅"],
                "conclusion": "_x 会飞",
                "certainty": 0.8,
                "description": "大多数鸟类会飞"
            }
        ]
        
        rule_ids = []
        for rule_data in rules:
            rule_id = reasoner.add_knowledge("rule", rule_data)
            rule_ids.append(rule_id)
            print(f"✓ 添加规则: {rule_data['name']}")
        
        print("\n2. 前向链式推理测试")
        print("-" * 40)
        
        result = reasoner.forward_chain("会飞", facts=["燕子 是 鸟类"])
        if result["success"]:
            print("✓ 前向推理成功!")
            print(f"  结果: {result['result']}")
            print(f"  执行时间: {result['execution_time']:.3f}秒")
        else:
            print(f"✗ 前向推理失败: {result['error']}")
        
        print("\n3. 后向链式推理测试")
        print("-" * 40)
        
        result = reasoner.backward_chain("会飞", facts=["企鹅 是 鸟类"])
        if result["success"]:
            print("✓ 后向推理成功!")
            print(f"  结果: {result['result']}")
        else:
            print(f"✗ 后向推理失败: {result['error']}")
        
        print("\n4. 模糊逻辑推理测试")
        print("-" * 40)
        
        fuzzy_facts = [
            {"variable": "温度", "membership_degree": 0.8},
            {"variable": "湿度", "membership_degree": 0.7}
        ]
        
        result = reasoner.fuzzy_reasoning(fuzzy_facts)
        if result["success"]:
            print("✓ 模糊推理成功!")
            print(f"  结果: {result['result']}")
        else:
            print(f"✗ 模糊推理失败: {result['error']}")
        
        print("\n5. 不确定性推理测试")
        print("-" * 40)
        
        uncertain_facts = [
            {"proposition": "今天下雨", "certainty": 0.7}
        ]
        
        result = reasoner.uncertain_reasoning(uncertain_facts)
        if result["success"]:
            print("✓ 不确定性推理成功!")
            print(f"  结果: {result['result']}")
        else:
            print(f"✗ 不确定性推理失败: {result['error']}")
        
        print("\n6. 双向推理测试")
        print("-" * 40)
        
        result = reasoner.bidirectional_reasoning("会飞", facts=["燕子 是 鸟类"])
        if result["success"]:
            print("✓ 双向推理成功!")
            print(f"  结果: {result['result']}")
        else:
            print(f"✗ 双向推理失败: {result['error']}")
        
        print("\n7. 批量推理测试")
        print("-" * 40)
        
        queries = [
            {"query": "会飞", "type": "forward_chain"},
            {"query": "是 鸟类", "type": "backward_chain"}
        ]
        
        batch_results = reasoner.batch_reason(queries)
        success_count = sum(1 for r in batch_results if r["success"])
        print(f"✓ 批量推理完成: {success_count}/{len(queries)} 成功")
        
        print("\n8. 知识库状态")
        print("-" * 40)
        
        stats = reasoner.get_statistics()
        print(f"✓ 知识库事实数: {stats['knowledge_base']['fact_count']}")
        print(f"✓ 规则库规则数: {stats['rule_base']['total_rules']}")
        print(f"✓ 成功查询数: {stats['reasoner']['successful_queries']}")
        print(f"✓ 总查询数: {stats['reasoner']['total_queries']}")
        
        print("\n9. 推理解释测试")
        print("-" * 40)
        
        result = reasoner.forward_chain("是 哺乳动物")
        if result["success"]:
            explanation = reasoner.explain_reasoning(result)
            print("✓ 推理解释:")
            print(explanation)
        
        print("\n10. 性能优化测试")
        print("-" * 40)
        
        optimization_results = reasoner.optimize_performance()
        print("✓ 性能优化完成")
        for key, value in optimization_results.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("测试完成! 符号逻辑推理引擎运行正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭推理引擎
        reasoner.shutdown()


if __name__ == "__main__":
    test_basic_functionality()