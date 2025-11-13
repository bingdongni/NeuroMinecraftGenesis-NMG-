#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号逻辑推理引擎测试文件
用于验证所有核心功能和组件的正常工作
"""

import sys
import os
import json
from datetime import datetime
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbolic.symbolic_reasoner import SymbolicReasoner, ReasoningConfig, ReasoningMode


class SymbolicReasonerTest:
    """符号推理引擎测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.test_results = []
        self.reasoner = None
        
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🧠 符号逻辑推理引擎测试")
        print("=" * 60)
        
        try:
            # 初始化推理引擎
            self._test_initialization()
            
            # 测试核心方法
            self._test_parse_logic_expression()
            self._test_forward_chain()
            self._test_backward_chain()
            self._test_fuzzy_reasoning()
            self._test_uncertain_reasoning()
            
            # 测试高级功能
            self._test_bidirectional_reasoning()
            self._test_batch_reasoning()
            self._test_knowledge_management()
            self._test_performance_monitoring()
            
            # 打印测试总结
            self._print_test_summary()
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {str(e)}")
            print(traceback.format_exc())
        finally:
            if self.reasoner:
                self.reasoner.shutdown()
    
    def _test_initialization(self):
        """测试初始化"""
        print("\n📋 测试1: 初始化推理引擎")
        try:
            config = ReasoningConfig(
                mode=ReasoningMode.AUTOMATIC,
                max_iterations=50,
                certainty_threshold=0.1,
                confidence_threshold=0.1,
                llm_enabled=False  # 测试时不启用LLM
            )
            self.reasoner = SymbolicReasoner("test_reasoner", config)
            print("✅ 推理引擎初始化成功")
            self._add_test_result("初始化测试", True, "推理引擎初始化成功")
        except Exception as e:
            print(f"❌ 推理引擎初始化失败: {str(e)}")
            self._add_test_result("初始化测试", False, str(e))
    
    def _test_parse_logic_expression(self):
        """测试逻辑表达式解析"""
        print("\n📋 测试2: 逻辑表达式解析")
        try:
            test_cases = [
                ("P → Q", "propositional"),
                ("∀x (Human(x) → Mortal(x))", "first_order"),
                ("◇P", "modal")
            ]
            
            for expression, logic_type in test_cases:
                result = self.reasoner.parse_logic_expression(expression, logic_type)
                if result["success"]:
                    print(f"✅ 解析 '{expression}' 成功")
                else:
                    print(f"❌ 解析 '{expression}' 失败: {result.get('error')}")
                    self._add_test_result("逻辑表达式解析", False, f"解析'{expression}'失败")
                    return
            
            print("✅ 逻辑表达式解析测试通过")
            self._add_test_result("逻辑表达式解析", True, "所有表达式解析成功")
        except Exception as e:
            print(f"❌ 逻辑表达式解析测试失败: {str(e)}")
            self._add_test_result("逻辑表达式解析", False, str(e))
    
    def _test_forward_chain(self):
        """测试前向链式推理"""
        print("\n📋 测试3: 前向链式推理")
        try:
            # 添加测试知识
            facts = [
                "鸟会飞",
                "企鹅是鸟",
                "羽毛是保暖的"
            ]
            
            rules = [
                {
                    "name": "如果它是鸟，那么它有羽毛",
                    "conditions": ["鸟会飞"],
                    "conclusion": "企鹅有羽毛",
                    "certainty": 0.9
                },
                {
                    "name": "如果有羽毛，那么它是保暖的",
                    "conditions": ["企鹅有羽毛"],
                    "conclusion": "企鹅是保暖的",
                    "certainty": 0.8
                }
            ]
            
            # 添加知识到推理引擎
            for fact in facts:
                self.reasoner.add_knowledge("fact", {
                    "subject": fact.split()[0],
                    "predicate": fact.split()[1], 
                    "object": " ".join(fact.split()[2:]),
                    "certainty": 1.0,
                    "source": "test"
                })
            
            for rule in rules:
                self.reasoner.add_knowledge("rule", {
                    "name": rule["name"],
                    "rule_type": "if_then",
                    "conditions": rule["conditions"],
                    "conclusion": rule["conclusion"],
                    "certainty": rule["certainty"],
                    "source": "test"
                })
            
            # 执行前向推理
            result = self.reasoner.forward_chain("企鹅是保暖的")
            
            if result["success"]:
                print("✅ 前向链式推理测试成功")
                print(f"   - 推理步骤数: {result['reasoning_steps']}")
                print(f"   - 执行时间: {result['execution_time']:.3f}秒")
                self._add_test_result("前向链式推理", True, "推理执行成功")
            else:
                print(f"❌ 前向链式推理失败: {result.get('error')}")
                self._add_test_result("前向链式推理", False, result.get('error'))
                
        except Exception as e:
            print(f"❌ 前向链式推理测试失败: {str(e)}")
            self._add_test_result("前向链式推理", False, str(e))
    
    def _test_backward_chain(self):
        """测试后向链式推理"""
        print("\n📋 测试4: 后向链式推理")
        try:
            result = self.reasoner.backward_chain("企鹅是保暖的")
            
            if result["success"]:
                print("✅ 后向链式推理测试成功")
                print(f"   - 推理步骤数: {result['reasoning_steps']}")
                print(f"   - 证明找到: {result['proof_found']}")
                self._add_test_result("后向链式推理", True, "推理执行成功")
            else:
                print(f"❌ 后向链式推理失败: {result.get('error')}")
                self._add_test_result("后向链式推理", False, result.get('error'))
                
        except Exception as e:
            print(f"❌ 后向链式推理测试失败: {str(e)}")
            self._add_test_result("后向链式推理", False, str(e))
    
    def _test_fuzzy_reasoning(self):
        """测试模糊逻辑推理"""
        print("\n📋 测试5: 模糊逻辑推理")
        try:
            fuzzy_facts = [
                {
                    "variable": "温度",
                    "value": "高",
                    "membership_degree": 0.8
                },
                {
                    "variable": "湿度", 
                    "value": "中",
                    "membership_degree": 0.6
                }
            ]
            
            result = self.reasoner.fuzzy_reasoning(fuzzy_facts)
            
            if result["success"]:
                print("✅ 模糊逻辑推理测试成功")
                print(f"   - 模糊事实数: {result['fuzzy_facts_count']}")
                print(f"   - 推理步骤数: {result['reasoning_steps']}")
                self._add_test_result("模糊逻辑推理", True, "推理执行成功")
            else:
                print(f"❌ 模糊逻辑推理失败: {result.get('error')}")
                self._add_test_result("模糊逻辑推理", False, result.get('error'))
                
        except Exception as e:
            print(f"❌ 模糊逻辑推理测试失败: {str(e)}")
            self._add_test_result("模糊逻辑推理", False, str(e))
    
    def _test_uncertain_reasoning(self):
        """测试不确定性推理"""
        print("\n📋 测试6: 不确定性推理")
        try:
            uncertain_facts = [
                {
                    "proposition": "可能会下雨",
                    "certainty": 0.7
                },
                {
                    "proposition": "温度会下降",
                    "certainty": 0.6
                }
            ]
            
            result = self.reasoner.uncertain_reasoning(uncertain_facts)
            
            if result["success"]:
                print("✅ 不确定性推理测试成功")
                print(f"   - 不确定性事实数: {result['uncertain_facts_count']}")
                print(f"   - 推理步骤数: {result['reasoning_steps']}")
                self._add_test_result("不确定性推理", True, "推理执行成功")
            else:
                print(f"❌ 不确定性推理失败: {result.get('error')}")
                self._add_test_result("不确定性推理", False, result.get('error'))
                
        except Exception as e:
            print(f"❌ 不确定性推理测试失败: {str(e)}")
            self._add_test_result("不确定性推理", False, str(e))
    
    def _test_bidirectional_reasoning(self):
        """测试双向推理"""
        print("\n📋 测试7: 双向推理")
        try:
            result = self.reasoner.bidirectional_reasoning("企鹅是保暖的")
            
            if result["success"]:
                print("✅ 双向推理测试成功")
                print(f"   - 前向路径: {result.get('forward_paths', 0)}")
                print(f"   - 后向路径: {result.get('backward_paths', 0)}")
                print(f"   - 总推理路径: {result.get('total_reasoning_paths', 0)}")
                self._add_test_result("双向推理", True, "推理执行成功")
            else:
                print(f"❌ 双向推理失败: {result.get('error')}")
                self._add_test_result("双向推理", False, result.get('error'))
                
        except Exception as e:
            print(f"❌ 双向推理测试失败: {str(e)}")
            self._add_test_result("双向推理", False, str(e))
    
    def _test_batch_reasoning(self):
        """测试批量推理"""
        print("\n📋 测试8: 批量推理")
        try:
            queries = [
                {"query": "企鹅是保暖的", "type": "forward_chain"},
                {"query": "鸟会飞", "type": "backward_chain"},
                {"type": "fuzzy", "fuzzy_facts": [{"variable": "测试", "value": "高", "membership_degree": 0.7}]}
            ]
            
            results = self.reasoner.batch_reason(queries)
            
            success_count = sum(1 for r in results if r.get("success", False))
            
            if success_count > 0:
                print(f"✅ 批量推理测试成功 ({success_count}/{len(results)} 成功)")
                self._add_test_result("批量推理", True, f"{success_count}/{len(results)} 查询成功")
            else:
                print("❌ 批量推理全部失败")
                self._add_test_result("批量推理", False, "所有查询都失败")
                
        except Exception as e:
            print(f"❌ 批量推理测试失败: {str(e)}")
            self._add_test_result("批量推理", False, str(e))
    
    def _test_knowledge_management(self):
        """测试知识管理"""
        print("\n📋 测试9: 知识管理")
        try:
            # 测试添加不同类型的知识
            test_knowledge = [
                {
                    "type": "fact",
                    "data": {
                        "subject": "测试主体",
                        "predicate": "是",
                        "object": "测试对象",
                        "certainty": 0.9
                    }
                },
                {
                    "type": "fuzzy_fact", 
                    "data": {
                        "variable": "测试变量",
                        "value": "高",
                        "membership_degree": 0.8
                    }
                }
            ]
            
            success_count = 0
            for knowledge in test_knowledge:
                result = self.reasoner.add_knowledge(knowledge["type"], knowledge["data"])
                if result:
                    success_count += 1
            
            if success_count == len(test_knowledge):
                print(f"✅ 知识管理测试成功 ({success_count}/{len(test_knowledge)} 成功)")
                self._add_test_result("知识管理", True, f"成功添加{success_count}项知识")
            else:
                print(f"❌ 知识管理测试部分失败 ({success_count}/{len(test_knowledge)})")
                self._add_test_result("知识管理", False, f"只成功添加{success_count}项知识")
                
        except Exception as e:
            print(f"❌ 知识管理测试失败: {str(e)}")
            self._add_test_result("知识管理", False, str(e))
    
    def _test_performance_monitoring(self):
        """测试性能监控"""
        print("\n📋 测试10: 性能监控")
        try:
            # 获取统计信息
            stats = self.reasoner.get_statistics()
            
            if stats:
                print("✅ 性能监控测试成功")
                print(f"   - 推理引擎统计: {len(stats.get('reasoner', {}))} 项")
                print(f"   - 知识库统计: {len(stats.get('knowledge_base', {}))} 项")
                print(f"   - 规则库统计: {len(stats.get('rule_base', {}))} 项")
                self._add_test_result("性能监控", True, "统计信息获取成功")
            else:
                print("❌ 性能监控获取统计信息失败")
                self._add_test_result("性能监控", False, "统计信息获取失败")
                
        except Exception as e:
            print(f"❌ 性能监控测试失败: {str(e)}")
            self._add_test_result("性能监控", False, str(e))
    
    def _add_test_result(self, test_name: str, success: bool, message: str):
        """添加测试结果"""
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def _print_test_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("📊 测试结果总结")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test_name']}: {result['message']}")
        
        print("\n✅ 测试完成!")
        
        # 保存测试结果到文件
        self._save_test_results()
    
    def _save_test_results(self):
        """保存测试结果到文件"""
        try:
            with open("test_results.json", "w", encoding="utf-8") as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print(f"\n📄 测试结果已保存到 test_results.json")
        except Exception as e:
            print(f"\n❌ 保存测试结果失败: {str(e)}")


def main():
    """主函数"""
    test_suite = SymbolicReasonerTest()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()