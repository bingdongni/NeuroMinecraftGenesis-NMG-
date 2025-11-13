#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域学习能力评估系统快速测试
Cross-Domain Learning Assessment System Quick Test

该脚本提供系统组件的基础功能测试，
确保所有核心组件都能正常工作。

作者: AI系统
日期: 2025-11-13
"""

import asyncio
import sys
import os

# 添加系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from worlds.real import (
            CrossDomainLearner,
            DomainAdapter,
            TransferAnalyzer,
            LearningEfficiency,
            AdaptationMetrics,
            create_cross_domain_learner,
            quick_assessment,
            system_health_check
        )
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_health_check():
    """测试系统健康检查"""
    print("\n测试系统健康检查...")
    
    try:
        from worlds.real import system_health_check
        health_status = system_health_check()
        
        if health_status['status'] == 'healthy':
            print("✓ 系统健康检查通过")
            print(f"  系统版本: {health_status['version']}")
            print(f"  检查时间: {health_status['timestamp']}")
            return True
        else:
            print("✗ 系统健康检查失败")
            print(f"  错误信息: {health_status.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"✗ 健康检查异常: {e}")
        return False

async def test_quick_assessment():
    """测试快速评估功能"""
    print("\n测试快速评估功能...")
    
    try:
        from worlds.real import quick_assessment
        
        print("  执行快速评估: game -> physics, social")
        result = await quick_assessment(
            source_domains=['game'],
            target_domains=['physics', 'social']
        )
        
        # 检查结果结构
        required_keys = ['overall_performance', 'domain_similarity', 'learning_results']
        for key in required_keys:
            if key not in result:
                print(f"✗ 缺少必需的结果字段: {key}")
                return False
        
        overall_score = result['overall_performance']['overall_score']
        print(f"✓ 快速评估完成")
        print(f"  总体得分: {overall_score:.3f}")
        print(f"  评估用时: {result['evaluation_duration']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 快速评估失败: {e}")
        return False

async def test_learning_efficiency():
    """测试学习效率评估"""
    print("\n测试学习效率评估...")
    
    try:
        from worlds.real import create_learning_efficiency
        
        evaluator = create_learning_efficiency({
            'speed_weight': 0.3,
            'quality_weight': 0.4
        })
        
        result = await evaluator.evaluate_learning_efficiency(
            domain='game',
            knowledge_base={'concepts': ['strategy', 'tactics']},
            evaluation_tasks={'tasks': ['classification']}
        )
        
        efficiency_score = result['efficiency_report'].overall_efficiency
        print(f"✓ 学习效率评估完成")
        print(f"  效率得分: {efficiency_score:.3f}")
        print(f"  学习模式: {result['efficiency_report'].learning_pattern}")
        
        return True
        
    except Exception as e:
        print(f"✗ 学习效率评估失败: {e}")
        return False

async def test_adaptation_metrics():
    """测试适应指标计算"""
    print("\n测试适应指标计算...")
    
    try:
        from worlds.real import create_adaptation_metrics
        
        calculator = create_adaptation_metrics({
            'speed_threshold': 0.1,
            'quality_threshold': 0.8
        })
        
        result = await calculator.evaluate_adaptation_speed(
            target_domain='physics',
            transferred_knowledge={'concepts': ['force', 'motion']},
            adaptation_tasks={'tasks': ['mechanics']}
        )
        
        adaptation_score = result['adaptation_report'].overall_adaptation_score
        print(f"✓ 适应指标计算完成")
        print(f"  适应评分: {adaptation_score:.3f}")
        print(f"  适应模式: {result['adaptation_report'].adaptation_pattern}")
        
        return True
        
    except Exception as e:
        print(f"✗ 适应指标计算失败: {e}")
        return False

async def test_domain_adapter():
    """测试领域适配器"""
    print("\n测试领域适配器...")
    
    try:
        from worlds.real import create_domain_adapter
        
        adapter = create_domain_adapter({
            'feature_adapter': {
                'adaptation_threshold': 0.7
            }
        })
        
        result = await adapter.adapt_knowledge(
            source_domains=['game'],
            target_domain='social',
            learner_agent=None
        )
        
        validation_score = result['quality_validation']['validation_score']
        print(f"✓ 领域适配完成")
        print(f"  适配质量: {validation_score:.3f}")
        print(f"  整合概念数: {len(result['integrated_result']['integrated_knowledge']['concepts'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ 领域适配失败: {e}")
        return False

async def test_transfer_analyzer():
    """测试迁移分析器"""
    print("\n测试迁移分析器...")
    
    try:
        from worlds.real import create_transfer_analyzer
        
        analyzer = create_transfer_analyzer({
            'efficiency_threshold': 0.7,
            'max_transfer_hops': 2
        })
        
        result = await analyzer.measure_transfer_efficiency(
            source_domains=['game'],
            target_domains=['physics'],
            knowledge_base={'concepts': ['strategy', 'movement']}
        )
        
        # 检查结果结构
        if 'physics' not in result:
            print("✗ 迁移分析结果缺少目标领域")
            return False
        
        best_efficiency = result['physics']['best_efficiency']
        print(f"✓ 迁移分析完成")
        print(f"  最佳效率: {best_efficiency:.3f}")
        print(f"  最佳源领域: {result['physics']['best_source']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 迁移分析失败: {e}")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("跨域学习能力评估系统快速测试")
    print("Cross-Domain Learning Assessment System Quick Test")
    print("=" * 60)
    
    test_results = []
    
    # 基础测试
    test_results.append(("模块导入", test_imports()))
    test_results.append(("健康检查", test_health_check()))
    
    # 功能测试
    test_results.append(("快速评估", await test_quick_assessment()))
    test_results.append(("学习效率", await test_learning_efficiency()))
    test_results.append(("适应指标", await test_adaptation_metrics()))
    test_results.append(("领域适配", await test_domain_adapter()))
    test_results.append(("迁移分析", await test_transfer_analyzer()))
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("-" * 30)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name:15} {status}")
    
    print(f"\n测试总结:")
    print(f"  总计测试: {total}")
    print(f"  通过测试: {passed}")
    print(f"  失败测试: {total - passed}")
    print(f"  通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print(f"\n🎉 所有测试通过！系统运行正常。")
        print("All tests passed! System is working properly.")
    else:
        print(f"\n⚠️  有测试失败，请检查相关组件。")
        print("Some tests failed, please check the related components.")
    
    print("=" * 60)
    
    return passed == total

def main():
    """主函数"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()