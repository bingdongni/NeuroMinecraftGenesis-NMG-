#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增功能的专门脚本
验证新添加的方法是否正常工作
"""

import sys
import json
import numpy as np
from datetime import datetime

# 导入系统组件
try:
    from transfer_evaluator import TransferEvaluator
    from performance_analyzer import PerformanceAnalyzer
    print("✓ 新功能模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

def test_transfer_evaluator_new_features():
    """测试 TransferEvaluator 的新增功能"""
    print("\n=== 测试 TransferEvaluator 新功能 ===")
    
    try:
        # 初始化评估器
        evaluator = TransferEvaluator()
        print("✓ 迁移评估器初始化成功")
        
        # 准备测试数据
        adapted_strategy = {
            "strategy_type": "grab_and_move",
            "minecraft_performance": 0.85,
            "mapped_strategy_id": "test_adapted"
        }
        
        execution_results = {
            "execution_data": [
                {
                    "actual_position": [0.9, 1.1, 1.0],
                    "target_position": [1.0, 1.0, 1.0],
                    "actual_orientation": [0.1, 0.0, 0.0],
                    "target_orientation": [0.0, 0.0, 0.0],
                    "success": True,
                    "execution_time": 2.5,
                    "completed": True,
                    "error_count": 0,
                    "final_state_correct": True
                },
                {
                    "actual_position": [1.1, 0.9, 1.1],
                    "target_position": [1.0, 1.0, 1.0],
                    "actual_orientation": [0.0, 0.1, 0.0],
                    "target_orientation": [0.0, 0.0, 0.0],
                    "success": True,
                    "execution_time": 2.3,
                    "completed": True,
                    "error_count": 1,
                    "final_state_correct": True
                }
            ],
            "overall_score": 0.82,
            "performance_by_complexity": {1: 0.85, 2: 0.78, 3: 0.75}
        }
        
        # 测试1：深入迁移质量分析
        print("\n--- 测试1: 深入迁移质量分析 ---")
        quality_analysis = evaluator.analyze_transfer_quality(
            adapted_strategy, execution_results, 
            quality_dimensions=['precision', 'consistency', 'efficiency']
        )
        print(f"✓ 迁移质量分析完成")
        print(f"  综合质量分数: {quality_analysis['overall_quality_score']:.2f}")
        print(f"  质量等级: {quality_analysis['quality_grade']}")
        print(f"  分析维度: {len(quality_analysis['quality_dimensions'])}")
        print(f"  发现质量问题: {len(quality_analysis['quality_issues'])} 个")
        
        # 测试2：多策略对比分析
        print("\n--- 测试2: 多策略对比分析 ---")
        strategies_data = {
            'strategy_a': {
                'strategy': {
                    'strategy_type': 'grab_and_move',
                    'mapped_strategy_id': 'test_a'
                },
                'results': {
                    'execution_data': [
                        {'actual_position': [0.9, 1.1, 1.0], 'target_position': [1.0, 1.0, 1.0], 'success': True, 'execution_time': 2.0},
                        {'actual_position': [1.1, 0.9, 1.1], 'target_position': [1.0, 1.0, 1.0], 'success': True, 'execution_time': 2.2}
                    ],
                    'overall_score': 0.85
                }
            },
            'strategy_b': {
                'strategy': {
                    'strategy_type': 'precision_grab',
                    'mapped_strategy_id': 'test_b'
                },
                'results': {
                    'execution_data': [
                        {'actual_position': [0.95, 1.05, 1.0], 'target_position': [1.0, 1.0, 1.0], 'success': True, 'execution_time': 2.8},
                        {'actual_position': [1.05, 0.95, 1.0], 'target_position': [1.0, 1.0, 1.0], 'success': True, 'execution_time': 2.9}
                    ],
                    'overall_score': 0.92
                }
            }
        }
        
        strategy_comparison = evaluator.compare_strategies(strategies_data)
        print(f"✓ 策略对比分析完成")
        print(f"  对比策略数: {strategy_comparison['strategies_count']}")
        print(f"  最佳策略: {strategy_comparison['best_strategy']}")
        print(f"  平均性能: {strategy_comparison['analysis_summary']['average_performance']:.2f}")
        print(f"  选择建议数: {len(strategy_comparison['selection_recommendations'])}")
        
        # 测试3：改进建议生成
        print("\n--- 测试3: 改进建议生成 ---")
        evaluation_result = {
            'overall_score': 0.75,
            'metrics': {'accuracy': 0.7, 'success_rate': 0.8, 'execution_time': 0.6},
            'performance_comparison': {
                'target_performance': {
                    'degradation_areas': [{'metric': 'execution_time', 'percentage': 15}]
                }
            },
            'statistical_analysis': {
                'confidence_interval': {'margin_of_error': 0.05}
            }
        }
        
        improvement_suggestions = evaluator.generate_improvement_suggestions(
            evaluation_result, suggestion_type="comprehensive"
        )
        print(f"✓ 改进建议生成完成")
        print(f"  建议类型: {improvement_suggestions['suggestion_type']}")
        print(f"  总建议数: {improvement_suggestions['recommendations_summary']['total_recommendations']}")
        print(f"  置信度: {improvement_suggestions['confidence_level']:.2f}")
        print(f"  当前性能: {improvement_suggestions['current_performance']['overall_score']:.2f}")
        
        print("\n✓ 所有 TransferEvaluator 新功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ TransferEvaluator 新功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_analyzer_new_features():
    """测试 PerformanceAnalyzer 的新增功能"""
    print("\n=== 测试 PerformanceAnalyzer 新功能 ===")
    
    try:
        # 初始化性能分析器
        analyzer = PerformanceAnalyzer()
        print("✓ 性能分析器初始化成功")
        
        # 测试1：性能趋势预测
        print("\n--- 测试1: 性能趋势预测 ---")
        performance_history = {
            'accuracy': [0.7, 0.75, 0.8, 0.82, 0.85],
            'success_rate': [0.8, 0.82, 0.85, 0.87, 0.9],
            'execution_time': [3.0, 2.8, 2.6, 2.5, 2.4]
        }
        
        trend_prediction = analyzer.predict_performance_trend(
            performance_history, prediction_horizon=5, confidence_level=0.95
        )
        print(f"✓ 性能趋势预测完成")
        print(f"  预测指标数: {trend_prediction['prediction_metadata']['metrics_predicted']}")
        print(f"  整体趋势: {trend_prediction['overall_trend']['overall_direction']}")
        print(f"  预测置信度: {trend_prediction['prediction_confidence']:.2f}")
        print(f"  预测质量: {trend_prediction['prediction_metadata']['prediction_quality']}")
        
        # 测试2：系统瓶颈识别
        print("\n--- 测试2: 系统瓶颈识别 ---")
        performance_metrics = {
            'accuracy': {
                'current_value': 0.75,
                'target_value': 0.9,
                'trend': 'stable'
            },
            'execution_time': {
                'current_value': 3.0,
                'target_value': 2.0,
                'trend': 'declining'
            }
        }
        
        resource_utilization = {
            'cpu': 0.95,  # CPU利用率95%，瓶颈
            'memory': 0.6,
            'storage': 0.3
        }
        
        system_constraints = {
            'max_concurrent_tasks': {
                'current_limit': 10,
                'required_capacity': 15
            }
        }
        
        bottleneck_analysis = analyzer.identify_bottlenecks(
            performance_metrics, resource_utilization, system_constraints
        )
        print(f"✓ 系统瓶颈识别完成")
        print(f"  发现瓶颈数: {bottleneck_analysis['analysis_summary']['total_bottlenecks_identified']}")
        print(f"  严重程度: {bottleneck_analysis['bottleneck_severity']['overall_severity']}")
        print(f"  分析置信度: {bottleneck_analysis['analysis_confidence']:.2f}")
        print(f"  紧急行动数: {bottleneck_analysis['analysis_summary']['immediate_action_required']}")
        
        # 测试3：资源分配优化
        print("\n--- 测试3: 资源分配优化 ---")
        current_allocation = {
            'cpu_cores': {'allocated': 8},
            'memory_gb': {'allocated': 16},
            'storage_tb': {'allocated': 2}
        }
        
        performance_requirements = {
            'cpu_cores': 10,
            'memory_gb': 20,
            'storage_tb': 1
        }
        
        resource_constraints = {
            'cpu_cores': 12,
            'memory_gb': 32,
            'storage_tb': 5
        }
        
        optimization_result = analyzer.optimize_resource_allocation(
            current_allocation, performance_requirements, resource_constraints,
            optimization_objective="balanced"
        )
        print(f"✓ 资源分配优化完成")
        print(f"  优化目标: {optimization_result['optimization_objective']}")
        print(f"  优化置信度: {optimization_result['optimization_confidence']:.2f}")
        print(f"  实施复杂度: {optimization_result['optimization_summary']['implementation_complexity']}")
        print(f"  预期性能提升: {optimization_result['optimization_summary']['expected_performance_improvement']:.1f}%")
        print(f"  优化阶段数: {len(optimization_result['implementation_plan']['implementation_phases'])}")
        
        print("\n✓ 所有 PerformanceAnalyzer 新功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ PerformanceAnalyzer 新功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("策略迁移系统新增功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 测试 TransferEvaluator 新功能
    test_results.append(("TransferEvaluator 新功能", test_transfer_evaluator_new_features()))
    
    # 测试 PerformanceAnalyzer 新功能
    test_results.append(("PerformanceAnalyzer 新功能", test_performance_analyzer_new_features()))
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("新功能测试总结:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有新功能测试都通过了！")
    else:
        print(f"\n⚠️  有 {total - passed} 个新功能测试失败")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)