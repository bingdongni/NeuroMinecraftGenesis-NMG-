#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略迁移系统测试脚本
验证系统各个组件的基本功能
"""

import sys
import time
import json
import numpy as np
from datetime import datetime

# 导入系统组件
try:
    from strategy_transfer import StrategyTransfer
    from knowledge_mapper import KnowledgeMapper
    from transfer_evaluator import TransferEvaluator
    from adaptation_engine import AdaptationEngine
    from performance_analyzer import PerformanceAnalyzer
    print("✓ 所有组件导入成功")
except ImportError as e:
    print(f"✗ 组件导入失败: {e}")
    sys.exit(1)


def test_strategy_transfer():
    """测试策略迁移主类"""
    print("\n=== 测试 StrategyTransfer ===")
    
    try:
        # 初始化系统
        transfer_system = StrategyTransfer()
        print("✓ 策略迁移系统初始化成功")
        
        # 开始迁移会话
        session_id = transfer_system.start_transfer_session("test_session")
        print(f"✓ 迁移会话创建成功: {session_id}")
        
        # 准备测试数据
        minecraft_data = {
            "scene_info": {
                "world_size": {"x": 10, "y": 5, "z": 10},
                "block_properties": {
                    "stone": {"hardness": 1.5, "density": 2.7},
                    "wood": {"hardness": 0.8, "density": 0.6}
                }
            },
            "action_sequences": [
                {
                    "action_type": "grab",
                    "position": [5, 2, 5],
                    "target_block": "stone",
                    "parameters": {"force": 0.8}
                }
            ],
            "performance_metrics": {
                "success_rate": 0.9,
                "execution_time": 2.0,
                "accuracy": 0.95
            }
        }
        
        # 测试策略提取
        strategy = transfer_system.extract_minecraft_strategy(minecraft_data, session_id)
        print(f"✓ 策略提取完成，类型: {strategy['strategy_type']}")
        print(f"  置信度: {strategy['confidence_score']:.2f}")
        
        # 测试状态查询
        status = transfer_system.get_transfer_status(session_id)
        print(f"✓ 状态查询成功: {status['status']}")
        
        # 完成会话
        summary = transfer_system.complete_transfer_session(session_id)
        print(f"✓ 会话完成，策略处理数: {summary['strategies_processed']}")
        
        return True
        
    except Exception as e:
        print(f"✗ StrategyTransfer 测试失败: {e}")
        return False


def test_knowledge_mapper():
    """测试知识映射器"""
    print("\n=== 测试 KnowledgeMapper ===")
    
    try:
        # 初始化映射器
        mapper = KnowledgeMapper()
        print("✓ 知识映射器初始化成功")
        
        # 准备测试策略
        test_strategy = {
            "strategy_id": "test_strategy",
            "strategy_type": "grab_and_move",
            "action_sequences": [
                {
                    "action_type": "grab",
                    "position": [1, 1, 1],
                    "parameters": {"force": 0.8}
                }
            ],
            "environmental_context": {
                "position": [1, 1, 1],
                "world_size": {"x": 10, "y": 5, "z": 10}
            }
        }
        
        # 执行映射
        mapping_result = mapper.map_strategy(test_strategy)
        print(f"✓ 策略映射完成")
        print(f"  映射置信度: {mapping_result['mapping_confidence']:.2f}")
        print(f"  映射动作数: {len(mapping_result['mapped_actions'])}")
        
        # 获取映射统计
        stats = mapper.get_mapping_statistics()
        print(f"✓ 映射统计: 总映射数 {stats['total_mappings_completed']}")
        
        return True
        
    except Exception as e:
        print(f"✗ KnowledgeMapper 测试失败: {e}")
        return False


def test_transfer_evaluator():
    """测试迁移评估器"""
    print("\n=== 测试 TransferEvaluator ===")
    
    try:
        # 初始化评估器
        evaluator = TransferEvaluator()
        print("✓ 迁移评估器初始化成功")
        
        # 准备测试数据
        adapted_strategy = {
            "adapted_strategy_id": "test_adapted",
            "strategy_type": "grab_and_move"
        }
        
        execution_results = {
            "execution_data": [
                {
                    "actual_position": [0.9, 1.1, 1.0],
                    "target_position": [1.0, 1.0, 1.0],
                    "success": True,
                    "execution_time": 2.5,
                    "error_count": 0
                },
                {
                    "actual_position": [1.1, 0.9, 1.1],
                    "target_position": [1.0, 1.0, 1.0],
                    "success": True,
                    "execution_time": 2.3,
                    "error_count": 1
                }
            ]
        }
        
        # 执行评估
        evaluation = evaluator.evaluate_transfer(adapted_strategy, execution_results)
        print(f"✓ 迁移评估完成")
        print(f"  总体评分: {evaluation['overall_score']:.2f}")
        print(f"  评估指标数: {len(evaluation['metrics'])}")
        
        # 获取评估摘要
        summary = evaluator.get_evaluation_summary()
        print(f"✓ 评估摘要: 总评估数 {summary['total_evaluations']}")
        
        # 测试新增功能：深入迁移质量分析
        quality_analysis = evaluator.analyze_transfer_quality(
            adapted_strategy, execution_results, 
            quality_dimensions=['precision', 'consistency', 'efficiency']
        )
        print(f"✓ 迁移质量分析完成")
        print(f"  综合质量分数: {quality_analysis['overall_quality_score']:.2f}")
        print(f"  质量等级: {quality_analysis['quality_grade']}")
        print(f"  发现质量问题: {len(quality_analysis['quality_issues'])} 个")
        
        # 测试新增功能：多策略对比
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
        
        # 测试新增功能：改进建议生成
        improvement_suggestions = evaluator.generate_improvement_suggestions(
            evaluation, suggestion_type="comprehensive"
        )
        print(f"✓ 改进建议生成完成")
        print(f"  建议类型: {improvement_suggestions['suggestion_type']}")
        print(f"  建议数量: {improvement_suggestions['recommendations_summary']['total_recommendations']}")
        print(f"  置信度: {improvement_suggestions['confidence_level']:.2f}")
        print(f"  当前性能: {improvement_suggestions['current_performance']['overall_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ TransferEvaluator 测试失败: {e}")
        return False


def test_adaptation_engine():
    """测试适应引擎"""
    print("\n=== 测试 AdaptationEngine ===")
    
    try:
        # 初始化适应引擎
        adapter = AdaptationEngine()
        print("✓ 适应引擎初始化成功")
        
        # 准备测试数据
        physical_strategy = {
            "mapped_strategy_id": "test_physical",
            "mapped_action_sequences": [
                {
                    "action_type": "grab",
                    "mapped_parameters": {
                        "target_position": {"x": 1.0, "y": 1.0, "z": 1.0}
                    }
                }
            ]
        }
        
        physical_environment = {
            "workspace_dimensions": {"width": 2.0, "height": 1.0, "depth": 2.0},
            "environmental_constraints": {
                "friction_coefficients": {"block_to_ground": 0.5}
            }
        }
        
        # 执行适应
        adaptation = adapter.adapt_strategy(physical_strategy, physical_environment)
        print(f"✓ 策略适应完成")
        print(f"  适应置信度: {adaptation['adaptation_confidence']:.2f}")
        print(f"  学习进度: {adaptation['learning_progress']:.2f}")
        
        # 报告性能数据
        strategy_id = physical_strategy["mapped_strategy_id"]
        adapter.report_performance(strategy_id, 0.85)
        print("✓ 性能数据报告成功")
        
        return True
        
    except Exception as e:
        print(f"✗ AdaptationEngine 测试失败: {e}")
        return False


def test_performance_analyzer():
    """测试性能分析器"""
    print("\n=== 测试 PerformanceAnalyzer ===")
    
    try:
        # 初始化性能分析器
        analyzer = PerformanceAnalyzer()
        print("✓ 性能分析器初始化成功")
        
        # 准备测试数据
        transfer_history = [
            {
                "session_id": "test_1",
                "metrics": {"accuracy": 0.85, "success_rate": 0.9},
                "overall_score": 0.87
            }
        ]
        
        evaluation_report = {
            "evaluation_id": "test_eval",
            "overall_score": 0.85,
            "metrics": {"accuracy": 0.85, "success_rate": 0.9}
        }
        
        current_metrics = {
            "accuracy": 0.85,
            "success_rate": 0.9,
            "execution_time": 2.5,
            "stability": 0.8
        }
        
        # 执行性能分析
        analysis = analyzer.analyze_performance(
            transfer_history, evaluation_report, current_metrics
        )
        print(f"✓ 性能分析完成")
        print(f"  优化置信度: {analysis['optimization_confidence']:.2f}")
        print(f"  分析方法数: {len(analysis['analysis_metadata']['analysis_methods_used'])}")
        
        # 生成性能报告
        report = analyzer.generate_performance_report()
        print(f"✓ 性能报告生成成功")
        print(f"  报告状态: {report['status']}")
        
        # 测试新增功能：性能趋势预测
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
        
        # 测试新增功能：系统瓶颈识别
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
        
        # 测试新增功能：资源分配优化
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
        
        return True
        
    except Exception as e:
        print(f"✗ PerformanceAnalyzer 测试失败: {e}")
        return False


def test_system_integration():
    """测试系统集成"""
    print("\n=== 测试系统集成 ===")
    
    try:
        # 初始化完整系统
        transfer_system = StrategyTransfer()
        print("✓ 完整系统初始化成功")
        
        # 创建测试会话
        session_id = transfer_system.start_transfer_session("integration_test")
        
        # 完整的迁移流程测试
        minecraft_data = {
            "scene_info": {"world_size": {"x": 10, "y": 5, "z": 10}},
            "action_sequences": [
                {
                    "action_type": "grab",
                    "position": [5, 2, 5],
                    "parameters": {"force": 0.8}
                }
            ],
            "performance_metrics": {"success_rate": 0.9, "execution_time": 2.0}
        }
        
        # 步骤1: 提取策略
        strategy = transfer_system.extract_minecraft_strategy(minecraft_data, session_id)
        
        # 步骤2: 映射策略
        physical_strategy = transfer_system.map_to_physical_world(strategy, session_id)
        
        # 步骤3: 适应策略
        physical_env = {
            "workspace_dimensions": {"width": 2.0, "height": 1.0, "depth": 2.0},
            "environmental_constraints": {"friction_coefficients": {"block_to_ground": 0.5}}
        }
        adapted_strategy = transfer_system.adapt_strategy(physical_strategy, physical_env, session_id)
        
        # 步骤4: 评估效果
        execution_results = {
            "execution_data": [
                {
                    "actual_position": [0.9, 1.1, 1.0],
                    "target_position": [1.0, 1.0, 1.0],
                    "success": True,
                    "execution_time": 2.5
                }
            ]
        }
        evaluation = transfer_system.evaluate_transfer(adapted_strategy, execution_results, session_id)
        
        # 步骤5: 优化性能
        optimization = transfer_system.optimize_transfer(evaluation, session_id)
        
        # 步骤6: 完成会话
        summary = transfer_system.complete_transfer_session(session_id)
        
        print("✓ 完整迁移流程测试成功")
        print(f"  总体评分: {evaluation['overall_score']:.2f}")
        print(f"  优化置信度: {optimization['confidence_score']:.2f}")
        
        # 获取系统统计
        stats = transfer_system.get_transfer_statistics()
        print(f"✓ 系统统计: 总会话数 {stats['total_sessions']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 系统集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("策略迁移系统功能测试")
    print("=" * 50)
    
    start_time = time.time()
    
    # 执行各项测试
    tests = [
        ("StrategyTransfer", test_strategy_transfer),
        ("KnowledgeMapper", test_knowledge_mapper),
        ("TransferEvaluator", test_transfer_evaluator),
        ("AdaptationEngine", test_adaptation_engine),
        ("PerformanceAnalyzer", test_performance_analyzer),
        ("系统集成", test_system_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    # 测试总结
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests:.1%}")
    print(f"测试耗时: {elapsed_time:.2f}秒")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过! 策略迁移系统运行正常。")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查相关组件。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)