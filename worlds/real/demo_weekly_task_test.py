#!/usr/bin/env python3
"""
每周真实世界任务测试系统演示脚本
================================

该脚本演示如何使用每周真实世界任务测试系统的各个组件，
包括创建测试套件、执行任务、记录性能、分析趋势和生成报告。

作者: NeuroMinecraftGenesis
版本: 1.0.0
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入测试系统组件
from worlds.real import (
    WeeklyTaskTest, 
    TaskScheduler, 
    PerformanceRecorder, 
    TrendAnalyzer, 
    ReportGenerator,
    create_weekly_test_system,
    run_weekly_test_now
)


def demo_basic_functionality():
    """演示基本功能"""
    print("=" * 60)
    print("每周真实世界任务测试系统演示")
    print("=" * 60)
    
    # 1. 创建并配置测试系统
    print("\n1. 创建测试系统...")
    test_system = create_weekly_test_system()
    
    # 获取系统状态
    status = test_system.get_test_status()
    print(f"   系统状态: {status['system_status']}")
    print(f"   测试套件: {status['test_suite']['name']}")
    
    # 2. 执行一次完整的测试
    print("\n2. 执行测试套件...")
    result = test_system.execute_test_suite()
    
    if result['success']:
        print(f"   ✅ 测试执行成功")
        print(f"   📊 执行任务数: {result['tasks_executed']}/{result['total_tasks']}")
        print(f"   ⏱️  执行时间: {result['duration_seconds']:.2f}秒")
        print(f"   🎯 成功率: {result['statistics']['success_rate']:.1%}")
        print(f"   📈 平均分数: {result['statistics']['average_score']:.3f}")
    else:
        print(f"   ❌ 测试执行失败: {result.get('error', '未知错误')}")
    
    # 3. 分析性能趋势
    print("\n3. 分析性能趋势...")
    trend_analysis = test_system.analyze_trends()
    
    if 'error' not in trend_analysis:
        summary = trend_analysis.get('analysis_summary', {})
        print(f"   📋 分析智能体数: {summary.get('total_agents_analyzed', 0)}")
        print(f"   📊 平均趋势强度: {summary.get('average_trend_strength', 0):.3f}")
        print(f"   🔮 预测置信度: {summary.get('average_prediction_confidence', 0):.3f}")
        print(f"   ⚠️  检测异常数: {summary.get('total_anomalies_detected', 0)}")
    else:
        print(f"   ⚠️  趋势分析失败: {trend_analysis['error']}")
    
    # 4. 生成报告
    print("\n4. 生成测试报告...")
    report_path = result.get('report_path', '')
    if report_path and os.path.exists(report_path):
        print(f"   📄 报告已生成: {report_path}")
    else:
        print(f"   ❌ 报告生成失败")
    
    return test_system


def demo_task_scheduler():
    """演示任务调度器功能"""
    print("\n" + "=" * 60)
    print("任务调度器演示")
    print("=" * 60)
    
    # 创建调度器
    scheduler = TaskScheduler()
    
    # 启动调度器
    scheduler.start_scheduler()
    print("✅ 任务调度器已启动")
    
    # 创建和调度任务
    from worlds.real.task_scheduler import create_scheduled_task, TaskPriority
    
    task = create_scheduled_task(
        task_name="导航测试任务",
        task_type="navigation_task",
        agent_id="agent_demo_001",
        priority=TaskPriority.HIGH
    )
    
    success = scheduler.schedule_task(task)
    print(f"📋 任务调度{'成功' if success else '失败'}: {task.task_name}")
    
    # 等待任务执行
    print("⏳ 等待任务执行...")
    time.sleep(3)
    
    # 获取调度器状态
    scheduler_status = scheduler.get_scheduler_status()
    print(f"📊 调度器状态:")
    print(f"   - 运行状态: {'运行中' if scheduler_status['is_running'] else '已停止'}")
    print(f"   - 待执行任务: {scheduler_status['pending_tasks']}")
    print(f"   - 运行中任务: {scheduler_status['running_tasks']}")
    print(f"   - 已完成任务: {scheduler_status['completed_tasks']}")
    print(f"   - 失败任务: {scheduler_status['failed_tasks']}")
    
    # 关闭调度器
    scheduler.shutdown()
    print("🛑 任务调度器已关闭")
    
    return scheduler


def demo_performance_recorder():
    """演示性能记录器功能"""
    print("\n" + "=" * 60)
    print("性能记录器演示")
    print("=" * 60)
    
    # 创建性能记录器
    recorder = PerformanceRecorder()
    print("✅ 性能记录器已创建")
    
    # 创建模拟性能记录
    from worlds.real.performance_recorder import create_performance_record
    
    records = []
    for i in range(5):
        record = create_performance_record(
            agent_id=f"agent_demo_{i:03d}",
            task_id=f"task_demo_{i:03d}",
            task_type="test_task",
            success=True if i < 4 else False,  # 前4个成功，最后1个失败
            score=0.5 + 0.1 * i,
            execution_time=30.0 + 10.0 * i,
            accuracy=0.7 + 0.05 * i,
            efficiency=0.6 + 0.08 * i
        )
        records.append(record)
        
        # 记录性能数据
        success = recorder.record_task_result(record)
        print(f"📊 记录{'成功' if success else '失败'}: 智能体 {record.agent_id}, 分数 {record.score}")
    
    # 获取性能摘要
    summary = recorder.get_performance_summary(days=1)
    print(f"\n📈 性能摘要:")
    print(f"   - 统计周期: {summary['period']}")
    print(f"   - 总任务数: {summary['total_tasks']}")
    print(f"   - 成功率: {summary['overall_success_rate']:.1%}")
    print(f"   - 平均分数: {summary['average_score']:.3f}")
    print(f"   - 平均执行时间: {summary['average_execution_time']:.1f}秒")
    
    # 获取统计信息
    stats = recorder.get_stats()
    print(f"\n📊 记录器统计:")
    print(f"   - 总记录数: {stats['total_records']}")
    print(f"   - 成功任务: {stats['successful_tasks']}")
    print(f"   - 失败任务: {stats['failed_tasks']}")
    print(f"   - 平均执行时间: {stats['average_execution_time']:.2f}秒")
    
    # 关闭记录器
    recorder.close()
    print("🛑 性能记录器已关闭")
    
    return recorder


def demo_trend_analyzer():
    """演示趋势分析器功能"""
    print("\n" + "=" * 60)
    print("趋势分析器演示")
    print("=" * 60)
    
    # 创建趋势分析器
    analyzer = TrendAnalyzer()
    print("✅ 趋势分析器已创建")
    
    # 创建模拟测试结果数据
    mock_results = []
    base_time = datetime.now() - timedelta(days=7)
    
    for day in range(7):
        for agent_id in ['agent_001', 'agent_002']:
            # 模拟每天每个智能体的测试结果
            result = type('MockResult', (), {
                'test_id': f'test_{day}_{agent_id}',
                'agent_id': agent_id,
                'task_name': 'navigation_task',
                'start_time': base_time + timedelta(days=day, hours=9),
                'end_time': base_time + timedelta(days=day, hours=10),
                'success': True,
                'score': 0.5 + 0.1 * day + 0.05 * (agent_id == 'agent_002'),  # agent_002表现更好
                'performance_metrics': {
                    'accuracy': 0.6 + 0.05 * day,
                    'efficiency': 0.5 + 0.08 * day
                },
                'resource_usage': {
                    'cpu_usage': 40 + 5 * day,
                    'memory_usage': 50 + 3 * day
                },
                'error_message': None
            })()
            
            mock_results.append(result)
    
    print(f"📊 创建了 {len(mock_results)} 个模拟测试结果")
    
    # 执行趋势分析
    print("\n🔍 执行性能趋势分析...")
    analysis_result = analyzer.analyze_performance_trends(mock_results)
    
    if 'error' not in analysis_result:
        summary = analysis_result.get('analysis_summary', {})
        print(f"✅ 趋势分析完成:")
        print(f"   - 分析智能体数: {summary.get('total_agents_analyzed', 0)}")
        print(f"   - 平均趋势强度: {summary.get('average_trend_strength', 0):.3f}")
        print(f"   - 预测置信度: {summary.get('average_prediction_confidence', 0):.3f}")
        print(f"   - 检测异常数: {summary.get('total_anomalies_detected', 0)}")
        print(f"   - 检测模式数: {summary.get('patterns_detected', 0)}")
        
        # 显示趋势分布
        trend_dist = analysis_result.get('trend_distribution', {})
        if trend_dist:
            print(f"\n📈 趋势分布:")
            for trend, count in trend_dist.items():
                print(f"   - {trend}: {count} 个智能体")
        
        # 显示整体建议
        recommendations = analysis_result.get('overall_recommendations', [])
        if recommendations:
            print(f"\n💡 整体建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    else:
        print(f"❌ 趋势分析失败: {analysis_result['error']}")
    
    return analyzer


def demo_report_generator():
    """演示报告生成器功能"""
    print("\n" + "=" * 60)
    print("报告生成器演示")
    print("=" * 60)
    
    # 创建报告生成器
    generator = ReportGenerator()
    print("✅ 报告生成器已创建")
    
    # 创建模拟报告数据
    mock_report_data = {
        'report_title': '2024年第46周测试报告',
        'test_period': {
            'start': '2024-11-11',
            'end': '2024-11-17'
        },
        'statistics': {
            'total_tests': 120,
            'successful_tests': 102,
            'failed_tests': 18,
            'success_rate': 0.85,
            'average_score': 0.78,
            'best_score': 0.95,
            'worst_score': 0.42,
            'average_execution_time': 145.3
        },
        'test_results': [
            {
                'test_id': 'test_001',
                'agent_id': 'agent_001',
                'task_name': 'navigation_task',
                'success': True,
                'score': 0.85,
                'start_time': '2024-11-17T09:00:00',
                'end_time': '2024-11-17T09:02:30',
                'performance_metrics': {
                    'accuracy': 0.88,
                    'efficiency': 0.82
                }
            }
            # 可以添加更多测试结果
        ],
        'trend_analysis': {
            'analysis_summary': {
                'total_agents_analyzed': 2,
                'average_trend_strength': 0.75,
                'average_prediction_confidence': 0.82,
                'total_anomalies_detected': 3,
                'patterns_detected': 1
            },
            'trend_distribution': {
                'increasing': 1,
                'stable': 1,
                'decreasing': 0
            }
        },
        'recommendations': [
            '继续优化智能体的导航算法',
            '加强对异常情况的处理能力',
            '提升整体执行效率'
        ]
    }
    
    # 生成报告文件路径
    report_dir = '/tmp/weekly_test_reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'weekly_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    
    # 生成周报
    print("\n📄 生成周报...")
    generated_path = generator.generate_weekly_report(mock_report_data, report_path)
    
    if generated_path and os.path.exists(generated_path):
        print(f"✅ 周报生成成功: {generated_path}")
        print(f"   文件大小: {os.path.getsize(generated_path)} 字节")
    else:
        print(f"❌ 周报生成失败")
    
    # 生成性能仪表板
    dashboard_path = os.path.join(report_dir, f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    print("\n📊 生成性能仪表板...")
    dashboard_generated = generator.generate_performance_dashboard(mock_report_data, dashboard_path)
    
    if dashboard_generated:
        print(f"✅ 仪表板生成成功: {dashboard_generated}")
    else:
        print(f"❌ 仪表板生成失败")
    
    # 清理临时文件
    generator.cleanup_temp_files()
    print("🧹 临时文件已清理")
    
    return generator


def demo_integration():
    """演示完整集成"""
    print("\n" + "=" * 60)
    print("完整集成演示")
    print("=" * 60)
    
    # 创建完整的测试系统
    test_system = create_weekly_test_system()
    
    print("🚀 执行完整的每周测试流程...")
    
    # 执行测试
    result = test_system.execute_test_suite()
    
    if result['success']:
        print("✅ 测试执行完成")
        
        # 分析趋势
        trend_analysis = test_system.analyze_trends()
        
        # 生成报告
        start_time = datetime.fromisoformat(result['start_time'])
        end_time = datetime.fromisoformat(result['end_time'])
        report_path = test_system.generate_weekly_report(start_time, end_time)
        
        print(f"\n📊 测试结果摘要:")
        print(f"   - 执行任务: {result['tasks_executed']}/{result['total_tasks']}")
        print(f"   - 成功率: {result['statistics']['success_rate']:.1%}")
        print(f"   - 平均分数: {result['statistics']['average_score']:.3f}")
        print(f"   - 执行时间: {result['duration_seconds']:.2f}秒")
        
        print(f"\n📈 趋势分析摘要:")
        if 'error' not in trend_analysis:
            summary = trend_analysis.get('analysis_summary', {})
            print(f"   - 分析智能体: {summary.get('total_agents_analyzed', 0)}个")
            print(f"   - 平均趋势强度: {summary.get('average_trend_strength', 0):.3f}")
            print(f"   - 检测异常: {summary.get('total_anomalies_detected', 0)}个")
        else:
            print(f"   - 趋势分析失败: {trend_analysis['error']}")
        
        print(f"\n📄 报告生成:")
        if report_path and os.path.exists(report_path):
            print(f"   - 周报路径: {report_path}")
            print(f"   - 报告大小: {os.path.getsize(report_path)} 字节")
        else:
            print(f"   - 报告生成失败")
        
    else:
        print(f"❌ 测试执行失败: {result.get('error', '未知错误')}")
    
    # 获取最终系统状态
    final_status = test_system.get_test_status()
    print(f"\n🏁 系统最终状态:")
    print(f"   - 系统状态: {final_status['system_status']}")
    print(f"   - 调度器状态: {final_status['scheduler_status']}")
    print(f"   - 历史测试数: {final_status['total_historical_tests']}")
    
    # 清理系统
    test_system.cleanup()
    print("🧹 系统已清理")
    
    return test_system


def main():
    """主演示函数"""
    print("🎯 每周真实世界任务测试系统演示")
    print("=" * 60)
    
    try:
        # 1. 基本功能演示
        test_system = demo_basic_functionality()
        
        # 2. 各组件演示
        scheduler = demo_task_scheduler()
        recorder = demo_performance_recorder()
        analyzer = demo_trend_analyzer()
        generator = demo_report_generator()
        
        # 3. 完整集成演示
        final_system = demo_integration()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)
        print("\n📋 演示总结:")
        print("✅ 每周任务测试系统 - 核心功能和API")
        print("✅ 任务调度器 - 定时任务和优先级管理") 
        print("✅ 性能记录器 - 数据收集和存储")
        print("✅ 趋势分析器 - 性能分析和预测")
        print("✅ 报告生成器 - 可视化报告和数据导出")
        print("✅ 完整集成 - 端到端测试流程")
        
        print("\n💡 提示:")
        print("- 所有组件都支持自定义配置")
        print("- 生成的报告保存在 /tmp/ 目录下")
        print("- 可以通过配置文件调整所有参数")
        print("- 系统支持多智能体并发测试")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()