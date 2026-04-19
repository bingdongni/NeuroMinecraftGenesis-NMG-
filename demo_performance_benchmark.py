#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准展示面板系统演示
Performance Benchmark System Demo

该脚本展示了如何使用性能基准展示面板系统的各项功能。

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from pathlib import Path

# 导入性能基准系统
from utils.visualization import (
    PerformanceBenchmark,
    global_benchmark,
    create_performance_benchmark,
    get_global_benchmark
)

def demo_performance_benchmark():
    """演示性能基准系统功能"""
    print("=" * 80)
    print("🚀 NeuroMinecraftGenesis - 性能基准展示面板系统演示")
    print("=" * 80)
    
    # 1. 创建性能基准实例
    print("\n📊 1. 创建性能基准系统实例...")
    config = {
        'update_interval': 30,
        'comparison_threshold': 0.1,
        'trend_analysis_window': 20,
        'export_formats': ['json', 'csv', 'html']
    }
    
    benchmark = create_performance_benchmark(config)
    print("✅ 性能基准系统初始化完成")
    
    # 2. 添加性能数据
    print("\n📈 2. 添加性能数据...")
    
    algorithms = ['DQN', 'PPO', 'DiscoRL', 'A3C', 'Rainbow', 'NeuroMinecraftGenesis']
    tasks = ['Atari Breakout', 'Minecraft Survival']
    
    for algorithm in algorithms:
        for task in tasks:
            metrics = generate_sample_metrics(algorithm, task)
            benchmark.add_performance_data(algorithm, task, metrics)
            print(f"   ✅ 添加数据: {algorithm} - {task}")
    
    # 3. 计算性能指标
    print("\n🧮 3. 计算性能指标...")
    
    for algorithm in algorithms:
        for task in tasks:
            metrics = benchmark.calculate_performance_metrics(algorithm, task)
            print(f"   ✅ 计算指标: {algorithm} - {task}")
            avg_reward = metrics.get('average_reward', 0)
            success_rate = metrics.get('success_rate', 0)
            overall_score = metrics.get('overall_score', 0)
            
            if isinstance(avg_reward, (int, float)):
                print(f"      - 平均奖励: {avg_reward:.1f}")
            else:
                print(f"      - 平均奖励: {avg_reward}")
                
            if isinstance(success_rate, (int, float)):
                print(f"      - 成功率: {success_rate*100:.1f}%")
            else:
                print(f"      - 成功率: {success_rate}")
                
            if isinstance(overall_score, (int, float)):
                print(f"      - 综合评分: {overall_score:.1f}")
            else:
                print(f"      - 综合评分: {overall_score}")
    
    # 4. 性能对比分析
    print("\n🔄 4. 执行性能对比分析...")
    
    # 与基线算法比较
    comparison = benchmark.compare_with_baselines(
        'NeuroMinecraftGenesis', 
        'Atari Breakout', 
        ['DQN', 'PPO', 'DiscoRL']
    )
    
    print("   ✅ 性能对比完成")
    print(f"   - 总体评估: {comparison.get('overall_assessment', {}).get('performance_level', 'unknown')}")
    print(f"   - 建议: {comparison.get('overall_assessment', {}).get('recommendation', 'N/A')}")
    
    # 5. 趋势分析
    print("\n📈 5. 执行趋势分析...")
    
    trend_analysis = benchmark.analyze_trends(
        'NeuroMinecraftGenesis',
        'Atari Breakout',
        15
    )
    
    print("   ✅ 趋势分析完成")
    if 'overall_trend' in trend_analysis:
        overall_trend = trend_analysis['overall_trend']
        print(f"   - 总体趋势: {overall_trend.get('overall_trend_type', 'unknown')}")
        print(f"   - 趋势强度: {overall_trend.get('trend_strength', 0):.2f}")
        print(f"   - 性能改善: {overall_trend.get('performance_improvement', 0)*100:.1f}%")
    
    # 6. 生成性能报告
    print("\n📄 6. 生成性能报告...")
    
    report_path = benchmark.generate_performance_report('NeuroMinecraftGenesis', 'html')
    print(f"   ✅ HTML报告已生成: {report_path}")
    
    json_report_path = benchmark.generate_performance_report('NeuroMinecraftGenesis', 'json')
    print(f"   ✅ JSON报告已生成: {json_report_path}")
    
    # 7. 导出基准数据
    print("\n💾 7. 导出基准数据...")
    
    csv_path = benchmark.export_benchmark_data('csv')
    print(f"   ✅ CSV数据已导出: {csv_path}")
    
    json_path = benchmark.export_benchmark_data('json')
    print(f"   ✅ JSON数据已导出: {json_path}")
    
    # 8. 获取性能总结
    print("\n📊 8. 获取性能总结...")
    
    summary = benchmark.get_performance_summary()
    print("   ✅ 性能总结:")
    print(f"   - 支持的算法: {len(summary.get('supported_baselines', {}))} 个")
    print(f"   - 当前算法: {len(summary.get('current_algorithms', []))} 个")
    print(f"   - 系统状态: {summary.get('system_status', 'unknown')}")
    
    # 显示实时指标
    real_time = summary.get('real_time_metrics', {})
    print("   - 实时性能指标:")
    for metric, value in real_time.items():
        print(f"     * {metric}: {value}")
    
    # 9. 更新实时指标
    print("\n🔄 9. 更新实时指标...")
    
    benchmark.update_real_time_metrics()
    print("   ✅ 实时指标已更新")
    
    # 10. 展示全局实例
    print("\n🌐 10. 使用全局实例...")
    
    global_instance = get_global_benchmark()
    print(f"   ✅ 全局实例获取成功: {type(global_instance).__name__}")
    
    # 使用便捷函数
    from utils.visualization import add_performance_data, calculate_performance_metrics
    
    add_performance_data('TestAlgorithm', 'TestTask', {'score': 85.5})
    print("   ✅ 使用便捷函数添加数据成功")
    
    print("\n🎉 演示完成！所有功能运行正常。")
    print("\n📋 生成的报告和文件:")
    
    # 列出生成的文件
    reports_dir = Path('reports')
    if reports_dir.exists():
        for file_path in reports_dir.glob('*'):
            print(f"   📄 {file_path}")
    
    print("\n💡 下一步操作建议:")
    print("   1. 查看生成的HTML报告")
    print("   2. 打开性能仪表板 HTML文件")
    print("   3. 分析性能数据和趋势")
    print("   4. 根据建议优化算法配置")
    
    return True

def generate_sample_metrics(algorithm: str, task: str) -> dict:
    """生成示例性能指标"""
    import random
    
    # 基础性能值
    base_rewards = {
        'DQN': 132.5, 'PPO': 145.2, 'DiscoRL': 128.7,
        'A3C': 138.9, 'Rainbow': 152.8, 'NeuroMinecraftGenesis': 156.3
    }
    
    # 算法特定调整
    algorithm_modifier = {
        'DQN': 0.85, 'PPO': 1.0, 'DiscoRL': 0.9,
        'A3C': 0.95, 'Rainbow': 1.05, 'NeuroMinecraftGenesis': 1.08
    }
    
    # 任务特定调整
    task_modifier = {'Atari Breakout': 1.2, 'Minecraft Survival': 1.1}
    
    base_reward = base_rewards.get(algorithm, 100)
    alg_mod = algorithm_modifier.get(algorithm, 1.0)
    task_mod = task_modifier.get(task, 1.0)
    
    # 添加随机波动
    reward = base_reward * alg_mod * task_mod + random.uniform(-10, 10)
    
    metrics = {
        'average_reward': reward,
        'success_rate': min(1.0, max(0.0, 0.5 + random.uniform(0, 0.5))),
        'exploration_efficiency': min(1.0, max(0.0, 0.6 + random.uniform(0, 0.4))),
        'learning_stability': min(1.0, max(0.0, 0.7 + random.uniform(0, 0.3))),
        'convergence_speed': min(1.0, max(0.0, 0.6 + random.uniform(0, 0.4))),
        'overall_score': min(100, max(0, reward / 2))
    }
    
    # 任务特定指标
    if task == 'Atari Breakout':
        metrics['breakout_score'] = int(reward * 5)
    elif task == 'Minecraft Survival':
        metrics['survival_rate'] = 1.0
    
    return metrics

def show_system_info():
    """显示系统信息"""
    print("\n🔧 系统配置信息:")
    print("   - Python版本:", sys.version.split()[0])
    print("   - 工作目录:", os.getcwd())
    
    # 检查依赖
    try:
        import numpy as np
        print("   - NumPy版本:", np.__version__)
    except ImportError:
        print("   - NumPy: 未安装")
    
    try:
        import pandas as pd
        print("   - Pandas版本:", pd.__version__)
    except ImportError:
        print("   - Pandas: 未安装")
    
    try:
        import matplotlib
        print("   - Matplotlib版本:", matplotlib.__version__)
    except ImportError:
        print("   - Matplotlib: 未安装")
    
    print("   - 支持的基线算法: DQN, PPO, DiscoRL, A3C, Rainbow")
    print("   - 支持的任务: Atari Breakout, Minecraft Survival")

def main():
    """主函数"""
    show_system_info()
    
    try:
        success = demo_performance_benchmark()
        if success:
            print("\n✅ 演示成功完成！")
            return 0
        else:
            print("\n❌ 演示过程中出现错误")
            return 1
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())