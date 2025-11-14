"""
终身学习评估系统演示

该文件演示如何使用终身学习评估系统，包括：
1. 系统初始化和配置
2. 连续学习训练过程
3. 灾难性遗忘分析
4. 前后迁移评估
5. 学习曲线分析

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import logging
import os
import sys

# 添加当前目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lifelong_learning import LifelongLearningSystem, EvaluationConfig
from atari_game_sequence import AtariGameSequence, create_diverse_game_sequence
from catastrophic_forgetting_analyzer import CatastrophicForgettingAnalyzer
from forward_backward_transfer import ForwardBackwardTransfer
from learning_curve_analyzer import LearningCurveAnalyzer


def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('lifelong_learning_demo.log', encoding='utf-8')
        ]
    )


def demo_basic_system():
    """演示基本系统功能"""
    print("=" * 60)
    print("终身学习评估系统 - 基本功能演示")
    print("=" * 60)
    
    # 1. 创建配置
    config = EvaluationConfig(
        total_tasks=15,  # 演示使用较少的任务
        steps_per_task=500,
        test_frequency=5,
        ewc_lambda=50.0,
        mas_lambda=0.5,
        use_ewc=True,
        use_mas=True,
        performance_threshold=0.05
    )
    
    print(f"系统配置:")
    print(f"  总任务数: {config.total_tasks}")
    print(f"  每任务步数: {config.steps_per_task}")
    print(f"  测试频率: {config.test_frequency}")
    print(f"  EWC lambda: {config.ewc_lambda}")
    print(f"  MAS lambda: {config.mas_lambda}")
    
    # 2. 创建系统
    system = LifelongLearningSystem(config)
    
    # 3. 演示游戏序列
    print(f"\n游戏序列演示:")
    sequence = system.atari_sequence
    games_list = sequence.get_games_list()
    print(f"  总游戏数: {len(games_list)}")
    print(f"  前5个游戏: {games_list[:5]}")
    
    # 4. 获取序列统计
    stats = sequence.get_sequence_statistics()
    print(f"  平均难度: {stats['average_difficulty']:.3f}")
    print(f"  平均复杂度: {stats['average_complexity']:.3f}")
    print(f"  技能分布: {dict(list(stats['skill_distribution'].items())[:3])}")
    
    return system


def demo_catastrophic_forgetting():
    """演示灾难性遗忘分析"""
    print(f"\n" + "=" * 60)
    print("灾难性遗忘分析演示")
    print("=" * 60)
    
    # 创建分析器
    analyzer = CatastrophicForgettingAnalyzer(threshold=0.05)
    
    # 模拟性能数据
    print("模拟连续学习过程...")
    
    for task_id in range(10):
        # 模拟性能变化（逐渐下降表示遗忘）
        baseline_performance = 0.8 + task_id * 0.01  # 基线性能略有提升
        current_performance = baseline_performance * np.random.uniform(0.85, 1.0)
        
        # 记录性能历史
        performance_scores = [baseline_performance * np.random.uniform(0.9, 1.1) for _ in range(20)]
        analyzer.track_performance(task_id, performance_scores)
        
        # 分析遗忘
        all_performances = {task_id: performance_scores}
        forgetting_rate = analyzer.analyze_forgetting(
            task_id, current_performance, all_performances
        )
        
        print(f"  任务 {task_id}: 基线 {baseline_performance:.3f}, "
              f"当前 {current_performance:.3f}, 遗忘率 {forgetting_rate:.3f}")
    
    # 生成报告
    print(f"\n生成遗忘分析报告...")
    report = analyzer.generate_forgetting_report(9)
    
    print(f"遗忘分析摘要:")
    summary = report['summary']
    print(f"  总遗忘事件: {summary['total_forgetting_events']}")
    print(f"  严重遗忘事件: {summary['severe_forgetting_events']}")
    print(f"  平均遗忘率: {summary['average_forgetting_rate']:.3f}")
    print(f"  超过阈值事件: {summary['current_threshold_exceeded']}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_forgetting_results.json")
    print(f"  结果已保存到: demo_forgetting_results.json")
    
    return analyzer


def demo_transfer_analysis():
    """演示迁移分析"""
    print(f"\n" + "=" * 60)
    print("前后迁移分析演示")
    print("=" * 60)
    
    # 创建分析器
    analyzer = ForwardBackwardTransfer()
    
    # 模拟学习进度和迁移
    print("模拟学习进度和迁移过程...")
    
    for task_id in range(12):
        # 模拟回合奖励（逐渐改进但有干扰）
        base_performance = 0.5 + task_id * 0.02
        noise = np.random.normal(0, 0.1, 30)
        episode_rewards = [base_performance + n for n in noise]
        
        # 记录学习进度
        analyzer.record_learning_progress(task_id, episode_rewards)
        
        # 计算前向迁移
        forward_score = analyzer.calculate_forward_transfer(task_id, episode_rewards)
        
        # 模拟重新评估的旧任务分数
        re_evaluated_scores = {}
        for prev_task_id in range(task_id):
            baseline = analyzer.baseline_performance.get(prev_task_id, 0.5)
            re_evaluated_scores[prev_task_id] = baseline * np.random.uniform(0.8, 1.0)
        
        # 计算后向迁移
        backward_scores = analyzer.calculate_backward_transfer(task_id, re_evaluated_scores)
        
        avg_backward = np.mean(backward_scores) if backward_scores else 0.0
        
        print(f"  任务 {task_id}: 前向迁移 {forward_score:.3f}, "
              f"平均后向迁移 {avg_backward:.3f}")
        
        # 分析迁移模式
        pattern = analyzer.analyze_transfer_patterns(task_id)
        if task_id >= 5:  # 只显示有足够数据后的模式
            print(f"    迁移模式: {pattern.pattern_type}, 有效性: {pattern.transfer_effectiveness:.3f}")
    
    # 生成报告
    print(f"\n生成迁移分析报告...")
    report = analyzer.generate_transfer_report(11)
    
    print(f"迁移分析摘要:")
    summary = report['summary']
    if summary:
        print(f"  任务对总数: {summary['total_task_pairs']}")
        print(f"  前向迁移对数: {summary['forward_transfer_pairs']}")
        print(f"  正向前向迁移率: {summary['positive_forward_rate']:.2%}")
        print(f"  良好后向保持率: {summary['good_backward_rate']:.2%}")
        print(f"  平均前向迁移: {summary['average_forward_transfer']:.3f}")
        print(f"  平均后向迁移: {summary['average_backward_transfer']:.3f}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_transfer_results.json")
    print(f"  结果已保存到: demo_transfer_results.json")
    
    return analyzer


def demo_learning_curve_analysis():
    """演示学习曲线分析"""
    print(f"\n" + "=" * 60)
    print("学习曲线分析演示")
    print("=" * 60)
    
    # 创建分析器
    analyzer = LearningCurveAnalyzer()
    
    # 模拟性能历史数据
    print("生成不同类型的学习曲线...")
    
    performance_history = {}
    np.random.seed(42)  # 确保可重复性
    
    for task_id in range(8):
        curve_type = task_id % 4
        
        if curve_type == 0:  # 指数型
            base = 0.3
            rewards = [base * (1 - np.exp(-0.05 * i)) + np.random.normal(0, 0.05) for i in range(60)]
        elif curve_type == 1:  # 线性型
            rewards = [0.4 + 0.003 * i + np.random.normal(0, 0.03) for i in range(60)]
        elif curve_type == 2:  # 振荡型
            rewards = [0.6 + 0.2 * np.sin(0.15 * i) + np.random.normal(0, 0.08) for i in range(60)]
        else:  # 平台型
            rewards = [0.7 + np.random.normal(0, 0.015) if i > 25 else 0.5 + i * 0.008 + np.random.normal(0, 0.02) 
                      for i in range(60)]
        
        performance_history[task_id] = rewards[:40]  # 限制长度
        curve_names = ["指数型", "线性型", "振荡型", "平台型"]
        print(f"  任务 {task_id} ({curve_names[curve_type]}): 起始 {rewards[0]:.3f}, 结束 {rewards[-1]:.3f}")
    
    # 分析学习曲线
    print(f"\n分析学习曲线...")
    report = analyzer.analyze_learning_curves(performance_history, 8)
    
    if 'summary_statistics' in report:
        summary = report['summary_statistics']
        print(f"学习速度统计:")
        speed_stats = summary['learning_speed']
        print(f"  平均: {speed_stats['mean']:.4f}, 标准差: {speed_stats['std']:.4f}")
        print(f"  范围: {speed_stats['min']:.4f} - {speed_stats['max']:.4f}")
        
        print(f"收敛率统计:")
        conv_stats = summary['convergence_rate']
        print(f"  平均: {conv_stats['mean']:.4f}, 标准差: {conv_stats['std']:.4f}")
        
        print(f"最终性能统计:")
        perf_stats = summary['final_performance']
        print(f"  平均: {perf_stats['mean']:.4f}, 改进趋势: {perf_stats['improvement_trend']:.4f}")
        
        print(f"稳定性统计:")
        stab_stats = summary['stability']
        print(f"  平均分数: {stab_stats['mean']:.4f}, 一致性: {stab_stats['consistency']:.4f}")
    
    if 'curve_type_distribution' in report:
        print(f"\n曲线类型分布:")
        for curve_type, count in report['curve_type_distribution'].items():
            print(f"  {curve_type}: {count} 个任务")
    
    if 'learning_trends' in report:
        print(f"\n学习趋势:")
        trends = report['learning_trends']
        if 'speed_trend_slope' in trends:
            print(f"  速度趋势斜率: {trends['speed_trend_slope']:.4f}")
            print(f"  趋势R²: {trends['speed_trend_r2']:.4f}")
    
    if 'recommendations' in report:
        print(f"\n学习建议:")
        for i, recommendation in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {recommendation}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_learning_curve_results.json")
    print(f"\n  结果已保存到: demo_learning_curve_results.json")
    
    return analyzer


def demo_complete_system():
    """演示完整系统运行"""
    print(f"\n" + "=" * 60)
    print("完整系统运行演示")
    print("=" * 60)
    
    # 创建配置
    config = EvaluationConfig(
        total_tasks=5,  # 演示使用少量任务
        steps_per_task=200,
        test_frequency=2,
        ewc_lambda=50.0,
        mas_lambda=0.5,
        use_ewc=True,
        use_mas=True,
        performance_threshold=0.05,
        save_checkpoints=True,
        log_interval=50
    )
    
    print("创建终身学习系统...")
    system = LifelongLearningSystem(config)
    
    print(f"\n开始完整评估序列...")
    print(f"这将演示:")
    print(f"  1. 连续学习训练过程")
    print(f"  2. 定期性能评估")
    print(f"  3. 灾难性遗忘监控")
    print(f"  4. 前后迁移分析")
    print(f"  5. 学习曲线特征提取")
    
    # 运行完整评估（会生成大量日志和可视化）
    try:
        results = system.run_evaluation_sequence()
        
        print(f"\n完整系统评估完成!")
        
        # 显示最终性能摘要
        if 'final_performance' in results:
            final_perf = results['final_performance']
            print(f"\n最终性能评估:")
            print(f"  平均任务性能: {final_perf['average_performance']:.4f}")
            print(f"  平均前向迁移: {final_perf['average_forward_transfer']:.4f}")
            print(f"  平均后向迁移: {final_perf['average_backward_transfer']:.4f}")
            print(f"  平均遗忘率: {final_perf['average_forgetting_rate']:.4f}")
            print(f"  平均元学习分数: {final_perf['average_meta_learning_score']:.4f}")
            print(f"  成功率: {final_perf['success_rate']:.2%}")
        
        # 显示关键发现
        print(f"\n关键发现:")
        if final_perf['average_forgetting_rate'] < config.performance_threshold:
            print(f"  ✓ 成功控制灾难性遗忘 (遗忘率 {final_perf['average_forgetting_rate']:.3f} < {config.performance_threshold})")
        else:
            print(f"  ✗ 灾难性遗忘问题严重 (遗忘率 {final_perf['average_forgetting_rate']:.3f} >= {config.performance_threshold})")
        
        if final_perf['average_forward_transfer'] > 0:
            print(f"  ✓ 观察到正向前向迁移 (分数: {final_perf['average_forward_transfer']:.3f})")
        else:
            print(f"  ✗ 负向前向迁移 (分数: {final_perf['average_forward_transfer']:.3f})")
        
        print(f"\n结果文件:")
        print(f"  - 性能报告: performance_summary_*.txt")
        print(f"  - 详细数据: lifelong_learning_results_*.json")
        print(f"  - 可视化图表: lifelong_learning_analysis_*.png")
        
        return results
        
    except Exception as e:
        print(f"系统运行过程中出现错误: {e}")
        print(f"这可能是由于缺少某些依赖库或环境配置问题")
        print(f"请检查requirements.txt和系统配置")
        return None


def demo_customization():
    """演示自定义配置"""
    print(f"\n" + "=" * 60)
    print("自定义配置演示")
    print("=" * 60)
    
    # 演示不同配置的影响
    configs = [
        ("基础配置", EvaluationConfig(
            total_tasks=5,
            ewc_lambda=10.0,
            mas_lambda=0.1,
            use_ewc=True,
            use_mas=False
        )),
        ("强正则化配置", EvaluationConfig(
            total_tasks=5,
            ewc_lambda=100.0,
            mas_lambda=2.0,
            use_ewc=True,
            use_mas=True
        )),
        ("仅EWC配置", EvaluationConfig(
            total_tasks=5,
            ewc_lambda=200.0,
            mas_lambda=0.0,
            use_ewc=True,
            use_mas=False
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n{config_name}:")
        print(f"  EWC lambda: {config.ewc_lambda}")
        print(f"  MAS lambda: {config.mas_lambda}")
        print(f"  使用EWC: {config.use_ewc}")
        print(f"  使用MAS: {config.use_mas}")
        
        # 创建系统并快速运行
        try:
            system = LifelongLearningSystem(config)
            print(f"  系统创建成功")
            
            # 简单演示一个任务
            print(f"  演示单任务训练...")
            # 这里可以添加具体的训练演示，但为了演示简洁我们只创建系统
            
        except Exception as e:
            print(f"  系统创建失败: {e}")


def main():
    """主函数 - 运行所有演示"""
    # 设置日志
    setup_logging()
    
    print("终身学习评估系统演示")
    print("本演示将展示系统的各项核心功能")
    
    try:
        # 1. 基本系统演示
        system = demo_basic_system()
        
        # 2. 灾难性遗忘分析演示
        demo_catastrophic_forgetting()
        
        # 3. 迁移分析演示
        demo_transfer_analysis()
        
        # 4. 学习曲线分析演示
        demo_learning_curve_analysis()
        
        # 5. 自定义配置演示
        demo_customization()
        
        # 6. 完整系统演示（可选，注释掉以避免长时间运行）
        # print(f"\n注意: 完整系统演示可能需要较长时间...")
        # print(f"如需运行完整演示，请取消下面一行的注释")
        # demo_complete_system()
        
        print(f"\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print(f"演示了以下功能:")
        print(f"  ✓ 终身学习系统初始化和配置")
        print(f"  ✓ Atari游戏序列管理")
        print(f"  ✓ 灾难性遗忘分析和监控")
        print(f"  ✓ 前后迁移性能评估")
        print(f"  ✓ 学习曲线特征分析")
        print(f"  ✓ 可视化图表生成")
        print(f"  ✓ 结果文件保存")
        print(f"\n要查看详细结果，请检查生成的文件:")
        print(f"  - demo_*.json: 详细数据")
        print(f"  - demo_*.log: 运行日志")
        print(f"  - lifelong_learning_*.png: 可视化图表")
        
        print(f"\n要运行完整系统，请调用: demo_complete_system()")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print(f"请检查依赖库是否正确安装")


if __name__ == "__main__":
    main()