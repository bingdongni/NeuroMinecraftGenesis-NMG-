"""
终身学习评估系统演示 - 简化版

该文件演示终身学习评估系统的核心功能，避免需要图形界面的可视化操作

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import json
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from atari_game_sequence import AtariGameSequence
from catastrophic_forgetting_analyzer import CatastrophicForgettingAnalyzer
from forward_backward_transfer import ForwardBackwardTransfer
from learning_curve_analyzer import LearningCurveAnalyzer


def demo_basic_system():
    """演示基本系统功能"""
    print("=" * 60)
    print("终身学习评估系统 - 基本功能演示")
    print("=" * 60)
    
    # 演示Atari游戏序列
    print("1. Atari游戏序列管理演示:")
    sequence = AtariGameSequence(total_games=10, steps_per_game=500)
    games_list = sequence.get_games_list()
    print(f"   总游戏数: {len(games_list)}")
    print(f"   前5个游戏: {games_list[:5]}")
    
    # 获取序列统计
    stats = sequence.get_sequence_statistics()
    print(f"   平均难度: {stats['average_difficulty']:.3f}")
    print(f"   平均复杂度: {stats['average_complexity']:.3f}")
    print(f"   技能分布: {dict(list(stats['skill_distribution'].items())[:3])}")
    
    # 保存序列信息
    sequence.save_sequence_info("demo_game_sequence.json")
    print(f"   序列信息已保存: demo_game_sequence.json")
    
    return sequence


def demo_catastrophic_forgetting():
    """演示灾难性遗忘分析"""
    print(f"\n2. 灾难性遗忘分析演示:")
    
    # 创建分析器
    analyzer = CatastrophicForgettingAnalyzer(threshold=0.05)
    
    # 模拟性能数据
    print("   模拟连续学习过程...")
    
    for task_id in range(8):
        # 模拟性能变化（逐渐下降表示遗忘）
        baseline_performance = 0.8 + task_id * 0.01
        current_performance = baseline_performance * np.random.uniform(0.85, 1.0)
        
        # 记录性能历史
        performance_scores = [baseline_performance * np.random.uniform(0.9, 1.1) for _ in range(20)]
        analyzer.track_performance(task_id, performance_scores)
        
        # 分析遗忘
        all_performances = {task_id: performance_scores}
        forgetting_rate = analyzer.analyze_forgetting(
            task_id, current_performance, all_performances
        )
        
        print(f"   任务 {task_id}: 基线 {baseline_performance:.3f}, "
              f"当前 {current_performance:.3f}, 遗忘率 {forgetting_rate:.3f}")
    
    # 生成报告
    print(f"\n   生成遗忘分析报告...")
    report = analyzer.generate_forgetting_report(7)
    
    print(f"   遗忘分析摘要:")
    summary = report['summary']
    print(f"     总遗忘事件: {summary['total_forgetting_events']}")
    print(f"     严重遗忘事件: {summary['severe_forgetting_events']}")
    print(f"     平均遗忘率: {summary['average_forgetting_rate']:.3f}")
    print(f"     超过阈值事件: {summary['current_threshold_exceeded']}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_forgetting_results.json")
    print(f"   结果已保存: demo_forgetting_results.json")


def demo_transfer_analysis():
    """演示迁移分析"""
    print(f"\n3. 前后迁移分析演示:")
    
    # 创建分析器
    analyzer = ForwardBackwardTransfer()
    
    # 模拟学习进度和迁移
    print("   模拟学习进度和迁移过程...")
    
    for task_id in range(8):
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
        
        print(f"   任务 {task_id}: 前向迁移 {forward_score:.3f}, "
              f"平均后向迁移 {avg_backward:.3f}")
    
    # 生成报告
    print(f"\n   生成迁移分析报告...")
    report = analyzer.generate_transfer_report(7)
    
    print(f"   迁移分析摘要:")
    summary = report['summary']
    if summary:
        print(f"     任务对总数: {summary['total_task_pairs']}")
        print(f"     正向前向迁移率: {summary['positive_forward_rate']:.2%}")
        print(f"     良好后向保持率: {summary['good_backward_rate']:.2%}")
        print(f"     平均前向迁移: {summary['average_forward_transfer']:.3f}")
        print(f"     平均后向迁移: {summary['average_backward_transfer']:.3f}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_transfer_results.json")
    print(f"   结果已保存: demo_transfer_results.json")


def demo_learning_curve_analysis():
    """演示学习曲线分析"""
    print(f"\n4. 学习曲线分析演示:")
    
    # 创建分析器
    analyzer = LearningCurveAnalyzer()
    
    # 模拟性能历史数据
    print("   生成不同类型的学习曲线...")
    
    performance_history = {}
    np.random.seed(42)  # 确保可重复性
    
    for task_id in range(6):
        curve_type = task_id % 3
        
        if curve_type == 0:  # 指数型
            base = 0.3
            rewards = [base * (1 - np.exp(-0.05 * i)) + np.random.normal(0, 0.05) for i in range(50)]
        elif curve_type == 1:  # 线性型
            rewards = [0.4 + 0.003 * i + np.random.normal(0, 0.03) for i in range(50)]
        else:  # 振荡型
            rewards = [0.6 + 0.2 * np.sin(0.15 * i) + np.random.normal(0, 0.08) for i in range(50)]
        
        performance_history[task_id] = rewards[:30]  # 限制长度
        curve_names = ["指数型", "线性型", "振荡型"]
        print(f"   任务 {task_id} ({curve_names[curve_type]}): 起始 {rewards[0]:.3f}, 结束 {rewards[-1]:.3f}")
    
    # 分析学习曲线
    print(f"\n   分析学习曲线...")
    report = analyzer.analyze_learning_curves(performance_history, 6)
    
    if 'summary_statistics' in report:
        summary = report['summary_statistics']
        print(f"   学习速度统计:")
        speed_stats = summary['learning_speed']
        print(f"     平均: {speed_stats['mean']:.4f}, 标准差: {speed_stats['std']:.4f}")
        
        print(f"   收敛率统计:")
        conv_stats = summary['convergence_rate']
        print(f"     平均: {conv_stats['mean']:.4f}, 标准差: {conv_stats['std']:.4f}")
        
        print(f"   最终性能统计:")
        perf_stats = summary['final_performance']
        print(f"     平均: {perf_stats['mean']:.4f}, 改进趋势: {perf_stats['improvement_trend']:.4f}")
    
    if 'curve_type_distribution' in report:
        print(f"   曲线类型分布:")
        for curve_type, count in report['curve_type_distribution'].items():
            print(f"     {curve_type}: {count} 个任务")
    
    # 保存结果（不生成图表）
    analyzer.save_analysis_results("demo_learning_curve_results.json")
    print(f"   结果已保存: demo_learning_curve_results.json")


def demo_comprehensive_analysis():
    """演示综合分析"""
    print(f"\n5. 综合分析演示:")
    
    print("   整合所有分析结果...")
    
    # 模拟综合指标
    comprehensive_metrics = {
        'learning_speed': 0.0245,
        'forgetting_control': 0.92,  # 遗忘控制率
        'transfer_effectiveness': 0.67,  # 迁移有效性
        'stability_score': 0.78,  # 稳定性分数
        'convergence_quality': 0.85,  # 收敛质量
        'meta_learning_ability': 0.43  # 元学习能力
    }
    
    print("   系统综合评估:")
    for metric, value in comprehensive_metrics.items():
        print(f"     {metric}: {value:.3f}")
    
    # 性能评级
    overall_score = np.mean(list(comprehensive_metrics.values()))
    print(f"\n   综合评分: {overall_score:.3f}")
    
    if overall_score >= 0.8:
        grade = "优秀 (A)"
    elif overall_score >= 0.7:
        grade = "良好 (B)"
    elif overall_score >= 0.6:
        grade = "中等 (C)"
    else:
        grade = "需要改进 (D)"
    
    print(f"   性能等级: {grade}")
    
    # 保存综合报告
    report = {
        'timestamp': '2025-11-13T16:30:00',
        'metrics': comprehensive_metrics,
        'overall_score': overall_score,
        'grade': grade,
        'recommendations': [
            "继续监控灾难性遗忘率",
            "优化元学习算法",
            "增加EWC正则化强度",
            "改进任务间迁移策略"
        ]
    }
    
    with open('demo_comprehensive_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"   综合报告已保存: demo_comprehensive_report.json")


def main():
    """主函数 - 运行简化演示"""
    print("终身学习评估系统演示（简化版）")
    print("本演示将展示系统的各项核心功能，但不包含图形可视化")
    
    try:
        # 1. 基本系统演示
        sequence = demo_basic_system()
        
        # 2. 灾难性遗忘分析演示
        demo_catastrophic_forgetting()
        
        # 3. 迁移分析演示
        demo_transfer_analysis()
        
        # 4. 学习曲线分析演示
        demo_learning_curve_analysis()
        
        # 5. 综合分析演示
        demo_comprehensive_analysis()
        
        print(f"\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print(f"演示了以下功能:")
        print(f"  ✓ Atari游戏序列管理和生成")
        print(f"  ✓ 灾难性遗忘分析和监控")
        print(f"  ✓ 前后迁移性能评估")
        print(f"  ✓ 学习曲线特征分析")
        print(f"  ✓ 综合性能评估")
        print(f"  ✓ 详细报告生成")
        
        print(f"\n生成的文件:")
        for filename in [
            'demo_game_sequence.json',
            'demo_forgetting_results.json',
            'demo_transfer_results.json',
            'demo_learning_curve_results.json',
            'demo_comprehensive_report.json'
        ]:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename}")
        
        print(f"\n要查看详细数据，请检查生成的JSON文件")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()