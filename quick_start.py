#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化系统快速使用指南

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def quick_demo():
    """快速演示"""
    print("🚀 进化系统快速演示")
    print("="*50)
    
    try:
        # 导入组件
        from core.evolution import EvolutionVisualizer, CheckpointManager
        from utils.visualization import EvolutionDashboard
        
        # 创建演示目录
        os.makedirs("quick_demo/data/evolution_logs", exist_ok=True)
        os.makedirs("quick_demo/models/genomes/history", exist_ok=True)
        os.makedirs("quick_demo/models/genomes/best", exist_ok=True)
        
        print("📊 创建进化组件...")
        
        # 创建可视化器
        visualizer = EvolutionVisualizer(
            population_size=30,
            genome_length=6,
            data_dir="quick_demo/data/evolution_logs",
            checkpoint_dir="quick_demo/models/genomes"
        )
        
        # 创建检查点管理器
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="quick_demo/models/genomes",
            auto_save_interval=3
        )
        
        print("🧬 运行模拟进化...")
        
        # 模拟进化过程
        for gen in range(8):
            # 生成种群和适应度
            population = [np.random.randn(6) for _ in range(30)]
            fitness_scores = [np.sum(ind**2) + np.random.normal(0, 0.2) for ind in population]
            
            # 更新可视化器
            visualizer.update_population_state(population, fitness_scores, gen)
            
            # 保存检查点
            if gen % 4 == 0:
                checkpoint_info = checkpoint_manager.save_checkpoint(
                    population, fitness_scores, gen, 
                    checkpoint_type="auto",
                    description=f"快速演示_第{gen}代"
                )
                print(f"  ✅ 代数 {gen}: 保存检查点 (适应度: {fitness_scores[0]:.3f})")
            
            # 生成可视化
            if gen % 4 == 0:
                visualizer.visualize_evolution_progress()
                print(f"  📈 代数 {gen}: 生成可视化")
        
        print("\n🔄 测试断点续跑功能...")
        
        # 保存最佳检查点
        best_checkpoint = checkpoint_manager.save_checkpoint(
            population, fitness_scores, 7,
            checkpoint_type="best",
            description="演示_最佳个体"
        )
        print(f"  ⭐ 保存最佳个体: Gen 7")
        
        # 尝试恢复
        load_result = checkpoint_manager.load_checkpoint(7)
        if load_result:
            print(f"  🔄 恢复成功: 从第 {load_result['state']['generation']} 代继续")
            print(f"     恢复适应度: {load_result['state']['best_fitness']:.4f}")
        else:
            print("  ⚠️  恢复失败")
        
        print("\n📱 生成仪表板...")
        
        # 创建仪表板
        dashboard = EvolutionDashboard(
            data_dir="quick_demo/data/evolution_logs",
            auto_reload=False
        )
        
        # 生成静态仪表板
        dashboard_path = "quick_demo/final_dashboard.png"
        dashboard.create_static_dashboard(dashboard_path, include_analysis=True)
        print(f"  📊 仪表板生成: {dashboard_path}")
        
        print("\n📋 生成进化报告...")
        
        # 获取进化摘要
        summary = visualizer.get_evolution_summary()
        summary_path = "quick_demo/evolution_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"  📄 进化摘要: {summary_path}")
        
        print("\n🎉 快速演示完成！")
        print("\n📁 生成的文件:")
        print("  - 仪表板图片: quick_demo/final_dashboard.png")
        print("  - 进化摘要: quick_demo/evolution_summary.json")
        print("  - 检查点目录: quick_demo/models/genomes/")
        print("  - 进化数据: quick_demo/data/evolution_logs/")
        
        # 显示摘要信息
        print(f"\n📊 演示统计:")
        print(f"  - 总代数: {summary.get('current_generation', 0)}")
        print(f"  - 最佳适应度: {summary.get('overall_best_fitness', 0):.4f}")
        print(f"  - 进化改善率: {summary.get('evolution_progress', {}).get('improvement_rate', 0):.6f}/代")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n📚 使用示例")
    print("="*50)
    
    examples = {
        "基础使用": """
from core.evolution import EvolutionVisualizer, CheckpointManager

# 创建可视化器
visualizer = EvolutionVisualizer(
    population_size=100,
    genome_length=20,
    data_dir="data/evolution_logs"
)

# 更新状态
for generation in range(50):
    population, fitness_scores = evolve_population()
    visualizer.update_population_state(population, fitness_scores, generation)
    
    if generation % 10 == 0:
        visualizer.visualize_evolution_progress()
""",
        
        "断点续跑": """
from core.evolution import CheckpointManager

# 保存检查点
checkpoint_manager = CheckpointManager()
checkpoint_manager.save_checkpoint(population, fitness_scores, generation)

# 恢复进化
load_result = checkpoint_manager.load_checkpoint()
if load_result:
    state = load_result['state']
    generation = state['generation']
    # ... 继续进化
""",
        
        "实时监控": """
from utils.visualization import EvolutionDashboard

# 创建仪表板
dashboard = EvolutionDashboard(
    data_dir="data/evolution_logs",
    update_interval=2.0
)

# 启动实时监控
dashboard.start_dashboard(show_live_updates=True)

# 或生成静态报告
dashboard.create_static_dashboard("evolution_report.png")
"""
    }
    
    for title, code in examples.items():
        print(f"\n🔹 {title}:")
        print(code)

def main():
    """主函数"""
    print("NeuroMinecraftGenesis - 进化系统快速使用")
    print("🎯 完整实现了进化可视化和断点续跑功能")
    
    # 显示功能列表
    features = [
        "✅ 实时进化曲线可视化",
        "✅ 适应度地形3D展示",
        "✅ 遗传多样性变化监控", 
        "✅ 种群进化历史记录",
        "✅ 断点保存和恢复",
        "✅ 实时状态监控仪表板",
        "✅ 详细的进化分析报告"
    ]
    
    print("\n🚀 已实现功能:")
    for feature in features:
        print(f"  {feature}")
    
    # 询问是否运行演示
    print("\n" + "="*50)
    choice = input("是否运行快速演示？(Y/n): ").strip().lower()
    
    if choice != 'n':
        success = quick_demo()
        if success:
            print("\n🎉 演示成功完成！")
            show_usage_examples()
        else:
            print("\n❌ 演示失败，请检查错误信息")
    else:
        print("\n💡 你可以运行以下命令进行测试:")
        print("  python core/evolution/evolution_demo.py    # 完整演示")
        print("  python test_evolution_system.py           # 基础测试")
    
    print("\n📖 更多信息请查看:")
    print("  - README_evolution_system.md     # 详细文档")
    print("  - evolution_system_completion_report.md  # 完成报告")

if __name__ == "__main__":
    main()