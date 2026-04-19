#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化系统简化测试
"""

import os
import sys
import numpy as np
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_basic_functionality():
    """测试基本功能"""
    print("测试进化系统基本功能...")
    
    # 创建必要的目录
    os.makedirs("test_data/evolution_logs", exist_ok=True)
    os.makedirs("test_models/genomes/history", exist_ok=True)
    
    # 测试 EvolutionVisualizer
    print("1. 测试 EvolutionVisualizer...")
    try:
        from core.evolution import EvolutionVisualizer
        
        visualizer = EvolutionVisualizer(
            population_size=10,
            genome_length=5,
            data_dir="test_data/evolution_logs",
            checkpoint_dir="test_models/genomes"
        )
        
        # 模拟几代数据
        for gen in range(3):
            population = [np.random.randn(5) for _ in range(10)]
            fitness_scores = [np.sum(ind**2) for ind in population]
            visualizer.update_population_state(population, fitness_scores, gen)
        
        print("  ✓ EvolutionVisualizer 创建和更新成功")
        
    except Exception as e:
        print(f"  ✗ EvolutionVisualizer 失败: {e}")
        return False
    
    # 测试 CheckpointManager
    print("2. 测试 CheckpointManager...")
    try:
        from core.evolution import CheckpointManager
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="test_models/genomes",
            auto_save_interval=2
        )
        
        # 保存检查点
        population = [np.random.randn(5) for _ in range(10)]
        fitness_scores = [np.sum(ind**2) for ind in population]
        
        checkpoint_info = checkpoint_manager.save_checkpoint(
            population, fitness_scores, 0, checkpoint_type="manual"
        )
        
        print(f"  ✓ 检查点保存成功: {checkpoint_info['generation']}")
        
        # 测试加载
        load_result = checkpoint_manager.load_checkpoint()
        if load_result:
            print(f"  ✓ 检查点加载成功: Gen {load_result['state']['generation']}")
        else:
            print("  ! 检查点加载返回None")
        
    except Exception as e:
        print(f"  ✗ CheckpointManager 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 EvolutionDashboard
    print("3. 测试 EvolutionDashboard...")
    try:
        from utils.visualization import EvolutionDashboard
        
        dashboard = EvolutionDashboard(
            data_dir="test_data/evolution_logs",
            auto_reload=False
        )
        
        # 创建静态仪表板
        output_path = "test_data/evolution_logs/test_dashboard.png"
        dashboard.create_static_dashboard(output_path, include_analysis=False)
        
        print(f"  ✓ 仪表板创建成功: {output_path}")
        
    except Exception as e:
        print(f"  ✗ EvolutionDashboard 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def create_simple_demo():
    """创建一个简单的演示"""
    print("\n创建简单演示...")
    
    try:
        from core.evolution import EvolutionVisualizer, CheckpointManager
        from utils.visualization import EvolutionDashboard
        
        # 创建组件
        visualizer = EvolutionVisualizer(
            population_size=20,
            genome_length=8,
            data_dir="demo_data/evolution_logs",
            checkpoint_dir="demo_models/genomes"
        )
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="demo_models/genomes",
            auto_save_interval=5
        )
        
        dashboard = EvolutionDashboard(
            data_dir="demo_data/evolution_logs"
        )
        
        # 模拟进化过程
        print("  运行模拟进化过程...")
        for gen in range(10):
            # 生成种群和适应度
            population = [np.random.randn(8) for _ in range(20)]
            fitness_scores = [np.sum(ind**2) + np.random.normal(0, 0.1) for ind in population]
            
            # 更新组件
            visualizer.update_population_state(population, fitness_scores, gen)
            
            # 保存检查点
            if gen % 5 == 0:
                checkpoint_info = checkpoint_manager.save_checkpoint(
                    population, fitness_scores, gen, checkpoint_type="auto"
                )
                print(f"    保存检查点: Gen {gen}")
            
            # 每3代生成可视化
            if gen % 3 == 0:
                visualizer.visualize_evolution_progress()
                print(f"    生成进度图: Gen {gen}")
        
        # 生成最终报告
        print("  生成最终报告...")
        
        # 静态仪表板
        dashboard_path = "demo_data/evolution_logs/final_dashboard.png"
        dashboard.create_static_dashboard(dashboard_path, include_analysis=True)
        print(f"    仪表板: {dashboard_path}")
        
        # 进化摘要
        summary = visualizer.get_evolution_summary()
        summary_path = "demo_data/evolution_logs/evolution_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"    摘要: {summary_path}")
        
        # 导出最佳个体
        if population:
            export_path = "demo_models/genomes/best_individual_export.json"
            checkpoint_manager.export_checkpoint(9, export_path)
            print(f"    最佳个体: {export_path}")
        
        print("\n✅ 演示完成！生成的文件:")
        print(f"  - 仪表板图片: {dashboard_path}")
        print(f"  - 进化摘要: {summary_path}")
        print(f"  - 最佳个体: {export_path}")
        print(f"  - 检查点目录: demo_models/genomes/")
        print(f"  - 进化日志: demo_data/evolution_logs/")
        
        return True
        
    except Exception as e:
        print(f"✗ 演示创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("NeuroMinecraftGenesis - 进化系统测试")
    print("="*50)
    
    # 运行基本功能测试
    if test_basic_functionality():
        print("\n✅ 基本功能测试通过")
    else:
        print("\n❌ 基本功能测试失败")
        return
    
    # 创建演示
    if create_simple_demo():
        print("\n✅ 演示创建成功")
    else:
        print("\n❌ 演示创建失败")
    
    print("\n" + "="*50)
    print("测试完成！")
    print("检查 demo_data/ 和 demo_models/ 目录查看生成的文件")

if __name__ == "__main__":
    main()