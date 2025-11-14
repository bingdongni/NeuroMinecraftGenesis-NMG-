#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化系统基础功能测试

验证核心组件是否正常工作

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import os
import sys
import numpy as np
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_evolution_visualizer():
    """测试进化可视化器"""
    print("测试 EvolutionVisualizer...")
    
    try:
        from core.evolution import EvolutionVisualizer
        
        # 创建可视化器
        visualizer = EvolutionVisualizer(
            population_size=20,
            genome_length=5,
            data_dir="test_data/evolution_logs",
            checkpoint_dir="test_models/genomes"
        )
        
        # 模拟几代数据
        for gen in range(5):
            population = [np.random.randn(5) for _ in range(20)]
            fitness_scores = [np.sum(ind**2) for ind in population]
            
            visualizer.update_population_state(population, fitness_scores, gen)
            
            if gen % 2 == 0:
                visualizer.visualize_evolution_progress()
        
        # 测试方法调用
        summary = visualizer.get_evolution_summary()
        
        print("  ✓ EvolutionVisualizer 测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ EvolutionVisualizer 测试失败: {e}")
        return False

def test_checkpoint_manager():
    """测试检查点管理器"""
    print("测试 CheckpointManager...")
    
    try:
        from core.evolution import CheckpointManager
        
        # 创建检查点管理器
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="test_models/genomes",
            auto_save_interval=3,
            max_checkpoints=10
        )
        
        # 保存检查点
        population = [np.random.randn(5) for _ in range(20)]
        fitness_scores = [np.sum(ind**2) for ind in population]
        
        checkpoint_info = checkpoint_manager.save_checkpoint(
            population, fitness_scores, 0, 
            checkpoint_type="auto"
        )
        
        # 加载检查点
        load_result = checkpoint_manager.load_checkpoint(0)
        
        # 列出检查点
        checkpoints = checkpoint_manager.list_checkpoints()
        
        print(f"  ✓ CheckpointManager 测试通过 - 保存/加载了 {len(checkpoints)} 个检查点")
        return True
        
    except Exception as e:
        print(f"  ✗ CheckpointManager 测试失败: {e}")
        return False

def test_evolution_dashboard():
    """测试进化仪表板"""
    print("测试 EvolutionDashboard...")
    
    try:
        from utils.visualization import EvolutionDashboard
        
        # 创建仪表板
        dashboard = EvolutionDashboard(
            data_dir="test_data/evolution_logs",
            update_interval=1.0,
            auto_reload=False  # 测试时关闭自动重载
        )
        
        # 生成静态仪表板
        output_path = "test_data/evolution_logs/test_dashboard.png"
        dashboard.create_static_dashboard(output_path, include_analysis=False)
        
        # 获取状态
        status = dashboard.get_current_status()
        
        print("  ✓ EvolutionDashboard 测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ EvolutionDashboard 测试失败: {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("测试集成功能...")
    
    try:
        from core.evolution import EvolutionVisualizer, CheckpointManager
        from utils.visualization import EvolutionDashboard
        
        # 创建组件
        visualizer = EvolutionVisualizer(
            population_size=10,
            genome_length=3,
            data_dir="test_data/evolution_logs",
            checkpoint_dir="test_models/genomes"
        )
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="test_models/genomes"
        )
        
        dashboard = EvolutionDashboard(
            data_dir="test_data/evolution_logs"
        )
        
        # 模拟完整的进化流程
        for gen in range(3):
            population = [np.random.randn(3) for _ in range(10)]
            fitness_scores = [np.sum(ind**2) for ind in population]
            
            # 更新可视化器
            visualizer.update_population_state(population, fitness_scores, gen)
            
            # 保存检查点
            if gen == 2:  # 只在最后一代保存
                checkpoint_info = checkpoint_manager.save_checkpoint(
                    population, fitness_scores, gen
                )
        
        # 生成可视化
        visualizer.visualize_evolution_progress()
        dashboard.create_static_dashboard("test_data/evolution_logs/integration_test.png")
        
        # 获取摘要
        summary = visualizer.get_evolution_summary()
        
        print("  ✓ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_data():
    """清理测试数据"""
    import shutil
    
    test_dirs = ["test_data", "test_models"]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"清理测试目录: {dir_path}")
            except Exception as e:
                print(f"清理目录失败 {dir_path}: {e}")

def main():
    """主测试函数"""
    print("NeuroMinecraftGenesis - 进化系统基础功能测试")
    print("="*60)
    
    # 确保测试目录存在
    os.makedirs("test_data/evolution_logs", exist_ok=True)
    os.makedirs("test_models/genomes", exist_ok=True)
    
    # 运行测试
    test_results = []
    
    test_results.append(test_evolution_visualizer())
    test_results.append(test_checkpoint_manager())
    test_results.append(test_evolution_dashboard())
    test_results.append(test_integration())
    
    # 统计结果
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统工作正常")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    # 询问是否清理测试数据
    cleanup = input("\n是否清理测试数据？(y/N): ").strip().lower()
    if cleanup == 'y':
        cleanup_test_data()
        print("测试数据已清理")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)