#!/usr/bin/env python3
"""
泛化能力压力测试系统快速验证脚本
================================

快速验证各个组件的基本功能
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments', 'cognition'))

def test_imports():
    """测试导入功能"""
    print("1. 测试模块导入...")
    try:
        from modded_minecraft_test import ModdedMinecraftTest
        from pybullet_test import PyBulletTest
        from reddit_dialogue_test import RedditDialogueTest
        from zero_shot_evaluator import ZeroShotEvaluator
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n2. 测试基本功能...")
    try:
        # 测试模组Minecraft
        mc_test = ModdedMinecraftTest()
        zero_shot = mc_test.run_zero_shot_test()
        print(f"  模组Minecraft零样本测试: {zero_shot:.3f}")
        
        # 测试PyBullet
        pb_test = PyBulletTest()
        zero_shot_pb = pb_test.run_zero_shot_test()
        print(f"  PyBullet零样本测试: {zero_shot_pb:.3f}")
        
        # 测试Reddit
        rd_test = RedditDialogueTest()
        zero_shot_rd = rd_test.run_zero_shot_test()
        print(f"  Reddit对话零样本测试: {zero_shot_rd:.3f}")
        
        # 测试评估器
        evaluator = ZeroShotEvaluator()
        print(f"  评估器初始化成功")
        
        print("✓ 基本功能测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False

def test_performance_metrics():
    """测试性能指标"""
    print("\n3. 测试性能指标...")
    try:
        evaluator = ZeroShotEvaluator()
        
        # 测试适应速度计算
        adaptation_speed = evaluator.calculate_adaptation_speed(0.3, 0.7, 50)
        print(f"  适应速度: {adaptation_speed:.6f}")
        
        # 测试学习效率计算
        learning_efficiency = evaluator.calculate_learning_efficiency(adaptation_speed, 0.8)
        print(f"  学习效率: {learning_efficiency:.3f}")
        
        print("✓ 性能指标测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 性能指标测试失败: {e}")
        return False

def main():
    """主验证函数"""
    print("=" * 60)
    print("泛化能力压力测试系统快速验证")
    print("=" * 60)
    
    results = []
    
    # 1. 测试导入
    results.append(test_imports())
    
    # 2. 测试基本功能
    results.append(test_basic_functionality())
    
    # 3. 测试性能指标
    results.append(test_performance_metrics())
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结:")
    print(f"成功: {sum(results)}/{len(results)}")
    if all(results):
        print("✓ 所有组件工作正常！")
        print("\n系统可以用于泛化能力测试。")
        print("\n主要功能包括:")
        print("• 模组Minecraft零样本测试")
        print("• PyBullet物理模拟器测试") 
        print("• Reddit对话能力测试")
        print("• 少样本适应评估")
        print("• 跨域迁移分析")
        print("• 性能基准比较")
        print("• 详细测试报告生成")
    else:
        print("✗ 部分组件存在问题，需要检查。")
    print("=" * 60)

if __name__ == "__main__":
    main()