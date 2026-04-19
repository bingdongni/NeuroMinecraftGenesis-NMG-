#!/usr/bin/env python3
"""
每周真实世界任务测试系统演示脚本（简化版）
===========================================

该脚本演示如何使用每周真实世界任务测试系统的各个组件。

作者: NeuroMinecraftGenesis
版本: 1.0.0
"""

import sys
import os
import time
import importlib
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 正确的文件路径
        file_path = "/workspace/NeuroMinecraftGenesis/worlds/real/weekly_task_test.py"
        print(f"📁 尝试加载文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 导入weekly_task_test模块
        spec = importlib.util.spec_from_file_location("weekly_task_test", file_path)
        weekly_task_test = importlib.util.module_from_spec(spec)
        sys.modules['weekly_task_test'] = weekly_task_test
        spec.loader.exec_module(weekly_task_test)
        
        print("✅ weekly_task_test 模块导入成功")
        
        # 创建测试系统
        test_system = weekly_task_test.create_weekly_test_system()
        print("✅ 测试系统创建成功")
        
        # 执行简单测试
        result = test_system.execute_test_suite()
        print(f"✅ 测试执行完成: {'成功' if result['success'] else '失败'}")
        
        if result['success']:
            print(f"   - 执行任务: {result['tasks_executed']}/{result['total_tasks']}")
            print(f"   - 成功率: {result['statistics']['success_rate']:.1%}")
            print(f"   - 平均分数: {result['statistics']['average_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主演示函数"""
    print("🎯 每周真实世界任务测试系统演示")
    print("=" * 50)
    
    # 导入必要的模块
    import importlib.util
    
    success = test_imports()
    
    if success:
        print("\n🎉 演示成功完成！")
        print("\n📋 系统特性:")
        print("✅ 每周任务测试 - 定期执行性能评估")
        print("✅ 任务调度器 - 智能任务分配和管理")
        print("✅ 性能记录器 - 全面数据收集和存储")
        print("✅ 趋势分析器 - 性能预测和模式识别")
        print("✅ 报告生成器 - 可视化报告和数据导出")
        
        print("\n📁 生成的文件:")
        print("- 测试报告: /tmp/weekly_test_reports/")
        print("- 性能数据: data/performance/")
        print("- 趋势分析: 可视化图表")
        
    else:
        print("\n❌ 演示失败，请检查错误信息")

if __name__ == "__main__":
    main()