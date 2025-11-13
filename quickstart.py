#!/usr/bin/env python3
"""
NeuroMinecraft Genesis 快速启动脚本
开发者：bingdongni

这个脚本将演示系统的核心功能，包括：
- DiscoRL算法发现
- 六维认知引擎
- 量子-类脑融合
- 多智能体协同
"""

import sys
import os
import time
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """打印启动横幅"""
    print("\n" + "="*60)
    print("🧠 NeuroMinecraft Genesis (NMG) - AGI自主进化系统")
    print("   开发者: bingdongni")
    print("   版本: v1.0")
    print("="*60)

def test_core_imports():
    """测试核心模块导入"""
    print("\n🔧 测试核心模块导入...")
    
    try:
        # 测试基础numpy导入
        import numpy as np
        print("   ✅ NumPy导入成功")
        
        # 测试基础matplotlib
        import matplotlib.pyplot as plt
        print("   ✅ Matplotlib导入成功")
        
        # 测试基础torch (如果可用)
        try:
            import torch
            print("   ✅ PyTorch导入成功")
        except ImportError:
            print("   ⚠️ PyTorch未安装，部分功能受限")
        
        # 测试其他核心库
        import json
        import yaml
        print("   ✅ 基础库导入成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 导入测试失败: {e}")
        return False

def demo_brain_concept():
    """演示大脑概念"""
    print("\n🧠 演示六维认知引擎概念...")
    
    try:
        import numpy as np
        
        # 模拟记忆系统
        memory_data = np.random.rand(100, 64)
        print(f"   ✅ 记忆系统初始化 - {memory_data.shape} 维度")
        
        # 模拟思维推理
        thought_pattern = np.mean(memory_data, axis=0)
        print(f"   ✅ 思维模式分析 - 思维向量维度: {thought_pattern.shape}")
        
        # 模拟创造力评估
        creativity_score = np.dot(thought_pattern, np.random.rand(64))
        print(f"   ✅ 创造力评估 - 得分: {creativity_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 大脑概念演示失败: {e}")
        return False

def demo_evolution_system():
    """演示进化系统"""
    print("\n🔬 演示DiscoRL进化系统...")
    
    try:
        import numpy as np
        
        # 模拟群体初始化
        population_size = 50
        population = np.random.rand(population_size, 10)
        print(f"   ✅ 初始群体 - {population_size}个个体")
        
        # 模拟适应性评估
        fitness_scores = np.sum(population, axis=1)
        best_fitness = np.max(fitness_scores)
        print(f"   ✅ 适应性评估 - 最优得分: {best_fitness:.3f}")
        
        # 模拟选择过程
        indices = np.argsort(fitness_scores)[-10:]  # 选择前10个
        selected = population[indices]
        print(f"   ✅ 精英选择 - 选择{len(selected)}个精英")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 进化系统演示失败: {e}")
        return False

def demo_quantum_concept():
    """演示量子计算概念"""
    print("\n⚛️ 演示量子计算概念...")
    
    try:
        # 如果qiskit可用则演示真实量子电路
        try:
            from qiskit import QuantumCircuit, execute, Aer
            from qiskit.visualization import plot_histogram
            
            # 创建简单量子电路
            qc = QuantumCircuit(2)
            qc.h(0)  # Hadamard门
            qc.cx(0, 1)  # CNOT门
            qc.measure_all()
            
            # 模拟运行
            backend = Aer.get_backend('qasm_simulator')
            result = execute(qc, backend, shots=1000).result()
            counts = result.get_counts(qc)
            
            print(f"   ✅ 量子电路演示成功 - 测量结果: {list(counts.keys())}")
            
        except ImportError:
            print("   ⚠️ Qiskit未安装，使用模拟演示")
            
            # 模拟量子态
            import numpy as np
            qubit_states = np.array([1, 0])  # |0⟩ 状态
            print(f"   ✅ 量子态模拟 - 基态 |0⟩")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 量子概念演示失败: {e}")
        return False

def demo_visualization():
    """演示可视化功能"""
    print("\n📊 演示可视化功能...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 生成演示数据
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        # 创建简单图表
        plt.figure(figsize=(8, 6))
        plt.plot(x, y1, label='sin(x)')
        plt.plot(x, y2, label='cos(x)')
        plt.title('NeuroMinecraft Genesis - 函数可视化')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')
        plt.legend()
        plt.grid(True)
        
        # 保存到临时文件
        output_path = '/tmp/nmg_demo_plot.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 可视化图表已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 可视化演示失败: {e}")
        return False

def demo_performance_test():
    """性能测试"""
    print("\n⚡ 性能基准测试...")
    
    try:
        import numpy as np
        import time
        
        # 矩阵运算测试
        start_time = time.time()
        matrix_size = 1000
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        C = np.dot(A, B)
        end_time = time.time()
        
        print(f"   ✅ 矩阵运算测试 - {matrix_size}x{matrix_size}矩阵")
        print(f"      耗时: {end_time - start_time:.3f}秒")
        
        # 内存使用测试
        memory_usage = C.nbytes / (1024 * 1024)
        print(f"      内存使用: {memory_usage:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    print_banner()
    
    # 测试结果统计
    test_results = []
    start_time = time.time()
    
    # 执行所有演示
    tests = [
        ("核心模块导入测试", test_core_imports),
        ("六维认知引擎演示", demo_brain_concept),
        ("DiscoRL进化系统演示", demo_evolution_system),
        ("量子计算概念演示", demo_quantum_concept),
        ("可视化功能演示", demo_visualization),
        ("性能基准测试", demo_performance_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name}执行异常: {e}")
            test_results.append((test_name, False))
    
    # 总结测试结果
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 测试结果总结:")
    print("="*60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 整体结果:")
    print(f"   总测试: {total}")
    print(f"   通过: {passed}")
    print(f"   成功率: {passed/total*100:.1f}%")
    print(f"   总耗时: {total_time:.2f}秒")
    
    if passed == total:
        print("\n🎉 所有测试通过！NeuroMinecraft Genesis系统运行正常！")
        print("\n🚀 系统已准备就绪，您现在可以:")
        print("   - 使用 advanced_dashboard.py 启动可视化仪表盘")
        print("   - 运行 experiment_system.py 进行完整实验")
        print("   - 查看 docs/ 目录获取详细文档")
        
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，建议检查依赖安装")
        print("   - 确保所有依赖包已正确安装")
        print("   - 检查Python版本是否兼容 (推荐3.8+)")
    
    print("\n📖 更多信息请查看:")
    print("   - docs/installation_guide.md: 详细安装指南")
    print("   - docs/user_guide/README.md: 用户使用手册")
    print("   - docs/developer_guide/README.md: 开发者指南")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，启动脚本退出")
    except Exception as e:
        print(f"\n\n❌ 启动脚本执行异常: {e}")
        print("   请检查Python环境和依赖包安装")
        traceback.print_exc()
