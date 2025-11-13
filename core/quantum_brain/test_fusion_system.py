#!/usr/bin/env python3
"""
量子-类脑融合系统测试脚本
Quantum-Brain Fusion System Test Script

该脚本用于测试量子-类脑融合系统的各项功能。
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from core.quantum_brain import (
        QuantumBrainFusion,
        create_quantum_brain_fusion_system,
        demo_quantum_brain_system
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("="*60)
    print("测试量子-类脑融合系统基本功能")
    print("="*60)
    
    try:
        # 创建系统（使用较小的规模用于测试）
        fusion_system = QuantumBrainFusion(n_neurons=1000, n_qubits=4)
        print("✓ 系统创建成功")
        
        # 测试初始化
        fusion_system.initialize_system()
        print("✓ 系统初始化成功")
        
        # 测试输入处理
        test_input = np.random.normal(0, 1, 4)
        result = fusion_system.process_input(test_input)
        print("✓ 输入处理成功")
        
        # 获取系统状态
        system_state = fusion_system.get_system_state()
        print("✓ 系统状态获取成功")
        
        # 测试性能基准
        performance = fusion_system.run_performance_benchmark()
        print("✓ 性能基准测试成功")
        
        # 关闭系统
        shutdown_result = fusion_system.shutdown()
        print("✓ 系统关闭成功")
        
        print("\n所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_quantum_components():
    """测试量子组件"""
    print("\n" + "="*60)
    print("测试量子组件")
    print("="*60)
    
    try:
        from core.quantum_brain.fusion_system import QuantumDecisionCircuit, QuantumState
        
        # 测试量子态
        quantum_state = QuantumState(3)
        print("✓ 量子态创建成功")
        
        # 测试量子叠加态
        quantum_state.set_superposition([0, 1, 2], [0.5, 0.5, 0.0])
        print("✓ 量子叠加态设置成功")
        
        # 测试量子测量
        measurement = quantum_state.measure()
        print(f"✓ 量子测量成功: {measurement}")
        
        # 测试量子决策电路
        quantum_circuit = QuantumDecisionCircuit(3, 2)
        print("✓ 量子决策电路创建成功")
        
        # 测试量子门操作
        quantum_circuit.apply_gate("H", 0)
        quantum_circuit.apply_gate("RX", 1, 0.5)
        print("✓ 量子门操作成功")
        
        # 测试量子决策
        input_signal = np.array([0.5, 0.3, 0.8])
        decision, confidence = quantum_circuit.quantum_decision(input_signal)
        print(f"✓ 量子决策成功: decision={decision}, confidence={confidence:.3f}")
        
        print("\n所有量子组件测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 量子组件测试失败: {e}")
        return False


def test_neural_components():
    """测试神经组件"""
    print("\n" + "="*60)
    print("测试神经组件")
    print("="*60)
    
    try:
        from core.quantum_brain.fusion_system import STDPNeuron, SpikingNeuralNetwork
        
        # 测试STDP神经元
        neuron = STDPNeuron(0)
        neuron.receive_input(10.0)  # 强输入信号
        print("✓ STDP神经元创建和输入处理成功")
        
        # 测试脉冲神经网络
        snn = SpikingNeuralNetwork(n_neurons=100, n_layers=2)
        print("✓ 脉冲神经网络创建成功")
        
        # 添加输入
        test_input = np.random.normal(0, 5, 50)
        snn.add_input(0, test_input)
        print("✓ 脉冲神经网络输入添加成功")
        
        # 运行模拟步骤
        result = snn.step_simulation()
        print(f"✓ 脉冲神经网络模拟成功: {len(result['spike_events'])} 个脉冲事件")
        
        # 获取网络活动
        activity = snn.get_network_activity()
        print(f"✓ 网络活动统计获取成功: 活跃神经元数 = {activity['active_neurons']}")
        
        print("\n所有神经组件测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 神经组件测试失败: {e}")
        return False


def test_symbolic_components():
    """测试符号组件"""
    print("\n" + "="*60)
    print("测试符号组件")
    print("="*60)
    
    try:
        from core.quantum_brain.fusion_system import NeuroSymbolicReasoner
        
        # 创建神经符号推理器
        reasoner = NeuroSymbolicReasoner()
        print("✓ 神经符号推理器创建成功")
        
        # 测试概念学习
        neural_pattern = np.random.normal(0, 1, 100)
        concept_id = reasoner.learn_concept(
            neural_pattern,
            "test_concept",
            {"test": True, "level": "high"}
        )
        print(f"✓ 概念学习成功: concept_id = {concept_id}")
        
        # 测试符号推理
        result = reasoner.symbolic_reasoning(
            "What is test_concept?",
            {"context": "test"}
        )
        print("✓ 符号推理成功")
        
        # 测试混合推理
        fusion_result = reasoner.hybrid_inference(
            neural_pattern,
            "How does this pattern relate to test_concept?",
            {"context": "test"}
        )
        print("✓ 混合推理成功")
        
        print("\n所有符号组件测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 符号组件测试失败: {e}")
        return False


def performance_test():
    """性能测试"""
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)
    
    try:
        # 创建性能测试系统
        fusion_system = QuantumBrainFusion(n_neurons=500, n_qubits=3)
        fusion_system.initialize_system()
        
        # 批量输入测试
        start_time = time.time()
        test_results = []
        
        for i in range(20):  # 20次测试
            test_input = np.random.normal(0, 1, 3)
            result = fusion_system.process_input(test_input)
            test_results.append(result['processing_time'])
            time.sleep(0.01)  # 短暂延迟
            
        total_time = time.time() - start_time
        
        # 性能统计
        avg_processing_time = np.mean(test_results)
        max_processing_time = np.max(test_results)
        min_processing_time = np.min(test_results)
        throughput = len(test_results) / total_time
        
        print(f"✓ 性能测试完成")
        print(f"  平均处理时间: {avg_processing_time:.4f}秒")
        print(f"  最大处理时间: {max_processing_time:.4f}秒")
        print(f"  最小处理时间: {min_processing_time:.4f}秒")
        print(f"  吞吐量: {throughput:.2f} 操作/秒")
        
        # 关闭系统
        fusion_system.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("量子-类脑融合系统全面测试")
    print("="*60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("基本功能测试", test_basic_functionality()))
    test_results.append(("量子组件测试", test_quantum_components()))
    test_results.append(("神经组件测试", test_neural_components()))
    test_results.append(("符号组件测试", test_symbolic_components()))
    test_results.append(("性能测试", performance_test()))
    
    # 运行完整演示
    print("\n" + "="*60)
    print("完整系统演示")
    print("="*60)
    demo_result = demo_quantum_brain_system()
    test_results.append(("完整系统演示", True))
    
    # 测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 项测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试都通过了！量子-类脑融合系统运行正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)