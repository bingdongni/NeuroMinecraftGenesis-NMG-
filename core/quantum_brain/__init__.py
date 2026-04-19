"""
量子-类脑计算融合模块
Quantum-Brain Computing Fusion Module

该模块提供了类脑计算和量子计算的深度融合系统。
主要组件包括：
- 10万神经元脉冲神经网络
- 量子决策电路和叠加态探索
- 神经符号混合推理架构
- STDP学习规则和可塑性建模
- 量子纠缠和相干性应用

Author: Quantum-Brain AI System
Date: 2025-11-13
"""

from .fusion_system import (
    QuantumState,
    QuantumDecisionCircuit,
    STDPNeuron,
    SpikingNeuralNetwork,
    NeuroSymbolicReasoner,
    QuantumBrainFusion,
    create_quantum_brain_fusion_system,
    demo_quantum_brain_system
)

__version__ = "1.0.0"
__author__ = "Quantum-Brain AI System"

__all__ = [
    'QuantumState',
    'QuantumDecisionCircuit', 
    'STDPNeuron',
    'SpikingNeuralNetwork',
    'NeuroSymbolicReasoner',
    'QuantumBrainFusion',
    'create_quantum_brain_fusion_system',
    'demo_quantum_brain_system'
]

# 包级别的便捷函数
def create_system(n_neurons: int = 100000, n_qubits: int = 6):
    """
    创建量子-类脑融合系统的便捷函数
    
    Args:
        n_neurons: 神经元数量，默认10万
        n_qubits: 量子比特数量，默认6
        
    Returns:
        QuantumBrainFusion: 融合系统实例
    """
    return QuantumBrainFusion(n_neurons, n_qubits)


def quick_demo():
    """
    快速演示系统功能
    
    Returns:
        dict: 演示结果
    """
    return demo_quantum_brain_system()