"""
NengoDL脉冲神经网络模块
实现基于NengoDL的高性能脉冲神经网络系统，支持10万神经元的实时模拟

主要功能：
- 皮层柱结构建模
- 突触连接和可塑性机制
- 实时性能优化
- 生物神经模拟

作者：NeuroMinecraft Genesis Team
版本：1.0.0
"""

from .spiking_neural_network import SpikingNeuralNetwork
from .cortical_column import CorticalColumn
from .synaptic_connections import SynapticConnection
from .neuron_population import NeuronPopulation
from .spiking_input import SpikingInput

__all__ = [
    'SpikingNeuralNetwork',
    'CorticalColumn', 
    'SynapticConnection',
    'NeuronPopulation',
    'SpikingInput'
]

__version__ = '1.0.0'
__author__ = 'NeuroMinecraft Genesis Team'