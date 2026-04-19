"""
皮层柱(Cortical Column)模块
实现基于生物皮层柱结构的脉冲神经网络组件

皮层柱是哺乳动物大脑皮层的基本功能单元，包含：
- 输入层(L1)：接收来自其他区域的输入
- 颗粒层(L2/3)：处理和传递信息  
- 棱锥层(L4)：接收丘脑输入
- 颗粒上层(L5)：输出层
- 分子层(L6)：反馈连接

作者：NeuroMinecraft Genesis Team
版本：1.0.0
"""

import numpy as np
import nengo
import nengo_dl
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
from dataclasses import dataclass
from enum import Enum


class LayerType(Enum):
    """皮层层类型枚举"""
    INPUT = "input"          # 输入层 (L1)
    GRANULAR = "granular"    # 颗粒层 (L2/3)
    PYRAMIDAL = "pyramidal"  # 棱锥层 (L4)
    OUTPUT = "output"        # 输出层 (L5)
    FEEDBACK = "feedback"    # 反馈层 (L6)


class ActivationPattern(Enum):
    """激活模式枚举"""
    EXCITATORY = "excitatory"      # 兴奋性激活
    INHIBITORY = "inhibitory"      # 抑制性激活
    BALANCED = "balanced"          # 平衡激活
    SPARSITY = "sparsity"          # 稀疏激活


@dataclass
class CorticalColumnConfig:
    """皮层柱配置参数"""
    # 基础参数
    num_layers: int = 6                    # 皮层层数
    neurons_per_layer: List[int] = None   # 每层神经元数量
    layer_types: List[LayerType] = None   # 层类型
    
    # 连接参数
    connection_probability: float = 0.1   # 连接概率
    excitatory_ratio: float = 0.8         # 兴奋性神经元比例
    synaptic_delay: float = 0.001         # 突触延迟(s)
    
    # 生物参数
    membrane_time_constant: float = 0.02  # 膜时间常数(s)
    refractory_period: float = 0.002      # 不应期(s)
    firing_threshold: float = 1.0         # 发放阈值
    
    # 可塑性参数
    stdp_enabled: bool = True             # 是否启用STDP
    learning_rate: float = 0.001          # 学习率
    plasticity_window: float = 0.02       # 可塑性时间窗口
    
    # 性能参数
    batch_size: int = 32                  # 批处理大小
    seed: int = 42                        # 随机种子


class CorticalColumn:
    """
    皮层柱类
    
    模拟生物皮层柱的基本结构和工作原理。
    皮层柱是皮层的基本功能单元，垂直排列并具有相似的感觉响应特性。
    
    结构特征：
    - 6层结构，每层有特定功能
    - 兴奋性和抑制性神经元混合
    - 层间和层内连接
    - 突触可塑性机制
    
    工作原理：
    - 自下而上的信息处理
    - 横向抑制和侧激活
    - 自上而下的反馈调节
    - 动态权重调整
    """
    
    def __init__(self, network: nengo.Network, config: Dict[str, Any] = None, 
                 seed: int = 42):
        """
        初始化皮层柱
        
        Args:
            network: 所属的Nengo网络
            config: 配置参数字典
            seed: 随机种子
        """
        self.network = network
        self.seed = seed
        self.logger = logging.getLogger(f"CorticalColumn_{id(self)}")
        
        # 设置默认配置
        self.config = self._setup_default_config(config)
        
        # 皮层结构
        self.layers = {}           # 皮层神经元群体
        self.layer_types = {}      # 层类型映射
        self.connections = {}      # 层间连接
        self.feedback_connections = {}  # 反馈连接
        
        # 生物神经元参数
        self.excitatory_neurons = {}   # 兴奋性神经元群体
        self.inhibitory_neurons = {}   # 抑制性神经元群体
        
        # 可塑性机制
        self.plasticity_rules = {}     # 可塑性规则
        self.synaptic_weights = {}     # 突触权重
        
        # 性能监控
        self.activity_metrics = {
            'layer_activity': {},
            'spike_rates': {},
            'connection_strength': {},
            'plasticity_indicators': {}
        }
        
        # 构建皮层柱结构
        self._build_cortical_structure()
        
        self.logger.info(f"皮层柱构建完成，包含{self.config.num_layers}层")
    
    def _setup_default_config(self, config: Dict[str, Any] = None) -> CorticalColumnConfig:
        """设置默认配置"""
        default_config = CorticalColumnConfig()
        
        if config:
            # 更新默认配置
            for key, value in config.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
        
        # 设置默认层参数
        if default_config.neurons_per_layer is None:
            default_config.neurons_per_layer = [100, 200, 300, 200, 100, 50]
        
        if len(default_config.neurons_per_layer) != default_config.num_layers:
            # 自动调整神经元数量
            scale_factor = default_config.num_layers / len(default_config.neurons_per_layer)
            default_config.neurons_per_layer = [
                int(n * scale_factor) for n in default_config.neurons_per_layer[:default_config.num_layers]
            ]
        
        if default_config.layer_types is None:
            default_config.layer_types = [
                LayerType.INPUT, LayerType.GRANULAR, LayerType.PYRAMIDAL,
                LayerType.PYRAMIDAL, LayerType.OUTPUT, LayerType.FEEDBACK
            ]
        
        return default_config
    
    def _build_cortical_structure(self) -> None:
        """构建皮层柱结构"""
        with self.network:
            # 创建各层神经元群体
            for layer_idx in range(self.config.num_layers):
                layer_name = f"layer_{layer_idx}"
                layer_type = self.config.layer_types[layer_idx]
                num_neurons = self.config.neurons_per_layer[layer_idx]
                
                # 创建兴奋性和抑制性神经元
                excitatory_count = int(num_neurons * self.config.excitatory_ratio)
                inhibitory_count = num_neurons - excitatory_count
                
                # 选择神经元类型
                neuron_type = self._select_neuron_type(layer_type)
                
                # 创建兴奋性神经元群体
                if excitatory_count > 0:
                    self.excitatory_neurons[f"{layer_name}_exc"] = nengo.Ensemble(
                        n_neurons=excitatory_count,
                        dimensions=1,
                        neuron_type=neuron_type,
                        seed=self.seed + layer_idx * 2
                    )
                
                # 创建抑制性神经元群体  
                if inhibitory_count > 0:
                    self.inhibitory_neurons[f"{layer_name}_inh"] = nengo.Ensemble(
                        n_neurons=inhibitory_count,
                        dimensions=1,
                        neuron_type=nengo.LIF(),  # 抑制性神经元通常使用LIF
                        seed=self.seed + layer_idx * 2 + 1
                    )
                
                self.layers[layer_name] = {
                    'excitatory': self.excitatory_neurons.get(f"{layer_name}_exc"),
                    'inhibitory': self.inhibitory_neurons.get(f"{layer_name}_inh"),
                    'type': layer_type,
                    'neurons': num_neurons
                }
                
                self.layer_types[layer_name] = layer_type
            
            # 创建层间连接
            self._build_layer_connections()
            
            # 创建反馈连接
            self._build_feedback_connections()
    
    def _select_neuron_type(self, layer_type: LayerType) -> nengo.NeuronType:
        """根据层类型选择神经元模型"""
        layer_neuron_mapping = {
            LayerType.INPUT: nengo.RectifiedLinear(),
            LayerType.GRANULAR: nengo.LIF(),
            LayerType.PYRAMIDAL: nengo.LIF(),
            LayerType.OUTPUT: nengo.SpikingRectifiedLinear(),
            LayerType.FEEDBACK: nengo.LIF()
        }
        
        return layer_neuron_mapping.get(layer_type, nengo.LIF())
    
    def _build_layer_connections(self) -> None:
        """构建层间前馈连接"""
        layer_names = list(self.layers.keys())
        
        for i in range(len(layer_names) - 1):
            source_layer = layer_names[i]
            target_layer = layer_names[i + 1]
            
            self._create_layer_connection(source_layer, target_layer)
    
    def _create_layer_connection(self, source_layer: str, target_layer: str) -> None:
        """创建层间连接"""
        source_info = self.layers[source_layer]
        target_info = self.layers[target_layer]
        
        # 兴奋性连接：源层兴奋性 -> 目标层兴奋性
        if (source_info['excitatory'] and target_info['excitatory'] and
            source_info['type'] != LayerType.FEEDBACK):
            
            connection_name = f"{source_layer}_to_{target_layer}_exc"
            self.connections[connection_name] = nengo.Connection(
                source_info['excitatory'].neurons,
                target_info['excitatory'].neurons,
                transform=nengo.dists.Uniform(-0.1, 0.2),  # 兴奋性权重偏置
                synapse=self.config.synaptic_delay,
                seed=self.seed + hash(connection_name) % 1000
            )
        
        # 抑制性连接：源层兴奋性 -> 目标层抑制性
        if (source_info['excitatory'] and target_info['inhibitory']):
            connection_name = f"{source_layer}_to_{target_layer}_inh"
            self.connections[connection_name] = nengo.Connection(
                source_info['excitatory'].neurons,
                target_info['inhibitory'].neurons,
                transform=nengo.dists.Uniform(0.1, 0.5),  # 较强的抑制性权重
                synapse=self.config.synaptic_delay,
                seed=self.seed + hash(connection_name) % 1000
            )
        
        # 反馈抑制：目标层抑制性 -> 源层兴奋性
        if target_info['inhibitory'] and source_info['excitatory']:
            connection_name = f"{target_layer}_inhibits_{source_layer}"
            self.connections[connection_name] = nengo.Connection(
                target_info['inhibitory'].neurons,
                source_info['excitatory'].neurons,
                transform=nengo.dists.Uniform(-0.3, -0.1),  # 负权重表示抑制
                synapse=self.config.synaptic_delay,
                seed=self.seed + hash(connection_name) % 1000
            )
    
    def _build_feedback_connections(self) -> None:
        """构建反馈连接"""
        layer_names = list(self.layers.keys())
        
        # 从输出层到各层的反馈
        for i, target_layer in enumerate(layer_names[:-1]):
            if target_layer != layer_names[0]:  # 不反馈到输入层
                source_layer = layer_names[-1]  # 使用输出层作为反馈源
                self._create_feedback_connection(source_layer, target_layer)
    
    def _create_feedback_connection(self, source_layer: str, target_layer: str) -> None:
        """创建反馈连接"""
        source_info = self.layers[source_layer]
        target_info = self.layers[target_layer]
        
        # 反馈兴奋性连接
        if source_info['excitatory'] and target_info['excitatory']:
            connection_name = f"feedback_{source_layer}_to_{target_layer}"
            self.feedback_connections[connection_name] = nengo.Connection(
                source_info['excitatory'].neurons,
                target_info['excitatory'].neurons,
                transform=nengo.dists.Uniform(-0.05, 0.1),  # 较弱的反馈
                synapse=self.config.synaptic_delay * 2,  # 反馈延迟较长
                seed=self.seed + hash(connection_name) % 1000
            )
    
    def add_external_input(self, input_source: Union[nengo.Node, nengo.Ensemble], 
                          target_layer: str) -> str:
        """
        添加外部输入到指定层
        
        Args:
            input_source: 输入源（节点或神经元群体）
            target_layer: 目标层名称
            
        Returns:
            str: 连接名称
        """
        if target_layer not in self.layers:
            raise ValueError(f"目标层 {target_layer} 不存在")
        
        target_info = self.layers[target_layer]
        connection_name = f"external_input_to_{target_layer}"
        
        with self.network:
            # 连接到兴奋性神经元
            if target_info['excitatory']:
                self.connections[connection_name] = nengo.Connection(
                    input_source,
                    target_info['excitatory'].neurons,
                    transform=nengo.dists.Uniform(0.1, 0.3),
                    synapse=self.config.synaptic_delay,
                    seed=self.seed + hash(connection_name) % 1000
                )
        
        self.logger.info(f"添加外部输入到层 {target_layer}")
        return connection_name
    
    def add_output_probe(self, layer_name: str) -> str:
        """
        添加输出探针
        
        Args:
            layer_name: 层名称
            
        Returns:
            str: 探针名称
        """
        if layer_name not in self.layers:
            raise ValueError(f"层 {layer_name} 不存在")
        
        layer_info = self.layers[layer_name]
        probe_name = f"probe_{layer_name}"
        
        with self.network:
            if layer_info['excitatory']:
                # 创建活动探针
                self.activity_metrics['layer_activity'][probe_name] = nengo.Probe(
                    layer_info['excitatory'].neurons,
                    synapse=0.01  # 平滑时间常数
                )
                
                # 创建脉冲探针
                self.activity_metrics['spike_rates'][f"{probe_name}_spikes"] = nengo.Probe(
                    layer_info['excitatory'].neurons,
                    'spikes'
                )
        
        self.logger.info(f"添加输出探针到层 {layer_name}")
        return probe_name
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """获取层信息"""
        if layer_name not in self.layers:
            raise ValueError(f"层 {layer_name} 不存在")
        
        layer_info = self.layers[layer_name]
        layer_type = layer_info['type']
        
        return {
            'name': layer_name,
            'type': layer_type.value,
            'total_neurons': layer_info['neurons'],
            'excitatory_count': layer_info['excitatory'].n_neurons if layer_info['excitatory'] else 0,
            'inhibitory_count': layer_info['inhibitory'].n_neurons if layer_info['inhibitory'] else 0,
            'connections_from': [name for name, conn in self.connections.items() 
                               if layer_name in name.split('_to_')[0]],
            'connections_to': [name for name, conn in self.connections.items() 
                             if layer_name in name.split('_to_')[1:]]
        }
    
    def get_cortical_info(self) -> Dict[str, Any]:
        """获取整个皮层柱信息"""
        return {
            'total_layers': self.config.num_layers,
            'total_neurons': sum(layer['neurons'] for layer in self.layers.values()),
            'excitatory_neurons': sum(
                layer['excitatory'].n_neurons for layer in self.layers.values() 
                if layer['excitatory']
            ),
            'inhibitory_neurons': sum(
                layer['inhibitory'].n_neurons for layer in self.layers.values()
                if layer['inhibitory']
            ),
            'layer_types': {name: layer['type'].value for name, layer in self.layers.items()},
            'connection_count': len(self.connections) + len(self.feedback_connections),
            'layer_configs': {
                name: {
                    'neurons': info['neurons'],
                    'type': info['type'].value,
                    'excitatory_ratio': (info['excitatory'].n_neurons / info['neurons'] 
                                       if info['excitatory'] else 0.0)
                }
                for name, info in self.layers.items()
            }
        }
    
    def enable_plasticity(self, layer_name: str, connection_type: str = "all") -> None:
        """
        启用突触可塑性
        
        Args:
            layer_name: 目标层名称
            connection_type: 连接类型 ("feedforward", "feedback", "all")
        """
        if self.config.stdp_enabled:
            # 这里可以添加STDP或其他可塑性规则
            # 由于NengoDL的限制，这里是一个示例实现
            plasticity_name = f"plasticity_{layer_name}_{connection_type}"
            self.plasticity_rules[plasticity_name] = {
                'type': 'STDP',
                'layer': layer_name,
                'connection_type': connection_type,
                'learning_rate': self.config.learning_rate,
                'window': self.config.plasticity_window
            }
            
            self.logger.info(f"在层 {layer_name} 启用 {connection_type} 可塑性")
    
    def analyze_activity_patterns(self, simulator_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析皮层活动模式"""
        analysis = {
            'layer_activation_patterns': {},
            'synchronization_metrics': {},
            'information_content': {},
            'plasticity_indicators': {}
        }
        
        for layer_name, layer_info in self.layers.items():
            # 计算层激活模式
            probe_name = f"probe_{layer_name}"
            if probe_name in simulator_results:
                activity = simulator_results[probe_name]
                analysis['layer_activation_patterns'][layer_name] = {
                    'mean_activity': np.mean(activity),
                    'max_activity': np.max(activity),
                    'sparsity': np.mean(activity > np.mean(activity) * 2),
                    'stability': np.std(activity) / (np.mean(activity) + 1e-8)
                }
        
        return analysis
    
    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_cortical_info()
        return (f"CorticalColumn(layers={info['total_layers']}, "
                f"neurons={info['total_neurons']}, "
                f"connections={info['connection_count']})")