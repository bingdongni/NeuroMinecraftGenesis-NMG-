"""
神经元群体(Neuron Population)模块
管理和控制脉冲神经元群体，支持大规模神经网络的组织和管理

功能特性：
- 多种神经元模型支持
- 神经元群体的动态创建和配置
- 群体间的层次化组织
- 性能监控和统计
- 内存和计算优化

作者：NeuroMinecraft Genesis Team
版本：1.0.0
"""

import numpy as np
import nengo
import nengo_dl
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class NeuronModel(Enum):
    """神经元模型枚举"""
    LIF = "lif"                    # Leaky Integrate-and-Fire
    IAF = "iaf"                    # Integrate-and-Fire
    SRLIF = "srlif"                # Stochastic Response LIF
    RELU = "relu"                  # ReLU神经元
    SPIKING_RELU = "spiking_relu"  # 脉冲ReLU
    SOFTMAX = "softmax"            # Softmax
    RATE = "rate"                  # Rate神经元


class PopulationType(Enum):
    """群体类型枚举"""
    EXCITATORY = "excitatory"    # 兴奋性群体
    INHIBITORY = "inhibitory"    # 抑制性群体
    MODULATORY = "modulatory"    # 调制性群体
    SENSOR = "sensor"            # 感觉神经元
    MOTOR = "motor"              # 运动神经元
    INTERNEURON = "interneuron"  # 中间神经元


@dataclass
class NeuronPopulationConfig:
    """神经元群体配置参数"""
    # 基础参数
    population_type: PopulationType = PopulationType.EXCITATORY
    neuron_model: NeuronModel = NeuronModel.LIF
    num_neurons: int = 1000      # 神经元数量
    dimensions: int = 1          # 维度数
    
    # 神经元参数
    max_rate: float = 100.0      # 最大发放率(Hz)
    refractory_period: float = 0.002  # 不应期(s)
    membrane_time_constant: float = 0.02  # 膜时间常数(s)
    firing_threshold: float = 1.0  # 发放阈值
    baseline_voltage: float = 0.0  # 基准电压
    
    # 连接参数
    connection_probability: float = 0.1  # 连接概率
    synaptic_delay: float = 0.001   # 突触延迟(s)
    
    # 可塑性参数
    plasticity_enabled: bool = True  # 是否启用可塑性
    learning_rate: float = 0.001     # 学习率
    
    # 性能参数
    seed: int = 42              # 随机种子
    device: str = '/cpu:0'     # 计算设备
    batch_size: int = 32       # 批处理大小


class NeuronPopulation:
    """
    神经元群体类
    
    管理一组功能相关的神经元，支持多种神经元模型和动态配置。
    
    核心功能：
    - 神经元模型的统一管理
    - 群体行为的统计分析
    - 神经元间的连接管理
    - 性能监控和优化
    """
    
    def __init__(self, network: nengo.Network, population_id: str,
                 config: Dict[str, Any] = None, seed: int = 42):
        """
        初始化神经元群体
        
        Args:
            network: 所属的Nengo网络
            population_id: 群体唯一标识符
            config: 配置参数
            seed: 随机种子
        """
        self.network = network
        self.population_id = population_id
        self.seed = seed
        self.logger = logging.getLogger(f"NeuronPopulation_{population_id}")
        
        # 配置
        self.config = self._setup_config(config)
        
        # 神经元群体对象
        self.ensemble = None
        self.neuron_type = None
        
        # 群体属性
        self.population_type = self.config.population_type
        self.neuron_model = self.config.neuron_model
        self.num_neurons = self.config.num_neurons
        self.dimensions = self.config.dimensions
        
        # 连接管理
        self.input_connections = {}    # 输入连接
        self.output_connections = {}   # 输出连接
        
        # 活动和统计
        self.activity_history = []     # 活动历史
        self.spike_history = []        # 脉冲历史
        self.activity_statistics = {
            'mean_activity': 0.0,
            'max_activity': 0.0,
            'sparsity': 0.0,
            'spike_rate': 0.0,
            'synchrony': 0.0
        }
        
        # 性能监控
        self.performance_metrics = {
            'simulation_time': 0.0,
            'memory_usage': 0.0,
            'connection_count': 0,
            'activity_level': 0.0
        }
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 创建神经元群体
        self._create_neuron_population()
        
        self.logger.info(f"创建神经元群体：{population_id} "
                        f"(类型：{self.population_type.value}, "
                        f"模型：{self.neuron_model.value}, "
                        f"数量：{self.num_neurons})")
    
    def _setup_config(self, config: Dict[str, Any] = None) -> NeuronPopulationConfig:
        """设置配置参数"""
        population_config = NeuronPopulationConfig()
        
        if config:
            for key, value in config.items():
                if hasattr(population_config, key):
                    setattr(population_config, key, value)
        
        return population_config
    
    def _create_neuron_population(self) -> None:
        """创建神经元群体"""
        with self.network:
            # 选择神经元类型
            self.neuron_type = self._select_neuron_type()
            
            # 创建神经元群体
            self.ensemble = nengo.Ensemble(
                n_neurons=self.num_neurons,
                dimensions=self.dimensions,
                neuron_type=self.neuron_type,
                seed=self.seed,
                label=f"ensemble_{self.population_id}"
            )
            
            # 配置神经元参数
            self._configure_neuron_parameters()
    
    def _select_neuron_type(self) -> nengo.NeuronType:
        """根据模型选择神经元类型"""
        model_mapping = {
            NeuronModel.LIF: nengo.LIF,
            NeuronModel.IAF: nengo.IAF,
            NeuronModel.RELU: nengo.RectifiedLinear,
            NeuronModel.SPIKING_RELU: nengo.SpikingRectifiedLinear,
            NeuronModel.SOFTMAX: nengo.Softmax,
            NeuronModel.RATE: nengo.LIFRate  # 使用LIF的速率版本
        }
        
        neuron_type_class = model_mapping.get(self.neuron_model, nengo.LIF)
        
        # 为特殊参数创建神经元类型实例
        if self.neuron_model == NeuronModel.LIF:
            return nengo.LIF(
                max_rate=self.config.max_rate,
                tau_rc=self.config.membrane_time_constant,
                tau_ref=self.config.refractory_period
            )
        elif self.neuron_model == NeuronModel.IAF:
            return nengo.IAF(
                tau_ref=self.config.refractory_period
            )
        elif self.neuron_model == NeuronModel.SRLIF:
            return nengo.SRLIF(
                max_rate=self.config.max_rate,
                tau_rc=self.config.membrane_time_constant
            )
        
        return neuron_type_class()
    
    def _configure_neuron_parameters(self) -> None:
        """配置神经元参数"""
        if hasattr(self.ensemble.neuron_type, 'max_rate'):
            self.ensemble.neuron_type.max_rate = self.config.max_rate
        
        if hasattr(self.ensemble.neuron_type, 'tau_rc'):
            self.ensemble.neuron_type.tau_rc = self.config.membrane_time_constant
        
        if hasattr(self.ensemble.neuron_type, 'tau_ref'):
            self.ensemble.neuron_type.tau_ref = self.config.refractory_period
    
    def connect_from(self, source: Union['NeuronPopulation', nengo.Ensemble, nengo.Node],
                    connection_type: str = "excitatory",
                    weight_range: Tuple[float, float] = (-0.1, 0.2),
                    probability: float = None) -> str:
        """
        从其他群体或节点连接到此群体
        
        Args:
            source: 源对象
            connection_type: 连接类型 ("excitatory", "inhibitory", "modulatory")
            weight_range: 权重范围
            probability: 连接概率（如果为None则使用群体默认配置）
            
        Returns:
            str: 连接标识符
        """
        connection_id = f"input_{len(self.input_connections)}"
        
        with self.network:
            # 确定权重变换
            if connection_type == "excitatory":
                transform = nengo.dists.Uniform(weight_range[0], weight_range[1])
            elif connection_type == "inhibitory":
                transform = nengo.dists.Uniform(-0.5, -0.1)
            else:  # modulatory
                transform = weight_range[0]
            
            # 创建连接
            if isinstance(source, NeuronPopulation):
                source_obj = source.ensemble
            elif isinstance(source, nengo.Ensemble):
                source_obj = source
            elif isinstance(source, nengo.Node):
                source_obj = source
            else:
                raise ValueError(f"不支持的源类型：{type(source)}")
            
            connection_prob = probability or self.config.connection_probability
            
            connection = nengo.Connection(
                source_obj,
                self.ensemble,
                transform=transform,
                synapse=self.config.synaptic_delay,
                seed=self.seed + len(self.input_connections)
            )
            
            # 记录连接信息
            self.input_connections[connection_id] = {
                'connection': connection,
                'source': source,
                'source_id': getattr(source, 'population_id', 'unknown'),
                'connection_type': connection_type,
                'weight_range': weight_range,
                'probability': connection_prob
            }
        
        self.logger.info(f"添加输入连接：{connection_id}")
        return connection_id
    
    def connect_to(self, target: Union['NeuronPopulation', nengo.Ensemble],
                  connection_type: str = "excitatory",
                  weight_range: Tuple[float, float] = (0.1, 0.3),
                  probability: float = None) -> str:
        """
        从此群体连接到其他群体
        
        Args:
            target: 目标对象
            connection_type: 连接类型
            weight_range: 权重范围
            probability: 连接概率
            
        Returns:
            str: 连接标识符
        """
        connection_id = f"output_{len(self.output_connections)}"
        
        with self.network:
            # 确定权重变换
            if connection_type == "excitatory":
                transform = nengo.dists.Uniform(weight_range[0], weight_range[1])
            elif connection_type == "inhibitory":
                transform = nengo.dists.Uniform(-0.3, -0.1)
            else:
                transform = weight_range[0]
            
            # 创建连接
            if isinstance(target, NeuronPopulation):
                target_obj = target.ensemble
            elif isinstance(target, nengo.Ensemble):
                target_obj = target
            else:
                raise ValueError(f"不支持的目标类型：{type(target)}")
            
            connection_prob = probability or self.config.connection_probability
            
            connection = nengo.Connection(
                self.ensemble,
                target_obj,
                transform=transform,
                synapse=self.config.synaptic_delay,
                seed=self.seed + len(self.output_connections)
            )
            
            # 记录连接信息
            self.output_connections[connection_id] = {
                'connection': connection,
                'target': target,
                'target_id': getattr(target, 'population_id', 'unknown'),
                'connection_type': connection_type,
                'weight_range': weight_range,
                'probability': connection_prob
            }
        
        self.logger.info(f"添加输出连接：{connection_id}")
        return connection_id
    
    def add_probe(self, probe_type: str = "activity", 
                  synapse: float = 0.01) -> str:
        """
        添加探针以监控群体活动
        
        Args:
            probe_type: 探针类型 ("activity", "spikes", "voltage")
            synapse: 平滑时间常数
            
        Returns:
            str: 探针标识符
        """
        probe_id = f"probe_{probe_type}_{len(self.activity_history)}"
        
        with self.network:
            if probe_type == "activity":
                probe = nengo.Probe(self.ensemble.neurons, synapse=synapse)
            elif probe_type == "spikes":
                probe = nengo.Probe(self.ensemble.neurons, 'spikes')
            elif probe_type == "voltage":
                if hasattr(self.ensemble.neuron_type, 'voltage'):
                    probe = nengo.Probe(self.ensemble, 'voltage')
                else:
                    self.logger.warning("该神经元模型不支持电压探针")
                    return None
            else:
                raise ValueError(f"不支持的探针类型：{probe_type}")
            
            # 添加到活动监控
            if probe_type not in self.activity_history:
                self.activity_history[probe_type] = []
            
            self.activity_history[probe_type].append(probe)
        
        self.logger.info(f"添加探针：{probe_id}")
        return probe_id
    
    def update_activity_statistics(self, activity_data: np.ndarray) -> None:
        """更新活动统计"""
        with self._lock:
            if activity_data.size == 0:
                return
            
            # 计算统计指标
            mean_activity = np.mean(activity_data)
            max_activity = np.max(activity_data)
            
            # 计算稀疏性（活跃神经元比例）
            if len(activity_data.shape) > 1:
                activity_flat = activity_data.flatten()
                sparsity = np.mean(activity_flat > (mean_activity + np.std(activity_data)))
            else:
                sparsity = np.mean(activity_data > mean_activity)
            
            # 计算同步性（基于相关性）
            if len(activity_data.shape) > 1 and activity_data.shape[1] > 1:
                correlation_matrix = np.corrcoef(activity_data.T)
                synchrony = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                synchrony = abs(synchrony)  # 使用绝对值
            else:
                synchrony = 0.0
            
            # 更新统计
            self.activity_statistics.update({
                'mean_activity': mean_activity,
                'max_activity': max_activity,
                'sparsity': sparsity,
                'spike_rate': mean_activity,  # 简化处理
                'synchrony': synchrony
            })
    
    def get_population_info(self) -> Dict[str, Any]:
        """获取群体信息"""
        return {
            'population_id': self.population_id,
            'population_type': self.population_type.value,
            'neuron_model': self.neuron_model.value,
            'num_neurons': self.num_neurons,
            'dimensions': self.dimensions,
            'activity_statistics': self.activity_statistics.copy(),
            'input_connections': {
                conn_id: {
                    'source_id': info['source_id'],
                    'connection_type': info['connection_type'],
                    'probability': info['probability']
                }
                for conn_id, info in self.input_connections.items()
            },
            'output_connections': {
                conn_id: {
                    'target_id': info['target_id'],
                    'connection_type': info['connection_type'],
                    'probability': info['probability']
                }
                for conn_id, info in self.output_connections.items()
            },
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def get_activity_summary(self) -> Dict[str, float]:
        """获取活动摘要"""
        return {
            'mean_activity': self.activity_statistics['mean_activity'],
            'sparsity': self.activity_statistics['sparsity'],
            'synchrony': self.activity_statistics['synchrony'],
            'total_connections': len(self.input_connections) + len(self.output_connections),
            'activity_level': self.activity_statistics['spike_rate']
        }
    
    def optimize_for_performance(self) -> None:
        """性能优化"""
        self.logger.info(f"优化群体 {self.population_id} 的性能...")
        
        # 内存优化
        if len(self.activity_history) > 100:
            # 保留最近的活动历史
            self.activity_history = self.activity_history[-50:]
        
        # 连接优化
        connection_overhead = (len(self.input_connections) + len(self.output_connections))
        if connection_overhead > self.num_neurons * 0.5:
            self.logger.warning(f"连接密度过高：{connection_overhead}/{self.num_neurons}")
        
        # 活动优化
        if self.activity_statistics['mean_activity'] < 0.01:
            self.logger.info("群体活动较低，考虑调整参数")
    
    def reset_activity(self) -> None:
        """重置活动历史和统计"""
        with self._lock:
            self.activity_history = []
            self.spike_history = []
            self.activity_statistics = {
                'mean_activity': 0.0,
                'max_activity': 0.0,
                'sparsity': 0.0,
                'spike_rate': 0.0,
                'synchrony': 0.0
            }
            
            self.logger.info(f"重置群体 {self.population_id} 的活动")
    
    def scale_population(self, scale_factor: float) -> None:
        """调整群体规模"""
        old_size = self.num_neurons
        new_size = int(old_size * scale_factor)
        
        if new_size < 10:
            self.logger.warning(f"缩放后群体规模过小：{new_size}")
            return
        
        # 更新配置
        self.config.num_neurons = new_size
        self.num_neurons = new_size
        
        # 注意：实际的网络对象不能动态更改大小
        # 这里只是更新配置信息
        self.logger.info(f"群体 {self.population_id} 规模从 {old_size} 调整为 {new_size}")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"NeuronPopulation({self.population_id}, "
                f"type={self.population_type.value}, "
                f"neurons={self.num_neurons}, "
                f"model={self.neuron_model.value})")