"""
突触连接(Synaptic Connections)模块
实现生物启发的突触连接和可塑性机制

突触是神经元间信息传递的关键结构，具有以下特征：
- 突触前和突触后结构
- 神经递质释放机制
- 突触权重可塑性(STDP、LTP/LTD)
- 时序依赖性可塑性
- 抑制和兴奋性连接

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
from abc import ABC, abstractmethod


class ConnectionType(Enum):
    """连接类型枚举"""
    EXCITATORY = "excitatory"     # 兴奋性连接
    INHIBITORY = "inhibitory"     # 抑制性连接
    MODULATORY = "modulatory"     # 调制性连接
    GAP_JUNCTION = "gap_junction" # 电突触


class PlasticityRule(Enum):
    """可塑性规则枚举"""
    STDP = "stdp"                 # 尖峰时序依赖性可塑性
    LTP_LTD = "ltp_ltd"          # 长时程增强/抑制
    HOMEOSTATIC = "homeostatic"   # 稳态可塑性
    BCM = "bcm"                  # BCM理论
    OJA = "oja"                  # Oja's rule


@dataclass
class SynapticConfig:
    """突触连接配置参数"""
    # 基础参数
    connection_type: ConnectionType = ConnectionType.EXCITATORY
    plasticity_rule: PlasticityRule = PlasticityRule.STDP
    
    # 突触参数
    max_weight: float = 1.0           # 最大突触权重
    min_weight: float = 0.0           # 最小突触权重
    initial_weight: float = 0.1       # 初始权重
    
    # 时序参数
    synaptic_delay: float = 0.001     # 突触延迟(s)
    decay_time: float = 0.005         # 衰减时间(s)
    refractory_time: float = 0.002    # 不应期(s)
    
    # 可塑性参数
    learning_rate: float = 0.001      # 学习率
    plasticity_window: float = 0.02   # 可塑性时间窗口(ms)
    homeostatic_target: float = 0.1   # 稳态目标激活率
    
    # 权重初始化参数
    weight_distribution: str = "uniform"  # "uniform", "normal", "exponential", "lognormal"
    weight_mean: float = 0.1          # 权重均值
    weight_std: float = 0.05          # 权重标准差
    
    # 性能参数
    enable_plasticity: bool = True    # 是否启用可塑性
    cache_size: int = 1000           # 缓存大小


class PlasticityMechanism(ABC):
    """可塑性机制抽象基类"""
    
    @abstractmethod
    def update_weights(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                      current_weights: np.ndarray, dt: float) -> np.ndarray:
        """更新突触权重
        
        Args:
            pre_spikes: 突触前神经元脉冲
            post_spikes: 突触后神经元脉冲
            current_weights: 当前突触权重
            dt: 时间步长
            
        Returns:
            更新后的突触权重
        """
        pass
    
    @abstractmethod
    def initialize(self, weight_matrix_shape: Tuple[int, int]) -> np.ndarray:
        """初始化权重矩阵
        
        Args:
            weight_matrix_shape: 权重矩阵形状
            
        Returns:
            初始权重矩阵
        """
        pass


class STDPPlasticity(PlasticityMechanism):
    """
    STDP (Spike-Timing Dependent Plasticity) 可塑性机制
    
    实现经典的尖峰时序依赖性可塑性：
    - 如果突触前脉冲在突触后脉冲之前到达：LTP (权重增加)
    - 如果突触前脉冲在突触后脉冲之后到达：LTD (权重减少)
    - 时间窗口内的脉冲对权重变化影响最大
    """
    
    def __init__(self, config: SynapticConfig):
        self.config = config
        self.logger = logging.getLogger(f"STDP_{id(self)}")
        
        # STDP参数
        self.a_plus = 0.005     # LTP幅度
        self.a_minus = 0.004    # LTD幅度
        self.tau_plus = 0.02    # LTP时间常数
        self.tau_minus = 0.02   # LTD时间常数
        self.tau_trace = 0.01   # 痕迹时间常数
        
        # 记录脉冲痕迹
        self.pre_traces = []    # 突触前痕迹
        self.post_traces = []   # 突触后痕迹
    
    def initialize(self, weight_matrix_shape: Tuple[int, int]) -> np.ndarray:
        """初始化权重矩阵"""
        rows, cols = weight_matrix_shape
        
        if self.config.weight_distribution == "uniform":
            weights = np.random.uniform(
                self.config.weight_mean - self.config.weight_std,
                self.config.weight_mean + self.config.weight_std,
                (rows, cols)
            )
        elif self.config.weight_distribution == "normal":
            weights = np.random.normal(
                self.config.weight_mean,
                self.config.weight_std,
                (rows, cols)
            )
        elif self.config.weight_distribution == "exponential":
            weights = np.random.exponential(
                self.config.weight_mean,
                (rows, cols)
            )
        else:  # lognormal
            weights = np.random.lognormal(
                np.log(self.config.weight_mean),
                self.config.weight_std,
                (rows, cols)
            )
        
        # 确保权重在允许范围内
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # 初始化痕迹
        self.pre_traces = np.zeros((rows, cols))
        self.post_traces = np.zeros((rows, cols))
        
        self.logger.info(f"STDP权重矩阵初始化完成：{weight_matrix_shape}")
        return weights
    
    def update_weights(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                      current_weights: np.ndarray, dt: float) -> np.ndarray:
        """更新权重"""
        rows, cols = current_weights.shape
        
        # 更新脉冲痕迹
        self.pre_traces *= np.exp(-dt / self.tau_trace)
        self.post_traces *= np.exp(-dt / self.tau_trace)
        
        # 积累新痕迹
        self.pre_traces += pre_spikes[:, np.newaxis]
        self.post_traces += post_spikes[np.newaxis, :]
        
        # 计算STDP权重变化
        weight_changes = np.zeros_like(current_weights)
        
        # LTP: 突触前脉冲在突触后脉冲之前
        ltp_contribution = self.a_plus * np.outer(self.pre_traces, post_spikes)
        ltp_contribution *= np.exp(-dt / self.tau_plus)
        
        # LTD: 突触前脉冲在突触后脉冲之后
        ltd_contribution = self.a_minus * np.outer(pre_spikes, self.post_traces)
        ltd_contribution *= np.exp(-dt / self.tau_minus)
        
        # 净权重变化
        weight_changes = ltp_contribution - ltd_contribution
        
        # 应用学习率
        weight_changes *= self.config.learning_rate
        
        # 更新权重
        new_weights = current_weights + weight_changes
        
        # 限制权重范围
        new_weights = np.clip(new_weights, self.config.min_weight, self.config.max_weight)
        
        return new_weights


class LTP_LTDPlasticity(PlasticityMechanism):
    """
    LTP/LTD (Long-Term Potentiation/Depression) 可塑性机制
    
    实现长时程增强和抑制机制，基于神经元激活频率和强度。
    """
    
    def __init__(self, config: SynapticConfig):
        self.config = config
        self.logger = logging.getLogger(f"LTP_LTD_{id(self)}")
        
        # LTP/LTD参数
        self.ltp_threshold = 0.1    # LTP阈值
        self.ltd_threshold = 0.05   # LTD阈值
        self.ltp_rate = 0.01        # LTP速率
        self.ltd_rate = 0.008       # LTD速率
        self.activity_threshold = 0.1  # 活动阈值
    
    def initialize(self, weight_matrix_shape: Tuple[int, int]) -> np.ndarray:
        """初始化权重矩阵"""
        rows, cols = weight_matrix_shape
        
        # 使用均匀分布初始化
        weights = np.random.uniform(
            self.config.weight_mean - self.config.weight_std,
            self.config.weight_mean + self.config.weight_std,
            (rows, cols)
        )
        
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        self.logger.info(f"LTP/LTD权重矩阵初始化完成：{weight_matrix_shape}")
        return weights
    
    def update_weights(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                      current_weights: np.ndarray, dt: float) -> np.ndarray:
        """更新权重"""
        rows, cols = current_weights.shape
        
        # 计算平均激活率
        pre_activity = np.mean(pre_spikes)
        post_activity = np.mean(post_spikes)
        
        # 计算权重变化
        weight_changes = np.zeros_like(current_weights)
        
        # LTP条件：高频激活
        if pre_activity > self.ltp_threshold and post_activity > self.activity_threshold:
            ltp_mask = pre_spikes[:, np.newaxis] > self.ltp_threshold
            weight_changes[ltp_mask] += self.ltp_rate * pre_spikes[:, np.newaxis][ltp_mask]
        
        # LTD条件：低频激活
        if (pre_activity < self.ltd_threshold or 
            (pre_activity > self.activity_threshold and post_activity < self.activity_threshold)):
            ltd_mask = pre_spikes[:, np.newaxis] > self.ltd_threshold
            weight_changes[ltp_mask] -= self.ltd_rate * pre_spikes[:, np.newaxis][ltp_mask]
        
        # 应用时间缩放
        weight_changes *= dt
        
        # 更新权重
        new_weights = current_weights + weight_changes
        new_weights = np.clip(new_weights, self.config.min_weight, self.config.max_weight)
        
        return new_weights


class SynapticConnection:
    """
    突触连接类
    
    实现生物启发的突触连接，包括多种可塑性机制。
    
    功能特征：
    - 多种突触类型（兴奋性、抑制性等）
    - 多种可塑性规则
    - 动态权重调整
    - 突触延迟模拟
    - 抑制机制
    """
    
    def __init__(self, network: nengo.Network, source_id: str, target_id: str,
                 config: Dict[str, Any] = None, seed: int = 42):
        """
        初始化突触连接
        
        Args:
            network: 所属的Nengo网络
            source_id: 源神经元群体ID
            target_id: 目标神经元群体ID
            config: 配置参数
            seed: 随机种子
        """
        self.network = network
        self.source_id = source_id
        self.target_id = target_id
        self.seed = seed
        self.logger = logging.getLogger(f"SynapticConn_{source_id}_to_{target_id}")
        
        # 设置配置
        self.config = self._setup_config(config)
        
        # 突触连接
        self.connection = None
        self.connection_type = self.config.connection_type
        
        # 可塑性机制
        self.plasticity_mechanism = self._create_plasticity_mechanism()
        
        # 权重管理
        self.weights = None
        self.weight_history = []
        self.weight_statistics = {
            'mean': 0.0,
            'std': 0.0,
            'sparsity': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0
        }
        
        # 性能监控
        self.activity_metrics = {
            'pre_spike_count': 0,
            'post_spike_count': 0,
            'connection_strength': 0.0,
            'plasticity_rate': 0.0
        }
        
        self.logger.info(f"创建突触连接：{source_id} -> {target_id}")
    
    def _setup_config(self, config: Dict[str, Any] = None) -> SynapticConfig:
        """设置配置"""
        synaptic_config = SynapticConfig()
        
        if config:
            for key, value in config.items():
                if hasattr(synaptic_config, key):
                    setattr(synaptic_config, key, value)
        
        return synaptic_config
    
    def _create_plasticity_mechanism(self) -> Optional[PlasticityMechanism]:
        """创建可塑性机制"""
        if not self.config.enable_plasticity:
            return None
        
        plasticity_rule = self.config.plasticity_rule
        
        if plasticity_rule == PlasticityRule.STDP:
            return STDPPlasticity(self.config)
        elif plasticity_rule == PlasticityRule.LTP_LTD:
            return LTP_LTDPlasticity(self.config)
        else:
            self.logger.warning(f"不支持的可塑性规则：{plasticity_rule}")
            return None
    
    def create_connection(self, source_ensemble: nengo.Ensemble, 
                         target_ensemble: nengo.Ensemble) -> nengo.Connection:
        """
        创建Nengo突触连接
        
        Args:
            source_ensemble: 源神经元群体
            target_ensemble: 目标神经元群体
            
        Returns:
            nengo.Connection: 突触连接对象
        """
        with self.network:
            # 选择权重分布
            weight_transform = self._create_weight_transform()
            
            # 创建连接
            self.connection = nengo.Connection(
                source_ensemble.neurons,
                target_ensemble.neurons,
                transform=weight_transform,
                synapse=self.config.synaptic_delay,
                seed=self.seed
            )
            
            # 初始化权重
            if self.plasticity_mechanism:
                weight_shape = (
                    source_ensemble.n_neurons,
                    target_ensemble.n_neurons
                )
                self.weights = self.plasticity_mechanism.initialize(weight_shape)
        
        self.logger.info(f"创建Nengo连接：{source_ensemble.n_neurons} -> {target_ensemble.n_neurons}")
        return self.connection
    
    def _create_weight_transform(self) -> Union[float, np.ndarray]:
        """创建权重变换矩阵"""
        if self.connection_type == ConnectionType.EXCITATORY:
            # 兴奋性连接：正权重
            if self.config.weight_distribution == "uniform":
                return nengo.dists.Uniform(
                    self.config.weight_mean, 
                    self.config.max_weight
                )
            else:
                return self.config.initial_weight
                
        elif self.connection_type == ConnectionType.INHIBITORY:
            # 抑制性连接：负权重
            return -self.config.initial_weight
            
        else:
            # 其他类型
            return self.config.initial_weight
    
    def update_synaptic_weights(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                               dt: float) -> np.ndarray:
        """
        更新突触权重
        
        Args:
            pre_spikes: 突触前神经元脉冲
            post_spikes: 突触后神经元脉冲  
            dt: 时间步长
            
        Returns:
            更新后的权重矩阵
        """
        if not self.plasticity_mechanism or self.weights is None:
            return self.weights
        
        # 保存旧权重
        old_weights = self.weights.copy()
        
        # 更新权重
        new_weights = self.plasticity_mechanism.update_weights(
            pre_spikes, post_spikes, self.weights, dt
        )
        
        # 计算权重变化统计
        weight_change = np.mean(np.abs(new_weights - old_weights))
        self.activity_metrics['plasticity_rate'] = weight_change / dt
        
        # 更新权重
        self.weights = new_weights
        self.weight_history.append(new_weights.copy())
        
        # 保持权重历史在缓存大小内
        if len(self.weight_history) > self.config.cache_size:
            self.weight_history.pop(0)
        
        # 更新权重统计
        self._update_weight_statistics()
        
        return self.weights
    
    def _update_weight_statistics(self) -> None:
        """更新权重统计信息"""
        if self.weights is not None:
            self.weight_statistics.update({
                'mean': np.mean(self.weights),
                'std': np.std(self.weights),
                'sparsity': np.mean(self.weights < 0.01),
                'max_weight': np.max(self.weights),
                'min_weight': np.min(self.weights)
            })
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'connection_type': self.connection_type.value,
            'plasticity_rule': self.config.plasticity_rule.value if self.plasticity_mechanism else None,
            'weight_statistics': self.weight_statistics.copy(),
            'activity_metrics': self.activity_metrics.copy(),
            'config': {
                'max_weight': self.config.max_weight,
                'min_weight': self.config.min_weight,
                'learning_rate': self.config.learning_rate,
                'synaptic_delay': self.config.synaptic_delay
            }
        }
    
    def add_leaky_integrator(self) -> nengo.Node:
        """添加泄漏积分器以模拟突触动力学"""
        with self.network:
            leaky_integrator = nengo.Node(
                output=lambda t, x: x[0] * np.exp(-x[1] / self.config.decay_time),
                size_in=2,
                label=f"leaky_integrator_{self.source_id}_to_{self.target_id}"
            )
            
            # 创建从输入到泄漏积分器的连接
            input_connection = nengo.Connection(
                self.connection.pre_obj,
                leaky_integrator[0],
                synapse=0
            )
            
            # 创建从泄漏积分器到输出的连接
            output_connection = nengo.Connection(
                leaky_integrator,
                self.connection.post_obj,
                synapse=self.config.synaptic_delay
            )
            
            self.logger.info("添加泄漏积分器")
            return leaky_integrator
    
    def add_short_term_plasticity(self, facilitation: float = 0.1, 
                                 depression: float = 0.1) -> None:
        """添加短期可塑性机制"""
        # 这里可以实现短期可塑性，如 facilitation 和 depression
        self.logger.info(f"添加短期可塑性：facilitation={facilitation}, depression={depression}")
    
    def analyze_weight_dynamics(self) -> Dict[str, Any]:
        """分析权重动态"""
        if not self.weight_history:
            return {'error': '没有权重历史数据'}
        
        # 权重轨迹分析
        weight_trajectories = np.array(self.weight_history)
        
        analysis = {
            'weight_evolution': {
                'trajectory_length': len(self.weight_history),
                'weight_range': {
                    'min': np.min(weight_trajectories),
                    'max': np.max(weight_trajectories)
                },
                'stability': np.std(weight_trajectories) / (np.mean(weight_trajectories) + 1e-8)
            },
            'plasticity_dynamics': {
                'change_rate': np.mean(np.diff(weight_trajectories, axis=0)),
                'convergence': self._analyze_convergence(weight_trajectories)
            },
            'final_weights': self.weights.tolist() if self.weights is not None else None
        }
        
        return analysis
    
    def _analyze_convergence(self, weight_trajectories: np.ndarray) -> float:
        """分析权重收敛"""
        if len(weight_trajectories) < 10:
            return 0.0
        
        # 计算权重变化趋势
        recent_changes = weight_trajectories[-10:] - weight_trajectories[-20:-10]
        convergence_score = 1.0 / (1.0 + np.mean(np.abs(recent_changes)))
        
        return convergence_score
    
    def reset_plasticity(self) -> None:
        """重置可塑性"""
        if self.plasticity_mechanism and hasattr(self.plasticity_mechanism, 'pre_traces'):
            # 重置STDP痕迹
            self.plasticity_mechanism.pre_traces.fill(0)
            self.plasticity_mechanism.post_traces.fill(0)
        
        # 重置权重历史
        self.weight_history = []
        self.activity_metrics['plasticity_rate'] = 0.0
        
        self.logger.info("突触可塑性已重置")
    
    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_connection_info()
        return (f"SynapticConnection({self.source_id}->{self.target_id}, "
                f"type={info['connection_type']}, "
                f"weights={info['weight_statistics']['mean']:.3f}±{info['weight_statistics']['std']:.3f})")


# 便利函数
def create_excitatory_connection(network: nengo.Network, source_id: str, target_id: str,
                                config: Dict[str, Any] = None) -> SynapticConnection:
    """创建兴奋性突触连接"""
    if config is None:
        config = {}
    config['connection_type'] = ConnectionType.EXCITATORY
    
    return SynapticConnection(network, source_id, target_id, config)


def create_inhibitory_connection(network: nengo.Network, source_id: str, target_id: str,
                                config: Dict[str, Any] = None) -> SynapticConnection:
    """创建抑制性突触连接"""
    if config is None:
        config = {}
    config['connection_type'] = ConnectionType.INHIBITORY
    
    return SynapticConnection(network, source_id, target_id, config)