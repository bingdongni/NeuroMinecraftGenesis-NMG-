"""
脉冲输入(Spiking Input)模块
提供各种类型的脉冲输入接口，支持外部刺激的注入和动态控制

功能特性：
- 多种输入模式（恒定、脉冲、噪声、模式输入等）
- 时间序列输入生成
- 外部数据输入
- 输入模式和强度的动态控制
- 输入同步和协调机制

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
from scipy import signal


class InputMode(Enum):
    """输入模式枚举"""
    CONSTANT = "constant"           # 恒定输入
    PULSE = "pulse"                 # 脉冲输入
    BURST = "burst"                 # 爆发式输入
    NOISE = "noise"                 # 噪声输入
    PATTERN = "pattern"             # 模式输入
    TEMPORAL = "temporal"           # 时间序列输入
    EXTERNAL = "external"           # 外部数据输入
    SINUSOID = "sinusoid"           # 正弦波输入


class InputType(Enum):
    """输入类型枚举"""
    SENSORY = "sensory"             # 感觉输入
    MODULATORY = "modulatory"       # 调制输入
    STIMULUS = "stimulus"           # 刺激输入
    NOISE = "noise"                 # 噪声输入
    TEACHING = "teaching"           # 教师信号输入


@dataclass
class SpikingInputConfig:
    """脉冲输入配置参数"""
    # 基础参数
    input_mode: InputMode = InputMode.CONSTANT
    input_type: InputType = InputType.SENSORY
    
    # 时间参数
    start_time: float = 0.0         # 开始时间(s)
    duration: float = 10.0          # 持续时间(s)
    dt: float = 0.001               # 时间步长(s)
    
    # 输入参数
    amplitude: float = 1.0          # 输入幅度
    frequency: float = 10.0         # 频率(Hz)
    phase: float = 0.0              # 相位
    
    # 噪声参数
    noise_level: float = 0.1        # 噪声水平
    noise_distribution: str = "gaussian"  # "gaussian", "uniform", "poisson"
    
    # 脉冲参数
    pulse_width: float = 0.01       # 脉冲宽度(s)
    pulse_rate: float = 10.0        # 脉冲频率(Hz)
    burst_length: float = 0.1       # 爆发长度(s)
    burst_rate: float = 1.0         # 爆发频率(Hz)
    
    # 模式参数
    pattern_duration: float = 1.0   # 模式持续时间(s)
    pattern_repeat: bool = True     # 是否重复模式
    
    # 外部数据参数
    external_data: np.ndarray = None  # 外部数据
    external_sampling_rate: float = 1000.0  # 外部数据采样率
    
    # 性能参数
    seed: int = 42                  # 随机种子
    cache_enabled: bool = True      # 是否启用缓存


class SpikingInput:
    """
    脉冲输入类
    
    提供各种类型的脉冲输入，支持外部刺激的注入和动态控制。
    
    支持的输入类型：
    - 恒定输入：基础输入水平
    - 脉冲输入：周期性脉冲
    - 噪声输入：高斯或均匀噪声
    - 模式输入：预定义的时间模式
    - 外部输入：真实数据驱动
    - 教师信号：有监督学习输入
    """
    
    def __init__(self, network: nengo.Network, input_id: str,
                 config: Dict[str, Any] = None, seed: int = 42):
        """
        初始化脉冲输入
        
        Args:
            network: 所属的Nengo网络
            input_id: 输入唯一标识符
            config: 配置参数
            seed: 随机种子
        """
        self.network = network
        self.input_id = input_id
        self.seed = seed
        self.logger = logging.getLogger(f"SpikingInput_{input_id}")
        
        # 配置
        self.config = self._setup_config(config)
        
        # 输入对象
        self.input_node = None
        self.input_array = None
        
        # 输入属性
        self.input_mode = self.config.input_mode
        self.input_type = self.config.input_type
        self.dimensions = 1  # 默认一维输入
        
        # 缓存管理
        self._cache = {}
        self._cache_lock = threading.RLock()
        
        # 活动监控
        self.activity_log = {
            'generation_time': 0.0,
            'last_input_value': 0.0,
            'average_input': 0.0,
            'input_variance': 0.0
        }
        
        # 统计信息
        self.statistics = {
            'total_generated_inputs': 0,
            'max_input_value': 0.0,
            'min_input_value': 0.0,
            'input_sparsity': 0.0
        }
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 创建输入节点
        self._create_input_node()
        
        self.logger.info(f"创建脉冲输入：{input_id} "
                        f"(模式：{self.input_mode.value}, 类型：{self.input_type.value})")
    
    def _setup_config(self, config: Dict[str, Any] = None) -> SpikingInputConfig:
        """设置配置参数"""
        input_config = SpikingInputConfig()
        
        if config:
            for key, value in config.items():
                if hasattr(input_config, key):
                    setattr(input_config, key, value)
        
        return input_config
    
    def _create_input_node(self) -> None:
        """创建输入节点"""
        with self.network:
            # 生成输入时间序列
            input_sequence = self._generate_input_sequence()
            
            # 创建输入节点
            self.input_node = nengo.Node(
                output=self._get_input_output_function(),
                size_in=0,
                size_out=self.dimensions,
                label=f"input_{self.input_id}"
            )
    
    def _generate_input_sequence(self) -> np.ndarray:
        """生成输入时间序列"""
        start_time = time.time()
        
        # 时间轴
        t_start = self.config.start_time
        t_end = t_start + self.config.duration
        t = np.arange(t_start, t_end, self.config.dt)
        
        # 根据输入模式生成序列
        if self.input_mode == InputMode.CONSTANT:
            input_sequence = self._generate_constant_input(t)
        elif self.input_mode == InputMode.PULSE:
            input_sequence = self._generate_pulse_input(t)
        elif self.input_mode == InputMode.BURST:
            input_sequence = self._generate_burst_input(t)
        elif self.input_mode == InputMode.NOISE:
            input_sequence = self._generate_noise_input(t)
        elif self.input_mode == InputMode.PATTERN:
            input_sequence = self._generate_pattern_input(t)
        elif self.input_mode == InputMode.SINUSOID:
            input_sequence = self._generate_sinusoid_input(t)
        elif self.input_mode == InputMode.TEMPORAL:
            input_sequence = self._generate_temporal_input(t)
        elif self.input_mode == InputMode.EXTERNAL:
            input_sequence = self._generate_external_input(t)
        else:
            input_sequence = self._generate_constant_input(t)
        
        # 缓存输入序列
        if self.config.cache_enabled:
            with self._cache_lock:
                self._cache['input_sequence'] = input_sequence
                self._cache['time_axis'] = t
        
        # 更新活动日志
        generation_time = time.time() - start_time
        self.activity_log['generation_time'] = generation_time
        
        self.input_array = input_sequence
        
        # 更新统计
        self.statistics['total_generated_inputs'] += len(input_sequence)
        self.statistics['max_input_value'] = max(self.statistics['max_input_value'], np.max(input_sequence))
        self.statistics['min_input_value'] = min(self.statistics['min_input_value'], np.min(input_sequence))
        
        return input_sequence
    
    def _generate_constant_input(self, t: np.ndarray) -> np.ndarray:
        """生成恒定输入"""
        return np.full_like(t, self.config.amplitude)
    
    def _generate_pulse_input(self, t: np.ndarray) -> np.ndarray:
        """生成脉冲输入"""
        # 计算脉冲周期
        period = 1.0 / self.config.pulse_rate
        
        # 生成周期性脉冲
        pulse_pattern = np.zeros_like(t)
        pulse_width_samples = int(self.config.pulse_width / self.config.dt)
        
        for pulse_start in np.arange(0, self.config.duration, period):
            start_idx = int(pulse_start / self.config.dt)
            end_idx = min(start_idx + pulse_width_samples, len(t))
            if start_idx < len(t):
                pulse_pattern[start_idx:end_idx] = self.config.amplitude
        
        # 添加噪声
        if self.config.noise_level > 0:
            noise = self._generate_noise(t.shape)
            pulse_pattern += noise * self.config.noise_level
        
        return pulse_pattern
    
    def _generate_burst_input(self, t: np.ndarray) -> np.ndarray:
        """生成爆发式输入"""
        burst_pattern = np.zeros_like(t)
        burst_width_samples = int(self.config.burst_length / self.config.dt)
        
        for burst_start in np.arange(0, self.config.duration, 1.0 / self.config.burst_rate):
            start_idx = int(burst_start / self.config.dt)
            end_idx = min(start_idx + burst_width_samples, len(t))
            if start_idx < len(t):
                burst_pattern[start_idx:end_idx] = self.config.amplitude
        
        return burst_pattern
    
    def _generate_noise_input(self, t: np.ndarray) -> np.ndarray:
        """生成噪声输入"""
        noise = self._generate_noise(t.shape)
        return noise * self.config.amplitude
    
    def _generate_pattern_input(self, t: np.ndarray) -> np.ndarray:
        """生成模式输入"""
        pattern_length_samples = int(self.config.pattern_duration / self.config.dt)
        total_samples = len(t)
        
        pattern_input = np.zeros_like(t)
        
        # 生成单个模式周期
        pattern_cycle = self._generate_single_pattern(pattern_length_samples)
        
        # 重复模式
        if self.config.pattern_repeat:
            for i in range(0, total_samples, pattern_length_samples):
                end_idx = min(i + pattern_length_samples, total_samples)
                pattern_input[i:end_idx] = pattern_cycle[:end_idx-i]
        else:
            # 单次模式
            if len(pattern_cycle) > 0:
                pattern_input[:min(len(pattern_cycle), total_samples)] = pattern_cycle[:total_samples]
        
        return pattern_input
    
    def _generate_single_pattern(self, length: int) -> np.ndarray:
        """生成单个模式周期"""
        if length <= 0:
            return np.array([])
        
        # 简单的模式：渐增-渐减
        t_pattern = np.linspace(0, 1, length)
        pattern = self.config.amplitude * np.sin(2 * np.pi * t_pattern)
        
        # 添加噪声
        if self.config.noise_level > 0:
            noise = self._generate_noise((length,))
            pattern += noise * self.config.noise_level
        
        return pattern
    
    def _generate_sinusoid_input(self, t: np.ndarray) -> np.ndarray:
        """生成正弦波输入"""
        omega = 2 * np.pi * self.config.frequency
        signal = self.config.amplitude * np.sin(omega * t + self.config.phase)
        
        # 添加噪声
        if self.config.noise_level > 0:
            noise = self._generate_noise(t.shape)
            signal += noise * self.config.noise_level
        
        return signal
    
    def _generate_temporal_input(self, t: np.ndarray) -> np.ndarray:
        """生成时间序列输入"""
        # 这里可以集成更复杂的时间序列模型
        # 暂时使用简单的多频率组合
        signal = np.zeros_like(t)
        
        # 多频率成分
        frequencies = [1.0, 3.0, 7.0, 15.0]  # Hz
        amplitudes = [0.5, 0.3, 0.2, 0.1]
        
        for freq, amp in zip(frequencies, amplitudes):
            omega = 2 * np.pi * freq
            signal += amp * np.sin(omega * t)
        
        # 归一化
        signal = signal * self.config.amplitude / np.max(np.abs(signal))
        
        return signal
    
    def _generate_external_input(self, t: np.ndarray) -> np.ndarray:
        """生成外部数据输入"""
        if self.config.external_data is None:
            self.logger.warning("未提供外部数据，使用默认输入")
            return self._generate_constant_input(t)
        
        # 重采样外部数据到目标时间轴
        external_data = self.config.external_data
        external_t = np.linspace(0, self.config.duration, len(external_data))
        
        # 插值
        input_sequence = np.interp(t, external_t, external_data)
        
        # 缩放和偏移
        if self.config.amplitude != 1.0:
            input_sequence = input_sequence * self.config.amplitude
        
        # 添加噪声
        if self.config.noise_level > 0:
            noise = self._generate_noise(t.shape)
            input_sequence += noise * self.config.noise_level
        
        return input_sequence
    
    def _generate_noise(self, shape: Tuple) -> np.ndarray:
        """生成噪声"""
        if self.config.noise_distribution == "gaussian":
            noise = np.random.normal(0, 1, shape)
        elif self.config.noise_distribution == "uniform":
            noise = np.random.uniform(-1, 1, shape)
        elif self.config.noise_distribution == "poisson":
            # 生成泊松噪声（适合理性神经元模型）
            rate = np.abs(self.config.amplitude)
            noise = np.random.poisson(rate, shape).astype(float) - rate
        else:
            noise = np.random.normal(0, 1, shape)
        
        return noise
    
    def _get_input_output_function(self) -> Callable:
        """获取输入节点的输出函数"""
        def input_function(t):
            # 查找缓存的输入序列
            if self.config.cache_enabled:
                with self._cache_lock:
                    if 'input_sequence' in self._cache and 'time_axis' in self._cache:
                        input_sequence = self._cache['input_sequence']
                        time_axis = self._cache['time_axis']
                        
                        # 插值获取当前时间的输入值
                        if len(time_axis) > 0:
                            current_input = np.interp(t, time_axis, input_sequence)
                            self.activity_log['last_input_value'] = current_input
                            return current_input
            
            # 备用计算（如果缓存不可用）
            return self._calculate_current_input(t)
        
        return input_function
    
    def _calculate_current_input(self, t: float) -> float:
        """计算当前时间点的输入值"""
        if self.input_mode == InputMode.CONSTANT:
            return self.config.amplitude
        elif self.input_mode == InputMode.PULSE:
            period = 1.0 / self.config.pulse_rate
            if (t % period) < self.config.pulse_width:
                return self.config.amplitude
            else:
                return 0.0
        elif self.input_mode == InputMode.SINUSOID:
            omega = 2 * np.pi * self.config.frequency
            return self.config.amplitude * np.sin(omega * t + self.config.phase)
        else:
            return self.config.amplitude
    
    def update_input_parameters(self, **kwargs) -> None:
        """动态更新输入参数"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    self.logger.info(f"更新输入参数 {key}: {old_value} -> {value}")
            
            # 清除缓存以重新生成
            if self.config.cache_enabled:
                with self._cache_lock:
                    self._cache.clear()
            
            # 重新生成输入序列
            if self.input_array is not None:
                self._generate_input_sequence()
    
    def set_external_data(self, data: np.ndarray, sampling_rate: float = None) -> None:
        """设置外部数据"""
        with self._lock:
            if sampling_rate is not None:
                self.config.external_sampling_rate = sampling_rate
            
            self.config.external_data = data.copy()
            
            # 清除缓存
            if self.config.cache_enabled:
                with self._cache_lock:
                    self._cache.clear()
            
            self.logger.info(f"设置外部数据：{len(data)} 个采样点")
    
    def modulate_amplitude(self, modulation_signal: Callable[[float], float]) -> None:
        """幅度调制"""
        # 保存原始输出函数
        original_output = self.input_node.output
        
        def modulated_output(t):
            base_input = original_output(t)
            modulation = modulation_signal(t)
            return base_input * modulation
        
        self.input_node.output = modulated_output
        self.logger.info("应用幅度调制")
    
    def get_input_info(self) -> Dict[str, Any]:
        """获取输入信息"""
        return {
            'input_id': self.input_id,
            'input_mode': self.input_mode.value,
            'input_type': self.input_type.value,
            'duration': self.config.duration,
            'amplitude': self.config.amplitude,
            'statistics': self.statistics.copy(),
            'activity_log': self.activity_log.copy(),
            'cache_enabled': self.config.cache_enabled,
            'cache_size': len(self._cache) if self.config.cache_enabled else 0
        }
    
    def get_input_statistics(self) -> Dict[str, float]:
        """获取输入统计信息"""
        if self.input_array is not None:
            self.statistics.update({
                'average_input': np.mean(self.input_array),
                'input_variance': np.var(self.input_array),
                'input_sparsity': np.mean(self.input_array == 0),
                'max_input_value': np.max(self.input_array),
                'min_input_value': np.min(self.input_array)
            })
        
        return self.statistics.copy()
    
    def plot_input_pattern(self, save_path: str = None) -> None:
        """绘制输入模式（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            
            if self.input_array is None:
                self.logger.warning("没有输入数据可绘制")
                return
            
            t = np.linspace(0, self.config.duration, len(self.input_array))
            
            plt.figure(figsize=(12, 6))
            plt.plot(t, self.input_array, linewidth=1)
            plt.title(f"脉冲输入模式：{self.input_id}")
            plt.xlabel("时间 (s)")
            plt.ylabel("输入幅度")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"输入模式图已保存：{save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib未安装，无法绘制输入模式")
    
    def reset(self) -> None:
        """重置输入"""
        with self._lock:
            self.activity_log = {
                'generation_time': 0.0,
                'last_input_value': 0.0,
                'average_input': 0.0,
                'input_variance': 0.0
            }
            
            self.statistics = {
                'total_generated_inputs': 0,
                'max_input_value': 0.0,
                'min_input_value': 0.0,
                'input_sparsity': 0.0
            }
            
            if self.config.cache_enabled:
                with self._cache_lock:
                    self._cache.clear()
            
            self.logger.info(f"重置输入：{self.input_id}")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"SpikingInput({self.input_id}, "
                f"mode={self.input_mode.value}, "
                f"type={self.input_type.value}, "
                f"duration={self.config.duration}s)")