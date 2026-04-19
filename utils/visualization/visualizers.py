#!/usr/bin/env python3
"""
NeuroMinecraft Genesis - 实时神经网络可视化
提供脉冲神经网络和注意力机制的可视化
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


class NeuralVisualizer:
    """神经网络可视化器"""

    def __init__(self, n_neurons: int = 100):
        self.n_neurons = n_neurons
        self.spike_history = []
        self.neuron_positions = self._generate_positions()

    def _generate_positions(self) -> np.ndarray:
        """生成神经元位置"""
        positions = np.random.rand(self.n_neurons, 3)
        return positions

    def update_spikes(self, spike_times: np.ndarray):
        """更新脉冲数据"""
        self.spike_history.append(spike_times)

        # 保持历史长度
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)

    def create_3d_visualization(self) -> go.Figure:
        """创建3D神经元可视化"""
        fig = go.Figure()

        # 绘制神经元
        colors = ['red' if self.spike_history[-1][i] > 0.5
                  else 'blue' for i in range(self.n_neurons)]

        fig.add_trace(go.Scatter3d(
            x=self.neuron_positions[:, 0],
            y=self.neuron_positions[:, 1],
            z=self.neuron_positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8
            )
        ))

        fig.update_layout(
            title='脉冲神经网络活动',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        return fig

    def create_spike_raster(self) -> go.Figure:
        """创建脉冲光栅图"""
        fig = go.Figure()

        for i, spikes in enumerate(self.spike_history):
            spike_times = np.where(spikes > 0.5)[0]
            fig.add_trace(go.Scatter(
                x=spike_times,
                y=[i] * len(spike_times),
                mode='markers',
                marker=dict(size=3, color='red')
            ))

        fig.update_layout(
            title='脉冲光栅图',
            xaxis_title='时间',
            yaxis_title='神经元'
        )

        return fig


class AttentionVisualizer:
    """注意力可视化器"""

    def __init__(self, seq_len: int = 10, n_heads: int = 8):
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.attention_weights = []

    def update_attention(self, weights: np.ndarray):
        """更新注意力权重"""
        self.attention_weights.append(weights)

        if len(self.attention_weights) > 50:
            self.attention_weights.pop(0)

    def create_attention_heatmap(self) -> go.Figure:
        """创建注意力热力图"""
        if not self.attention_weights:
            return go.Figure()

        weights = self.attention_weights[-1]

        fig = go.Figure(data=go.Heatmap(
            z=weights,
            colorscale='Viridis',
            showscale=True
        ))

        fig.update_layout(
            title='注意力权重热力图',
            xaxis_title='Key',
            yaxis_title='Query'
        )

        return fig

    def create_head_comparison(self) -> go.Figure:
        """创建多头注意力比较图"""
        if not self.attention_weights:
            return go.Figure()

        weights = self.attention_weights[-1]

        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[f'Head {i}' for i in range(self.n_heads)]
        )

        for i in range(self.n_heads):
            row = i // 4 + 1
            col = i % 4 + 1

            fig.add_trace(
                go.Heatmap(z=weights[i]),
                row=row, col=col
            )

        fig.update_layout(
            title='多头注意力对比',
            height=600
        )

        return fig


class EvolutionVisualizer:
    """进化过程可视化器"""

    def __init__(self):
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

    def add_generation(self, generation: int, best: float, avg: float, diversity: float):
        """添加一代数据"""
        self.history['generations'].append(generation)
        self.history['best_fitness'].append(best)
        self.history['avg_fitness'].append(avg)
        self.history['diversity'].append(diversity)

    def create_fitness_plot(self) -> go.Figure:
        """创建适应度曲线"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.history['generations'],
            y=self.history['best_fitness'],
            mode='lines+markers',
            name='最佳适应度',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=self.history['generations'],
            y=self.history['avg_fitness'],
            mode='lines+markers',
            name='平均适应度',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='适应度进化曲线',
            xaxis_title='代数',
            yaxis_title='适应度',
            hovermode='x unified'
        )

        return fig

    def create_diversity_plot(self) -> go.Figure:
        """创建多样性曲线"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.history['generations'],
            y=self.history['diversity'],
            mode='lines+markers',
            name='种群多样性',
            fill='tozeroy',
            line=dict(color='green')
        ))

        fig.update_layout(
            title='种群多样性变化',
            xaxis_title='代数',
            yaxis_title='多样性'
        )

        return fig

    def create_pareto_front(self, objectives: list) -> go.Figure:
        """创建帕累托前沿"""
        fig = go.Figure()

        for obj in objectives:
            fig.add_trace(go.Scatter(
                x=[o[0] for o in obj],
                y=[o[1] for o in obj],
                mode='markers',
                marker=dict(size=10)
            ))

        fig.update_layout(
            title='帕累托前沿',
            xaxis_title='目标1',
            yaxis_title='目标2'
        )

        return fig


class QuantumVisualizer:
    """量子系统可视化器"""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.amplitudes_history = []

    def update_amplitudes(self, amplitudes: np.ndarray):
        """更新量子态振幅"""
        self.amplitudes_history.append(amplitudes)

        if len(self.amplitudes_history) > 50:
            self.amplitudes_history.pop(0)

    def create_bloch_sphere(self) -> go.Figure:
        """创建布洛赫球可视化"""
        # 简化的布洛赫球
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        fig = go.Figure()

        # 球面
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale='Blues',
            opacity=0.3,
            showscale=False
        ))

        # 状态向量
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=3)
        ))

        fig.update_layout(
            title='量子态布洛赫球表示',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )

        return fig

    def create_amplitude_plot(self) -> go.Figure:
        """创建振幅可视化"""
        if not self.amplitudes_history:
            return go.Figure()

        amplitudes = self.amplitudes_history[-1]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['振幅模', '相位']
        )

        probs = np.abs(amplitudes) ** 2
        phases = np.angle(amplitudes)

        fig.add_trace(
            go.Bar(x=list(range(len(probs))), y=probs, name='概率'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=list(range(len(phases))), y=phases, name='相位'),
            row=1, col=2
        )

        fig.update_layout(
            title='量子态测量概率',
            height=400
        )

        return fig


def demo_visualization():
    """演示可视化功能"""
    print("=" * 60)
    print(" 神经网络和量子可视化演示 ")
    print("=" * 60)

    # 创建可视化器
    neural = NeuralVisualizer(n_neurons=50)
    attention = AttentionVisualizer(seq_len=10, n_heads=8)
    evolution = EvolutionVisualizer()
    quantum = QuantumVisualizer(n_qubits=4)

    # 生成数据
    print("\n📊 生成模拟数据...")

    for i in range(50):
        # 神经脉冲
        spikes = np.random.rand(50) > 0.7
        neural.update_spikes(spikes)

        # 注意力
        att_weights = np.random.rand(8, 10, 10)
        attention.update_attention(att_weights)

        # 进化
        evolution.add_generation(
            generation=i,
            best=np.random.uniform(0.5, 0.9),
            avg=np.random.uniform(0.3, 0.7),
            diversity=np.random.uniform(0.2, 0.8)
        )

        # 量子态
        amps = np.random.rand(16) + 1j * np.random.rand(16)
        amps = amps / np.sqrt(np.sum(np.abs(amps) ** 2))
        quantum.update_amplitudes(amps)

        time.sleep(0.1)

    print("\n✅ 可视化数据生成完成！")
    print("\n生成的可视化:")
    print("  1. 3D神经元网络活动")
    print("  2. 脉冲光栅图")
    print("  3. 注意力热力图")
    print("  4. 进化适应度曲线")
    print("  5. 量子态布洛赫球")
    print("  6. 量子振幅概率图")

    return neural, attention, evolution, quantum


if __name__ == "__main__":
    demo_visualization()
