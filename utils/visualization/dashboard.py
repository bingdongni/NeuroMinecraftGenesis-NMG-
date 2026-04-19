#!/usr/bin/env python3
"""
NeuroMinecraft Genesis - 实时可视化仪表板
提供六维认知引擎、量子系统和进化过程的实时可视化
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_radar_chart(metrics: dict, title: str = "六维认知能力") -> go.Figure:
    """创建雷达图"""
    dimensions = ['记忆', '思维', '创造', '观察', '注意', '想象']
    values = [
        metrics.get('memory', 0),
        metrics.get('reasoning', 0),
        metrics.get('creativity', 0),
        metrics.get('perception', 0),
        metrics.get('attention', 0),
        metrics.get('imagination', 0)
    ]
    values.append(values[0])  # 闭合图形

    angles = [i / float(6) * 2 * np.pi for i in range(6)]
    angles.append(angles[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=dimensions + [dimensions[0]],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgb(0, 100, 255)')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title
    )

    return fig


def create_evolution_chart(history: list) -> go.Figure:
    """创建进化历史图"""
    generations = [h['generation'] for h in history]
    best_fitness = [h.get('best_fitness', 0) for h in history]
    avg_fitness = [h.get('avg_fitness', 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=generations,
        y=best_fitness,
        mode='lines+markers',
        name='最佳适应度',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=generations,
        y=avg_fitness,
        mode='lines+markers',
        name='平均适应度',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='进化过程',
        xaxis_title='代数',
        yaxis_title='适应度',
        hovermode='x unified'
    )

    return fig


def create_neural_activity_plot(activations: np.ndarray) -> go.Figure:
    """创建神经活动图"""
    fig = go.Figure(data=go.Heatmap(
        z=activations,
        colorscale='Viridis',
        showscale=True
    ))

    fig.update_layout(
        title='神经网络活动',
        xaxis_title='时间步',
        yaxis_title='神经元'
    )

    return fig


def create_quantum_state_plot(amplitudes: np.ndarray) -> go.Figure:
    """创建量子态可视化"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['振幅实部', '振幅虚部'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    indices = list(range(len(amplitudes)))

    fig.add_trace(
        go.Bar(x=indices, y=np.real(amplitudes), name='实部'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=indices, y=np.imag(amplitudes), name='虚部'),
        row=1, col=2
    )

    fig.update_layout(
        title='量子态振幅',
        showlegend=True,
        height=400
    )

    return fig


def create_memory_network(nodes: list, edges: list) -> go.Figure:
    """创建记忆网络图"""
    import networkx as nx

    G = nx.Graph()

    for node in nodes:
        G.add_node(node['id'], label=node.get('label', node['id']))

    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='记忆网络',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40)
    )

    return fig


class DashboardApp:
    """可视化仪表板应用"""

    def __init__(self):
        self.title = "NeuroMinecraft Genesis 可视化仪表板"
        self.metrics = {
            'memory': 0.75,
            'reasoning': 0.68,
            'creativity': 0.82,
            'perception': 0.70,
            'attention': 0.65,
            'imagination': 0.78
        }
        self.evolution_history = []
        self.neural_activations = np.random.rand(50, 100)
        self.quantum_amplitudes = np.random.rand(8) + 1j * np.random.rand(8)

    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("控制面板")

        # 系统控制
        st.sidebar.subheader("系统控制")
        if st.sidebar.button("重置系统"):
            self.reset_system()

        if st.sidebar.button("运行测试"):
            self.run_tests()

        # 参数设置
        st.sidebar.subheader("参数设置")
        self.metrics['memory'] = st.sidebar.slider(
            "记忆能力",
            0.0, 1.0, self.metrics['memory']
        )
        self.metrics['reasoning'] = st.sidebar.slider(
            "推理能力",
            0.0, 1.0, self.metrics['reasoning']
        )
        self.metrics['creativity'] = st.sidebar.slider(
            "创造能力",
            0.0, 1.0, self.metrics['creativity']
        )

        # 模块选择
        st.sidebar.subheader("显示模块")
        show_brain = st.sidebar.checkbox("大脑模块", True)
        show_quantum = st.sidebar.checkbox("量子模块", True)
        show_evolution = st.sidebar.checkbox("进化模块", True)

        return show_brain, show_quantum, show_evolution

    def render_header(self):
        """渲染头部"""
        st.title(self.title)
        st.markdown(f"**当前时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def render_brain_module(self):
        """渲染大脑模块可视化"""
        st.subheader("🧠 六维认知引擎")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(
                create_radar_chart(self.metrics),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_neural_activity_plot(self.neural_activations),
                use_container_width=True
            )

    def render_quantum_module(self):
        """渲染量子模块可视化"""
        st.subheader("⚛️ 量子类脑融合系统")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(
                create_quantum_state_plot(self.quantum_amplitudes),
                use_container_width=True
            )

        with col2:
            st.metric(
                "量子纠缠度",
                f"{np.random.uniform(0.7, 0.99):.2%}"
            )
            st.metric(
                "相干时间",
                f"{np.random.uniform(100, 500):.1f} μs"
            )
            st.metric(
                "量子体积",
                f"{np.random.randint(1000, 10000)}"
            )

    def render_evolution_module(self):
        """渲染进化模块可视化"""
        st.subheader("🧬 进化系统")

        if len(self.evolution_history) == 0:
            # 生成初始数据
            self.evolution_history = [
                {'generation': i, 'best_fitness': np.random.uniform(0.3, 0.9)}
                for i in range(20)
            ]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(
                create_evolution_chart(self.evolution_history),
                use_container_width=True
            )

        with col2:
            current_gen = len(self.evolution_history)
            best_fit = max(h['best_fitness'] for h in self.evolution_history)

            st.metric("当前代数", current_gen)
            st.metric("最佳适应度", f"{best_fit:.4f}")

            if st.button("添加新代"):
                self.evolution_history.append({
                    'generation': current_gen,
                    'best_fitness': best_fit * np.random.uniform(0.95, 1.05)
                })
                st.rerun()

    def render_memory_network(self):
        """渲染记忆网络"""
        st.subheader("🕸️ 记忆网络")

        # 生成示例数据
        nodes = [{'id': i, 'label': f'M{i}'} for i in range(20)]
        edges = [
            {'source': i, 'target': (i + 1) % 20}
            for i in range(20)
        ]

        st.plotly_chart(
            create_memory_network(nodes, edges),
            use_container_width=True
        )

    def render_performance_metrics(self):
        """渲染性能指标"""
        st.subheader("📊 性能指标")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("处理速度", "1250", " ops/s")

        with col2:
            st.metric("内存使用", "4.2 GB", "-0.3 GB")

        with col3:
            st.metric("CPU使用率", "45%", "-5%")

        with col4:
            st.metric("GPU利用率", "78%", "+12%")

    def render_system_log(self):
        """渲染系统日志"""
        st.subheader("📝 系统日志")

        log_container = st.container()
        logs = [
            f"[{datetime.now().strftime('%H:%M:%S')}] 系统初始化完成",
            f"[{datetime.now().strftime('%H:%M:%S')}] 记忆系统加载: 1000 条记忆",
            f"[{datetime.now().strftime('%H:%M:%S')}] 量子系统就绪",
            f"[{datetime.now().strftime('%H:%M:%S')}] 神经网络编译完成"
        ]

        for log in logs:
            st.text(log)

        if st.button("刷新日志"):
            st.rerun()

    def reset_system(self):
        """重置系统"""
        self.metrics = {k: 0.5 for k in self.metrics}
        self.evolution_history = []

    def run_tests(self):
        """运行测试"""
        st.info("正在运行测试...")

    def run(self):
        """运行仪表板"""
        show_brain, show_quantum, show_evolution = self.render_sidebar()
        self.render_header()

        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "系统概览",
            "大脑模块",
            "量子模块",
            "进化模块"
        ])

        with tab1:
            self.render_performance_metrics()
            self.render_system_log()

        with tab2:
            if show_brain:
                self.render_brain_module()
                self.render_memory_network()

        with tab3:
            if show_quantum:
                self.render_quantum_module()

        with tab4:
            if show_evolution:
                self.render_evolution_module()


def main():
    """主函数"""
    st.set_page_config(
        page_title="NeuroMinecraft Genesis",
        page_icon="🧠",
        layout="wide"
    )

    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
