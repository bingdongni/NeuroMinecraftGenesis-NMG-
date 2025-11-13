"""
六维实时监控主面板
================

Streamlit实时监控主面板，展示六维认知能力的实时变化趋势和性能指标。

主要功能：
- 六维认知能力实时显示：记忆力、思维力、创造力、观察力、注意力、想象力
- 每5秒自动刷新数据更新
- 实时曲线图显示能力发展趋势
- 性能指标展示：精确检索准确率、思维深度、创新性动作占比等
- 数据持久化和历史记录追踪

Author: Claude Code Agent
Date: 2025-11-13
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import datetime
import json
import os
import numpy as np
from typing import Dict, List, Any

from six_dimension_monitor import SixDimensionMonitor
from memory_monitor import MemoryMonitor
from thinking_monitor import ThinkingMonitor
from creativity_monitor import CreativityMonitor
from observation_monitor import ObservationMonitor
from attention_monitor import AttentionMonitor
from imagination_monitor import ImaginationMonitor


class StreamlitDashboard:
    """
    Streamlit实时监控主面板类
    
    功能：
    - 管理六维认知能力监控
    - 实时数据展示和可视化
    - 自动刷新机制
    - 性能指标计算
    """
    
    def __init__(self):
        """初始化主面板"""
        # 设置页面配置
        self.setup_page_config()
        
        # 初始化会话状态
        self.init_session_state()
        
        # 初始化监控器
        self.init_monitors()
        
        # 数据存储路径
        self.data_dir = "/workspace/data/monitoring_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def setup_page_config(self):
        """设置Streamlit页面配置"""
        st.set_page_config(
            page_title="六维认知能力监控面板",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def init_session_state(self):
        """初始化会话状态变量"""
        if 'monitor_start_time' not in st.session_state:
            st.session_state.monitor_start_time = time.time()
        
        if 'refresh_count' not in st.session_state:
            st.session_state.refresh_count = 0
        
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = []
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def init_monitors(self):
        """初始化六维能力监控器"""
        # 创建六维能力监控主类
        self.six_dimension_monitor = SixDimensionMonitor()
        
        # 创建各个维度的监控器
        self.memory_monitor = MemoryMonitor()
        self.thinking_monitor = ThinkingMonitor()
        self.creativity_monitor = CreativityMonitor()
        self.observation_monitor = ObservationMonitor()
        self.attention_monitor = AttentionMonitor()
        self.imagination_monitor = ImaginationMonitor()
    
    def create_header(self):
        """创建页面头部"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🧠 六维认知能力实时监控面板")
            st.markdown("---")
            
            # 运行状态显示
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            runtime = time.time() - st.session_state.monitor_start_time
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("当前时间", current_time)
            with col_b:
                st.metric("运行时间", f"{runtime/3600:.2f}小时")
            with col_c:
                st.metric("刷新次数", st.session_state.refresh_count)
    
    def create_sidebar(self):
        """创建侧边栏控制面板"""
        st.sidebar.header("⚙️ 控制面板")
        
        # 自动刷新控制
        st.sidebar.checkbox("自动刷新 (5秒)", key="auto_refresh", 
                           help="开启后每5秒自动更新数据")
        
        # 监控维度选择
        st.sidebar.subheader("📊 监控维度选择")
        dimensions = {
            "记忆力": self.memory_monitor,
            "思维力": self.thinking_monitor,
            "创造力": self.creativity_monitor,
            "观察力": self.observation_monitor,
            "注意力": self.attention_monitor,
            "想象力": self.imagination_monitor
        }
        
        selected_dimensions = []
        for dim_name, monitor in dimensions.items():
            if st.sidebar.checkbox(dim_name, value=True):
                selected_dimensions.append(dim_name)
        
        # 图表类型选择
        st.sidebar.subheader("📈 图表类型")
        chart_types = {
            "实时曲线图": "line",
            "雷达图": "radar",
            "柱状图": "bar",
            "热力图": "heatmap"
        }
        
        selected_chart_type = st.sidebar.selectbox("选择图表类型", list(chart_types.keys()))
        
        # 历史数据管理
        st.sidebar.subheader("📁 数据管理")
        if st.sidebar.button("保存当前数据"):
            self.save_historical_data()
        
        if st.sidebar.button("清空历史数据"):
            st.session_state.historical_data = []
            st.sidebar.success("历史数据已清空")
        
        return selected_dimensions, chart_types[selected_chart_type]
    
    def update_data(self):
        """更新监控数据"""
        # 更新所有监控器的数据
        current_data = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "memory": self.memory_monitor.get_current_metrics(),
            "thinking": self.thinking_monitor.get_current_metrics(),
            "creativity": self.creativity_monitor.get_current_metrics(),
            "observation": self.observation_monitor.get_current_metrics(),
            "attention": self.attention_monitor.get_current_metrics(),
            "imagination": self.imagination_monitor.get_current_metrics()
        }
        
        # 添加到历史数据
        st.session_state.historical_data.append(current_data)
        
        # 限制历史数据长度（保留最近1000个数据点）
        if len(st.session_state.historical_data) > 1000:
            st.session_state.historical_data = st.session_state.historical_data[-1000:]
        
        # 增加刷新计数
        st.session_state.refresh_count += 1
        
        return current_data
    
    def create_six_dimension_overview(self, data: Dict[str, Any]):
        """创建六维能力总览"""
        st.header("🎯 六维认知能力总览")
        
        # 获取六个维度的当前分数
        dimensions_scores = []
        dimension_names = ["记忆力", "思维力", "创造力", "观察力", "注意力", "想象力"]
        dimension_keys = ["memory", "thinking", "creativity", "observation", "attention", "imagination"]
        
        for key in dimension_keys:
            score = data.get(key, {}).get("overall_score", 0)
            dimensions_scores.append(score)
        
        # 创建六列显示每个维度的当前状态
        cols = st.columns(6)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        
        for i, (col, name, score, color) in enumerate(zip(cols, dimension_names, dimensions_scores, colors)):
            with col:
                # 创建指标卡片
                delta_score = 0
                if len(st.session_state.historical_data) > 1:
                    prev_score = st.session_state.historical_data[-2].get(dimension_keys[i], {}).get("overall_score", 0)
                    delta_score = score - prev_score
                
                st.metric(
                    label=name,
                    value=f"{score:.1f}%",
                    delta=f"{delta_score:+.1f}%",
                    help=f"当前{name}得分：{score:.1f}%"
                )
        
        # 创建雷达图显示整体能力分布
        if len(dimensions_scores) == 6:
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=dimensions_scores + [dimensions_scores[0]],  # 闭合雷达图
                theta=dimension_names + [dimension_names[0]],
                fill='toself',
                name='当前能力',
                line_color='#FF6B6B',
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="六维认知能力雷达图",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    def create_performance_metrics(self, data: Dict[str, Any]):
        """创建性能指标面板"""
        st.header("📈 性能指标")
        
        # 计算总体性能指标
        memory_metrics = data.get("memory", {})
        thinking_metrics = data.get("thinking", {})
        creativity_metrics = data.get("creativity", {})
        observation_metrics = data.get("observation", {})
        attention_metrics = data.get("attention", {})
        imagination_metrics = data.get("imagination", {})
        
        # 核心性能指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 精确检索准确率
            accuracy = memory_metrics.get("retrieval_accuracy", 0.0)
            st.metric(
                label="精确检索准确率",
                value=f"{accuracy*100:.1f}%",
                help="记忆力模块的精确检索准确率"
            )
        
        with col2:
            # 思维深度指数
            depth = thinking_metrics.get("thinking_depth", 0.0)
            st.metric(
                label="思维深度指数",
                value=f"{depth:.2f}",
                help="思维力的深度分析指数"
            )
        
        with col3:
            # 创新性动作占比
            innovation_ratio = creativity_metrics.get("innovation_ratio", 0.0)
            st.metric(
                label="创新性动作占比",
                value=f"{innovation_ratio*100:.1f}%",
                help="创造力模块的创新性动作比例"
            )
        
        with col4:
            # 观察敏锐度
            observation_acuity = observation_metrics.get("observation_acuity", 0.0)
            st.metric(
                label="观察敏锐度",
                value=f"{observation_acuity:.1f}%",
                help="观察力模块的敏锐度指数"
            )
        
        # 详细性能指标表格
        st.subheader("详细性能指标")
        
        performance_data = []
        for dim_name, dim_key in zip(
            ["记忆力", "思维力", "创造力", "观察力", "注意力", "想象力"],
            ["memory", "thinking", "creativity", "observation", "attention", "imagination"]
        ):
            metrics = data.get(dim_key, {})
            performance_data.append({
                "维度": dim_name,
                "当前得分": f"{metrics.get('overall_score', 0):.1f}%",
                "响应时间": f"{metrics.get('response_time', 0):.3f}s",
                "稳定性": f"{metrics.get('stability', 0):.2f}",
                "效率": f"{metrics.get('efficiency', 0):.1f}%"
            })
        
        st.dataframe(performance_data, use_container_width=True)
    
    def create_realtime_charts(self, selected_dimensions: List[str], chart_type: str):
        """创建实时图表"""
        st.header("📊 实时趋势图表")
        
        if len(st.session_state.historical_data) < 2:
            st.warning("需要至少2个数据点才能显示趋势图")
            return
        
        # 准备数据
        timestamps = []
        dimension_data = {dim: [] for dim in selected_dimensions}
        
        dimension_mapping = {
            "记忆力": "memory",
            "思维力": "thinking", 
            "创造力": "creativity",
            "观察力": "observation",
            "注意力": "attention",
            "想象力": "imagination"
        }
        
        for data_point in st.session_state.historical_data[-50:]:  # 显示最近50个数据点
            timestamps.append(datetime.datetime.fromtimestamp(data_point["timestamp"]).strftime("%H:%M:%S"))
            
            for dim_name in selected_dimensions:
                dim_key = dimension_mapping.get(dim_name, dim_name.lower())
                score = data_point.get(dim_key, {}).get("overall_score", 0)
                dimension_data[dim_name].append(score)
        
        # 根据图表类型创建不同的可视化
        if chart_type == "line":
            self.create_line_chart(timestamps, dimension_data)
        elif chart_type == "radar":
            self.create_radar_chart(dimension_data)
        elif chart_type == "bar":
            self.create_bar_chart(dimension_data)
        elif chart_type == "heatmap":
            self.create_heatmap(timestamps, dimension_data)
    
    def create_line_chart(self, timestamps: List[str], dimension_data: Dict[str, List[float]]):
        """创建实时曲线图"""
        fig = go.Figure()
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        
        for i, (dim_name, scores) in enumerate(dimension_data.items()):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name=dim_name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="六维认知能力实时变化趋势",
            xaxis_title="时间",
            yaxis_title="能力得分 (%)",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_radar_chart(self, dimension_data: Dict[str, List[float]]):
        """创建雷达图"""
        # 使用最新数据点
        latest_scores = {dim: scores[-1] if scores else 0 for dim, scores in dimension_data.items()}
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(latest_scores.values()) + [list(latest_scores.values())[0]],
            theta=list(latest_scores.keys()) + [list(latest_scores.keys())[0]],
            fill='toself',
            name='当前状态',
            line_color='#FF6B6B',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="当前六维能力雷达图",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_bar_chart(self, dimension_data: Dict[str, List[float]]):
        """创建柱状图"""
        # 使用最新数据点
        latest_scores = {dim: scores[-1] if scores else 0 for dim, scores in dimension_data.items()}
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(latest_scores.keys()),
            y=list(latest_scores.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:len(latest_scores)]
        ))
        
        fig.update_layout(
            title="六维认知能力当前得分",
            xaxis_title="认知维度",
            yaxis_title="能力得分 (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_heatmap(self, timestamps: List[str], dimension_data: Dict[str, List[float]]):
        """创建热力图"""
        # 准备热力图数据
        heatmap_data = []
        for dim_name in dimension_data:
            heatmap_data.append(dimension_data[dim_name])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=timestamps,
            y=list(dimension_data.keys()),
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="六维能力热力图（时间序列）",
            xaxis_title="时间",
            yaxis_title="认知维度",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def save_historical_data(self):
        """保存历史数据到文件"""
        if st.session_state.historical_data:
            filename = f"monitoring_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.historical_data, f, ensure_ascii=False, indent=2)
            
            st.sidebar.success(f"数据已保存至: {filename}")
        else:
            st.sidebar.warning("没有可保存的数据")
    
    def run_dashboard(self):
        """运行主面板"""
        # 创建页面头部
        self.create_header()
        
        # 创建侧边栏控制面板
        selected_dimensions, chart_type = self.create_sidebar()
        
        # 更新数据
        current_data = self.update_data()
        
        # 创建主要面板
        tab1, tab2, tab3, tab4 = st.tabs(["总览", "性能指标", "实时图表", "历史数据"])
        
        with tab1:
            self.create_six_dimension_overview(current_data)
        
        with tab2:
            self.create_performance_metrics(current_data)
        
        with tab3:
            if selected_dimensions:
                self.create_realtime_charts(selected_dimensions, chart_type)
            else:
                st.warning("请在左侧面板中选择要显示的维度")
        
        with tab4:
            self.create_historical_data_view()
    
    def create_historical_data_view(self):
        """创建历史数据查看器"""
        st.subheader("📊 历史数据查看")
        
        if not st.session_state.historical_data:
            st.info("暂无历史数据")
            return
        
        # 数据统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据点总数", len(st.session_state.historical_data))
        with col2:
            st.metric("监控时长", f"{(st.session_state.historical_data[-1]['timestamp'] - st.session_state.historical_data[0]['timestamp'])/60:.1f}分钟")
        with col3:
            st.metric("平均刷新间隔", f"{(st.session_state.historical_data[-1]['timestamp'] - st.session_state.historical_data[0]['timestamp'])/(len(st.session_state.historical_data)-1):.1f}秒")
        
        # 最近数据点查看
        st.subheader("最近10个数据点")
        recent_data = st.session_state.historical_data[-10:]
        
        formatted_data = []
        for data_point in recent_data:
            dt = datetime.datetime.fromtimestamp(data_point["timestamp"]).strftime("%H:%M:%S")
            memory_score = data_point.get("memory", {}).get("overall_score", 0)
            thinking_score = data_point.get("thinking", {}).get("overall_score", 0)
            creativity_score = data_point.get("creativity", {}).get("overall_score", 0)
            observation_score = data_point.get("observation", {}).get("overall_score", 0)
            attention_score = data_point.get("attention", {}).get("overall_score", 0)
            imagination_score = data_point.get("imagination", {}).get("overall_score", 0)
            
            formatted_data.append({
                "时间": dt,
                "记忆力": f"{memory_score:.1f}%",
                "思维力": f"{thinking_score:.1f}%",
                "创造力": f"{creativity_score:.1f}%",
                "观察力": f"{observation_score:.1f}%",
                "注意力": f"{attention_score:.1f}%",
                "想象力": f"{imagination_score:.1f}%"
            })
        
        st.dataframe(formatted_data, use_container_width=True)
        
        # 数据导出选项
        st.subheader("数据导出")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("导出为JSON"):
                json_str = json.dumps(st.session_state.historical_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="下载JSON文件",
                    data=json_str,
                    file_name=f"cognitive_monitoring_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("导出为CSV"):
                import pandas as pd
                df = pd.DataFrame(st.session_state.historical_data)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="下载CSV文件", 
                    data=csv,
                    file_name=f"cognitive_monitoring_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def main():
    """主函数"""
    # 创建仪表板实例
    dashboard = StreamlitDashboard()
    
    # 运行仪表板
    dashboard.run_dashboard()
    
    # 自动刷新机制
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()