"""
24小时连续实验系统
================

该模块实现了六维能力增长的24小时连续实验主系统，整合所有核心组件：
- LongTermRetention：24小时实验主控制器
- 实时Streamlit界面
- 多组对照实验设计
- 实时数据分析和可视化
- 自动报告生成
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .cognitive_tracker import CognitiveTracker, CognitiveMetrics
from .hourly_monitor import HourlyMonitor, MonitorStatus
from .trend_analyzer import TrendAnalyzer, TrendAnalysis
from .statistical_analyzer import StatisticalAnalyzer, StatisticalResult

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentGroup(Enum):
    """实验组类型"""
    BASELINE = "基线组"
    SINGLE_OPTIMIZATION = "单维优化组"
    MULTI_OPTIMIZATION = "六维协同组"

class ExperimentStatus(Enum):
    """实验状态"""
    INITIALIZING = "初始化中"
    RUNNING = "运行中"
    PAUSED = "暂停"
    COMPLETED = "已完成"
    ERROR = "错误"
    STOPPED = "已停止"

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    group_type: ExperimentGroup
    duration_hours: int = 24
    evaluation_interval: int = 3600  # 1小时
    optimization_weights: Dict[str, float] = None
    parallel_runs: int = 5  # 每组并行运行次数
    random_seed: int = 42

@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime]
    status: ExperimentStatus
    metrics_data: Dict[int, Dict]  # hour -> metrics
    trend_analysis: Dict[str, TrendAnalysis]
    statistical_results: Dict[str, Any]
    export_path: str

class LongTermRetention:
    """24小时连续实验主控制器"""
    
    def __init__(self, streamlit_app: bool = True):
        """
        初始化24小时实验系统
        
        Args:
            streamlit_app: 是否启动Streamlit界面
        """
        self.streamlit_app = streamlit_app
        self.experiments: Dict[str, LongTermRetention.ExperimentInstance] = {}
        self.active_experiment: Optional[str] = None
        
        # 实验控制
        self.status = ExperimentStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.total_runs = 3  # 3个实验组
        self.completed_runs = 0
        
        # 组件初始化
        self.tracker: Optional[CognitiveTracker] = None
        self.monitor: Optional[HourlyMonitor] = None
        self.trend_analyzer: Optional[TrendAnalyzer] = None
        self.statistical_analyzer: Optional[StatisticalAnalyzer] = None
        
        # 数据存储
        self.experiment_data: Dict[str, ExperimentResult] = {}
        self.results_queue = queue.Queue()
        
        # 实时控制
        self.control_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Streamlit配置
        if self.streamlit_app:
            self.setup_streamlit_interface()
        
        # 实验配置
        self.experiment_configs = {
            '基线组': ExperimentConfig(
                name="基线组",
                group_type=ExperimentGroup.BASELINE,
                optimization_weights={'memory': 1.0, 'thinking': 1.0, 'creativity': 1.0, 
                                   'observation': 1.0, 'attention': 1.0, 'imagination': 1.0}
            ),
            '单维优化组': ExperimentConfig(
                name="单维优化组",
                group_type=ExperimentGroup.SINGLE_OPTIMIZATION,
                optimization_weights={'memory': 2.0, 'thinking': 1.0, 'creativity': 1.0, 
                                   'observation': 1.0, 'attention': 1.0, 'imagination': 1.0}
            ),
            '六维协同组': ExperimentConfig(
                name="六维协同组",
                group_type=ExperimentGroup.MULTI_OPTIMIZATION,
                optimization_weights={'memory': 1.5, 'thinking': 1.5, 'creativity': 1.5, 
                                   'observation': 1.5, 'attention': 1.5, 'imagination': 1.5}
            )
        }
        
        self.status = ExperimentStatus.INITIALIZING
        logger.info("24小时实验系统初始化完成")
    
    class ExperimentInstance:
        """单个实验实例"""
        def __init__(self, experiment_id: str, config: ExperimentConfig):
            self.experiment_id = experiment_id
            self.config = config
            self.tracker = CognitiveTracker(f"agent_{experiment_id}")
            self.monitor = HourlyMonitor(self.tracker, config.evaluation_interval)
            self.trend_analyzer = TrendAnalyzer()
            self.statistical_analyzer = StatisticalAnalyzer()
            
            # 状态跟踪
            self.status = ExperimentStatus.INITIALIZING
            self.start_time: Optional[datetime] = None
            self.end_time: Optional[datetime] = None
            self.results: Optional[ExperimentResult] = None
            
            # 数据缓存
            self.metrics_cache: List[CognitiveMetrics] = []
            self.performance_data: Dict[str, List[float]] = {}
    
    def setup_streamlit_interface(self):
        """设置Streamlit实时界面"""
        if not hasattr(self, '_streamlit_setup'):
            st.set_page_config(
                page_title="24小时认知能力实验监控",
                page_icon="🧠",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # 主标题
            st.title("🧠 六维能力增长24小时连续实验系统")
            st.markdown("### 实时监控认知能力发展趋势")
            
            # 侧边栏控制
            st.sidebar.header("实验控制")
            
            # 实验状态显示
            if 'experiment_status' not in st.session_state:
                st.session_state.experiment_status = "未开始"
            if 'current_hour' not in st.session_state:
                st.session_state.current_hour = 0
            if 'metrics_data' not in st.session_state:
                st.session_state.metrics_data = {}
            if 'charts_data' not in st.session_state:
                st.session_state.charts_data = {}
            
            # 实时图表容器
            self.charts_container = st.container()
            
            # 控制按钮
            col1, col2, col3 = st.sidebar.columns(3)
            
            with col1:
                if st.button("开始实验", type="primary"):
                    self.start_full_experiment()
            
            with col2:
                if st.button("暂停实验"):
                    self.pause_experiment()
            
            with col3:
                if st.button("停止实验"):
                    self.stop_experiment()
            
            # 实验组选择
            st.sidebar.subheader("实验组配置")
            selected_groups = st.sidebar.multiselect(
                "选择实验组",
                list(self.experiment_configs.keys()),
                default=list(self.experiment_configs.keys())
            )
            
            # 运行次数设置
            runs_per_group = st.sidebar.slider("每组运行次数", 1, 10, 5)
            
            # 设置完成标记
            self._streamlit_setup = True
            
            logger.info("Streamlit界面设置完成")
    
    def start_full_experiment(self) -> bool:
        """
        启动完整的24小时实验（包含3个对照组）
        
        Returns:
            是否成功启动
        """
        if self.status == ExperimentStatus.RUNNING:
            logger.warning("实验已在运行中")
            return False
        
        try:
            self.status = ExperimentStatus.RUNNING
            self.start_time = datetime.now()
            self.completed_runs = 0
            
            # 启动控制线程
            self.control_thread = threading.Thread(target=self._experiment_control_loop, daemon=True)
            self.control_thread.start()
            
            if self.streamlit_app:
                st.session_state.experiment_status = "运行中"
            
            logger.info("24小时完整实验已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动实验失败: {e}")
            self.status = ExperimentStatus.ERROR
            return False
    
    def _experiment_control_loop(self):
        """实验控制主循环"""
        try:
            for group_name, config in self.experiment_configs.items():
                if self.stop_event.is_set():
                    break
                
                logger.info(f"开始实验组: {group_name}")
                
                # 创建实验实例
                experiment_id = f"{group_name}_{int(time.time())}"
                experiment = self.ExperimentInstance(experiment_id, config)
                
                # 设置优化权重
                if config.optimization_weights:
                    experiment.tracker.set_weights(config.optimization_weights)
                
                # 启动监控
                experiment.monitor.add_callback('hourly_update', 
                                              lambda data: self._on_hourly_update(experiment_id, data))
                experiment.monitor.add_callback('completion', 
                                              lambda data: self._on_experiment_completion(experiment_id, data))
                
                experiment.status = ExperimentStatus.RUNNING
                experiment.start_time = datetime.now()
                
                # 运行实验
                success = self._run_single_experiment(experiment)
                
                if success:
                    self.completed_runs += 1
                    logger.info(f"实验组 {group_name} 完成")
                else:
                    logger.error(f"实验组 {group_name} 失败")
            
            # 所有实验完成后进行统计分析
            if not self.stop_event.is_set():
                self._perform_statistical_analysis()
                
            self.status = ExperimentStatus.COMPLETED
            
            if self.streamlit_app:
                st.session_state.experiment_status = "已完成"
            
        except Exception as e:
            logger.error(f"实验控制循环出错: {e}")
            self.status = ExperimentStatus.ERROR
    
    def _run_single_experiment(self, experiment: 'LongTermRetention.ExperimentInstance') -> bool:
        """
        运行单个24小时实验
        
        Args:
            experiment: 实验实例
            
        Returns:
            是否成功完成
        """
        try:
            # 启动24小时监控
            success = experiment.monitor.start_monitoring()
            
            if success:
                # 等待实验完成
                while (experiment.monitor.status == MonitorStatus.RUNNING and 
                       not self.stop_event.is_set()):
                    time.sleep(60)  # 每分钟检查一次
                
                if experiment.monitor.status == MonitorStatus.STOPPED:
                    # 收集结果
                    results = self._collect_experiment_results(experiment)
                    self.experiment_data[experiment.experiment_id] = results
                    experiment.results = results
                    experiment.status = ExperimentStatus.COMPLETED
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"单实验运行失败: {e}")
            experiment.status = ExperimentStatus.ERROR
            return False
    
    def _collect_experiment_results(self, experiment: 'LongTermRetention.ExperimentInstance') -> ExperimentResult:
        """
        收集实验结果
        
        Args:
            experiment: 实验实例
            
        Returns:
            实验结果对象
        """
        # 获取监控数据
        hourly_data = experiment.monitor.get_hourly_data()
        
        # 转换为认知指标
        metrics_data = {}
        metrics_history = []
        
        for hour, data in hourly_data.items():
            dimension_scores = data.get('dimension_scores', {})
            metrics = CognitiveMetrics(
                timestamp=datetime.fromisoformat(data['timestamp']),
                memory_score=dimension_scores.get('memory', 50),
                thinking_score=dimension_scores.get('thinking', 50),
                creativity_score=dimension_scores.get('creativity', 50),
                observation_score=dimension_scores.get('observation', 50),
                attention_score=dimension_scores.get('attention', 50),
                imagination_score=dimension_scores.get('imagination', 50)
            )
            metrics_data[hour] = data
            metrics_history.append(metrics)
        
        # 进行趋势分析
        trend_analysis = experiment.trend_analyzer.analyze_all_dimensions(metrics_history)
        
        # 创建结果对象
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            config=experiment.config,
            start_time=experiment.start_time,
            end_time=datetime.now(),
            status=ExperimentStatus.COMPLETED,
            metrics_data=metrics_data,
            trend_analysis={k: asdict(v) for k, v in trend_analysis.items()},
            statistical_results={},
            export_path=f"experiment_results_{experiment.experiment_id}.json"
        )
        
        logger.info(f"实验结果收集完成: {experiment.experiment_id}")
        return result
    
    def _on_hourly_update(self, experiment_id: str, data: Dict):
        """处理每小时数据更新"""
        # 更新会话状态
        if 'metrics_data' not in st.session_state:
            st.session_state.metrics_data = {}
        
        if experiment_id not in st.session_state.metrics_data:
            st.session_state.metrics_data[experiment_id] = {}
        
        st.session_state.metrics_data[experiment_id][data['hour']] = data
        
        # 更新图表数据
        if 'charts_data' not in st.session_state:
            st.session_state.charts_data = {}
        
        if experiment_id not in st.session_state.charts_data:
            st.session_state.charts_data[experiment_id] = {
                'hours': [],
                'scores': {'memory': [], 'thinking': [], 'creativity': [],
                          'observation': [], 'attention': [], 'imagination': []}
            }
        
        # 添加新数据点
        st.session_state.charts_data[experiment_id]['hours'].append(data['hour'])
        
        dimension_scores = data['dimension_scores']
        for dimension in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
            st.session_state.charts_data[experiment_id]['scores'][dimension].append(
                dimension_scores.get(dimension, 50)
            )
        
        # 实时更新图表
        if self.streamlit_app:
            self.update_realtime_charts()
    
    def _on_experiment_completion(self, experiment_id: str, data: Dict):
        """处理实验完成事件"""
        logger.info(f"实验完成: {experiment_id}")
        
        if self.streamlit_app:
            st.success(f"实验组 {experiment_id} 已完成!")
    
    def update_realtime_charts(self):
        """更新实时图表"""
        try:
            with self.charts_container:
                if not st.session_state.charts_data:
                    st.info("暂无数据，请开始实验")
                    return
                
                # 创建6个子图
                dimensions = ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']
                dimension_names = ['记忆力', '思维力', '创造力', '观察力', '注意力', '想象力']
                
                # 使用Plotly创建交互式图表
                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=dimension_names,
                    specs=[[{"secondary_y": False}] * 3] * 2
                )
                
                colors = px.colors.qualitative.Set1
                exp_index = 0
                
                # 为每个实验组添加数据
                for experiment_id, chart_data in st.session_state.charts_data.items():
                    hours = chart_data['hours']
                    scores = chart_data['scores']
                    
                    if not hours:
                        continue
                    
                    color = colors[exp_index % len(colors)]
                    
                    for i, (dim, dim_name) in enumerate(zip(dimensions, dimension_names)):
                        row = (i // 3) + 1
                        col = (i % 3) + 1
                        
                        fig.add_trace(
                            go.Scatter(
                                x=hours,
                                y=scores[dim],
                                mode='lines+markers',
                                name=f'{experiment_id}_{dim_name}',
                                line=dict(color=color, width=2),
                                marker=dict(size=6),
                                hovertemplate=f'<b>{experiment_id}</b><br>' +
                                            f'{dim_name}: %{{y:.1f}}<br>' +
                                            '时间: 第%{x}小时<extra></extra>'
                            ),
                            row=row, col=col
                        )
                
                # 更新布局
                fig.update_layout(
                    title={
                        'text': '六维认知能力实时发展趋势',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20}
                    },
                    height=800,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # 更新x轴和y轴
                for i in range(1, 7):
                    row = (i - 1) // 3 + 1
                    col = (i - 1) % 3 + 1
                    
                    fig.update_xaxes(
                        title_text="时间 (小时)",
                        row=row, col=col,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                    
                    fig.update_yaxes(
                        title_text="能力分数",
                        range=[0, 100],
                        row=row, col=col,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示实验进度
                if st.session_state.charts_data:
                    total_hours = 24
                    completed_hours = max([len(data['hours']) for data in st.session_state.charts_data.values()]) if st.session_state.charts_data else 0
                    progress = min(100, (completed_hours / total_hours) * 100)
                    
                    st.progress(progress)
                    st.text(f"实验进度: {completed_hours}/{total_hours} 小时 ({progress:.1f}%)")
        
        except Exception as e:
            logger.error(f"更新实时图表失败: {e}")
            st.error("图表更新失败")
    
    def _perform_statistical_analysis(self):
        """执行统计显著性分析"""
        logger.info("开始统计显著性分析")
        
        try:
            if len(self.experiment_data) < 2:
                logger.warning("实验数据不足，无法进行统计分析")
                return
            
            # 准备数据
            analysis_data = {}
            
            for experiment_id, result in self.experiment_data.items():
                group_name = result.config.name
                if group_name not in analysis_data:
                    analysis_data[group_name] = {}
                
                # 收集各维度数据
                for hour in range(24):
                    if hour in result.metrics_data:
                        dimension_scores = result.metrics_data[hour].get('dimension_scores', {})
                        for dimension, score in dimension_scores.items():
                            if dimension not in analysis_data[group_name]:
                                analysis_data[group_name][dimension] = []
                            analysis_data[group_name][dimension].append(score)
            
            # 创建统计报告
            for dimension in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
                if dimension in analysis_data:
                    groups_data = {name: data[dimension] for name, data in analysis_data.items() 
                                 if dimension in data and len(data[dimension]) > 0}
                    
                    if len(groups_data) >= 2:
                        # 创建统计分析器实例
                        analyzer = StatisticalAnalyzer()
                        
                        # 执行方差分析
                        try:
                            anova_result, comparisons = analyzer.anova_analysis(
                                groups_data, dimension
                            )
                            
                            # 存储结果
                            for exp_id, result in self.experiment_data.items():
                                if result.config.name not in result.statistical_results:
                                    result.statistical_results[result.config.name] = {}
                                result.statistical_results[result.config.name][dimension] = {
                                    'anova_statistic': anova_result.statistic,
                                    'p_value': anova_result.p_value,
                                    'effect_size': anova_result.effect_size,
                                    'significance': anova_result.significance_level
                                }
                        except Exception as e:
                            logger.error(f"维度 {dimension} 统计分析失败: {e}")
            
            # 生成最终报告
            self._generate_final_report()
            
            logger.info("统计显著性分析完成")
            
        except Exception as e:
            logger.error(f"统计分析执行失败: {e}")
    
    def _generate_final_report(self):
        """生成最终实验报告"""
        try:
            report = {
                'experiment_summary': {
                    'total_experiments': len(self.experiment_data),
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'total_duration_hours': 24
                },
                'experiment_results': {},
                'comparative_analysis': {},
                'conclusions': []
            }
            
            # 汇总各实验结果
            for exp_id, result in self.experiment_data.items():
                group_name = result.config.name
                report['experiment_results'][group_name] = {
                    'duration_hours': 24,
                    'final_scores': {},
                    'improvement_rates': {},
                    'trend_summary': {}
                }
                
                # 计算最终分数和改进率
                if result.metrics_data:
                    final_data = result.metrics_data.get(23, {})  # 第24小时数据
                    baseline_data = result.metrics_data.get(0, {})  # 第1小时数据
                    
                    if final_data and baseline_data:
                        final_scores = final_data.get('dimension_scores', {})
                        baseline_scores = baseline_data.get('dimension_scores', {})
                        
                        for dimension in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
                            final_score = final_scores.get(dimension, 50)
                            baseline_score = baseline_scores.get(dimension, 50)
                            improvement_rate = ((final_score - baseline_score) / baseline_score) * 100
                            
                            report['experiment_results'][group_name]['final_scores'][dimension] = final_score
                            report['experiment_results'][group_name]['improvement_rates'][dimension] = improvement_rate
                
                # 添加趋势摘要
                if 'trend_summary' in result.trend_analysis:
                    report['experiment_results'][group_name]['trend_summary'] = result.trend_analysis['trend_summary']
            
            # 生成结论
            conclusions = []
            
            # 比较各组性能
            group_performance = {}
            for group_name, data in report['experiment_results'].items():
                avg_improvement = np.mean(list(data['improvement_rates'].values()))
                group_performance[group_name] = avg_improvement
            
            if group_performance:
                best_group = max(group_performance, key=group_performance.get)
                worst_group = min(group_performance, key=group_performance.get)
                
                conclusions.append(f"实验组性能排序: {sorted(group_performance.items(), key=lambda x: x[1], reverse=True)}")
                conclusions.append(f"表现最佳组: {best_group} (平均改进率: {group_performance[best_group]:.1f}%)")
                conclusions.append(f"表现最差组: {worst_group} (平均改进率: {group_performance[worst_group]:.1f}%)")
            
            report['conclusions'] = conclusions
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"24h_experiment_report_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"最终实验报告已生成: {report_path}")
            
            if self.streamlit_app:
                st.success(f"实验完成! 报告已保存到: {report_path}")
        
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
    
    def pause_experiment(self) -> bool:
        """暂停当前实验"""
        if self.status == ExperimentStatus.RUNNING and self.monitor:
            success = self.monitor.pause_monitoring()
            if success:
                self.status = ExperimentStatus.PAUSED
                if self.streamlit_app:
                    st.session_state.experiment_status = "暂停"
                logger.info("实验已暂停")
            return success
        return False
    
    def resume_experiment(self) -> bool:
        """恢复实验"""
        if self.status == ExperimentStatus.PAUSED and self.monitor:
            success = self.monitor.resume_monitoring()
            if success:
                self.status = ExperimentStatus.RUNNING
                if self.streamlit_app:
                    st.session_state.experiment_status = "运行中"
                logger.info("实验已恢复")
            return success
        return False
    
    def stop_experiment(self) -> bool:
        """停止实验"""
        if self.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.stop_event.set()
            if self.monitor:
                self.monitor.stop_monitoring()
            
            self.status = ExperimentStatus.STOPPED
            if self.streamlit_app:
                st.session_state.experiment_status = "已停止"
            
            logger.info("实验已停止")
            return True
        return False
    
    def get_experiment_status(self) -> Dict:
        """获取实验状态"""
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
            'completed_runs': self.completed_runs,
            'total_runs': self.total_runs,
            'completion_rate': (self.completed_runs / self.total_runs) * 100
        }
    
    def export_all_results(self, directory: str = "experiment_exports") -> bool:
        """导出所有实验结果"""
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            
            for exp_id, result in self.experiment_data.items():
                filepath = os.path.join(directory, f"{exp_id}_results.json")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"所有实验结果已导出到目录: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"导出实验结果失败: {e}")
            return False

def main():
    """主函数 - 启动24小时实验系统"""
    logger.info("启动24小时认知能力实验系统")
    
    # 创建实验系统
    experiment_system = LongTermRetention(streamlit_app=True)
    
    try:
        if experiment_system.streamlit_app:
            # 启动Streamlit应用
            import subprocess
            import sys
            
            # 获取当前脚本目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 启动Streamlit
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                __file__,
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ])
        else:
            # 命令行模式运行
            while experiment_system.status != ExperimentStatus.COMPLETED:
                time.sleep(10)
                logger.info(f"实验状态: {experiment_system.status.value}")
    
    except KeyboardInterrupt:
        logger.info("用户中断，停止实验系统")
        experiment_system.stop_experiment()
    except Exception as e:
        logger.error(f"实验系统运行出错: {e}")
        experiment_system.status = ExperimentStatus.ERROR

if __name__ == "__main__":
    import os
    main()