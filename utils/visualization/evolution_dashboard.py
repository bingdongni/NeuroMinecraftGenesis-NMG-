#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolution Dashboard - Real-time monitoring of evolution process

Features:
- Real-time evolution curve display
- Best individual tracking
- Diversity monitoring
- Auto-load latest data
- Interactive visualization

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EvolutionDashboard:
    """
    进化过程实时监控仪表板
    
    提供实时可视化监控和自动数据更新功能
    """
    
    def __init__(self,
                 data_dir: str = "data/evolution_logs",
                 update_interval: float = 2.0,
                 auto_reload: bool = True,
                 dashboard_config: Optional[Dict[str, Any]] = None):
        """
        初始化进化仪表板
        
        Args:
            data_dir: 进化数据目录
            update_interval: 更新间隔（秒）
            auto_reload: 是否自动重新加载数据
            dashboard_config: 仪表板配置
        """
        self.data_dir = data_dir
        self.update_interval = update_interval
        self.auto_reload = auto_reload
        
        # 仪表板配置
        self.config = dashboard_config or {
            'show_fitness_curve': True,
            'show_diversity_plot': True,
            'show_population_heatmap': True,
            'show_3d_trajectory': True,
            'show_best_individual': True,
            'show_species_evolution': True,
            'max_history_points': 500,
            'animation_speed': 100  # ms per frame
        }
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 数据存储
        self.evolution_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'species_count': [],
            'timestamps': [],
            'population_data': {},
            'best_individuals': {}
        }
        
        # 图表对象
        self.fig = None
        self.axes = {}
        self.animation_obj = None
        self.last_update_time = None
        self.last_generation = -1
        
        # 性能监控
        self.performance_metrics = {
            'update_count': 0,
            'last_update_duration': 0.0,
            'average_update_duration': 0.0,
            'data_points_loaded': 0
        }
        
        self.logger.info("Evolution dashboard initialized")
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def start_dashboard(self, 
                       show_live_updates: bool = True,
                       save_snapshots: bool = True,
                       snapshot_interval: int = 50):
        """
        启动进化仪表板
        
        Args:
            show_live_updates: 是否显示实时更新
            save_snapshots: 是否保存快照
            snapshot_interval: 快照保存间隔
        """
        self.logger.info("Starting evolution dashboard...")
        
        # 创建仪表板界面
        self._create_dashboard_layout()
        
        # 初始化数据
        self._load_latest_data()
        
        if show_live_updates:
            # 启动实时更新动画
            self.animation_obj = animation.FuncAnimation(
                self.fig, self._update_dashboard,
                interval=self.config['animation_speed'],
                blit=False
            )
        
        try:
            # 显示仪表板
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            self.logger.info("用户中断，关闭仪表板")
        finally:
            self.cleanup()
    
    def create_static_dashboard(self, 
                               output_path: str,
                               include_analysis: bool = True) -> str:
        """
        创建静态仪表板并保存
        
        Args:
            output_path: 输出文件路径
            include_analysis: 是否包含分析信息
            
        Returns:
            保存的文件路径
        """
        self.logger.info("创建静态仪表板...")
        
        # 加载数据
        self._load_latest_data()
        
        # 创建静态图表
        self._create_dashboard_layout()
        self._update_dashboard(None)
        
        # 添加分析信息
        if include_analysis:
            self._add_analysis_overlay()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        self.logger.info(f"静态仪表板保存至: {output_path}")
        return output_path
    
    def _create_dashboard_layout(self):
        """创建仪表板布局"""
        # 计算子图布局
        plot_configs = [
            ('fitness', self.config['show_fitness_curve']),
            ('diversity', self.config['show_diversity_plot']),
            ('heatmap', self.config['show_population_heatmap']),
            ('trajectory', self.config['show_3d_trajectory']),
            ('best', self.config['show_best_individual']),
            ('species', self.config['show_species_evolution'])
        ]
        
        active_plots = [name for name, enabled in plot_configs if enabled]
        plot_count = len(active_plots)
        
        if plot_count == 0:
            self.logger.warning("没有启用的图表配置")
            return
        
        # 确定布局
        if plot_count <= 2:
            rows, cols = 1, plot_count
        elif plot_count <= 4:
            rows, cols = 2, 2
        elif plot_count <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # 创建图形
        self.fig = plt.figure(figsize=(6*cols, 4*rows))
        self.fig.suptitle('Real-time Evolution Monitoring Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # 创建子图
        self.axes = {}
        plot_index = 0
        
        # 适应度曲线
        if self.config['show_fitness_curve']:
            self.axes['fitness'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_fitness_plot()
            plot_index += 1
        
        # 多样性图
        if self.config['show_diversity_plot']:
            self.axes['diversity'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_diversity_plot()
            plot_index += 1
        
        # 种群热图
        if self.config['show_population_heatmap']:
            self.axes['heatmap'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_heatmap_plot()
            plot_index += 1
        
        # 3D轨迹
        if self.config['show_3d_trajectory']:
            self.axes['trajectory'] = self.fig.add_subplot(rows, cols, plot_index + 1, projection='3d')
            self._setup_trajectory_plot()
            plot_index += 1
        
        # 最佳个体
        if self.config['show_best_individual']:
            self.axes['best'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_best_individual_plot()
            plot_index += 1
        
        # 物种进化
        if self.config['show_species_evolution']:
            self.axes['species'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_species_plot()
            plot_index += 1
        
        # 添加状态显示（仅在有空间时）
        if 'status' not in self.axes and plot_index < rows * cols:
            self.axes['status'] = self.fig.add_subplot(rows, cols, plot_index + 1)
            self._setup_status_display()
        
        # 添加信息面板
        self._add_info_panel()
    
    def _setup_fitness_plot(self):
        """设置适应度曲线图"""
        ax = self.axes['fitness']
        ax.set_title('Evolution Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10)
        
        # 初始化线条
        self.lines_fitness = {
            'best': ax.plot([], [], 'r-', linewidth=3, label='Best Fitness', marker='o', markersize=3)[0],
            'avg': ax.plot([], [], 'b-', linewidth=2, label='Average Fitness', marker='s', markersize=2)[0],
            'worst': ax.plot([], [], 'g-', linewidth=1, label='Worst Fitness', marker='^', markersize=2)[0],
            'fill': ax.fill_between([], [], [], alpha=0.2, color='orange')
        }
        
        ax.legend(loc='upper left')
    
    def _setup_diversity_plot(self):
        """设置多样性图"""
        ax = self.axes['diversity']
        ax.set_title('Genetic Diversity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity Index')
        ax.grid(True, alpha=0.3)
        
        # 初始化线条
        self.line_diversity = ax.plot([], [], 'purple', linewidth=2, marker='D', markersize=3)[0]
        
        # 添加参考线
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='高多样性阈值')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='低多样性阈值')
        ax.legend(loc='upper right')
    
    def _setup_heatmap_plot(self):
        """设置种群热图"""
        ax = self.axes['heatmap']
        ax.set_title('Population Genotype Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gene Locus')
        ax.set_ylabel('Individuals')
        
        # 初始化热图
        self.heatmap = ax.imshow(np.zeros((10, 10)), cmap='viridis', aspect='auto')
        plt.colorbar(self.heatmap, ax=ax, shrink=0.8)
    
    def _setup_trajectory_plot(self):
        """设置3D轨迹图"""
        ax = self.axes['trajectory']
        ax.set_title('3D Evolution Trajectory', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Fitness')
        ax.set_ylabel('Best Fitness')
        ax.set_zlabel('Diversity')
        
        # 初始化轨迹线
        self.trajectory_line = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)[0]
        self.trajectory_points = ax.scatter([], [], [], c='red', s=50, alpha=0.8)
    
    def _setup_best_individual_plot(self):
        """设置最佳个体图"""
        ax = self.axes['best']
        ax.set_title('Best Individual Genome', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gene Locus')
        ax.set_ylabel('Gene Value')
        ax.grid(True, alpha=0.3)
        
        # 初始化基因组线
        self.best_genome_line = ax.plot([], [], 'ro-', linewidth=2, markersize=4)[0]
    
    def _setup_species_plot(self):
        """设置物种进化图"""
        ax = self.axes['species']
        ax.set_title('物种数量变化', fontsize=14, fontweight='bold')
        ax.set_xlabel('代数')
        ax.set_ylabel('物种数量')
        ax.grid(True, alpha=0.3)
        
        # 初始化物种线
        self.species_line = ax.plot([], [], 'm-', linewidth=2, marker='*', markersize=4)[0]
    
    def _setup_status_display(self):
        """设置状态显示"""
        ax = self.axes['status']
        ax.set_title('系统状态', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 初始化状态文本
        self.status_text = ax.text(0.1, 0.8, '', transform=ax.transAxes, 
                                  fontsize=12, verticalalignment='top',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _add_info_panel(self):
        """添加信息面板"""
        if len(self.evolution_data['generations']) == 0:
            return
        
        current_gen = self.evolution_data['generations'][-1] if self.evolution_data['generations'] else 0
        best_fitness = self.evolution_data['best_fitness'][-1] if self.evolution_data['best_fitness'] else 0
        
        # 添加总览信息
        info_text = f"""
当前代数: {current_gen}
最佳适应度: {best_fitness:.4f}
更新次数: {self.performance_metrics['update_count']}
最后更新: {datetime.now().strftime('%H:%M:%S')}
        """
        
        # 在图的右下角添加信息框
        self.fig.text(0.02, 0.02, info_text.strip(), fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    
    def _load_latest_data(self):
        """加载最新数据"""
        try:
            start_time = time.time()
            
            # 扫描数据目录
            data_files = self._scan_data_files()
            if not data_files:
                self.logger.warning("未找到进化数据文件")
                return
            
            # 加载最新的数据文件
            latest_file = max(data_files, key=lambda x: os.path.getmtime(x))
            generation = self._extract_generation_from_filename(latest_file)
            
            # 如果是新数据，才加载
            if generation > self.last_generation:
                self._load_data_file(latest_file)
                self.last_generation = generation
            
            # 更新性能指标
            load_duration = time.time() - start_time
            self.performance_metrics['last_update_duration'] = load_duration
            self.performance_metrics['update_count'] += 1
            
            # 更新平均时间
            if self.performance_metrics['update_count'] == 1:
                self.performance_metrics['average_update_duration'] = load_duration
            else:
                n = self.performance_metrics['update_count']
                self.performance_metrics['average_update_duration'] = (
                    (self.performance_metrics['average_update_duration'] * (n - 1) + load_duration) / n
                )
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
    
    def _scan_data_files(self) -> List[str]:
        """扫描数据目录中的文件"""
        data_files = []
        
        if not os.path.exists(self.data_dir):
            return data_files
        
        # 查找JSON格式的进化数据文件
        for file in os.listdir(self.data_dir):
            if file.startswith('generation_') and file.endswith('.json'):
                file_path = os.path.join(self.data_dir, file)
                data_files.append(file_path)
        
        return data_files
    
    def _extract_generation_from_filename(self, filename: str) -> int:
        """从文件名中提取代数"""
        basename = os.path.basename(filename)
        # 从 "generation_000123.json" 中提取 "000123"
        gen_str = basename.replace('generation_', '').replace('.json', '')
        try:
            return int(gen_str)
        except ValueError:
            return 0
    
    def _load_data_file(self, file_path: str):
        """加载单个数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            generation = data.get('generation', 0)
            pop_stats = data.get('population_stats', {})
            
            # 更新进化数据
            self.evolution_data['generations'].append(generation)
            self.evolution_data['best_fitness'].append(pop_stats.get('best_fitness', 0.0))
            self.evolution_data['avg_fitness'].append(pop_stats.get('avg_fitness', 0.0))
            self.evolution_data['worst_fitness'].append(pop_stats.get('worst_fitness', 0.0))
            self.evolution_data['diversity'].append(pop_stats.get('diversity', 0.0))
            self.evolution_data['species_count'].append(pop_stats.get('species_count', 1))
            self.evolution_data['timestamps'].append(data.get('timestamp', datetime.now().isoformat()))
            
            # 限制历史数据长度
            max_points = self.config['max_history_points']
            for key in ['generations', 'best_fitness', 'avg_fitness', 'worst_fitness', 
                       'diversity', 'species_count', 'timestamps']:
                if len(self.evolution_data[key]) > max_points:
                    self.evolution_data[key] = self.evolution_data[key][-max_points:]
            
            self.performance_metrics['data_points_loaded'] += 1
            
            self.logger.debug(f"加载数据: Gen {generation}")
            
        except Exception as e:
            self.logger.error(f"加载数据文件失败 {file_path}: {e}")
    
    def _update_dashboard(self, frame):
        """更新仪表板数据"""
        try:
            # 加载新数据
            self._load_latest_data()
            
            # 更新各个图表
            self._update_fitness_plot()
            self._update_diversity_plot()
            self._update_heatmap()
            self._update_trajectory()
            self._update_best_individual()
            self._update_species_plot()
            self._update_status_display()
            
            # 更新总览信息
            self._update_info_panel()
            
        except Exception as e:
            self.logger.error(f"更新仪表板失败: {e}")
    
    def _update_fitness_plot(self):
        """更新适应度曲线"""
        if 'fitness' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        ax = self.axes['fitness']
        generations = self.evolution_data['generations']
        
        # 更新数据
        self.lines_fitness['best'].set_data(generations, self.evolution_data['best_fitness'])
        self.lines_fitness['avg'].set_data(generations, self.evolution_data['avg_fitness'])
        self.lines_fitness['worst'].set_data(generations, self.evolution_data['worst_fitness'])
        
        # 更新填充区域
        self.lines_fitness['fill'].set_data(generations, self.evolution_data['best_fitness'])
        self.lines_fitness['fill'].set_ydata(self.evolution_data['avg_fitness'])
        
        # 更新坐标轴范围
        if len(generations) > 0:
            ax.set_xlim(max(0, generations[-1] - 100), generations[-1] + 10)
            
            fitness_values = (self.evolution_data['best_fitness'] + 
                            self.evolution_data['avg_fitness'] + 
                            self.evolution_data['worst_fitness'])
            if fitness_values:
                ax.set_ylim(min(fitness_values) * 0.9, max(fitness_values) * 1.1)
    
    def _update_diversity_plot(self):
        """更新多样性图"""
        if 'diversity' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        generations = self.evolution_data['generations']
        diversity_values = self.evolution_data['diversity']
        
        self.line_diversity.set_data(generations, diversity_values)
        
        # 更新坐标轴
        if len(generations) > 0:
            self.axes['diversity'].set_xlim(max(0, generations[-1] - 100), generations[-1] + 10)
    
    def _update_heatmap(self):
        """更新种群热图"""
        if 'heatmap' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        # 生成示例热图数据（实际应用中使用真实种群数据）
        ax = self.axes['heatmap']
        
        # 模拟种群热图
        gen = self.evolution_data['generations'][-1] if self.evolution_data['generations'] else 0
        np.random.seed(gen)  # 确保可重现性
        
        # 生成热图数据
        population_size = 50
        genome_length = min(20, 10 + gen // 10)  # 随代数增长
        
        heatmap_data = np.random.randn(population_size, genome_length)
        
        # 更新热图
        self.heatmap.set_data(heatmap_data)
        
        # 更新坐标轴
        ax.set_xlim(0, genome_length)
        ax.set_ylim(0, population_size)
    
    def _update_trajectory(self):
        """更新3D轨迹"""
        if 'trajectory' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        ax = self.axes['trajectory']
        
        # 3D轨迹数据
        avg_fitness = self.evolution_data['avg_fitness']
        best_fitness = self.evolution_data['best_fitness']
        diversity = self.evolution_data['diversity']
        
        self.trajectory_line.set_data(avg_fitness, best_fitness)
        self.trajectory_line.set_3d_properties(diversity)
        
        # 更新散点
        if len(avg_fitness) > 0:
            self.trajectory_points._offsets3d = (avg_fitness, best_fitness, diversity)
    
    def _update_best_individual(self):
        """更新最佳个体图"""
        if 'best' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        ax = self.axes['best']
        
        # 模拟最佳个体基因组
        gen = self.evolution_data['generations'][-1]
        genome_length = min(20, 5 + gen // 5)
        
        np.random.seed(gen + 42)  # 确保可重现性
        genome = np.sin(np.linspace(0, 4*np.pi, genome_length)) + 0.1 * np.random.randn(genome_length)
        
        gene_positions = list(range(genome_length))
        
        self.best_genome_line.set_data(gene_positions, genome)
        ax.set_xlim(0, genome_length)
    
    def _update_species_plot(self):
        """更新物种进化图"""
        if 'species' not in self.axes or len(self.evolution_data['generations']) == 0:
            return
        
        generations = self.evolution_data['generations']
        species_counts = self.evolution_data['species_count']
        
        self.species_line.set_data(generations, species_counts)
        
        # 更新坐标轴
        if len(generations) > 0:
            self.axes['species'].set_xlim(max(0, generations[-1] - 100), generations[-1] + 10)
    
    def _update_status_display(self):
        """更新状态显示"""
        if 'status' not in self.axes:
            return
        
        ax = self.axes['status']
        
        # 计算状态信息
        if len(self.evolution_data['generations']) == 0:
            status_info = "等待数据..."
        else:
            current_gen = self.evolution_data['generations'][-1]
            best_fitness = self.evolution_data['best_fitness'][-1] if self.evolution_data['best_fitness'] else 0
            
            # 分析进化状态
            if len(self.evolution_data['best_fitness']) >= 10:
                recent_improvement = (self.evolution_data['best_fitness'][-1] - 
                                    self.evolution_data['best_fitness'][-10])
                if recent_improvement > 0.1:
                    trend = "📈 快速进化"
                elif recent_improvement > 0:
                    trend = "📊 稳步进化"
                else:
                    trend = "📉 进化停滞"
            else:
                trend = "🔄 早期阶段"
            
            status_info = f"""
代数: {current_gen}
最佳适应度: {best_fitness:.4f}
进化趋势: {trend}

数据点: {self.performance_metrics['data_points_loaded']}
更新次数: {self.performance_metrics['update_count']}
平均更新时间: {self.performance_metrics['average_update_duration']*1000:.1f}ms
            """.strip()
        
        self.status_text.set_text(status_info)
    
    def _update_info_panel(self):
        """更新总览信息面板"""
        # 更新右下角的信息框
        if hasattr(self, 'fig'):
            info_texts = self.fig.texts[1:] if len(self.fig.texts) > 1 else []
            
            for text_obj in info_texts:
                if '当前代数' in text_obj.get_text():
                    current_gen = self.evolution_data['generations'][-1] if self.evolution_data['generations'] else 0
                    best_fitness = self.evolution_data['best_fitness'][-1] if self.evolution_data['best_fitness'] else 0
                    
                    info_text = f"""
当前代数: {current_gen}
最佳适应度: {best_fitness:.4f}
更新次数: {self.performance_metrics['update_count']}
最后更新: {datetime.now().strftime('%H:%M:%S')}
                    """.strip()
                    
                    text_obj.set_text(info_text)
                    break
    
    def _add_analysis_overlay(self):
        """添加分析覆盖层"""
        if len(self.evolution_data['generations']) == 0:
            return
        
        # 添加进化分析文本
        analysis_text = self._generate_evolution_analysis()
        
        # 在图的左上角添加分析信息
        self.fig.text(0.02, 0.95, analysis_text, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                     verticalalignment='top')
    
    def _generate_evolution_analysis(self) -> str:
        """生成进化分析报告"""
        if len(self.evolution_data['generations']) < 2:
            return "数据不足，无法分析"
        
        generations = self.evolution_data['generations']
        best_fitness = self.evolution_data['best_fitness']
        diversity = self.evolution_data['diversity']
        
        # 计算分析指标
        total_generations = generations[-1]
        fitness_improvement = best_fitness[-1] - best_fitness[0]
        avg_diversity = np.mean(diversity) if diversity else 0
        
        # 进化效率分析
        if total_generations > 0:
            evolution_rate = fitness_improvement / total_generations
        else:
            evolution_rate = 0
        
        # 多样性分析
        if avg_diversity > 1.0:
            diversity_status = "丰富"
        elif avg_diversity > 0.5:
            diversity_status = "适中"
        else:
            diversity_status = "不足"
        
        # 停滞期检测
        stagnation_period = self._detect_stagnation_period()
        
        analysis = f"""
进化分析报告
━━━━━━━━━━━━━
总代数: {total_generations}
适应度改善: {fitness_improvement:.4f}
进化速率: {evolution_rate:.6f}/代
多样性状态: {diversity_status}
停滞期: {stagnation_period}代

建议:
{self._generate_recommendations(evolution_rate, avg_diversity, stagnation_period)}
        """.strip()
        
        return analysis
    
    def _detect_stagnation_period(self, window_size: int = 20) -> int:
        """检测进化停滞期"""
        if len(self.evolution_data['best_fitness']) < window_size:
            return 0
        
        recent_fitness = self.evolution_data['best_fitness'][-window_size:]
        fitness_variance = np.var(recent_fitness)
        
        # 如果方差很小，认为是停滞期
        return window_size if fitness_variance < 1e-6 else 0
    
    def _generate_recommendations(self, evolution_rate: float, diversity: float, stagnation: int) -> str:
        """生成建议"""
        recommendations = []
        
        if stagnation > 10:
            recommendations.append("• 增加变异率以打破停滞")
            recommendations.append("• 考虑引入新的基因变种")
        
        if diversity < 0.3:
            recommendations.append("• 种群多样性不足，增加选择压力")
        
        if evolution_rate < 0.001:
            recommendations.append("• 进化速率较慢，调整参数")
        elif evolution_rate > 0.1:
            recommendations.append("• 进化过快，注意收敛质量")
        
        if not recommendations:
            recommendations.append("• 当前进化状态良好")
        
        return "\n".join(recommendations)
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前仪表板状态
        
        Returns:
            状态信息字典
        """
        status = {
            'dashboard_active': self.fig is not None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'current_generation': self.evolution_data['generations'][-1] if self.evolution_data['generations'] else 0,
            'data_points_available': len(self.evolution_data['generations']),
            'performance_metrics': self.performance_metrics.copy(),
            'auto_reload_enabled': self.auto_reload,
            'update_interval': self.update_interval,
            'configured_plots': [name for name, enabled in [
                ('fitness_curve', self.config['show_fitness_curve']),
                ('diversity', self.config['show_diversity_plot']),
                ('heatmap', self.config['show_population_heatmap']),
                ('trajectory', self.config['show_3d_trajectory']),
                ('best_individual', self.config['show_best_individual']),
                ('species_evolution', self.config['show_species_evolution'])
            ] if enabled]
        }
        
        return status
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.animation_obj:
                self.animation_obj.event_source.stop()
            
            if self.fig:
                plt.close(self.fig)
            
            self.logger.info("仪表板资源清理完成")
            
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")


if __name__ == "__main__":
    # 测试代码
    print("EvolutionDashboard 模块测试")
    
    # 创建仪表板
    dashboard = EvolutionDashboard(
        data_dir="../data/evolution_logs",
        update_interval=1.0,
        auto_reload=True
    )
    
    # 创建静态仪表板
    output_path = "test_dashboard.png"
    dashboard.create_static_dashboard(output_path, include_analysis=True)
    
    # 获取状态
    status = dashboard.get_current_status()
    print("仪表板状态:", json.dumps(status, indent=2, ensure_ascii=False))
    
    print(f"EvolutionDashboard 测试完成，输出: {output_path}")