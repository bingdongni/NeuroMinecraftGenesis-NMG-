#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化可视化器 - 实时监控进化过程

实现功能：
- 实时进化曲线可视化
- 适应度地形3D展示
- 遗传多样性变化监控
- 种群进化历史记录
- 进化树可视化

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
import logging
from collections import deque
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EvolutionVisualizer:
    """
    进化可视化器类
    
    提供实时进化监控、3D可视化和数据记录功能
    """
    
    def __init__(self, 
                 population_size: int,
                 genome_length: int,
                 data_dir: str = "data/evolution_logs",
                 checkpoint_dir: str = "models/genomes",
                 max_history: int = 1000):
        """
        初始化进化可视化器
        
        Args:
            population_size: 种群大小
            genome_length: 基因组长度
            data_dir: 进化数据保存目录
            checkpoint_dir: 检查点保存目录
            max_history: 最大历史记录数量
        """
        self.population_size = population_size
        self.genome_length = genome_length
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.max_history = max_history
        
        # 创建必要的目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 进化历史数据存储
        self.evolution_history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'species_count': [],
            'timestamp': []
        }
        
        # 当前种群状态
        self.current_population = []
        self.current_fitness = []
        self.current_generation = 0
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 图表句柄
        self.fig = None
        self.axes = {}
        
        self.logger.info("进化可视化器初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_dir, 'evolution_visualizer.log')),
                logging.StreamHandler()
            ]
        )
    
    def update_population_state(self, 
                               population: List[np.ndarray], 
                               fitness_scores: List[float],
                               generation: int):
        """
        更新种群状态并记录数据
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数列表
            generation: 当前代数
        """
        self.current_population = population
        self.current_fitness = fitness_scores
        self.current_generation = generation
        
        # 计算统计指标
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        worst_fitness = min(fitness_scores)
        diversity = self._calculate_genetic_diversity(population)
        
        # 计算物种数量（基于基因相似度聚类）
        species_count = self._estimate_species_count(population)
        
        # 记录历史数据
        self.evolution_history['generation'].append(generation)
        self.evolution_history['best_fitness'].append(best_fitness)
        self.evolution_history['avg_fitness'].append(avg_fitness)
        self.evolution_history['worst_fitness'].append(worst_fitness)
        self.evolution_history['diversity'].append(diversity)
        self.evolution_history['species_count'].append(species_count)
        self.evolution_history['timestamp'].append(datetime.now().isoformat())
        
        # 限制历史记录数量
        for key in self.evolution_history:
            if len(self.evolution_history[key]) > self.max_history:
                self.evolution_history[key] = self.evolution_history[key][-self.max_history:]
        
        self.logger.info(f"Generation {generation}: Best={best_fitness:.4f}, "
                        f"Avg={avg_fitness:.4f}, Diversity={diversity:.4f}")
    
    def visualize_evolution_progress(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        实时进化进度可视化并保存数据
        
        Args:
            save_path: 额外的保存路径
            
        Returns:
            保存的进度数据字典
        """
        if len(self.evolution_history['generation']) == 0:
            self.logger.warning("没有可用的进化历史数据")
            return {}
        
        # 创建主图
        if self.fig is None:
            self.fig, self.axes = self._create_evolution_plots()
        else:
            self._update_evolution_plots()
        
        # 保存当前代数数据到JSON文件
        progress_data = self._save_current_generation_data()
        
        # 更新图表
        self.fig.suptitle(f'进化进度监控 - 第{self.current_generation}代', 
                         fontsize=16, fontweight='bold')
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.data_dir, f'evolution_progress_gen{self.current_generation}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 不显示图像，专注于数据记录
        plt.close(self.fig)
        self.fig = None
        
        self.logger.info(f"进化进度可视化保存至: {save_path}")
        return progress_data
    
    def _create_evolution_plots(self) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """创建进化图表布局"""
        fig = plt.figure(figsize=(16, 12))
        
        # 创建子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        axes = {}
        
        # 1. 适应度曲线图
        axes['fitness'] = fig.add_subplot(gs[0, :2])
        axes['fitness'].set_title('进化曲线', fontsize=14, fontweight='bold')
        axes['fitness'].set_xlabel('代数')
        axes['fitness'].set_ylabel('适应度')
        axes['fitness'].grid(True, alpha=0.3)
        
        # 2. 遗传多样性图
        axes['diversity'] = fig.add_subplot(gs[0, 2])
        axes['diversity'].set_title('遗传多样性', fontsize=14, fontweight='bold')
        axes['diversity'].set_xlabel('代数')
        axes['diversity'].set_ylabel('多样性指数')
        axes['diversity'].grid(True, alpha=0.3)
        
        # 3. 物种数量变化
        axes['species'] = fig.add_subplot(gs[1, 0])
        axes['species'].set_title('物种数量', fontsize=14, fontweight='bold')
        axes['species'].set_xlabel('代数')
        axes['species'].set_ylabel('物种数')
        axes['species'].grid(True, alpha=0.3)
        
        # 4. 适应度分布直方图
        axes['fitness_dist'] = fig.add_subplot(gs[1, 1])
        axes['fitness_dist'].set_title(f'适应度分布 (第{self.current_generation}代)', 
                                      fontsize=14, fontweight='bold')
        axes['fitness_dist'].set_xlabel('适应度')
        axes['fitness_dist'].set_ylabel('个体数量')
        axes['fitness_dist'].grid(True, alpha=0.3)
        
        # 5. 基因型分布热图
        axes['genotype'] = fig.add_subplot(gs[1, 2])
        axes['genotype'].set_title('基因型分布', fontsize=14, fontweight='bold')
        axes['genotype'].set_xlabel('基因位置')
        axes['genotype'].set_ylabel('个体')
        
        # 6. 进化速率分析
        axes['evolution_rate'] = fig.add_subplot(gs[2, :2])
        axes['evolution_rate'].set_title('进化速率分析', fontsize=14, fontweight='bold')
        axes['evolution_rate'].set_xlabel('代数')
        axes['evolution_rate'].set_ylabel('适应度变化率')
        axes['evolution_rate'].grid(True, alpha=0.3)
        
        # 7. 相空间轨迹
        axes['phase_space'] = fig.add_subplot(gs[2, 2], projection='3d')
        axes['phase_space'].set_title('3D进化轨迹', fontsize=14, fontweight='bold')
        axes['phase_space'].set_xlabel('平均适应度')
        axes['phase_space'].set_ylabel('最佳适应度')
        axes['phase_space'].set_zlabel('多样性')
        
        return fig, axes
    
    def _update_evolution_plots(self):
        """更新进化图表"""
        if not self.axes:
            return
            
        history = self.evolution_history
        if len(history['generation']) == 0:
            return
        
        generation = history['generation']
        best_fitness = history['best_fitness']
        avg_fitness = history['avg_fitness']
        diversity = history['diversity']
        species_count = history['species_count']
        
        # 更新适应度曲线
        axes = self.axes
        axes['fitness'].clear()
        axes['fitness'].plot(generation, best_fitness, 'r-', linewidth=2, label='最佳适应度', marker='o', markersize=3)
        axes['fitness'].plot(generation, avg_fitness, 'b-', linewidth=2, label='平均适应度', marker='s', markersize=3)
        axes['fitness'].fill_between(generation, best_fitness, avg_fitness, alpha=0.2, color='green')
        axes['fitness'].legend()
        axes['fitness'].grid(True, alpha=0.3)
        axes['fitness'].set_title('进化曲线')
        axes['fitness'].set_xlabel('代数')
        axes['fitness'].set_ylabel('适应度')
        
        # 更新遗传多样性
        axes['diversity'].clear()
        axes['diversity'].plot(generation, diversity, 'g-', linewidth=2, marker='^', markersize=3)
        axes['diversity'].grid(True, alpha=0.3)
        axes['diversity'].set_title('遗传多样性')
        axes['diversity'].set_xlabel('代数')
        axes['diversity'].set_ylabel('多样性指数')
        
        # 更新物种数量
        axes['species'].clear()
        axes['species'].plot(generation, species_count, 'm-', linewidth=2, marker='D', markersize=3)
        axes['species'].grid(True, alpha=0.3)
        axes['species'].set_title('物种数量')
        axes['species'].set_xlabel('代数')
        axes['species'].set_ylabel('物种数')
        
        # 更新适应度分布
        if len(self.current_fitness) > 0:
            axes['fitness_dist'].clear()
            axes['fitness_dist'].hist(self.current_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes['fitness_dist'].axvline(np.mean(self.current_fitness), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(self.current_fitness):.3f}')
            axes['fitness_dist'].legend()
            axes['fitness_dist'].set_title(f'适应度分布 (第{self.current_generation}代)')
            axes['fitness_dist'].set_xlabel('适应度')
            axes['fitness_dist'].set_ylabel('个体数量')
        
        # 更新基因型分布
        if len(self.current_population) > 0:
            axes['genotype'].clear()
            # 计算基因型分布矩阵
            genotype_matrix = self._create_genotype_matrix()
            im = axes['genotype'].imshow(genotype_matrix, cmap='viridis', aspect='auto')
            axes['genotype'].set_title('基因型分布')
            axes['genotype'].set_xlabel('基因位置')
            axes['genotype'].set_ylabel('个体')
            plt.colorbar(im, ax=axes['genotype'])
        
        # 更新进化速率
        if len(avg_fitness) > 1:
            axes['evolution_rate'].clear()
            evolution_rate = np.diff(avg_fitness)
            axes['evolution_rate'].plot(generation[1:], evolution_rate, 'orange', linewidth=2, marker='*', markersize=3)
            axes['evolution_rate'].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes['evolution_rate'].grid(True, alpha=0.3)
            axes['evolution_rate'].set_title('进化速率分析')
            axes['evolution_rate'].set_xlabel('代数')
            axes['evolution_rate'].set_ylabel('适应度变化率')
        
        # 更新3D轨迹
        if len(avg_fitness) > 1:
            axes['phase_space'].clear()
            axes['phase_space'].plot(avg_fitness, best_fitness, diversity, 'b-', linewidth=2, alpha=0.7)
            # 标记起点和终点
            if len(avg_fitness) > 0:
                axes['phase_space'].scatter(avg_fitness[0], best_fitness[0], diversity[0], 
                                          color='green', s=100, label='起点', marker='o')
                axes['phase_space'].scatter(avg_fitness[-1], best_fitness[-1], diversity[-1], 
                                          color='red', s=100, label='终点', marker='s')
            axes['phase_space'].set_xlabel('平均适应度')
            axes['phase_space'].set_ylabel('最佳适应度')
            axes['phase_space'].set_zlabel('多样性')
            axes['phase_space'].legend()
    
    def plot_fitness_landscape(self, 
                              gene_range: Tuple[float, float] = (-5.0, 5.0),
                              resolution: int = 50,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制3D适应度地形图
        
        Args:
            gene_range: 基因值范围
            resolution: 地形分辨率
            save_path: 保存路径
            
        Returns:
            地形分析数据
        """
        self.logger.info("开始绘制3D适应度地形图...")
        
        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 生成网格
        x = np.linspace(gene_range[0], gene_range[1], resolution)
        y = np.linspace(gene_range[0], gene_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # 计算适应度地形（使用示例函数）
        Z = self._calculate_fitness_landscape(X, Y)
        
        # 绘制3D表面
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # 添加当前种群点
        if len(self.current_population) > 0:
            for i, individual in enumerate(self.current_population[:20]):  # 只显示前20个个体
                if len(individual) >= 2:
                    ax.scatter(individual[0], individual[1], 
                             self.current_fitness[i] if i < len(self.current_fitness) else 0,
                             color='red', s=50, alpha=0.8)
        
        # 添加最佳个体
        if len(self.current_population) > 0:
            best_idx = np.argmax(self.current_fitness)
            best_individual = self.current_population[best_idx]
            if len(best_individual) >= 2:
                ax.scatter(best_individual[0], best_individual[1], 
                         self.current_fitness[best_idx],
                         color='gold', s=200, marker='*', 
                         label=f'最佳个体 (适应度: {self.current_fitness[best_idx]:.3f})')
                ax.legend()
        
        # 设置标签和标题
        ax.set_xlabel('基因 1', fontsize=12)
        ax.set_ylabel('基因 2', fontsize=12)
        ax.set_zlabel('适应度', fontsize=12)
        ax.set_title('3D适应度地形图', fontsize=16, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.data_dir, f'fitness_landscape_gen{self.current_generation}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分析地形特征
        landscape_analysis = self._analyze_landscape(Z, X, Y)
        
        self.logger.info(f"适应度地形图保存至: {save_path}")
        return {
            'landscape_data': {
                'x_range': gene_range,
                'y_range': gene_range,
                'resolution': resolution,
                'peak_values': landscape_analysis['peaks'],
                'valley_values': landscape_analysis['valleys'],
                'gradient_analysis': landscape_analysis['gradients']
            },
            'save_path': save_path
        }
    
    def render_evolution_tree(self, 
                             max_depth: int = 10,
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        渲染进化树可视化
        
        Args:
            max_depth: 最大深度
            save_path: 保存路径
            
        Returns:
            进化树数据
        """
        self.logger.info("开始渲染进化树...")
        
        # 创建进化树数据
        evolution_tree_data = self._create_evolution_tree_data(max_depth)
        
        # 使用matplotlib绘制进化树
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # 绘制进化树
        self._draw_evolution_tree(ax, evolution_tree_data)
        
        # 设置标题和标签
        ax.set_title('进化树可视化', fontsize=16, fontweight='bold')
        ax.set_xlabel('进化时间轴', fontsize=12)
        ax.set_ylabel('物种谱系', fontsize=12)
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.data_dir, f'evolution_tree_gen{self.current_generation}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存进化树数据
        tree_data_path = save_path.replace('.png', '_data.json')
        with open(tree_data_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tree_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"进化树保存至: {save_path}")
        return {
            'tree_data': evolution_tree_data,
            'save_path': save_path,
            'data_path': tree_data_path
        }
    
    def _calculate_genetic_diversity(self, population: List[np.ndarray]) -> float:
        """
        计算遗传多样性指数
        
        Args:
            population: 种群
            
        Returns:
            多样性指数
        """
        if len(population) < 2:
            return 0.0
        
        # 计算种群基因型距离矩阵
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        
        # 使用平均距离作为多样性指标
        diversity = np.mean(distances) if distances else 0.0
        
        return diversity
    
    def _estimate_species_count(self, population: List[np.ndarray], 
                               threshold: float = 0.5) -> int:
        """
        估算物种数量（基于基因相似度聚类）
        
        Args:
            population: 种群
            threshold: 相似度阈值
            
        Returns:
            估算的物种数量
        """
        if len(population) < 2:
            return 1
        
        # 简单的基于距离的聚类
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # 构建距离矩阵
        distances = np.zeros((len(population), len(population)))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances[i, j] = distances[j, i] = dist
        
        # 层次聚类
        if len(population) > 2:
            condensed_dist = distances[np.triu_indices_from(distances, k=1)]
            linkage_matrix = linkage(condensed_dist, method='average')
            cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
            return len(np.unique(cluster_labels))
        else:
            return 1
    
    def _save_current_generation_data(self) -> Dict[str, Any]:
        """保存当前代数的数据到JSON"""
        current_data = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'population_stats': {
                'best_fitness': float(max(self.current_fitness)) if self.current_fitness else 0.0,
                'avg_fitness': float(np.mean(self.current_fitness)) if self.current_fitness else 0.0,
                'worst_fitness': float(min(self.current_fitness)) if self.current_fitness else 0.0,
                'diversity': self._calculate_genetic_diversity(self.current_population),
                'species_count': self._estimate_species_count(self.current_population)
            },
            'population_size': len(self.current_population),
            'genome_length': self.genome_length
        }
        
        # 保存到JSON文件
        json_path = os.path.join(self.data_dir, f'generation_{self.current_generation:05d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        return current_data
    
    def _create_genotype_matrix(self) -> np.ndarray:
        """创建基因型分布矩阵"""
        if len(self.current_population) == 0:
            return np.zeros((10, min(self.genome_length, 20)))
        
        # 取前20个个体和前20个基因位点
        num_individuals = min(20, len(self.current_population))
        num_genes = min(20, self.genome_length)
        
        matrix = np.zeros((num_individuals, num_genes))
        for i in range(num_individuals):
            individual = self.current_population[i]
            for j in range(num_genes):
                if j < len(individual):
                    matrix[i, j] = individual[j]
        
        return matrix
    
    def _calculate_fitness_landscape(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        计算适应度地形函数
        
        Args:
            X: X坐标网格
            Y: Y坐标网格
            
        Returns:
            适应度值网格
        """
        # 使用多模态函数创建复杂地形
        Z = np.zeros_like(X)
        
        # 添加多个峰值
        centers = [(1, 1), (-1, -1), (2, -2), (-2, 2)]
        for cx, cy in centers:
            Z += np.exp(-((X - cx)**2 + (Y - cy)**2))
        
        # 添加噪声
        Z += 0.1 * np.random.normal(0, 1, Z.shape)
        
        return Z
    
    def _analyze_landscape(self, Z: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Dict[str, List[Tuple]]:
        """分析地形特征"""
        # 找到峰值
        from scipy.ndimage import maximum_filter, minimum_filter
        
        max_filtered = maximum_filter(Z, size=5)
        peaks = np.where(Z == max_filtered)
        peak_values = [(X[i, j], Y[i, j], Z[i, j]) for i, j in zip(peaks[0], peaks[1])]
        
        # 找到谷值
        min_filtered = minimum_filter(Z, size=5)
        valleys = np.where(Z == min_filtered)
        valley_values = [(X[i, j], Y[i, j], Z[i, j]) for i, j in zip(valleys[0], valleys[1])]
        
        # 梯度分析
        grad_x, grad_y = np.gradient(Z)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        steep_regions = np.where(gradient_magnitude > np.percentile(gradient_magnitude, 90))
        gradients = [(X[i, j], Y[i, j], gradient_magnitude[i, j]) for i, j in zip(steep_regions[0], steep_regions[1])]
        
        return {
            'peaks': peak_values[:10],  # 限制数量
            'valleys': valley_values[:10],
            'gradients': gradients[:20]
        }
    
    def _create_evolution_tree_data(self, max_depth: int) -> Dict[str, Any]:
        """创建进化树数据结构"""
        tree_data = {
            'root': {
                'id': 'generation_0',
                'generation': 0,
                'fitness': 0.0,
                'children': [],
                'species_id': 0
            },
            'current_generation': self.current_generation,
            'max_depth': max_depth,
            'total_species': 0,
            'branching_events': []
        }
        
        # 如果有进化历史，构建树结构
        if len(self.evolution_history['generation']) > 1:
            self._build_evolution_tree(tree_data)
        
        return tree_data
    
    def _build_evolution_tree(self, tree_data: Dict[str, Any]):
        """构建进化树"""
        history = self.evolution_history
        
        # 简化版本：基于物种数量变化构建树
        species_counts = history['species_count']
        fitness_values = history['best_fitness']
        
        current_species_id = 1
        last_branch_gen = 0
        
        for gen in range(1, len(species_counts)):
            # 如果物种数量显著增加，认为发生了分支事件
            if species_counts[gen] > species_counts[gen-1] * 1.5:
                branch_event = {
                    'generation': gen,
                    'parent_species': current_species_id - 1,
                    'new_species': current_species_id,
                    'fitness_improvement': fitness_values[gen] - fitness_values[gen-1]
                }
                tree_data['branching_events'].append(branch_event)
                current_species_id += 1
                last_branch_gen = gen
    
    def _draw_evolution_tree(self, ax: plt.Axes, tree_data: Dict[str, Any]):
        """绘制进化树"""
        history = self.evolution_history
        
        if len(history['generation']) == 0:
            ax.text(0.5, 0.5, '暂无进化数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            return
        
        generation = history['generation']
        
        # 绘制主干
        ax.plot([0, 0], [0, len(generation)-1], 'k-', linewidth=3, label='进化主干')
        
        # 绘制分支
        species_count = history['species_count']
        fitness_values = history['best_fitness']
        
        for i, (gen, species, fitness) in enumerate(zip(generation, species_count, fitness_values)):
            # 标记主要节点
            ax.plot(0, i, 'ro', markersize=8)
            
            # 绘制分支
            if species > 1:
                branch_length = min(species * 0.5, 3.0)
                ax.plot([0, branch_length], [i, i], 'b-', linewidth=2, alpha=0.7)
                ax.plot(branch_length, i, 'bs', markersize=6)
            
            # 添加世代标签
            ax.text(-0.5, i, f'Gen {gen}', ha='right', va='center', fontsize=8)
        
        # 添加进化速率信息
        ax.text(3.5, len(generation)-1, 
               f'总代数: {generation[-1]}\n'
               f'最终适应度: {fitness_values[-1]:.3f}\n'
               f'峰值适应度: {max(fitness_values):.3f}',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
               fontsize=10)
        
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, len(generation))
        ax.set_aspect('equal')
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        获取进化过程摘要
        
        Returns:
            进化摘要数据
        """
        if len(self.evolution_history['generation']) == 0:
            return {"message": "暂无进化数据"}
        
        history = self.evolution_history
        best_fitness_idx = np.argmax(history['best_fitness'])
        
        summary = {
            'current_generation': self.current_generation,
            'total_generations': len(history['generation']),
            'overall_best_fitness': float(history['best_fitness'][best_fitness_idx]),
            'overall_best_generation': int(history['generation'][best_fitness_idx]),
            'current_diversity': float(history['diversity'][-1]) if history['diversity'] else 0.0,
            'evolution_progress': {
                'improvement_rate': float((history['best_fitness'][-1] - history['best_fitness'][0]) / 
                                        max(1, self.current_generation)),
                'stagnation_period': self._detect_stagnation(),
                'convergence_level': float(history['best_fitness'][-1] / max(history['best_fitness']))
            },
            'species_evolution': {
                'current_species_count': int(history['species_count'][-1]) if history['species_count'] else 1,
                'max_species_count': int(max(history['species_count'])) if history['species_count'] else 1,
                'diversity_trend': self._analyze_diversity_trend()
            }
        }
        
        return summary
    
    def _detect_stagnation(self) -> int:
        """检测进化停滞期"""
        if len(self.evolution_history['best_fitness']) < 10:
            return 0
        
        # 检查最近10代是否有显著改善
        recent_fitness = self.evolution_history['best_fitness'][-10:]
        fitness_variance = np.var(recent_fitness)
        
        # 如果方差很小，认为是停滞期
        return 10 if fitness_variance < 1e-6 else 0
    
    def _analyze_diversity_trend(self) -> str:
        """分析多样性趋势"""
        if len(self.evolution_history['diversity']) < 5:
            return "数据不足"
        
        diversity_values = self.evolution_history['diversity']
        
        # 计算趋势
        recent_trend = np.mean(diversity_values[-5:]) - np.mean(diversity_values[-10:-5])
        
        if recent_trend > 0.01:
            return "多样性增加"
        elif recent_trend < -0.01:
            return "多样性减少"
        else:
            return "多样性稳定"


if __name__ == "__main__":
    # 测试代码
    print("EvolutionVisualizer 模块测试")
    
    # 创建可视化器
    visualizer = EvolutionVisualizer(
        population_size=100,
        genome_length=10
    )
    
    # 模拟进化数据
    for gen in range(20):
        # 模拟种群
        population = [np.random.randn(10) for _ in range(100)]
        fitness_scores = [np.sum(individual**2) + np.random.normal(0, 0.1) for individual in population]
        
        # 更新状态
        visualizer.update_population_state(population, fitness_scores, gen)
        
        # 每5代保存一次可视化
        if gen % 5 == 0:
            visualizer.visualize_evolution_progress()
    
    # 生成3D地形图
    visualizer.plot_fitness_landscape()
    
    # 生成进化树
    visualizer.render_evolution_tree()
    
    # 获取摘要
    summary = visualizer.get_evolution_summary()
    print("进化摘要:", json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("EvolutionVisualizer 测试完成")