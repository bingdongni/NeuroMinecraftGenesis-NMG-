#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化可视化和断点续跑系统演示

展示完整的进化监控、可视化和断点续跑功能

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import os
import sys
import time
import numpy as np
import json
from typing import Dict, List, Optional
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.evolution.evolution_visualizer import EvolutionVisualizer
from core.evolution.checkpoint_manager import CheckpointManager
from utils.visualization.evolution_dashboard import EvolutionDashboard


class EvolutionDemo:
    """
    进化系统演示类
    
    展示完整的进化可视化和断点续跑流程
    """
    
    def __init__(self,
                 population_size: int = 100,
                 genome_length: int = 10,
                 total_generations: int = 50):
        """
        初始化演示系统
        
        Args:
            population_size: 种群大小
            genome_length: 基因组长度
            total_generations: 总代数
        """
        self.population_size = population_size
        self.genome_length = genome_length
        self.total_generations = total_generations
        
        # 初始化组件
        self.visualizer = EvolutionVisualizer(
            population_size=population_size,
            genome_length=genome_length,
            data_dir="data/evolution_logs",
            checkpoint_dir="models/genomes",
            max_history=1000
        )
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="models/genomes",
            auto_save_interval=5,
            max_checkpoints=50,
            backup_enabled=True
        )
        
        self.dashboard = EvolutionDashboard(
            data_dir="data/evolution_logs",
            update_interval=2.0,
            auto_reload=True
        )
        
        # 进化状态
        self.current_population = []
        self.current_fitness = []
        self.current_generation = 0
        
        print(f"进化系统演示初始化完成")
        print(f"种群大小: {population_size}")
        print(f"基因组长度: {genome_length}")
        print(f"总代数: {total_generations}")
    
    def run_full_evolution(self):
        """运行完整的进化过程演示"""
        print("\n" + "="*60)
        print("开始完整进化过程演示")
        print("="*60)
        
        # 阶段1：正常进化
        print("\n阶段1: 正常进化过程...")
        self._run_evolution_phase(30, "阶段1_正常进化")
        
        # 阶段2：演示断点保存
        print("\n阶段2: 保存检查点...")
        self._save_checkpoint_demo()
        
        # 阶段3：模拟系统重启，恢复进化
        print("\n阶段3: 模拟系统重启，恢复进化...")
        self._simulate_restart_and_continue()
        
        # 阶段4：继续进化
        print("\n阶段4: 继续进化过程...")
        self._run_evolution_phase(self.total_generations - 30, "阶段4_继续进化")
        
        # 生成最终报告
        print("\n阶段5: 生成最终报告...")
        self._generate_final_report()
        
        print("\n进化过程演示完成！")
    
    def _run_evolution_phase(self, generations: int, phase_name: str):
        """
        运行指定阶段的进化过程
        
        Args:
            generations: 进化的代数
            phase_name: 阶段名称
        """
        print(f"\n运行 {phase_name}，共 {generations} 代...")
        
        for gen in range(generations):
            actual_gen = self.current_generation + gen + 1
            
            # 模拟进化过程
            population, fitness_scores = self._simulate_evolution_step(actual_gen)
            
            # 更新状态
            self.current_population = population
            self.current_fitness = fitness_scores
            self.current_generation = actual_gen
            
            # 更新可视化器
            self.visualizer.update_population_state(
                population, fitness_scores, actual_gen
            )
            
            # 自动保存检查点
            if self.checkpoint_manager.should_auto_save(actual_gen):
                checkpoint_info = self.checkpoint_manager.save_checkpoint(
                    population, fitness_scores, actual_gen,
                    checkpoint_type="auto",
                    description=f"{phase_name}_自动保存"
                )
                print(f"  Gen {actual_gen}: 保存检查点 - Fitness: {fitness_scores[0]:.4f}")
            
            # 每10代生成可视化
            if gen % 10 == 0 or gen == generations - 1:
                self._generate_visualizations(actual_gen, phase_name)
            
            # 打印进度
            if gen % 5 == 0:
                print(f"  完成 {gen+1}/{generations} 代 (总计第 {actual_gen} 代)")
        
        print(f"{phase_name} 完成")
    
    def _simulate_evolution_step(self, generation: int) -> tuple:
        """
        模拟单步进化
        
        Args:
            generation: 当前代数
            
        Returns:
            (population, fitness_scores) 元组
        """
        # 如果是第一次进化，初始化种群
        if not self.current_population:
            population = [np.random.randn(self.genome_length) for _ in range(self.population_size)]
        else:
            # 基于上一代进行进化
            population = self._evolve_population(self.current_population, generation)
        
        # 计算适应度（使用Rastrigin函数的变种）
        fitness_scores = []
        for individual in population:
            # Rastrigin函数: f(x) = 10*n + sum(xi^2 - 10*cos(2*pi*xi))
            # 我们要最小化，所以取负值作为我们的适应度
            base_fitness = -10 * self.genome_length
            individual_fitness = base_fitness - np.sum(
                individual**2 - 10 * np.cos(2 * np.pi * individual)
            )
            # 添加噪声和进化趋势
            individual_fitness += 0.1 * generation  # 进化改善
            individual_fitness += np.random.normal(0, 0.5)  # 随机噪声
            
            fitness_scores.append(individual_fitness)
        
        # 排序（适应度高的在前）
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        return population, fitness_scores
    
    def _evolve_population(self, previous_population: List[np.ndarray], generation: int) -> List[np.ndarray]:
        """
        基于上一代进化出新种群
        
        Args:
            previous_population: 上一代种群
            generation: 当前代数
            
        Returns:
            新一代种群
        """
        new_population = []
        
        # 精英保留（前10%）
        elite_count = max(1, self.population_size // 10)
        for i in range(elite_count):
            new_population.append(previous_population[i].copy())
        
        # 交叉和变异
        mutation_rate = max(0.01, 0.1 * (1 - generation / (self.total_generations + 50)))  # 逐渐减少变异率
        
        while len(new_population) < self.population_size:
            # 选择两个父代（锦标赛选择）
            parent1 = self._tournament_selection(previous_population)
            parent2 = self._tournament_selection(previous_population)
            
            # 交叉
            if np.random.random() < 0.8:  # 80%交叉率
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 变异
            child = self._mutate(child, mutation_rate)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population: List[np.ndarray], tournament_size: int = 3) -> np.ndarray:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament = [population[i] for i in tournament_indices]
        
        # 返回适应度最高的个体（这里简化为随机选择）
        return tournament[np.random.randint(tournament_size)]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """单点交叉"""
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        return child
    
    def _mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """高斯变异"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
        
        return mutated
    
    def _generate_visualizations(self, generation: int, phase_name: str):
        """生成可视化数据"""
        try:
            # 生成进化进度图
            progress_path = f"data/evolution_logs/{phase_name}_progress_gen{generation:05d}.png"
            self.visualizer.visualize_evolution_progress(progress_path)
            
            # 每20代生成3D地形图
            if generation % 20 == 0:
                landscape_path = f"data/evolution_logs/{phase_name}_landscape_gen{generation:05d}.png"
                self.visualizer.plot_fitness_landscape(save_path=landscape_path)
            
            # 每30代生成进化树
            if generation % 30 == 0:
                tree_path = f"data/evolution_logs/{phase_name}_tree_gen{generation:05d}.png"
                self.visualizer.render_evolution_tree(save_path=tree_path)
                
        except Exception as e:
            print(f"  生成可视化时出错: {e}")
    
    def _save_checkpoint_demo(self):
        """演示检查点保存功能"""
        print("演示手动保存检查点...")
        
        # 保存一个手动检查点
        checkpoint_info = self.checkpoint_manager.save_checkpoint(
            self.current_population,
            self.current_fitness,
            self.current_generation,
            checkpoint_type="manual",
            description="手动保存的检查点 - 演示用途"
        )
        
        print(f"手动检查点保存: Gen {self.current_generation}")
        print(f"文件路径: {checkpoint_info['filepath']}")
        
        # 列出所有检查点
        checkpoints = self.checkpoint_manager.list_checkpoints(limit=10)
        print(f"\n当前共有 {len(checkpoints)} 个检查点")
        for cp in checkpoints[:3]:
            print(f"  Gen {cp['generation']}: {cp['type']} - Fitness: {cp['best_fitness']:.4f}")
    
    def _simulate_restart_and_continue(self):
        """模拟系统重启并恢复进化"""
        print("\n模拟系统重启...")
        
        # 模拟系统关闭和重启
        print("  - 保存当前状态...")
        print("  - 关闭系统...")
        print("  - 重启系统...")
        print("  - 加载检查点...")
        
        # 加载最新的检查点
        load_result = self.checkpoint_manager.load_checkpoint()
        
        if load_result:
            state = load_result['state']
            self.current_generation = state['generation']
            self.current_population = state['population']
            self.current_fitness = state['fitness_scores']
            
            print(f"  恢复成功! 从第 {self.current_generation} 代继续")
            print(f"  最佳适应度: {state['best_fitness']:.4f}")
            
            # 恢复可视化器状态（简化处理）
            self.visualizer.current_generation = self.current_generation
            
        else:
            print("  警告: 无法加载检查点，从头开始")
    
    def _generate_final_report(self):
        """生成最终报告"""
        print("\n生成最终进化报告...")
        
        # 获取进化摘要
        summary = self.visualizer.get_evolution_summary()
        
        # 生成静态仪表板
        dashboard_path = "data/evolution_logs/final_evolution_dashboard.png"
        self.dashboard.create_static_dashboard(dashboard_path, include_analysis=True)
        
        # 导出最佳个体
        if self.current_population:
            best_individual = self.current_population[0]
            export_path = "models/genomes/final_best_individual.json"
            self.checkpoint_manager.export_checkpoint(
                self.current_generation,
                export_path,
                include_visualizer_data=True
            )
        
        # 打印摘要报告
        print("\n" + "="*60)
        print("进化过程最终报告")
        print("="*60)
        print(f"总代数: {summary.get('current_generation', 0)}")
        print(f"最佳适应度: {summary.get('overall_best_fitness', 0):.6f}")
        print(f"进化改善率: {summary.get('evolution_progress', {}).get('improvement_rate', 0):.6f}/代")
        print(f"最终多样性: {summary.get('current_diversity', 0):.4f}")
        print(f"当前物种数: {summary.get('species_evolution', {}).get('current_species_count', 1)}")
        
        print("\n生成的文件:")
        print(f"  - 最终仪表板: {dashboard_path}")
        print(f"  - 最佳个体数据: {export_path}")
        print(f"  - 进化日志: data/evolution_logs/")
        print(f"  - 检查点: models/genomes/")
        
        # 保存摘要报告
        report_path = "data/evolution_logs/evolution_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  - 详细报告: {report_path}")
    
    def demonstrate_checkpoint_recovery(self):
        """演示检查点恢复功能"""
        print("\n" + "="*60)
        print("检查点恢复功能演示")
        print("="*60)
        
        # 先运行一些进化
        self._run_evolution_phase(15, "恢复演示_初始进化")
        
        # 保存一个特殊的检查点
        checkpoint_info = self.checkpoint_manager.save_checkpoint(
            self.current_population,
            self.current_fitness,
            self.current_generation,
            checkpoint_type="best",
            description="恢复演示专用检查点"
        )
        
        print(f"\n保存恢复检查点: Gen {self.current_generation}")
        
        # 清空当前状态
        self.current_population = []
        self.current_fitness = []
        self.current_generation = 0
        
        # 尝试恢复
        print("\n尝试恢复...")
        load_result = self.checkpoint_manager.load_checkpoint(
            generation=checkpoint_info['generation']
        )
        
        if load_result:
            print("恢复成功!")
            print(f"  恢复代数: {load_result['state']['generation']}")
            print(f"  恢复适应度: {load_result['state']['best_fitness']:.4f}")
            print(f"  继续进化建议: {load_result['resume_recommendations']}")
            
            # 使用恢复的状态继续进化
            self.current_population = load_result['state']['population']
            self.current_fitness = load_result['state']['fitness_scores']
            self.current_generation = load_result['state']['generation']
            
            # 运行几代继续验证
            self._run_evolution_phase(5, "恢复验证_继续进化")
            
        else:
            print("恢复失败!")
    
    def generate_dashboard_demo(self):
        """生成仪表板演示"""
        print("\n" + "="*60)
        print("仪表板功能演示")
        print("="*60)
        
        # 先运行一些进化生成数据
        self._run_evolution_phase(20, "仪表板演示_数据生成")
        
        # 生成静态仪表板
        dashboard_path = "data/evolution_logs/dashboard_demo.png"
        self.dashboard.create_static_dashboard(dashboard_path, include_analysis=True)
        
        # 获取仪表板状态
        status = self.dashboard.get_current_status()
        print(f"\n仪表板状态:")
        print(f"  激活状态: {status['dashboard_active']}")
        print(f"  数据点数: {status['data_points_available']}")
        print(f"  当前代数: {status['current_generation']}")
        print(f"  配置图表: {len(status['configured_plots'])} 个")
        
        print(f"\n静态仪表板已生成: {dashboard_path}")
        
        # 显示配置信息
        print("\n仪表板配置:")
        print(f"  - 显示适应度曲线: {'是' if self.dashboard.config['show_fitness_curve'] else '否'}")
        print(f"  - 显示多样性图: {'是' if self.dashboard.config['show_diversity_plot'] else '否'}")
        print(f"  - 显示3D轨迹: {'是' if self.dashboard.config['show_3d_trajectory'] else '否'}")
        print(f"  - 自动更新间隔: {self.dashboard.update_interval}秒")


def main():
    """主函数 - 运行完整的演示"""
    print("NeuroMinecraftGenesis - 进化可视化和断点续跑系统演示")
    print("="*80)
    
    # 创建演示实例
    demo = EvolutionDemo(
        population_size=50,
        genome_length=8,
        total_generations=40
    )
    
    try:
        # 选择演示模式
        print("\n请选择演示模式:")
        print("1. 完整进化过程演示（包括断点续跑）")
        print("2. 检查点恢复功能演示")
        print("3. 仪表板功能演示")
        print("4. 所有功能演示")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            demo.run_full_evolution()
        elif choice == "2":
            demo.demonstrate_checkpoint_recovery()
        elif choice == "3":
            demo.generate_dashboard_demo()
        elif choice == "4":
            demo.demonstrate_checkpoint_recovery()
            demo.generate_dashboard_demo()
            demo.run_full_evolution()
        else:
            print("无效选择，运行完整演示")
            demo.run_full_evolution()
            
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n演示结束")


if __name__ == "__main__":
    main()