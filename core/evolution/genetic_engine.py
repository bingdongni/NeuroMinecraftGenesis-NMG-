"""
遗传算法引擎 - 多目标优化的进化算法实现

本模块实现了基于DEAP框架的NSGA-II多目标遗传算法引擎，专门用于智能体学习规则的进化优化。
引擎支持智能体的并行评估、动态环境复杂度调整和实时性能监控。

核心特性:
- NSGA-II多目标优化算法实现
- 智能体学习规则进化（50维浮点向量）
- 并行化适应度评估（使用Ray库）
- 动态环境复杂度调整
- 完整的性能监控和统计
- 精英保留和多样性维护

算法参数:
- 种群大小: 16个智能体
- 学习规则维度: 50维浮点向量
- 进化代数: 50代
- 交叉率: 70%
- 变异率: 20%
- 选择算子: NSGA-II

作者: NeuroMinecraftGenesis Team
日期: 2025-11-13
"""

import numpy as np
import random
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime

# DEAP相关导入
from deap import creator, base, tools, algorithms
from deap.algorithms import varAnd

# 导入本地模块
from .nsga_ii import NSGA2Selector
from .population_manager import PopulationManager, Agent


class GeneticEngine:
    """
    遗传算法引擎
    
    基于NSGA-II的多目标优化遗传算法引擎，专门用于智能体学习规则的进化。
    支持并行评估、动态环境复杂度调整和完整的性能监控。
    
    主要功能:
    - 种群初始化和管理
    - 适应度评估（支持并行化）
    - NSGA-II选择、交叉、变异操作
    - 进化过程监控和统计
    - 结果输出和分析
    
    Attributes:
        population_manager: 种群管理器
        nsga2_selector: NSGA-II选择算子
        fitness_evaluator: 适应度评估函数
        toolbox: DEAP工具箱
        config: 算法配置参数
        statistics: 进化过程统计
        logger: 日志记录器
    """
    
    def __init__(self, population_size: int = 16, rule_dim: int = 50, 
                 crossover_rate: float = 0.7, mutation_rate: float = 0.2):
        """
        初始化遗传算法引擎
        
        Args:
            population_size: 种群大小，默认16
            rule_dim: 学习规则向量维度，默认50
            crossover_rate: 交叉率，默认0.7 (70%)
            mutation_rate: 变异率，默认0.2 (20%)
        """
        self.population_size = population_size
        self.rule_dim = rule_dim
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 初始化组件
        self.population_manager = PopulationManager(population_size, rule_dim)
        self.nsga2_selector = NSGA2Selector()
        self.fitness_evaluator = None  # 需要外部设置
        
        # 进化过程统计
        self.statistics = {
            'generation_history': [],
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': [],
            'convergence_history': [],
            'pareto_front_sizes': [],
            'evaluation_times': []
        }
        
        # 算法配置
        self.config = {
            'generations': 50,
            'elite_ratio': 0.25,
            'complexity_progression': True,
            'parallel_evaluation': True,
            'save_frequency': 10,
            'log_frequency': 5
        }
        
        # 初始化DEAP框架
        self._setup_deap()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_logging()
        
        self.logger.info(f"遗传算法引擎初始化完成 - 种群大小: {population_size}, "
                        f"规则维度: {rule_dim}, 交叉率: {crossover_rate}, 变异率: {mutation_rate}")
    
    def _setup_logging(self):
        """设置日志记录器"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_deap(self):
        """设置DEAP框架"""
        # 创建适应度类（多目标最大化）
        creator.create("FitnessMulti", tuple, weights=(1.0, 1.0, 1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
        
        # 创建基础工具箱
        self.toolbox = base.Toolbox()
        
        # 注册基因生成器（50维浮点向量，范围[-1, 1]）
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, self.rule_dim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传操作算子
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", self.nsga2_selector.select)
    
    def set_fitness_evaluator(self, evaluator_func: Callable):
        """
        设置适应度评估函数
        
        Args:
            evaluator_func: 接受学习规则参数，返回(游戏得分, 学习速度, 泛化能力)的函数
        """
        self.fitness_evaluator = evaluator_func
        self.logger.info("适应度评估函数已设置")
    
    def initialize_population(self) -> List:
        """
        初始化种群
        
        初始化16个智能体，每个智能体包含50维学习规则向量。
        使用多样化的初始化策略确保种群探索能力。
        
        Returns:
            List: 初始化后的DEAP个体列表
        """
        self.logger.info("开始初始化种群...")
        
        # 初始化种群管理器中的Agent
        agents = self.population_manager.initialize_population()
        
        # 转换为DEAP个体格式
        population = []
        for agent in agents:
            individual = creator.Individual(agent.learning_rule.copy())
            individual.agent_id = agent.id
            population.append(individual)
        
        self.logger.info(f"种群初始化完成，包含 {len(population)} 个个体")
        return population
    
    def evaluate_fitness(self, population: List) -> List[Tuple[float, float, float]]:
        """
        并行评估种群中每个个体的适应度
        
        使用Ray库在多个进程间分配评估任务，提高计算效率。
        适应度为三目标最大化：游戏得分、学习速度、泛化能力。
        
        Args:
            population: 待评估的种群
            
        Returns:
            List[Tuple[float, float, float]]: 每个个体的适应度元组列表
        """
        start_time = time.time()
        
        fitnesses = []
        
        if self.config['parallel_evaluation'] and self.fitness_evaluator:
            try:
                # 尝试使用并行评估
                fitnesses = self._parallel_fitness_evaluation(population)
            except Exception as e:
                self.logger.warning(f"并行评估失败，使用串行评估: {e}")
                fitnesses = self._sequential_fitness_evaluation(population)
        else:
            # 串行评估
            fitnesses = self._sequential_fitness_evaluation(population)
        
        evaluation_time = time.time() - start_time
        self.statistics['evaluation_times'].append(evaluation_time)
        
        self.logger.info(f"适应度评估完成 - 耗时: {evaluation_time:.2f}s, "
                        f"平均时间: {evaluation_time/len(population):.3f}s/个体")
        
        return fitnesses
    
    def _parallel_fitness_evaluation(self, population: List) -> List[Tuple[float, float, float]]:
        """并行适应度评估（使用Ray库）"""
        try:
            import ray
            
            # 远程评估函数
            @ray.remote
            def evaluate_single(individual):
                if self.fitness_evaluator:
                    return self.fitness_evaluator(individual.tolist())
                else:
                    return self._default_evaluation(individual.tolist())
            
            # 提交所有评估任务
            futures = [evaluate_single.remote(ind) for ind in population]
            
            # 收集结果
            fitnesses = ray.get(futures)
            
        except ImportError:
            raise Exception("Ray库未安装，无法进行并行评估")
        
        return fitnesses
    
    def _sequential_fitness_evaluation(self, population: List) -> List[Tuple[float, float, float]]:
        """串行适应度评估"""
        fitnesses = []
        
        for i, individual in enumerate(population):
            try:
                if self.fitness_evaluator:
                    fitness = self.fitness_evaluator(individual.tolist())
                else:
                    fitness = self._default_evaluation(individual.tolist())
                
                fitnesses.append(fitness)
                
                # 更新种群管理器中的Agent适应度
                agent_id = getattr(individual, 'agent_id', i)
                self.population_manager.update_agent_fitness(agent_id, fitness)
                
            except Exception as e:
                self.logger.error(f"评估个体 {i} 时出错: {e}")
                # 使用默认适应度
                fitness = self._default_evaluation(individual.tolist())
                fitnesses.append(fitness)
        
        return fitnesses
    
    def _evaluate_individual(self, individual) -> Tuple[float, float, float]:
        """DEAP评估函数接口"""
        if self.fitness_evaluator:
            return self.fitness_evaluator(individual.tolist())
        else:
            return self._default_evaluation(individual.tolist())
    
    def _default_evaluation(self, learning_rule: List[float]) -> Tuple[float, float, float]:
        """
        默认适应度评估函数
        
        当未提供自定义评估函数时使用的默认评估。
        模拟三个目标：游戏得分、学习速度、泛化能力。
        
        Args:
            learning_rule: 50维学习规则向量
            
        Returns:
            Tuple[float, float, float]: (游戏得分, 学习速度, 泛化能力)
        """
        # 模拟评估逻辑
        rule_array = np.array(learning_rule)
        
        # 游戏得分：基于参数的正负平衡和方差
        game_score = np.mean(rule_array) * 10 + np.std(rule_array) * 5 + np.random.normal(0, 1)
        
        # 学习速度：基于参数变化幅度
        learning_speed = np.std(rule_array) * 20 + np.abs(np.mean(rule_array)) * 10
        
        # 泛化能力：基于参数的分布均匀性
        generalization = 1.0 / (1.0 + np.std(rule_array)) * 15 + random.uniform(-2, 2)
        
        # 添加随机性模拟环境噪声
        game_score += random.uniform(-1, 1)
        learning_speed += random.uniform(-1, 1)
        generalization += random.uniform(-1, 1)
        
        # 确保非负值
        game_score = max(0, game_score)
        learning_speed = max(0, learning_speed)
        generalization = max(0, generalization)
        
        return (game_score, learning_speed, generalization)
    
    def evolve(self, generations: int = 50) -> Tuple[List, List[Tuple]]:
        """
        执行进化算法
        
        运行完整的遗传算法进化过程，包括选择、交叉、变异操作。
        支持动态环境复杂度调整和实时性能监控。
        
        Args:
            generations: 进化代数，默认50
            
        Returns:
            Tuple[List, List[Tuple]]: (最终种群, 适应度历史)
        """
        self.config['generations'] = generations
        
        self.logger.info(f"开始进化算法 - 代数: {generations}")
        start_time = time.time()
        
        # 初始化种群
        population = self.initialize_population()
        
        # 进化主循环
        for generation in range(generations):
            generation_start_time = time.time()
            
            # 更新环境复杂度
            self.population_manager.generation = generation
            if self.config['complexity_progression']:
                self.population_manager.update_complexity_level()
            
            # 评估适应度
            fitnesses = self.evaluate_fitness(population)
            
            # 更新统计信息
            self._update_statistics(population, fitnesses, generation)
            
            # 记录进度
            if generation % self.config['log_frequency'] == 0:
                self._log_generation_progress(generation, fitnesses, generation_start_time)
            
            # 保存中间结果
            if generation % self.config['save_frequency'] == 0:
                self._save_intermediate_results(generation, population, fitnesses)
            
            # 选择、交叉、变异
            if generation < generations - 1:  # 最后一代不需要产生后代
                offspring = self.varAnd(population, fitnesses)
                population = offspring
        
        # 进化完成，保存最终结果
        final_time = time.time() - start_time
        self.logger.info(f"进化算法完成 - 总耗时: {final_time:.2f}s")
        
        # 返回最终种群和统计信息
        return population, self.statistics
    
    def varAnd(self, population: List, fitnesses: List[Tuple[float, float, float]]) -> List:
        """
        NSGA-II风格的变体算法
        
        执行选择、交叉、变异操作，生成新的后代种群。
        使用NSGA-II选择算子维持多目标优化的帕累托最优解集。
        
        算法流程:
        1. 使用NSGA-II选择父代
        2. 对选择的个体进行交叉
        3. 对交叉结果进行变异
        4. 合并父代和后代，选择最优个体
        
        Args:
            population: 当前种群
            fitnesses: 当前种群适应度
            
        Returns:
            List: 新一代种群
        """
        # 选择父代（精英保留）
        elites = self.population_manager.get_elite_agents(self.config['elite_ratio'])
        elite_individuals = []
        
        for elite_agent in elites:
            # 找到对应的DEAP个体
            for ind in population:
                if getattr(ind, 'agent_id', -1) == elite_agent.id:
                    elite_individuals.append(ind)
                    break
        
        # 使用NSGA-II选择剩余个体
        remaining_size = self.population_size - len(elite_individuals)
        if remaining_size > 0:
            selected_parents = self.nsga2_selector.select(
                population, fitnesses, k=remaining_size
            )
        else:
            selected_parents = []
        
        # 合并精英和选择的个体作为父代
        parents = elite_individuals + selected_parents
        
        # 生成后代
        offspring = []
        
        # 交叉操作
        for _ in range(self.population_size // 2):
            if len(parents) >= 2:
                # 随机选择两个父代
                parent1, parent2 = random.sample(parents, 2)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = tools.cxBlend(parent1, parent2, alpha=0.5)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                offspring.extend([child1, child2])
        
        # 变异操作
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                mutated = tools.mutGaussian(offspring[i], mu=0, sigma=0.1, indpb=0.1)
                offspring[i] = mutated[0]
        
        # 确保种群大小
        while len(offspring) < self.population_size:
            offspring.append(random.choice(population).copy())
        
        # 限制种群大小
        offspring = offspring[:self.population_size]
        
        return offspring
    
    def _update_statistics(self, population: List, fitnesses: List[Tuple], generation: int):
        """更新进化统计信息"""
        # 计算平均适应度
        avg_fitness = np.mean([np.mean(f) for f in fitnesses])
        
        # 计算最佳适应度
        best_fitness = max([np.mean(f) for f in fitnesses])
        
        # 计算种群多样性
        diversity = self.population_manager.calculate_population_diversity()
        
        # 获取帕累托前沿大小
        pareto_fronts = self.nsga2_selector.get_pareto_fronts(population, fitnesses)
        pareto_front_size = len(pareto_fronts[0]) if pareto_fronts else 0
        
        # 更新统计记录
        self.statistics['generation_history'].append(generation)
        self.statistics['avg_fitness_history'].append(avg_fitness)
        self.statistics['best_fitness_history'].append(best_fitness)
        self.statistics['diversity_history'].append(diversity)
        self.statistics['pareto_front_sizes'].append(pareto_front_size)
        
        # 更新收敛指标
        if len(self.statistics['avg_fitness_history']) > 1:
            recent_improvement = (avg_fitness - 
                                self.statistics['avg_fitness_history'][-2])
            self.statistics['convergence_history'].append(recent_improvement)
        else:
            self.statistics['convergence_history'].append(0)
    
    def _log_generation_progress(self, generation: int, fitnesses: List, start_time: float):
        """记录代数进度"""
        elapsed_time = time.time() - start_time
        
        # 计算统计信息
        avg_performance = np.mean([np.mean(f) for f in fitnesses])
        best_performance = max([np.mean(f) for f in fitnesses])
        diversity = self.population_manager.calculate_population_diversity()
        
        self.logger.info(
            f"代数 {generation}: "
            f"平均性能={avg_performance:.4f}, "
            f"最佳性能={best_performance:.4f}, "
            f"多样性={diversity:.4f}, "
            f"复杂度={self.population_manager.get_complexity_level()}, "
            f"耗时={elapsed_time:.2f}s"
        )
        
        # 输出详细统计
        stats = self.population_manager.get_performance_statistics()
        self.logger.info(
            f"详细统计: 游戏得分={stats['performance_stats']['game_score']['mean']:.3f}±{stats['performance_stats']['game_score']['std']:.3f}, "
            f"学习速度={stats['performance_stats']['learning_speed']['mean']:.3f}±{stats['performance_stats']['learning_speed']['std']:.3f}, "
            f"泛化能力={stats['performance_stats']['generalization']['mean']:.3f}±{stats['performance_stats']['generalization']['std']:.3f}"
        )
    
    def _save_intermediate_results(self, generation: int, population: List, fitnesses: List):
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolution_generation_{generation}_{timestamp}.json"
        
        # 保存种群状态
        self.population_manager.save_population_state(filename)
        
        # 额外保存进化统计
        stats_data = {
            'generation': generation,
            'statistics': self.statistics,
            'population_fitness': fitnesses,
            'pareto_fronts': self.nsga2_selector.get_pareto_fronts(population, fitnesses),
            'timestamp': timestamp
        }
        
        with open(f"evolution_stats_{generation}_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    def get_best_solutions(self, count: int = 5) -> List[Dict]:
        """
        获取最优解
        
        Args:
            count: 返回解的数量
            
        Returns:
            List[Dict]: 包含学习规则和适应度的解列表
        """
        if not self.population_manager.agents:
            return []
        
        best_agents = self.population_manager.get_best_agents(count)
        
        solutions = []
        for agent in best_agents:
            solutions.append({
                'learning_rule': agent.learning_rule.tolist(),
                'fitness': agent.fitness,
                'avg_performance': agent.get_avg_performance(),
                'complexity_level': agent.complexity_level,
                'age': agent.age
            })
        
        return solutions
    
    def get_pareto_front_solutions(self) -> List[Dict]:
        """
        获取帕累托前沿解集
        
        Returns:
            List[Dict]: 帕累托前沿解集
        """
        if not self.population_manager.agents:
            return []
        
        # 获取当前种群
        population = self.initialize_population()  # 这里应该有实际保存的种群
        fitnesses = [agent.fitness for agent in self.population_manager.agents]
        
        # 获取帕累托前沿
        pareto_fronts = self.nsga2_selector.get_pareto_fronts(population, fitnesses)
        if not pareto_fronts:
            return []
        
        # 转换为解格式
        pareto_solutions = []
        for agent_idx in pareto_fronts[0]:
            agent = self.population_manager.agents[agent_idx]
            pareto_solutions.append({
                'learning_rule': agent.learning_rule.tolist(),
                'fitness': agent.fitness,
                'avg_performance': agent.get_avg_performance(),
                'complexity_level': agent.complexity_level
            })
        
        return pareto_solutions
    
    def export_results(self, filename: str):
        """
        导出完整的进化结果
        
        Args:
            filename: 导出文件名
        """
        results = {
            'config': self.config,
            'statistics': self.statistics,
            'best_solutions': self.get_best_solutions(),
            'pareto_front_solutions': self.get_pareto_front_solutions(),
            'final_stats': self.population_manager.get_performance_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"进化结果已导出到: {filename}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        运行结果分析
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        analysis = {
            'convergence_analysis': self._analyze_convergence(),
            'diversity_analysis': self._analyze_diversity(),
            'pareto_analysis': self._analyze_pareto_front(),
            'performance_analysis': self._analyze_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_convergence(self) -> Dict:
        """分析收敛性"""
        fitness_history = self.statistics['avg_fitness_history']
        
        if len(fitness_history) < 10:
            return {'status': 'insufficient_data'}
        
        # 计算收敛趋势
        recent_fitness = fitness_history[-10:]
        trend_slope = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
        
        # 计算改进率
        initial_fitness = np.mean(fitness_history[:10])
        final_fitness = np.mean(fitness_history[-10:])
        improvement_rate = (final_fitness - initial_fitness) / initial_fitness if initial_fitness > 0 else 0
        
        return {
            'trend_slope': trend_slope,
            'improvement_rate': improvement_rate,
            'is_converging': trend_slope < 0.01,
            'final_performance': final_fitness,
            'total_improvement': final_fitness - initial_fitness
        }
    
    def _analyze_diversity(self) -> Dict:
        """分析多样性"""
        diversity_history = self.statistics['diversity_history']
        
        return {
            'initial_diversity': diversity_history[0] if diversity_history else 0,
            'final_diversity': diversity_history[-1] if diversity_history else 0,
            'diversity_trend': 'stable' if len(diversity_history) > 1 else 'unknown',
            'avg_diversity': np.mean(diversity_history) if diversity_history else 0
        }
    
    def _analyze_pareto_front(self) -> Dict:
        """分析帕累托前沿"""
        pareto_sizes = self.statistics['pareto_front_sizes']
        
        return {
            'final_pareto_size': pareto_sizes[-1] if pareto_sizes else 0,
            'avg_pareto_size': np.mean(pareto_sizes) if pareto_sizes else 0,
            'pareto_trend': 'stable' if len(set(pareto_sizes[-5:])) <= 2 else 'evolving'
        }
    
    def _analyze_performance(self) -> Dict:
        """分析性能"""
        best_fitness_history = self.statistics['best_fitness_history']
        avg_fitness_history = self.statistics['avg_fitness_history']
        
        return {
            'best_final_performance': max(best_fitness_history) if best_fitness_history else 0,
            'avg_final_performance': avg_fitness_history[-1] if avg_fitness_history else 0,
            'performance_gap': (max(best_fitness_history) - avg_fitness_history[-1]) if best_fitness_history and avg_fitness_history else 0,
            'consistency': 1.0 - (np.std(avg_fitness_history[-10:]) / np.mean(avg_fitness_history[-10:])) if avg_fitness_history and len(avg_fitness_history) >= 10 else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于收敛性分析
        if 'convergence_analysis' in self.statistics:
            conv = self.statistics['convergence_analysis']
            if conv.get('is_converging', False):
                recommendations.append("算法已收敛，可以考虑增加环境复杂度或调整参数")
            if conv.get('improvement_rate', 0) < 0.01:
                recommendations.append("性能改进缓慢，建议调整交叉率和变异率")
        
        # 基于多样性分析
        if 'diversity_analysis' in self.statistics:
            div = self.statistics['diversity_analysis']
            if div.get('final_diversity', 0) < 0.3:
                recommendations.append("种群多样性较低，建议增加变异率或调整选择压力")
        
        # 基于性能分析
        if 'performance_analysis' in self.statistics:
            perf = self.statistics['performance_analysis']
            if perf.get('consistency', 0) < 0.7:
                recommendations.append("性能不够稳定，建议增加种群大小或优化评估函数")
        
        if not recommendations:
            recommendations.append("算法运行良好，保持当前参数设置")
        
        return recommendations


# 创建全局实例（单例模式）
_genetic_engine_instance = None

def get_genetic_engine(**kwargs) -> GeneticEngine:
    """
    获取遗传算法引擎实例（单例模式）
    
    Args:
        **kwargs: 引擎初始化参数
        
    Returns:
        GeneticEngine: 遗传算法引擎实例
    """
    global _genetic_engine_instance
    
    if _genetic_engine_instance is None:
        _genetic_engine_instance = GeneticEngine(**kwargs)
    
    return _genetic_engine_instance


def create_genetic_engine(**kwargs) -> GeneticEngine:
    """
    创建新的遗传算法引擎实例
    
    Args:
        **kwargs: 引擎初始化参数
        
    Returns:
        GeneticEngine: 新的遗传算法引擎实例
    """
    return GeneticEngine(**kwargs)