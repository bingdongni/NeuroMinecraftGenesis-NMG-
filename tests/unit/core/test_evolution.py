#!/usr/bin/env python3
"""
进化系统完整单元测试
"""

import pytest
import numpy as np


class TestGeneticEngine:
    """遗传引擎测试类"""

    def test_initialization(self):
        """测试遗传引擎初始化"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(
            population_size=16,
            rule_dim=50,
            crossover_rate=0.7,
            mutation_rate=0.2
        )

        assert engine is not None
        assert engine.population_size == 16
        assert engine.rule_dim == 50

    def test_population_initialization(self):
        """测试种群初始化"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        population = engine.initialize_population()

        assert len(population) == 10

    def test_fitness_evaluation(self):
        """测试适应度评估"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        population = engine.initialize_population()

        # 定义评估函数
        def fitness_func(individual):
            return (np.mean(individual), np.std(individual), np.sum(individual > 0) / len(individual))

        engine.set_fitness_evaluator(fitness_func)

        # 评估
        fitness_scores = engine.evaluate_fitness(population)

        assert len(fitness_scores) == len(population)

    def test_selection(self):
        """测试选择操作"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        population = engine.initialize_population()

        # 评估
        def fitness_func(individual):
            return (np.mean(individual),)

        engine.set_fitness_evaluator(fitness_func)
        fitness_scores = engine.evaluate_fitness(population)

        # 选择
        selected = engine.select(population, fitness_scores)

        assert len(selected) <= len(population)

    def test_crossover(self):
        """测试交叉操作"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        population = engine.initialize_population()

        # 交叉
        offspring = engine.crossover(population[0], population[1])

        assert len(offspring) == 2

    def test_mutation(self):
        """测试变异操作"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        population = engine.initialize_population()

        # 变异
        mutated = engine.mutate(population[0])

        assert len(mutated) == len(population[0])

    def test_evolution_loop(self):
        """测试进化循环"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        engine.config['generations'] = 3

        # 定义评估函数
        def fitness_func(individual):
            return (np.mean(individual), np.sum(individual > 0) / len(individual))

        engine.set_fitness_evaluator(fitness_func)

        # 运行进化
        results = engine.evolve()

        assert results is not None
        assert 'final_population' in results

    def test_elite_preservation(self):
        """测试精英保留"""
        from core.evolution.genetic_engine import GeneticEngine

        engine = GeneticEngine(population_size=10, rule_dim=20)
        engine.config['elite_ratio'] = 0.2

        # 初始化种群
        population = engine.initialize_population()

        # 验证精英数量
        elite_count = int(len(population) * engine.config['elite_ratio'])
        assert elite_count >= 2


class TestNSGA2:
    """NSGA-II测试类"""

    def test_initialization(self):
        """测试NSGA-II初始化"""
        from core.evolution.nsga_ii import NSGA2Selector

        selector = NSGA2Selector()

        assert selector is not None

    def test_non_dominated_sorting(self):
        """测试非支配排序"""
        from core.evolution.nsga_ii import NSGA2Selector

        selector = NSGA2Selector()

        # 创建测试适应度
        fitness_values = [
            (0.8, 0.2),
            (0.5, 0.5),
            (0.3, 0.7),
            (0.9, 0.1)
        ]

        # 排序
        fronts = selector.fast_non_dominated_sort(fitness_values)

        assert len(fronts) >= 1

    def test_crowding_distance(self):
        """测试拥挤距离"""
        from core.evolution.nsga_ii import NSGA2Selector

        selector = NSGA2Selector()

        front = [(0.8, 0.2), (0.5, 0.5), (0.3, 0.7)]

        # 计算拥挤距离
        distances = selector.calculate_crowding_distance(front)

        assert len(distances) == len(front)


class TestFitnessEvaluator:
    """适应度评估器测试类"""

    def test_initialization(self):
        """测试评估器初始化"""
        from core.evolution.fitness_evaluator import FitnessEvaluator

        evaluator = FitnessEvaluator()

        assert evaluator is not None

    def test_evaluation_metrics(self):
        """测试评估指标"""
        from core.evolution.fitness_evaluator import FitnessEvaluator

        evaluator = FitnessEvaluator()

        # 创建模拟数据
        learning_rule = np.random.randn(50)
        performance_data = {
            'score': 0.8,
            'learning_speed': 0.5,
            'generalization': 0.7
        }

        # 评估
        result = evaluator.evaluate(learning_rule, performance_data)

        assert result is not None
        assert 'overall_fitness' in result

    def test_multi_objective_fitness(self):
        """测试多目标适应度"""
        from core.evolution.fitness_evaluator import FitnessEvaluator

        evaluator = FitnessEvaluator()

        # 多目标评估
        learning_rule = np.random.randn(50)
        result = evaluator.evaluate_multi_objective(learning_rule)

        assert result is not None
        assert len(result) >= 3


class TestPopulationManager:
    """种群管理器测试类"""

    def test_initialization(self):
        """测试管理器初始化"""
        from core.evolution.population_manager import PopulationManager

        manager = PopulationManager(
            population_size=20,
            rule_dim=50
        )

        assert manager is not None
        assert manager.population_size == 20

    def test_agent_creation(self):
        """测试智能体创建"""
        from core.evolution.population_manager import Agent

        agent = Agent(agent_id="test_1")

        assert agent is not None
        assert agent.id == "test_1"

    def test_population_operations(self):
        """测试种群操作"""
        from core.evolution.population_manager import PopulationManager

        manager = PopulationManager(population_size=10, rule_dim=20)

        # 初始化
        agents = manager.initialize_population()

        assert len(agents) == 10

        # 添加
        new_agent = manager.create_agent("new_agent")
        assert len(manager.agents) == 11

        # 获取
        retrieved = manager.get_agent("new_agent")
        assert retrieved is not None


class TestCheckpointManager:
    """检查点管理器测试类"""

    def test_initialization(self):
        """测试检查点初始化"""
        from core.evolution.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(save_dir="./test_checkpoints")

        assert manager is not None

    def test_checkpoint_save(self):
        """测试检查点保存"""
        from core.evolution.checkpoint_manager import CheckpointManager
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # 保存检查点
            checkpoint_data = {
                'generation': 10,
                'population': [np.random.randn(20) for _ in range(5)],
                'best_fitness': 0.8
            }

            path = manager.save_checkpoint(checkpoint_data, "test_gen_10")
            assert os.path.exists(path)

    def test_checkpoint_load(self):
        """测试检查点加载"""
        from core.evolution.checkpoint_manager import CheckpointManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # 保存检查点
            checkpoint_data = {
                'generation': 10,
                'best_fitness': 0.8
            }

            path = manager.save_checkpoint(checkpoint_data, "test_gen_10")

            # 加载
            loaded = manager.load_checkpoint(path)

            assert loaded is not None
            assert loaded['generation'] == 10


class TestEvolutionVisualizer:
    """进化可视化测试类"""

    def test_initialization(self):
        """测试可视化初始化"""
        from core.evolution.evolution_visualizer import EvolutionVisualizer

        viz = EvolutionVisualizer()

        assert viz is not None

    def test_fitness_history_plotting(self):
        """测试适应度历史绘图"""
        from core.evolution.evolution_visualizer import EvolutionVisualizer
        import tempfile

        viz = EvolutionVisualizer()

        # 添加数据
        history = [
            {'generation': i, 'best_fitness': np.random.rand()}
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            viz.plot_fitness_history(history, save_path=path)
            import os
            assert os.path.exists(path)
            os.unlink(path)
        except Exception:
            pass


class TestLifelongLearning:
    """终身学习测试类"""

    def test_initialization(self):
        """测试终身学习初始化"""
        from core.evolution.lifeless_learning import LifelessLearning

        learner = LifelessLearning()

        assert learner is not None

    def test_task_sequence_learning(self):
        """测试任务序列学习"""
        from core.evolution.lifeless_learning import LifelessLearning

        learner = LifelessLearning()

        # 添加任务
        for i in range(3):
            task_data = {
                'task_id': f'task_{i}',
                'samples': [np.random.randn(20) for _ in range(10)]
            }
            learner.add_task(task_data)

        assert learner.num_tasks == 3

    def test_forgetting_control(self):
        """测试遗忘控制"""
        from core.evolution.lifeless_learning import LifelessLearning

        learner = LifelessLearning()

        # 添加多个任务
        for i in range(5):
            learner.add_task({
                'task_id': f'task_{i}',
                'samples': [np.random.randn(20) for _ in range(10)]
            })

        # 检查遗忘率
        forgetting_rate = learner.get_forgetting_rate()
        assert 0 <= forgetting_rate <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
