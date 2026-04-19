#!/usr/bin/env python3
"""
算法发现模块测试
测试DiscoRL核心和算法发现功能
"""

import pytest
import numpy as np
import torch


class TestDiscoRLCore:
    """DiscoRL核心测试"""

    def test_disco_rl_initialization(self):
        """测试DiscoRL初始化"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=100,
            state_dim=64,
            action_dim=8
        )

        assert core is not None
        assert core.n_agents == 100

    def test_algorithm_discovery(self):
        """测试算法发现"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=50,
            state_dim=32,
            action_dim=4
        )

        result = core.run_discovery_cycle(iterations=5)

        assert result is not None
        assert "best_algorithm" in result or "population" in result

    def test_population_evaluation(self):
        """测试种群评估"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=20,
            state_dim=16,
            action_dim=4
        )

        population = core.initialize_population()
        assert len(population) > 0

        fitness_scores = core.evaluate_population(population)
        assert len(fitness_scores) == len(population)

    def test_genetic_operations(self):
        """测试遗传操作"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=10,
            state_dim=8,
            action_dim=4
        )

        parent1 = np.random.randn(10)
        parent2 = np.random.randn(10)

        # 测试交叉
        child = core.crossover(parent1, parent2)
        assert len(child) == len(parent1)

        # 测试变异
        mutated = core.mutate(child, mutation_rate=0.1)
        assert len(mutated) == len(child)

    def test_selection(self):
        """测试选择操作"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=10,
            state_dim=8,
            action_dim=4
        )

        population = core.initialize_population()
        fitness_scores = np.random.rand(len(population))

        selected = core.selection(population, fitness_scores, n_select=5)
        assert len(selected) == 5

    def test_algorithm_encoding(self):
        """测试算法编码"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=10,
            state_dim=8,
            action_dim=4
        )

        algorithm = core.encode_algorithm()
        assert algorithm is not None

        decoded = core.decode_algorithm(algorithm)
        assert decoded is not None

    def test_reward_calculation(self):
        """测试奖励计算"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=10,
            state_dim=8,
            action_dim=4
        )

        performance_metrics = {
            "task_success": 0.8,
            "efficiency": 0.7,
            "generalization": 0.6
        }

        reward = core.calculate_reward(performance_metrics)
        assert reward >= 0

    def test_convergence_check(self):
        """测试收敛检查"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore

        core = DiscoRLCore(
            n_agents=50,
            state_dim=32,
            action_dim=8
        )

        # 运行几代
        history = []
        for _ in range(10):
            result = core.run_discovery_cycle(iterations=1)
            if "best_fitness" in result:
                history.append(result["best_fitness"])

        converged = core.check_convergence(history)
        assert isinstance(converged, bool)

    def test_save_load_algorithms(self):
        """测试算法保存加载"""
        from core.algorithm_discovery.disco_rl_core import DiscoRLCore
        import tempfile
        import os

        core = DiscoRLCore(
            n_agents=20,
            state_dim=16,
            action_dim=4
        )

        # 运行发现
        core.run_discovery_cycle(iterations=5)

        # 保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as f:
            temp_path = f.name

        try:
            core.save_best_algorithm(temp_path)

            # 加载
            loaded = core.load_algorithm(temp_path)
            assert loaded is not None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestAlgorithmSpace:
    """算法空间测试"""

    def test_action_space_definition(self):
        """测试动作空间定义"""
        from core.algorithm_discovery.disco_rl_core import ActionSpace

        space = ActionSpace()

        # 定义基本操作
        space.add_operation("gradient_descent", {
            "type": "optimizer",
            "params": {"lr": 0.01}
        })

        space.add_operation("momentum", {
            "type": "optimizer",
            "params": {"momentum": 0.9}
        })

        assert len(space.operations) >= 2

    def test_hyperparameter_sampling(self):
        """测试超参数采样"""
        from core.algorithm_discovery.disco_rl_core import HyperParameterSpace

        space = HyperParameterSpace()

        space.add_parameter("learning_rate", "float", min=0.0001, max=0.1)
        space.add_parameter("batch_size", "int", min=16, max=256)
        space.add_parameter("optimizer", "choice", options=["adam", "sgd", "rmsprop"])

        sample = space.sample()
        assert "learning_rate" in sample
        assert "batch_size" in sample
        assert "optimizer" in sample


class TestEvolutionaryDynamics:
    """进化动态测试"""

    def test_fitness_landscape(self):
        """测试适应度景观"""
        from core.algorithm_discovery.disco_rl_core import FitnessLandscape

        landscape = FitnessLandscape(dimensions=2, resolution=10)

        # 评估点
        point = np.array([0.5, 0.5])
        fitness = landscape.evaluate(point)

        assert isinstance(fitness, (int, float))

    def test_population_diversity(self):
        """测试种群多样性"""
        from core.algorithm_discovery.disco_rl_core import DiversityMetric

        metric = DiversityMetric()

        population = [np.random.randn(10) for _ in range(20)]
        diversity = metric.calculate(population)

        assert 0 <= diversity <= 1

    def test_selection_pressure(self):
        """测试选择压力"""
        from core.algorithm_discovery.disco_rl_core import SelectionPressure

        pressure = SelectionPressure()

        fitness_scores = np.random.rand(50)

        selected = pressure.select(fitness_scores, n=10, pressure=2.0)
        assert len(selected) == 10


class TestGeneralizationTesting:
    """泛化测试"""

    def test_cross_environment_transfer(self):
        """测试跨环境迁移"""
        from core.algorithm_discovery.disco_rl_core import GeneralizationTester

        tester = GeneralizationTester()

        source_performance = 0.8
        target_performance = tester.test_transfer(source_performance, "new_env")

        assert isinstance(target_performance, (int, float))

    def test_generalization_score(self):
        """测试泛化评分"""
        from core.algorithm_discovery.disco_rl_core import GeneralizationScore

        score = GeneralizationScore()

        # 测试多个环境
        performances = [0.7, 0.8, 0.75, 0.6]
        gen_score = score.calculate(performances)

        assert 0 <= gen_score <= 1

    def test_adaptation_rate(self):
        """测试适应率"""
        from core.algorithm_discovery.disco_rl_core import AdaptationRate

        rate = AdaptationRate()

        history = [0.5, 0.6, 0.65, 0.7, 0.75]
        adaptation = rate.calculate(history)

        assert isinstance(adaptation, (int, float))


class TestDiscoRLAgent:
    """DiscoRL智能体测试"""

    def test_agent_initialization(self):
        """测试智能体初始化"""
        from agents.evolution.disco_rl_agent import DiscoRLAgent

        agent = DiscoRLAgent(
            agent_id="test_001",
            state_dim=32,
            action_dim=4
        )

        assert agent.agent_id == "test_001"
        assert agent.state_dim == 32

    def test_algorithm_execution(self):
        """测试算法执行"""
        from agents.evolution.disco_rl_agent import DiscoRLAgent

        agent = DiscoRLAgent(
            agent_id="test_001",
            state_dim=16,
            action_dim=4
        )

        state = np.random.randn(16)
        action = agent.execute_algorithm(state)

        assert action is not None

    def test_performance_tracking(self):
        """测试性能跟踪"""
        from agents.evolution.disco_rl_agent import DiscoRLAgent

        agent = DiscoRLAgent(
            agent_id="test_001",
            state_dim=8,
            action_dim=4
        )

        # 记录性能
        for i in range(10):
            agent.record_performance(score=np.random.rand())

        history = agent.get_performance_history()
        assert len(history) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
