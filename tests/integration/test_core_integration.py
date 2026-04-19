#!/usr/bin/env python3
"""
核心模块集成测试
测试各核心模块之间的协作
"""

import pytest
import numpy as np
import torch
import asyncio


class TestBrainModuleIntegration:
    """大脑模块集成测试"""

    @pytest.mark.asyncio
    async def test_perception_to_cognition(self):
        """测试从感知到认知的流程"""
        from core.brain.perception_module import PerceptionModule
        from core.brain.prefrontal_cortex import PrefrontalCortex

        # 初始化模块
        perception = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=128)
        cognition = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)

        # 模拟图像输入
        image = torch.randn(2, 3, 224, 224)
        visual_features = perception.process_visual(image)

        # 验证特征形状
        assert visual_features.shape[0] == 2

    @pytest.mark.asyncio
    async def test_memory_to_reasoning(self):
        """测试从记忆到推理的流程"""
        from core.brain.hippocampus import Hippocampus
        from core.brain.prefrontal_cortex import PrefrontalCortex

        # 初始化模块
        memory = Hippocampus(max_capacity=100, embedding_dim=128)
        reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)

        # 存储记忆
        memory_item = {
            'content': 'Test memory',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        memory.store_episodic(memory_item)

        # 检索并推理
        query = np.random.randn(128)
        retrieved = memory.retrieve(query, top_k=3)

        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_attention_to_memory(self):
        """测试从注意力到记忆的流程"""
        from core.brain.thalamic_gate import ThalamicGate
        from core.brain.hippocampus import Hippocampus

        # 初始化模块
        attention = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)
        memory = Hippocampus(max_capacity=100, embedding_dim=128)

        # 处理注意力
        features = torch.randn(4, 10, 64)
        attended = attention.compute_attention(features)

        # 将注意力结果存储到记忆
        memory_item = {
            'content': 'Attended features',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        memory.store_episodic(memory_item)

        assert len(memory.episodic_memory) > 0

    @pytest.mark.asyncio
    async def test_imagination_to_planning(self):
        """测试从想象到规划的流程"""
        from core.brain.imagination_engine import ImaginationEngine
        from core.brain.prefrontal_cortex import PrefrontalCortex

        # 初始化模块
        imagination = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)
        planning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)

        # 生成想象
        current_state = np.random.randn(64).astype(np.float32)
        prediction = imagination.predict_future_state(current_state, steps=5)

        # 验证预测
        assert 'predicted_states' in prediction
        assert len(prediction['predicted_states']) == 5

    @pytest.mark.asyncio
    async def test_full_cognitive_pipeline(self):
        """测试完整认知流程"""
        from core.brain.perception_module import PerceptionModule
        from core.brain.thalamic_gate import ThalamicGate
        from core.brain.hippocampus import Hippocampus
        from core.brain.imagination_engine import ImaginationEngine
        from core.brain.prefrontal_cortex import PrefrontalCortex

        # 初始化所有模块
        perception = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=128)
        attention = ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8)
        memory = Hippocampus(max_capacity=1000, embedding_dim=128)
        imagination = ImaginationEngine(state_dim=128, hidden_dim=256, spatial_dim=3)
        reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=3)

        # 1. 感知阶段
        image = torch.randn(1, 3, 224, 224)
        features = perception.process_visual(image)

        # 2. 注意力阶段
        attended = attention.compute_attention(features.unsqueeze(1))

        # 3. 记忆阶段
        memory_item = {
            'content': 'Integrated experience',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        memory.store_episodic(memory_item)

        # 4. 想象阶段
        state = features.squeeze().numpy()
        prediction = imagination.predict_future_state(state, steps=3)

        # 验证流程
        assert prediction is not None


class TestQuantumBrainIntegration:
    """量子类脑集成测试"""

    @pytest.mark.asyncio
    async def test_quantum_neural_fusion(self):
        """测试量子神经融合"""
        from core.quantum_brain.fusion_system import QuantumBrainFusion
        from core.brain.thalamic_gate import ThalamicGate

        # 初始化模块
        fusion = QuantumBrainFusion(n_neurons=1000, n_qubits=5)
        attention = ThalamicGate(input_dim=8, hidden_dim=128, num_attention_heads=4)

        # 初始化融合系统
        fusion.initialize_system()

        # 处理输入
        input_signal = np.random.randn(8)
        output = fusion.process_input(input_signal)

        # 获取神经激活
        activations = fusion.get_neural_activations()

        assert activations is not None
        assert len(activations) == 1000

    @pytest.mark.asyncio
    async def test_quantum_decision_integration(self):
        """测试量子决策集成"""
        from core.quantum_brain.fusion_system import QuantumBrainFusion
        from core.brain.prefrontal_cortex import PrefrontalCortex

        # 初始化模块
        fusion = QuantumBrainFusion(n_neurons=500, n_qubits=4)
        reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=3)

        fusion.initialize_system()

        # 做出决策
        decision_input = np.random.randn(8)
        quantum_decision, confidence = fusion.make_fusion_decision(decision_input)

        assert quantum_decision in [0, 1]
        assert 0 <= confidence <= 1


class TestEvolutionCognitionIntegration:
    """进化认知集成测试"""

    def test_evolution_to_cognition(self):
        """测试从进化到认知的流程"""
        from core.evolution.genetic_engine import GeneticEngine
        from core.brain.hippocampus import Hippocampus

        # 初始化模块
        evolution = GeneticEngine(population_size=10, rule_dim=20)
        memory = Hippocampus(max_capacity=100, embedding_dim=64)

        # 运行进化
        def fitness_func(individual):
            return (np.mean(individual), np.sum(individual > 0) / len(individual))

        evolution.set_fitness_evaluator(fitness_func)
        population = evolution.initialize_population()

        # 评估
        fitness_scores = evolution.evaluate_fitness(population)

        # 将最佳个体的规则存储到记忆
        best_idx = np.argmax([f[0] for f in fitness_scores])
        best_individual = population[best_idx]

        memory_item = {
            'content': 'Evolved learning rules',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': best_individual
        }
        memory.store_episodic(memory_item)

        assert len(memory.episodic_memory) > 0


class TestAgentWorldIntegration:
    """智能体世界集成测试"""

    @pytest.mark.asyncio
    async def test_agent_perception_action(self):
        """测试智能体感知-动作循环"""
        from agents.single.action_executor import ActionExecutor, ActionType
        from core.brain.perception_module import PerceptionModule

        # 初始化模块
        executor = ActionExecutor()
        perception = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟感知
        image = torch.randn(1, 3, 224, 224)
        features = perception.process_visual(image)

        # 根据感知执行动作
        result = await executor.execute_action(ActionType.MOVE_FORWARD, distance=3.0)

        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_learning_loop(self):
        """测试智能体学习循环"""
        from agents.single.intelligent_agent_system import IntelligentAgentSystem
        from core.brain.hippocampus import Hippocampus

        # 初始化模块
        agent = IntelligentAgentSystem()
        memory = Hippocampus(max_capacity=100, embedding_dim=128)

        await agent.start_system()

        # 执行动作
        await agent.execute_atom_actions()

        # 存储学习经验
        memory_item = {
            'content': 'Agent learning experience',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        memory.store_episodic(memory_item)

        await agent.stop_system()

        assert len(memory.episodic_memory) > 0


class TestMultiAgentIntegration:
    """多智能体集成测试"""

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """测试多智能体协作"""
        from agents.multi.tribal_society import TribalSociety
        from agents.multi.collective_memory import CollectiveMemory

        # 初始化系统
        society = TribalSociety(num_agents=3)
        society.create_agents()

        # 初始化集体记忆
        memory = CollectiveMemory()

        # 创建多个智能体的知识
        for i in range(3):
            memory.store(agent_id=f"agent_{i}", content=f"Knowledge from agent {i}")

        # 验证集体记忆
        assert len(memory.memories) == 3

    @pytest.mark.asyncio
    async def test_multi_agent_evolution(self):
        """测试多智能体进化"""
        from agents.mass_evolution.multi_agent_society import MultiAgentSociety
        from core.evolution.fitness_evaluator import FitnessEvaluator

        # 初始化系统
        society = MultiAgentSociety(num_agents=20)
        evaluator = FitnessEvaluator()

        # 运行进化
        society.evolve(generations=2)

        # 获取历史
        history = society.get_evolution_history()

        assert len(history) > 0


class TestCrossWorldIntegration:
    """跨世界集成测试"""

    def test_real_to_virtual_mapping(self):
        """测试真实到虚拟世界的映射"""
        from worlds.real.cross_domain_learner import CrossDomainLearner
        from worlds.procgen.world_generator import WorldGenerator

        # 初始化模块
        learner = CrossDomainLearner()
        world_gen = WorldGenerator(world_size=(64, 64), seed=42)

        # 生成虚拟世界
        terrain = world_gen.generate_terrain()

        # 映射真实特征到虚拟世界
        real_features = np.random.randn(64)
        virtual_features = learner.map_to_virtual(real_features)

        assert virtual_features is not None

    def test_knowledge_transfer_across_domains(self):
        """测试跨域知识迁移"""
        from worlds.real.cross_domain_learner import CrossDomainLearner
        from worlds.real.strategy_transfer import StrategyTransfer

        # 初始化模块
        learner = CrossDomainLearner()
        transfer = StrategyTransfer()

        # 创建迁移数据
        source_data = np.random.randn(50, 64)
        target_data = np.random.randn(50, 64)

        learner.train_transfer(source_data, target_data)

        # 提取并应用策略
        demonstrations = [np.random.randn(20, 10) for _ in range(5)]
        strategy = transfer.extract_strategy(demonstrations)

        assert strategy is not None


class TestSystemIntegration:
    """系统级集成测试"""

    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """测试完整系统初始化"""
        from core.brain.perception_module import PerceptionModule
        from core.brain.hippocampus import Hippocampus
        from core.brain.prefrontal_cortex import PrefrontalCortex
        from core.quantum_brain.fusion_system import QuantumBrainFusion
        from agents.single.intelligent_agent_system import IntelligentAgentSystem

        # 初始化所有组件
        modules = {
            'perception': PerceptionModule(input_dim=128, hidden_dim=256, output_dim=128),
            'memory': Hippocampus(max_capacity=100, embedding_dim=128),
            'reasoning': PrefrontalCortex(llm_mode='local', max_reasoning_steps=3),
            'quantum': QuantumBrainFusion(n_neurons=500, n_qubits=4),
            'agent': IntelligentAgentSystem()
        }

        # 初始化量子系统
        modules['quantum'].initialize_system()

        # 启动智能体
        await modules['agent'].start_system()

        # 验证所有组件
        assert modules['perception'] is not None
        assert modules['memory'] is not None
        assert modules['quantum'].is_initialized()
        assert modules['agent'].system_started

        # 清理
        modules['quantum'].shutdown()
        await modules['agent'].stop_system()

    def test_evolution_cognition_action_integration(self):
        """测试进化-认知-动作集成"""
        from core.evolution.genetic_engine import GeneticEngine
        from core.brain.hippocampus import Hippocampus
        from core.brain.prefrontal_cortex import PrefrontalCortex
        from agents.single.action_executor import ActionExecutor, ActionType

        # 初始化模块
        evolution = GeneticEngine(population_size=10, rule_dim=20)
        memory = Hippocampus(max_capacity=100, embedding_dim=64)
        reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=3)
        executor = ActionExecutor()

        # 运行进化获取最优策略
        def fitness_func(individual):
            return (np.mean(individual),)

        evolution.set_fitness_evaluator(fitness_func)
        population = evolution.initialize_population()
        fitness_scores = evolution.evaluate_fitness(population)

        # 选择最佳个体
        best_idx = np.argmax([f[0] for f in fitness_scores])
        best_rules = population[best_idx]

        # 存储到记忆
        memory.store_episodic({
            'content': 'Evolved strategy',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': best_rules,
            'context': np.random.randn(20)
        })

        assert len(memory.episodic_memory) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
