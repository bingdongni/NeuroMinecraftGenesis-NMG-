#!/usr/bin/env python3
"""
NeuroMinecraft Genesis - 完整示例代码集
包含所有核心模块的使用示例
"""

import numpy as np
import torch
import asyncio
import time


def example_1_hello_world():
    """
    示例1: Hello World - 基础系统初始化
    """
    print("=" * 60)
    print("示例1: Hello World - 基础系统初始化")
    print("=" * 60)

    from core.brain.hippocampus import Hippocampus
    from core.brain.prefrontal_cortex import PrefrontalCortex

    # 初始化海马体
    memory = Hippocampus(max_capacity=1000, embedding_dim=128)
    print(f"✅ 海马体初始化完成: 容量={memory.max_capacity}, 嵌入维度={memory.embedding_dim}")

    # 初始化前额叶
    cortex = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)
    print(f"✅ 前额叶初始化完成: LLM模式={cortex.llm_mode.value}")

    return memory, cortex


def example_2_memory_system():
    """
    示例2: 记忆系统使用
    """
    print("\n" + "=" * 60)
    print("示例2: 记忆系统使用")
    print("=" * 60)

    from core.brain.hippocampus import Hippocampus

    # 初始化
    memory = Hippocampus(max_capacity=100, embedding_dim=64)

    # 存储情景记忆
    for i in range(5):
        memory_data = {
            'content': f'记忆事件 {i}',
            'timestamp': time.time(),
            'emotion': np.random.randn(5),  # 情感数据
            'sensory_data': np.random.randn(10),  # 感觉数据
            'context': np.random.randn(20)  # 上下文
        }
        key = memory.store_episodic(memory_data)
        print(f"  存储记忆 {i+1}: {key}")

    # 存储语义记忆
    semantic_data = {
        'concept': '人工智能',
        'definition': '使机器具有人类智能的科学',
        'features': np.random.randn(10)
    }
    memory.store_semantic('AI', semantic_data)
    print("  ✅ 语义记忆存储完成")

    # 检索记忆
    query = np.random.randn(64)
    results = memory.retrieve(query, top_k=3)
    print(f"  检索结果: {len(results)} 条记忆")

    # 获取性能指标
    metrics = memory.get_performance_metrics()
    print(f"  性能指标: 存储={metrics['storage_count']}, 检索={metrics['retrieval_count']}")


async def example_3_reasoning_system():
    """
    示例3: 推理系统使用
    """
    print("\n" + "=" * 60)
    print("示例3: 推理系统使用")
    print("=" * 60)

    from core.brain.prefrontal_cortex import PrefrontalCortex, LLMMode

    # 初始化推理引擎
    cortex = PrefrontalCortex(llm_mode=LLMMode.LOCAL, max_reasoning_steps=10)

    # 测试推理问题
    problems = [
        "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？",
        "今天天气很好，我应该出去散步还是在家工作？"
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n  问题 {i}: {problem[:30]}...")

        result = await cortex.chain_of_thought_reasoning(
            problem=problem,
            context={"domain": "逻辑推理"}
        )

        print(f"  ✅ 推理完成: 质量={result.get('quality_score', 0):.2f}")
        if result.get('final_conclusion'):
            conclusion = result['final_conclusion']
            print(f"  结论: {conclusion.get('conclusion', '无')[:50]}...")


def example_4_attention_system():
    """
    示例4: 注意力系统使用
    """
    print("\n" + "=" * 60)
    print("示例4: 注意力系统使用")
    print("=" * 60)

    from core.brain.thalamic_gate import ThalamicGate

    # 初始化注意力系统
    gate = ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8)
    print(f"✅ 注意力系统初始化: 输入维度={gate.input_dim}, 头数={gate.num_attention_heads}")

    # 模拟输入特征
    batch_size = 4
    seq_len = 20
    features = torch.randn(batch_size, seq_len, 128)

    # 计算注意力
    attention_weights = gate.compute_attention(features)
    print(f"✅ 注意力权重计算: 形状={attention_weights.shape}")

    # 焦点切换
    for i in range(3):
        output = gate.focus_attention(features, focus_strength=0.8)
        print(f"  焦点切换 {i+1}: 完成")

    # 获取注意力指标
    metrics = gate.get_attention_metrics()
    print(f"  注意力指标: 总切换={metrics['total_switches']}, 效率={metrics.get('efficiency', 0):.2f}")


def example_5_imagination_system():
    """
    示例5: 想象力系统使用
    """
    print("\n" + "=" * 60)
    print("示例5: 想象力系统使用")
    print("=" * 60)

    from core.brain.imagination_engine import ImaginationEngine

    # 初始化想象力引擎
    engine = ImaginationEngine(
        state_dim=64,
        hidden_dim=128,
        spatial_dim=3,
        temporal_dim=4
    )
    print("✅ 想象力引擎初始化完成")

    # 当前状态
    current_state = np.random.randn(64).astype(np.float32)
    spatial_info = np.array([10.0, 20.0, 5.0], dtype=np.float32)
    temporal_info = np.array([0.0, 1.0, 0.5, 0.1], dtype=np.float32)

    # 预测未来
    prediction = engine.predict_future_state(
        current_state,
        steps=5,
        spatial_info=spatial_info,
        temporal_info=temporal_info
    )

    print(f"✅ 未来预测完成: 预测步数={len(prediction['predicted_states'])}")
    print(f"  新颖性分数: {prediction['novelty_score']:.2f}")

    # 生成反事实场景
    real_action = np.array([1.0, 0.0, 0.5])
    counterfactuals = engine.generate_counterfactual(current_state, real_action, num_alternatives=3)

    print(f"✅ 反事实场景生成: {len(counterfactuals)} 个替代场景")


async def example_6_quantum_system():
    """
    示例6: 量子类脑融合系统使用
    """
    print("\n" + "=" * 60)
    print("示例6: 量子类脑融合系统使用")
    print("=" * 60)

    from core.quantum_brain.fusion_system import QuantumBrainFusion

    # 初始化量子融合系统
    fusion = QuantumBrainFusion(n_neurons=1000, n_qubits=5)
    fusion.initialize_system()
    print(f"✅ 量子融合系统初始化: 神经元={fusion.n_neurons}, 量子比特={fusion.n_qubits}")

    # 处理输入
    input_signal = np.random.randn(8)
    output = fusion.process_input(input_signal)
    print(f"✅ 输入处理完成: 输出维度={len(output)}")

    # 量子决策
    decision, confidence = fusion.make_fusion_decision(input_signal)
    print(f"✅ 量子决策: 决策={decision}, 置信度={confidence:.2f}")

    # 获取系统状态
    system_state = fusion.get_system_state()
    print(f"✅ 系统状态: 量子维度={system_state['quantum']['dimension']}")

    # 关闭系统
    fusion.shutdown()
    print("  系统已关闭")


def example_7_evolution_system():
    """
    示例7: 进化系统使用
    """
    print("\n" + "=" * 60)
    print("示例7: 进化系统使用")
    print("=" * 60)

    from core.evolution.genetic_engine import GeneticEngine

    # 初始化进化引擎
    engine = GeneticEngine(
        population_size=16,
        rule_dim=50,
        crossover_rate=0.7,
        mutation_rate=0.2
    )
    print(f"✅ 进化引擎初始化: 种群={engine.population_size}, 维度={engine.rule_dim}")

    # 定义适应度函数
    def fitness_func(individual):
        # 多目标适应度: 分数、学习速度、泛化能力
        score = np.mean(individual)
        learning_speed = np.sum(individual > 0) / len(individual)
        generalization = np.std(individual)
        return (score, learning_speed, generalization)

    engine.set_fitness_evaluator(fitness_func)

    # 初始化种群
    population = engine.initialize_population()
    print(f"✅ 种群初始化: {len(population)} 个个体")

    # 评估适应度
    fitness_scores = engine.evaluate_fitness(population)
    print(f"✅ 适应度评估完成: {len(fitness_scores)} 个分数")

    # 显示最佳个体
    best_idx = np.argmax([f[0] for f in fitness_scores])
    print(f"  最佳适应度: {fitness_scores[best_idx][0]:.4f}")


async def example_8_agent_system():
    """
    示例8: 智能体系统使用
    """
    print("\n" + "=" * 60)
    print("示例8: 智能体系统使用")
    print("=" * 60)

    from agents.single.intelligent_agent_system import IntelligentAgentSystem

    # 初始化智能体系统
    agent = IntelligentAgentSystem()
    print("✅ 智能体系统初始化完成")

    # 启动系统
    await agent.start_system()
    print("  系统已启动")

    # 测试原子动作
    print("\n  测试原子动作:")
    await agent.execute_atom_actions()

    # 测试技能
    print("\n  测试技能:")
    await agent.execute_skill_actions()

    # 获取性能指标
    metrics = agent.motion_controller.get_performance_metrics()
    print(f"\n  性能指标: 成功率={metrics['success_rate']:.2%}")

    # 停止系统
    await agent.stop_system()
    print("  系统已停止")


def example_9_multi_agent_system():
    """
    示例9: 多智能体系统使用
    """
    print("\n" + "=" * 60)
    print("示例9: 多智能体系统使用")
    print("=" * 60)

    from agents.multi.tribal_society import TribalSociety

    # 初始化部落社会
    society = TribalSociety(num_agents=10)
    print(f"✅ 部落社会初始化: {society.num_agents} 个智能体")

    # 创建智能体
    society.create_agents()
    print(f"  智能体创建完成: {len(society.agents)} 个")

    # 形成文化
    culture = society.form_culture(iterations=5)
    print(f"✅ 文化形成完成: 共享信念={len(culture.get('shared_beliefs', []))}")

    # 知识共享
    result = society.share_knowledge()
    print(f"✅ 知识共享完成: 共享知识={len(result.get('shared_knowledge', []))}")


def example_10_world_integration():
    """
    示例10: 世界集成系统使用
    """
    print("\n" + "=" * 60)
    print("示例10: 世界集成系统使用")
    print("=" * 60)

    from worlds.procgen.world_generator import WorldGenerator

    # 初始化世界生成器
    generator = WorldGenerator(world_size=(64, 64), seed=42)
    print("✅ 世界生成器初始化: 尺寸=64x64, 种子=42")

    # 生成地形
    terrain = generator.generate_terrain()
    print(f"✅ 地形生成完成: 形状={terrain.shape}")

    # 分布资源
    resources = generator.distribute_resources(terrain)
    print(f"✅ 资源分布完成: {len(resources)} 个资源点")


async def run_all_examples():
    """
    运行所有示例
    """
    print("\n" + "=" * 70)
    print(" NeuroMinecraft Genesis - 完整示例代码集 ")
    print("=" * 70)

    # 运行所有示例
    example_1_hello_world()
    example_2_memory_system()
    await example_3_reasoning_system()
    example_4_attention_system()
    example_5_imagination_system()
    await example_6_quantum_system()
    example_7_evolution_system()
    await example_8_agent_system()
    example_9_multi_agent_system()
    example_10_world_integration()

    print("\n" + "=" * 70)
    print(" 所有示例运行完成！ ")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
