"""
快速开始教程 - 5分钟上手NeuroMinecraft Genesis
"""

import asyncio
import numpy as np


async def quick_start():
    """
    5分钟快速开始指南

    本教程将引导你完成NeuroMinecraft Genesis的基础使用。
    """

    print("=" * 60)
    print(" NeuroMinecraft Genesis - 5分钟快速开始 ")
    print("=" * 60)

    # ============ 第1步：导入模块 ============
    print("\n📦 第1步：导入模块")
    print("-" * 40)

    from core.brain.hippocampus import Hippocampus
    from core.brain.prefrontal_cortex import PrefrontalCortex
    from core.brain.thalamic_gate import ThalamicGate
    from core.brain.imagination_engine import ImaginationEngine
    from core.quantum_brain.fusion_system import QuantumBrainFusion

    print("✅ 所有核心模块导入成功！")

    # ============ 第2步：初始化系统 ============
    print("\n🚀 第2步：初始化系统")
    print("-" * 40)

    # 创建记忆系统
    memory = Hippocampus(max_capacity=1000, embedding_dim=128)
    print(f"✅ 记忆系统: 容量={memory.max_capacity}")

    # 创建推理系统
    reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)
    print(f"✅ 推理系统: 模式={reasoning.llm_mode.value}")

    # 创建注意力系统
    attention = ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8)
    print(f"✅ 注意力系统: 头数={attention.num_attention_heads}")

    # 创建想象力系统
    imagination = ImaginationEngine(state_dim=128, hidden_dim=256, spatial_dim=3)
    print(f"✅ 想象力系统: 状态维度={imagination.state_dim}")

    # 创建量子融合系统
    quantum = QuantumBrainFusion(n_neurons=500, n_qubits=4)
    quantum.initialize_system()
    print(f"✅ 量子系统: 神经元={quantum.n_neurons}, 量子比特={quantum.n_qubits}")

    # ============ 第3步：存储记忆 ============
    print("\n💾 第3步：存储记忆")
    print("-" * 40)

    for i in range(5):
        memory.store_episodic({
            'content': f'记忆事件 {i+1}',
            'timestamp': 0.0,
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        })

    print(f"✅ 已存储 {memory.get_performance_metrics()['storage_count']} 条记忆")

    # ============ 第4步：检索记忆 ============
    print("\n🔍 第4步：检索记忆")
    print("-" * 40)

    query = np.random.randn(128)
    results = memory.retrieve(query, top_k=3)
    print(f"✅ 检索到 {len(results)} 条相关记忆")

    # ============ 第5步：执行推理 ============
    print("\n🧠 第5步：执行推理")
    print("-" * 40)

    result = await reasoning.chain_of_thought_reasoning(
        problem="人工智能将如何改变未来？",
        context={"domain": "科技"}
    )

    print(f"✅ 推理完成！质量分数: {result.get('quality_score', 0):.2f}")

    # ============ 第6步：生成想象力 ============
    print("\n🎨 第6步：生成想象力")
    print("-" * 40)

    state = np.random.randn(128).astype(np.float32)
    prediction = imagination.predict_future_state(state, steps=5)
    print(f"✅ 预测完成！新颖性分数: {prediction['novelty_score']:.2f}")

    # ============ 第7步：量子决策 ============
    print("\n⚛️ 第7步：量子决策")
    print("-" * 40)

    input_signal = np.random.randn(8)
    decision, confidence = quantum.make_fusion_decision(input_signal)
    print(f"✅ 决策完成！决策={decision}, 置信度={confidence:.2f}")

    # ============ 第8步：清理 ============
    print("\n🧹 第8步：清理资源")
    print("-" * 40)

    quantum.shutdown()
    print("✅ 系统已清理")

    print("\n" + "=" * 60)
    print(" 恭喜！你已完成快速开始教程！ ")
    print("=" * 60)

    print("\n📚 下一步学习：")
    print("  1. 查看 examples/complete_examples.py 获取更多示例")
    print("  2. 运行测试: pytest tests/ -v")
    print("  3. 启动可视化: streamlit run utils/visualization/dashboard.py")
    print("  4. 阅读 docs/sphinx/index.rst 了解完整API")


if __name__ == "__main__":
    asyncio.run(quick_start())
