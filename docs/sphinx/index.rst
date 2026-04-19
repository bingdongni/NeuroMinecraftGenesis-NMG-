# NeuroMinecraft Genesis API Documentation

欢迎使用 NeuroMinecraft Genesis API 文档！

## 项目概述

NeuroMinecraft Genesis 是一个集成了神经科学、量子计算和人工智能的自主进化认知系统。

## 主要模块

### 核心模块 (core)

| 模块 | 描述 |
|------|------|
| [brain](core/brain.rst) | 六维认知引擎 |
| [quantum](core/quantum.rst) | 量子类脑融合 |
| [evolution](core/evolution.rst) | 进化算法 |
| [perception](core/perception.rst) | 多模态感知 |

### 智能体模块 (agents)

| 模块 | 描述 |
|------|------|
| [single](agents/single.rst) | 单智能体系统 |
| [multi](agents/multi.rst) | 多智能体系统 |
| [mass_evolution](agents/mass_evolution.rst) | 大规模进化 |

### 世界模块 (worlds)

| 模块 | 描述 |
|------|------|
| [integrated](worlds/integrated.rst) | 三世界集成 |
| [procgen](worlds/procgen.rst) | 程序化生成 |
| [real](worlds/real.rst) | 真实世界 |

## 快速开始

```python
from core.brain import Hippocampus, PrefrontalCortex
from core.quantum_brain import QuantumBrainFusion

# 初始化系统
memory = Hippocampus(max_capacity=1000, embedding_dim=128)
reasoning = PrefrontalCortex(llm_mode='local', max_reasoning_steps=5)
quantum = QuantumBrainFusion(n_neurons=1000, n_qubits=5)

# 运行系统
quantum.initialize_system()
```

## API 参考

### core.brain 模块

#### Hippocampus

海马体记忆系统，负责情景记忆和空间记忆的存储与检索。

```python
from core.brain.hippocampus import Hippocampus

# 初始化
memory = Hippocampus(max_capacity=1000, embedding_dim=128)

# 存储记忆
memory.store_episodic({
    'content': 'Experience',
    'timestamp': time.time(),
    'emotion': np.random.randn(5),
    'sensory_data': np.random.randn(10),
    'context': np.random.randn(20)
})

# 检索记忆
results = memory.retrieve(query, top_k=5)
```

#### PrefrontalCortex

前额叶推理引擎，负责链式推理和逻辑演绎。

```python
from core.brain.prefrontal_cortex import PrefrontalCortex, LLMMode

# 初始化
cortex = PrefrontalCortex(llm_mode=LLMMode.HYBRID, max_reasoning_steps=10)

# 执行推理
result = await cortex.chain_of_thought_reasoning(
    problem="What is the meaning of life?",
    context={"domain": "philosophy"}
)
```

### core.quantum_brain 模块

#### QuantumBrainFusion

量子类脑融合系统。

```python
from core.quantum_brain.fusion_system import QuantumBrainFusion

# 初始化
fusion = QuantumBrainFusion(n_neurons=1000, n_qubits=5)
fusion.initialize_system()

# 处理输入
output = fusion.process_input(input_signal)

# 做出决策
decision, confidence = fusion.make_fusion_decision(input_signal)
```

### agents.single 模块

#### IntelligentAgentSystem

智能体动作系统。

```python
from agents.single.intelligent_agent_system import IntelligentAgentSystem

# 初始化
agent = IntelligentAgentSystem()
await agent.start_system()

# 执行动作
await agent.execute_atom_actions()
await agent.execute_skill_actions()

await agent.stop_system()
```

## 教程

- [快速开始指南](tutorials/getting_started.md)
- [六维认知引擎教程](tutorials/cognition_engine.md)
- [进化系统教程](tutorials/evolution_system.md)
- [多智能体系统教程](tutorials/multi_agent.md)

## 示例

查看 `examples/` 目录获取完整示例代码。
