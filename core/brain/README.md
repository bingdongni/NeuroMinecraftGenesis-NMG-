# 前额叶推理引擎 (PrefrontalCortex)

## 系统概述

前额叶推理引擎是思维力系统的核心推理模块，模拟人脑前额叶皮层的符号推理能力，提供链式推理、矛盾检测、信念管理和多步计划执行等高级认知功能。

## 核心特性

### 🧠 高级推理能力
- **链式推理**: 支持最多20步的深度推理，每步生成中间结论
- **置信度评估**: 每步推理都包含置信度评分，用于质量控制
- **推理质量分析**: 完整的推理过程记录，支持思维深度分析

### ⚖️ 智能矛盾管理
- **实时矛盾检测**: 维护信念图谱一致性，检测冲突强度
- **自动信念修正**: 基于证据强度的智能信念更新
- **冲突强度分级**: 直接矛盾、逻辑矛盾、语义矛盾三级分类

### 🌐 双模式LLM集成
- **GPT-3.5-turbo API**: 高质量云端推理能力
- **DialoGPT-small本地模型**: 离线推理保障
- **混合模式**: 自动切换，确保推理连续性

### 📊 完整性能监控
- **推理成功率追踪**: 目标≥70%
- **矛盾检测率控制**: 目标<5%
- **多维性能指标**: 详细的系统表现分析

## 核心方法

### 1. `chain_of_thought_reasoning(problem, context)`
链式推理核心方法，支持多步逻辑推理。

```python
# 使用示例
result = await cortex.chain_of_thought_reasoning(
    problem="如果明天下雨，我需要带伞吗？",
    context={"weather": "预报", "time": "明日"}
)

print(f"推理质量: {result['quality_score']:.3f}")
print(f"最终结论: {result['final_conclusion']['conclusion']}")
```

### 2. `detect_contradiction()`
矛盾检测方法，实时维护信念图谱一致性。

```python
# 检测矛盾
contradictions = await cortex.detect_contradiction()
print(f"发现 {len(contradictions)} 个矛盾")

for contradiction in contradictions:
    if contradiction.conflict_intensity > 0.8:
        print(f"高强度矛盾: {contradiction.node_a} vs {contradiction.node_b}")
```

### 3. `belief_revision(contradiction)`
信念修正方法，基于证据重新审视相关信念。

```python
# 修正矛盾信念
revision_result = await cortex.belief_revision(contradiction)
if revision_result['success']:
    print("信念修正成功")
    print(f"执行的操作: {revision_result['result']['revision_actions']}")
```

### 4. `create_belief_graph()`
构建完整的信念图谱NetworkX结构。

```python
# 构建信念图谱
belief_graph = cortex.create_belief_graph()
print(f"信念节点: {belief_graph.number_of_nodes()}")
print(f"关系边: {belief_graph.number_of_edges()}")
```

## 使用模式

### 初始化配置

```python
from core.brain.prefrontal_cortex import PrefrontalCortex, LLMMode

# 混合模式（推荐）
cortex = PrefrontalCortex(
    llm_mode=LLMMode.HYBRID,
    api_key="your_openai_key",  # 可选
    local_model_path="path/to/model",  # 可选
    max_reasoning_steps=20
)

# 仅API模式
api_cortex = PrefrontalCortex(
    llm_mode=LLMMode.API,
    api_key="your_openai_key"
)

# 仅本地模式
local_cortex = PrefrontalCortex(
    llm_mode=LLMMode.LOCAL,
    local_model_path="path/to/DialoGPT-small"
)
```

### 完整工作流程示例

```python
import asyncio
from core.brain.prefrontal_cortex import PrefrontalCortex, LLMMode

async def reasoning_workflow():
    # 初始化引擎
    cortex = PrefrontalCortex(llm_mode=LLMMode.HYBRID)
    
    # 1. 执行链式推理
    result = await cortex.chain_of_thought_reasoning(
        "分析人工智能对社会的影响",
        context={"focus": "正面影响"}
    )
    
    if result['success']:
        print(f"推理深度: {result['reasoning_depth']} 步")
        print(f"质量评分: {result['quality_score']:.3f}")
    
    # 2. 检测和修正矛盾
    contradictions = await cortex.detect_contradiction()
    for contradiction in contradictions[:3]:  # 处理前3个矛盾
        await cortex.belief_revision(contradiction)
    
    # 3. 构建信念图谱
    belief_graph = cortex.create_belief_graph()
    
    # 4. 查看性能指标
    metrics = cortex.get_performance_metrics()
    print(f"推理成功率: {metrics['reasoning_success_rate']:.1%}")

# 运行工作流程
asyncio.run(reasoning_workflow())
```

## 技术参数

### 推理配置
- **最大推理步数**: 20步（可配置）
- **置信度阈值**: 
  - 高置信度: 0.8
  - 中置信度: 0.6  
  - 低置信度: 0.4
  - 矛盾触发阈值: 0.6

### 性能目标
- **推理成功率**: ≥70%
- **矛盾检测率**: <5%
- **信念修正成功率**: ≥80%
- **推理质量评分**: 基于置信度一致性、推理深度等因素

### 系统架构
- **数据结构**: 基于dataclass的强类型设计
- **图谱结构**: NetworkX DiGraph支持复杂关系
- **并发支持**: 全面异步实现
- **异常处理**: 完整的错误处理和降级策略

## 依赖要求

### 必需依赖
```
numpy>=1.21.0
networkx>=2.6.0
```

### 可选依赖（LLM功能）
```
# GPT-3.5-turbo API支持
openai>=0.27.0

# 本地DialoGPT支持
transformers>=4.20.0
torch>=1.11.0
```

### 安装命令
```bash
# 基础依赖
pip install numpy networkx

# LLM支持（可选）
pip install openai
pip install transformers torch
```

## 异常处理

系统实现了完善的异常处理机制：

1. **LLM调用失败**: 自动降级到后备推理策略
2. **推理超时**: 超时保护，防止无限循环
3. **数据结构错误**: 类型检查和数据验证
4. **性能监控**: 实时跟踪系统表现

## 扩展开发

### 添加新的推理策略
```python
class CustomPrefrontalCortex(PrefrontalCortex):
    async def custom_reasoning_strategy(self, problem: str):
        # 实现自定义推理逻辑
        pass
```

### 自定义矛盾检测规则
```python
def custom_contradiction_check(self, content_a: str, content_b: str) -> float:
    # 实现自定义矛盾检测逻辑
    return conflict_score
```

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进这个系统。请确保：
1. 添加适当的测试用例
2. 遵循现有的代码风格
3. 更新相关文档

## 更新日志

### v1.0.0 (2025-11-13)
- ✅ 实现完整的PrefrontalCortex类
- ✅ 实现链式推理机制（最多20步）
- ✅ 实现矛盾检测和信念修正
- ✅ 实现信念图谱构建（NetworkX）
- ✅ 支持GPT-3.5-turbo API和本地DialoGPT
- ✅ 完整的性能监控和指标统计
- ✅ 详细的中文注释和异常处理

---

**注意**: 本系统为思维力系统的核心组件，需要与其他脑模块（如创意记忆、想象力引擎）协同工作。