# Agents模块

AI代理系统核心模块，负责创建和管理不同类型的智能代理。

## 子模块

### single/
单代理系统，处理独立的AI智能体
- 单个代理的感知、决策和行动
- 独立的认知架构
- 自主学习和适应

### multi/
多代理协作系统，处理代理间的交互
- 代理间通信协议
- 协作策略和任务分配
- 群体智能涌现

## 主要类和方法

```python
from .single import SingleAgent
from .multi import MultiAgentSystem
```

## 使用示例

```python
# 创建单代理
agent = SingleAgent(brain_model="neural_network")

# 创建多代理系统
system = MultiAgentSystem(num_agents=5)
```