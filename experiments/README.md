# Experiments模块

实验模块，用于运行和评估各种AI实验。

## 子模块

### memory/
记忆系统实验
- 短期记忆测试
- 长期记忆实验
- 情景记忆模拟
- 记忆巩固机制

### evolution/
进化实验
- 代际演化测试
- 适应度函数评估
- 变异和选择实验
- 进化速度测试

### cognition/
认知实验
- 学习能力测试
- 推理能力评估
- 决策制定实验
- 注意力机制研究

## 实验运行

```python
from .memory import MemoryExperiment
from .evolution import EvolutionExperiment  
from .cognition import CognitionExperiment

# 运行记忆实验
exp = MemoryExperiment(agent_id="agent_1")
results = exp.run()

# 运行进化实验
evo_exp = EvolutionExperiment(population_size=100)
evo_results = evo_exp.run()
```