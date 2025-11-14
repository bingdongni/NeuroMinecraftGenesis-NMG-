# Utils模块

工具和实用程序模块，提供项目运行所需的各种辅助工具。

## 子模块

### visualization/
数据可视化工具
- 神经网络结构图
- 进化过程动画
- 代理行为轨迹图
- 实时监控面板

### logging/
日志记录系统
- 结构化日志格式
- 多级别日志控制
- 日志轮转和归档
- 性能监控日志

## 主要功能

```python
from .visualization import Visualizer
from .logging import LogManager

# 创建可视化器
viz = Visualizer()
viz.plot_neural_network(brain_model)
viz.animate_evolution(evolution_data)

# 配置日志系统
logger = LogManager.get_logger("NeuroMinecraft")
logger.info("System initialized")
```