# 进化可视化和断点续跑系统

## 系统概述

本系统为NeuroMinecraftGenesis项目提供了完整的进化可视化和断点续跑功能，支持实时监控进化过程、自动保存检查点以及断点恢复功能。

## 核心组件

### 1. EvolutionVisualizer (进化可视化器)

负责实时监控和可视化进化过程，包括：

- **实时进化曲线可视化**：显示最佳、平均、最差适应度变化
- **适应度地形3D展示**：绘制3D适应度地形图
- **遗传多样性变化监控**：追踪种群多样性变化
- **种群进化历史记录**：保存完整的进化历史数据
- **进化树可视化**：生成进化树图表

#### 核心方法：

- `visualize_evolution_progress()` - 实时保存每代数据到JSON
- `plot_fitness_landscape()` - 绘制3D适应度地形图  
- `render_evolution_tree()` - 进化树可视化展示
- `get_evolution_summary()` - 获取进化过程摘要

### 2. CheckpointManager (检查点管理器)

负责管理进化检查点，支持断点续跑功能：

- **自动保存检查点**：按间隔自动保存最佳个体
- **手动保存检查点**：支持手动触发保存
- **断点恢复功能**：从指定检查点恢复进化
- **检查点管理**：维护检查点历史，清理旧文件
- **数据完整性验证**：确保恢复数据的完整性

#### 核心方法：

- `save_checkpoint()` - 保存检查点
- `load_checkpoint()` - 加载检查点
- `list_checkpoints()` - 列出所有检查点
- `cleanup_old_checkpoints()` - 清理旧检查点

### 3. EvolutionDashboard (进化仪表板)

提供实时监控仪表板：

- **实时数据更新**：自动加载最新进化数据
- **多图表展示**：适应度曲线、多样性图、热图等
- **交互式界面**：支持动态更新和状态监控
- **静态报告生成**：生成静态仪表板图片
- **性能监控**：跟踪系统性能指标

## 文件结构

```
NeuroMinecraftGenesis/
├── core/
│   └── evolution/
│       ├── evolution_visualizer.py      # 进化可视化器
│       ├── checkpoint_manager.py        # 检查点管理器
│       ├── __init__.py                  # 模块初始化
│       └── evolution_demo.py            # 演示脚本
├── utils/
│   └── visualization/
│       ├── evolution_dashboard.py       # 进化仪表板
│       └── __init__.py
├── models/
│   └── genomes/                         # 检查点存储目录
│       ├── history/                     # 历史检查点
│       └── best/                        # 最佳个体保存
└── data/
    └── evolution_logs/                  # 进化日志和可视化
        ├── generation_*.json            # 每代数据
        ├── evolution_progress_*.png     # 进化进度图
        ├── fitness_landscape_*.png      # 3D地形图
        └── evolution_tree_*.png         # 进化树图
```

## 使用示例

### 1. 基础使用

```python
from core.evolution import EvolutionVisualizer, CheckpointManager
import numpy as np

# 创建可视化器
visualizer = EvolutionVisualizer(
    population_size=100,
    genome_length=20,
    data_dir="data/evolution_logs",
    checkpoint_dir="models/genomes"
)

# 创建检查点管理器
checkpoint_manager = CheckpointManager(
    checkpoint_dir="models/genomes",
    auto_save_interval=10
)

# 模拟进化过程
for generation in range(50):
    # 生成种群和适应度
    population = [np.random.randn(20) for _ in range(100)]
    fitness_scores = [np.sum(ind**2) for ind in population]
    
    # 更新可视化器
    visualizer.update_population_state(population, fitness_scores, generation)
    
    # 自动保存检查点
    if checkpoint_manager.should_auto_save(generation):
        checkpoint_info = checkpoint_manager.save_checkpoint(
            population, fitness_scores, generation
        )
        print(f"Gen {generation}: 保存检查点")
    
    # 每10代生成可视化
    if generation % 10 == 0:
        visualizer.visualize_evolution_progress()
        visualizer.plot_fitness_landscape()
```

### 2. 断点续跑

```python
# 从最新检查点恢复
load_result = checkpoint_manager.load_checkpoint()
if load_result:
    state = load_result['state']
    generation = state['generation']
    population = state['population']
    fitness_scores = state['fitness_scores']
    
    print(f"从第 {generation} 代继续进化")
    
    # 继续进化过程...
    for gen in range(generation, 100):
        # ... 继续进化代码
```

### 3. 仪表板监控

```python
from utils.visualization import EvolutionDashboard

# 创建仪表板
dashboard = EvolutionDashboard(
    data_dir="data/evolution_logs",
    update_interval=2.0
)

# 生成静态仪表板
dashboard.create_static_dashboard(
    output_path="evolution_dashboard.png",
    include_analysis=True
)

# 获取状态信息
status = dashboard.get_current_status()
print(f"当前代数: {status['current_generation']}")
```

## 运行演示

系统提供了完整的演示脚本：

```bash
cd NeuroMinecraftGenesis
python core/evolution/evolution_demo.py
```

演示包含：
1. 完整进化过程演示（包括断点续跑）
2. 检查点恢复功能演示  
3. 仪表板功能演示

## 配置参数

### EvolutionVisualizer参数

```python
visualizer = EvolutionVisualizer(
    population_size=100,        # 种群大小
    genome_length=20,           # 基因组长度
    data_dir="data/evolution_logs",  # 数据保存目录
    checkpoint_dir="models/genomes", # 检查点目录
    max_history=1000            # 最大历史记录数
)
```

### CheckpointManager参数

```python
checkpoint_manager = CheckpointManager(
    checkpoint_dir="models/genomes", # 检查点目录
    auto_save_interval=10,      # 自动保存间隔
    max_checkpoints=100,        # 最大检查点数量
    backup_enabled=True         # 是否启用备份
)
```

### EvolutionDashboard参数

```python
dashboard = EvolutionDashboard(
    data_dir="data/evolution_logs",  # 数据目录
    update_interval=2.0,        # 更新间隔(秒)
    auto_reload=True,           # 自动重新加载
    dashboard_config={           # 仪表板配置
        'show_fitness_curve': True,
        'show_diversity_plot': True,
        'show_3d_trajectory': True,
        'show_best_individual': True,
        'show_species_evolution': True,
        'max_history_points': 500,
        'animation_speed': 100
    }
)
```

## 技术特性

### 实时监控
- 自动检测新数据文件
- 实时更新可视化图表
- 性能监控和状态追踪

### 断点机制
- 自动保存间隔控制
- 最佳个体优先保存
- 数据完整性验证
- 自动清理旧检查点

### 可视化功能
- 多维度进化数据展示
- 3D适应度地形图
- 进化树可视化
- 实时状态仪表板

### 数据管理
- JSON格式的进化数据
- 自动目录结构管理
- 数据压缩和归档
- 跨平台兼容性

## 注意事项

1. **数据目录**: 确保有足够的磁盘空间存储进化数据和检查点
2. **性能考虑**: 大量数据点时建议限制历史记录数量
3. **内存管理**: 定期清理旧检查点，避免内存占用过高
4. **数据备份**: 重要实验建议定期备份检查点数据

## 扩展功能

系统支持以下扩展：

- 自定义适应度函数
- 多种可视化样式
- 分布式检查点存储
- 实时数据导出
- 进化分析报告生成

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖包已安装
2. **数据文件未找到**: 检查数据目录路径配置
3. **可视化显示异常**: 检查matplotlib后端设置
4. **检查点加载失败**: 验证数据文件完整性

### 日志查看

系统会生成详细的日志文件：
- `data/evolution_logs/evolution_visualizer.log`
- 控制台实时日志输出

## 版本历史

- **v1.0.0** (2025-11-13): 初始版本，支持基础可视化和断点功能