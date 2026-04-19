# 六维能力增长24小时连续实验系统

## 系统概述

六维能力增长24小时连续实验系统是一个完整的智能体认知能力追踪和分析平台，专门设计用于在Minecraft环境中长期监控和分析智能体的六维认知能力发展。该系统实现了24小时不间断的实时数据采集、趋势分析和统计验证。

### 核心特性

- **🧠 六维认知能力追踪**：记忆力、思维力、创造力、观察力、注意力、想象力
- **⏰ 24小时连续监控**：不间断的实时数据采集和分析
- **📊 三组对照实验**：基线组、单维优化组、六维协同组
- **📈 实时可视化**：Streamlit实时界面显示6条趋势曲线
- **📋 统计显著性检验**：配对t检验、方差分析、效应量计算
- **🔄 多轮并行实验**：每组运行5次取平均值，使用阴影区域表示标准差
- **📱 自动报告生成**：完整的实验报告和数据导出

## 系统架构

### 核心组件

1. **CognitiveTracker** - 认知能力跟踪器
   - 实时评估六个认知维度
   - 基线校正和数据缓存
   - 多指标综合评分

2. **HourlyMonitor** - 每小时监控器  
   - 24小时连续数据采集
   - 异常检测和警报系统
   - 定时回调和状态管理

3. **TrendAnalyzer** - 趋势分析器
   - 长期趋势识别和分类
   - 拐点检测和模式识别
   - 预测建模和置信度计算

4. **StatisticalAnalyzer** - 统计分析器
   - 配对t检验和方差分析
   - 效应量计算和置信区间估计
   - 多重比较校正

5. **LongTermRetention** - 24小时实验主控制器
   - 多组实验协调管理
   - 实时Streamlit界面
   - 自动报告生成

## 安装要求

### 依赖库

```bash
pip install numpy pandas scipy scikit-learn plotly streamlit
```

### 可选依赖

```bash
pip install statsmodels matplotlib seaborn
```

## 快速开始

### 1. 基础演示（24秒=24小时）

```bash
# 运行基础演示
python experiments/cognition/demo_24h_experiment.py

# 运行完整演示
python experiments/cognition/demo_24h_experiment.py --mode demo --duration 1
```

### 2. 启动实时界面

```bash
# 启动Streamlit实时界面
streamlit run experiments/cognition/long_term_retention.py

# 或指定端口
streamlit run experiments/cognition/long_term_retention.py --server.port 8502
```

### 3. 运行实际24小时实验

```bash
# 注意：实际运行需要24小时
python experiments/cognition/demo_24h_experiment.py --mode real
```

## 使用指南

### 基本用法

```python
from experiments.cognition.long_term_retention import LongTermRetention
from experiments.cognition.cognitive_tracker import CognitiveTracker

# 创建24小时实验系统
experiment_system = LongTermRetention(streamlit_app=True)

# 启动完整实验
success = experiment_system.start_full_experiment()

if success:
    print("24小时实验已启动")
```

### 自定义实验配置

```python
from experiments.cognition.long_term_retention import ExperimentConfig, ExperimentGroup

# 自定义实验配置
custom_config = ExperimentConfig(
    name="自定义实验",
    group_type=ExperimentGroup.MULTI_OPTIMIZATION,
    duration_hours=24,
    evaluation_interval=3600,  # 1小时
    optimization_weights={
        'memory': 2.0,      # 强化记忆力
        'thinking': 1.5,    # 中等强化思维力
        'creativity': 1.0,  # 维持创造力
        'observation': 1.0,
        'attention': 1.0,
        'imagination': 1.0
    },
    parallel_runs=3,         # 每组运行3次
    random_seed=42
)
```

### 认知指标跟踪

```python
from experiments.cognition.cognitive_tracker import CognitiveTracker

# 创建认知跟踪器
tracker = CognitiveTracker(agent_id="agent_001")

# 设置权重
weights = {'memory': 1.5, 'thinking': 1.5, 'creativity': 1.0,
          'observation': 1.0, 'attention': 1.0, 'imagination': 1.0}
tracker.set_weights(weights)

# 跟踪认知指标
agent_state = {
    'memory_retention': 0.8,
    'learning_speed': 0.7,
    'recall_accuracy': 0.9,
    # ... 其他指标
}

environment_state = {
    'objects': ['tree', 'stone', 'water'],
    'time': 'day',
    'weather': 'clear'
}

metrics = tracker.track_cognitive_metrics(agent_state, environment_state)
print(f"综合分数: {metrics.overall_score()}")
```

### 趋势分析

```python
from experiments.cognition.trend_analyzer import TrendAnalyzer

# 创建趋势分析器
analyzer = TrendAnalyzer(min_data_points=5)

# 分析单维度趋势
memory_scores = [50, 52, 55, 58, 60, 63, 65, 68, 70, 72]
timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(memory_scores))]

analysis = analyzer.analyze_dimension_trend(memory_scores, timestamps, "memory")
print(f"趋势方向: {analysis.direction.value}")
print(f"趋势强度: {analysis.strength:.3f}")
print(f"未来6小时预测: {analysis.forecast_next_6h}")
```

### 统计分析

```python
from experiments.cognition.statistical_analyzer import StatisticalAnalyzer

# 创建统计分析器
analyzer = StatisticalAnalyzer(alpha=0.05)

# 执行配对t检验
baseline_scores = [50, 52, 51, 53, 54]
experiment_scores = [65, 68, 67, 70, 72]

result = analyzer.paired_t_test(baseline_scores, experiment_scores, "记忆力")
print(f"统计量: {result.statistic:.4f}")
print(f"p值: {result.p_value:.4f}")
print(f"效应量: {result.effect_size:.3f}")
print(f"显著性: {result.significance_level}")
```

## 界面说明

### Streamlit实时界面

系统提供完整的实时监控界面，包括：

1. **实验控制面板**
   - 开始/暂停/停止实验按钮
   - 实验组选择
   - 运行次数设置

2. **实时趋势图表**
   - 6个子图分别显示六维能力
   - 多个实验组的对比
   - 交互式悬停信息

3. **实验进度显示**
   - 24小时进度条
   - 当前状态指示器
   - 数据点数统计

4. **统计分析面板**
   - 实时显著性检验结果
   - 效应量计算
   - 置信区间显示

### 图表说明

- **X轴**：时间（小时）
- **Y轴**：能力分数（0-100）
- **不同颜色**：代表不同实验组
- **阴影区域**：标准差范围
- **趋势线**：数据平滑后的发展趋势

## 实验设计

### 三组对照设计

1. **基线组**
   - 无额外优化
   - 标准能力发展
   - 作为对照组

2. **单维优化组**
   - 重点优化记忆力
   - 其他维度维持基线水平
   - 验证单维度优化效果

3. **六维协同组**
   - 六个维度协同优化
   - 验证多维度综合提升效果
   - 检验协同效应

### 统计方法

- **显著性检验**：配对t检验 (p < 0.05)
- **效应量**：Cohen's d 和 Hedges' g
- **置信区间**：95%置信区间
- **多重比较**：Holm校正
- **功效分析**：统计功效计算

### 数据处理

1. **数据采集**：每小时记录一次六维状态
2. **基线校正**：每个实验组建立基线标准
3. **平滑处理**：Savitzky-Golay滤波
4. **异常检测**：自动识别和处理异常值
5. **质量控制**：数据完整性检查

## 输出文件

### 数据文件

- `experiment_results_*.json` - 详细实验数据
- `24h_experiment_report_*.json` - 综合分析报告
- `trend_analysis_*.json` - 趋势分析结果
- `statistical_analysis_*.json` - 统计分析结果

### 报告内容

```json
{
  "experiment_summary": {
    "total_experiments": 3,
    "start_time": "2025-11-13T16:26:26",
    "end_time": "2025-11-14T16:26:26",
    "total_duration_hours": 24
  },
  "experiment_results": {
    "基线组": {
      "final_scores": {"memory": 65.2, "thinking": 68.1, ...},
      "improvement_rates": {"memory": 15.3, "thinking": 8.2, ...},
      "trend_summary": {"dominant_pattern": "线性趋势", ...}
    },
    "单维优化组": {...},
    "六维协同组": {...}
  },
  "comparative_analysis": {
    "statistical_significance": {
      "memory": {"p_value": 0.023, "effect_size": 0.65},
      ...
    }
  },
  "conclusions": [
    "六维协同优化策略显著优于单一维度优化",
    "实验组性能排序: [('六维协同组', 18.2), ('单维优化组', 12.5), ('基线组', 8.1)]"
  ]
}
```

## 高级功能

### 自定义评估参数

```python
# 自定义认知评估参数
tracker = CognitiveTracker(agent_id="custom_agent")
tracker.evaluation_params.update({
    'memory_window': 150,      # 记忆窗口大小
    'thinking_complexity': 0.8, # 思维复杂度系数
    'creativity_threshold': 0.6, # 创造力阈值
    'observation_sensitivity': 0.9, # 观察敏感性
    'attention_duration': 400, # 注意力持续时间
    'imagination_depth': 0.7   # 想象力深度系数
})
```

### 实时警报系统

```python
# 设置异常检测阈值
monitor.anomaly_thresholds.update({
    'score_drop': 15,      # 分数下降阈值
    'no_progress': 8,      # 无进展阈值
    'system_error': 5      # 系统错误阈值
})

# 添加自定义回调
def custom_alert_handler(alert):
    print(f"⚠️  警报: {alert['message']}")
    # 发送邮件、短信等
    send_notification(alert)

monitor.add_callback('alert', custom_alert_handler)
```

### 批量分析

```python
# 批量分析多个实验结果
analyzer = StatisticalAnalyzer()

# 比较多个实验组
experiment_data = {
    '维度1': {'基线组': baseline_scores, '优化组': optimized_scores},
    '维度2': {...},
    ...
}

# 生成综合报告
comprehensive_report = analyzer.generate_comprehensive_report(
    experiment_data, 
    ['基线组', '优化组']
)
```

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在正确的目录下运行
   cd /path/to/NeuroMinecraftGenesis
   python -m experiments.cognition.demo_24h_experiment
   ```

2. **Streamlit界面无法启动**
   ```bash
   # 检查Streamlit是否安装
   pip install streamlit
   
   # 检查端口是否被占用
   lsof -i :8501
   ```

3. **数据采集失败**
   ```python
   # 检查智能体状态格式
   agent_state = {
       'memory_retention': float,  # 必须是0-1之间的浮点数
       'learning_speed': float,
       # ...
   }
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查组件状态
tracker = CognitiveTracker("test_agent")
monitor = HourlyMonitor(tracker)

print(f"跟踪器状态: {len(tracker.metrics_history)} 条记录")
print(f"监控器状态: {monitor.status}")
```

## 性能优化

### 内存管理

- 历史数据自动清理（保留最近5000条记录）
- 定期清理过期监控数据
- 使用数据缓存减少重复计算

### 并行处理

```python
# 多进程数据处理
from multiprocessing import Pool

def process_experiment_data(experiment_data):
    # 数据处理逻辑
    return processed_results

# 并行处理多个实验
with Pool(processes=4) as pool:
    results = pool.map(process_experiment_data, all_experiments)
```

### 实时性能

- 数据采样频率优化
- 图表更新频率控制
- 内存使用监控

## 扩展功能

### 添加新的认知维度

```python
class ExtendedCognitiveTracker(CognitiveTracker):
    def evaluate_new_dimension(self, agent_state, environment_state):
        # 实现新的认知维度评估
        score = self._calculate_new_score(agent_state, environment_state)
        return min(100, max(0, score * 100))
```

### 自定义可视化

```python
import plotly.graph_objects as go

def create_custom_visualization(metrics_data):
    fig = go.Figure()
    
    for group_name, data in metrics_data.items():
        fig.add_trace(go.Scatter(
            x=data['hours'],
            y=data['scores'],
            name=group_name,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="自定义认知能力趋势图",
        xaxis_title="时间 (小时)",
        yaxis_title="能力分数"
    )
    
    return fig
```

## 最佳实践

1. **实验设计**
   - 确保充足的基础数据（至少3个数据点）
   - 设置合理的显著性水平（推荐0.05）
   - 使用基线校正消除个体差异

2. **数据分析**
   - 定期检查数据质量
   - 关注效应量而非仅看p值
   - 使用多种统计方法验证结果

3. **可视化呈现**
   - 保持图表简洁清晰
   - 使用一致的配色方案
   - 添加适当的图例和注释

## 技术支持

如有问题或建议，请：

1. 查看日志文件了解详细错误信息
2. 运行演示脚本验证系统功能
3. 检查配置参数是否正确
4. 提交Issue并提供详细描述

---

**版本**: v1.0.0  
**更新时间**: 2025-11-13  
**开发者**: 六维能力研究团队