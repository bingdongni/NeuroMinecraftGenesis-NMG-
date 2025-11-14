# 泛化能力压力测试系统 - 快速启动指南

## 系统简介

这是一个全面的泛化能力测试系统，用于评估智能体在未见任务上的零样本迁移能力。系统包含三个不同认知领域的测试模块，提供从零样本到少样本的完整评估框架。

## 核心特性

- ✅ **跨域测试**: 游戏环境、物理模拟、社交对话
- ✅ **零样本评估**: 在未见任务上的直接表现测试
- ✅ **少样本适应**: 有限学习后的改进效果评估
- ✅ **适应速度计算**: (少样本性能-零样本性能)/50
- ✅ **跨域迁移分析**: 不同领域间知识相关性评估
- ✅ **详细报告生成**: JSON、CSV、TXT多格式输出

## 快速开始

### 1. 环境准备

确保Python环境可用：
```bash
python --version  # 建议Python 3.7+
```

### 2. 运行演示

```bash
cd /workspace/NeuroMinecraftGenesis
python demo_generalization_test.py
```

### 3. 单模块测试

#### 模组Minecraft测试
```python
from experiments.cognition.modded_minecraft_test import ModdedMinecraftTest

# 创建测试实例
mc_test = ModdedMinecraftTest()

# 运行零样本测试
zero_shot_score = mc_test.run_zero_shot_test()
print(f"零样本分数: {zero_shot_score:.3f}")

# 运行少样本测试
few_shot_score = mc_test.run_few_shot_test(baseline_score=zero_shot_score)
print(f"少样本分数: {few_shot_score:.3f}")

# 生成详细报告
report = mc_test.generate_detailed_report()
```

#### PyBullet物理模拟器测试
```python
from experiments.cognition.pybullet_test import PyBulletTest

# 创建测试实例
pb_test = PyBulletTest()

# 运行测试
zero_shot_score = pb_test.run_zero_shot_test()
few_shot_score = pb_test.run_few_shot_test(baseline_score=zero_shot_score)

# 物理理解评估
physics_understanding = pb_test.evaluate_physics_understanding()
print("物理理解能力:", physics_understanding)
```

#### Reddit对话测试
```python
from experiments.cognition.reddit_dialogue_test import RedditDialogueTest

# 创建测试实例
rd_test = RedditDialogueTest()

# 运行测试
zero_shot_score = rd_test.run_zero_shot_test()
few_shot_score = rd_test.run_few_shot_test(baseline_score=zero_shot_score)

# 社交认知评估
social_metrics = rd_test.evaluate_social_understanding()
print("社交认知指标:", social_metrics)
```

### 4. 综合测试

```python
from experiments.cognition.generalization_test import GeneralizationTest

# 创建综合测试实例
general_test = GeneralizationTest(
    output_dir="/path/to/output",
    max_few_shot_attempts=50
)

# 运行综合测试
report = general_test.run_comprehensive_test()

# 生成性能摘要
summary = general_test.generate_performance_summary(report)
print(summary)

# 导出结果
json_file = general_test.export_results("json")
csv_file = general_test.export_results("csv")
txt_file = general_test.export_results("txt")
```

### 5. 性能评估

```python
from experiments.cognition.zero_shot_evaluator import ZeroShotEvaluator, DomainType, PerformanceMetrics

# 创建评估器
evaluator = ZeroShotEvaluator(confidence_level=0.95)

# 模拟性能数据
mock_metrics = evaluator.evaluate_single_task(
    task_name="minecraft_terralith",
    domain=DomainType.MINECRAFT,
    zero_shot_results=[0.45, 0.52, 0.38],
    few_shot_results=[0.78, 0.82, 0.75],
    max_attempts=50
)

# 生成综合评估报告
evaluation_report = evaluator.comprehensive_evaluation([mock_metrics])

# 基准对比
benchmark_comparison = evaluator.benchmark_comparison(evaluation_report)
print("基准对比结果:", benchmark_comparison)
```

## 测试结果解读

### 性能指标

1. **零样本分数 (0.0-1.0)**
   - >0.7: 优秀的零样本迁移能力
   - 0.4-0.7: 良好的迁移能力
   - <0.4: 迁移能力有限

2. **适应速度 (0.0-0.1)**
   - >0.01: 快速适应
   - 0.005-0.01: 正常适应
   - <0.005: 适应较慢

3. **学习效率 (0.0-1.0)**
   - >0.7: 高效学习
   - 0.4-0.7: 中等效率
   - <0.4: 学习效率低

### 领域性能

- **Minecraft测试**: 空间推理和环境适应能力
- **PyBullet测试**: 物理理解和精确操作能力
- **Reddit测试**: 社交认知和交流能力

## 扩展开发

### 添加新的测试任务

1. 继承基础测试类
2. 实现零样本和少样本测试方法
3. 定义性能评估指标
4. 集成到综合测试系统中

```python
from experiments.cognition.generalization_test import GeneralizationTest

class NewTaskTest:
    def __init__(self):
        # 初始化新任务环境
        pass
    
    def run_zero_shot_test(self):
        # 实现零样本测试逻辑
        return 0.5
    
    def run_few_shot_test(self, max_attempts, baseline_score):
        # 实现少样本测试逻辑
        return 0.8
```

### 自定义评估指标

```python
from experiments.cognition.zero_shot_evaluator import EvaluationMetric

# 自定义评估权重
evaluator.metric_weights = {
    EvaluationMetric.ACCURACY: 0.3,
    EvaluationMetric.SUCCESS_RATE: 0.25,
    EvaluationMetric.ADAPTATION_SPEED: 0.25,
    EvaluationMetric.LEARNING_EFFICIENCY: 0.2
}
```

## 故障排除

### 常见问题

1. **模块导入错误**
   - 确保Python路径正确设置
   - 检查所有依赖文件是否存在

2. **测试超时**
   - 减少测试样本数量
   - 调整测试时间限制

3. **内存不足**
   - 清理中间结果
   - 批处理大规模测试

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行测试时将显示详细日志信息
```

## 性能优化

### 1. 并行测试
```python
from concurrent.futures import ThreadPoolExecutor

def run_parallel_tests():
    tasks = [mc_test, pb_test, rd_test]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda t: t.run_zero_shot_test(), tasks))
    return results
```

### 2. 缓存结果
```python
import functools

@functools.lru_cache(maxsize=128)
def cached_zero_shot_test(task_config):
    return run_test(task_config)
```

### 3. 批量评估
```python
def batch_evaluate(metrics_list):
    evaluator = ZeroShotEvaluator()
    return evaluator.comprehensive_evaluation(metrics_list)
```

## 进一步阅读

- 详细API文档请参考各模块的docstring
- 测试用例和示例代码请查看demo脚本
- 性能基准和评估方法请参考完整报告

---

**快速启动完成！** 🎉

现在您可以开始使用泛化能力压力测试系统来评估智能体的跨域迁移能力了。