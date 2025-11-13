# 神经符号混合架构 (Neuro-Symbolic Architecture)

## 概述

本项目实现了一个完整的神经符号混合架构，整合了神经网络和符号推理的优势。该架构支持神经-符号双向映射、实时知识提取、混合推理和动态学习。

## 核心组件

### 1. NeuroSymbolicArchitecture (核心架构类)
- **位置**: `core/symbolic/neuro_symbolic_architecture.py`
- **功能**: 整合所有组件，提供统一的神经符号接口
- **主要方法**:
  - `initialize_architecture()`: 初始化架构组件
  - `extract_symbolic_knowledge()`: 从神经网络中提取符号知识
  - `initialize_network()`: 基于符号知识初始化神经网络
  - `hybrid_reasoning()`: 执行神经符号混合推理

### 2. NeuralSymbolicBridge (神经符号桥接器)
- **位置**: `core/symbolic/neural_symbolic_bridge.py`
- **功能**: 实现神经网络和符号表示之间的双向映射
- **主要方法**:
  - `translate_neural_to_symbolic()`: 神经网络激活转为符号表示
  - `translate_symbolic_to_neural()`: 符号规则转为神经网络结构
  - `validate_consistency()`: 验证映射一致性

### 3. SymbolExtraction (符号知识提取器)
- **位置**: `core/symbolic/symbol_extraction.py`
- **功能**: 从神经网络激活中提取符号知识
- **主要方法**:
  - `extract_symbolic_knowledge()`: 提取符号概念、关系和规则
  - `build_knowledge_graph()`: 构建知识图谱
  - `assess_knowledge_quality()`: 评估知识质量

### 4. NeuralInitialization (神经网络初始化器)
- **位置**: `core/symbolic/neural_initialization.py`
- **功能**: 基于符号知识初始化神经网络权重
- **主要方法**:
  - `initialize_network_from_knowledge()`: 基于符号知识初始化网络
  - `apply_symbolic_constraints()`: 应用符号约束
  - `optimize_weights_for_constraints()`: 优化权重以满足约束

### 5. HybridReasoning (混合推理引擎)
- **位置**: `core/symbolic/hybrid_reasoning.py`
- **功能**: 执行多种模式的神经符号混合推理
- **主要方法**:
  - `reason()`: 执行混合推理
  - `neural_reasoning()`: 纯神经网络推理
  - `symbolic_reasoning()`: 纯符号推理
  - `hybrid_reasoning()`: 混合推理

## 快速开始

### 1. 基本使用

```python
from core.symbolic import NeuroSymbolicArchitecture

# 配置网络和符号参数
network_config = {
    "input_dim": 128,
    "hidden_dims": [256, 128],
    "output_dim": 64,
    "activation": "relu"
}

symbolic_config = {
    "activation_threshold": 0.5,
    "inference_depth": 3,
    "initialization_method": "knowledge_guided"
}

# 创建架构
architecture = NeuroSymbolicArchitecture(
    network_config, 
    symbolic_config, 
    inference_mode="hybrid"
)

# 初始化架构
knowledge_base = {
    "concepts": {
        "concept1": {
            "attributes": {"feature": "value"},
            "neural_representation": {"neuron_indices": [0, 1, 2], "weights": [0.8, 0.7, 0.9]},
            "confidence": 0.9
        }
    },
    "relations": {},
    "rules": []
}

architecture.initialize_architecture(knowledge_base)

# 执行混合推理
import torch
input_data = torch.randn(128)
result = architecture.hybrid_reasoning(input_data)
print(f"推理结果: {result}")
```

### 2. 符号知识提取

```python
# 从神经网络激活中提取符号知识
neural_activations = torch.randn(128)
extraction_result = architecture.extract_symbolic_knowledge(
    neural_activations,
    context={"task": "classification"}
)

print(f"提取的概念: {extraction_result['concepts']}")
print(f"提取的关系: {extraction_result['relations']}")
print(f"生成的规则: {extraction_result['rules']}")
```

### 3. 基于符号知识的网络初始化

```python
# 使用符号知识初始化神经网络
symbolic_knowledge = {
    "concepts": {
        "object_recognition": {
            "neural_representation": {
                "neuron_indices": [10, 11, 12, 13, 14],
                "weights": [0.8, 0.9, 0.7, 0.6, 0.8]
            },
            "confidence": 0.9
        }
    },
    "relations": {},
    "rules": []
}

initialization_result = architecture.initialize_network(
    symbolic_knowledge,
    optimization_config={"max_iterations": 100}
)

print(f"初始化验证分数: {initialization_result['validation_score']}")
```

## 技术特性

### 1. 神经符号双向映射
- 神经网络激活 ↔ 符号表示
- 符号规则 ↔ 网络权重
- 保持映射一致性

### 2. 实时知识转换
- 动态提取符号知识
- 实时推理路径选择
- 自适应权重调整

### 3. 多模式推理
- **神经模式**: 纯神经网络推理
- **符号模式**: 纯符号推理
- **混合模式**: 神经符号协同推理
- **自适应模式**: 根据输入特征自动选择

### 4. 质量保证
- 知识一致性验证
- 推理结果验证
- 性能监控和报告

### 5. 可扩展性
- 模块化设计
- 插件式组件
- 配置化参数

## 配置参数

### 神经网络配置
```python
network_config = {
    "input_dim": 128,           # 输入维度
    "hidden_dims": [256, 128],  # 隐藏层维度列表
    "output_dim": 64,           # 输出维度
    "activation": "relu",       # 激活函数
    "dropout_rate": 0.1,        # Dropout率
    "learning_rate": 0.001      # 学习率
}
```

### 符号推理配置
```python
symbolic_config = {
    "activation_threshold": 0.5,           # 激活阈值
    "concept_clustering": "dbscan",        # 概念聚类方法
    "relation_threshold": 0.7,            # 关系阈值
    "inference_depth": 3,                 # 推理深度
    "confidence_threshold": 0.5,          # 置信度阈值
    "initialization_method": "knowledge_guided",  # 初始化方法
    "prior_strength": 0.8,               # 先验强度
    "parallel_reasoning": True,          # 并行推理
    "adaptive_mode_selection": True      # 自适应模式选择
}
```

## 示例演示

运行完整演示：
```bash
python core/symbolic/demo_neuro_symbolic.py
```

演示包括：
1. 架构初始化
2. 符号知识提取
3. 神经网络初始化
4. 多种模式推理
5. 性能分析

## 性能优化

### 1. 缓存机制
- 推理结果缓存
- 映射规则缓存
- 动态缓存管理

### 2. 并行处理
- 多线程推理
- 异步知识提取
- 并行权重优化

### 3. 自适应优化
- 动态策略选择
- 性能监控
- 自动参数调优

## 错误处理

所有组件都包含完善的错误处理：
- 初始化验证
- 运行时异常捕获
- 优雅降级机制
- 详细错误日志

## 监控和报告

### 1. 性能监控
- 推理时间统计
- 成功率跟踪
- 内存使用监控
- 缓存命中率分析

### 2. 质量评估
- 知识一致性评分
- 推理准确性评估
- 系统健康检查

### 3. 生成报告
```python
# 获取性能报告
performance_report = architecture.get_performance_report()
print(f"平均推理时间: {performance_report['inference_statistics']['average_time']}")

# 获取架构状态
state = architecture.get_architecture_state()
print(f"架构状态: {state}")
```

## 扩展和定制

### 1. 自定义推理策略
```python
# 在HybridReasoning中添加新的推理策略
def custom_reasoning_strategy(self, input_data):
    # 实现自定义推理逻辑
    pass
```

### 2. 扩展知识表示
```python
# 在SymbolExtraction中添加新的知识提取方法
def custom_knowledge_extraction(self, neural_data):
    # 实现自定义知识提取
    pass
```

### 3. 自定义融合策略
```python
# 在NeuralSymbolicBridge中添加新的结果融合方法
def custom_fusion_method(self, neural_result, symbolic_result):
    # 实现自定义融合逻辑
    pass
```

## 依赖项

主要依赖：
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Scikit-learn >= 0.24.0
- NetworkX >= 2.5.0
- SciPy >= 1.6.0

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎贡献！请确保：
1. 代码符合PEP8标准
2. 包含充分的测试
3. 更新相关文档
4. 通过所有现有测试

## 更新日志

### v1.0.0 (2025-11-13)
- 初始版本发布
- 实现核心神经符号混合架构
- 支持多模式推理
- 包含完整的演示和文档