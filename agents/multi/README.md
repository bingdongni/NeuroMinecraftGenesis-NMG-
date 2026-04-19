# 多智能体社会系统开发文档

## 项目概述

这是一个完整的多智能体社会系统，模拟16个智能体构成的部落，实现集体智能、协作协议和社会认知机制。

## 核心特性

### 🤖 多智能体架构
- **16个智能体** 构成一个完整的部落
- **8种性格类型**: 领导者、协作者、探索者、建造者、研究者、保护者、创造者、分析者
- **个体特征建模**: 能量、动机、社交倾向、风险容忍度、学习速度、创造力
- **私有目标管理**: 个人建筑偏好、探索方向选择、技能发展

### 🧠 集体记忆系统 (`collective_memory.py`)
- **公共危险区域标注**: 实时记录和共享危险区域信息
- **资源热点坐标**: 精确的资源位置和质量信息
- **有效建造蓝图**: 共享的建筑设计和施工方案
- **知识版本控制**: 可靠性评分、验证机制、记忆衰减
- **智能检索**: 基于类型、标签、空间的智能搜索
- **知识融合**: 多个记忆的自动整合和优化

### 🗣️ 社会认知能力 (`social_cognition.py`)
- **意图识别**: 通过行为分析理解其他智能体的目标
- **信任建立**: 基于合作历史的动态信任评估
- **社会学习**: 观察学习、直接教学、模仿学习
- **领导力选举**: 基于多标准的动态领导者选举
- **社交网络分析**: 中心性、聚类系数、网络密度分析
- **社交推荐**: 协作、学习、信任的智能推荐

### 🤝 协作协议系统 (`collaboration_protocol.py`)
- **任务分配**: 基于能力和工作负载的智能分配
- **资源分享**: 动态资源分配和共享网络
- **冲突解决**: 智能冲突检测和多种解决策略
- **集体决策**: 投票机制、决策门槛、集体智慧
- **协作效率优化**: 实时性能监控和优化建议

### 🏛️ 部落社会系统 (`tribal_society.py`)
- **集体智能涌现**: 超越个体总和的群体智慧
- **专业化分工**: 基于性格和能力的任务专业化
- **适应性学习**: 对环境变化的集体适应
- **社会网络演化**: 动态的社交关系网络
- **实时模拟**: 可配置的模拟速度和事件系统

## 文件结构

```
agents/multi/
├── __init__.py                    # 模块初始化
├── collective_memory.py          # 集体记忆系统
├── social_cognition.py           # 社会认知能力
├── collaboration_protocol.py     # 协作协议系统
├── tribal_society.py            # 部落社会系统核心
├── test_multi_agent_system.py    # 完整测试套件
└── README.md                     # 本文档
```

## 核心系统架构

### 数据流图
```
智能体行为 → 社会认知系统 → 协作协议
     ↓              ↓            ↓
集体记忆 ← 社会交互 ← 任务执行
     ↓              ↓            ↓
集体智能 ← 记忆更新 ← 决策制定
```

### 主要类结构

```python
# 集体记忆系统
CollectiveMemory
├── MemoryEntry (记忆条目)
├── 索引系统 (类型、标签、空间)
└── 知识图谱管理

# 社会认知系统  
SocialCognitionSystem
├── 行为记录 (SocialAction)
├── 意图分析 (Intention)
├── 学习事件 (SocialLearningEvent)
└── 信任网络管理

# 协作协议系统
CollaborationProtocol
├── 任务管理 (Task)
├── 资源分配 (Resource)
├── 冲突解决 (Conflict)
└── 决策机制 (DecisionProposal)

# 部落社会系统
TribalSociety
├── 智能体管理 (AgentCharacteristics)
├── 社交网络
├── 集体智能计算
└── 模拟引擎
```

## 使用指南

### 1. 基本使用

```python
from tribal_society import TribalSociety

# 创建部落
tribe = TribalSociety(agent_count=16)

# 启动模拟
tribe.start_simulation()

# 获取部落状态
status = tribe.get_tribal_status()
print(f"集体智能评分: {status['collective_metrics']['collective_intelligence']:.3f}")
```

### 2. 测试系统

```python
# 运行完整测试
python test_multi_agent_system.py

# 运行演示
from tribal_society import run_comprehensive_demo
demo_path = run_comprehensive_demo()
```

### 3. 自定义配置

```python
# 创建自定义部落
config = {
    'agent_count': 12,  # 自定义智能体数量
    'memory_capacity': 3000,  # 记忆容量
    'simulation_speed': 2.0   # 模拟速度
}

tribe = create_tribal_society_with_config(config)
```

## 核心功能详解

### 集体记忆机制

```python
# 创建危险区域记忆
danger_memory = create_danger_zone_memory(
    x=100, y=0, z=50,
    danger_type="creeper_spawn",
    description="苦力怕刷新点",
    contributor_id="agent_0"
)

# 存储到集体记忆
memory_id = collective_memory.store_memory(danger_memory)

# 检索相关记忆
danger_zones = collective_memory.get_danger_zones(spatial_range=((0,0,0), (200,50,200)))
```

### 社会认知能力

```python
# 分析智能体意图
intentions = social_cognition.analyze_intentions("agent_0")

# 建立信任模型
trust_model = social_cognition.build_trust_model("agent_0")

# 发起社会学习
learning_success = social_cognition.initiate_social_learning(
    learner_id="agent_0",
    teacher_id="agent_1", 
    skill_type="exploration",
    learning_method="observation"
)
```

### 协作协议

```python
# 创建任务
task = create_simple_task(
    TaskType.CONSTRUCTION,
    "建造防御塔",
    TaskPriority.HIGH
)

task_id = collaboration_protocol.create_task(task)

# 分配任务
assignment_success = collaboration_protocol.assign_task(task_id, "agent_0")

# 资源分享
share_success = collaboration_protocol.share_resource(
    resource_id="iron_ore",
    sharer_id="agent_1",
    recipient_id="agent_0", 
    quantity=10
)
```

## 集体智能指标

### 核心指标
- **集体智能评分**: 0.0-1.0，综合评估群体智慧水平
- **社会网络密度**: 智能体间连接密度
- **知识共享程度**: 集体记忆利用率
- **协作效率**: 任务完成和资源利用效率
- **适应性**: 对环境变化的适应能力
- **创新能力**: 基于创造力和协作的创新水平

### 涌现现象
- **超越个体总和**: 群体表现超过单独个体的简单相加
- **分工专业化**: 自然形成的能力分工
- **知识融合**: 多源信息的自动整合
- **自适应学习**: 集体层面学习新技能

## 模拟配置

### 模拟参数
```python
simulation_config = {
    "agent_count": 16,           # 智能体数量
    "simulation_speed": 1.0,     # 模拟速度倍数
    "memory_capacity": 5000,     # 记忆容量限制
    "event_frequency": 12,       # 事件触发频率(步数)
    "cleanup_interval": 48,      # 清理间隔(步数)
}
```

### 世界事件
- **资源发现**: 新的资源位置发现
- **危险出现**: 新的威胁和危险区域
- **技术突破**: 重要技术进展
- **环境变化**: 天气、季节等环境变化

## 性能特征

### 扩展性
- 支持 8-32 个智能体
- 内存使用与智能体数量线性增长
- 计算复杂度 O(n²) 在社交网络分析中

### 性能指标
- 模拟步执行时间: < 0.01秒 (16智能体)
- 记忆检索延迟: < 0.001秒
- 决策计算时间: < 0.005秒

## 应用场景

### 游戏AI
- NPC社会系统
- 团队协作AI
- 动态故事生成

### 分布式系统
- 多机器人协调
- 云服务协作
- 网络自治系统

### 研究领域
- 社会学建模
- 集体行为研究
- 人工智能集体智慧

## 扩展开发

### 添加新性格类型
```python
class NewPersonality(Enum):
    MEDITATOR = "meditator"

def _create_new_personality_characteristics(agent_id: str, personality: NewPersonality):
    # 实现新性格特征生成
    pass
```

### 添加新任务类型
```python
# 在collaboration_protocol.py中添加
class NewTaskType(Enum):
    MEDITATION = "meditation"

# 更新任务分配逻辑
def _evaluate_agent_suitability_for_meditation(agent, task):
    # 评估适合性逻辑
    pass
```

### 自定义记忆类型
```python
def create_custom_memory(content, memory_type, contributor_id):
    return MemoryEntry(
        id="",
        content=content,
        memory_type=memory_type,
        timestamp=datetime.now(),
        reliability_score=0.8,
        contributor_id=contributor_id,
        # ... 其他字段
    )
```

## 故障排除

### 常见问题

1. **内存不足**
   - 降低memory_capacity参数
   - 增加cleanup_interval频率

2. **性能下降**
   - 减少智能体数量
   - 降低模拟速度
   - 优化记忆清理策略

3. **社交网络稀疏**
   - 调整初始化连接数量
   - 增加社交事件频率

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
tribe = TribalSociety(agent_count=8)
```

## 开发团队

- **系统架构**: 集体智能设计
- **核心算法**: 社会认知、协作协议
- **测试框架**: 完整测试套件
- **文档**: 详细中文注释和说明

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

**注意**: 本系统专为研究目的设计，在生产环境中使用前请进行充分测试和验证。