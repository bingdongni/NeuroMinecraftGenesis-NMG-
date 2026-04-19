# 智能体动作系统

## 系统概述

本系统实现了27种原子动作和组合技能库的智能体动作系统，具备以下核心功能：

### 🎯 核心特性

1. **27种原子动作**
   - 8方向移动（前后左右 + 四个对角线方向）
   - 跳跃和飞行（跳跃、双跳、飞行控制）
   - 攻击和交互（普通攻击、右键交互、破坏方块）
   - 物品操作（放置方块、使用物品、丢弃物品）

2. **组合技能库**
   - 建造技能（房屋建造、农场搭建、防御工事）
   - 采集技能（矿物开采、树木采伐、水流收集）
   - 战斗技能（怪物清击、防御策略、逃脱路线）
   - 探索技能（地图绘制、资源发现、路径规划）

3. **10Hz动作控制**
   - 100ms控制周期
   - 动作优先级管理
   - 动作序列执行
   - 并行/顺序执行支持

4. **技能学习系统**
   - 经验值积累
   - 技能熟练度跟踪
   - 技能进化机制
   - 智能推荐系统

## 📁 文件结构

```
agents/single/
├── __init__.py                   # 包初始化文件
├── action_executor.py            # 原子动作执行器 (27种动作)
├── skill_library.py              # 组合技能库和学习系统
├── motion_controller.py          # 10Hz动作控制器
├── intelligent_agent_system.py   # 综合测试系统
└── README.md                     # 本文档
```

## 🚀 快速开始

### 1. 基本使用

```python
import asyncio
from action_executor import ActionExecutor, ActionType
from skill_library import SkillLibrary
from motion_controller import MotionController

async def basic_example():
    # 创建系统组件
    action_executor = ActionExecutor()
    skill_library = SkillLibrary(action_executor)
    motion_controller = MotionController(action_executor, skill_library)
    
    # 启动控制器
    await motion_controller.start()
    
    # 调度一个原子动作
    action_id = motion_controller.create_and_schedule_action(
        ActionType.MOVE_FORWARD,
        parameters={'distance': 5.0}
    )
    
    # 等待执行
    await asyncio.sleep(1.0)
    
    # 检查结果
    status = motion_controller.get_action_status(action_id)
    print(f"动作状态: {status}")
    
    # 停止控制器
    await motion_controller.stop()

# 运行示例
asyncio.run(basic_example())
```

### 2. 执行技能

```python
async def skill_example():
    action_executor = ActionExecutor()
    skill_library = SkillLibrary(action_executor)
    motion_controller = MotionController(action_executor, skill_library)
    
    await motion_controller.start()
    
    # 执行建造技能
    skill_id = motion_controller.create_and_schedule_action(
        "simple_house",
        parameters={
            'size': {'width': 3, 'length': 4},
            'materials': {'wood': 20, 'stone': 15},
            'quality': 0.9
        }
    )
    
    await asyncio.sleep(2.0)
    
    # 查看技能熟练度
    skill_info = skill_library.get_skill_info("simple_house")
    print(f"技能熟练度: {skill_info['mastery_level']}")
    
    await motion_controller.stop()
```

### 3. 使用动作序列

```python
async def sequence_example():
    action_executor = ActionExecutor()
    skill_library = SkillLibrary(action_executor)
    motion_controller = MotionController(action_executor, skill_library)
    
    await motion_controller.start()
    
    # 创建序列
    sequence = motion_controller.create_action_sequence(
        "build_sequence",
        "建造序列",
        parallel_execution=False
    )
    
    # 添加动作到序列
    motion_controller.add_action_to_sequence("build_sequence", ActionType.MOVE_FORWARD)
    motion_controller.add_action_to_sequence("build_sequence", ActionType.PLACE_BLOCK)
    motion_controller.add_action_to_sequence("build_sequence", ActionType.JUMP)
    
    # 启动序列
    await motion_controller.start_sequence("build_sequence")
    
    # 等待完成
    await asyncio.sleep(3.0)
    
    # 检查序列状态
    seq_status = motion_controller.get_sequence_status("build_sequence")
    print(f"序列状态: {seq_status}")
    
    await motion_controller.stop()
```

## 📚 详细API文档

### ActionExecutor (动作执行器)

负责执行27种原子动作。

#### 主要方法

- `execute_action(action_type, **kwargs)` - 执行原子动作
- `get_action_statistics()` - 获取动作统计信息
- `reset_state()` - 重置状态

#### 支持的原子动作

**移动动作 (8种)**:
- `MOVE_FORWARD` - 向前移动
- `MOVE_BACKWARD` - 向后移动
- `MOVE_LEFT` - 向左移动
- `MOVE_RIGHT` - 向右移动
- `MOVE_FORWARD_LEFT` - 左前移动
- `MOVE_FORWARD_RIGHT` - 右前移动
- `MOVE_BACKWARD_LEFT` - 左后移动
- `MOVE_BACKWARD_RIGHT` - 右后移动

**跳跃和飞行动作 (7种)**:
- `JUMP` - 跳跃
- `DOUBLE_JUMP` - 双跳
- `FLY_UP` - 向上飞行
- `FLY_DOWN` - 向下飞行
- `FLY_FORWARD` - 向前飞行
- `FLY_BACKWARD` - 向后飞行
- `FLY_STOP` - 停止飞行

**攻击和交互动作 (3种)**:
- `ATTACK` - 攻击
- `RIGHT_CLICK` - 右键交互
- `DESTROY_BLOCK` - 破坏方块

**物品操作动作 (5种)**:
- `PLACE_BLOCK` - 放置方块
- `USE_ITEM` - 使用物品
- `DROP_ITEM` - 丢弃物品
- `INVENTORY_OPEN` - 打开背包
- `INVENTORY_CLOSE` - 关闭背包

### SkillLibrary (技能库)

管理组合技能和技能学习系统。

#### 主要方法

- `execute_skill(skill_name, **kwargs)` - 执行技能
- `get_skill_info(skill_name)` - 获取技能信息
- `get_recommended_skills()` - 获取推荐技能
- `get_skills_by_category(category)` - 按分类获取技能

#### 技能分类

**建造技能**:
- `simple_house` - 简易房屋建造
- `farm_construction` - 农场搭建
- `defense_structure` - 防御工事建造
- `advanced_architecture` - 高级建筑技术

**采集技能**:
- `basic_mining` - 基础矿物开采
- `tree_harvesting` - 树木采伐
- `deep_mining` - 深层矿物开采
- `water_collection` - 水流收集技术
- `precious_mining` - 珍贵矿物开采

**战斗技能**:
- `basic_combat` - 基础战斗
- `group_combat` - 群体攻击
- `defensive_strategy` - 防御策略
- `escape_route` - 逃脱路线规划
- `elite_combat` - 精英怪物战斗

**探索技能**:
- `basic_exploration` - 基础探索
- `resource_discovery` - 资源发现
- `path_planning` - 路径规划
- `terrain_analysis` - 地形分析
- `remote_exploration` - 远程探索

### MotionController (动作控制器)

提供10Hz频率的动作调度和优先级控制。

#### 主要方法

- `start()` - 启动控制器
- `stop()` - 停止控制器
- `create_and_schedule_action()` - 创建并调度动作
- `cancel_action(action_id)` - 取消动作
- `create_action_sequence()` - 创建动作序列
- `get_performance_metrics()` - 获取性能指标

#### 动作优先级

- `EMERGENCY` (0) - 紧急任务
- `HIGH` (1) - 高优先级
- `NORMAL` (2) - 普通优先级
- `LOW` (3) - 低优先级
- `BACKGROUND` (4) - 后台任务

## 🧪 测试和验证

### 运行综合测试

```bash
cd agents/single
python intelligent_agent_system.py
```

测试将验证：
- ✅ 27种原子动作的执行
- ✅ 组合技能的使用
- ✅ 动作优先级系统
- ✅ 动作序列管理
- ✅ 技能学习机制
- ✅ 系统性能指标

### 单元测试

```python
# 测试单个组件
from action_executor import ActionExecutor, ActionType

async def test_action_executor():
    executor = ActionExecutor()
    result = await executor.execute_action(ActionType.JUMP, height=2.0)
    print(f"跳跃结果: {result.success}")
    
asyncio.run(test_action_executor())
```

## 📊 性能特性

### 控制频率
- **10Hz**: 每100ms执行一次控制循环
- **低延迟**: 平均动作延迟 < 10ms
- **高并发**: 支持最多5个并发动作

### 内存管理
- 动作历史记录限制: 1000条
- 技能执行历史: 每技能100条
- 自动清理过期动作

### 扩展性
- 支持自定义原子动作
- 支持添加新技能
- 支持自定义优先级
- 支持动态调整并发数

## 🔧 配置选项

### 控制器配置

```python
# 设置最大并发动作数
motion_controller.set_max_concurrent_actions(8)

# 设置控制频率（默认10Hz）
motion_controller.control_frequency = 20.0  # 20Hz
```

### 技能学习配置

```python
# 设置经验值倍数
skill_library.experience_multiplier = 1.5

# 导出技能数据
skill_library.export_skill_data("my_skills.json")

# 导入技能数据
skill_library.import_skill_data("my_skills.json")
```

## 🐛 故障排除

### 常见问题

1. **动作执行超时**
   - 检查 `timeout` 参数设置
   - 确认系统资源充足

2. **技能执行失败**
   - 检查前置技能要求
   - 确认有足够的资源

3. **序列执行卡住**
   - 检查依赖关系设置
   - 确认暂停条件

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看队列状态
queue_status = motion_controller.get_queue_status()
print(f"队列状态: {queue_status}")

# 查看性能指标
metrics = motion_controller.get_performance_metrics()
print(f"性能指标: {metrics}")
```

## 🤝 贡献指南

欢迎贡献代码和建议！

### 开发规范
- 使用中文注释
- 遵循PEP 8代码风格
- 添加适当的测试用例
- 更新文档

### 扩展系统
1. 添加新的原子动作到 `ActionExecutor`
2. 添加新技能到 `SkillLibrary`
3. 更新 `motion_controller.py` 支持新功能

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 📞 联系我们

如有问题或建议，请联系开发团队。

---

🎯 **让智能体动作系统为您的AI应用提供强大的动作执行能力！**