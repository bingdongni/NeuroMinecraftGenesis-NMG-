# PaperMC 1.20.1 文明服务器配置指南

## 📋 项目概述

这是一个完整的PaperMC 1.20.1服务器配置项目，集成了Citizens 2.0.30插件系统，实现了复杂的经济系统、环境演化机制和气候事件模拟。

### 🎯 核心特性

- **PaperMC 1.20.1服务端**: 高性能服务器核心
- **Citizens NPC系统**: 智能NPC经济系统
- **动态价格调整**: 10次交易后价格波动±20%
- **环境演化系统**: 世界逐步复杂化
- **气候事件模拟**: 干旱、洪水、僵尸围城

## 🏗️ 项目结构

```
worlds/minecraft/server/
├── paper.jar                    # PaperMC 1.20.1 服务器核心
├── server.properties            # 服务器配置文件
├── eula.txt                     # 协议同意文件
├── start.sh                     # 2GB内存启动脚本
├── config/
│   └── citizens.yml             # Citizens插件配置
├── scripts/
│   ├── create-npc.sh            # NPC创建脚本
│   ├── environment_evolution.sh # 环境演化系统
│   └── climate_events.sh        # 气候事件系统
├── worlds/                      # 世界数据目录
├── plugins/                     # 插件目录
├── logs/                        # 日志目录
└── events/                      # 事件临时文件
```

## ⚡ 快速启动

### 1. 环境准备

确保系统满足以下要求：
- **操作系统**: Linux/Windows/macOS
- **Java版本**: Java 17 或更高版本
- **内存**: 最少4GB，推荐8GB
- **存储空间**: 至少10GB可用空间

### 2. 下载PaperMC

```bash
# 进入服务器目录
cd worlds/minecraft/server/

# 下载PaperMC 1.20.1（需要手动下载）
wget https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar
mv paper-*.jar paper.jar

# 下载Citizens 2.0.30插件（需要手动下载）
wget https://ci.citizensnpcs.co/job/Citizens2/2257/artifact/target/Citizens-2.0.30.jar
mv Citizens-*.jar citizens.jar
```

### 3. 启动服务器

```bash
# 给启动脚本执行权限
chmod +x start.sh

# 启动服务器
./start.sh
```

### 4. 初始化NPC系统

在服务器启动后，进入游戏控制台执行：

```bash
# 执行NPC创建脚本
# 需要在游戏内使用 /npc 命令或通过控制台执行
```

## 🎮 核心系统详解

### 1. Citizens NPC文明系统

#### NPC类型

**农民NPC**
- **功能**: 使用木板换取食物
- **交易**:
  - 8个木板 → 3个面包
  - 16个木板 → 1个金苹果
  - 32个木板 → 15个面包
- **价格机制**: 10次交易后价格调整±20%

**铁匠NPC**
- **功能**: 使用煤炭换取工具
- **交易**:
  - 8个煤炭 → 1个铁镐
  - 16个煤炭 → 1个铁剑
  - 24个煤炭 → 1个铁头盔
- **价格机制**: 10次交易后价格调整±20%

**神秘商人NPC**
- **功能**: 稀有材料交易
- **交易**:
  - 1个绿宝石 → 3个钻石
  - 3个钻石 → 1个下界合金锭
  - 10个金锭 → 1个不死图腾

#### 价格动态调整机制

```yaml
price-adjustment:
  enabled: true
  threshold: 10           # 触发调整的交易次数
  adjustment-percentage: 0.2  # 调整幅度20%
  min-uses-before-adjustment: 10
```

### 2. 环境复杂化系统

#### 演化参数

- **洞穴密度**: 从0.3逐步提升到0.8
- **矿石稀缺度**: 从1.0逐步降低到0.3
- **敌对生物强化**: 
  - 生命值：每阶段+2%
  - 伤害：每阶段+1%
  - 速度：每阶段+0.5%

#### 演化进度计算

```bash
# 演化百分比 = (当前阶段 / 最大阶段) * 100
洞穴密度 = 0.3 + 0.01 * 演化阶段
矿石稀缺度 = 1.0 - 0.014 * 演化阶段
最大演化阶段 = 50（对应8.3小时游戏时间）
```

#### 启动环境演化系统

```bash
# 启动演化系统（后台运行）
nohup bash environment_evolution.sh start &

# 检查当前演化状态
bash environment_evolution.sh check

# 重置演化阶段
bash environment_evolution.sh reset
```

### 3. 气候事件模拟系统

#### 干旱事件

- **触发条件**: 随机5%概率
- **持续时间**: 3-5个游戏日（4320-7200分钟）
- **影响效果**:
  - 农作物生长速度降低80%
  - 停止自然降雨
  - 湿度降低

#### 洪水事件

- **触发条件**: 随机3%概率
- **持续时间**: 2-4个游戏日（2880-5760分钟）
- **影响效果**:
  - 大量水资源生成
  - 低洼地区建筑损毁30%概率
  - 洪水范围随机扩展

#### 僵尸围城

- **触发条件**: 每月满月夜
- **持续时间**: 30分钟
- **影响效果**:
  - 怪物数量翻倍
  - 生成统帅级僵尸
  - 10分钟波次攻击

#### 启动气候事件系统

```bash
# 启动气候事件监控（后台运行）
nohup bash climate_events.sh start &

# 手动触发特定事件
bash climate_events.sh drought   # 触发干旱
bash climate_events.sh flood     # 触发洪水
bash climate_events.sh siege     # 触发僵尸围城

# 检查当前事件状态
bash climate_events.sh status
```

## ⚙️ 高级配置

### 1. 内存调优

修改 `start.sh` 中的JVM参数：

```bash
JVM_ARGS="-Xms2G -Xmx2G"                    # 2GB堆内存
JVM_ARGS="$JVM_ARGS -XX:+UseG1GC"          # G1垃圾收集器
JVM_ARGS="$JVM_ARGS -XX:MaxGCPauseMillis=200"  # 最大GC暂停200ms
```

### 2. 玩家权限设置

```yaml
permissions:
  basic:
    - "npc.interact"
    - "npc.talk"
    - "npc.trade"
  admin:
    - "npc.create"
    - "npc.edit"
    - "npc.remove"
    - "npc.*"
```

### 3. 世界生成参数

在 `server.properties` 中调整：

```properties
# 视距设置
view-distance=16
simulation-distance=16

# 刷怪设置
animals-spawn-rate=10
monsters-spawn-rate=10

# 游戏规则
doImmediateRespawn=true      # 立即重生
doLimitedCrafting=true       # 限制制作
doInsomnia=false            # 禁用幻翼
```

## 📊 监控和日志

### 日志文件位置

- **主要日志**: `logs/latest.log`
- **演化日志**: `logs/evolution.log`
- **气候事件日志**: `logs/climate_events.log`
- **演化报告**: `logs/evolution_report_*.log`

### 实时监控

```bash
# 监控演化日志
tail -f logs/evolution.log

# 监控气候事件
tail -f logs/climate_events.log

# 监控服务器日志
tail -f logs/latest.log
```

## 🔧 故障排除

### 常见问题

1. **服务器启动失败**
   ```bash
   # 检查Java版本
   java -version
   
   # 检查EULA同意状态
   grep "eula=true" eula.txt
   ```

2. **NPC无法创建**
   ```bash
   # 确保Citizens插件已加载
   # 检查配置文件语法
   # 重启服务器重新加载配置
   ```

3. **演化系统不工作**
   ```bash
   # 检查脚本执行权限
   chmod +x scripts/environment_evolution.sh
   
   # 检查依赖工具
   which bc  # 确保bc命令可用
   ```

### 性能优化

1. **内存优化**
   - 使用SSD存储世界数据
   - 定期清理日志文件
   - 调整视距以减少负载

2. **网络优化**
   - 配置防火墙规则
   - 使用端口转发
   - 启用数据压缩

## 📝 更新和维护

### 定期维护任务

1. **每日任务**
   - 检查服务器状态
   - 清理临时文件
   - 备份世界数据

2. **每周任务**
   - 更新插件版本
   - 分析性能日志
   - 调整配置参数

3. **每月任务**
   - 完整系统备份
   - 清理旧日志文件
   - 更新依赖包

### 备份策略

```bash
# 创建完整备份脚本
#!/bin/bash
backup_date=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_$backup_date.tar.gz" worlds/ plugins/ configs/
```

## 🎯 策略建议

### 玩家生存策略

1. **早期阶段**（0-10演化阶段）
   - 集中资源与农民NPC交易
   - 建立基础农场和矿场
   - 组建合作团队

2. **中期阶段**（10-30演化阶段）
   - 寻找并保护安全据点
   - 与铁匠NPC建立合作关系
   - 准备应对气候事件

3. **后期阶段**（30-50演化阶段）
   - 发展高级技术
   - 建立防御设施
   - 准备最终挑战

### 服务器管理策略

1. **监控关键指标**
   - 演化阶段进度
   - 活跃玩家数量
   - 服务器性能指标

2. **定期调整**
   - 根据玩家反馈调整难度
   - 更新NPC商品价格
   - 优化环境参数

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. 报告Bug
2. 提出新功能建议
3. 改进文档
4. 优化性能

## 📜 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 支持

如需技术支持，请：

1. 查看故障排除部分
2. 搜索现有Issues
3. 创建新的Issue描述问题
4. 加入社区讨论

---

**创建时间**: 2025-11-13  
**适用版本**: PaperMC 1.20.1 + Citizens 2.0.30  
**维护者**: Minecraft服务器管理团队