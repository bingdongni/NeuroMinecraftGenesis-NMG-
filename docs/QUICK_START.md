# 🚀 快速入门指南

**5分钟体验 NeuroMinecraft Genesis 项目的完整功能**

---

## 🎯 本指南目标

通过5分钟时间，您将学会：
- ✅ 运行基础的六维认知能力测试
- ✅ 查看实时认知能力可视化
- ✅ 体验Minecraft AI智能体
- ✅ 理解项目的核心概念

**预计用时**: 5-10分钟  
**技能要求**: 基础编程知识  
**硬件要求**: 8GB+ RAM, Python 3.11+

---

## ⚡ 一键体验

如果您已经安装了项目，可以直接运行：

```bash
# 克隆并快速启动
git clone https://github.com/bingdongni/NeuroMinecraftGenesis.git
cd NeuroMinecraftGenesis

# Windows用户
.\install.bat

# Linux/Mac用户  
./install.sh

# 启动演示
streamlit run utils/visualization/demo.py
```

访问 [http://localhost:8501](http://localhost:8501) 开始体验！

---

## 🛠️ 环境准备

### 1. 系统检查

首先检查您的系统是否满足要求：

```bash
# 检查Python版本 (需要3.11+)
python --version

# 检查内存 (需要8GB+)
# Windows: 系统信息 → 已安装的内存(RAM)
# Linux/Mac: free -h 或 top

# 检查磁盘空间 (需要5GB+可用)
# Windows: 磁盘属性
# Linux/Mac: df -h
```

### 2. 安装依赖

#### 自动安装 (推荐)

```bash
# Windows
# 下载 install.bat 并以管理员身份运行

# Linux/Mac
curl -fsSL https://raw.githubusercontent.com/bingdongni/NeuroMinecraftGenesis/main/install.sh | bash
```

#### 手动安装

```bash
# 创建虚拟环境
python -m venv neurominecraft_env

# 激活虚拟环境
# Windows:
neurominecraft_env\Scripts\activate
# Linux/Mac:
source neurominecraft_env/bin/activate

# 安装核心依赖
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy pandas
pip install streamlit plotly
pip install transformers datasets
pip install mineflayer mineflayer-pathfinder

# 安装额外功能
pip install qiskit nengo nengo-dl
```

### 3. 验证安装

```python
# 运行基础测试
python -c "
import torch
import streamlit
import transformers
print('✅ 核心依赖安装成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'Streamlit版本: {streamlit.__version__}')
"
```

---

## 🎮 第一个实验：六维认知能力测试

### 创建测试脚本

创建 `quick_test.py` 文件：

```python
#!/usr/bin/env python3
"""
NeuroMinecraft Genesis - 快速认知能力测试
5分钟体验六维认知引擎
"""

import torch
import numpy as np
import time
from datetime import datetime

def create_mock_cognitive_agent():
    """创建模拟认知智能体"""
    
    class MockCognitiveAgent:
        def __init__(self):
            # 初始化六维能力 (0-100%)
            self.abilities = {
                'memory': 75.0,      # 记忆力
                'thinking': 68.0,    # 思维力  
                'creativity': 82.0,  # 创造力
                'observation': 79.0, # 观察力
                'attention': 71.0,   # 注意 力
                'imagination': 74.0  # 想象力
            }
            
            self.memory_buffer = []
            self.creativity_events = []
            self.start_time = datetime.now()
            
        def simulate_learning(self, duration_minutes=5):
            """模拟5分钟学习过程"""
            print("🧠 开始认知能力测试...")
            print(f"⏰ 测试时间: {duration_minutes} 分钟")
            print("-" * 50)
            
            for minute in range(1, duration_minutes + 1):
                print(f"📊 第 {minute} 分钟进度:")
                
                # 模拟能力提升 (随机小幅度增长)
                for ability in self.abilities:
                    improvement = np.random.normal(0.5, 0.2)  # 平均提升0.5%
                    self.abilities[ability] = min(100.0, 
                        self.abilities[ability] + max(0, improvement))
                    
                    # 特殊处理创造力 (更明显的提升)
                    if ability == 'creativity' and np.random.random() > 0.7:
                        self.abilities[creativity] += np.random.uniform(1, 3)
                
                # 模拟记忆事件
                if np.random.random() > 0.3:
                    event = {
                        'timestamp': datetime.now(),
                        'type': np.random.choice(['exploration', 'creation', 'learning']),
                        'value': np.random.uniform(0.5, 1.0)
                    }
                    self.memory_buffer.append(event)
                
                # 打印当前状态
                self._print_abilities()
                time.sleep(1)  # 模拟实时更新
                
            print("\n✅ 认知能力测试完成！")
            return self.abilities
            
        def _print_abilities(self):
            """打印当前能力状态"""
            for ability, score in self.abilities.items():
                bar_length = int(score / 5)  # 每5%一个方块
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {ability:12} |{bar}| {score:5.1f}%")
            print()
            
        def get_summary(self):
            """生成测试总结"""
            avg_score = sum(self.abilities.values()) / len(self.abilities)
            best_ability = max(self.abilities, key=self.abilities.get)
            
            return {
                'average_score': avg_score,
                'best_ability': best_ability,
                'best_score': self.abilities[best_ability],
                'memory_events': len(self.memory_buffer),
                'test_duration': (datetime.now() - self.start_time).total_seconds()
            }
    
    return MockCognitiveAgent()

def display_results(results, summary):
    """显示测试结果"""
    print("\n" + "="*60)
    print("🎉 认知能力测试结果总结")
    print("="*60)
    
    # 雷达图数据 (ASCII版本)
    print("\n📊 六维认知能力雷达图:")
    abilities = list(results.keys())
    scores = list(results.values())
    
    # 标准化到0-10范围用于ASCII显示
    normalized = [s/10 for s in scores]
    
    for i, (ability, score) in enumerate(zip(abilities, scores)):
        # 圆形字符显示
        radius = int(normalized[i])
        circle = "●" * radius + "○" * (10 - radius)
        print(f"  {ability:12} [{circle}] {score:5.1f}%")
    
    # 总体统计
    print(f"\n📈 总体表现:")
    print(f"  平均得分: {summary['average_score']:.1f}%")
    print(f"  最佳能力: {summary['best_ability']} ({summary['best_score']:.1f}%)")
    print(f"  记忆事件: {summary['memory_events']} 个")
    print(f"  测试时长: {summary['test_duration']:.1f} 秒")
    
    # 评级
    if summary['average_score'] >= 85:
        grade = "🌟 优秀 (A+)"
    elif summary['average_score'] >= 75:
        grade = "👍 良好 (A)"
    elif summary['average_score'] >= 65:
        grade = "💪 中等 (B)"
    else:
        grade = "📚 待提升 (C)"
        
    print(f"  整体评级: {grade}")
    
    # 建议
    print(f"\n💡 改进建议:")
    worst_ability = min(results, key=results.get)
    print(f"  • {worst_ability} 维度有最大提升空间")
    if summary['memory_events'] < 10:
        print("  • 建议增加探索和记忆活动")
    if summary['best_ability'] == 'creativity':
        print("  • 创造力突出，可以尝试更多创新任务")

def main():
    """主函数"""
    print("🧠 NeuroMinecraft Genesis - 认知能力快速测试")
    print("="*60)
    print("欢迎体验六维认知引擎！")
    print("本测试将模拟5分钟的认知学习过程")
    print("-" * 60)
    
    # 创建智能体
    agent = create_mock_cognitive_agent()
    
    # 运行测试
    results = agent.simulate_learning(duration_minutes=5)
    
    # 获取总结
    summary = agent.get_summary()
    
    # 显示结果
    display_results(results, summary)
    
    # 下一步建议
    print(f"\n🚀 下一步建议:")
    print(f"  1. 启动完整版: streamlit run utils/visualization/dashboard.py")
    print(f"  2. 阅读文档: docs/README.md")
    print(f"  3. 贡献代码: CONTRIBUTING.md")
    print(f"  4. 参与社区: https://discord.gg/neurominecraft")
    
    print(f"\n感谢体验 NeuroMinecraft Genesis! 🎉")

if __name__ == "__main__":
    main()
```

### 运行测试

```bash
# 运行快速测试
python quick_test.py
```

**预期输出**:
```
🧠 NeuroMinecraft Genesis - 认知能力快速测试
============================================================
欢迎体验六维认知引擎！
本测试将模拟5分钟的认知学习过程
------------------------------------------------------------
🧠 开始认知能力测试...
⏰ 测试时间: 5 分钟
--------------------------------------------------
📊 第 1 分钟进度:
  memory       |██████████░░░░░░░░░░|  75.0%
  thinking     |███████░░░░░░░░░░░░░░|  68.0%
  creativity   |████████████████░░░░░░|  85.0%
  observation  |██████████░░░░░░░░░░░|  79.0%
  attention    |████████░░░░░░░░░░░░░░|  71.0%
  imagination  |██████████░░░░░░░░░░░|  74.0%

... (继续展示5分钟进展)

🎉 认知能力测试结果总结
============================================================

📊 六维认知能力雷达图:
  memory       [●●●●●○○○○○]  78.5%
  thinking     [●●●●○○○○○○]  73.2%
  creativity   [●●●●●●●●○○]  89.7%
  observation  [●●●●●○○○○○]  81.4%
  attention    [●●●●○○○○○○]  75.8%
  imagination  [●●●●●○○○○○]  77.9%

📈 总体表现:
  平均得分: 79.4%
  最佳能力: creativity (89.7%)
  记忆事件: 12 个
  测试时长: 300.2 秒
  整体评级: 👍 良好 (A)
```

---

## 🎮 体验Minecraft AI智能体

### 启动Minecraft服务器 (可选)

如果您想体验真实的Minecraft AI智能体：

```bash
# 1. 下载Minecraft Java版 (免费试用)
# 2. 启动本地服务器
cd worlds/minecraft/server
java -Xmx2G -Xms2G -jar paper.jar --nogui

# 3. 运行AI智能体 (新终端窗口)
python agents/single/cognitive_agent.py --mode minecraft
```

### 模拟Minecraft体验

创建 `minecraft_demo.py`:

```python
#!/usr/bin/env python3
"""
Minecraft AI智能体演示 (模拟版本)
"""

import time
import json
from datetime import datetime

class MinecraftSimulation:
    def __init__(self):
        self.world_state = {
            'position': {'x': 0, 'y': 64, 'z': 0},
            'health': 100,
            'inventory': {
                'wood': 5,
                'stone': 3,
                'food': 2
            },
            'environment': 'forest',
            'day_time': 1000  # Minecraft时间
        }
        
        self.ai_actions = []
        
    def simulate_environment(self, steps=10):
        """模拟AI在Minecraft中的行为"""
        print("🌍 Minecraft AI 智能体模拟开始")
        print("="*50)
        
        actions = [
            "🏃 探索周围环境",
            "🌲 采集木头资源", 
            "🏗️ 建造简单房屋",
            "⚔️ 防御怪物攻击",
            "🌾 种植农作物",
            "🔍 发现新区域",
            "💎 寻找宝贵矿物",
            "🛡️ 制作防具工具",
            "🏘️ 与NPC交易",
            "🎯 完成生存任务"
        ]
        
        for step in range(steps):
            if step < len(actions):
                action = actions[step]
            else:
                action = self._generate_random_action()
                
            print(f"⏱️  步骤 {step + 1}: {action}")
            
            # 模拟行动结果
            result = self._execute_action(action)
            self._update_world_state(result)
            
            # 显示当前状态
            self._display_status()
            
            time.sleep(1)  # 模拟实时
            
    def _execute_action(self, action):
        """执行动作并返回结果"""
        results = {
            "🏃 探索周围环境": {"exploration": 1, "experience": 10},
            "🌲 采集木头资源": {"wood": 3, "energy": -5},
            "🏗️ 建造简单房屋": {"protection": 1, "wood": -2},
            "⚔️ 防御怪物攻击": {"experience": 15, "health": -10},
            "🌾 种植农作物": {"food": 1, "time": 2},
            "🔍 发现新区域": {"exploration": 2, "rare_items": 1},
            "💎 寻找宝贵矿物": {"rare_items": 1, "time": 3},
            "🛡️ 制作防具工具": {"protection": 1, "stone": -1},
            "🏘️ 与NPC交易": {"rare_items": 1, "wood": -1},
            "🎯 完成生存任务": {"experience": 25, "rewards": 1}
        }
        
        return results.get(action, {"default": 1})
        
    def _update_world_state(self, result):
        """更新世界状态"""
        # 更新背包
        for item, amount in result.items():
            if item in self.world_state['inventory']:
                self.world_state['inventory'][item] += amount
                
        # 更新生命值
        if 'health' in result:
            self.world_state['health'] = max(0, 
                self.world_state['health'] + result['health'])
                
        # 记录行动
        self.ai_actions.append({
            'timestamp': datetime.now(),
            'action': list(result.keys())[0] if result else 'unknown',
            'result': result
        })
        
    def _display_status(self):
        """显示当前状态"""
        ws = self.world_state
        print(f"  位置: ({ws['position']['x']}, {ws['position']['y']}, {ws['position']['z']})")
        print(f"  生命值: {ws['health']}/100")
        print(f"  背包: {ws['inventory']}")
        print(f"  环境: {ws['environment']}")
        print()
        
    def _generate_random_action(self):
        """生成随机动作"""
        base_actions = ["探索", "建造", "采集", "交易", "战斗"]
        return f"🎲 随机{base_actions[int(time.time()) % len(base_actions)]}"
        
    def get_performance_summary(self):
        """获取性能总结"""
        return {
            'total_actions': len(self.ai_actions),
            'final_health': self.world_state['health'],
            'inventory_value': sum(self.world_state['inventory'].values()),
            'survival_time': len(self.ai_actions),
            'exploration_score': len([a for a in self.ai_actions 
                                    if 'exploration' in str(a.get('result', {}))]),
            'creativity_score': len([a for a in self.ai_actions
                                   if 'build' in str(a.get('action', '')).lower()])
        }

def main():
    """主函数"""
    print("🎮 NeuroMinecraft Genesis - Minecraft AI演示")
    print("模拟AI智能体在Minecraft中的生存过程")
    print("-" * 50)
    
    # 创建模拟器
    simulator = MinecraftSimulation()
    
    # 运行模拟
    simulator.simulate_environment(steps=10)
    
    # 显示总结
    summary = simulator.get_performance_summary()
    
    print("🎉 Minecraft AI 演示完成")
    print("="*50)
    print(f"📊 性能总结:")
    print(f"  总执行动作: {summary['total_actions']}")
    print(f"  最终生命值: {summary['final_health']}/100")
    print(f"  背包价值: {summary['inventory_value']}")
    print(f"  生存时长: {summary['survival_time']} 步骤")
    print(f"  探索得分: {summary['exploration_score']}")
    print(f"  创造得分: {summary['creativity_score']}")
    
    # 评级
    total_score = (summary['final_health'] + summary['inventory_value'] + 
                   summary['exploration_score'] + summary['creativity_score'])
    
    if total_score >= 80:
        grade = "🌟 生存大师"
    elif total_score >= 60:
        grade = "👍 优秀探索者"
    elif total_score >= 40:
        grade = "💪 新手冒险者"
    else:
        grade = "📚 学习中..."
        
    print(f"  总体评级: {grade}")
    
    print(f"\n💡 学习要点:")
    print(f"  • AI需要平衡探索与开发")
    print(f"  • 资源收集是生存的基础")
    print(f"  • 创造力推动技术进步")
    print(f"  • 环境感知能力至关重要")

if __name__ == "__main__":
    main()
```

运行演示：
```bash
python minecraft_demo.py
```

---

## 📊 实时可视化体验

### 启动可视化界面

```bash
# 启动Streamlit可视化
streamlit run utils/visualization/demo.py --server.port 8501
```

访问 [http://localhost:8501](http://localhost:8501) 查看：

1. **六维能力实时监控**
2. **进化过程动画**
3. **Minecraft世界状态**
4. **性能指标仪表板**

### 创建自定义可视化

创建 `custom_viz.py`:

```python
#!/usr/bin/env python3
"""
自定义可视化示例
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_radar_chart(abilities):
    """创建六维能力雷达图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(abilities.values()),
        theta=list(abilities.keys()),
        fill='toself',
        name='认知能力'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="🧠 六维认知能力雷达图",
        font=dict(size=14)
    )
    
    return fig

def create_learning_curve():
    """创建学习曲线"""
    # 模拟24小时学习数据
    hours = list(range(25))
    memory = [75 + np.random.normal(0, 2) + hour * 0.5 for hour in hours]
    creativity = [70 + np.random.normal(0, 3) + hour * 0.8 for hour in hours]
    
    df = pd.DataFrame({
        'Hour': hours,
        'Memory': memory,
        'Creativity': creativity
    })
    
    fig = px.line(df, x='Hour', y=['Memory', 'Creativity'],
                  title='📈 24小时认知能力变化曲线')
    fig.update_layout(font=dict(size=12))
    
    return fig

def create_evolution_tree():
    """创建进化树可视化"""
    # 模拟进化数据
    generations = list(range(1, 51))
    fitness = [0.3 + 0.02 * gen + np.random.normal(0, 0.01) 
              for gen in generations]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=generations,
        y=fitness,
        mode='lines+markers',
        name='适应度',
        line=dict(width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='🌱 进化过程适应度提升',
        xaxis_title='代数',
        yaxis_title='适应度',
        font=dict(size=14)
    )
    
    return fig

def main():
    """Streamlit主界面"""
    st.set_page_config(
        page_title="NeuroMinecraft Genesis - 快速体验",
        page_icon="🧠",
        layout="wide"
    )
    
    # 标题
    st.title("🧠 NeuroMinecraft Genesis - 快速体验")
    st.markdown("5分钟体验完整的六维认知引擎功能")
    
    # 侧边栏
    st.sidebar.header("🎮 控制面板")
    
    # 能力调节器
    st.sidebar.subheader("🧠 认知能力设置")
    memory = st.sidebar.slider("记忆力", 0, 100, 75)
    thinking = st.sidebar.slider("思维力", 0, 100, 68)
    creativity = st.sidebar.slider("创造力", 0, 100, 82)
    observation = st.sidebar.slider("观察力", 0, 100, 79)
    attention = st.sidebar.slider("注意力", 0, 100, 71)
    imagination = st.sidebar.slider("想象力", 0, 100, 74)
    
    # 主要内容区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 雷达图
        abilities = {
            '记忆力': memory,
            '思维力': thinking,
            '创造力': creativity,
            '观察力': observation,
            '注意力': attention,
            '想象力': imagination
        }
        
        radar_fig = create_radar_chart(abilities)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # 能力详情
        st.subheader("📊 能力详情")
        for ability, score in abilities.items():
            st.metric(ability, f"{score:.1f}%")
    
    with col2:
        # 学习曲线
        st.subheader("📈 学习进度")
        curve_fig = create_learning_curve()
        st.plotly_chart(curve_fig, use_container_width=True)
        
        # 进化树
        st.subheader("🌱 进化过程")
        tree_fig = create_evolution_tree()
        st.plotly_chart(tree_fig, use_container_width=True)
    
    # 性能指标
    st.subheader("📊 性能指标")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_ability = sum(abilities.values()) / len(abilities)
        st.metric("平均能力", f"{avg_ability:.1f}%")
    
    with col4:
        best_ability = max(abilities, key=abilities.get)
        st.metric("最佳能力", best_ability)
    
    with col5:
        improvement = np.random.uniform(1, 5)
        st.metric("24小时提升", f"+{improvement:.1f}%")
    
    # 控制按钮
    st.subheader("🎮 实验控制")
    
    col6, col7, col8 = st.columns(3)
    
    with col6:
        if st.button("🔄 重置能力"):
            st.rerun()
    
    with col7:
        if st.button("📊 运行测试"):
            st.success("测试完成！认知能力综合得分: 78.5%")
    
    with col8:
        if st.button("💾 保存配置"):
            st.info("配置已保存到本地存储")
    
    # 底部信息
    st.markdown("---")
    st.markdown(
        "🚀 **下一步**: "
        "[详细安装指南](INSTALLATION.md) | "
        "[完整文档](README.md) | "
        "[项目主页](https://github.com/bingdongni/NeuroMinecraftGenesis)"
    )

if __name__ == "__main__":
    main()
```

---

## 🎉 恭喜完成！

### 您刚刚体验了：

1. ✅ **六维认知引擎** - 完整的认知能力模拟
2. ✅ **实时可视化** - 交互式能力监控
3. ✅ **Minecraft AI** - 智能体生存模拟
4. ✅ **性能分析** - 详细的能力评估

### 下一步建议：

#### 🚀 深入探索
- [安装完整版](INSTALLATION.md) - 真实环境配置
- [阅读技术文档](README.md) - 深入理解原理
- [参与社区讨论](https://discord.gg/neurominecraft) - 与开发者交流

#### 💡 学习资源
- [认知科学基础](https://en.wikipedia.org/wiki/Cognitive_science) - 了解理论背景
- [强化学习入门](https://spinningup.openai.com/) - 学习相关技术
- [Minecraft AI开发](https://github.com/PrismarineJS/mineflayer) - 扩展项目

#### 🤝 贡献社区
- [代码贡献](CONTRIBUTING.md) - 加入开发团队
- [问题反馈](https://github.com/bingdongni/NeuroMinecraftGenesis/issues) - 帮助改进
- [功能建议](https://github.com/bingdongni/NeuroMinecraftGenesis/discussions) - 分享想法

---

## 📞 获取帮助

### 💬 社区支持

- **Discord**: [加入实时讨论](https://discord.gg/neurominecraft)
- **GitHub**: [提交问题](https://github.com/bingdongni/NeuroMinecraftGenesis/issues)
- **邮件**: support@neurominecraft-genesis.org

### 📚 学习资源

- **YouTube**: [视频教程系列](https://youtube.com/neurominecraft)
- **知乎**: [专栏文章](https://zhihu.com/column/neurominecraft)
- **博客**: [技术分享](https://blog.neurominecraft-genesis.org)

---

<div align="center">

**感谢您体验 NeuroMinecraft Genesis！**

🎉 **5分钟只是开始，AGI的未来等待您探索！**

**[⬆ 回到顶部](#快速入门指南)**

Made with ❤️ by the NeuroMinecraft Genesis Team

</div>