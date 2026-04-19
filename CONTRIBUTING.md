# 贡献指南

**感谢您对 NeuroMinecraft Genesis 项目的兴趣！我们欢迎各种形式的贡献。**

---

## 🤝 如何贡献

### 💡 贡献类型

我们欢迎以下类型的贡献：

- 🐛 **Bug修复** - 修复代码中的错误
- ✨ **新功能** - 添加新特性或改进
- 📝 **文档** - 改进文档、教程或示例
- 🧪 **测试** - 添加或改进测试用例
- 🎨 **界面** - 改进可视化或用户界面
- 💬 **讨论** - 参与问题讨论和社区建设

---

## 🚀 快速开始

### 1. Fork 仓库

点击页面右上角的 "Fork" 按钮，将仓库复制到您的GitHub账户。

### 2. 克隆到本地

```bash
git clone https://github.com/YOUR_USERNAME/NeuroMinecraftGenesis.git
cd NeuroMinecraftGenesis
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 4. 进行更改

按照我们的代码规范进行开发，确保代码质量和可读性。

### 5. 提交更改

```bash
git add .
git commit -m "feat: add new cognitive dimension tracker"
```

### 6. 推送分支

```bash
git push origin feature/your-feature-name
```

### 7. 创建 Pull Request

在GitHub上创建Pull Request，填写详细的描述信息。

---

## 📋 开发环境设置

### 系统要求

- Python 3.11+
- Node.js 18+
- Git
- 推荐: 16GB+ RAM

### 自动安装 (Windows)

```bash
# 在项目根目录运行
.\install.bat
```

### 手动安装 (Linux/MacOS)

```bash
# 克隆仓库
git clone https://github.com/bingdongni/NeuroMinecraftGenesis.git
cd NeuroMinecraftGenesis

# 安装Python依赖
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 安装Node.js依赖
npm install

# 运行开发服务器
npm run dev
```

### 验证安装

```bash
# 运行基础测试
python tests/test_basic_functionality.py

# 启动可视化界面
streamlit run utils/visualization/dashboard.py
```

---

## 📝 代码规范

### Python 代码规范

我们遵循 PEP 8 标准，并使用以下工具确保代码质量：

#### 格式规范
```python
# 好的示例
def calculate_cognitive_score(
    memory_vector: torch.Tensor,
    attention_weights: torch.Tensor,
    creativity_threshold: float = 0.5
) -> float:
    """
    计算六维认知能力综合得分
    
    Args:
        memory_vector: 记忆向量 (512维)
        attention_weights: 注意力权重 (512维)
        creativity_threshold: 创造力阈值
        
    Returns:
        认知能力得分 (0-1范围)
    """
    # 实现逻辑
    pass
```

#### 不推荐的代码
```python
# 避免这种写法
def f(v,w,t):
    # 缺少类型提示和文档
    pass
```

#### 导入规范
```python
# 标准库导入
import os
import sys
from pathlib import Path

# 第三方库导入
import numpy as np
import torch
import streamlit as st

# 本地模块导入
from core.brain.hippocampus import HippocampusMemory
from utils.visualization.dashboard import CognitiveDashboard
```

### Git 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 标准：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 提交类型
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更改
- `style`: 代码格式化
- `refactor`: 重构代码
- `test`: 添加测试
- `chore`: 构建或工具更改

#### 示例
```bash
# 新功能
git commit -m "feat: add quantum decision circuit"

# Bug修复
git commit -m "fix: resolve memory consolidation bug"

# 文档更新
git commit -m "docs: update installation guide"
```

---

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_memory_system.py

# 生成覆盖率报告
python -m pytest --cov=core tests/
```

### 编写测试

```python
import pytest
import torch
from core.brain.hippocampus import HippocampusMemory

class TestHippocampusMemory:
    def test_memory_storage(self):
        """测试记忆存储功能"""
        memory = HippocampusMemory(capacity=1000)
        
        # 创建测试数据
        state = torch.randn(512)
        action = "move_forward"
        reward = 1.0
        
        # 存储记忆
        memory_id = memory.store(state, action, reward)
        
        # 验证存储
        assert memory_id is not None
        assert len(memory.memories) == 1
    
    def test_memory_retrieval(self):
        """测试记忆检索功能"""
        memory = HippocampusMemory(capacity=1000)
        
        # 存储测试记忆
        state = torch.randn(512)
        memory.store(state, "test_action", 1.0)
        
        # 检索记忆
        retrieved = memory.retrieve(state, top_k=5)
        
        assert len(retrieved) > 0
        assert all("action" in item for item in retrieved)
```

### 测试覆盖率要求

- 新功能测试覆盖率 > 90%
- Bug修复必须包含测试用例
- 关键模块测试覆盖率 > 95%

---

## 📚 文档贡献

### 文档类型

- **API文档**: 函数、类的详细说明
- **教程**: 逐步指导文档
- **示例**: 代码使用示例
- **说明**: 设计思路和技术细节

### 文档格式

#### 函数文档
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    简短描述函数功能
    
    详细描述函数的工作原理、算法实现等。
    
    Args:
        param1: 参数1的说明
        param2: 参数2的说明
        
    Returns:
        返回值的说明
        
    Raises:
        ExceptionType: 异常情况的说明
        
    Examples:
        >>> function_name(value1, value2)
        expected_output
        
    Note:
        重要注意事项或限制条件
    """
```

#### README章节
```markdown
## 功能特性

### 核心算法
- 算法1：详细说明
- 算法2：详细说明

### 使用示例
```python
# 导入模块
from core.brain.hippocampus import HippocampusMemory

# 创建实例
memory = HippocampusMemory(capacity=1000)

# 使用示例
result = memory.process(data)
print(result)
```
```

---

## 🎨 可视化贡献

### 界面设计原则

1. **简洁明了**: 界面简洁，避免复杂操作
2. **响应式**: 适配不同屏幕尺寸
3. **实时性**: 数据更新及时
4. **美观性**: 使用一致的配色方案

### 颜色规范

```css
/* 主色调 */
--primary-color: #1f77b4;
--secondary-color: #ff7f0e;
--success-color: #2ca02c;
--warning-color: #d62728;
--info-color: #9467bd;

/* 背景色 */
--bg-dark: #2c3e50;
--bg-light: #ecf0f1;
--bg-white: #ffffff;
```

### 组件规范

```python
import streamlit as st
import plotly.graph_objects as go

class CognitiveVisualization:
    def create_ability_chart(self, abilities: dict):
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
            title="六维认知能力雷达图"
        )
        
        return fig
```

---

## 🐛 Bug报告指南

### 报告模板

在 GitHub Issues 中使用以下模板：

```markdown
## Bug 描述
简要描述bug现象

## 复现步骤
1. 打开...
2. 点击...
3. 滚动到...
4. 看到错误

## 期望行为
描述您期望发生什么

## 实际行为
描述实际发生了什么

## 屏幕截图
如果适用，添加屏幕截图

## 环境信息
- OS: [e.g. Windows 11]
- Python版本: [e.g. 3.11.5]
- 项目版本: [e.g. v1.0.0]
- 显卡: [e.g. NVIDIA RTX 3080]

## 额外信息
添加任何其他关于问题的信息
```

### Bug优先级

- **Critical**: 系统崩溃，数据丢失
- **High**: 主要功能无法使用
- **Medium**: 功能受限但有解决方案
- **Low**: 轻微的不便或美化问题

---

## 💡 功能请求指南

### 请求模板

```markdown
## 功能描述
简要描述您希望的功能

## 问题背景
这个功能解决了什么问题？

## 期望解决方案
您希望这个功能如何工作？

## 替代方案
您考虑过其他解决方案吗？

## 额外信息
添加任何其他相关信息
```

### 功能优先级

我们根据以下因素评估功能优先级：

1. **社区需求**: 多少用户需要此功能
2. **技术可行性**: 实现难度和时间成本
3. **项目目标**: 与AGI目标的一致性
4. **影响力**: 对项目发展的潜在影响

---

## 🔍 代码审查指南

### 审查检查清单

#### 代码质量
- [ ] 代码遵循项目规范
- [ ] 函数和变量命名清晰
- [ ] 适当的注释和文档
- [ ] 没有明显的性能问题

#### 功能正确性
- [ ] 代码实现符合需求
- [ ] 边界情况处理正确
- [ ] 错误处理完善
- [ ] 测试用例覆盖充分

#### 可维护性
- [ ] 模块解耦合理
- [ ] 接口设计清晰
- [ ] 便于扩展和维护
- [ ] 依赖关系明确

### 审查评论示例

```python
# 好的评论
"建议将这个魔法数字提取为常量，提高可读性"
"考虑添加类型提示，使接口更清晰"
"这个函数有点长，建议拆分为更小的函数"

# 避免的评论
"写得不好"
"这不对"
"重写"
```

---

## 📊 性能贡献

### 性能优化指南

1. **测量先行**: 使用分析工具找出瓶颈
2. **渐进优化**: 一次只优化一个方面
3. **保持可读性**: 优化不能损害代码质量
4. **充分测试**: 确保优化后功能正确

### 性能测试

```python
import time
import cProfile
from memory_profiler import profile

@profile
def performance_critical_function():
    """性能关键函数"""
    start_time = time.time()
    
    # 执行核心逻辑
    result = complex_calculation()
    
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.4f}秒")
    
    return result

# 使用示例
if __name__ == "__main__":
    cProfile.run('performance_critical_function()')
```

---

## 🌍 国际化贡献

### 支持语言

当前支持：
- 🇨🇳 简体中文 (主要)
- 🇺🇸 English
- 🇯🇵 日本語 (计划中)
- 🇩🇪 Deutsch (计划中)

### 添加新语言

1. 在 `i18n/` 目录下创建语言文件
2. 翻译所有界面文本
3. 更新配置以包含新语言
4. 添加相应的测试用例

### 语言文件格式

```json
{
  "ui": {
    "title": "标题",
    "start_button": "开始",
    "stop_button": "停止"
  },
  "messages": {
    "welcome": "欢迎使用",
    "error": "发生错误"
  },
  "cognitive_dimensions": {
    "memory": "记忆",
    "thinking": "思维",
    "creativity": "创造",
    "observation": "观察",
    "attention": "注意",
    "imagination": "想象"
  }
}
```

---

## 🎓 学习资源

### 推荐学习路径

#### 初学者
1. **Python基础**: [官方教程](https://docs.python.org/3/tutorial/)
2. **机器学习**: [Andrew Ng课程](https://www.coursera.org/learn/machine-learning)
3. **深度学习**: [Fast.ai](https://www.fast.ai/)
4. **强化学习**: [Spinning Up](https://spinningup.openai.com/)

#### 进阶者
1. **认知科学**: 《认知心理学》
2. **神经科学**: 《大脑如何工作》
3. **进化算法**: DEAP框架文档
4. **量子计算**: Qiskit教材

#### 专家级
1. **AGI研究论文**: 阅读最新arXiv论文
2. **开源贡献**: 参与其他AGI项目
3. **会议参与**: NeurIPS, ICML, ICLR等
4. **研究合作**: 与顶级实验室合作

### 内部资源

- 📖 **技术文档**: `/docs/technical/`
- 🎥 **视频教程**: `/docs/tutorials/`
- 💬 **开发者群**: Discord/微信群
- 📝 **设计笔记**: `/docs/design/`

---

## 🏆 贡献者认可

### 贡献等级

#### 🌱 新手贡献者 (1-5贡献)
- GitHub个人资料徽章
- 项目邮件列表加入
- 贡献者证书

#### 🌿 活跃贡献者 (6-20贡献)
- GitHub组织邀请
- 项目wiki编辑权限
- 月度贡献者感谢

#### 🌳 核心贡献者 (21-50贡献)
- 代码审查权限
- 发布管理权限
- 年度聚会邀请

#### 🌟 维护者 (50+贡献)
- 项目共同维护者
- 技术决策投票权
- 商业合作参与权

### 致谢方式

在每个版本发布时，我们会在 `CHANGELOG.md` 中特别感谢贡献者：

```markdown
## 贡献者

感谢以下贡献者的出色工作：

- **@username** - 贡献描述
- **@username** - 贡献描述
- **@username** - 贡献描述

特别感谢:
- 核心维护者团队
- 早期测试者
- 社区支持者
```

---

## 📞 获取帮助

### 联系方式

- 💬 **Discord**: [加入社区](https://discord.gg/neurominecraft)
- 📧 **邮件**: dev@neurominecraft-genesis.org
- 🐦 **Twitter**: [@NeuroMinecraft](https://twitter.com/NeuroMinecraft)
- 🐛 **Bug报告**: [GitHub Issues](https://github.com/bingdongni/NeuroMinecraftGenesis/issues)

### 问答社区

- **Stack Overflow**: 使用标签 `neurominecraft-genesis`
- **Reddit**: r/NeuroMinecraftGenesis
- **知乎**: 专栏"AGI探索之路"

### 在线办公时间

每周四晚上8点-10点 (GMT+8)，核心团队在线答疑

---

## 📜 行为准则

### 我们的承诺

为了促进开放和欢迎的环境，我们作为贡献者和维护者承诺，无论年龄、体型、残疾、民族、性别认同和表达、经验水平、教育、社会经济地位、国籍、个人外貌、种族、宗教信仰或性认同和取向，都为每个人提供无骚扰的体验。

### 不可接受的行为

- 使用性化的语言或图像，以及不受欢迎的性关注或追求
- 恶意评论、侮辱/贬损评论、个人或政治攻击
- 公开或私下的骚扰
- 未经明确许可，发布他人的私人信息，如物理或电子地址
- 在专业环境中可能被合理认为不合适的其他行为

### 执法

可以通过 dev@neurominecraft-genesis.org 向项目团队报告违反行为准则的事件。所有的投诉都将得到审查和调查。

---

## 🎉 开始贡献

准备好开始贡献了吗？

1. 选择一个您感兴趣的问题
2. 在GitHub上留言说明您的计划
3. 开始开发
4. 提交Pull Request
5. 等待代码审查
6. 合并到主分支

**让我们一起构建AGI的未来！**

---

<div align="center">

**感谢您对 NeuroMinecraft Genesis 的贡献！**

**[⬆ 回到顶部](#贡献指南)**

Made with ❤️ by the NeuroMinecraft Genesis Team

</div>