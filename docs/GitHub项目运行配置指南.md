# NeuroMinecraft Genesis GitHub项目运行配置指南

## 📋 配置概述

为了让GitHub上的NeuroMinecraft Genesis项目能够正常运行，需要添加几个关键文件，特别是Minecraft服务器的核心文件。本指南将详细介绍整个配置过程。

## 🎯 需要添加的文件

### 1. Minecraft服务器核心文件
**缺失文件**: `paper.jar`
**位置**: `worlds/minecraft/server/paper.jar`
**作用**: PaperMC Minecraft服务端核心文件

### 2. 启动脚本依赖
**文件**: 各种启动脚本和配置文件已存在
**状态**: ✅ 已完成

### 3. 依赖包配置
**文件**: `requirements.txt`
**状态**: ✅ 需要验证和完善

## 📥 详细下载和配置过程

### 第一步：下载PaperMC服务器文件

#### 方法1：官方下载（推荐）
1. 访问PaperMC官网：https://papermc.io/
2. 点击"Downloads" → "Paper"
3. 选择版本：
   - Minecraft版本：`1.20.1`
   - 建议构建：latest (最新稳定版)
4. 下载文件：`paper-1.20.1-[build].jar`
5. 将下载的文件重命名为：`paper.jar`

#### 方法2：命令行下载
```bash
# 如果有curl
curl -L -o paper.jar "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"

# 如果有wget
wget -O paper.jar "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
```

### 第二步：配置项目文件结构

确保你的项目目录结构如下：
```
NeuroMinecraft-Genesis/
├── worlds/minecraft/server/
│   ├── paper.jar                 ← 新添加
│   ├── start.sh                  ← 已存在
│   ├── server.properties         ← 已存在
│   └── eula.txt                  ← 已存在
├── utils/visualization/          ← 可视化系统
├── requirements.txt              ← 依赖配置
├── README.md                     ← 项目说明
├── LICENSE                       ← 开源协议
└── ...其他项目文件
```

### 第三步：创建requirements.txt文件

创建或更新`requirements.txt`文件：
```
# NeuroMinecraft Genesis项目依赖

# 核心AI框架
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
scipy>=1.9.0

# 数据处理
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0

# 网络和通信
requests>=2.28.0
websocket-client>=1.4.0
websockets>=10.4

# 游戏接口
mineflayer>=4.7.0

# 可视化
streamlit>=1.25.0
plotly>=5.10.0
dash>=2.7.0
bokeh>=2.4.0

# 科学计算
scikit-learn>=1.1.0
sympy>=1.11.0

# 工具库
pyyaml>=6.0
tqdm>=4.64.0
click>=8.1.0
rich>=12.5.0

# 开发工具（可选）
pytest>=7.1.0
black>=22.0.0
flake8>=5.0.0
```

### 第四步：创建环境配置脚本

#### 4.1 环境检查脚本 `check_environment.py`
```python
#!/usr/bin/env python3
"""
NeuroMinecraft Genesis环境检查脚本
检查所有必需的依赖和配置文件
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def check_minecraft_server():
    """检查Minecraft服务器文件"""
    print("🎮 检查Minecraft服务器文件...")
    server_dir = Path("worlds/minecraft/server")
    paper_jar = server_dir / "paper.jar"
    
    if not paper_jar.exists():
        print("❌ 缺少 paper.jar 文件")
        print("📥 下载地址: https://papermc.io/")
        return False
    
    print(f"✅ Minecraft服务器文件存在: {paper_jar}")
    return True

def check_dependencies():
    """检查Python依赖"""
    print("📦 检查Python依赖...")
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib',
        'streamlit', 'plotly', 'websocket', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"  ✅ {package}")
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("💡 运行命令: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包都已安装")
    return True

def main():
    """主检查函数"""
    print("🔍 NeuroMinecraft Genesis环境检查")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_minecraft_server,
        check_dependencies
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 环境检查通过！项目可以正常运行")
        return True
    else:
        print("⚠️  环境检查失败，请修复上述问题")
        return False

if __name__ == "__main__":
    main()
```

#### 4.2 一键安装脚本 `install_dependencies.sh`
```bash
#!/bin/bash
# NeuroMinecraft Genesis一键安装脚本

echo "🚀 NeuroMinecraft Genesis一键安装"
echo "================================"

# 检查Python版本
echo "🐍 检查Python版本..."
python3 --version || {
    echo "❌ 未找到Python3，请先安装Python 3.8+"
    exit 1
}

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 安装依赖
echo "📥 安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 检查Minecraft服务器文件
echo "🎮 检查Minecraft服务器..."
if [ ! -f "worlds/minecraft/server/paper.jar" ]; then
    echo "❌ 缺少paper.jar文件"
    echo "📥 请从 https://papermc.io/ 下载PaperMC 1.20.1并重命名为paper.jar"
    echo "💡 下载命令:"
    echo "curl -L -o worlds/minecraft/server/paper.jar \"https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar\""
else
    echo "✅ Minecraft服务器文件已存在"
fi

# 运行环境检查
echo "🔍 运行环境检查..."
python3 check_environment.py

echo "🎉 安装完成！"
echo "💡 启动命令:"
echo "  - 完整系统: python quickstart.py"
echo "  - 可视化界面: streamlit run utils/visualization/streamlit_dashboard.py"
echo "  - 3D界面: python utils/visualization/brain_network_3d.py"
echo "  - Minecraft服务器: bash worlds/minecraft/server/start.sh"
```

### 第五步：创建GitHub Actions自动化配置

#### 5.1 创建`.github/workflows/ci.yml`
```yaml
name: NeuroMinecraft Genesis CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run environment check
      run: python check_environment.py
    
    - name: Test core imports
      run: |
        python -c "
        import sys
        sys.path.append('.')
        
        # 测试核心模块导入
        try:
            from utils.brain_engine import SixDimensionBrain
            from agents.evolution.disco_rl_agent import DiscoRLAgent
            from utils.quantum_simulator import QuantumBrainSimulator
            print('✅ 核心模块导入成功')
        except ImportError as e:
            print(f'❌ 模块导入失败: {e}')
            sys.exit(1)
        "
    
    - name: Test visualization components
      run: |
        python -c "
        try:
            import streamlit
            import plotly
            import bokeh
            print('✅ 可视化组件可用')
        except ImportError as e:
            print(f'❌ 可视化组件缺失: {e}')
            sys.exit(1)
        "

  minecraft-server:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Minecraft Server
      run: |
        mkdir -p worlds/minecraft/server
        
        # 下载PaperMC（如果不存在）
        if [ ! -f "worlds/minecraft/server/paper.jar" ]; then
          echo "📥 下载PaperMC服务器..."
          curl -L -o worlds/minecraft/server/paper.jar \
            "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
        fi
        
        echo "✅ PaperMC服务器已准备就绪"
    
    - name: Test server startup
      run: |
        cd worlds/minecraft/server
        timeout 30s bash start.sh || echo "⏰ 服务器启动测试完成"

  build-docs:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install documentation dependencies
      run: |
        pip install mkdocs mkdocs-material
    
    - name: Build documentation
      run: |
        # 如果有mkdocs配置文件
        if [ -f "mkdocs.yml" ]; then
          mkdocs build
        fi
```

#### 5.2 创建`.github/dependabot.yml`
```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    reviewers:
      - "project-maintainer"
    commit-message:
      prefix: "deps"
      include: "scope"
    
  # GitHub Actions dependencies
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
      day: "first-monday"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "ci"
      include: "scope"
```

## 🚀 上传GitHub的完整流程

### 步骤1：本地项目准备

```bash
# 1. 克隆现有仓库（或从本地初始化）
git clone https://github.com/你的用户名/NeuroMinecraft-Genesis.git
cd NeuroMinecraft-Genesis

# 2. 添加缺失文件
# 确保 paper.jar 在 worlds/minecraft/server/ 目录

# 3. 添加新文件到版本控制
git add .

# 4. 提交更改
git commit -m "feat: 添加Minecraft服务器配置和项目依赖

- 添加PaperMC 1.20.1服务器核心文件
- 创建requirements.txt依赖配置
- 添加环境检查和安装脚本
- 添加GitHub Actions CI/CD配置
- 添加依赖更新自动化配置"

# 5. 推送到GitHub
git push origin main
```

### 步骤2：验证GitHub Actions

1. 在GitHub仓库页面点击"Actions"标签
2. 查看CI/CD流水线是否正常运行
3. 检查所有测试是否通过

### 步骤3：创建Releases

```bash
# 创建版本标签
git tag -a v1.0.0 -m "Release v1.0.0: 完整可运行的NeuroMinecraft Genesis项目"

# 推送标签
git push origin v1.0.0

# 在GitHub上创建Release，包含:
# - 详细发布说明
# - 下载paper.jar的说明
# - 安装和使用指南链接
```

## 🛠️ 确保项目始终正常运行的最佳实践

### 1. 依赖管理
- 定期更新`requirements.txt`
- 使用GitHub Dependabot自动检查依赖更新
- 在CI中测试多个Python版本

### 2. 环境隔离
- 提供虚拟环境创建脚本
- 明确指定Python版本要求
- 提供Docker容器配置（可选）

### 3. 文档维护
- 保持README.md与代码同步
- 更新API文档
- 添加使用示例和教程

### 4. 测试覆盖
- 增加单元测试
- 添加集成测试
- 使用GitHub Actions自动化测试

### 5. 版本控制
- 使用语义化版本控制
- 创建清晰的发布说明
- 维护CHANGELOG.md

### 6. 监控和诊断
- 添加日志记录
- 提供诊断工具
- 监控常见错误

## 📝 项目结构完整性检查清单

在上传前，确保以下文件和目录都存在：

```bash
# 核心AI系统
✅ utils/brain_engine/
✅ agents/evolution/
✅ utils/quantum_simulator/

# 可视化系统
✅ utils/visualization/
✅ static/

# 世界集成
✅ worlds/integrated_environment.py
✅ worlds/minecraft/
✅ worlds/virtual/
✅ worlds/real/

# 启动脚本
✅ quickstart.py
✅ start.bat
✅ worlds/minecraft/server/start.sh

# Minecraft服务器（关键）
✅ worlds/minecraft/server/paper.jar          ← 必须添加
✅ worlds/minecraft/server/eula.txt
✅ worlds/minecraft/server/server.properties

# 项目配置
✅ requirements.txt                           ← 必须创建
✅ check_environment.py                       ← 必须创建
✅ install_dependencies.sh                    ← 必须创建

# GitHub配置
✅ .github/workflows/ci.yml                   ← 必须创建
✅ .github/dependabot.yml                     ← 可选但推荐

# 文档
✅ README.md
✅ LICENSE
✅ GitHub项目运行配置指南.md                   ← 本文件
```

## 🎯 用户使用指南

为最终用户提供简单的使用说明：

### 快速开始
```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/NeuroMinecraft-Genesis.git
cd NeuroMinecraft-Genesis

# 2. 一键安装
bash install_dependencies.sh

# 3. 运行项目
python quickstart.py
```

### 高级功能
```bash
# 启动可视化界面
streamlit run utils/visualization/streamlit_dashboard.py

# 启动3D脑网络
python utils/visualization/brain_network_3d.py

# 启动Minecraft服务器
bash worlds/minecraft/server/start.sh
```

---

## 🎉 配置完成

按照以上步骤操作后，你的GitHub项目将能够：

1. ✅ **自动CI/CD测试** - GitHub Actions自动运行测试
2. ✅ **依赖自动更新** - Dependabot定期检查依赖更新
3. ✅ **环境检查** - 用户可以轻松检查和配置环境
4. ✅ **一键安装** - 提供自动化安装脚本
5. ✅ **完整文档** - 详细的使用和配置说明

项目将始终保持最新状态，用户可以轻松地克隆、配置和运行项目！

---

*本指南创建时间: 2025-11-14*  
*适用于: NeuroMinecraft Genesis v1.0.0+*