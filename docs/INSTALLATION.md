# 🛠️ 详细安装指南

**完整的 NeuroMinecraft Genesis 环境配置指南**

---

## 📋 目录

1. [系统要求](#系统要求)
2. [快速安装](#快速安装)
3. [手动安装](#手动安装)
4. [开发环境配置](#开发环境配置)
5. [Minecraft服务器设置](#minecraft服务器设置)
6. [GPU加速配置](#gpu加速配置)
7. [故障排除](#故障排除)
8. [性能优化](#性能优化)

---

## 💻 系统要求

### 最低要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10/11, Ubuntu 18.04+, macOS 10.15+ | Windows 11, Ubuntu 20.04+, macOS 12+ |
| **内存** | 8GB RAM | 16GB+ RAM |
| **存储** | 10GB 可用空间 | 50GB+ SSD |
| **处理器** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7+ |
| **网络** | 宽带连接 (下载依赖) | 宽带连接 |

### 软件依赖

| 软件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| **Python** | 3.8+ | 3.11+ |
| **Node.js** | 14+ | 18+ |
| **Git** | 2.20+ | 最新版 |
| **Java** | 8+ | 17+ (Minecraft服务器) |

---

## ⚡ 快速安装

### 一键安装脚本

#### Windows 用户

创建 `install.bat` 文件：

```batch
@echo off
echo ========================================
echo NeuroMinecraft Genesis 安装程序
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.11+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Node.js，请先安装Node.js 18+
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

echo [信息] 环境检查通过，开始安装...

:: 创建虚拟环境
echo [步骤 1/5] 创建Python虚拟环境...
python -m venv neurominecraft_env
call neurominecraft_env\Scripts\activate.bat

:: 升级pip
echo [步骤 2/5] 升级pip...
python -m pip install --upgrade pip

:: 安装核心依赖
echo [步骤 3/5] 安装核心依赖...
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy pandas matplotlib seaborn
pip install streamlit plotly

:: 安装AI/ML依赖
echo [步骤 4/5] 安装AI/ML依赖...
pip install transformers datasets tokenizers
pip install scikit-learn gymnasium stable-baselines3

:: 安装Minecraft相关
echo [步骤 5/5] 安装Minecraft AI依赖...
npm install mineflayer mineflayer-pathfinder mineflayer-collectblock vec3 ws

echo.
echo ========================================
echo ✅ 安装完成！
echo ========================================
echo.
echo 启动命令:
echo   激活环境: neurominecraft_env\Scripts\activate
echo   运行演示: streamlit run docs/QUICK_START.py
echo.
pause
```

#### Linux/macOS 用户

创建 `install.sh` 文件：

```bash
#!/bin/bash

echo "========================================"
echo "NeuroMinecraft Genesis 安装程序"
echo "========================================"
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.11+"
    exit 1
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "[错误] 未找到Node.js，请先安装Node.js 18+"
    echo "安装命令: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs"
    exit 1
fi

echo "[信息] 环境检查通过，开始安装..."

# 创建虚拟环境
echo "[步骤 1/5] 创建Python虚拟环境..."
python3 -m venv neurominecraft_env
source neurominecraft_env/bin/activate

# 升级pip
echo "[步骤 2/5] 升级pip..."
python3 -m pip install --upgrade pip

# 安装核心依赖
echo "[步骤 3/5] 安装核心依赖..."
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy pandas matplotlib seaborn
pip install streamlit plotly

# 安装AI/ML依赖
echo "[步骤 4/5] 安装AI/ML依赖..."
pip install transformers datasets tokenizers
pip install scikit-learn gymnasium stable-baselines3

# 安装Minecraft相关
echo "[步骤 5/5] 安装Minecraft AI依赖..."
npm install mineflayer mineflayer-pathfinder mineflayer-collectblock vec3 ws

echo
echo "========================================"
echo "✅ 安装完成！"
echo "========================================"
echo
echo "启动命令:"
echo "  激活环境: source neurominecraft_env/bin/activate"
echo "  运行演示: streamlit run docs/QUICK_START.py"
echo
```

运行安装脚本：

```bash
# Windows
install.bat

# Linux/macOS
chmod +x install.sh
./install.sh
```

---

## 🔧 手动安装

### 1. Python环境配置

#### 安装Python 3.11

**Windows**:
```bash
# 下载Python 3.11: https://www.python.org/downloads/
# 安装时勾选 "Add Python to PATH"
```

**Ubuntu/Debian**:
```bash
# 添加deadsnakes PPA
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**macOS**:
```bash
# 使用Homebrew
brew install python@3.11

# 或从官网下载: https://www.python.org/downloads/macos/
```

#### 创建虚拟环境

```bash
# 进入项目目录
cd NeuroMinecraftGenesis

# 创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 验证虚拟环境
which python  # 应该显示 venv 路径
python --version  # 应该显示 3.11.x
```

### 2. 安装Python依赖

#### 核心依赖

```bash
# 升级pip
pip install --upgrade pip setuptools wheel

# 科学计算基础
pip install numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0

# 机器学习框架
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn>=1.0.0

# 数据可视化
pip install matplotlib>=3.5.0 seaborn>=0.11.0 plotly>=5.0.0

# Web界面
pip install streamlit>=1.20.0

# 强化学习
pip install gymnasium>=0.26.0 stable-baselines3>=2.0.0

# 自然语言处理
pip install transformers>=4.20.0 datasets>=2.0.0 tokenizers>=0.12.0
```

#### 可选依赖

```bash
# 量子计算 (可选)
pip install qiskit>=0.40.0

# 类脑计算 (可选)
pip install nengo>=4.6.0 nengo-dl>=3.1.0

# 高性能计算 (可选)
pip install numba>=0.56.0

# 分布式训练 (可选)
pip install ray[default]>=2.2.0

# 监控和日志 (可选)
pip install wandb>=0.13.0 tensorboard>=2.10.0
```

### 3. 安装Node.js依赖

```bash
# 初始化npm项目
npm init -y

# 安装Minecraft AI相关包
npm install mineflayer@4.10.0
npm install mineflayer-pathfinder@2.4.0
npm install mineflayer-collectblock@1.1.0
npm install vec3@0.1.8
npm install ws@8.13.0
npm install mineflayer-npc@2.0.0
```

### 4. 验证安装

创建 `test_installation.py`:

```python
#!/usr/bin/env python3
"""
安装验证脚本
"""

import sys
import importlib
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    else:
        print("✅ Python版本满足要求")
        return True

def check_package(package_name, import_name=None):
    """检查包安装状态"""
    try:
        if import_name:
            importlib.import_module(import_name)
        else:
            importlib.import_module(package_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def check_node_packages():
    """检查Node.js包"""
    try:
        result = subprocess.run(['npm', 'list', '--depth=0'], 
                              capture_output=True, text=True)
        if 'mineflayer' in result.stdout:
            print("✅ Node.js依赖已安装")
            return True
        else:
            print("❌ Node.js依赖未安装")
            return False
    except FileNotFoundError:
        print("❌ npm未找到")
        return False

def main():
    """主函数"""
    print("🔍 NeuroMinecraft Genesis 安装验证")
    print("=" * 50)
    
    all_good = True
    
    # 检查Python版本
    if not check_python_version():
        all_good = False
    
    print()
    
    # 检查核心包
    core_packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('streamlit', 'streamlit'),
        ('transformers', 'transformers'),
        ('scikit-learn', 'sklearn'),
        ('plotly', 'plotly'),
    ]
    
    print("📦 Python包检查:")
    for package, import_name in core_packages:
        if not check_package(package, import_name):
            all_good = False
    
    print()
    
    # 检查可选包
    optional_packages = [
        ('qiskit', 'qiskit'),
        ('nengo', 'nengo'),
        ('numba', 'numba'),
        ('ray', 'ray'),
    ]
    
    print("📦 可选包检查:")
    for package, import_name in optional_packages:
        check_package(package, import_name)  # 可选包不影响主功能
    
    print()
    
    # 检查Node.js依赖
    print("📦 Node.js包检查:")
    if not check_node_packages():
        all_good = False
    
    print()
    
    # 结果汇总
    print("=" * 50)
    if all_good:
        print("🎉 安装验证通过！可以开始使用项目了")
        print()
        print("🚀 快速开始:")
        print("  python docs/QUICK_START.py")
        print("  streamlit run docs/QUICK_START.py")
    else:
        print("❌ 安装验证失败，请检查上述错误")
        print()
        print("💡 解决建议:")
        print("  1. 检查错误信息并安装缺失的包")
        print("  2. 参考本文档重新安装")
        print("  3. 在GitHub Issues中寻求帮助")

if __name__ == "__main__":
    main()
```

运行验证：

```bash
python test_installation.py
```

---

## 🛠️ 开发环境配置

### 推荐IDE配置

#### VS Code配置

创建 `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
```

#### Git配置

创建 `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
neurominecraft_env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/minecraft_episodes/*.dat
data/brain_scans/*.h5
data/evolution_logs/*.json

# Models
models/checkpoints/*.pt
models/genomes/*.pkl

# Logs
logs/
*.log

# Environment variables
.env
.env.local
.env.development
.env.test
.env.production

# OS
.DS_Store
Thumbs.db

# Minecraft
worlds/minecraft/server/world/
worlds/minecraft/server/world_nether/
worlds/minecraft/server/world_the_end/
```

### 测试环境

创建 `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    cognitive: marks tests related to cognitive functions
    minecraft: marks tests requiring Minecraft
```

### 代码格式化

创建 `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.coverage.run]
source = ["core", "agents", "utils"]
omit = [
    "*/tests/*",
    "*/test_*",
    "setup.py",
    "*/conftest.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

---

## 🎮 Minecraft服务器设置

### 1. 下载Minecraft服务器

```bash
# 创建服务器目录
mkdir -p worlds/minecraft/server
cd worlds/minecraft/server

# 下载PaperMC服务器 (高性能Minecraft服务端)
wget https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/196/downloads/paper-1.20.1-196.jar -O paper.jar

# 或在Windows上使用PowerShell
Invoke-WebRequest -Uri "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/196/downloads/paper-1.20.1-196.jar" -OutFile "paper.jar"
```

### 2. 服务器配置

创建 `start.bat` (Windows):

```batch
@echo off
java -Xmx4G -Xms2G -jar paper.jar --nogui --no-jline
pause
```

创建 `start.sh` (Linux/macOS):

```bash
#!/bin/bash
java -Xmx4G -Xms2G -jar paper.jar --nogui --no-jline
```

创建 `server.properties`:

```properties
# Minecraft服务器配置
gamemode=survival
difficulty=normal
spawn-protection=16
max-players=20
online-mode=false
enable-command-block=true
enable-query=false
enable-rcon=false
enable-status=false

# 世界设置
level-name=world
level-type=minecraft\\normal
generator-settings={}

# 生物和怪物
spawn-monsters=true
spawn-animals=true
spawn-npcs=true

# 游戏规则
do-daylight-cycle=true
do-weather-cycle=true
do-mob-spawning=true
do-insomnia=true

# 服务器性能
view-distance=16
simulation-distance=16
entity-broadcast-range-percentage=100

# AI智能体友好设置
broadcast-rcon-to-ops=true
broadcast-console-to-ops=true
```

### 3. 首次启动

```bash
# 启动服务器 (首次运行会自动生成世界)
java -Xmx2G -Xms2G -jar paper.jar --nogui
```

首次启动后，会生成 `eula.txt` 文件，需要同意EULA：

```text
# 编辑 eula.txt
eula=true
```

### 4. 安装Citizens插件 (NPC系统)

```bash
# 下载Citizens插件
wget https://github.com/CitizensDev/Citizens/releases/download/2.0.30/Citizens-2.0.30.jar -O plugins/Citizens-2.0.30.jar

# 重启服务器后会自动加载插件
```

### 5. 验证安装

创建 `test_minecraft_server.py`:

```python
#!/usr/bin/env python3
"""
Minecraft服务器连接测试
"""

import asyncio
from mineflayer import MinecraftData, mineflayer

async def test_server_connection():
    """测试服务器连接"""
    try:
        # 连接到本地服务器
        bot = mineflayer.create_bot({
            'host': 'localhost',
            'port': 25565,
            'username': 'NeuroMinecraftAI',
        })
        
        # 等待连接
        await bot.wait_until_ready()
        
        print("✅ Minecraft服务器连接成功")
        print(f"服务器版本: {bot.version}")
        print(f"在线玩家: {len(bot.players)}")
        
        # 测试基本操作
        await bot.chat.send('/gamemode survival @s')
        
        # 移动到安全位置
        await bot.moveto.move_to(0, 64, 0)
        
        print("✅ 基本操作测试成功")
        
        # 断开连接
        bot.quit("测试完成")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请检查Minecraft服务器是否运行在localhost:25565")

if __name__ == "__main__":
    asyncio.run(test_server_connection())
```

运行测试：

```bash
python test_minecraft_server.py
```

---

## 🏃‍♂️ GPU加速配置

### CUDA支持 (可选)

如果您有NVIDIA GPU，可以启用CUDA加速：

#### 检查GPU支持

```bash
# 检查CUDA版本
nvidia-smi

# 检查cuDNN
python -c "import torch; print(torch.backends.cudnn.enabled)"
```

#### 安装CUDA版本的PyTorch

```bash
# 卸载CPU版本
pip uninstall torch torchvision

# 安装CUDA版本 (根据您的CUDA版本选择)
# CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 12.1
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 验证GPU安装

创建 `test_gpu.py`:

```python
#!/usr/bin/env python3
"""
GPU加速测试
"""

import torch

def test_gpu_support():
    """测试GPU支持"""
    print("🖥️ GPU加速测试")
    print("=" * 30)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print("✅ CUDA可用")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # GPU性能测试
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        
        # 矩阵乘法测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        y = torch.mm(x, x)
        end_time.record()
        
        torch.cuda.synchronize()
        
        gpu_time = start_time.elapsed_time(end_time)
        print(f"GPU矩阵乘法时间: {gpu_time:.2f}ms")
        
        # CPU对比
        x_cpu = x.cpu()
        start_time.record()
        y_cpu = torch.mm(x_cpu, x_cpu)
        end_time.record()
        
        cpu_time = start_time.elapsed_time(end_time)
        print(f"CPU矩阵乘法时间: {cpu_time:.2f}ms")
        print(f"GPU加速比: {cpu_time/gpu_time:.1f}x")
        
    else:
        print("❌ CUDA不可用，将使用CPU")
        
        # CPU性能基准
        x = torch.randn(1000, 1000)
        
        import time
        start_time = time.time()
        y = torch.mm(x, x)
        end_time = time.time()
        
        cpu_time = (end_time - start_time) * 1000
        print(f"CPU矩阵乘法时间: {cpu_time:.2f}ms")

if __name__ == "__main__":
    test_gpu_support()
```

运行测试：

```bash
python test_gpu.py
```

---

## 🚨 故障排除

### 常见问题及解决方案

#### 1. Python相关问题

**问题**: `ModuleNotFoundError: No module named 'torch'`

**解决方案**:
```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 重新安装PyTorch
pip uninstall torch torchvision
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**问题**: `Microsoft Visual C++ 14.0 is required`

**解决方案** (Windows):
```bash
# 安装 Microsoft C++ Build Tools
# 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 或安装简化版
pip install --upgrade pip setuptools
pip install --only-binary=all numpy scipy
```

#### 2. Node.js相关问题

**问题**: `npm install` 失败

**解决方案**:
```bash
# 清理npm缓存
npm cache clean --force

# 更新npm版本
npm install -g npm@latest

# 使用yarn替代 (推荐)
npm install -g yarn
yarn install
```

**问题**: `mineflayer` 连接失败

**解决方案**:
```python
# 检查Minecraft服务器状态
import socket

def check_minecraft_port():
    host = 'localhost'
    port = 25565
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ Minecraft服务器端口开放")
        else:
            print("❌ Minecraft服务器端口未开放")
            print("请启动Minecraft服务器或检查防火墙设置")
    except Exception as e:
        print(f"连接检查失败: {e}")

check_minecraft_port()
```

#### 3. 性能问题

**问题**: 内存不足

**解决方案**:
```python
# 减少批处理大小
BATCH_SIZE = 16  # 原来是32

# 使用更小的模型
model_name = "microsoft/DialoGPT-small"  # 原来是medium

# 启用内存映射
torch.set_grad_enabled(False)

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**问题**: 推理速度慢

**解决方案**:
```python
# 启用混合精度推理
from torch.cuda.amp import autocast

with autocast():
    result = model(input_data)

# 使用TensorRT优化 (GPU)
import torch_tensorrt
```

#### 4. 安装权限问题

**问题**: `Permission denied`

**解决方案** (Linux/macOS):
```bash
# 使用用户安装
pip install --user package_name

# 或创建专用目录
mkdir ~/neuro_packages
export PYTHONPATH=~/neuro_packages:$PYTHONPATH
pip install --target ~/neuro_packages package_name
```

**问题**: `SSL certificate verification failed`

**解决方案**:
```bash
# 临时禁用SSL验证 (不推荐)
pip install --trusted-host pypi.org --trusted-host pypi.python.org package_name

# 或更新证书
pip install --upgrade certifi
```

### 诊断工具

创建 `diagnostic.py`:

```python
#!/usr/bin/env python3
"""
系统诊断工具
"""

import os
import sys
import subprocess
import platform

def check_system_info():
    """检查系统信息"""
    print("🖥️ 系统信息")
    print("=" * 30)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"架构: {platform.machine()}")
    print(f"处理器: {platform.processor()}")
    
    # 内存信息
    try:
        if platform.system() == "Windows":
            import psutil
            memory = psutil.virtual_memory()
            print(f"内存: {memory.total // (1024**3)}GB 总计, {memory.available // (1024**3)}GB 可用")
        else:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                total = int(meminfo.split('MemTotal:')[1].split('kB')[0]) * 1024
                print(f"内存: {total // (1024**3)}GB 总计")
    except:
        print("无法获取内存信息")

def check_dependencies():
    """检查依赖状态"""
    print("\n📦 依赖检查")
    print("=" * 30)
    
    packages = [
        'torch', 'numpy', 'scipy', 'pandas', 
        'matplotlib', 'streamlit', 'transformers',
        'mineflayer', 'qiskit', 'nengo'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")

def check_network():
    """检查网络连接"""
    print("\n🌐 网络检查")
    print("=" * 30)
    
    import urllib.request
    
    test_urls = [
        'https://pypi.org/',
        'https://huggingface.co/',
        'https://github.com/'
    ]
    
    for url in test_urls:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"✅ {url}")
        except:
            print(f"❌ {url}")

def check_minecraft():
    """检查Minecraft服务器"""
    print("\n🎮 Minecraft检查")
    print("=" * 30)
    
    import socket
    
    host = 'localhost'
    port = 25565
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ Minecraft服务器运行中")
        else:
            print("❌ Minecraft服务器未运行")
            print("提示: java -Xmx2G -Xms2G -jar paper.jar --nogui")
    except Exception as e:
        print(f"检查失败: {e}")

def main():
    """主函数"""
    print("🔍 NeuroMinecraft Genesis 系统诊断")
    print("=" * 50)
    
    check_system_info()
    check_dependencies()
    check_network()
    check_minecraft()
    
    print("\n" + "=" * 50)
    print("诊断完成！")

if __name__ == "__main__":
    main()
```

运行诊断：

```bash
python diagnostic.py
```

---

## ⚡ 性能优化

### 内存优化

#### Python内存管理

```python
import gc
import os

# 启用垃圾回收
gc.enable()

# 设置内存限制
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, hard))  # 2GB限制

# 定期清理内存
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
# 在长时间运行中定期调用
import threading
def memory_monitor():
    while True:
        cleanup_memory()
        time.sleep(300)  # 每5分钟清理一次

threading.Thread(target=memory_monitor, daemon=True).start()
```

#### 数据加载优化

```python
from torch.utils.data import DataLoader, Dataset
import torch

class OptimizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data = np.array(data, dtype=np.float32)
        
    def __getitem__(self, idx):
        # 返回内存映射的数据
        return torch.from_numpy(self.data[idx])
    
    def __len__(self):
        return len(self.data)

# 使用内存映射文件
import mmap

class MemoryMappedDataset:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_item(self, idx, size=1024):
        offset = idx * size
        self.mmap.seek(offset)
        data = self.mmap.read(size)
        return torch.from_numpy(np.frombuffer(data, dtype=np.float32))
```

### CPU优化

#### 多进程处理

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def parallel_processing_example():
    """并行处理示例"""
    
    def process_chunk(chunk):
        # 处理数据块
        result = heavy_computation(chunk)
        return result
    
    # 创建进程池
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # 分发任务
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        results = list(executor.map(process_chunk, chunks))
    
    return results
```

#### 矢量化操作

```python
import numpy as np
import torch

def vectorized_operations():
    """使用矢量化优化计算"""
    
    # 批量处理而非循环
    # 好的做法
    batch_inputs = torch.randn(100, 512)
    outputs = model(batch_inputs)  # 一次处理100个样本
    
    # 不好的做法
    # for i in range(100):
    #     output = model(batch_inputs[i:i+1])
```

### GPU优化

#### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training():
    """混合精度训练"""
    
    scaler = GradScaler()
    
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

#### 模型并行

```python
import torch.nn as nn

class ParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.DataParallel(nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ))
        self.classifier = nn.DataParallel(nn.Linear(512, 10))
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 🎉 安装完成检查

### 最终验证脚本

创建 `final_check.py`:

```python
#!/usr/bin/env python3
"""
最终安装验证
"""

import torch
import streamlit
import subprocess
import sys

def comprehensive_check():
    """综合检查"""
    print("🎉 NeuroMinecraft Genesis 安装完成验证")
    print("=" * 50)
    
    checks = []
    
    # 1. Python环境
    if sys.version_info >= (3, 8):
        checks.append(("Python版本", "✅", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
    else:
        checks.append(("Python版本", "❌", "版本过低"))
    
    # 2. 核心库
    core_libs = [
        ('PyTorch', 'torch', '2.0.1'),
        ('NumPy', 'numpy', '1.21+'),
        ('Streamlit', 'streamlit', '1.20+'),
        ('Transformers', 'transformers', '4.20+'),
    ]
    
    for name, lib, version in core_libs:
        try:
            module = __import__(lib)
            actual_version = getattr(module, '__version__', 'unknown')
            checks.append((name, "✅", f"v{actual_version}"))
        except ImportError:
            checks.append((name, "❌", "未安装"))
    
    # 3. GPU支持
    if torch.cuda.is_available():
        checks.append(("GPU加速", "✅", f"{torch.cuda.get_device_name(0)}"))
    else:
        checks.append(("GPU加速", "⚠️", "不可用 (将使用CPU)"))
    
    # 4. Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(("Node.js", "✅", result.stdout.strip()))
        else:
            checks.append(("Node.js", "❌", "未安装"))
    except FileNotFoundError:
        checks.append(("Node.js", "❌", "未安装"))
    
    # 5. Minecraft服务器连接
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 25565))
        sock.close()
        
        if result == 0:
            checks.append(("Minecraft服务器", "✅", "运行中"))
        else:
            checks.append(("Minecraft服务器", "⚠️", "未运行"))
    except:
        checks.append(("Minecraft服务器", "⚠️", "无法检测"))
    
    # 显示检查结果
    for name, status, info in checks:
        print(f"{status} {name:<15} {info}")
    
    # 统计通过率
    passed = sum(1 for _, status, _ in checks if status == "✅")
    total = len(checks)
    success_rate = passed / total * 100
    
    print(f"\n📊 检查结果: {passed}/{total} 通过 ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 安装验证通过！可以开始使用项目了")
        print("\n🚀 快速开始:")
        print("  python docs/QUICK_START.py")
        print("  streamlit run docs/QUICK_START.py")
    else:
        print("⚠️ 部分检查未通过，建议修复后再使用")
        print("\n💡 获取帮助:")
        print("  查看故障排除章节")
        print("  GitHub Issues: https://github.com/bingdongni/NeuroMinecraftGenesis/issues")

if __name__ == "__main__":
    comprehensive_check()
```

运行最终检查：

```bash
python final_check.py
```

---

<div align="center">

**恭喜！您已完成NeuroMinecraft Genesis的完整安装！**

🎉 **现在可以开始探索AGI的未来了！**

**[⬆ 回到顶部](#详细安装指南)**

Made with ❤️ by the NeuroMinecraft Genesis Team

</div>