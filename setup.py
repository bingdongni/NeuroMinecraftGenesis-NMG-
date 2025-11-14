#!/usr/bin/env python3
"""
NeuroMinecraft Genesis 项目初始化脚本
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """创建项目基本结构"""
    
    # 项目根目录
    project_root = Path(__file__).parent
    
    print("🚀 初始化 NeuroMinecraft Genesis 项目...")
    
    # 检查是否在正确的目录
    if not (project_root / "README.md").exists():
        print("❌ 错误：请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 创建示例配置文件
    create_config_files(project_root)
    
    # 创建示例Python文件
    create_example_files(project_root)
    
    print("✅ 项目初始化完成！")
    print("\n📁 创建的文件结构：")
    print(project_structure_info())

def create_config_files(project_root):
    """创建配置文件"""
    config_dir = project_root / "config"
    
    # 环境配置示例
    env_example = """# 环境配置示例
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# Minecraft服务器配置
MINECRAFT_SERVER_HOST=localhost
MINECRAFT_SERVER_PORT=25565

# 数据库配置
DATABASE_URL=sqlite:///neurogenesis.db

# 模型配置
MODEL_CACHE_DIR=./models/cache
PRETRAINED_MODEL_DIR=./models/pretrained
"""
    
    (config_dir / ".env.example").write_text(env_example)
    
    # 项目配置
    project_config = """project:
  name: "NeuroMinecraft Genesis"
  version: "1.0.0"
  description: "AI agents in Minecraft worlds with evolution and neuroscience"

agents:
  single:
    brain_model: "neural_network"
    learning_rate: 0.001
  
  multi:
    max_agents: 10
    communication_protocol: "distributed"

worlds:
  minecraft:
    server_config: "server.properties"
    world_seed: "genesis"
  
  procgen:
    size: 1000
    complexity: 0.8

evolution:
  population_size: 100
  mutation_rate: 0.1
  selection_pressure: 0.8
"""
    
    (config_dir / "project.yaml").write_text(project_config)

def create_example_files(project_root):
    """创建示例Python文件"""
    
    # 主模块示例
    main_example = '''"""NeuroMinecraft Genesis 主模块"""

from .agents import SingleAgent, MultiAgentSystem
from .core import BrainModel, EvolutionEngine
from .worlds import MinecraftWorld

__version__ = "1.0.0"
__all__ = ["SingleAgent", "MultiAgentSystem", "BrainModel", "EvolutionEngine", "MinecraftWorld"]
'''
    
    (project_root / "__init__.py").write_text(main_example)
    
    # 快速开始示例
    quickstart = '''"""快速开始示例"""

from NeuroMinecraftGenesis import SingleAgent, MinecraftWorld

def main():
    # 创建Minecraft世界
    world = MinecraftWorld()
    
    # 创建AI代理
    agent = SingleAgent()
    
    # 运行代理
    agent.run_in_world(world)

if __name__ == "__main__":
    main()
'''
    
    (project_root / "quickstart.py").write_text(quickstart)

def project_structure_info():
    """返回项目结构信息"""
    return """
NeuroMinecraftGenesis/
├── agents/           # AI代理模块
├── core/            # 核心算法
├── worlds/          # 环境世界
├── utils/           # 工具模块
├── experiments/     # 实验模块
├── models/          # 模型存储
├── data/            # 数据存储
├── docs/            # 文档
├── config/          # 配置文件
├── quickstart.py    # 快速开始示例
└── README.md        # 项目说明
"""

if __name__ == "__main__":
    create_project_structure()