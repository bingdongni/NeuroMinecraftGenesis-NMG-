#!/bin/bash
# NeuroMinecraft Genesis 安装脚本
# 开发者：bingdongni

echo "████████████████████████████████████████████████████████████████"
echo "█                                                              █"
echo "█         NeuroMinecraft Genesis (NMG) 安装程序                █"
echo "█                    版本 1.0                                    █"
echo "█                                                              █"
echo "█                    开发者: bingdongni                        █"
echo "█                                                              █"
echo "████████████████████████████████████████████████████████████████"

echo ""
echo "📦 开始安装 NeuroMinecraft Genesis..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

echo "✅ Python 环境检查通过"

echo ""
echo "📦 安装依赖包..."

# 升级pip
python3 -m pip install --upgrade pip

# 安装依赖
python3 -m pip install -r requirements.txt

echo ""
echo "✅ NeuroMinecraft Genesis 安装完成！"
echo ""
echo "🚀 快速启动命令:"
echo "    python3 quickstart.py"
echo ""
