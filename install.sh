#!/bin/bash
echo "NeuroMinecraft Genesis - 安装脚本"
echo "================================"
echo

echo "[1/6] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.11+"
    exit 1
fi

echo "[2/6] 升级pip..."
python3 -m pip install --upgrade pip

echo "[3/6] 安装核心依赖包..."
python3 -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu
python3 -m pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.3
python3 -m pip install opencv-python==4.8.0.76 Pillow==10.0.0
python3 -m pip install transformers==4.30.2 datasets==2.13.1
python3 -m pip install nengo==4.1.0 qiskit==2.2.3 qiskit-aer==0.12.0
python3 -m pip install streamlit==1.25.0 plotly==5.15.0

echo "[4/6] 安装可视化工具..."
python3 -m pip install matplotlib==3.7.2 seaborn==0.12.2
python3 -m pip install networkx==3.1 pyvis==0.3.2

echo "[5/6] 安装实用工具..."
python3 -m pip install requests==2.31.0 aiohttp==3.8.5
python3 -m pip install pyyaml==6.0.1 rich==13.4.2
python3 -m pip install tqdm==4.65.0 loguru==0.7.0

echo "[6/6] 创建必要目录..."
mkdir -p models/genomes
mkdir -p data/evolution_logs
mkdir -p data/brain_scans
mkdir -p demo_models/genomes
mkdir -p demo_data/evolution_logs

echo
echo "============================"
echo "安装完成！"
echo "============================"
echo
echo "快速启动："
echo "  python3 simple_test.py           - 运行基础测试"
echo "  streamlit run utils/visualization/advanced_dashboard.py  - 启动可视化界面"
echo
echo "文档请查看 README.md"