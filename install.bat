@echo off
echo NeuroMinecraft Genesis - 一键安装脚本
echo ========================================
echo.

echo [1/6] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.11+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [2/6] 升级pip...
python -m pip install --upgrade pip

echo [3/6] 安装核心依赖包...
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu
pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.3
pip install opencv-python==4.8.0.76 Pillow==10.0.0
pip install transformers==4.30.2 datasets==2.13.1
pip install nengo==4.1.0 qiskit==2.2.3 qiskit-aer==0.12.0
pip install streamlit==1.25.0 plotly==5.15.0

echo [4/6] 安装可视化工具...
pip install matplotlib==3.7.2 seaborn==0.12.2
pip install networkx==3.1 pyvis==0.3.2

echo [5/6] 安装实用工具...
pip install requests==2.31.0 aiohttp==3.8.5
pip install pyyaml==6.0.1 rich==13.4.2
pip install tqdm==4.65.0 loguru==0.7.0

echo [6/6] 创建必要目录...
mkdir models\genomes >nul 2>&1
mkdir data\evolution_logs >nul 2>&1
mkdir data\brain_scans >nul 2>&1
mkdir demo_models\genomes >nul 2>&1
mkdir demo_data\evolution_logs >nul 2>&1

echo.
echo ================================
echo 安装完成！
echo ================================
echo.
echo 快速启动：
echo   python simple_test.py           - 运行基础测试
echo   streamlit run utils\visualization\advanced_dashboard.py  - 启动可视化界面
echo.
echo 文档请查看 README.md
echo.
pause