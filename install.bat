@echo off
chcp 65001 >nul
title NeuroMinecraft Genesis 安装程序
color 0A

echo.
echo ████████████████████████████████████████████████████████████████
echo █                                                              █
echo █         NeuroMinecraft Genesis (NMG) 安装程序                █
echo █                    版本 1.0                                    █
echo █                                                              █
echo █                    开发者: bingdongni                        █
echo █                                                              █
echo ████████████████████████████████████████████████████████████████
echo.

echo 📦 开始安装 NeuroMinecraft Genesis...
echo.

echo 🐍 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python 环境检查通过

echo.
echo 📦 安装依赖包...
pip install -r requirements.txt --upgrade

if errorlevel 1 (
    echo ❌ 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)

echo.
echo ✅ NeuroMinecraft Genesis 安装完成！
echo.
echo 🚀 快速启动命令:
echo    python quickstart.py
echo.
pause
