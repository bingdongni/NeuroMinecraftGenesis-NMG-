@echo off
REM NeuroMinecraft Genesis Windows安装脚本
REM 创建时间: 2025-11-14
REM 项目: NeuroMinecraft Genesis v1.0.0+

setlocal enabledelayedexpansion

REM 颜色代码 (Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "BOLD=[1m"
set "NC=[0m"

REM 打印函数
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🔍 %~1%NC%
echo %BLUE%================================%NC%
echo.

print_success() {
    echo %GREEN%✅ %~1%NC%
}

print_error() {
    echo %RED%❌ %~1%NC%
}

print_warning() {
    echo %YELLOW%⚠️  %~1%NC%
}

print_info() {
    echo %CYAN%ℹ️  %~1%NC%
}

print_bold() {
    echo %BOLD%%~1%NC%
}

REM 检查操作系统
:check_os
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🔍 操作系统检查%NC%
echo %BLUE%================================%NC%
echo.

print_success "检测到Windows系统"
ver
goto :check_python

REM 检查Python版本
:check_python
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🐍 Python环境检查%NC%
echo %BLUE%================================%NC%
echo.

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo %GREEN%✅ 找到Python: %PYTHON_VERSION% %NC%
    
    python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo %GREEN%✅ Python版本符合要求 (^>=3.8^) %NC%
        set "PYTHON_CMD=python"
        set "PIP_CMD=pip"
    ) else (
        echo %RED%❌ Python版本过低，需要Python 3.8+ %NC%
        echo 💡 请从 https://python.org 下载安装Python
        pause
        exit /b 1
    )
) else (
    python3 --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set "PYTHON_VERSION=%%i"
        echo %GREEN%✅ 找到Python: %PYTHON_VERSION% %NC%
        set "PYTHON_CMD=python3"
        set "PIP_CMD=pip3"
    ) else (
        echo %RED%❌ 未找到Python，请先安装Python 3.8+ %NC%
        echo 💡 请从 https://python.org 下载安装Python
        pause
        exit /b 1
    )
)

REM 检查pip
:check_pip
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%📦 Pip包管理器检查%NC%
echo %BLUE%================================%NC%
echo.

%PIP_CMD% --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "tokens=2" %%i in ('%PIP_CMD% --version 2^>^&1') do set "PIP_VERSION=%%i"
    echo %GREEN%✅ 找到pip: %PIP_VERSION% %NC%
) else (
    echo %RED%❌ 未找到pip，请安装Python包管理器 %NC%
    pause
    exit /b 1
)

REM 虚拟环境配置
:create_venv
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🏠 虚拟环境配置%NC%
echo %BLUE%================================%NC%
echo.

set /p create_venv="是否创建虚拟环境？(y/n): "
if /i "!create_venv!"=="y" (
    set "VENV_DIR=venv"
    
    if exist "!VENV_DIR!" (
        echo %YELLOW%⚠️  虚拟环境目录已存在 %NC%
        set /p recreate="删除现有环境并重新创建？(y/n): "
        if /i "!recreate!"=="y" (
            rmdir /s /q "!VENV_DIR!"
        ) else (
            echo %CYAN%ℹ️  使用现有虚拟环境 %NC%
            set "USE_VENV=true"
            goto :install_dependencies
        )
    )
    
    echo %CYAN%ℹ️  创建虚拟环境... %NC%
    %PYTHON_CMD% -m venv "!VENV_DIR!"
    
    echo %CYAN%ℹ️  激活虚拟环境 %NC%
    call "!VENV_DIR!\Scripts\activate.bat"
    
    if !ERRORLEVEL! equ 0 (
        echo %GREEN%✅ 虚拟环境创建并激活成功 %NC%
        set "USE_VENV=true"
    ) else (
        echo %RED%❌ 虚拟环境创建失败 %NC%
        set "USE_VENV=false"
    )
) else (
    echo %CYAN%ℹ️  跳过虚拟环境创建，使用系统Python %NC%
    set "USE_VENV=false"
)

REM 安装依赖
:install_dependencies
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%📦 Python依赖安装%NC%
echo %BLUE%================================%NC%
echo.

if not exist "requirements.txt" (
    echo %RED%❌ 未找到requirements.txt文件 %NC%
    pause
    exit /b 1
)

echo %CYAN%ℹ️  更新pip... %NC%
%PIP_CMD% install --upgrade pip

echo %CYAN%ℹ️  安装项目依赖... %NC%
echo %YELLOW%⚠️  这可能需要几分钟时间，请耐心等待... %NC%

%PIP_CMD% install -r requirements.txt
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ 依赖安装完成 %NC%
) else (
    echo %RED%❌ 依赖安装失败 %NC%
    echo %CYAN%ℹ️  请检查网络连接或手动安装: %PIP_CMD% install -r requirements.txt %NC%
    pause
    exit /b 1
)

REM 检查Minecraft服务器
:check_minecraft_server
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🎮 Minecraft服务器检查%NC%
echo %BLUE%================================%NC%
echo.

set "SERVER_DIR=worlds\minecraft\server"
set "PAPER_JAR=!SERVER_DIR!\paper.jar"
set "EULA_FILE=!SERVER_DIR!\eula.txt"

if exist "!PAPER_JAR!" (
    for %%A in ("!PAPER_JAR!") do set "PAPER_SIZE=%%~zA"
    set /a PAPER_SIZE_MB=!PAPER_SIZE!/1024/1024
    echo %GREEN%✅ Minecraft服务器文件存在 (大小: !PAPER_SIZE_MB! MB) %NC%
) else (
    echo %YELLOW%⚠️  缺少paper.jar文件 %NC%
    echo %CYAN%ℹ️  📥 下载地址: https://papermc.io/ %NC%
    echo %CYAN%ℹ️  💡 自动下载命令: %NC%
    echo %CYAN%curl -L -o worlds/minecraft/server/paper.jar \ %NC%
    echo %CYON%"https://api.papermc.io/v2/projects/paper/versions/1.20.1/ \ %NC%
    echo %CYON%builds/latest/downloads/paper-1.20.1-latest.jar" %NC%
    
    set /p download_paper="是否现在下载PaperMC服务器？(y/n): "
    if /i "!download_paper!"=="y" (
        if not exist "!SERVER_DIR!" mkdir "!SERVER_DIR!"
        echo %CYAN%ℹ️  正在下载PaperMC... %NC%
        
        REM 检查是否有curl或wget
        where curl >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            curl -L -o "!PAPER_JAR!" "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
        ) else (
            where wget >nul 2>&1
            if !ERRORLEVEL! equ 0 (
                wget -O "!PAPER_JAR!" "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
            ) else (
                echo %RED%❌ 未找到curl或wget，请手动下载 %NC%
                echo %CYAN%💡 访问 https://papermc.io/ 下载PaperMC 1.20.1 %NC%
                pause
                exit /b 1
            )
        )
        
        if exist "!PAPER_JAR!" (
            for %%A in ("!PAPER_JAR!") do set "PAPER_SIZE=%%~zA"
            set /a PAPER_SIZE_MB=!PAPER_SIZE!/1024/1024
            echo %GREEN%✅ PaperMC下载完成 (大小: !PAPER_SIZE_MB! MB) %NC%
        ) else (
            echo %RED%❌ PaperMC下载失败 %NC%
            pause
            exit /b 1
        )
    ) else (
        echo %CYAN%ℹ️  跳过PaperMC下载 %NC%
    )
)

REM 检查EULA文件
if exist "!EULA_FILE!" (
    findstr /i "eula=true" "!EULA_FILE!" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo %GREEN%✅ EULA协议已同意 %NC%
    ) else (
        echo %YELLOW%⚠️  EULA协议未同意，服务器可能无法启动 %NC%
    )
) else (
    echo %YELLOW%⚠️  EULA文件不存在 %NC%
)

REM 检查Java环境
:check_java
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%☕ Java环境检查%NC%
echo %BLUE%================================%NC%
echo.

java -version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ Java已安装 %NC%
    java -version 2>&1 | findstr version
    echo %CYAN%ℹ️  💡 建议使用Java 17+ %NC%
) else (
    echo %YELLOW%⚠️  Java未安装 (Minecraft服务器需要) %NC%
    echo %CYAN%ℹ️  💡 安装建议: %NC%
    echo %CYAN%从 https://adoptium.net/ 下载安装 %NC%
)

REM 运行环境检查
:run_environment_check
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🔍 运行环境检查%NC%
echo %BLUE%================================%NC%
echo.

if exist "check_environment.py" (
    echo %CYAN%ℹ️  运行项目环境检查... %NC%
    python check_environment.py
    if !ERRORLEVEL! equ 0 (
        echo %GREEN%✅ 环境检查通过 %NC%
    ) else (
        echo %YELLOW%⚠️  环境检查发现问题，请查看上述输出 %NC%
    )
) else (
    echo %YELLOW%⚠️  未找到check_environment.py脚本 %NC%
)

REM 测试核心功能
:test_core_functions
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🧪 核心功能测试%NC%
echo %BLUE%================================%NC%
echo.

echo %CYAN%ℹ️  测试核心模块导入... %NC%

REM 测试基本导入
python -c "import numpy; print(f'NumPy {numpy.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ NumPy导入测试通过 %NC%
) else (
    echo %RED%❌ NumPy导入测试失败 %NC%
)

python -c "import pandas; print(f'Pandas {pandas.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ Pandas导入测试通过 %NC%
) else (
    echo %RED%❌ Pandas导入测试失败 %NC%
)

python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ Matplotlib导入测试通过 %NC%
) else (
    echo %RED%❌ Matplotlib导入测试失败 %NC%
)

python -c "import torch; print(f'PyTorch {torch.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ PyTorch导入测试通过 %NC%
) else (
    echo %RED%❌ PyTorch导入测试失败 %NC%
)

python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ Streamlit导入测试通过 %NC%
) else (
    echo %RED%❌ Streamlit导入测试失败 %NC%
)

python -c "import plotly; print(f'Plotly {plotly.__version__}')" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo %GREEN%✅ Plotly导入测试通过 %NC%
) else (
    echo %RED%❌ Plotly导入测试失败 %NC%
)

REM 创建启动脚本
:create_startup_scripts
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🚀 创建启动脚本%NC%
echo %BLUE%================================%NC%
echo.

REM Windows启动脚本已存在，检查其他脚本
if exist "start_project.bat" (
    echo %GREEN%✅ 启动脚本已存在: start_project.bat %NC%
) else (
    echo %CYAN%ℹ️  创建启动脚本: start_project.bat %NC%
    REM 脚本已在install_dependencies.sh中创建
)

if exist "start_project.sh" (
    echo %GREEN%✅ Unix启动脚本已存在: start_project.sh %NC%
) else (
    echo %YELLOW%⚠️  Unix启动脚本缺失，建议添加start_project.sh %NC%
)

REM 显示完成信息
:show_completion_info
echo.
echo %BLUE%================================%NC%
echo %BOLD%%BLUE%🎉 安装完成%NC%
echo %BLUE%================================%NC%
echo.

echo %GREEN%✅ NeuroMinecraft Genesis 安装完成！%NC%
echo.
echo %BOLD%📋 下一步操作:%NC%
echo %CYAN%ℹ️  1. 运行环境检查:%NC%
echo %CYAN%   python check_environment.py%NC%
echo.
echo %CYAN%ℹ️  2. 启动项目:%NC%
echo %CYAN%   start_project.bat%NC%
echo.
echo %CYAN%ℹ️  3. 其他启动方式:%NC%
echo %CYAN%   python quickstart.py%NC%
echo %CYAN%   streamlit run utils/visualization/streamlit_dashboard.py%NC%
echo %CYAN%   python utils/visualization/brain_network_3d.py%NC%
echo.
echo %CYAN%ℹ️  4. Minecraft服务器:%NC%
echo %CYAN%   worlds\minecraft\server\start.bat%NC%
echo.
echo %BOLD%💡 使用提示:%NC%
echo %CYAN%ℹ️  • 首次运行可能需要下载模型文件，请保持网络连接%NC%
echo %CYAN%ℹ️  • Minecraft服务器需要至少2GB内存%NC%
echo %CYAN%ℹ️  • 如遇到问题，运行 python check_environment.py 诊断%NC%
echo.
echo %BOLD%🌟 项目地址: https://github.com/你的用户名/NeuroMinecraft-Genesis%NC%
echo.

if "%USE_VENV%"=="true" (
    echo %YELLOW%⚠️  每次使用前记得激活虚拟环境:%NC%
    echo %CYAN%   venv\Scripts\activate%NC%
    echo.
)

echo %GREEN%✨ Windows安装脚本执行完成！%NC%
echo.

pause
exit /b 0