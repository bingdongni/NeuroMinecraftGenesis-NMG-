#!/bin/bash
# NeuroMinecraft Genesis一键安装脚本
# 创建时间: 2025-11-14
# 项目: NeuroMinecraft Genesis v1.0.0+

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印函数
print_color() {
    echo -e "${1}${2}${NC}"
}

print_header() {
    print_color "\n${BLUE}================================${NC}"
    print_color "${BOLD}${BLUE}🔍 $1${NC}"
    print_color "${BLUE}================================${NC}\n"
}

print_success() {
    print_color "${GREEN}✅ $1${NC}"
}

print_error() {
    print_color "${RED}❌ $1${NC}"
}

print_warning() {
    print_color "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    print_color "${CYAN}ℹ️  $1${NC}"
}

print_bold() {
    print_color "${BOLD}$1${NC}"
}

# 检查操作系统
check_os() {
    print_header "操作系统检查"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "检测到Linux系统"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "检测到macOS系统"
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        print_success "检测到Windows系统 (Git Bash/Cygwin)"
        OS="windows"
    else
        print_warning "未知操作系统: $OSTYPE"
        OS="unknown"
    fi
    
    uname -a
}

# 检查Python版本
check_python() {
    print_header "Python环境检查"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "找到Python: $PYTHON_VERSION"
        
        # 提取版本号
        PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python版本符合要求 (>=3.8)"
            PYTHON_CMD="python3"
            PIP_CMD="pip3"
        else
            print_error "Python版本过低，需要Python 3.8+"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version)
        print_success "找到Python: $PYTHON_VERSION"
        
        PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python版本符合要求 (>=3.8)"
            PYTHON_CMD="python"
            PIP_CMD="pip"
        else
            print_error "Python版本过低，需要Python 3.8+"
            exit 1
        fi
    else
        print_error "未找到Python3，请先安装Python 3.8+"
        exit 1
    fi
}

# 检查pip
check_pip() {
    print_header "Pip包管理器检查"
    
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version)
        print_success "找到pip: $PIP_VERSION"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_VERSION=$(pip --version)
        print_success "找到pip: $PIP_VERSION"
        PIP_CMD="pip"
    else
        print_error "未找到pip，请安装Python包管理器"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    print_header "虚拟环境配置"
    
    read -p "是否创建虚拟环境？(y/n): " create_venv
    if [[ "$create_venv" =~ ^[Yy]$ ]]; then
        VENV_DIR="venv"
        
        if [ -d "$VENV_DIR" ]; then
            print_warning "虚拟环境目录已存在"
            read -p "删除现有环境并重新创建？(y/n): " recreate
            if [[ "$recreate" =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_DIR"
            else
                print_info "使用现有虚拟环境"
                USE_VENV=true
                return
            fi
        fi
        
        print_info "创建虚拟环境..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        
        if [ "$OS" = "windows" ]; then
            print_info "激活虚拟环境 (Windows):"
            print_color "${CYAN}source venv/Scripts/activate${NC}"
            source "$VENV_DIR/Scripts/activate"
        else
            print_info "激活虚拟环境 (Unix/macOS):"
            print_color "${CYAN}source venv/bin/activate${NC}"
            source "$VENV_DIR/bin/activate"
        fi
        
        USE_VENV=true
        print_success "虚拟环境创建并激活成功"
    else
        USE_VENV=false
        print_info "跳过虚拟环境创建，使用系统Python"
    fi
}

# 安装依赖
install_dependencies() {
    print_header "Python依赖安装"
    
    if [ ! -f "requirements.txt" ]; then
        print_error "未找到requirements.txt文件"
        exit 1
    fi
    
    print_info "更新pip..."
    $PIP_CMD install --upgrade pip
    
    print_info "安装项目依赖..."
    print_warning "这可能需要几分钟时间，请耐心等待..."
    
    if $PIP_CMD install -r requirements.txt; then
        print_success "依赖安装完成"
    else
        print_error "依赖安装失败"
        print_info "请检查网络连接或手动安装: $PIP_CMD install -r requirements.txt"
        exit 1
    fi
}

# 检查Minecraft服务器
check_minecraft_server() {
    print_header "Minecraft服务器检查"
    
    SERVER_DIR="worlds/minecraft/server"
    PAPER_JAR="$SERVER_DIR/paper.jar"
    EULA_FILE="$SERVER_DIR/eula.txt"
    
    if [ -f "$PAPER_JAR" ]; then
        PAPER_SIZE=$(du -h "$PAPER_JAR" | cut -f1)
        print_success "Minecraft服务器文件存在 (大小: $PAPER_SIZE)"
    else
        print_warning "缺少paper.jar文件"
        print_info "📥 下载地址: https://papermc.io/"
        print_info "💡 自动下载命令:"
        print_color "${CYAN}curl -L -o worlds/minecraft/server/paper.jar \\"${NC}"
        print_color "${CYAN}  \"https://api.papermc.io/v2/projects/paper/versions/1.20.1/\\"${NC}"
        print_color "${CYAN}  builds/latest/downloads/paper-1.20.1-latest.jar\"${NC}"
        
        read -p "是否现在下载PaperMC服务器？(y/n): " download_paper
        if [[ "$download_paper" =~ ^[Yy]$ ]]; then
            mkdir -p "$SERVER_DIR"
            print_info "正在下载PaperMC..."
            
            if command -v curl &> /dev/null; then
                curl -L -o "$PAPER_JAR" \
                    "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
            elif command -v wget &> /dev/null; then
                wget -O "$PAPER_JAR" \
                    "https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/latest/downloads/paper-1.20.1-latest.jar"
            else
                print_error "未找到curl或wget，请手动下载"
                exit 1
            fi
            
            if [ -f "$PAPER_JAR" ]; then
                PAPER_SIZE=$(du -h "$PAPER_JAR" | cut -f1)
                print_success "PaperMC下载完成 (大小: $PAPER_SIZE)"
            else
                print_error "PaperMC下载失败"
                exit 1
            fi
        else
            print_info "跳过PaperMC下载"
        fi
    fi
    
    # 检查EULA文件
    if [ -f "$EULA_FILE" ]; then
        if grep -q "eula=true" "$EULA_FILE" 2>/dev/null; then
            print_success "EULA协议已同意"
        else
            print_warning "EULA协议未同意，服务器可能无法启动"
        fi
    else
        print_warning "EULA文件不存在"
    fi
}

# 检查Java环境
check_java() {
    print_header "Java环境检查"
    
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1)
        print_success "Java已安装: $JAVA_VERSION"
        
        # 检查版本号
        JAVA_MAJOR=$(java -version 2>&1 | head -n 1 | grep -oP '(?<=version ")\d+' || echo 0)
        if [ "$JAVA_MAJOR" -ge 17 ]; then
            print_success "Java版本符合要求 (>=17)"
        else
            print_warning "Java版本过低，建议使用Java 17+"
        fi
    else
        print_warning "Java未安装 (Minecraft服务器需要)"
        print_info "💡 安装建议:"
        print_color "${CYAN}# Ubuntu/Debian:${NC}"
        print_color "${CYAN}sudo apt update && sudo apt install openjdk-17-jdk${NC}"
        print_color "${CYAN}# CentOS/RHEL:${NC}"
        print_color "${CYAN}sudo yum install java-17-openjdk-devel${NC}"
        print_color "${CYAN}# macOS:${NC}"
        print_color "${CYAN}brew install openjdk@17${NC}"
        print_color "${CYAN}# Windows:${NC}"
        print_color "${CYAN}从 https://adoptium.net/ 下载安装${NC}"
    fi
}

# 运行环境检查
run_environment_check() {
    print_header "运行环境检查"
    
    if [ -f "check_environment.py" ]; then
        print_info "运行项目环境检查..."
        if $PYTHON_CMD check_environment.py; then
            print_success "环境检查通过"
        else
            print_warning "环境检查发现问题，请查看上述输出"
        fi
    else
        print_warning "未找到check_environment.py脚本"
    fi
}

# 测试核心功能
test_core_functions() {
    print_header "核心功能测试"
    
    print_info "测试核心模块导入..."
    
    # 测试基本导入
    test_imports=(
        "import numpy; print(f'NumPy {numpy.__version__}')"
        "import pandas; print(f'Pandas {pandas.__version__}')"
        "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
        "import torch; print(f'PyTorch {torch.__version__}')"
        "import streamlit; print(f'Streamlit {streamlit.__version__}')"
        "import plotly; print(f'Plotly {plotly.__version__}')"
    )
    
    failed_imports=0
    for test_import in "${test_imports[@]}"; do
        if $PYTHON_CMD -c "$test_import" 2>/dev/null; then
            print_success "导入测试通过"
        else
            print_error "导入测试失败: $test_import"
            ((failed_imports++))
        fi
    done
    
    if [ $failed_imports -eq 0 ]; then
        print_success "所有核心模块导入测试通过"
    else
        print_warning "$failed_imports 个模块导入失败"
    fi
}

# 创建启动脚本
create_startup_scripts() {
    print_header "创建启动脚本"
    
    # 创建快速启动脚本
    cat > "start_project.sh" << 'EOF'
#!/bin/bash
# NeuroMinecraft Genesis 快速启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_color() {
    echo -e "${1}${2}${NC}"
}

print_color "\n${BLUE}🚀 NeuroMinecraft Genesis 启动${NC}\n"

# 检查虚拟环境
if [ -d "venv" ]; then
    print_color "${YELLOW}激活虚拟环境...${NC}"
    source venv/bin/activate
fi

# 选择启动方式
echo "请选择启动方式:"
echo "1) 完整系统演示 (quickstart.py)"
echo "2) 可视化仪表板 (Streamlit)"
echo "3) 3D脑网络界面"
echo "4) 启动Minecraft服务器"
echo "5) 查看环境检查报告"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        print_color "${GREEN}启动完整系统演示...${NC}"
        python quickstart.py
        ;;
    2)
        print_color "${GREEN}启动可视化仪表板...${NC}"
        streamlit run utils/visualization/streamlit_dashboard.py
        ;;
    3)
        print_color "${GREEN}启动3D脑网络界面...${NC}"
        python utils/visualization/brain_network_3d.py
        ;;
    4)
        print_color "${GREEN}启动Minecraft服务器...${NC}"
        bash worlds/minecraft/server/start.sh
        ;;
    5)
        print_color "${GREEN}查看环境检查报告...${NC}"
        python check_environment.py
        ;;
    *)
        print_color "${RED}无效选择${NC}"
        ;;
esac
EOF

    chmod +x "start_project.sh"
    print_success "创建启动脚本: start_project.sh"
    
    # 创建Windows批处理文件
    cat > "start_project.bat" << 'EOF'
@echo off
chcp 65001 >nul
echo 🚀 NeuroMinecraft Genesis 启动

REM 检查虚拟环境
if exist venv (
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

echo 请选择启动方式:
echo 1) 完整系统演示 (quickstart.py)
echo 2) 可视化仪表板 (Streamlit)
echo 3) 3D脑网络界面
echo 4) 启动Minecraft服务器
echo 5) 查看环境检查报告

set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" (
    echo 启动完整系统演示...
    python quickstart.py
) else if "%choice%"=="2" (
    echo 启动可视化仪表板...
    streamlit run utils/visualization/streamlit_dashboard.py
) else if "%choice%"=="3" (
    echo 启动3D脑网络界面...
    python utils/visualization/brain_network_3d.py
) else if "%choice%"=="4" (
    echo 启动Minecraft服务器...
    bash worlds/minecraft/server/start.sh
) else if "%choice%"=="5" (
    echo 查看环境检查报告...
    python check_environment.py
) else (
    echo 无效选择
)

pause
EOF

    print_success "创建启动脚本: start_project.bat"
}

# 显示安装完成信息
show_completion_info() {
    print_header "安装完成"
    
    print_success "🎉 NeuroMinecraft Genesis 安装完成！"
    
    print_bold "\n📋 下一步操作:"
    print_info "1. 运行环境检查:"
    print_color "${CYAN}  python check_environment.py${NC}"
    
    print_info "\n2. 启动项目:"
    if [ "$OS" = "windows" ]; then
        print_color "${CYAN}  start_project.bat${NC}"
    else
        print_color "${CYAN}  ./start_project.sh${NC}"
    fi
    
    print_info "\n3. 其他启动方式:"
    print_color "${CYAN}  python quickstart.py${NC}"
    print_color "${CYAN}  streamlit run utils/visualization/streamlit_dashboard.py${NC}"
    print_color "${CYAN}  python utils/visualization/brain_network_3d.py${NC}"
    
    print_info "\n4. Minecraft服务器:"
    print_color "${CYAN}  bash worlds/minecraft/server/start.sh${NC}"
    
    print_info "\n📁 重要文件:"
    print_color "${CYAN}  requirements.txt          - 依赖配置${NC}"
    print_color "${CYAN}  check_environment.py      - 环境检查${NC}"
    print_color "${CYAN}  environment_check_report.json - 检查报告${NC}"
    
    if [ "$USE_VENV" = true ]; then
        print_warning "\n🔄 每次使用前记得激活虚拟环境:"
        if [ "$OS" = "windows" ]; then
            print_color "${CYAN}  venv\\Scripts\\activate${NC}"
        else
            print_color "${CYAN}  source venv/bin/activate${NC}"
        fi
    fi
    
    print_bold "\n💡 使用提示:"
    print_info "• 首次运行可能需要下载模型文件，请保持网络连接"
    print_info "• Minecraft服务器需要至少2GB内存"
    print_info "• 如遇到问题，运行 python check_environment.py 诊断"
    
    print_info "\n🌟 项目地址: https://github.com/你的用户名/NeuroMinecraft-Genesis"
}

# 主安装流程
main() {
    print_color "\n${BOLD}${PURPLE}🚀 NeuroMinecraft Genesis 一键安装脚本${NC}"
    print_color "${PURPLE}===============================================${NC}\n"
    
    check_os
    check_python
    check_pip
    create_venv
    install_dependencies
    check_minecraft_server
    check_java
    run_environment_check
    test_core_functions
    create_startup_scripts
    show_completion_info
    
    print_color "\n${GREEN}✨ 安装脚本执行完成！${NC}\n"
}

# 错误处理
set +e
trap 'print_error "安装过程中发生错误，请查看上方信息"; exit 1' ERR

# 运行主函数
main "$@"