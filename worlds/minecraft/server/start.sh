#!/bin/bash
# ================================================
# PaperMC 1.20.1 服务器启动脚本
# 内存分配: 2GB
# Java版本: Java 17+
# 创建时间: 2025-11-13
# ================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# 检查Java版本
check_java() {
    if ! command -v java &> /dev/null; then
        print_error "Java未安装！请安装Java 17或更高版本"
        exit 1
    fi

    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    print_info "检测到Java版本: $JAVA_VERSION"
    
    # 检查Java版本是否为17或更高
    if ! java -version 2>&1 | grep -qE "(1[7-9]|2[0-9])"; then
        print_warning "建议使用Java 17或更高版本以获得最佳性能"
    fi
}

# 检查内存
check_memory() {
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 4096 ]; then
        print_warning "系统内存可能不足4GB，建议增加内存以获得最佳性能"
    fi
    print_success "可用内存: ${TOTAL_MEM}MB"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要目录..."
    mkdir -p logs
    mkdir -p plugins
    mkdir -p world
    mkdir -p world_nether
    mkdir -p world_the_end
    mkdir -p config
    mkdir -p scripts
    print_success "目录创建完成"
}

# 检查文件完整性
check_files() {
    print_info "检查服务器文件完整性..."
    
    if [ ! -f "paper.jar" ]; then
        print_error "未找到paper.jar文件！请确保PaperMC 1.20.1位于服务器目录"
        exit 1
    fi
    
    if [ ! -f "eula.txt" ]; then
        print_error "未找到eula.txt文件！"
        exit 1
    fi
    
    # 检查EULA是否同意
    if ! grep -q "eula=true" eula.txt; then
        print_warning "EULA协议未同意！请编辑eula.txt文件并设置eula=true"
        print_info "可以使用以下命令编辑: nano eula.txt"
        exit 1
    fi
    
    print_success "文件完整性检查通过"
}

# 设置JVM参数
set_jvm_args() {
    JVM_ARGS="-Xms2G -Xmx2G"
    JVM_ARGS="$JVM_ARGS -XX:+UseG1GC"                    # 使用G1垃圾收集器
    JVM_ARGS="$JVM_ARGS -XX:+ParallelRefProcEnabled"     # 启用并行引用处理
    JVM_ARGS="$JVM_ARGS -XX:MaxGCPauseMillis=200"       # 最大GC暂停时间200ms
    JVM_ARGS="$JVM_ARGS -XX:+UnlockExperimentalVMOptions"
    JVM_ARGS="$JVM_ARGS -XX:+DisableExplicitGC"         # 禁用显式GC
    JVM_ARGS="$JVM_ARGS -XX:+AlwaysPreTouch"            # 预分配内存
    JVM_ARGS="$JVM_ARGS -XX:G1NewSizePercent=30"        # 新生代大小
    JVM_ARGS="$JVM_ARGS -XX:G1MaxNewSizePercent=40"     # 最大新生代大小
    JVM_ARGS="$JVM_ARGS -XX:G1HeapRegionSize=8M"        # 堆区域大小
    JVM_ARGS="$JVM_ARGS -XX:ReservedCodeCacheSize=512M" # 代码缓存大小
    JVM_ARGS="$JVM_ARGS -XX:SoftMaxHeapSize=2G"         # 软最大堆大小
    
    print_info "JVM参数: $JVM_ARGS"
}

# 启动服务器
start_server() {
    print_info "启动PaperMC服务器..."
    print_success "=================================="
    print_success "       PaperMC 1.20.1 启动中..."
    print_success "  内存分配: 2GB | 垃圾收集: G1GC"
    print_success "=================================="
    
    # 启动Java进程
    java $JVM_ARGS -jar paper.jar --nogui
}

# 清理函数
cleanup() {
    print_info "正在关闭服务器..."
    # 在这里可以添加清理逻辑，如保存世界数据、关闭数据库连接等
}

# 信号处理
trap cleanup EXIT INT TERM

# 主函数
main() {
    print_info "PaperMC 1.20.1 文明服务器启动脚本"
    print_info "====================================="
    
    check_java
    check_memory
    create_directories
    check_files
    set_jvm_args
    start_server
}

# 执行主函数
main "$@"