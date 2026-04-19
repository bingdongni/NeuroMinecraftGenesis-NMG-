# ================================================
# Minecraft文明服务器总控脚本
# PaperMC 1.20.1 服务器管理工具
# 创建时间: 2025-11-13
# ================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 服务器目录
SERVER_DIR="/workspace/worlds/minecraft/server"
PID_FILE="$SERVER_DIR/server.pid"
EVOLUTION_PID_FILE="$SERVER_DIR/evolution.pid"
CLIMATE_PID_FILE="$SERVER_DIR/climate.pid"

# 打印函数
print_header() {
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

# 检查服务器目录
check_server_directory() {
    if [ ! -d "$SERVER_DIR" ]; then
        print_error "服务器目录不存在: $SERVER_DIR"
        exit 1
    fi
    
    cd "$SERVER_DIR" || exit 1
}

# 检查Java环境
check_java() {
    if ! command -v java &> /dev/null; then
        print_error "Java未安装！请安装Java 17或更高版本"
        exit 1
    fi
    
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    print_success "Java版本: $JAVA_VERSION"
}

# 启动服务器
start_server() {
    print_header "启动PaperMC服务器"
    
    # 检查服务器是否已在运行
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_warning "服务器已在运行中"
        return 1
    fi
    
    # 检查必需文件
    if [ ! -f "paper.jar" ]; then
        print_error "未找到paper.jar文件！"
        return 1
    fi
    
    if [ ! -f "eula.txt" ]; then
        print_error "未找到eula.txt文件！"
        return 1
    fi
    
    if ! grep -q "eula=true" eula.txt; then
        print_error "EULA协议未同意！请编辑eula.txt文件并设置eula=true"
        return 1
    fi
    
    # 创建必要目录
    mkdir -p logs plugins world
    
    # 给启动脚本执行权限
    chmod +x start.sh
    
    # 启动服务器
    print_info "正在启动服务器..."
    nohup ./start.sh > logs/server.log 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 5
    
    # 检查启动状态
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_success "服务器启动成功！PID: $(cat $PID_FILE)"
        print_info "服务器日志: logs/server.log"
        print_info "控制台: tail -f logs/server.log"
    else
        print_error "服务器启动失败！"
        return 1
    fi
}

# 停止服务器
stop_server() {
    print_header "停止PaperMC服务器"
    
    if [ ! -f "$PID_FILE" ]; then
        print_warning "服务器未在运行"
        return 1
    fi
    
    local server_pid=$(cat "$PID_FILE")
    
    if ! kill -0 "$server_pid" 2>/dev/null; then
        print_warning "服务器进程不存在，清理PID文件"
        rm -f "$PID_FILE"
        return 1
    fi
    
    print_info "正在停止服务器 (PID: $server_pid)..."
    
    # 优雅停止
    kill -TERM "$server_pid"
    
    # 等待10秒
    local count=0
    while kill -0 "$server_pid" 2>/dev/null && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # 如果还在运行，强制杀死
    if kill -0 "$server_pid" 2>/dev/null; then
        print_warning "优雅停止失败，强制停止..."
        kill -KILL "$server_pid"
    fi
    
    # 清理文件
    rm -f "$PID_FILE"
    
    print_success "服务器已停止"
}

# 重启服务器
restart_server() {
    print_header "重启PaperMC服务器"
    stop_server
    sleep 2
    start_server
}

# 检查服务器状态
check_server_status() {
    print_header "服务器状态检查"
    
    echo -e "${BLUE}服务器状态:${NC}"
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        local server_pid=$(cat "$PID_FILE")
        local runtime=$(ps -p "$server_pid" -o etime= 2>/dev/null || echo "未知")
        echo -e "  ${GREEN}● 运行中${NC} (PID: $server_pid, 运行时间: $runtime)"
        
        # 检查系统服务
        if [ -f "$EVOLUTION_PID_FILE" ] && kill -0 $(cat "$EVOLUTION_PID_FILE") 2>/dev/null; then
            echo -e "  ${GREEN}● 环境演化系统${NC} 运行中"
        else
            echo -e "  ${RED}● 环境演化系统${NC} 未运行"
        fi
        
        if [ -f "$CLIMATE_PID_FILE" ] && kill -0 $(cat "$CLIMATE_PID_FILE") 2>/dev/null; then
            echo -e "  ${GREEN}● 气候事件系统${NC} 运行中"
        else
            echo -e "  ${RED}● 气候事件系统${NC} 未运行"
        fi
    else
        echo -e "  ${RED}● 未运行${NC}"
        rm -f "$PID_FILE"
    fi
    
    echo ""
    echo -e "${BLUE}系统信息:${NC}"
    echo "  Java版本: $(java -version 2>&1 | head -1)"
    echo "  服务器目录: $SERVER_DIR"
    echo "  当前时间: $(date)"
    echo "  负载情况: $(uptime | awk -F'load average:' '{print $2}')"
}

# 启动环境演化系统
start_evolution_system() {
    print_header "启动环境演化系统"
    
    if [ -f "$EVOLUTION_PID_FILE" ] && kill -0 $(cat "$EVOLUTION_PID_FILE") 2>/dev/null; then
        print_warning "环境演化系统已在运行中"
        return 1
    fi
    
    # 给脚本执行权限
    chmod +x scripts/environment_evolution.sh
    
    # 启动系统
    print_info "正在启动环境演化系统..."
    nohup bash scripts/environment_evolution.sh start > logs/evolution.log 2>&1 &
    echo $! > "$EVOLUTION_PID_FILE"
    
    sleep 2
    
    if [ -f "$EVOLUTION_PID_FILE" ] && kill -0 $(cat "$EVOLUTION_PID_FILE") 2>/dev/null; then
        print_success "环境演化系统启动成功！PID: $(cat $EVOLUTION_PID_FILE)"
        print_info "演化日志: logs/evolution.log"
    else
        print_error "环境演化系统启动失败！"
        return 1
    fi
}

# 启动气候事件系统
start_climate_system() {
    print_header "启动气候事件系统"
    
    if [ -f "$CLIMATE_PID_FILE" ] && kill -0 $(cat "$CLIMATE_PID_FILE") 2>/dev/null; then
        print_warning "气候事件系统已在运行中"
        return 1
    fi
    
    # 给脚本执行权限
    chmod +x scripts/climate_events.sh
    
    # 启动系统
    print_info "正在启动气候事件系统..."
    nohup bash scripts/climate_events.sh start > logs/climate_events.log 2>&1 &
    echo $! > "$CLIMATE_PID_FILE"
    
    sleep 2
    
    if [ -f "$CLIMATE_PID_FILE" ] && kill -0 $(cat "$CLIMATE_PID_FILE") 2>/dev/null; then
        print_success "气候事件系统启动成功！PID: $(cat $CLIMATE_PID_FILE)"
        print_info "气候事件日志: logs/climate_events.log"
    else
        print_error "气候事件系统启动失败！"
        return 1
    fi
}

# 启动所有系统
start_all_systems() {
    print_header "启动所有系统"
    
    start_server
    start_evolution_system
    start_climate_system
    
    print_success "所有系统启动完成！"
    print_info "查看状态: $0 status"
}

# 停止所有系统
stop_all_systems() {
    print_header "停止所有系统"
    
    # 停止系统服务
    if [ -f "$CLIMATE_PID_FILE" ] && kill -0 $(cat "$CLIMATE_PID_FILE") 2>/dev/null; then
        print_info "停止气候事件系统..."
        kill $(cat "$CLIMATE_PID_FILE")
        rm -f "$CLIMATE_PID_FILE"
    fi
    
    if [ -f "$EVOLUTION_PID_FILE" ] && kill -0 $(cat "$EVOLUTION_PID_FILE") 2>/dev/null; then
        print_info "停止环境演化系统..."
        kill $(cat "$EVOLUTION_PID_FILE")
        rm -f "$EVOLUTION_PID_FILE"
    fi
    
    # 停止服务器
    stop_server
    
    print_success "所有系统已停止"
}

# 初始化NPC
init_npcs() {
    print_header "初始化NPC系统"
    
    if [ ! -f "$PID_FILE" ] || ! kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_error "服务器未运行，无法初始化NPC"
        return 1
    fi
    
    # 给脚本执行权限
    chmod +x scripts/create-npc.sh
    
    print_info "正在创建NPC..."
    print_warning "请在游戏内使用 /npc 命令来执行NPC创建命令"
    print_info "或者将 scripts/create-npc.sh 中的命令复制到控制台执行"
    
    # 这里可以添加实际的NPC创建逻辑
    print_info "NPC初始化命令已在 scripts/create-npc.sh 中准备"
}

# 查看日志
view_logs() {
    local log_type=${1:-"server"}
    
    case "$log_type" in
        "server")
            if [ -f "logs/server.log" ]; then
                tail -f logs/server.log
            else
                print_error "服务器日志不存在"
            fi
            ;;
        "evolution")
            if [ -f "logs/evolution.log" ]; then
                tail -f logs/evolution.log
            else
                print_error "演化日志不存在"
            fi
            ;;
        "climate")
            if [ -f "logs/climate_events.log" ]; then
                tail -f logs/climate_events.log
            else
                print_error "气候事件日志不存在"
            fi
            ;;
        "all")
            tail -f logs/*.log
            ;;
        *)
            echo "可用日志类型: server, evolution, climate, all"
            ;;
    esac
}

# 备份系统
backup_system() {
    print_header "系统备份"
    
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_dir="backups/backup_$backup_date"
    
    mkdir -p "$backup_dir"
    
    print_info "正在创建备份..."
    
    # 备份世界数据
    if [ -d "world" ]; then
        print_info "备份世界数据..."
        tar -czf "$backup_dir/world.tar.gz" world/
    fi
    
    # 备份配置文件
    print_info "备份配置文件..."
    cp server.properties "$backup_dir/"
    cp eula.txt "$backup_dir/"
    cp -r config "$backup_dir/"
    
    # 备份插件
    if [ -d "plugins" ]; then
        print_info "备份插件..."
        tar -czf "$backup_dir/plugins.tar.gz" plugins/
    fi
    
    # 备份日志
    if [ -d "logs" ]; then
        print_info "备份日志..."
        tar -czf "$backup_dir/logs.tar.gz" logs/
    fi
    
    print_success "备份完成: $backup_dir"
    print_info "备份大小: $(du -sh "$backup_dir" | cut -f1)"
}

# 清理系统
clean_system() {
    print_header "系统清理"
    
    print_warning "这将清理临时文件和旧日志，确定继续吗？ (y/N)"
    read -r confirm
    
    if [ "$confirm" != "y" ]; then
        print_info "取消清理操作"
        return
    fi
    
    # 清理旧日志（保留7天）
    print_info "清理7天前的日志文件..."
    find logs/ -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    
    # 清理事件临时文件
    print_info "清理事件临时文件..."
    rm -rf events/* 2>/dev/null || true
    
    # 清理临时文件
    print_info "清理临时文件..."
    rm -rf tmp/* 2>/dev/null || true
    
    # 清理空目录
    print_info "清理空目录..."
    find . -type d -empty -delete 2>/dev/null || true
    
    print_success "系统清理完成"
}

# 显示帮助信息
show_help() {
    print_header "Minecraft文明服务器管理工具"
    
    echo -e "${CYAN}可用命令:${NC}"
    echo "  $0 start              - 启动服务器"
    echo "  $0 stop               - 停止服务器"
    echo "  $0 restart            - 重启服务器"
    echo "  $0 status             - 检查系统状态"
    echo ""
    echo -e "${CYAN}系统控制:${NC}"
    echo "  $0 start-all          - 启动所有系统"
    echo "  $0 stop-all           - 停止所有系统"
    echo "  $0 start-evolution    - 启动环境演化系统"
    echo "  $0 start-climate      - 启动气候事件系统"
    echo ""
    echo -e "${CYAN}系统管理:${NC}"
    echo "  $0 init-npcs          - 初始化NPC系统"
    echo "  $0 backup             - 创建系统备份"
    echo "  $0 clean              - 清理临时文件"
    echo "  $0 logs [type]        - 查看日志 (server/evolution/climate/all)"
    echo ""
    echo -e "${CYAN}示例:${NC}"
    echo "  $0 start-all          # 启动所有服务"
    echo "  $0 logs evolution     # 查看演化日志"
    echo "  $0 backup             # 创建备份"
}

# 主程序入口
main() {
    check_server_directory
    check_java
    
    case "$1" in
        "start")
            start_server
            ;;
        "stop")
            stop_server
            ;;
        "restart")
            restart_server
            ;;
        "status")
            check_server_status
            ;;
        "start-all")
            start_all_systems
            ;;
        "stop-all")
            stop_all_systems
            ;;
        "start-evolution")
            start_evolution_system
            ;;
        "start-climate")
            start_climate_system
            ;;
        "init-npcs")
            init_npcs
            ;;
        "backup")
            backup_system
            ;;
        "clean")
            clean_system
            ;;
        "logs")
            view_logs "$2"
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo "使用 '$0 help' 查看可用命令"
            exit 1
            ;;
    esac
}

# 执行主程序
main "$@"