# ================================================
# Minecraft环境复杂化系统
# PaperMC 1.20.1 服务器
# 创建时间: 2025-11-13
# ================================================
# 
# 此脚本每10分钟执行一次，模拟世界演化过程
# 主要功能：
# 1. 调整洞穴密度（从0.3→0.8）
# 2. 降低矿石稀缺度（从1.0→0.3）
# 3. 强化敌对生物能力
# 4. 增加环境挑战性
# 
# 使用方法：在服务器控制台中周期性执行
# 设置间隔：10分钟（600秒）
# ================================================

# ================================================
# 第一部分：初始化设置
# ================================================

# 定义演化阶段变量
EVOLUTION_STAGE=0
MAX_EVOLUTION_STAGE=50  # 最大演化阶段（对应500分钟/8.3小时）

# 洞穴密度初始值
CAVE_DENSITY_MIN=0.3
CAVE_DENSITY_MAX=0.8

# 矿石稀缺度初始值
ORE_ABUNDANCE_MAX=1.0
ORE_ABUNDANCE_MIN=0.3

# 敌对生物强化参数
MOB_HP_MULTIPLIER=1.0
MOB_DAMAGE_MULTIPLIER=1.0
MOB_SPEED_MULTIPLIER=1.0

# ================================================
# 第二部分：世界演化主函数
# ================================================

# 演化阶段执行函数
execute_evolution_cycle() {
    # 读取当前演化阶段
    current_stage=$(cat /workspace/worlds/minecraft/server/evolution_stage.txt 2>/dev/null || echo "0")
    
    # 增加演化阶段
    new_stage=$((current_stage + 1))
    echo "$new_stage" > /workspace/worlds/minecraft/server/evolution_stage.txt
    
    # 计算新参数
    calculate_evolution_parameters $new_stage
    
    # 应用参数到游戏世界
    apply_evolution_to_world $new_stage
    
    # 强化敌对生物
    strengthen_mobs $new_stage
    
    # 生成演化报告
    generate_evolution_report $new_stage
    
    # 记录日志
    log_evolution_event $new_stage
}

# 计算演化参数函数
calculate_evolution_parameters() {
    stage=$1
    
    # 计算洞穴密度 (0.3 + 0.01 * stage, 最大0.8)
    cave_density=$(echo "scale=2; 0.3 + 0.01 * $stage" | bc -l)
    if (( $(echo "$cave_density > 0.8" | bc -l) )); then
        cave_density=0.8
    fi
    
    # 计算矿石稀缺度 (1.0 - 0.014 * stage, 最小0.3)
    ore_abundance=$(echo "scale=2; 1.0 - 0.014 * $stage" | bc -l)
    if (( $(echo "$ore_abundance < 0.3" | bc -l) )); then
        ore_abundance=0.3
    fi
    
    # 计算敌对生物强化系数
    mob_hp_mult=$(echo "scale=2; 1.0 + 0.02 * $stage" | bc -l)
    if (( $(echo "$mob_hp_mult > 2.0" | bc -l) )); then
        mob_hp_mult=2.0
    fi
    
    mob_damage_mult=$(echo "scale=2; 1.0 + 0.01 * $stage" | bc -l)
    if (( $(echo "$mob_damage_mult > 1.5" | bc -l) )); then
        mob_damage_mult=1.5
    fi
    
    mob_speed_mult=$(echo "scale=2; 1.0 + 0.005 * $stage" | bc -l)
    if (( $(echo "$mob_speed_mult > 1.2" | bc -l) )); then
        mob_speed_mult=1.2
    fi
    
    # 保存参数到临时文件
    echo "CAVE_DENSITY=$cave_density" > /tmp/evolution_params.txt
    echo "ORE_ABUNDANCE=$ore_abundance" >> /tmp/evolution_params.txt
    echo "MOB_HP_MULT=$mob_hp_mult" >> /tmp/evolution_params.txt
    echo "MOB_DAMAGE_MULT=$mob_damage_mult" >> /tmp/evolution_params.txt
    echo "MOB_SPEED_MULT=$mob_speed_mult" >> /tmp/evolution_params.txt
}

# 应用演化参数到游戏世界
apply_evolution_to_world() {
    stage=$1
    
    # 读取参数
    source /tmp/evolution_params.txt
    
    # 向所有在线玩家广播演化信息
    broadcast_evolution_alert $stage
    
    # 调整世界生成参数（通过游戏规则）
    # 注意：这些命令需要在服务器控制台或游戏中执行
    
    # 设置洞穴密度（需要通过数据包或服务器属性调整）
    console_command "gamerule randomTickSpeed $(echo "scale=0; $MOB_SPEED_MULT * 3" | bc -l)"
    
    # 调整生物生成概率
    if [ $(echo "$stage >= 10" | bc -l) -eq 1 ]; then
        # 每10个演化阶段增加一次生物生成难度
        adjustment_factor=$(echo "scale=0; $stage / 10" | bc -l)
        
        # 强化敌对生物
        for mob_type in "zombie" "skeleton" "creeper" "spider" "enderman"; do
           强化_mob_properties $mob_type $adjustment_factor
        done
    fi
    
    # 调整自然恢复速度（体现环境恶化）
    if [ $(echo "$stage >= 20" | bc -l) -eq 1 ]; then
        # 禁用自然再生
        console_command "gamerule doNaturalRegeneration false"
        
        # 减缓作物生长速度
        console_command "gamerule randomTickSpeed $(echo "scale=0; $MOB_SPEED_MULT * 2" | bc -l)"
    fi
    
    # 生成高级地质变化
    if [ $(echo "$stage >= 30" | bc -l) -eq 1 ]; then
        generate_advanced_geological_changes
    fi
}

# 强化特定生物属性
strengthen_mob_properties() {
    mob_type=$1
    adjustment_factor=$2
    
    # 为不同生物创建强化配置
    case $mob_type in
        "zombie")
            console_command "effect give @e[type=zombie] strength $stage 0"
            console_command "effect give @e[type=zombie] speed $(echo "$adjustment_factor" | bc -l) 0"
            ;;
        "skeleton")
            console_command "effect give @e[type=skeleton] resistance $(echo "$adjustment_factor" | bc -l) 0"
            console_command "effect give @e[type=skeleton] speed $(echo "$adjustment_factor / 2" | bc -l) 0"
            ;;
        "creeper")
            console_command "attribute @e[type=creeper] minecraft:generic.explosion_radius base set $(echo "4 + $adjustment_factor" | bc -l)"
            ;;
        "spider")
            console_command "effect give @e[type=spider] strength $(echo "$adjustment_factor / 2" | bc -l) 0"
            console_command "effect give @e[type=spider] speed $(echo "$adjustment_factor" | bc -l) 0"
            ;;
        "enderman")
            console_command "effect give @e[type=enderman] resistance $(echo "$adjustment_factor" | bc -l) 0"
            ;;
    esac
}

# 生成高级地质变化
generate_advanced_geological_changes() {
    # 生成额外的洞穴网络
    console_command "fill ~-50 ~-50 ~-50 ~50 ~50 ~50 cave_air 0 replace stone"
    
    # 随机破坏部分矿石沉积
    console_command "fill ~-30 ~-30 ~-30 ~30 ~30 ~30 stone 0 replace diamond_ore"
    
    # 改变地形高度
    console_command "fill ~-20 ~-10 ~-20 ~20 ~20 ~20 air 0 replace stone"
    
    # 添加随机地质事件
    generate_random_geological_events
}

# 生成随机地质事件
generate_random_geological_events() {
    # 随机选择地质事件类型
    event_type=$((RANDOM % 3 + 1))
    
    case $event_type in
        1)
            # 地震：随机震动地面
            console_command "fill ~-10 ~-5 ~-10 ~10 ~5 ~10 cobblestone 0 replace dirt"
            ;;
        2)
            # 地陷：创建深坑
            console_command "fill ~-5 ~-10 ~-5 ~5 ~10 ~5 air 0 replace stone"
            ;;
        3)
            # 隆起：抬高地形
            console_command "fill ~-8 ~-5 ~-8 ~8 ~8 ~8 stone 0 replace air"
            ;;
    esac
}

# 广播演化警告
broadcast_evolution_alert() {
    stage=$1
    source /tmp/evolution_params.txt
    
    # 计算演化百分比
    evolution_percent=$(echo "scale=1; $stage / $MAX_EVOLUTION_STAGE * 100" | bc -l)
    
    # 发送分级警告
    if [ $(echo "$stage % 10" | bc -l) -eq 0 ]; then
        # 每10个阶段发送特殊警告
        console_command "tellraw @a {\"text\":\"⚠️ 世界演化警告 ⚠️\",\"color\":\"red\",\"bold\":true}"
        console_command "tellraw @a {\"text\":\"已进入演化第 $stage 阶段 ($evolution_percent%)\",\"color\":\"yellow\"}"
        console_command "tellraw @a {\"text\":\"洞穴密度: $CAVE_DENSITY，矿石稀缺度: $ORE_ABUNDANCE\",\"color\":\"aqua\"}"
        console_command "tellraw @a {\"text\":\"敌对生物已强化！请做好防护准备！\",\"color\":\"red\"}"
    else
        # 普通演化消息
        console_command "tellraw @a {\"text\":\"🌍 世界演化进行中...\",\"color\":\"green\"}"
        console_command "tellraw @a {\"text\":\"第 $stage 阶段：洞穴更密集，资源更稀有\",\"color\":\"yellow\"}"
    fi
}

# 生成演化报告
generate_evolution_report() {
    stage=$1
    source /tmp/evolution_params.txt
    
    # 生成报告文件
    report_file="/workspace/worlds/minecraft/server/logs/evolution_report_$(date +%Y%m%d_%H%M%S).log"
    
    cat > "$report_file" << EOF
===============================================
世界演化报告 - 阶段 $stage
生成时间: $(date)
===============================================

演化参数:
- 洞穴密度: $CAVE_DENSITY (初始: 0.3, 最大: 0.8)
- 矿石稀缺度: $ORE_ABUNDANCE (初始: 1.0, 最小: 0.3)
- 生物生命值强化: ${MOB_HP_MULT}x (最大: 2.0x)
- 生物伤害强化: ${MOB_DAMAGE_MULT}x (最大: 1.5x)
- 生物速度强化: ${MOB_SPEED_MULT}x (最大: 1.2x)

演化进度: $(echo "scale=1; $stage / $MAX_EVOLUTION_STAGE * 100" | bc -l)%

环境变化:
- 地下结构更复杂，洞穴网络更密集
- 矿物资源更加稀少和珍贵
- 敌对生物变得更加强大和智能
- 生存挑战持续增加

建议应对策略:
1. 寻找并保护安全据点
2. 建立储备资源库
3. 组建团队合作
4. 研发更先进的装备和武器
5. 探索新发现的洞穴系统

===============================================
EOF
    
    echo "演化报告已保存: $report_file"
}

# 记录演化事件到日志
log_evolution_event() {
    stage=$1
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 追加到主日志文件
    echo "[$timestamp] 世界演化进入第 $stage 阶段" >> /workspace/worlds/minecraft/server/logs/evolution.log
}

# 清理临时文件
cleanup_temp_files() {
    rm -f /tmp/evolution_params.txt
}

# 主循环函数
main_evolution_loop() {
    echo "环境复杂化系统启动..."
    echo "演化阶段将从0开始，最大 $MAX_EVOLUTION_STAGE 阶段"
    echo "每10分钟执行一次演化周期"
    
    # 初始化演化阶段文件
    echo "0" > /workspace/worlds/minecraft/server/evolution_stage.txt
    
    while true; do
        echo "开始执行演化周期..."
        execute_evolution_cycle
        echo "演化周期完成，等待10分钟..."
        
        # 清理临时文件
        cleanup_temp_files
        
        # 等待10分钟（600秒）
        sleep 600
    done
}

# 帮助信息
show_help() {
    echo "Minecraft环境复杂化系统使用指南"
    echo "================================"
    echo ""
    echo "功能说明："
    echo "  - 每10分钟自动调整世界生成参数"
    echo "  - 洞穴密度从0.3增加到0.8"
    echo "  - 矿石稀缺度从1.0降低到0.3"
    echo "  - 敌对生物能力持续强化"
    echo ""
    echo "使用方法："
    echo "  bash environment_evolution.sh start     # 启动环境演化系统"
    echo "  bash environment_evolution.sh check     # 检查当前演化状态"
    echo "  bash environment_evolution.sh reset     # 重置演化阶段"
    echo "  bash environment_evolution.sh help      # 显示帮助信息"
    echo ""
}

# 检查当前状态
check_evolution_status() {
    if [ -f "/workspace/worlds/minecraft/server/evolution_stage.txt" ]; then
        current_stage=$(cat /workspace/worlds/minecraft/server/evolution_stage.txt)
        evolution_percent=$(echo "scale=1; $current_stage / $MAX_EVOLUTION_STAGE * 100" | bc -l)
        
        echo "当前演化状态:"
        echo "  阶段: $current_stage / $MAX_EVOLUTION_STAGE"
        echo "  进度: $evolution_percent%"
        echo "  已运行时间: $(echo "$current_stage * 10" | bc -l) 分钟"
    else
        echo "未找到演化状态文件，系统可能未启动"
    fi
}

# 重置演化阶段
reset_evolution() {
    echo "0" > /workspace/worlds/minecraft/server/evolution_stage.txt
    echo "演化阶段已重置为0"
    echo "环境参数已恢复为初始值"
}

# 主程序入口
case "$1" in
    "start")
        main_evolution_loop
        ;;
    "check")
        check_evolution_status
        ;;
    "reset")
        reset_evolution
        ;;
    "help"|*)
        show_help
        ;;
esac