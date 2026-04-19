# ================================================
# Minecraft气候事件模拟系统
# PaperMC 1.20.1 服务器
# 创建时间: 2025-11-13
# ================================================
# 
# 系统功能：
# 1. 干旱事件：农作物生长速度降低80%，持续3-5天
# 2. 洪水事件：冲毁低洼建筑和农田
# 3. 僵尸围城：每月满月夜发生，持续30分钟
# 
# 执行间隔：每5分钟检查一次
# ================================================

# 初始化变量
CLIMATE_EVENTS_ENABLED=true
LOG_FILE="/workspace/worlds/minecraft/server/logs/climate_events.log"
EVENTS_DIR="/workspace/worlds/minecraft/server/events"

# 创建必要目录
mkdir -p "$EVENTS_DIR"

# ================================================
# 干旱事件系统
# ================================================

# 执行干旱事件
trigger_drought_event() {
    local duration_minutes=$((RANDOM % 4320 + 2160))  # 3-5天（分钟）
    local event_id="drought_$(date +%Y%m%d_%H%M%S)"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 干旱事件开始，ID: $event_id，预期持续时间: $duration_minutes 分钟" >> "$LOG_FILE"
    
    # 设置干旱状态文件
    cat > "$EVENTS_DIR/current_drought.txt" << EOF
event_id=$event_id
start_time=$(date +%s)
duration=$duration_minutes
crop_growth_multiplier=0.2
original_growth_rate=$(get_original_growth_rate)
EOF
    
    # 应用干旱效果到游戏
    apply_drought_effects
    
    # 开始干旱监控循环
    monitor_drought_event $event_id $duration_minutes
}

# 获取原始生长速度
get_original_growth_rate() {
    # 默认随机tick速度为3
    echo "3"
}

# 应用干旱效果
apply_drought_effects() {
    echo "正在应用干旱效果..."
    
    # 降低作物生长速度80%
    console_command "gamerule randomTickSpeed 1"
    
    # 停止自然降雨
    console_command "weather clear"
    
    # 降低湿度
    console_command "time set day"
    
    # 向玩家广播干旱警告
    broadcast_drought_warning
}

# 广播干旱警告
broadcast_drought_warning() {
    console_command "tellraw @a {\"text\":\"☀️ 干旱警报！☀️\",\"color\":\"gold\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"农作物生长速度降低80%！\",\"color\":\"red\"}"
    console_command "tellraw @a {\"text\":\"请节省水资源和食物储备！\",\"color\":\"yellow\"}"
    console_command "tellraw @a {\"text\":\"预计持续3-5个游戏日\",\"color\":\"gray\"}"
}

# 监控干旱事件
monitor_drought_event() {
    local event_id=$1
    local duration_minutes=$2
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_minutes=$(( (current_time - start_time) / 60 ))
        
        # 检查是否超过持续时间
        if [ $elapsed_minutes -ge $duration_minutes ]; then
            end_drought_event "$event_id"
            break
        fi
        
        # 每30分钟发布一次进度报告
        if [ $((elapsed_minutes % 30)) -eq 0 ]; then
            local remaining_minutes=$((duration_minutes - elapsed_minutes))
            console_command "tellraw @a {\"text\":\"干旱仍在继续... 剩余时间: $remaining_minutes 分钟\",\"color\":\"yellow\"}"
        fi
        
        sleep 300  # 5分钟检查一次
    done
}

# 结束干旱事件
end_drought_event() {
    local event_id=$1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 干旱事件结束，ID: $event_id" >> "$LOG_FILE"
    
    # 恢复正常的作物生长速度
    console_command "gamerule randomTickSpeed 3"
    
    # 删除干旱状态文件
    rm -f "$EVENTS_DIR/current_drought.txt"
    
    # 广播干旱结束
    console_command "tellraw @a {\"text\":\"💧 干旱结束！💧\",\"color\":\"blue\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"农作物生长速度恢复正常\",\"color\":\"green\"}"
}

# ================================================
# 洪水事件系统
# ================================================

# 执行洪水事件
trigger_flood_event() {
    local duration_minutes=$((RANDOM % 2880 + 1440))  # 2-4天（分钟）
    local event_id="flood_$(date +%Y%m%d_%H%M%S)"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 洪水事件开始，ID: $event_id，预期持续时间: $duration_minutes 分钟" >> "$LOG_FILE"
    
    # 设置洪水状态文件
    cat > "$EVENTS_DIR/current_flood.txt" << EOF
event_id=$event_id
start_time=$(date +%s)
duration=$duration_minutes
water_level=5
affected_areas=swamp,river,beach
building_damage_chance=0.3
EOF
    
    # 应用洪水效果
    apply_flood_effects
    
    # 开始洪水监控
    monitor_flood_event $event_id $duration_minutes
}

# 应用洪水效果
apply_flood_effects() {
    echo "正在应用洪水效果..."
    
    # 设定为雨天
    console_command "weather rain"
    
    # 生成大量水源
    generate_water_sources
    
    # 随机破坏低洼地区建筑
    damage_low_lying_buildings
    
    # 广播洪水警告
    broadcast_flood_warning
}

# 生成水源
generate_water_sources() {
    echo "在主要河流区域生成水源..."
    
    # 在河流源头添加水
    console_command "fill ~-20 ~-5 ~-20 ~20 ~-5 ~20 water 0 replace lava"
    console_command "fill ~-15 ~-3 ~-15 ~15 ~-3 ~15 water 0 replace stone"
    
    # 创建新的支流
    console_command "fill ~-30 ~-5 ~-30 ~30 ~-5 ~30 water 0 replace dirt"
    console_command "fill ~-25 ~-2 ~-25 ~25 ~-2 ~25 water 0 replace sand"
}

# 破坏低洼建筑
damage_low_lying_buildings() {
    echo "检查并破坏低洼建筑..."
    
    # 随机选择一些低洼区域
    for i in {1..5}; do
        local x=$((RANDOM % 200 - 100))
        local z=$((RANDOM % 200 - 100))
        local damage_chance=30  # 30%概率
        
        # 检查并破坏建筑
        console_command "fill ~$x ~-5 ~$z ~$((x+10)) ~5 ~$((z+10)) air 0 replace stone"
        console_command "fill ~$x ~-3 ~$z ~$((x+10)) ~3 ~$((z+10)) air 0 replace wood"
        console_command "fill ~$x ~-2 ~$z ~$((x+10)) ~2 ~$((z+10)) air 0 replace planks"
        
        # 通知玩家
        console_command "tellraw @a {\"text\":\"🌊 洪水冲毁了 $x, -5, $z 附近的建筑\",\"color\":\"blue\"}"
    done
}

# 广播洪水警告
broadcast_flood_warning() {
    console_command "tellraw @a {\"text\":\"🌊 洪水警报！🌊\",\"color\":\"blue\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"低洼地区的建筑可能受损！\",\"color\":\"red\"}"
    console_command "tellraw @a {\"text\":\"请迁往高地避难！\",\"color\":\"yellow\"}"
    console_command "tellraw @a {\"text\":\"预计持续2-4个游戏日\",\"color\":\"gray\"}"
}

# 监控洪水事件
monitor_flood_event() {
    local event_id=$1
    local duration_minutes=$2
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_minutes=$(( (current_time - start_time) / 60 ))
        
        if [ $elapsed_minutes -ge $duration_minutes ]; then
            end_flood_event "$event_id"
            break
        fi
        
        # 每60分钟扩展洪水范围
        if [ $((elapsed_minutes % 60)) -eq 0 ]; then
            expand_flood_area
        fi
        
        # 每30分钟发布进度报告
        if [ $((elapsed_minutes % 30)) -eq 0 ]; then
            local remaining_minutes=$((duration_minutes - elapsed_minutes))
            console_command "tellraw @a {\"text\":\"洪水仍在继续... 剩余时间: $remaining_minutes 分钟\",\"color\":\"blue\"}"
        fi
        
        sleep 300  # 5分钟检查一次
    done
}

# 扩展洪水范围
expand_flood_area() {
    echo "扩展洪水影响范围..."
    
    # 随机选择一个方向扩展
    local direction=$((RANDOM % 4 + 1))
    case $direction in
        1) console_command "fill ~-50 ~-5 ~-10 ~50 ~-5 ~10 water 0 replace dirt";;  # 北
        2) console_command "fill ~-50 ~-5 ~-10 ~50 ~-5 ~10 water 0 replace dirt";;  # 南  
        3) console_command "fill ~-10 ~-5 ~-50 ~10 ~-5 ~50 water 0 replace dirt";;  # 西
        4) console_command "fill ~-10 ~-5 ~-50 ~10 ~-5 ~50 water 0 replace dirt";;  # 东
    esac
    
    console_command "tellraw @a {\"text\":\"🌊 洪水范围继续扩大！\",\"color\":\"blue\"}"
}

# 结束洪水事件
end_flood_event() {
    local event_id=$1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 洪水事件结束，ID: $event_id" >> "$LOG_FILE"
    
    # 恢复到正常天气
    console_command "weather clear"
    console_command "time set day"
    
    # 清理多余水源
    console_command "fill ~-100 ~-5 ~-100 ~100 ~-5 ~100 air 0 replace water"
    
    # 删除洪水状态文件
    rm -f "$EVENTS_DIR/current_flood.txt"
    
    # 广播洪水结束
    console_command "tellraw @a {\"text\":\"🌞 洪水消退！🌞\",\"color\":\"green\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"水位正在下降，请注意清理积水\",\"color\":\"blue\"}"
}

# ================================================
# 僵尸围城系统
# ================================================

# 检查是否为满月
is_full_moon() {
    local time_of_day=$(/workspace/worlds/minecraft/server/scripts/get_time_of_day.sh)
    if [ "$time_of_day" = "night" ]; then
        local day_count=$(/workspace/worlds/minecraft/server/scripts/get_day_count.sh)
        # 满月每 2418 个游戏日出现一次（近似现实）
        local full_moon_cycle=2418
        if [ $((day_count % full_moon_cycle)) -eq 0 ]; then
            return 0  # 是满月
        fi
    fi
    return 1  # 不是满月
}

# 执行僵尸围城
trigger_zombie_siege() {
    local event_id="siege_$(date +%Y%m%d_%H%M%S)"
    local duration_minutes=30  # 30分钟
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 僵尸围城事件开始，ID: $event_id，持续时间: $duration_minutes 分钟" >> "$LOG_FILE"
    
    # 设置围城状态文件
    cat > "$EVENTS_DIR/current_siege.txt" << EOF
event_id=$event_id
start_time=$(date +%s)
duration=$duration_minutes
mob_wave_duration=10
mob_count_multiplier=2.0
affected_radius=200
EOF
    
    # 应用围城效果
    apply_siege_effects
    
    # 开始围城监控
    monitor_zombie_siege $event_id $duration_minutes
}

# 应用围城效果
apply_siege_effects() {
    echo "正在应用僵尸围城效果..."
    
    # 确保是夜晚
    console_command "time set night"
    
    # 增强怪物生成
    console_command "gamerule doMobSpawning true"
    console_command "gamerule spawnRadius 10"
    
    # 生成大量僵尸
    spawn_zombie_wave
    
    # 设置NPC进入防御模式
    set_npcs_defensive_mode
    
    # 广播围城警告
    broadcast_siege_warning
}

# 生成僵尸波次
spawn_zombie_wave() {
    local affected_radius=200
    
    echo "生成第一波僵尸..."
    
    # 在玩家附近生成大量僵尸
    for player in $(console_command "list"); do
        console_command "execute at $player run spreadplayers ~ ~ $affected_radius 100 false @e[type=zombie,distance=..$affected_radius]"
        console_command "execute at $player run summon zombie ~10 ~ ~"
        console_command "execute at $player run summon zombie ~-10 ~ ~"
        console_command "execute at $player run summon zombie ~ ~ ~10"
        console_command "execute at $player run summon zombie ~ ~ ~-10"
        console_command "execute at $player run summon zombie ~15 ~ ~15"
        console_command "execute at $player run summon zombie ~-15 ~ ~15"
    done
    
    # 生成特殊强化僵尸
    spawn_enhanced_zombies
}

# 生成强化僵尸
spawn_enhanced_zombies() {
    echo "生成强化僵尸..."
    
    # 生成统帅级僵尸
    console_command "summon zombie ~50 ~ ~ {CustomName:\"僵尸统帅\",CustomNameVisible:true,Health:100.0,Attributes:[{Name:\"minecraft:generic.max_health\",Base:100.0}],Invulnerable:false}"
    
    # 生成群体僵尸
    for i in {1..5}; do
        console_command "summon zombie ~$((i*20)) ~ ~ {CustomName:\"群体僵尸\",CustomNameVisible:true,Health:50.0,Attributes:[{Name:\"minecraft:generic.max_health\",Base:50.0}]}"
    done
}

# 设置NPC防御模式
set_npcs_defensive_mode() {
    echo "设置NPC进入防御模式..."
    
    # 修改商人NPC文本
    console_command "npc modify \"村民农夫\" --text \"危险！僵尸来袭！快购买食物准备避难！\""
    console_command "npc modify \"村庄铁匠\" --text \"紧急锻造！武器需求激增！快来购买装备！\""
    console_command "npc modify \"神秘商人\" --text \"稀有物品大减价！有助于对抗僵尸！\""
}

# 广播围城警告
broadcast_siege_warning() {
    console_command "tellraw @a {\"text\":\"🧟 僵尸围城警报！🧟\",\"color\":\"red\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"满月夜降临，大量僵尸正在集结！\",\"color\":\"yellow\"}"
    console_command "tellraw @a {\"text\":\"请组队防御或寻找安全避难所！\",\"color\":\"red\"}"
    console_command "tellraw @a {\"text\":\"围城持续30分钟\",\"color\":\"gray\"}"
}

# 监控僵尸围城
monitor_zombie_siege() {
    local event_id=$1
    local duration_minutes=$2
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_minutes=$(( (current_time - start_time) / 60 ))
        
        if [ $elapsed_minutes -ge $duration_minutes ]; then
            end_zombie_siege "$event_id"
            break
        fi
        
        # 每10分钟生成新波次僵尸
        if [ $((elapsed_minutes % 10)) -eq 0 ] && [ $elapsed_minutes -gt 0 ]; then
            spawn_zombie_wave
            console_command "tellraw @a {\"text\":\"🧟 新一波僵尸来袭！\",\"color\":\"red\"}"
        fi
        
        # 给予玩家生存奖励
        if [ $elapsed_minutes -eq 15 ]; then
            give_survival_rewards
        fi
        
        # 每5分钟发布进展
        if [ $((elapsed_minutes % 5)) -eq 0 ]; then
            local remaining_minutes=$((duration_minutes - elapsed_minutes))
            console_command "tellraw @a {\"text\":\"僵尸围城进行中... 剩余时间: $remaining_minutes 分钟\",\"color\":\"red\"}"
        fi
        
        sleep 300  # 5分钟检查一次
    done
}

# 给予生存奖励
give_survival_rewards() {
    echo "给予僵尸围城生存奖励..."
    
    # 给仍在生存的玩家奖励
    console_command "give @a[tag=alive] emerald 5"
    console_command "give @a[tag=alive] diamond_sword"
    console_command "give @a[tag=alive] golden_apple 3"
    
    # 标记存活玩家
    console_command "tag @a[tag=alive] remove alive"
}

# 结束僵尸围城
end_zombie_siege() {
    local event_id=$1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 僵尸围城事件结束，ID: $event_id" >> "$LOG_FILE"
    
    # 清除大量僵尸
    console_command "kill @e[type=zombie]"
    
    # 恢复正常的怪物生成
    console_command "gamerule doMobSpawning true"
    console_command "gamerule spawnRadius 8"
    
    # 恢复NPC正常状态
    restore_npc_normal_state
    
    # 删除围城状态文件
    rm -f "$EVENTS_DIR/current_siege.txt"
    
    # 广播围城结束
    console_command "tellraw @a {\"text\":\"🌅 僵尸围城结束！🌅\",\"color\":\"green\",\"bold\":true}"
    console_command "tellraw @a {\"text\":\"恭喜幸存者！奖励已发放\",\"color\":\"yellow\"}"
}

# 恢复NPC正常状态
restore_npc_normal_state() {
    console_command "npc modify \"村民农夫\" --text \"你好！我是一名农民。\\n我用木板换取食物，帮你度过饥饿期。\\n右键点击我来查看价格！\""
    console_command "npc modify \"村庄铁匠\" --text \"欢迎光临！我是村庄铁匠。\\n用煤炭我可以为你锻造最好的工具！\\n右边点击来查看我的锻造技能！\""
    console_command "npc modify \"神秘商人\" --text \"神秘商人: 我有世界上最好的珍宝！\\n钻石、附魔材料... 只要你有足够的绿宝石！\\n快来看看我的宝藏吧！\""
}

# ================================================
# 主系统监控循环
# ================================================

# 主循环函数
main_climate_monitoring() {
    echo "气候事件模拟系统启动..."
    echo "系统将每5分钟检查一次事件触发条件"
    
    while true; do
        check_climate_events
        
        sleep 300  # 5分钟检查一次
    done
}

# 检查气候事件触发条件
check_climate_events() {
    local current_hour=$(date +%H)
    
    # 检查是否有正在进行的干旱
    if [ ! -f "$EVENTS_DIR/current_drought.txt" ]; then
        # 随机触发干旱事件（低概率）
        local drought_chance=5  # 5%概率
        if [ $((RANDOM % 100)) -lt $drought_chance ]; then
            trigger_drought_event
        fi
    fi
    
    # 检查是否有正在进行的洪水
    if [ ! -f "$EVENTS_DIR/current_flood.txt" ]; then
        # 随机触发洪水事件（低概率）
        local flood_chance=3  # 3%概率
        if [ $((RANDOM % 100)) -lt $flood_chance ]; then
            trigger_flood_event
        fi
    fi
    
    # 检查满月僵尸围城
    if [ ! -f "$EVENTS_DIR/current_siege.txt" ] && is_full_moon; then
        trigger_zombie_siege
    fi
}

# 写入控制台命令（示例函数，需要根据实际服务器API调整）
console_command() {
    echo "执行控制台命令: $1"
    # 这里需要实现实际的Minecraft服务器控制台交互
    # 可以使用rc接口、send_command端点或文件监控等方式
}

# 显示帮助信息
show_climate_help() {
    echo "Minecraft气候事件模拟系统使用指南"
    echo "=================================="
    echo ""
    echo "事件类型："
    echo "  - 干旱事件：农作物生长速度降低80%，持续3-5天"
    echo "  - 洪水事件：随机破坏低洼建筑，持续2-4天"
    echo "  - 僵尸围城：每月满月夜发生，持续30分钟"
    echo ""
    echo "使用方法："
    echo "  bash climate_events.sh start     # 启动气候事件系统"
    echo "  bash climate_events.sh drought   # 手动触发干旱事件"
    echo "  bash climate_events.sh flood     # 手动触发洪水事件"
    echo "  bash climate_events.sh siege     # 手动触发僵尸围城"
    echo "  bash climate_events.sh status    # 检查当前事件状态"
    echo "  bash climate_events.sh help      # 显示帮助信息"
}

# 检查当前事件状态
check_climate_status() {
    echo "当前气候事件状态："
    
    if [ -f "$EVENTS_DIR/current_drought.txt" ]; then
        echo "  干旱事件：正在发生"
        cat "$EVENTS_DIR/current_drought.txt" | grep -E "duration|crop_growth_multiplier"
    else
        echo "  干旱事件：无"
    fi
    
    if [ -f "$EVENTS_DIR/current_flood.txt" ]; then
        echo "  洪水事件：正在发生"
        cat "$EVENTS_DIR/current_flood.txt" | grep -E "duration|water_level"
    else
        echo "  洪水事件：无"
    fi
    
    if [ -f "$EVENTS_DIR/current_siege.txt" ]; then
        echo "  僵尸围城：正在发生"
        cat "$EVENTS_DIR/current_siege.txt" | grep -E "duration|affected_radius"
    else
        echo "  僵尸围城：无"
    fi
}

# 主程序入口
case "$1" in
    "start")
        main_climate_monitoring
        ;;
    "drought")
        trigger_drought_event
        ;;
    "flood")
        trigger_flood_event
        ;;
    "siege")
        trigger_zombie_siege
        ;;
    "status")
        check_climate_status
        ;;
    "help"|*)
        show_climate_help
        ;;
esac