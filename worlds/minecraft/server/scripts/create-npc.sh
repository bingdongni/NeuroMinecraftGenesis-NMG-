# ================================================
# Citizens NPC创建脚本
# PaperMC 1.20.1 服务器
# 创建时间: 2025-11-13
# ================================================

# 本文件包含在服务器启动后执行的NPC创建命令
# 请在游戏内使用 /npc 命令或通过控制台执行这些命令

# ================================================
# 第一部分：创建基础NPC
# ================================================

# 创建农民NPC
# ================================================
# 命令含义：
# create: 创建一个新NPC
# --name: 设置NPC名称
# --type: 设置NPC类型
# --traits: 设置NPC特质
# --trait: trait参数可以添加多个

/npc create "村民农夫" --type player --traits text,speaker,merchant

# 为农民NPC设置外观和基本属性
/npc modify "村民农夫" --lookat
/npc modify "村民农夫" --talk-close
/npc modify "村民农夫" --text "&e你好！我是一名农民。&f\n&7我用木板换取食物，帮你度过饥饿期。\n&e右键点击我来查看价格！"

/npc modify "村民农夫" --trait trade \
  --trade wood 8 bread 3 \
  --trade wood 16 golden_apple 1 \
  --trade wood 32 bread 15

# 创建铁匠NPC
# ================================================
/npc create "村庄铁匠" --type player --traits text,speaker,merchant

# 为铁匠NPC设置外观和基本属性
/npc modify "村庄铁匠" --lookat
/npc modify "村庄铁匠" --talk-close
/npc modify "村庄铁匠" --text "&e欢迎光临！我是村庄铁匠。&f\n&7用煤炭我可以为你锻造最好的工具！\n&e右边点击来查看我的锻造技能！"

/npc modify "村庄铁匠" --trait trade \
  --trade coal 8 iron_pickaxe 1 \
  --trade coal 16 iron_sword 1 \
  --trade coal 24 iron_helmet 1

# 创建物资商人NPC
# ================================================
/npc create "神秘商人" --type player --traits text,speaker,merchant

# 为商人NPC设置外观和基本属性
/npc modify "神秘商人" --lookat
/npc modify "神秘商人" --talk-close
/npc modify "神秘商人" --text "&a神秘商人:&f 我有世界上最好的珍宝！&f\n&7钻石、附魔材料... 只要你有足够的绿宝石！\n&e快来看看我的宝藏吧！"

/npc modify "神秘商人" --trait trade \
  --trade emerald 1 diamond 3 \
  --trade diamond 3 netherite_ingot 1 \
  --trade gold_ingot 10 totem_of_undying 1

# ================================================
# 第二部分：创建高级管理NPC
# ================================================

# 创建世界守护者（管理员NPC）
/npc create "世界守护者" --type player --traits text,speaker,weather,spawning

/npc modify "世界守护者" --text "&b世界守护者:&f 我守护着这个世界的平衡。&f\n&7我可以控制天气、调节世界难度。\n&c请谨慎使用我的力量！"

/npc modify "世界守护者" --permission "world.guardian"
/npc modify "世界守护者" --lookat

# 创建环境监控员（生态管理NPC）
/npc create "生态监控员" --type player --traits text,speaker,environment

/npc modify "生态监控员" --text "&2生态监控员:&f 监测着世界的演化进程。&f\n&7每当世界演化发生时，我会发布警告。\n&e准备好迎接更严酷的环境吧！"

/npc modify "生态监控员" --permission "environment.monitor"

# ================================================
# 第三部分：NPC外观和位置设置
# ================================================

# 为农民NPC设置村民外观
/npc modify "村民农夫" --skin "farmer"
/npc modify "村民农夫" --gear "wooden_hoe"
/npc modify "村民农夫" --color "brown" --hat "brown"

# 为铁匠NPC设置铁匠外观
/npc modify "村庄铁匠" --skin "blacksmith"
/npc modify "村庄铁匠" --gear "iron_pickaxe"
/npc modify "村庄铁匠" --color "gray" --apron "black"

# 为商人NPC设置商人外观
/npc modify "神秘商人" --skin "merchant"
/npc modify "神秘商人" --gear "emerald"
/npc modify "神秘商人" --color "green" --coat "green"

# ================================================
# 第四部分：NPC群体和智能设置
# ================================================

# 设置NPC群体
/npc group create "商人"
/npc group assign "村民农夫" "商人"
/npc group assign "村庄铁匠" "商人"
/npc group assign "神秘商人" "商人"

/npc group create "管理者"
/npc group assign "世界守护者" "管理者"
/npc group assign "生态监控员" "管理者"

# 设置NPC智能
/npc modify "村民农夫" --smart-talk true
/npc modify "村庄铁匠" --smart-talk true
/npc modify "神秘商人" --smart-talk true
/npc modify "世界守护者" --smart-talk true

# 设置NPC记忆功能
/npc modify "村民农夫" --remember-players true
/npc modify "村庄铁匠" --remember-players true
/npc modify "神秘商人" --remember-players true

# ================================================
# 第五部分：NPC防沉浸功能
# ================================================

# 设置NPC不透明化
/npc modify "村民农夫" --ghost false
/npc modify "村庄铁匠" --ghost false
/npc modify "神秘商人" --ghost false

# 设置NPC交互动画
/npc modify "村民农夫" --animation-wave
/npc modify "村庄铁匠" --animation-wave
/npc modify "神秘商人" --animation-wave

# ================================================
# 第六部分：保存配置
# ================================================

# 保存所有NPC到文件
/npc save citizens-npc-config.yml

# 输出完成信息
console: "NPC系统创建完成！共创建了6个NPC："
console: "- 村民农夫：木板换取食物商人"
console: "- 村庄铁匠：煤炭换取工具商人"
console: "- 神秘商人：珍贵材料商人"
console: "- 世界守护者：天气和环境管理者"
console: "- 生态监控员：演化监控系统"