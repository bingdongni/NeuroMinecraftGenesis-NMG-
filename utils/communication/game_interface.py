#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏接口工具类
为Mineflayer桥接系统提供高级游戏操作接口

核心功能：
- 游戏状态管理和查询
- 路径规划和导航
- 物品栏管理和交易
- 方块操作和建造
- 战斗系统集成
- 技能库管理
- 性能监控和优化
"""

import asyncio
import math
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..single.websocket_client import WebSocketBridge, ActionCommand, ActionType, Position, InventoryItem, GameEvent, ActionSequence

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """方块类型枚举"""
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    WOOD = 4
    WATER = 8
    LAVA = 10
    COBBLESTONE = 4
    PLANK = 5
    SAND = 12
    GRAVEL = 13
    COAL_ORE = 16
    IRON_ORE = 15
    GOLD_ORE = 14
    DIAMOND_ORE = 56
    REDSTONE_ORE = 73
    LAPIS_ORE = 21


class ItemType(Enum):
    """物品类型枚举"""
    WOODEN_SWORD = 268
    STONE_SWORD = 272
    IRON_SWORD = 267
    DIAMOND_SWORD = 276
    WOODEN_PICKAXE = 270
    STONE_PICKAXE = 274
    IRON_PICKAXE = 257
    DIAMOND_PICKAXE = 278
    WOODEN_AXE = 271
    STONE_AXE = 275
    IRON_AXE = 258
    DIAMOND_AXE = 279
    WOODEN_SHOVEL = 269
    STONE_SHOVEL = 273
    IRON_SHOVEL = 256
    DIAMOND_SHOVEL = 277


@dataclass
class PathNode:
    """路径节点"""
    x: float
    y: float
    z: float
    cost: float = 0.0
    heuristic: float = 0.0
    parent: Optional['PathNode'] = None
    
    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic
    
    def to_position(self) -> Position:
        return Position(self.x, self.y, self.z)


@dataclass
class GameObjective:
    """游戏目标"""
    objective_type: str
    target: Any
    priority: int = 1
    deadline: Optional[float] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PathPlanner:
    """路径规划器"""
    
    def __init__(self, max_range: int = 20):
        self.max_range = max_range
        self.blocked_positions = set()
    
    def plan_path(self, start: Position, goal: Position) -> List[Position]:
        """规划从起点到终点的路径"""
        try:
            # 简化的A*算法实现
            open_set = [PathNode(start.x, start.y, start.z)]
            closed_set = set()
            node_map = {}
            
            while open_set:
                # 选择代价最小的节点
                current = min(open_set, key=lambda n: n.total_cost)
                open_set.remove(current)
                
                if (abs(current.x - goal.x) < 1 and 
                    abs(current.y - goal.y) < 1 and 
                    abs(current.z - goal.z) < 1):
                    # 找到目标，重建路径
                    return self._reconstruct_path(current, node_map)
                
                closed_set.add((current.x, current.y, current.z))
                
                # 探索邻居节点
                for neighbor in self._get_neighbors(current):
                    if (neighbor.x, neighbor.y, neighbor.z) in closed_set:
                        continue
                    
                    if (neighbor.x, neighbor.y, neighbor.z) in self.blocked_positions:
                        continue
                    
                    # 计算代价
                    tentative_cost = current.cost + self._distance(current, neighbor)
                    
                    if neighbor not in open_set or tentative_cost < neighbor.cost:
                        neighbor.parent = current
                        neighbor.cost = tentative_cost
                        neighbor.heuristic = self._heuristic(neighbor, goal)
                        
                        if neighbor not in open_set:
                            open_set.append(neighbor)
                        node_map[(neighbor.x, neighbor.y, neighbor.z)] = neighbor
            
            # 如果没有找到路径，返回直线路径
            return [start, goal]
            
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            return [start, goal]
    
    def _get_neighbors(self, node: PathNode) -> List[PathNode]:
        """获取邻居节点"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    neighbor = PathNode(
                        node.x + dx,
                        node.y + dy,
                        node.z + dz
                    )
                    
                    # 检查是否在有效范围内
                    if (abs(neighbor.x) <= self.max_range and
                        0 <= neighbor.y <= 128 and  # Minecraft高度限制
                        abs(neighbor.z) <= self.max_range):
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _distance(self, node1: PathNode, node2: PathNode) -> float:
        """计算两点距离"""
        return math.sqrt(
            (node1.x - node2.x) ** 2 +
            (node1.y - node2.y) ** 2 +
            (node1.z - node2.z) ** 2
        )
    
    def _heuristic(self, node: PathNode, goal: Position) -> float:
        """启发式函数（到目标的直线距离）"""
        return math.sqrt(
            (node.x - goal.x) ** 2 +
            (node.y - goal.y) ** 2 +
            (node.z - goal.z) ** 2
        )
    
    def _reconstruct_path(self, end_node: PathNode, node_map: Dict) -> List[Position]:
        """重建路径"""
        path = []
        current = end_node
        
        while current:
            path.append(current.to_position())
            current = current.parent
        
        return list(reversed(path))


class GameInterface:
    """游戏接口主类"""
    
    def __init__(self, bridge: WebSocketBridge):
        self.bridge = bridge
        self.path_planner = PathPlanner()
        self.objectives = []
        self.performance_metrics = {
            'actions_executed': 0,
            'actions_successful': 0,
            'path_length_avg': 0.0,
            'response_time_avg': 0.0,
            'total_distance_traveled': 0.0
        }
        
        # 注册事件回调
        self._register_event_callbacks()
    
    def _register_event_callbacks(self):
        """注册事件回调"""
        self.bridge.add_event_callback('position_update', self._on_position_update)
        self.bridge.add_event_callback('health_update', self._on_health_update)
        self.bridge.add_event_callback('inventory_update', self._on_inventory_update)
        self.bridge.add_event_callback('attack_event', self._on_attack_event)
        self.bridge.add_event_callback('damage_event', self._on_damage_event)
    
    async def _on_position_update(self, event: GameEvent):
        """位置更新事件处理"""
        if event.data and 'position' in event.data:
            self.performance_metrics['response_time_avg'] = (
                self.performance_metrics['response_time_avg'] * 0.9 +
                event.timestamp * 0.1
            )
    
    async def _on_health_update(self, event: GameEvent):
        """生命值更新事件处理"""
        if event.data and event.data.get('health', 0) < 10:
            logger.warning("⚠️ 生命值过低，考虑寻找治疗物品")
            await self._handle_low_health()
    
    async def _on_inventory_update(self, event: GameEvent):
        """物品栏更新事件处理"""
        # 物品栏变化时的处理逻辑
        pass
    
    async def _on_attack_event(self, event: GameEvent):
        """攻击事件处理"""
        logger.info(f"⚔️ 攻击事件: {event.data}")
    
    async def _on_damage_event(self, event: GameEvent):
        """受伤事件处理"""
        logger.warning(f"💔 受伤事件: {event.data}")
        await self._handle_damage(event.data)
    
    async def _handle_low_health(self):
        """处理低生命值"""
        inventory = self.bridge.get_inventory()
        
        # 寻找治疗物品（面包、胡萝卜等）
        healing_items = ['bread', 'golden_apple', 'carrot']
        for item in inventory:
            if item.item_name in healing_items:
                logger.info(f"🍖 使用治疗物品: {item.item_name}")
                await self.bridge.use_item()
                break
    
    async def _handle_damage(self, damage_data: Dict[str, Any]):
        """处理受伤"""
        # 简单的反击逻辑
        if damage_data.get('cause') == 'mob':
            await self.bridge.attack()
    
    # === 移动和导航 ===
    
    async def move_to(self, x: float, y: float, z: float, max_retries: int = 3) -> bool:
        """移动到指定位置"""
        current_pos = self.bridge.get_current_position()
        if not current_pos:
            logger.error("❌ 无法获取当前位置")
            return False
        
        goal = Position(x, y, z)
        
        for attempt in range(max_retries):
            logger.info(f"🎯 尝试移动到 ({x}, {y}, {z}), 尝试 {attempt + 1}/{max_retries}")
            
            # 规划路径
            path = self.path_planner.plan_path(current_pos, goal)
            if len(path) < 2:
                logger.warning("⚠️ 无法规划有效路径")
                return False
            
            # 沿路径移动
            success = await self._follow_path(path)
            if success:
                # 更新性能指标
                self.performance_metrics['actions_executed'] += len(path)
                self.performance_metrics['actions_successful'] += len(path)
                self.performance_metrics['path_length_avg'] = (
                    self.performance_metrics['path_length_avg'] * 0.9 +
                    len(path) * 0.1
                )
                return True
            
            logger.warning(f"⚠️ 移动尝试 {attempt + 1} 失败")
            await asyncio.sleep(1)  # 等待1秒后重试
        
        logger.error(f"❌ 经过 {max_retries} 次尝试后仍无法到达目标位置")
        return False
    
    async def _follow_path(self, path: List[Position]) -> bool:
        """沿路径移动"""
        for i in range(1, len(path)):  # 跳过起点
            current = path[i-1]
            target = path[i]
            
            # 计算移动方向
            dx = target.x - current.x
            dz = target.z - current.z
            
            # 选择移动方向
            if abs(dx) > abs(dz):
                # 主要沿X轴移动
                direction = 'east' if dx > 0 else 'west'
            else:
                # 主要沿Z轴移动
                direction = 'south' if dz > 0 else 'north'
            
            # 检查是否需要跳跃
            if target.y > current.y + 0.5:
                await self.bridge.jump(direction, 500)
            else:
                await self.bridge.move(direction, 500)
            
            # 等待移动完成
            await asyncio.sleep(0.5)
            
            # 检查是否到达目标附近
            current_pos = self.bridge.get_current_position()
            if current_pos and current_pos.distance_to(target) < 1:
                continue
            else:
                logger.warning(f"⚠️ 未能在预期时间内到达路径点 {i}")
                return False
        
        # 检查是否到达最终目标
        final_pos = self.bridge.get_current_position()
        if final_pos and final_pos.distance_to(path[-1]) < 2:
            logger.info("✅ 成功到达目标位置")
            return True
        else:
            logger.warning("⚠️ 未能在最终目标附近")
            return False
    
    # === 方块操作 ===
    
    async def mine_block(self, block_type: int, max_range: int = 5) -> bool:
        """挖掘指定类型的方块"""
        current_pos = self.bridge.get_current_position()
        if not current_pos:
            return False
        
        # 搜索附近的方块
        target_block = await self._find_block_nearby(current_pos, block_type, max_range)
        if not target_block:
            logger.warning(f"⚠️ 在附近未找到类型为 {block_type} 的方块")
            return False
        
        logger.info(f"⛏️ 开始挖掘方块: {target_block}")
        
        # 移动到方块附近
        block_pos = Position(target_block['x'], target_block['y'], target_block['z'])
        approach_pos = Position(block_pos.x - 1, block_pos.y, block_pos.z)
        
        # 确保有合适的工具
        await self._ensure_appropriate_tool(block_type)
        
        # 移动到挖掘位置
        await self.move_to(approach_pos.x, approach_pos.y, approach_pos.z)
        
        # 执行挖掘动作
        await self.bridge.execute_skill('mine_block', {
            'block_type': block_type,
            'direction': 'any'
        })
        
        # 等待挖掘完成
        await asyncio.sleep(3)  # 挖掘时间根据方块硬度调整
        
        return True
    
    async def place_block(self, block_type: int, target_position: Position) -> bool:
        """放置方块"""
        logger.info(f"🧱 放置方块类型 {block_type} 到位置 {target_position}")
        
        # 确保有方块
        if not await self._has_block_in_inventory(block_type):
            logger.warning(f"⚠️ 物品栏中没有类型为 {block_type} 的方块")
            return False
        
        # 移动到放置位置附近
        approach_pos = Position(target_position.x - 1, target_position.y, target_position.z)
        await self.move_to(approach_pos.x, approach_pos.y, approach_pos.z)
        
        # 执行放置动作
        await self.bridge.execute_skill('place_block', {
            'block_type': block_type,
            'position': target_position.to_dict()
        })
        
        return True
    
    async def _find_block_nearby(self, center_pos: Position, block_type: int, max_range: int) -> Optional[Dict[str, Any]]:
        """在附近搜索方块"""
        # 这里需要与Minecraft世界交互来查找方块
        # 简化实现，实际应查询实际世界状态
        for x in range(-max_range, max_range + 1):
            for y in range(-max_range, max_range + 1):
                for z in range(-max_range, max_range + 1):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    
                    test_pos = {
                        'x': int(center_pos.x + x),
                        'y': int(center_pos.y + y),
                        'z': int(center_pos.z + z)
                    }
                    
                    # 这里需要实际的方块查询逻辑
                    # 返回一个示例结果
                    if abs(x) <= 3 and abs(y) <= 3 and abs(z) <= 3:
                        return test_pos
        
        return None
    
    async def _ensure_appropriate_tool(self, block_type: int):
        """确保有合适的工具"""
        inventory = self.bridge.get_inventory()
        
        # 根据方块类型选择合适的工具
        tool_mapping = {
            BlockType.STONE.value: ItemType.STONE_PICKAXE.value,
            BlockType.COAL_ORE.value: ItemType.STONE_PICKAXE.value,
            BlockType.IRON_ORE.value: ItemType.IRON_PICKAXE.value,
            BlockType.GOLD_ORE.value: ItemType.IRON_PICKAXE.value,
            BlockType.DIAMOND_ORE.value: ItemType.DIAMOND_PICKAXE.value,
            BlockType.WOOD.value: ItemType.WOODEN_AXE.value,
            BlockType.PLANK.value: ItemType.WOODEN_AXE.value
        }
        
        required_tool = tool_mapping.get(block_type)
        if required_tool:
            # 检查物品栏中是否有合适的工具
            for item in inventory:
                if item.item_id == required_tool:
                    logger.info(f"✅ 找到合适的工具: {item.item_name}")
                    return
            
            # 如果没有合适的工具，尝试使用现有的最佳工具
            best_tool = await self._find_best_available_tool(inventory, block_type)
            if best_tool:
                logger.info(f"⚠️ 没有完美工具，使用现有最佳工具: {best_tool.item_name}")
    
    async def _find_best_available_tool(self, inventory: List[InventoryItem], block_type: int) -> Optional[InventoryItem]:
        """寻找最佳可用工具"""
        # 简化的工具选择逻辑
        stone_tools = [ItemType.DIAMOND_PICKAXE.value, ItemType.IRON_PICKAXE.value, ItemType.STONE_PICKAXE.value]
        wood_tools = [ItemType.DIAMOND_AXE.value, ItemType.IRON_AXE.value, ItemType.WOODEN_AXE.value]
        stone_blocks = [BlockType.STONE.value, BlockType.COAL_ORE.value, BlockType.IRON_ORE.value, 
                       BlockType.GOLD_ORE.value, BlockType.DIAMOND_ORE.value, BlockType.COBBLESTONE.value]
        wood_blocks = [BlockType.WOOD.value, BlockType.PLANK.value]
        
        if block_type in stone_blocks:
            return next((item for item in inventory if item.item_id in stone_tools), None)
        elif block_type in wood_blocks:
            return next((item for item in inventory if item.item_id in wood_tools), None)
        
        return None
    
    async def _has_block_in_inventory(self, block_type: int) -> bool:
        """检查物品栏中是否有指定类型的方块"""
        inventory = self.bridge.get_inventory()
        for item in inventory:
            if item.item_id == block_type and item.count > 0:
                return True
        return False
    
    # === 建造系统 ===
    
    async def build_structure(self, structure_type: str, start_position: Position, size: int = 3) -> bool:
        """建造结构"""
        logger.info(f"🏗️ 开始建造结构: {structure_type}")
        
        structures = {
            'house': self._build_house,
            'tower': self._build_tower,
            'bridge': self._build_bridge,
            'wall': self._build_wall
        }
        
        if structure_type in structures:
            return await structures[structure_type](start_position, size)
        else:
            logger.error(f"❌ 未知的结构类型: {structure_type}")
            return False
    
    async def _build_house(self, start_pos: Position, size: int) -> bool:
        """建造房屋"""
        logger.info(f"🏠 建造 {size}x{size} 的房屋")
        
        # 墙
        for x in range(size):
            for y in range(3):
                wall_pos = Position(start_pos.x + x, start_pos.y + y, start_pos.z)
                await self.place_block(BlockType.WOOD.value, wall_pos)
                
                wall_pos2 = Position(start_pos.x + x, start_pos.y + y, start_pos.z + size - 1)
                await self.place_block(BlockType.WOOD.value, wall_pos2)
        
        # 前门和窗户
        for y in range(2):
            door_pos = Position(start_pos.x + size//2, start_pos.y + y, start_pos.z)
            await self.place_block(BlockType.AIR.value, door_pos)  # 空门
            
            window_pos = Position(start_pos.x, start_pos.y + 1 + y, start_pos.z + 1)
            await self.place_block(BlockType.GLASS.value if hasattr(BlockType, 'GLASS') else BlockType.AIR.value, window_pos)
        
        # 屋顶（简化）
        for x in range(size + 2):
            for z in range(size + 2):
                roof_pos = Position(start_pos.x + x - 1, start_pos.y + 3, start_pos.z + z - 1)
                await self.place_block(BlockType.PLANK.value, roof_pos)
        
        return True
    
    async def _build_tower(self, start_pos: Position, height: int) -> bool:
        """建造塔楼"""
        logger.info(f"🏯 建造高度为 {height} 的塔楼")
        
        for y in range(height):
            for x in range(3):
                for z in range(3):
                    if x == 1 and z == 1:  # 中心位置是空的
                        continue
                    
                    block_pos = Position(start_pos.x + x, start_pos.y + y, start_pos.z + z)
                    await self.place_block(BlockType.STONE.value, block_pos)
        
        # 顶部
        for x in range(3):
            for z in range(3):
                top_pos = Position(start_pos.x + x, start_pos.y + height, start_pos.z + z)
                await self.place_block(BlockType.STONE.value, top_pos)
        
        return True
    
    async def _build_bridge(self, start_pos: Position, length: int) -> bool:
        """建造桥梁"""
        logger.info(f"🌉 建造长度为 {length} 的桥梁")
        
        for x in range(length):
            for z in range(2):
                bridge_pos = Position(start_pos.x + x, start_pos.y, start_pos.z + z)
                await self.place_block(BlockType.PLANK.value, bridge_pos)
            
            # 添加护栏
            rail_pos = Position(start_pos.x + x, start_pos.y + 1, start_pos.z)
            await self.place_block(BlockType.WOOD.value, rail_pos)
            
            rail_pos2 = Position(start_pos.x + x, start_pos.y + 1, start_pos.z + 1)
            await self.place_block(BlockType.WOOD.value, rail_pos2)
        
        return True
    
    async def _build_wall(self, start_pos: Position, length: int) -> bool:
        """建造围墙"""
        logger.info(f"🧱 建造长度为 {length} 的围墙")
        
        for x in range(length):
            for y in range(3):
                wall_pos = Position(start_pos.x + x, start_pos.y + y, start_pos.z)
                await self.place_block(BlockType.COBBLESTONE.value, wall_pos)
        
        return True
    
    # === 物品栏管理 ===
    
    async def get_item_info(self, item_id: int) -> Optional[Dict[str, Any]]:
        """获取物品信息"""
        inventory = self.bridge.get_inventory()
        for item in inventory:
            if item.item_id == item_id:
                return {
                    'item_id': item.item_id,
                    'item_name': item.item_name,
                    'count': item.count,
                    'durability': item.durability,
                    'max_durability': item.max_durability,
                    'slot': item.slot
                }
        return None
    
    async def count_items(self, item_id: int) -> int:
        """统计指定物品的数量"""
        total_count = 0
        inventory = self.bridge.get_inventory()
        for item in inventory:
            if item.item_id == item_id:
                total_count += item.count
        return total_count
    
    async def has_item(self, item_id: int, min_count: int = 1) -> bool:
        """检查是否有指定数量的物品"""
        return await self.count_items(item_id) >= min_count
    
    async def drop_item(self, item_id: int, count: int) -> bool:
        """丢弃指定物品"""
        logger.info(f"🗑️ 丢弃 {count} 个 {item_id} 物品")
        
        # 这里需要实现丢弃逻辑
        # 由于Minecraft API限制，简化实现
        return await self.bridge.use_item()  # 临时使用use动作代替
    
    # === 战斗系统 ===
    
    async def hunt_mob(self, mob_type: str, strategy: str = "aggressive") -> bool:
        """狩猎怪物"""
        logger.info(f"🎯 开始狩猎怪物: {mob_type}, 策略: {strategy}")
        
        # 寻找怪物
        mob = await self._find_mob_nearby(mob_type)
        if not mob:
            logger.warning(f"⚠️ 未找到附近的 {mob_type} 怪物")
            return False
        
        # 根据策略执行动作
        mob_pos = Position(mob['x'], mob['y'], mob['z'])
        current_pos = self.bridge.get_current_position()
        
        if not current_pos:
            return False
        
        distance = current_pos.distance_to(mob_pos)
        
        if strategy == "aggressive":
            await self._aggressive_combat(mob_pos, distance)
        elif strategy == "defensive":
            await self._defensive_combat(mob_pos, distance)
        elif strategy == "kiting":
            await self._kiting_combat(mob_pos, distance)
        
        return True
    
    async def _find_mob_nearby(self, mob_type: str, max_range: int = 10) -> Optional[Dict[str, Any]]:
        """寻找附近的怪物"""
        # 这里需要与Minecraft世界交互来查找实体
        # 简化实现
        current_pos = self.bridge.get_current_position()
        if current_pos:
            return {
                'type': mob_type,
                'x': current_pos.x + 3,
                'y': current_pos.y,
                'z': current_pos.z
            }
        return None
    
    async def _aggressive_combat(self, mob_pos: Position, distance: float):
        """积极战斗策略"""
        # 移动到怪物附近
        approach_pos = Position(mob_pos.x - 1, mob_pos.y, mob_pos.z)
        await self.move_to(approach_pos.x, approach_pos.y, approach_pos.z)
        
        # 持续攻击
        for _ in range(5):
            await self.bridge.attack()
            await asyncio.sleep(0.5)
    
    async def _defensive_combat(self, mob_pos: Position, distance: float):
        """防御战斗策略"""
        # 保持距离，逐步接近
        target_pos = Position(mob_pos.x - 2, mob_pos.y, mob_pos.z)
        await self.move_to(target_pos.x, target_pos.y, target_pos.z)
        
        # 间歇性攻击
        for _ in range(3):
            await self.bridge.attack()
            await asyncio.sleep(1)
    
    async def _kiting_combat(self, mob_pos: Position, distance: float):
        """风筝战斗策略"""
        # 围绕怪物移动攻击
        positions = [
            Position(mob_pos.x - 1, mob_pos.y, mob_pos.z),
            Position(mob_pos.x, mob_pos.y, mob_pos.z + 1),
            Position(mob_pos.x + 1, mob_pos.y, mob_pos.z),
            Position(mob_pos.x, mob_pos.y, mob_pos.z - 1)
        ]
        
        for pos in positions:
            await self.move_to(pos.x, pos.y, pos.z)
            await self.bridge.attack()
            await asyncio.sleep(0.5)
    
    # === 目标管理 ===
    
    def add_objective(self, objective: GameObjective):
        """添加游戏目标"""
        self.objectives.append(objective)
        self.objectives.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"📋 添加目标: {objective.objective_type}")
    
    async def complete_objective(self, objective: GameObjective):
        """完成目标"""
        objective.status = "completed"
        logger.info(f"✅ 完成目标: {objective.objective_type}")
    
    async def fail_objective(self, objective: GameObjective, reason: str = ""):
        """目标失败"""
        objective.status = "failed"
        logger.error(f"❌ 目标失败: {objective.objective_type}, 原因: {reason}")
    
    async def process_objectives(self):
        """处理目标队列"""
        for objective in self.objectives[:]:  # 遍历副本
            if objective.status != "pending":
                continue
            
            # 检查截止时间
            if objective.deadline and time.time() > objective.deadline:
                await self.fail_objective(objective, "超时")
                continue
            
            objective.status = "in_progress"
            
            try:
                success = await self._execute_objective(objective)
                if success:
                    await self.complete_objective(objective)
                else:
                    await self.fail_objective(objective, "执行失败")
                
            except Exception as e:
                logger.error(f"❌ 目标执行异常: {e}")
                await self.fail_objective(objective, str(e))
    
    async def _execute_objective(self, objective: GameObjective) -> bool:
        """执行具体目标"""
        objective_type = objective.objective_type
        target = objective.target
        
        if objective_type == "move_to":
            return await self.move_to(*target)
        elif objective_type == "mine_block":
            return await self.mine_block(target)
        elif objective_type == "build_structure":
            return await self.build_structure(*target)
        elif objective_type == "hunt_mob":
            return await self.hunt_mob(*target)
        elif objective_type == "collect_items":
            return await self._collect_items(*target)
        else:
            logger.error(f"❌ 未知的目标类型: {objective_type}")
            return False
    
    async def _collect_items(self, item_id: int, count: int) -> bool:
        """收集物品"""
        current_count = await self.count_items(item_id)
        target_count = current_count + count
        
        logger.info(f"🎒 收集 {count} 个物品，目标总数: {target_count}")
        
        # 这里需要实现自动收集逻辑
        # 简化实现
        while await self.count_items(item_id) < target_count:
            # 寻找和收集物品的逻辑
            await self._search_and_collect_item(item_id)
            await asyncio.sleep(1)
        
        return True
    
    async def _search_and_collect_item(self, item_id: int):
        """搜索并收集物品"""
        # 简化实现，模拟物品收集
        logger.debug(f"🔍 搜索物品: {item_id}")
        await asyncio.sleep(0.1)
    
    # === 性能监控 ===
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self):
        """重置性能指标"""
        self.performance_metrics = {
            'actions_executed': 0,
            'actions_successful': 0,
            'path_length_avg': 0.0,
            'response_time_avg': 0.0,
            'total_distance_traveled': 0.0
        }


# 工具函数
def create_movement_sequence(directions: List[str], step_duration: int = 500) -> ActionSequence:
    """创建移动序列"""
    from ..single.websocket_client import ActionSequence
    
    def movement_callback(bridge):
        sequence = ActionSequence(bridge)
        sequence.move_sequence(directions, step_duration)
        return sequence.execute()
    
    return movement_callback


def create_build_sequence(structure_type: str, start_pos: Position, size: int = 3) -> Callable:
    """创建建造序列"""
    def build_callback(interface: GameInterface):
        return interface.build_structure(structure_type, start_pos, size)
    
    return build_callback


# 示例使用
async def example_game_interface_usage():
    """游戏接口使用示例"""
    from ..single.websocket_client import WebSocketBridge
    
    # 创建桥接和接口
    bridge = WebSocketBridge()
    game_interface = GameInterface(bridge)
    
    # 连接
    if await bridge.connect():
        try:
            # 基础移动
            await game_interface.move_to(10, 64, 10)
            
            # 挖掘方块
            await game_interface.mine_block(BlockType.COAL_ORE.value)
            
            # 建造房屋
            current_pos = bridge.get_current_position()
            if current_pos:
                await game_interface.build_structure('house', current_pos, 5)
            
            # 狩猎怪物
            await game_interface.hunt_mob('zombie')
            
            # 添加并处理目标
            move_obj = GameObjective(
                objective_type="move_to",
                target=(20, 64, 20),
                priority=1
            )
            game_interface.add_objective(move_obj)
            
            mine_obj = GameObjective(
                objective_type="mine_block",
                target=BlockType.DIAMOND_ORE.value,
                priority=2
            )
            game_interface.add_objective(mine_obj)
            
            await game_interface.process_objectives()
            
            # 性能统计
            metrics = game_interface.get_performance_metrics()
            print(f"性能指标: {metrics}")
            
        finally:
            await bridge.disconnect()


if __name__ == "__main__":
    asyncio.run(example_game_interface_usage())