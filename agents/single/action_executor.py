#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作执行器 - 实现27种原子动作的执行

这个模块负责执行智能体的各种原子动作，包括：
1. 8方向移动动作
2. 跳跃和飞行动作  
3. 攻击和交互动作
4. 物品操作动作

作者：MiniMax智能体
创建时间：2025-11-13
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass
import asyncio


class ActionType(Enum):
    """动作类型枚举"""
    # 移动动作
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_FORWARD_LEFT = auto()
    MOVE_FORWARD_RIGHT = auto()
    MOVE_BACKWARD_LEFT = auto()
    MOVE_BACKWARD_RIGHT = auto()
    
    # 跳跃和飞行动作
    JUMP = auto()
    DOUBLE_JUMP = auto()
    FLY_UP = auto()
    FLY_DOWN = auto()
    FLY_FORWARD = auto()
    FLY_BACKWARD = auto()
    FLY_STOP = auto()
    
    # 攻击和交互动作
    ATTACK = auto()
    RIGHT_CLICK = auto()
    DESTROY_BLOCK = auto()
    
    # 物品操作动作
    PLACE_BLOCK = auto()
    USE_ITEM = auto()
    DROP_ITEM = auto()
    INVENTORY_OPEN = auto()
    INVENTORY_CLOSE = auto()


@dataclass
class ActionResult:
    """动作执行结果"""
    success: bool
    duration: float
    message: str = ""
    data: Dict[str, Any] = None


class ActionExecutor:
    """原子动作执行器
    
    负责执行27种基础原子动作，提供统一的动作执行接口
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_position = (0.0, 0.0, 0.0)
        self.inventory = {}
        self.health = 100
        self.energy = 100
        
        # 动作执行历史
        self.action_history = []
        self.max_history = 100
        
    async def execute_action(self, action_type: ActionType, **kwargs) -> ActionResult:
        """
        执行指定的原子动作
        
        Args:
            action_type: 动作类型
            **kwargs: 动作参数
            
        Returns:
            ActionResult: 动作执行结果
        """
        start_time = time.time()
        
        try:
            # 根据动作类型调用相应的处理方法
            if action_type in [ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD, 
                             ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT,
                             ActionType.MOVE_FORWARD_LEFT, ActionType.MOVE_FORWARD_RIGHT,
                             ActionType.MOVE_BACKWARD_LEFT, ActionType.MOVE_BACKWARD_RIGHT]:
                result = await self._execute_movement(action_type, **kwargs)
            elif action_type in [ActionType.JUMP, ActionType.DOUBLE_JUMP,
                               ActionType.FLY_UP, ActionType.FLY_DOWN, 
                               ActionType.FLY_FORWARD, ActionType.FLY_BACKWARD, ActionType.FLY_STOP]:
                result = await self._execute_jump_fly(action_type, **kwargs)
            elif action_type in [ActionType.ATTACK, ActionType.RIGHT_CLICK, ActionType.DESTROY_BLOCK]:
                result = await self._execute_attack_interact(action_type, **kwargs)
            elif action_type in [ActionType.PLACE_BLOCK, ActionType.USE_ITEM, ActionType.DROP_ITEM,
                               ActionType.INVENTORY_OPEN, ActionType.INVENTORY_CLOSE]:
                result = await self._execute_inventory_action(action_type, **kwargs)
            else:
                result = ActionResult(False, 0, f"未知动作类型: {action_type}")
                
            # 记录执行历史
            self._record_action_history(action_type, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"执行动作 {action_type} 失败: {str(e)}")
            duration = time.time() - start_time
            return ActionResult(False, duration, f"执行失败: {str(e)}")
    
    async def _execute_movement(self, action_type: ActionType, **kwargs) -> ActionResult:
        """执行移动动作"""
        distance = kwargs.get('distance', 1.0)
        speed = kwargs.get('speed', 1.0)
        
        # 模拟移动延迟
        move_time = distance / speed * 0.1  # 基础移动时间
        await asyncio.sleep(move_time)
        
        # 计算新的位置
        new_position = self._calculate_new_position(action_type, distance)
        old_position = self.current_position
        self.current_position = new_position
        
        self.logger.info(f"移动动作: {action_type.name} 从 {old_position} 到 {new_position}")
        
        return ActionResult(
            success=True,
            duration=move_time,
            message=f"移动成功，距离: {distance:.2f}",
            data={'from': old_position, 'to': new_position}
        )
    
    async def _execute_jump_fly(self, action_type: ActionType, **kwargs) -> ActionResult:
        """执行跳跃和飞行动作"""
        height = kwargs.get('height', 1.0)
        duration = kwargs.get('duration', 1.0)
        
        if action_type == ActionType.JUMP:
            # 普通跳跃
            if self.energy < 10:
                return ActionResult(False, 0, "能量不足，无法跳跃")
            self.energy -= 10
            
            jump_height = min(height, 5.0)  # 最大跳跃高度限制
            await asyncio.sleep(0.1)  # 跳跃动作时间
            
            new_pos = (self.current_position[0], 
                      self.current_position[1] + jump_height,
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"跳跃执行: 高度 {jump_height}")
            return ActionResult(True, 0.1, f"跳跃成功，高度: {jump_height}")
            
        elif action_type == ActionType.DOUBLE_JUMP:
            # 双跳
            if self.energy < 20:
                return ActionResult(False, 0, "能量不足，无法双跳")
            self.energy -= 20
            
            await asyncio.sleep(0.15)
            new_pos = (self.current_position[0], 
                      self.current_position[1] + height * 2,
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"双跳执行: 高度 {height * 2}")
            return ActionResult(True, 0.15, f"双跳成功，高度: {height * 2}")
            
        elif action_type == ActionType.FLY_UP:
            # 向上飞行
            if self.energy < 5:
                return ActionResult(False, 0, "能量不足，无法飞行")
            self.energy -= 5 * duration
            
            new_pos = (self.current_position[0], 
                      self.current_position[1] + height,
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"向上飞行: 高度 {height}")
            return ActionResult(True, duration, f"向上飞行成功，高度: {height}")
            
        elif action_type == ActionType.FLY_DOWN:
            # 向下飞行
            new_pos = (self.current_position[0], 
                      max(0, self.current_position[1] - height),
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"向下飞行: 高度 {height}")
            return ActionResult(True, duration, f"向下飞行成功，高度: {height}")
            
        elif action_type == ActionType.FLY_FORWARD:
            # 向前飞行
            if self.energy < 5:
                return ActionResult(False, 0, "能量不足，无法飞行")
            self.energy -= 5 * duration
            
            new_pos = (self.current_position[0] + height, 
                      self.current_position[1],
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"向前飞行: 距离 {height}")
            return ActionResult(True, duration, f"向前飞行成功，距离: {height}")
            
        elif action_type == ActionType.FLY_BACKWARD:
            # 向后飞行
            if self.energy < 5:
                return ActionResult(False, 0, "能量不足，无法飞行")
            self.energy -= 5 * duration
            
            new_pos = (self.current_position[0] - height, 
                      self.current_position[1],
                      self.current_position[2])
            self.current_position = new_pos
            
            self.logger.info(f"向后飞行: 距离 {height}")
            return ActionResult(True, duration, f"向后飞行成功，距离: {height}")
            
        elif action_type == ActionType.FLY_STOP:
            # 停止飞行
            await asyncio.sleep(0.05)
            self.logger.info("停止飞行")
            return ActionResult(True, 0.05, "停止飞行成功")
        
        return ActionResult(False, 0, f"未知的跳跃/飞行动作: {action_type}")
    
    async def _execute_attack_interact(self, action_type: ActionType, **kwargs) -> ActionResult:
        """执行攻击和交互动作"""
        target = kwargs.get('target', None)
        damage = kwargs.get('damage', 10)
        
        if action_type == ActionType.ATTACK:
            # 普通攻击
            if self.energy < 5:
                return ActionResult(False, 0, "能量不足，无法攻击")
            self.energy -= 5
            
            await asyncio.sleep(0.1)  # 攻击动作时间
            
            # 模拟攻击伤害
            attack_result = {
                'damage_dealt': damage,
                'target': target,
                'hit': True
            }
            
            self.logger.info(f"攻击执行: 对 {target} 造成 {damage} 伤害")
            return ActionResult(True, 0.1, f"攻击成功，造成 {damage} 伤害", attack_result)
            
        elif action_type == ActionType.RIGHT_CLICK:
            # 右键交互
            await asyncio.sleep(0.05)  # 交互时间
            
            interaction_result = {
                'target': target,
                'interaction_type': 'right_click',
                'success': True
            }
            
            self.logger.info(f"右键交互: 对 {target}")
            return ActionResult(True, 0.05, "右键交互成功", interaction_result)
            
        elif action_type == ActionType.DESTROY_BLOCK:
            # 破坏方块
            block_type = kwargs.get('block_type', 'stone')
            if self.inventory.get('tool', 0) <= 0:
                return ActionResult(False, 0, "没有合适的工具")
            
            await asyncio.sleep(0.5)  # 破坏方块时间
            
            destroy_result = {
                'block_type': block_type,
                'block_position': target,
                'materials_gained': self._calculate_materials_gained(block_type)
            }
            
            self.logger.info(f"破坏方块: {block_type} 在 {target}")
            return ActionResult(True, 0.5, f"成功破坏 {block_type} 方块", destroy_result)
        
        return ActionResult(False, 0, f"未知的攻击/交互动作: {action_type}")
    
    async def _execute_inventory_action(self, action_type: ActionType, **kwargs) -> ActionResult:
        """执行物品操作动作"""
        item_id = kwargs.get('item_id', None)
        quantity = kwargs.get('quantity', 1)
        slot = kwargs.get('slot', None)
        
        if action_type == ActionType.PLACE_BLOCK:
            # 放置方块
            if not self.inventory.get(item_id, 0) >= quantity:
                return ActionResult(False, 0, f"物品 {item_id} 数量不足")
            
            self.inventory[item_id] = self.inventory.get(item_id, 0) - quantity
            position = kwargs.get('position', self.current_position)
            
            await asyncio.sleep(0.1)  # 放置方块时间
            
            place_result = {
                'item_id': item_id,
                'quantity': quantity,
                'position': position
            }
            
            self.logger.info(f"放置方块: {quantity}x {item_id} 在 {position}")
            return ActionResult(True, 0.1, f"成功放置 {quantity}x {item_id}", place_result)
            
        elif action_type == ActionType.USE_ITEM:
            # 使用物品
            if not self.inventory.get(item_id, 0) >= 1:
                return ActionResult(False, 0, f"没有物品 {item_id}")
            
            await asyncio.sleep(0.2)  # 使用物品时间
            
            use_result = {
                'item_id': item_id,
                'effects': self._calculate_item_effects(item_id)
            }
            
            self.logger.info(f"使用物品: {item_id}")
            return ActionResult(True, 0.2, f"成功使用 {item_id}", use_result)
            
        elif action_type == ActionType.DROP_ITEM:
            # 丢弃物品
            if not self.inventory.get(item_id, 0) >= quantity:
                return ActionResult(False, 0, f"物品 {item_id} 数量不足")
            
            self.inventory[item_id] = self.inventory.get(item_id, 0) - quantity
            position = self.current_position
            
            await asyncio.sleep(0.05)  # 丢弃物品时间
            
            drop_result = {
                'item_id': item_id,
                'quantity': quantity,
                'position': position
            }
            
            self.logger.info(f"丢弃物品: {quantity}x {item_id}")
            return ActionResult(True, 0.05, f"成功丢弃 {quantity}x {item_id}", drop_result)
            
        elif action_type == ActionType.INVENTORY_OPEN:
            # 打开背包
            await asyncio.sleep(0.1)
            self.logger.info("打开背包")
            return ActionResult(True, 0.1, "背包已打开")
            
        elif action_type == ActionType.INVENTORY_CLOSE:
            # 关闭背包
            await asyncio.sleep(0.05)
            self.logger.info("关闭背包")
            return ActionResult(True, 0.05, "背包已关闭")
        
        return ActionResult(False, 0, f"未知的物品操作动作: {action_type}")
    
    def _calculate_new_position(self, action_type: ActionType, distance: float) -> Tuple[float, float, float]:
        """计算移动后的新位置"""
        x, y, z = self.current_position
        
        if action_type == ActionType.MOVE_FORWARD:
            x += distance
        elif action_type == ActionType.MOVE_BACKWARD:
            x -= distance
        elif action_type == ActionType.MOVE_LEFT:
            z -= distance
        elif action_type == ActionType.MOVE_RIGHT:
            z += distance
        elif action_type == ActionType.MOVE_FORWARD_LEFT:
            x += distance * 0.707  # 对角线移动
            z -= distance * 0.707
        elif action_type == ActionType.MOVE_FORWARD_RIGHT:
            x += distance * 0.707
            z += distance * 0.707
        elif action_type == ActionType.MOVE_BACKWARD_LEFT:
            x -= distance * 0.707
            z -= distance * 0.707
        elif action_type == ActionType.MOVE_BACKWARD_RIGHT:
            x -= distance * 0.707
            z += distance * 0.707
            
        return (x, y, z)
    
    def _calculate_materials_gained(self, block_type: str) -> Dict[str, int]:
        """计算破坏方块获得的材料"""
        material_map = {
            'stone': {'stone': 1},
            'wood': {'wood': 1},
            'sand': {'sand': 4},
            'clay': {'clay': 4},
            'iron_ore': {'iron_ingot': 1},
            'gold_ore': {'gold_ingot': 1},
            'diamond_ore': {'diamond': 1}
        }
        return material_map.get(block_type, {})
    
    def _calculate_item_effects(self, item_id: str) -> Dict[str, Any]:
        """计算使用物品的效果"""
        effects_map = {
            'apple': {'health_restore': 10},
            'bread': {'health_restore': 5},
            'potion_health': {'health_restore': 30},
            'potion_energy': {'energy_restore': 50},
            'torch': {'light_source': True}
        }
        return effects_map.get(item_id, {})
    
    def _record_action_history(self, action_type: ActionType, result: ActionResult):
        """记录动作执行历史"""
        history_entry = {
            'timestamp': time.time(),
            'action_type': action_type.name,
            'success': result.success,
            'duration': result.duration,
            'message': result.message
        }
        
        self.action_history.append(history_entry)
        
        # 保持历史记录在限制范围内
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """获取动作执行统计信息"""
        total_actions = len(self.action_history)
        successful_actions = sum(1 for entry in self.action_history if entry['success'])
        
        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'average_duration': sum(entry['duration'] for entry in self.action_history) / total_actions if total_actions > 0 else 0,
            'current_position': self.current_position,
            'inventory': self.inventory,
            'health': self.health,
            'energy': self.energy
        }
    
    def reset_state(self):
        """重置状态"""
        self.current_position = (0.0, 0.0, 0.0)
        self.inventory = {}
        self.health = 100
        self.energy = 100
        self.action_history = []
        self.logger.info("状态已重置")


# 测试和示例代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    async def test_actions():
        """测试各种动作执行"""
        executor = ActionExecutor()
        
        print("=== 动作执行器测试 ===")
        
        # 测试移动动作
        print("\n1. 测试移动动作:")
        for action in [ActionType.MOVE_FORWARD, ActionType.MOVE_LEFT, ActionType.MOVE_FORWARD_RIGHT]:
            result = await executor.execute_action(action, distance=2.0)
            print(f"   {action.name}: {result.success} - {result.message}")
        
        # 测试跳跃动作
        print("\n2. 测试跳跃动作:")
        for action in [ActionType.JUMP, ActionType.DOUBLE_JUMP]:
            result = await executor.execute_action(action, height=2.0)
            print(f"   {action.name}: {result.success} - {result.message}")
        
        # 测试飞行动作
        print("\n3. 测试飞行动作:")
        executor.energy = 100  # 重置能量
        result = await executor.execute_action(ActionType.FLY_UP, height=10.0, duration=2.0)
        print(f"   FLY_UP: {result.success} - {result.message}")
        
        # 测试攻击动作
        print("\n4. 测试攻击动作:")
        result = await executor.execute_action(ActionType.ATTACK, target="creeper", damage=15)
        print(f"   ATTACK: {result.success} - {result.message}")
        
        # 测试物品操作
        print("\n5. 测试物品操作:")
        executor.inventory = {"stone": 5, "apple": 2}  # 添加一些测试物品
        result = await executor.execute_action(ActionType.PLACE_BLOCK, item_id="stone", quantity=2)
        print(f"   PLACE_BLOCK: {result.success} - {result.message}")
        
        # 显示统计信息
        print("\n6. 动作统计:")
        stats = executor.get_action_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # 运行测试
    asyncio.run(test_actions())