# -*- coding: utf-8 -*-
"""
桥接系统模块初始化
"""

from .game_interface import GameInterface, ActionType, BlockType, ItemType, Position, InventoryItem, GameObjective

__all__ = [
    'GameInterface',
    'ActionType', 
    'BlockType',
    'ItemType',
    'Position',
    'InventoryItem', 
    'GameObjective'
]