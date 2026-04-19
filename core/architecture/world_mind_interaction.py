"""
外部世界-内部心智-交互行动完整架构
External World - Internal Mind - Interactive Action Architecture

此模块实现了完整的认知架构，集成了外部世界建模、内部心智处理和交互行动执行
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import deque
import uuid


class WorldType(Enum):
    """世界类型枚举"""
    REAL_WORLD = "real_world"
    VIRTUAL_WORLD = "virtual_world"
    GAME_WORLD = "game_world"


class ActionType(Enum):
    """行动类型枚举"""
    PERCEPTION = "perception"
    COGNITION = "cognition"
    MOTION = "motion"
    MANIPULATION = "manipulation"
    COMMUNICATION = "communication"


@dataclass
class PerceptionData:
    """感知数据结构"""
    timestamp: float = field(default_factory=time.time)
    modality: str = ""
    content: Any = None
    confidence: float = 0.0
    spatial_info: Dict[str, Any] = field(default_factory=dict)
    temporal_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveState:
    """认知状态数据结构"""
    six_dimensions: Dict[str, float] = field(default_factory=lambda: {
        "attention": 0.0,
        "memory": 0.0,
        "emotion": 0.0,
        "reasoning": 0.0,
        "creativity": 0.0,
        "meta_cognition": 0.0
    })
    attention_focus: List[str] = field(default_factory=list)
    active_memories: List[Any] = field(default_factory=list)
    current_goals: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActionCommand:
    """行动命令数据结构"""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    expected_outcome: Any = None


class ExternalWorldInterface(ABC):
    """外部世界接口抽象基类"""
    
    @abstractmethod
    async def perceive(self, modality: str) -> PerceptionData:
        """感知指定模态的信息"""
        pass
    
    @abstractmethod
    async def act(self, command: ActionCommand) -> bool:
        """执行行动命令"""
        pass
    
    @abstractmethod
    async def get_spatial_info(self) -> Dict[str, Any]:
        """获取空间信息"""
        pass
    
    @abstractmethod
    async def get_temporal_context(self) -> Dict[str, Any]:
        """获取时间上下文"""
        pass


class RealWorldInterface(ExternalWorldInterface):
    """真实世界接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensors = self._initialize_sensors()
        self.actuators = self._initialize_actuators()
        self.spatial_map = {}
        self.temporal_buffer = deque(maxlen=1000)
        
    def _initialize_sensors(self) -> Dict[str, Any]:
        """初始化传感器"""
        return {
            "camera": {"active": True, "resolution": (1920, 1080)},
            "microphone": {"active": True, "sample_rate": 44100},
            "imu": {"active": True, "sampling_rate": 100},
            "lidar": {"active": False, "range": 100},
            "gps": {"active": True, "accuracy": 5}
        }
    
    def _initialize_actuators(self) -> Dict[str, Any]:
        """初始化执行器"""
        return {
            "speakers": {"active": True, "volume": 0.8},
            "motors": {"active": True, "precision": 0.01},
            "leds": {"active": True, "count": 64},
            "vibration": {"active": False}
        }
    
    async def perceive(self, modality: str) -> PerceptionData:
        """感知真实世界信息"""
        if modality == "visual":
            # 模拟视觉感知
            frame_data = {
                "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "depth": np.random.rand(480, 640) * 10.0,
                "timestamp": time.time()
            }
            return PerceptionData(
                modality="visual",
                content=frame_data,
                confidence=0.85,
                spatial_info={"location": (0, 0, 0), "orientation": (0, 0, 0)},
                temporal_info={"frame_id": int(time.time() * 30)}
            )
        
        elif modality == "audio":
            # 模拟音频感知
            audio_data = {
                "waveform": np.random.randn(44100),  # 1秒音频
                "sample_rate": 44100,
                "timestamp": time.time()
            }
            return PerceptionData(
                modality="audio",
                content=audio_data,
                confidence=0.92,
                spatial_info={"direction": np.random.rand(3), "distance": np.random.rand() * 5},
                temporal_info={"duration": 1.0}
            )
        
        elif modality == "tactile":
            # 模拟触觉感知
            tactile_data = {
                "pressure": np.random.rand(4, 4),  # 4x4触觉阵列
                "temperature": 25.0 + np.random.randn(4, 4) * 2,
                "timestamp": time.time()
            }
            return PerceptionData(
                modality="tactile",
                content=tactile_data,
                confidence=0.78,
                spatial_info={"contact_points": [(i, j) for i in range(4) for j in range(4)]},
                temporal_info={"pressure_variance": np.var(tactile_data["pressure"])}
            )
        
        else:
            raise ValueError(f"不支持的感知模态: {modality}")
    
    async def act(self, command: ActionCommand) -> bool:
        """执行真实世界行动"""
        if command.action_type == ActionType.MOTION:
            # 模拟运动执行
            print(f"执行运动命令: {command.parameters}")
            await asyncio.sleep(0.1)  # 模拟执行时间
            return True
        
        elif command.action_type == ActionType.COMMUNICATION:
            # 模拟通信执行
            print(f"执行通信命令: {command.parameters}")
            await asyncio.sleep(0.2)
            return True
        
        elif command.action_type == ActionType.MANIPULATION:
            # 模拟操作执行
            print(f"执行操作命令: {command.parameters}")
            await asyncio.sleep(0.5)
            return True
        
        return False
    
    async def get_spatial_info(self) -> Dict[str, Any]:
        """获取真实世界空间信息"""
        return {
            "position": np.random.rand(3) * 100,  # 3D位置
            "orientation": np.random.rand(3) * 2 * np.pi,  # 3D方向
            "velocity": np.random.randn(3),  # 3D速度
            "objects": [
                {"id": i, "position": np.random.rand(3) * 10, "type": f"object_{i}"}
                for i in range(10)
            ]
        }
    
    async def get_temporal_context(self) -> Dict[str, Any]:
        """获取真实世界时间上下文"""
        return {
            "current_time": time.time(),
            "past_events": list(self.temporal_buffer)[-10:],
            "future_predictions": [time.time() + i for i in range(1, 6)],
            "time_series": np.random.randn(100)
        }


class VirtualWorldInterface(ExternalWorldInterface):
    """虚拟世界接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.virtual_objects = {}
        self.physics_engine = self._initialize_physics()
        self.render_engine = self._initialize_renderer()
        
    def _initialize_physics(self) -> Dict[str, Any]:
        """初始化物理引擎"""
        return {
            "gravity": 9.81,
            "friction": 0.8,
            "elasticity": 0.6,
            "collision_detection": True
        }
    
    def _initialize_renderer(self) -> Dict[str, Any]:
        """初始化渲染引擎"""
        return {
            "resolution": (1920, 1080),
            "fps": 60,
            "lighting": True,
            "shadows": True
        }
    
    async def perceive(self, modality: str) -> PerceptionData:
        """感知虚拟世界信息"""
        if modality == "visual":
            # 生成虚拟视觉数据
            scene_data = {
                "3d_scene": {
                    "objects": self.virtual_objects,
                    "lighting": {"intensity": 0.8, "position": np.array([0, 10, 5])},
                    "camera": {"position": np.array([0, 5, 10]), "target": np.array([0, 0, 0])}
                },
                "depth_buffer": np.random.rand(480, 640) * 50.0,
                "timestamp": time.time()
            }
            return PerceptionData(
                modality="virtual_visual",
                content=scene_data,
                confidence=0.95,
                spatial_info={"virtual_bounds": [(-50, 50), (-50, 50), (-50, 50)]},
                temporal_info={"simulation_step": int(time.time() * 60)}
            )
        
        return PerceptionData(modality=modality, content={}, confidence=0.0)
    
    async def act(self, command: ActionCommand) -> bool:
        """执行虚拟世界行动"""
        if command.action_type == ActionType.MANIPULATION:
            # 虚拟对象操作
            obj_id = command.parameters.get("object_id")
            action = command.parameters.get("action")
            
            if obj_id in self.virtual_objects:
                if action == "move":
                    new_pos = command.parameters.get("position")
                    self.virtual_objects[obj_id]["position"] = new_pos
                    print(f"移动虚拟对象 {obj_id} 到位置 {new_pos}")
                    return True
                elif action == "rotate":
                    rotation = command.parameters.get("rotation")
                    self.virtual_objects[obj_id]["rotation"] = rotation
                    print(f"旋转虚拟对象 {obj_id}: {rotation}")
                    return True
        
        return False
    
    async def get_spatial_info(self) -> Dict[str, Any]:
        """获取虚拟世界空间信息"""
        return {
            "virtual_bounds": [(-50, 50), (-50, 50), (-50, 50)],
            "objects": list(self.virtual_objects.values()),
            "physics_state": self.physics_engine,
            "camera_state": {"position": np.array([0, 5, 10]), "target": np.array([0, 0, 0])}
        }
    
    async def get_temporal_context(self) -> Dict[str, Any]:
        """获取虚拟世界时间上下文"""
        return {
            "simulation_time": time.time(),
            "frame_count": int(time.time() * 60),
            "physics_timestep": 1/60.0,
            "performance_metrics": {"fps": 60, "frame_time": 1/60.0}
        }


class GameWorldInterface(ExternalWorldInterface):
    """游戏世界接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.game_state = {}
        self.game_engine = self._initialize_game_engine()
        
    def _initialize_game_engine(self) -> Dict[str, Any]:
        """初始化游戏引擎"""
        return {
            "game_type": "3d_adventure",
            "difficulty": "medium",
            "physics": "realistic",
            "ai_behavior": "adaptive"
        }
    
    async def perceive(self, modality: str) -> PerceptionData:
        """感知游戏世界信息"""
        if modality == "game_state":
            game_data = {
                "player_state": {
                    "health": 100,
                    "position": np.random.rand(3) * 100,
                    "inventory": ["sword", "shield", "potion"],
                    "level": 5
                },
                "environment": {
                    "weather": "sunny",
                    "time_of_day": 12.0,
                    "npcs": [{"id": i, "state": "idle"} for i in range(20)]
                },
                "ui_elements": {
                    "health_bar": 100,
                    "minimap": np.random.randint(0, 255, (256, 256, 3)),
                    "quest_log": ["Kill 10 wolves", "Find the treasure"]
                },
                "timestamp": time.time()
            }
            return PerceptionData(
                modality="game_state",
                content=game_data,
                confidence=0.99,
                spatial_info={"game_map": "dungeon_level_3"},
                temporal_info={"game_time": 3600.0}
            )
        
        return PerceptionData(modality=modality, content={}, confidence=0.0)
    
    async def act(self, command: ActionCommand) -> bool:
        """执行游戏世界行动"""
        if command.action_type == ActionType.MOTION:
            # 游戏角色移动
            direction = command.parameters.get("direction")
            distance = command.parameters.get("distance", 1.0)
            print(f"游戏角色向{direction}移动{distance}单位")
            return True
        
        elif command.action_type == ActionType.MANIPULATION:
            # 游戏交互
            interaction = command.parameters.get("interaction")
            target = command.parameters.get("target")
            print(f"与{target}进行交互: {interaction}")
            return True
        
        return False
    
    async def get_spatial_info(self) -> Dict[str, Any]:
        """获取游戏世界空间信息"""
        return {
            "map_layout": {"width": 1000, "height": 1000, "tile_size": 1},
            "entities": [
                {"id": f"entity_{i}", "position": np.random.rand(3) * 100, "type": "npc"}
                for i in range(50)
            ],
            "navigation_mesh": {"nodes": 1000, "edges": 5000}
        }
    
    async def get_temporal_context(self) -> Dict[str, Any]:
        """获取游戏世界时间上下文"""
        return {
            "game_time": time.time(),
            "session_duration": 1800.0,  # 30分钟
            "save_points": [0, 300, 600, 900, 1200, 1500, 1800],
            "quest_timers": {"main_quest": 3600, "side_quest": 1800}
        }


class SixDimensionCognitiveEngine:
    """六维认知引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimensions = {
            "attention": AttentionModule(config.get("attention", {})),
            "memory": MemoryModule(config.get("memory", {})),
            "emotion": EmotionModule(config.get("emotion", {})),
            "reasoning": ReasoningModule(config.get("reasoning", {})),
            "creativity": CreativityModule(config.get("creativity", {})),
            "meta_cognition": MetaCognitionModule(config.get("meta_cognition", {}))
        }
        self.cognitive_state = CognitiveState()
        self.integration_weights = self._initialize_integration_weights()
        
    def _initialize_integration_weights(self) -> np.ndarray:
        """初始化维度间整合权重"""
        # 6x6权重矩阵，表示各维度间的相互影响
        weights = np.random.rand(6, 6)
        weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化
        return weights
    
    async def process_perception(self, perception: PerceptionData) -> CognitiveState:
        """处理感知输入，更新认知状态"""
        # 各维度并行处理感知
        tasks = []
        for dim_name, module in self.dimensions.items():
            if hasattr(module, 'process_perception'):
                tasks.append(module.process_perception(perception))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 更新认知状态
        for i, (dim_name, result) in enumerate(zip(self.dimensions.keys(), results)):
            if not isinstance(result, Exception) and result is not None:
                if dim_name == "attention":
                    self.cognitive_state.six_dimensions["attention"] = result.get("level", 0.0)
                    self.cognitive_state.attention_focus = result.get("focus", [])
                elif dim_name == "memory":
                    self.cognitive_state.six_dimensions["memory"] = result.get("strength", 0.0)
                    self.cognitive_state.active_memories = result.get("active_memories", [])
                elif dim_name == "emotion":
                    self.cognitive_state.six_dimensions["emotion"] = result.get("intensity", 0.0)
                    self.cognitive_state.emotional_state = result.get("state", {})
                elif dim_name == "reasoning":
                    self.cognitive_state.six_dimensions["reasoning"] = result.get("clarity", 0.0)
                elif dim_name == "creativity":
                    self.cognitive_state.six_dimensions["creativity"] = result.get("novelty", 0.0)
                elif dim_name == "meta_cognition":
                    self.cognitive_state.six_dimensions["meta_cognition"] = result.get("awareness", 0.0)
        
        # 维度间交互整合
        await self._integrate_dimensions()
        
        return self.cognitive_state
    
    async def _integrate_dimensions(self):
        """整合六个认知维度"""
        # 计算各维度的综合影响
        dimension_values = np.array(list(self.cognitive_state.six_dimensions.values()))
        
        # 维度间影响传播
        for i in range(6):
            for j in range(6):
                if i != j:
                    influence = dimension_values[i] * self.integration_weights[i][j]
                    dimension_values[j] += influence * 0.1  # 较小的相互影响
        
        # 更新认知状态
        for i, dim_name in enumerate(self.dimensions.keys()):
            self.cognitive_state.six_dimensions[dim_name] = np.clip(dimension_values[i], 0, 1)
    
    async def generate_action_intention(self) -> List[ActionCommand]:
        """基于认知状态生成行动意图"""
        intentions = []
        
        # 基于注意力决定行动优先级
        if self.cognitive_state.six_dimensions["attention"] > 0.5:
            # 关注当前焦点对象
            for focus in self.cognitive_state.attention_focus[:3]:  # 最多处理3个焦点
                intention = ActionCommand(
                    action_type=ActionType.PERCEPTION,
                    parameters={"focus_target": focus, "priority": 1},
                    priority=int(self.cognitive_state.six_dimensions["attention"] * 100)
                )
                intentions.append(intention)
        
        # 基于推理能力决定认知行动
        if self.cognitive_state.six_dimensions["reasoning"] > 0.3:
            intention = ActionCommand(
                action_type=ActionType.COGNITION,
                parameters={
                    "reasoning_type": "decision_making",
                    "data": self.cognitive_state.active_memories,
                    "priority": int(self.cognitive_state.six_dimensions["reasoning"] * 100)
                },
                priority=int(self.cognitive_state.six_dimensions["reasoning"] * 100)
            )
            intentions.append(intention)
        
        # 基于情感状态决定通信行动
        if self.cognitive_state.six_dimensions["emotion"] > 0.2:
            intention = ActionCommand(
                action_type=ActionType.COMMUNICATION,
                parameters={
                    "emotion_expression": self.cognitive_state.emotional_state,
                    "priority": int(self.cognitive_state.six_dimensions["emotion"] * 100)
                },
                priority=int(self.cognitive_state.six_dimensions["emotion"] * 80)
            )
            intentions.append(intention)
        
        # 基于创造力决定探索行动
        if self.cognitive_state.six_dimensions["creativity"] > 0.4:
            intention = ActionCommand(
                action_type=ActionType.MOTION,
                parameters={
                    "exploration_mode": True,
                    "novel_path": True,
                    "priority": int(self.cognitive_state.six_dimensions["creativity"] * 60)
                },
                priority=int(self.cognitive_state.six_dimensions["creativity"] * 60)
            )
            intentions.append(intention)
        
        # 按优先级排序
        intentions.sort(key=lambda x: x.priority, reverse=True)
        return intentions[:5]  # 最多返回5个意图


class AttentionModule:
    """注意力模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.attention_buffer = deque(maxlen=100)
        self.focus_history = []
        
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理注意力感知"""
        # 计算注意力强度
        if perception.modality == "visual":
            # 基于视觉显著性计算注意力
            saliency = self._calculate_visual_saliency(perception.content)
            attention_level = min(saliency * 1.2, 1.0)
        else:
            attention_level = perception.confidence
        
        # 确定注意焦点
        focus_targets = self._determine_focus_targets(perception)
        
        # 更新注意力历史
        self.attention_buffer.append({
            "timestamp": perception.timestamp,
            "level": attention_level,
            "modality": perception.modality,
            "focus": focus_targets
        })
        
        return {
            "level": attention_level,
            "focus": focus_targets,
            "buffer_size": len(self.attention_buffer)
        }
    
    def _calculate_visual_saliency(self, visual_data: Dict[str, Any]) -> float:
        """计算视觉显著性"""
        if "image" in visual_data:
            image = visual_data["image"]
            # 简单的边缘检测显著性
            grad_x = np.gradient(image, axis=1)
            grad_y = np.gradient(image, axis=0)
            saliency = np.mean(np.sqrt(grad_x**2 + grad_y**2)) / 255.0
            return saliency
        return 0.5
    
    def _determine_focus_targets(self, perception: PerceptionData) -> List[str]:
        """确定注意焦点目标"""
        targets = []
        
        if perception.modality == "visual" and "spatial_info" in perception.spatial_info:
            spatial = perception.spatial_info
            if "objects" in spatial:
                targets.extend([obj.get("type", f"object_{i}") for i, obj in enumerate(spatial["objects"][:3])])
        
        if not targets:
            targets = [perception.modality]
        
        return targets


class MemoryModule:
    """记忆模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.working_memory = deque(maxlen=10)
        self.episodic_memory = deque(maxlen=1000)
        self.semantic_memory = {}
        
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理记忆感知"""
        # 工作记忆更新
        self.working_memory.append({
            "timestamp": perception.timestamp,
            "content": perception.content,
            "modality": perception.modality
        })
        
        # 情景记忆编码
        if len(self.working_memory) > 3:  # 积累足够信息后编码情景
            episode = {
                "start_time": self.working_memory[0]["timestamp"],
                "end_time": self.working_memory[-1]["timestamp"],
                "modalities": [item["modality"] for item in self.working_memory],
                "content_summary": self._summarize_episode(self.working_memory)
            }
            self.episodic_memory.append(episode)
        
        # 语义记忆更新
        await self._update_semantic_memory(perception)
        
        # 计算记忆强度
        memory_strength = min(len(self.working_memory) / 10.0, 1.0)
        
        return {
            "strength": memory_strength,
            "active_memories": list(self.working_memory),
            "episodic_count": len(self.episodic_memory),
            "semantic_keys": list(self.semantic_memory.keys())[:10]
        }
    
    def _summarize_episode(self, working_memory: deque) -> Dict[str, Any]:
        """总结情景片段"""
        modalities = [item["modality"] for item in working_memory]
        modality_counts = {mod: modalities.count(mod) for mod in set(modalities)}
        
        return {
            "duration": working_memory[-1]["timestamp"] - working_memory[0]["timestamp"],
            "dominant_modality": max(modality_counts, key=modality_counts.get),
            "modality_diversity": len(set(modalities))
        }
    
    async def _update_semantic_memory(self, perception: PerceptionData):
        """更新语义记忆"""
        # 提取语义概念
        concepts = self._extract_concepts(perception)
        for concept in concepts:
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = {
                    "frequency": 1,
                    "first_seen": perception.timestamp,
                    "last_seen": perception.timestamp,
                    "associations": set()
                }
            else:
                self.semantic_memory[concept]["frequency"] += 1
                self.semantic_memory[concept]["last_seen"] = perception.timestamp
    
    def _extract_concepts(self, perception: PerceptionData) -> List[str]:
        """提取概念"""
        concepts = [perception.modality]
        
        if perception.spatial_info:
            concepts.extend(list(perception.spatial_info.keys()))
        
        if perception.temporal_info:
            concepts.extend(list(perception.temporal_info.keys()))
        
        return list(set(concepts))


class EmotionModule:
    """情感模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.emotional_state = {
            "valence": 0.0,  # 效价 (-1到1)
            "arousal": 0.5,  # 唤醒度 (0到1)
            "dominance": 0.5  # 支配度 (0到1)
        }
        self.emotion_history = deque(maxlen=100)
        
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理情感感知"""
        # 基于感知内容更新情感状态
        emotion_change = self._calculate_emotion_response(perception)
        
        # 更新当前情感状态
        for dimension, change in emotion_change.items():
            self.emotional_state[dimension] = np.clip(
                self.emotional_state[dimension] + change, -1, 1
            )
        
        # 记录情感历史
        self.emotion_history.append({
            "timestamp": perception.timestamp,
            "state": self.emotional_state.copy()
        })
        
        # 计算情感强度
        emotion_intensity = np.mean([abs(v) for v in self.emotional_state.values()])
        
        return {
            "intensity": emotion_intensity,
            "state": self.emotional_state.copy(),
            "valence": self.emotional_state["valence"],
            "arousal": self.emotional_state["arousal"],
            "dominance": self.emotional_state["dominance"]
        }
    
    def _calculate_emotion_response(self, perception: PerceptionData) -> Dict[str, float]:
        """计算情感反应"""
        response = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        # 基于置信度影响唤醒度
        confidence_change = (perception.confidence - 0.5) * 0.2
        response["arousal"] += confidence_change
        
        # 基于模态类型影响效价
        if perception.modality in ["visual", "audio"]:
            response["valence"] += 0.1
        elif perception.modality == "tactile":
            response["valence"] += 0.05
        
        # 基于空间信息影响支配度
        if perception.spatial_info:
            num_objects = len(perception.spatial_info.get("objects", []))
            if num_objects > 5:
                response["dominance"] += 0.1
            elif num_objects < 2:
                response["dominance"] -= 0.05
        
        return response


class ReasoningModule:
    """推理模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reasoning_rules = self._load_reasoning_rules()
        self.inference_chain = []
        self.decision_history = deque(maxlen=50)
        
    def _load_reasoning_rules(self) -> List[Dict[str, Any]]:
        """加载推理规则"""
        return [
            {
                "condition": "high_confidence",
                "action": "increase_certainty",
                "weight": 0.8
            },
            {
                "condition": "multiple_sources",
                "action": "cross_validate",
                "weight": 0.9
            },
            {
                "condition": "temporal_consistency",
                "action": "strengthen_belief",
                "weight": 0.7
            }
        ]
    
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理推理感知"""
        # 推理清晰度计算
        reasoning_clarity = self._calculate_reasoning_clarity(perception)
        
        # 应用推理规则
        reasoning_outcome = self._apply_reasoning_rules(perception)
        
        # 更新推理链
        self.inference_chain.append({
            "timestamp": perception.timestamp,
            "input": perception,
            "outcome": reasoning_outcome,
            "confidence": reasoning_clarity
        })
        
        return {
            "clarity": reasoning_clarity,
            "outcome": reasoning_outcome,
            "rule_applications": len([r for r in self.reasoning_rules if r["condition"] in str(perception.content)]),
            "chain_length": len(self.inference_chain)
        }
    
    def _calculate_reasoning_clarity(self, perception: PerceptionData) -> float:
        """计算推理清晰度"""
        clarity_factors = []
        
        # 置信度影响
        clarity_factors.append(perception.confidence)
        
        # 时间一致性影响
        if len(self.inference_chain) > 0:
            last_perception = self.inference_chain[-1]["input"]
            temporal_consistency = 1.0 - abs(perception.timestamp - last_perception.timestamp) / 10.0
            clarity_factors.append(max(temporal_consistency, 0.0))
        
        # 模态一致性影响
        if len(self.inference_chain) > 0:
            modalities = [item["input"].modality for item in self.inference_chain[-5:]]
            modality_consistency = modalities.count(perception.modality) / len(modalities)
            clarity_factors.append(modality_consistency)
        
        return np.mean(clarity_factors) if clarity_factors else 0.5
    
    def _apply_reasoning_rules(self, perception: PerceptionData) -> Dict[str, Any]:
        """应用推理规则"""
        outcome = {"confidence_adjustment": 0.0, "inferences": []}
        
        for rule in self.reasoning_rules:
            if self._evaluate_condition(rule["condition"], perception):
                inference = self._generate_inference(rule, perception)
                outcome["inferences"].append(inference)
                outcome["confidence_adjustment"] += rule["weight"] * 0.1
        
        return outcome
    
    def _evaluate_condition(self, condition: str, perception: PerceptionData) -> bool:
        """评估推理条件"""
        if condition == "high_confidence":
            return perception.confidence > 0.7
        elif condition == "multiple_sources":
            return len(perception.spatial_info) > 2
        elif condition == "temporal_consistency":
            return len(self.inference_chain) > 0 and abs(perception.timestamp - self.inference_chain[-1]["timestamp"]) < 5.0
        
        return False
    
    def _generate_inference(self, rule: Dict[str, Any], perception: PerceptionData) -> str:
        """生成推理结果"""
        if rule["action"] == "increase_certainty":
            return f"基于高置信度({perception.confidence:.2f})，增强对当前感知的确定性"
        elif rule["action"] == "cross_validate":
            return f"通过多源信息交叉验证提高可信度"
        elif rule["action"] == "strengthen_belief":
            return f"时间一致性增强了对当前感知的信念"
        
        return "应用推理规则生成结论"


class CreativityModule:
    """创造力模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.creative_ideas = []
        self.novelty_tracker = deque(maxlen=100)
        self.association_network = {}
        
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理创造力感知"""
        # 生成新颖联想
        novel_associations = self._generate_associations(perception)
        
        # 计算新颖性得分
        novelty_scores = [self._calculate_novelty_score(assoc) for assoc in novel_associations]
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
        
        # 记录创意想法
        if avg_novelty > 0.3:  # 新颖性阈值
            creative_idea = {
                "timestamp": perception.timestamp,
                "source": perception.modality,
                "associations": novel_associations,
                "novelty_score": avg_novelty
            }
            self.creative_ideas.append(creative_idea)
        
        return {
            "novelty": avg_novelty,
            "associations": novel_associations,
            "idea_count": len(self.creative_ideas),
            "creativity_level": min(avg_novelty * 1.5, 1.0)
        }
    
    def _generate_associations(self, perception: PerceptionData) -> List[str]:
        """生成联想"""
        associations = []
        
        # 基于模态的联想
        if perception.modality == "visual":
            associations.extend([
                "看到形状 → 联想到建筑",
                "看到颜色 → 联想到情绪",
                "看到运动 → 联想到物理"
            ])
        
        elif perception.modality == "audio":
            associations.extend([
                "听到频率 → 联想到音乐",
                "听到节奏 → 联想到舞蹈",
                "听到声音 → 联想到记忆"
            ])
        
        # 基于空间信息的联想
        if "objects" in perception.spatial_info:
            num_objects = len(perception.spatial_info["objects"])
            if num_objects > 5:
                associations.append(f"多个物体({num_objects}) → 联想到复杂系统")
        
        # 基于时间信息的联想
        if "duration" in perception.temporal_info:
            duration = perception.temporal_info["duration"]
            if duration > 5.0:
                associations.append(f"长时间({duration:.1f}s) → 联想到持续过程")
        
        return associations
    
    def _calculate_novelty_score(self, association: str) -> float:
        """计算新颖性得分"""
        # 检查是否与历史联想重复
        historical_associations = [idea["associations"] for idea in self.creative_ideas[-10:]]
        historical_count = sum(1 for assoc_list in historical_associations for assoc in assoc_list if assoc == association)
        
        # 基础新颖性 + 重复惩罚
        base_novelty = np.random.uniform(0.2, 0.8)  # 随机生成基础新颖性
        repetition_penalty = historical_count * 0.2
        novelty_score = max(base_novelty - repetition_penalty, 0.0)
        
        return min(novelty_score, 1.0)


class MetaCognitionModule:
    """元认知模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_awareness = {
            "cognitive_load": 0.5,
            "task_difficulty": 0.5,
            "performance_estimate": 0.5,
            "learning_rate": 0.5
        }
        self.strategy_history = deque(maxlen=50)
        
    async def process_perception(self, perception: PerceptionData) -> Dict[str, Any]:
        """处理元认知感知"""
        # 评估认知负荷
        cognitive_load = self._assess_cognitive_load(perception)
        
        # 估算任务难度
        task_difficulty = self._estimate_task_difficulty(perception)
        
        # 性能估计
        performance_estimate = self._estimate_performance(perception)
        
        # 学习率调整
        learning_rate = self._adjust_learning_rate(cognitive_load, task_difficulty)
        
        # 更新元认知状态
        self.meta_awareness.update({
            "cognitive_load": cognitive_load,
            "task_difficulty": task_difficulty,
            "performance_estimate": performance_estimate,
            "learning_rate": learning_rate
        })
        
        return {
            "awareness": np.mean(list(self.meta_awareness.values())),
            "cognitive_load": cognitive_load,
            "task_difficulty": task_difficulty,
            "performance_estimate": performance_estimate,
            "learning_rate": learning_rate,
            "strategies": self._recommend_strategies(cognitive_load, task_difficulty)
        }
    
    def _assess_cognitive_load(self, perception: PerceptionData) -> float:
        """评估认知负荷"""
        load_factors = []
        
        # 模态复杂度
        modality_complexity = {
            "visual": 0.8,
            "audio": 0.6,
            "tactile": 0.7,
            "game_state": 0.9,
            "virtual_visual": 0.85
        }
        load_factors.append(modality_complexity.get(perception.modality, 0.5))
        
        # 置信度影响
        if perception.confidence < 0.3:
            load_factors.append(0.9)  # 低置信度增加负荷
        elif perception.confidence > 0.8:
            load_factors.append(0.3)  # 高置信度降低负荷
        
        # 信息量影响
        info_size = len(str(perception.content))
        info_load = min(info_size / 10000.0, 1.0)  # 归一化信息量
        load_factors.append(info_load)
        
        return np.mean(load_factors)
    
    def _estimate_task_difficulty(self, perception: PerceptionData) -> float:
        """估算任务难度"""
        difficulty_factors = []
        
        # 模态固有难度
        modality_difficulty = {
            "visual": 0.6,
            "audio": 0.4,
            "tactile": 0.7,
            "game_state": 0.8,
            "virtual_visual": 0.75
        }
        difficulty_factors.append(modality_difficulty.get(perception.modality, 0.5))
        
        # 空间信息复杂度
        if perception.spatial_info:
            spatial_complexity = len(perception.spatial_info) / 10.0
            difficulty_factors.append(min(spatial_complexity, 1.0))
        
        # 时间约束影响
        if "duration" in perception.temporal_info:
            duration = perception.temporal_info["duration"]
            if duration < 1.0:
                difficulty_factors.append(0.8)  # 短时间约束增加难度
        
        return np.mean(difficulty_factors) if difficulty_factors else 0.5
    
    def _estimate_performance(self, perception: PerceptionData) -> float:
        """估算性能"""
        # 基于置信度和历史性能
        base_performance = perception.confidence
        
        # 历史性能影响
        if len(self.strategy_history) > 0:
            recent_performance = np.mean([
                strategy.get("performance", 0.5) for strategy in list(self.strategy_history)[-5:]
            ])
            base_performance = (base_performance + recent_performance) / 2.0
        
        return base_performance
    
    def _adjust_learning_rate(self, cognitive_load: float, task_difficulty: float) -> float:
        """调整学习率"""
        # 高负荷低难度 → 慢学习
        # 低负荷高难度 → 快学习
        # 平衡时正常学习
        
        if cognitive_load > 0.7 and task_difficulty < 0.3:
            return 0.3  # 慢学习
        elif cognitive_load < 0.3 and task_difficulty > 0.7:
            return 0.8  # 快学习
        else:
            return 0.5  # 正常学习
    
    def _recommend_strategies(self, cognitive_load: float, task_difficulty: float) -> List[str]:
        """推荐认知策略"""
        strategies = []
        
        if cognitive_load > 0.7:
            strategies.extend(["分解任务", "减少并行处理"])
        
        if task_difficulty > 0.7:
            strategies.extend(["增加练习", "寻求外部帮助"])
        
        if cognitive_load < 0.4 and task_difficulty < 0.4:
            strategies.extend(["挑战更高难度", "尝试新方法"])
        
        if not strategies:
            strategies = ["保持当前策略"]
        
        return strategies


class EmbodiedIntelligenceModule:
    """具身智能模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.body_model = self._initialize_body_model()
        self.action_planner = ActionPlanner(config.get("planning", {}))
        self.sensor_fusion = SensorFusion(config.get("sensors", {}))
        
    def _initialize_body_model(self) -> Dict[str, Any]:
        """初始化身体模型"""
        return {
            "joints": {
                "head": {"position": (0, 1.6, 0), "rotation": (0, 0, 0)},
                "torso": {"position": (0, 1.0, 0), "rotation": (0, 0, 0)},
                "left_arm": {"position": (-0.5, 1.0, 0), "rotation": (0, 0, 0)},
                "right_arm": {"position": (0.5, 1.0, 0), "rotation": (0, 0, 0)},
                "left_leg": {"position": (-0.2, 0.3, 0), "rotation": (0, 0, 0)},
                "right_leg": {"position": (0.2, 0.3, 0), "rotation": (0, 0, 0)}
            },
            "capabilities": {
                "vision": {"range": 100, "field_of_view": 120},
                "hearing": {"frequency_range": (20, 20000), "directional": True},
                "tactile": {"sensitivity": 0.8, "spatial_resolution": 4},
                "locomotion": {"max_speed": 5.0, "terrain_adaptation": True},
                "manipulation": {"precision": 0.95, "force_control": True}
            }
        }
    
    async def plan_action(self, intention: ActionCommand, cognitive_state: CognitiveState) -> List[ActionCommand]:
        """规划具身行动"""
        # 基于认知状态和行动意图生成具体行动
        planned_actions = await self.action_planner.generate_plan(
            intention, cognitive_state, self.body_model
        )
        
        # 融合多感官信息优化行动
        optimized_actions = await self.sensor_fusion.optimize_actions(
            planned_actions, self.body_model
        )
        
        return optimized_actions
    
    async def execute_action(self, action: ActionCommand, world_interface: ExternalWorldInterface) -> bool:
        """执行具身行动"""
        try:
            # 验证行动可行性
            if not await self._validate_action_feasibility(action):
                return False
            
            # 执行行动
            success = await world_interface.act(action)
            
            # 更新身体模型
            if success:
                await self._update_body_model(action)
            
            return success
            
        except Exception as e:
            print(f"行动执行失败: {e}")
            return False
    
    async def _validate_action_feasibility(self, action: ActionCommand) -> bool:
        """验证行动可行性"""
        if action.action_type == ActionType.MOTION:
            # 检查运动参数
            required_params = ["direction", "distance"]
            for param in required_params:
                if param not in action.parameters:
                    return False
            
            # 检查距离限制
            distance = action.parameters.get("distance", 0)
            if distance > self.body_model["capabilities"]["locomotion"]["max_speed"]:
                return False
        
        elif action.action_type == ActionType.MANIPULATION:
            # 检查操作精度要求
            precision_required = action.parameters.get("precision", 0.8)
            if precision_required > self.body_model["capabilities"]["manipulation"]["precision"]:
                return False
        
        return True
    
    async def _update_body_model(self, action: ActionCommand):
        """更新身体模型"""
        if action.action_type == ActionType.MOTION:
            # 更新位置信息
            direction = np.array(action.parameters.get("direction", [1, 0, 0]))
            distance = action.parameters.get("distance", 1.0)
            
            # 更新各关节位置（简化模型）
            for joint_name, joint_info in self.body_model["joints"].items():
                current_pos = np.array(joint_info["position"])
                new_pos = current_pos + direction * distance
                self.body_model["joints"][joint_name]["position"] = tuple(new_pos)


class ActionPlanner:
    """行动规划器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planning_horizon = config.get("horizon", 5)  # 规划步数
        self.action_library = self._initialize_action_library()
        
    def _initialize_action_library(self) -> Dict[str, Any]:
        """初始化行动库"""
        return {
            "locomotion": {
                "walk_forward": {"duration": 1.0, "energy_cost": 0.1},
                "walk_backward": {"duration": 1.0, "energy_cost": 0.12},
                "turn_left": {"duration": 0.5, "energy_cost": 0.05},
                "turn_right": {"duration": 0.5, "energy_cost": 0.05},
                "stop": {"duration": 0.1, "energy_cost": 0.0}
            },
            "manipulation": {
                "grab": {"duration": 0.5, "precision_required": 0.9},
                "release": {"duration": 0.3, "precision_required": 0.7},
                "push": {"duration": 0.8, "precision_required": 0.6},
                "pull": {"duration": 0.8, "precision_required": 0.6}
            },
            "communication": {
                "speak": {"duration": 1.0, "volume_level": 0.8},
                "gesture": {"duration": 0.6, "expressiveness": 0.7},
                "facial_expression": {"duration": 0.2, "intensity": 0.8}
            }
        }
    
    async def generate_plan(self, intention: ActionCommand, cognitive_state: CognitiveState, body_model: Dict[str, Any]) -> List[ActionCommand]:
        """生成行动计划"""
        plan = []
        
        if intention.action_type == ActionType.MOTION:
            plan = await self._plan_motion(intention, cognitive_state, body_model)
        elif intention.action_type == ActionType.MANIPULATION:
            plan = await self._plan_manipulation(intention, cognitive_state, body_model)
        elif intention.action_type == ActionType.COMMUNICATION:
            plan = await self._plan_communication(intention, cognitive_state, body_model)
        elif intention.action_type == ActionType.COGNITION:
            plan = await self._plan_cognition(intention, cognitive_state, body_model)
        
        return plan
    
    async def _plan_motion(self, intention: ActionCommand, cognitive_state: CognitiveState, body_model: Dict[str, Any]) -> List[ActionCommand]:
        """规划运动行动"""
        plan = []
        
        direction = intention.parameters.get("direction")
        distance = intention.parameters.get("distance", 1.0)
        
        if intention.parameters.get("exploration_mode"):
            # 探索模式：添加随机扰动
            exploration_noise = np.random.randn(3) * 0.1
            direction = np.array(direction) + exploration_noise
            direction = direction / np.linalg.norm(direction)
        
        # 分解长距离移动为多个短步
        max_step = 2.0
        steps = max(1, int(distance / max_step))
        step_distance = distance / steps
        
        for i in range(steps):
            action = ActionCommand(
                action_type=ActionType.MOTION,
                parameters={
                    "direction": direction.tolist(),
                    "distance": step_distance,
                    "step": i + 1
                },
                priority=intention.priority - i
            )
            plan.append(action)
        
        return plan
    
    async def _plan_manipulation(self, intention: ActionCommand, cognitive_state: CognitiveState, body_model: Dict[str, Any]) -> List[ActionCommand]:
        """规划操作行动"""
        plan = []
        
        manipulation_type = intention.parameters.get("manipulation_type")
        target = intention.parameters.get("target")
        
        # 添加准备动作
        pre_action = ActionCommand(
            action_type=ActionType.MOTION,
            parameters={
                "direction": [0, 0, 1],
                "distance": 0.5,
                "purpose": "positioning"
            },
            priority=intention.priority - 2
        )
        plan.append(pre_action)
        
        # 执行主要操作
        main_action = ActionCommand(
            action_type=ActionType.MANIPULATION,
            parameters={
                "manipulation_type": manipulation_type,
                "target": target,
                "precision": intention.parameters.get("precision", 0.8)
            },
            priority=intention.priority - 1
        )
        plan.append(main_action)
        
        # 添加恢复动作
        post_action = ActionCommand(
            action_type=ActionType.MOTION,
            parameters={
                "direction": [0, 0, -1],
                "distance": 0.5,
                "purpose": "recovery"
            },
            priority=intention.priority - 3
        )
        plan.append(post_action)
        
        return plan
    
    async def _plan_communication(self, intention: ActionCommand, cognitive_state: CognitiveState, body_model: Dict[str, Any]) -> List[ActionCommand]:
        """规划通信行动"""
        plan = []
        
        emotion_expression = intention.parameters.get("emotion_expression")
        
        # 面部表情
        facial_action = ActionCommand(
            action_type=ActionType.COMMUNICATION,
            parameters={
                "communication_type": "facial_expression",
                "emotion": emotion_expression,
                "intensity": cognitive_state.six_dimensions["emotion"]
            },
            priority=intention.priority
        )
        plan.append(facial_action)
        
        # 语音表达
        if cognitive_state.six_dimensions["emotion"] > 0.3:
            voice_action = ActionCommand(
                action_type=ActionType.COMMUNICATION,
                parameters={
                    "communication_type": "speech",
                    "content": "表达情感状态",
                    "tone": "emotional"
                },
                priority=intention.priority - 1
            )
            plan.append(voice_action)
        
        return plan
    
    async def _plan_cognition(self, intention: ActionCommand, cognitive_state: CognitiveState, body_model: Dict[str, Any]) -> List[ActionCommand]:
        """规划认知行动"""
        plan = []
        
        # 认知行动通常不直接产生身体动作
        # 但可能触发注意力调整或记忆巩固
        
        reasoning_type = intention.parameters.get("reasoning_type")
        data = intention.parameters.get("data", [])
        
        if reasoning_type == "decision_making":
            # 决策制定行动
            decision_action = ActionCommand(
                action_type=ActionType.COGNITION,
                parameters={
                    "cognitive_action": "evaluate_options",
                    "data_sources": data,
                    "decision_criteria": ["utility", "probability", "value"]
                },
                priority=intention.priority
            )
            plan.append(decision_action)
        
        return plan


class SensorFusion:
    """传感器融合"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensor_weights = self._initialize_sensor_weights()
        self.fusion_history = deque(maxlen=100)
        
    def _initialize_sensor_weights(self) -> Dict[str, float]:
        """初始化传感器权重"""
        return {
            "visual": 0.9,
            "audio": 0.7,
            "tactile": 0.8,
            "lidar": 0.95,
            "imu": 0.6,
            "gps": 0.85
        }
    
    async def optimize_actions(self, actions: List[ActionCommand], body_model: Dict[str, Any]) -> List[ActionCommand]:
        """基于传感器融合优化行动"""
        optimized_actions = []
        
        for action in actions:
            # 计算行动风险
            risk_score = self._calculate_action_risk(action, body_model)
            
            # 基于风险调整行动参数
            optimized_action = await self._adjust_action_parameters(action, risk_score)
            optimized_actions.append(optimized_action)
        
        return optimized_actions
    
    def _calculate_action_risk(self, action: ActionCommand, body_model: Dict[str, Any]) -> float:
        """计算行动风险"""
        risk_factors = []
        
        if action.action_type == ActionType.MOTION:
            distance = action.parameters.get("distance", 1.0)
            distance_risk = min(distance / 10.0, 1.0)
            risk_factors.append(distance_risk)
            
            # 检查环境复杂度
            if "objects" in action.parameters:
                object_density = len(action.parameters["objects"]) / 20.0
                risk_factors.append(min(object_density, 1.0))
        
        elif action.action_type == ActionType.MANIPULATION:
            precision_required = action.parameters.get("precision", 0.8)
            precision_risk = 1.0 - precision_required
            risk_factors.append(precision_risk)
        
        return np.mean(risk_factors) if risk_factors else 0.3
    
    async def _adjust_action_parameters(self, action: ActionCommand, risk_score: float) -> ActionCommand:
        """调整行动参数"""
        optimized_action = ActionCommand(
            action_type=action.action_type,
            parameters=action.parameters.copy(),
            priority=action.priority,
            timestamp=action.timestamp
        )
        
        if risk_score > 0.7:
            # 高风险：降低执行速度，增加检查
            if action.action_type == ActionType.MOTION:
                optimized_action.parameters["distance"] *= 0.5  # 减半距离
                optimized_action.parameters["cautious_mode"] = True
            elif action.action_type == ActionType.MANIPULATION:
                optimized_action.parameters["precision"] *= 0.8  # 降低精度要求
                optimized_action.parameters["verification_steps"] = 2
        
        return optimized_action


class WorldModelBuilder:
    """世界模型构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spatial_model = SpatialIntelligence(config.get("spatial", {}))
        self.temporal_model = TemporalIntelligence(config.get("temporal", {}))
        self.causal_model = CausalIntelligence(config.get("causal", {}))
        self.world_representation = {}
        
    async def build_world_model(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """构建世界模型"""
        # 空间智能处理
        spatial_representation = await self.spatial_model.process_spatial_information(perceptions)
        
        # 时间智能处理
        temporal_representation = await self.temporal_model.process_temporal_information(perceptions)
        
        # 因果智能处理
        causal_representation = await self.causal_model.process_causal_relationships(perceptions)
        
        # 整合表示
        world_model = {
            "spatial": spatial_representation,
            "temporal": temporal_representation,
            "causal": causal_representation,
            "objects": self._extract_objects(perceptions),
            "events": self._extract_events(perceptions),
            "rules": self._derive_rules(perceptions),
            "timestamp": time.time()
        }
        
        self.world_representation = world_model
        return world_model
    
    def _extract_objects(self, perceptions: List[PerceptionData]) -> List[Dict[str, Any]]:
        """提取对象"""
        objects = []
        
        for perception in perceptions:
            if hasattr(perception, 'spatial_info') and perception.spatial_info and "objects" in perception.spatial_info:
                spatial_objects = perception.spatial_info["objects"]
                if isinstance(spatial_objects, list):
                    objects.extend(spatial_objects)
        
        # 去除重复对象
        unique_objects = []
        seen_ids = set()
        for obj in objects:
            obj_id = obj.get("id", hash(str(obj)))  # 如果没有id，使用对象哈希
            if obj_id not in seen_ids:
                unique_objects.append(obj)
                seen_ids.add(obj_id)
        
        return unique_objects
    
    def _extract_events(self, perceptions: List[PerceptionData]) -> List[Dict[str, Any]]:
        """提取事件"""
        events = []
        
        for i, perception in enumerate(perceptions):
            try:
                event = {
                    "id": f"event_{i}",
                    "timestamp": perception.timestamp,
                    "modality": perception.modality,
                    "content_hash": hash(str(perception.content)) if perception.content is not None else 0
                }
                events.append(event)
            except Exception as e:
                # 如果内容无法哈希，使用简化的标识
                event = {
                    "id": f"event_{i}",
                    "timestamp": perception.timestamp,
                    "modality": perception.modality,
                    "content_hash": i  # 使用索引作为哈希
                }
                events.append(event)
        
        return events
    
    def _derive_rules(self, perceptions: List[PerceptionData]) -> List[Dict[str, Any]]:
        """推导规则"""
        rules = []
        
        # 基于时间序列推导简单规则
        if len(perceptions) > 1:
            for i in range(len(perceptions) - 1):
                current = perceptions[i]
                next_perc = perceptions[i + 1]
                
                # 时间间隔规则
                time_interval = next_perc.timestamp - current.timestamp
                if time_interval < 5.0:  # 5秒内
                    rule = {
                        "type": "temporal_sequence",
                        "condition": current.modality,
                        "consequence": next_perc.modality,
                        "interval": time_interval,
                        "confidence": min(perception.confidence for perception in [current, next_perc])
                    }
                    rules.append(rule)
        
        return rules


class SpatialIntelligence:
    """空间智能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spatial_memory = {}
        self.navigation_graph = {}
        
    async def process_spatial_information(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """处理空间信息"""
        spatial_data = {
            "objects": [],
            "locations": [],
            "relationships": [],
            "navigation_map": {},
            "coordinate_systems": {}
        }
        
        for perception in perceptions:
            if hasattr(perception, 'spatial_info') and perception.spatial_info:
                spatial = perception.spatial_info
                
                # 提取位置信息
                if "position" in spatial:
                    spatial_data["locations"].append({
                        "position": spatial["position"],
                        "confidence": perception.confidence,
                        "timestamp": perception.timestamp
                    })
                
                # 提取对象信息
                if "objects" in spatial:
                    for obj in spatial["objects"]:
                        spatial_data["objects"].append({
                            "id": obj.get("id"),
                            "position": obj.get("position"),
                            "type": obj.get("type"),
                            "confidence": perception.confidence
                        })
        
        # 构建导航图
        spatial_data["navigation_map"] = await self._build_navigation_map(spatial_data["locations"])
        
        return spatial_data
    
    async def _build_navigation_map(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建导航图"""
        if len(locations) < 2:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        # 创建节点
        for i, loc in enumerate(locations):
            nodes.append({
                "id": f"node_{i}",
                "position": loc["position"],
                "confidence": loc["confidence"]
            })
        
        # 创建边（简单的全连接图，实际情况中需要更智能的连接）
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                distance = np.linalg.norm(np.array(nodes[i]["position"]) - np.array(nodes[j]["position"]))
                edges.append({
                    "from": nodes[i]["id"],
                    "to": nodes[j]["id"],
                    "weight": distance,
                    "traversable": distance < 20.0  # 可通行距离阈值
                })
        
        return {"nodes": nodes, "edges": edges}


class TemporalIntelligence:
    """时间智能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_sequence = []
        self.prediction_model = {}
        
    async def process_temporal_information(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """处理时间信息"""
        temporal_data = {
            "sequences": [],
            "patterns": [],
            "predictions": [],
            "temporal_relationships": []
        }
        
        # 按时间排序
        sorted_perceptions = sorted(perceptions, key=lambda x: x.timestamp)
        
        # 提取时间序列
        temporal_sequence = [
            {
                "timestamp": perc.timestamp,
                "modality": perc.modality,
                "content_type": type(perc.content).__name__
            }
            for perc in sorted_perceptions
        ]
        
        temporal_data["sequences"].append(temporal_sequence)
        
        # 检测模式
        patterns = self._detect_temporal_patterns(temporal_sequence)
        temporal_data["patterns"] = patterns
        
        # 生成预测
        predictions = self._generate_temporal_predictions(temporal_sequence)
        temporal_data["predictions"] = predictions
        
        return temporal_data
    
    def _detect_temporal_patterns(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测时间模式"""
        patterns = []
        
        if len(sequence) < 3:
            return patterns
        
        # 检测模态转换模式
        modalities = [item["modality"] for item in sequence]
        modality_transitions = [(modalities[i], modalities[i+1]) for i in range(len(modalities)-1)]
        
        # 统计转换频率
        transition_counts = {}
        for transition in modality_transitions:
            transition_str = f"{transition[0]} -> {transition[1]}"
            transition_counts[transition_str] = transition_counts.get(transition_str, 0) + 1
        
        # 提取频繁模式
        for transition, count in transition_counts.items():
            if count >= 2:  # 至少出现2次
                patterns.append({
                    "type": "modality_transition",
                    "pattern": transition,
                    "frequency": count,
                    "confidence": count / len(modality_transitions)
                })
        
        return patterns
    
    def _generate_temporal_predictions(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成时间预测"""
        predictions = []
        
        if len(sequence) < 2:
            return predictions
        
        # 基于最近模式预测
        recent_modalities = [item["modality"] for item in sequence[-3:]]
        
        # 简单的模式匹配预测
        if len(recent_modalities) >= 2:
            last_transition = (recent_modalities[-2], recent_modalities[-1])
            
            # 基于历史数据预测下一个模态
            predicted_modality = self._predict_next_modality(last_transition)
            
            prediction = {
                "type": "modality_prediction",
                "input_pattern": f"{last_transition[0]} -> {last_transition[1]}",
                "predicted_next": predicted_modality,
                "confidence": 0.6,  # 基础置信度
                "timestamp": time.time()
            }
            predictions.append(prediction)
        
        return predictions
    
    def _predict_next_modality(self, last_transition: Tuple[str, str]) -> str:
        """预测下一个模态"""
        # 简单的转换概率模型
        transition_probs = {
            ("visual", "audio"): {"visual": 0.4, "audio": 0.3, "tactile": 0.3},
            ("audio", "visual"): {"visual": 0.5, "audio": 0.3, "tactile": 0.2},
            ("visual", "tactile"): {"visual": 0.4, "audio": 0.2, "tactile": 0.4},
            ("tactile", "visual"): {"visual": 0.6, "audio": 0.2, "tactile": 0.2},
            ("audio", "tactile"): {"visual": 0.3, "audio": 0.3, "tactile": 0.4},
            ("tactile", "audio"): {"visual": 0.3, "audio": 0.4, "tactile": 0.3}
        }
        
        probs = transition_probs.get(last_transition, {"visual": 0.33, "audio": 0.33, "tactile": 0.33})
        
        # 基于概率随机选择
        modalities = list(probs.keys())
        weights = list(probs.values())
        predicted = np.random.choice(modalities, p=weights)
        
        return predicted


class CausalIntelligence:
    """因果智能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.causal_relations = {}
        self.causal_strength = {}
        
    async def process_causal_relationships(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """处理因果关系"""
        causal_data = {
            "causal_relations": [],
            "cause_effect_chains": [],
            "intervention_effects": [],
            "confounding_factors": []
        }
        
        # 检测因果关系
        causal_relations = self._detect_causal_relations(perceptions)
        causal_data["causal_relations"] = causal_relations
        
        # 构建因果链
        cause_effect_chains = self._build_causal_chains(causal_relations)
        causal_data["cause_effect_chains"] = cause_effect_chains
        
        return causal_data
    
    def _detect_causal_relations(self, perceptions: List[PerceptionData]) -> List[Dict[str, Any]]:
        """检测因果关系"""
        relations = []
        
        # 简单的因果检测：基于时间先后和相关性
        for i in range(len(perceptions) - 1):
            cause_perc = perceptions[i]
            effect_perc = perceptions[i + 1]
            
            time_gap = effect_perc.timestamp - cause_perc.timestamp
            
            # 假设短时间内的变化可能存在因果关系
            if time_gap < 10.0:  # 10秒内
                confidence = max(cause_perc.confidence, effect_perc.confidence) * (1.0 - time_gap / 10.0)
                
                relation = {
                    "cause": {
                        "modality": cause_perc.modality,
                        "timestamp": cause_perc.timestamp
                    },
                    "effect": {
                        "modality": effect_perc.modality,
                        "timestamp": effect_perc.timestamp
                    },
                    "time_gap": time_gap,
                    "confidence": confidence,
                    "strength": self._calculate_causal_strength(cause_perc, effect_perc)
                }
                relations.append(relation)
        
        return relations
    
    def _calculate_causal_strength(self, cause: PerceptionData, effect: PerceptionData) -> float:
        """计算因果强度"""
        # 基于模态转换和置信度计算因果强度
        modality_change = 1.0 if cause.modality != effect.modality else 0.5
        confidence_alignment = (cause.confidence + effect.confidence) / 2.0
        temporal_closeness = 1.0 / (1.0 + abs(effect.timestamp - cause.timestamp) / 5.0)
        
        strength = (modality_change + confidence_alignment + temporal_closeness) / 3.0
        return min(strength, 1.0)
    
    def _build_causal_chains(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建因果链"""
        chains = []
        
        if len(relations) < 2:
            return chains
        
        # 简单的链式连接
        current_chain = [relations[0]]
        
        for i in range(1, len(relations)):
            prev_relation = relations[i - 1]
            current_relation = relations[i]
            
            # 检查是否可以连接成链
            if (prev_relation["effect"]["timestamp"] <= current_relation["cause"]["timestamp"] and
                prev_relation["effect"]["modality"] == current_relation["cause"]["modality"]):
                current_chain.append(current_relation)
            else:
                # 保存当前链，开始新链
                if len(current_chain) > 1:
                    chains.append({
                        "chain": current_chain,
                        "length": len(current_chain),
                        "total_confidence": np.mean([rel["confidence"] for rel in current_chain])
                    })
                current_chain = [current_relation]
        
        # 添加最后一个链
        if len(current_chain) > 1:
            chains.append({
                "chain": current_chain,
                "length": len(current_chain),
                "total_confidence": np.mean([rel["confidence"] for rel in current_chain])
            })
        
        return chains


class CrossDomainKnowledgeTransfer:
    """跨域知识迁移机制"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_knowledge = {}
        self.transfer_strategies = self._initialize_transfer_strategies()
        self.knowledge_graph = {}
        
    def _initialize_transfer_strategies(self) -> Dict[str, Any]:
        """初始化迁移策略"""
        return {
            "structural_mapping": {
                "description": "基于结构相似性的映射",
                "applicability": ["visual->virtual", "audio->communication"]
            },
            "functional_mapping": {
                "description": "基于功能相似性的映射",
                "applicability": ["manipulation->game_control", "locomotion->navigation"]
            },
            "semantic_mapping": {
                "description": "基于语义相似性的映射",
                "applicability": ["emotion->expression", "attention->focus"]
            },
            "temporal_mapping": {
                "description": "基于时间模式相似性的映射",
                "applicability": ["any->any"]
            }
        }
    
    async def transfer_knowledge(self, source_domain: str, target_domain: str, knowledge: Any) -> Dict[str, Any]:
        """跨域知识迁移"""
        transfer_result = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "transfer_strategy": None,
            "transferred_knowledge": None,
            "success_rate": 0.0,
            "adaptations": []
        }
        
        # 选择迁移策略
        strategy = self._select_transfer_strategy(source_domain, target_domain, knowledge)
        transfer_result["transfer_strategy"] = strategy
        
        if not strategy:
            transfer_result["success_rate"] = 0.0
            return transfer_result
        
        # 执行知识迁移
        if strategy == "structural_mapping":
            transferred_knowledge = await self._structural_mapping(source_domain, target_domain, knowledge)
        elif strategy == "functional_mapping":
            transferred_knowledge = await self._functional_mapping(source_domain, target_domain, knowledge)
        elif strategy == "semantic_mapping":
            transferred_knowledge = await self._semantic_mapping(source_domain, target_domain, knowledge)
        elif strategy == "temporal_mapping":
            transferred_knowledge = await self._temporal_mapping(source_domain, target_domain, knowledge)
        else:
            transferred_knowledge = knowledge  # 直接传递
        
        transfer_result["transferred_knowledge"] = transferred_knowledge
        transfer_result["success_rate"] = 0.8  # 基础成功率
        
        # 应用适应性调整
        adaptations = await self._apply_domain_adaptations(
            transferred_knowledge, source_domain, target_domain
        )
        transfer_result["adaptations"] = adaptations
        transfer_result["success_rate"] *= (1.0 - len(adaptations) * 0.1)  # 调整成功率
        
        return transfer_result
    
    def _select_transfer_strategy(self, source_domain: str, target_domain: str, knowledge: Any) -> Optional[str]:
        """选择迁移策略"""
        # 基于域组合选择策略
        domain_pair = f"{source_domain}->{target_domain}"
        
        # 检查具体适用性
        for strategy, info in self.transfer_strategies.items():
            if any(domain_pair in app or "any->any" in app for app in info["applicability"]):
                return strategy
        
        # 默认策略
        if "visual" in [source_domain, target_domain]:
            return "structural_mapping"
        elif "manipulation" in [source_domain, target_domain] or "locomotion" in [source_domain, target_domain]:
            return "functional_mapping"
        else:
            return "semantic_mapping"
    
    async def _structural_mapping(self, source_domain: str, target_domain: str, knowledge: Any) -> Any:
        """结构映射迁移"""
        if isinstance(knowledge, dict):
            # 保持结构不变，更换域标签
            mapped_knowledge = knowledge.copy()
            for key, value in mapped_knowledge.items():
                if isinstance(value, str) and source_domain in value:
                    mapped_knowledge[key] = value.replace(source_domain, target_domain)
            return mapped_knowledge
        
        elif isinstance(knowledge, list):
            # 映射列表中的结构元素
            mapped_list = []
            for item in knowledge:
                if isinstance(item, dict) and "domain" in item:
                    item["domain"] = target_domain
                mapped_list.append(item)
            return mapped_list
        
        return knowledge
    
    async def _functional_mapping(self, source_domain: str, target_domain: str, knowledge: Any) -> Any:
        """功能映射迁移"""
        # 功能映射字典
        function_mappings = {
            "manipulation->game_control": {
                "grab": "select",
                "push": "interact",
                "release": "deselect"
            },
            "locomotion->navigation": {
                "walk_forward": "move_forward",
                "turn_left": "rotate_left",
                "turn_right": "rotate_right"
            }
        }
        
        domain_key = f"{source_domain}->{target_domain}"
        mappings = function_mappings.get(domain_key, {})
        
        if isinstance(knowledge, dict):
            mapped_knowledge = knowledge.copy()
            for key, value in mapped_knowledge.items():
                if isinstance(value, str) and value in mappings:
                    mapped_knowledge[key] = mappings[value]
            return mapped_knowledge
        
        return knowledge
    
    async def _semantic_mapping(self, source_domain: str, target_domain: str, knowledge: Any) -> Any:
        """语义映射迁移"""
        # 语义映射：基于概念相似性
        semantic_mappings = {
            "emotion->expression": {
                "happy": "smile",
                "sad": "frown",
                "angry": "fierce_expression",
                "surprised": "wide_eyes"
            },
            "attention->focus": {
                "visual_focus": "camera_focus",
                "auditory_focus": "microphone_direction",
                "tactile_focus": "sensor_precision"
            }
        }
        
        domain_key = f"{source_domain}->{target_domain}"
        mappings = semantic_mappings.get(domain_key, {})
        
        if isinstance(knowledge, dict):
            mapped_knowledge = knowledge.copy()
            for key, value in mapped_knowledge.items():
                if isinstance(value, str) and value in mappings:
                    mapped_knowledge[key] = mappings[value]
            return mapped_knowledge
        
        return knowledge
    
    async def _temporal_mapping(self, source_domain: str, target_domain: str, knowledge: Any) -> Any:
        """时间模式映射迁移"""
        # 保持时间模式，但适应目标域的时序特征
        if isinstance(knowledge, dict):
            mapped_knowledge = knowledge.copy()
            
            # 调整时间参数
            if "duration" in mapped_knowledge:
                source_speed = self._get_domain_speed(source_domain)
                target_speed = self._get_domain_speed(target_domain)
                speed_ratio = target_speed / source_speed if source_speed > 0 else 1.0
                mapped_knowledge["duration"] *= speed_ratio
            
            if "frequency" in mapped_knowledge:
                mapped_knowledge["frequency"] = target_domain
            
            return mapped_knowledge
        
        return knowledge
    
    def _get_domain_speed(self, domain: str) -> float:
        """获取域特征速度"""
        speed_map = {
            "real_world": 1.0,
            "virtual_world": 1.2,
            "game_world": 1.5
        }
        return speed_map.get(domain, 1.0)
    
    async def _apply_domain_adaptations(self, knowledge: Any, source_domain: str, target_domain: str) -> List[str]:
        """应用域适应调整"""
        adaptations = []
        
        # 精度调整
        if source_domain == "real_world" and target_domain == "game_world":
            adaptations.append("降低精度要求以适应游戏物理引擎")
            knowledge = self._adjust_precision(knowledge, 0.8)
        
        # 时延调整
        if target_domain == "virtual_world":
            adaptations.append("增加计算延迟以适应渲染过程")
            knowledge = self._adjust_timing(knowledge, 0.1)
        
        # 资源限制调整
        if target_domain == "game_world":
            adaptations.append("应用资源使用限制")
            knowledge = self._apply_resource_limits(knowledge)
        
        return adaptations
    
    def _adjust_precision(self, knowledge: Any, factor: float) -> Any:
        """调整精度"""
        if isinstance(knowledge, dict):
            adjusted = knowledge.copy()
            for key, value in adjusted.items():
                if isinstance(value, float):
                    adjusted[key] = value * factor
            return adjusted
        return knowledge
    
    def _adjust_timing(self, knowledge: Any, delay: float) -> Any:
        """调整时间参数"""
        if isinstance(knowledge, dict):
            adjusted = knowledge.copy()
            if "duration" in adjusted:
                adjusted["duration"] += delay
            return adjusted
        return knowledge
    
    def _apply_resource_limits(self, knowledge: Any) -> Any:
        """应用资源限制"""
        if isinstance(knowledge, dict):
            adjusted = knowledge.copy()
            # 限制数组大小
            for key, value in adjusted.items():
                if isinstance(value, list) and len(value) > 100:
                    adjusted[key] = value[:100]
            return adjusted
        return knowledge


class WorldMindInteractionOrchestrator:
    """世界-心智-交互编排器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_interfaces = self._initialize_world_interfaces(config.get("worlds", {}))
        self.cognitive_engine = SixDimensionCognitiveEngine(config.get("cognition", {}))
        self.embodied_intelligence = EmbodiedIntelligenceModule(config.get("embodiment", {}))
        self.world_model_builder = WorldModelBuilder(config.get("world_model", {}))
        self.knowledge_transfer = CrossDomainKnowledgeTransfer(config.get("transfer", {}))
        
        self.current_world_type = WorldType.REAL_WORLD
        self.active_perceptions = []
        self.execution_history = deque(maxlen=1000)
        
    def _initialize_world_interfaces(self, world_configs: Dict[str, Any]) -> Dict[WorldType, ExternalWorldInterface]:
        """初始化世界接口"""
        interfaces = {}
        
        if world_configs.get("real_world", {}).get("enabled", True):
            interfaces[WorldType.REAL_WORLD] = RealWorldInterface(world_configs.get("real_world", {}))
        
        if world_configs.get("virtual_world", {}).get("enabled", True):
            interfaces[WorldType.VIRTUAL_WORLD] = VirtualWorldInterface(world_configs.get("virtual_world", {}))
        
        if world_configs.get("game_world", {}).get("enabled", True):
            interfaces[WorldType.GAME_WORLD] = GameWorldInterface(world_configs.get("game_world", {}))
        
        return interfaces
    
    async def run_interaction_cycle(self, duration: float = 60.0) -> Dict[str, Any]:
        """运行完整交互循环"""
        cycle_results = {
            "start_time": time.time(),
            "duration": duration,
            "cycles_completed": 0,
            "perceptions_processed": 0,
            "actions_executed": 0,
            "world_model_updates": 0,
            "knowledge_transfers": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                cycle_start = time.time()
                
                # 1. 感知阶段
                perceptions = await self._perception_phase()
                cycle_results["perceptions_processed"] += len(perceptions)
                
                # 2. 认知处理阶段
                cognitive_states = await self._cognition_phase(perceptions)
                
                # 3. 行动规划阶段
                action_plans = await self._planning_phase(cognitive_states)
                
                # 4. 执行阶段
                execution_results = await self._execution_phase(action_plans)
                cycle_results["actions_executed"] += len(execution_results)
                
                # 5. 世界模型更新阶段
                world_model = await self._world_model_phase(perceptions + execution_results)
                if world_model:
                    cycle_results["world_model_updates"] += 1
                
                # 6. 知识迁移阶段
                transfers = await self._knowledge_transfer_phase(perceptions)
                cycle_results["knowledge_transfers"] += len(transfers)
                
                cycle_results["cycles_completed"] += 1
                
                # 循环间隔
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, 0.1 - cycle_duration)  # 最小10Hz
                await asyncio.sleep(sleep_time)
            
            cycle_results["end_time"] = time.time()
            cycle_results["actual_duration"] = cycle_results["end_time"] - cycle_results["start_time"]
            
        except Exception as e:
            cycle_results["errors"].append(str(e))
            print(f"交互循环错误: {e}")
        
        return cycle_results
    
    async def _perception_phase(self) -> List[PerceptionData]:
        """感知阶段"""
        perceptions = []
        current_interface = self.world_interfaces.get(self.current_world_type)
        
        if not current_interface:
            return perceptions
        
        # 并行感知多种模态
        modalities = ["visual", "audio", "tactile"]
        if self.current_world_type == WorldType.GAME_WORLD:
            modalities = ["game_state"]
        elif self.current_world_type == WorldType.VIRTUAL_WORLD:
            modalities = ["virtual_visual"]
        
        tasks = []
        for modality in modalities:
            tasks.append(current_interface.perceive(modality))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if not isinstance(result, Exception) and result is not None:
                perceptions.append(result)
                self.active_perceptions.append(result)
        
        # 保持感知缓存大小
        if len(self.active_perceptions) > 1000:
            self.active_perceptions = self.active_perceptions[-500:]
        
        return perceptions
    
    async def _cognition_phase(self, perceptions: List[PerceptionData]) -> List[CognitiveState]:
        """认知处理阶段"""
        cognitive_states = []
        
        # 每个感知输入都产生一个认知状态
        tasks = []
        for perception in perceptions:
            task = self.cognitive_engine.process_perception(perception)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if not isinstance(result, Exception):
                cognitive_states.append(result)
        
        return cognitive_states
    
    async def _planning_phase(self, cognitive_states: List[CognitiveState]) -> List[List[ActionCommand]]:
        """行动规划阶段"""
        all_action_plans = []
        
        for cognitive_state in cognitive_states:
            # 生成行动意图
            intentions = await self.cognitive_engine.generate_action_intention()
            
            # 为每个意图生成具体行动计划
            plans = []
            for intention in intentions:
                plan = await self.embodied_intelligence.plan_action(
                    intention, cognitive_state
                )
                plans.extend(plan)
            
            all_action_plans.append(plans)
        
        return all_action_plans
    
    async def _execution_phase(self, action_plans: List[List[ActionCommand]]) -> List[PerceptionData]:
        """执行阶段"""
        execution_results = []
        current_interface = self.world_interfaces.get(self.current_world_type)
        
        if not current_interface:
            return execution_results
        
        # 并行执行所有行动计划
        tasks = []
        for plan in action_plans:
            for action in plan:
                task = self.embodied_intelligence.execute_action(action, current_interface)
                tasks.append((action, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # 收集执行结果
        action_list = [action for plan in action_plans for action in plan]
        for i, (action, result) in enumerate(zip(action_list, results)):
            if not isinstance(result, Exception) and result:
                # 创建执行结果感知（使用正确的数据结构）
                execution_perception = PerceptionData(
                    timestamp=time.time(),
                    modality="execution_result",
                    content={
                        "action": str(action.action_type),
                        "parameters": action.parameters,
                        "success": True
                    },
                    confidence=0.9,
                    spatial_info={},
                    temporal_info={"execution_time": time.time()}
                )
                execution_results.append(execution_perception)
        
        # 记录执行历史
        for plan in action_plans:
            for action in plan:
                self.execution_history.append({
                    "timestamp": time.time(),
                    "action": action.action_type,
                    "parameters": action.parameters,
                    "world_type": self.current_world_type.value
                })
        
        return execution_results
    
    async def _world_model_phase(self, all_perceptions: List[PerceptionData]) -> Optional[Dict[str, Any]]:
        """世界模型更新阶段"""
        if len(all_perceptions) < 3:  # 需要足够数据
            return None
        
        try:
            world_model = await self.world_model_builder.build_world_model(all_perceptions[-20:])
            return world_model
        except Exception as e:
            print(f"世界模型更新失败: {e}")
            return None
    
    async def _knowledge_transfer_phase(self, perceptions: List[PerceptionData]) -> List[Dict[str, Any]]:
        """知识迁移阶段"""
        transfers = []
        
        # 尝试跨域迁移（如果有多个世界接口）
        world_types = list(self.world_interfaces.keys())
        
        if len(world_types) > 1:
            # 从当前域迁移到其他域
            for target_world in world_types:
                if target_world != self.current_world_type:
                    for perception in perceptions[-3:]:  # 迁移最近的感知
                        transfer_result = await self.knowledge_transfer.transfer_knowledge(
                            self.current_world_type.value,
                            target_world.value,
                            perception.content
                        )
                        transfers.append(transfer_result)
        
        return transfers
    
    async def switch_world(self, new_world_type: WorldType) -> bool:
        """切换世界"""
        if new_world_type not in self.world_interfaces:
            print(f"世界类型 {new_world_type.value} 不可用")
            return False
        
        print(f"切换世界从 {self.current_world_type.value} 到 {new_world_type.value}")
        self.current_world_type = new_world_type
        
        # 触发知识迁移
        if len(self.active_perceptions) > 0:
            latest_perception = self.active_perceptions[-1]
            transfer_result = await self.knowledge_transfer.transfer_knowledge(
                self.current_world_type.value,
                new_world_type.value,
                latest_perception.content
            )
            print(f"跨域知识迁移完成: {transfer_result['success_rate']:.2f}")
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "current_world": self.current_world_type.value,
            "available_worlds": [world.value for world in self.world_interfaces.keys()],
            "active_perceptions_count": len(self.active_perceptions),
            "execution_history_size": len(self.execution_history),
            "cognitive_state": {
                dimension: self.cognitive_engine.cognitive_state.six_dimensions[dimension]
                for dimension in self.cognitive_engine.cognitive_state.six_dimensions
            }
        }


# 配置示例
DEFAULT_CONFIG = {
    "worlds": {
        "real_world": {
            "enabled": True,
            "sensors": {
                "camera": {"resolution": [1920, 1080]},
                "microphone": {"sample_rate": 44100}
            }
        },
        "virtual_world": {
            "enabled": True,
            "physics": {"gravity": 9.81, "friction": 0.8}
        },
        "game_world": {
            "enabled": True,
            "game_engine": {"type": "3d_adventure", "difficulty": "medium"}
        }
    },
    "cognition": {
        "attention": {"buffer_size": 100},
        "memory": {"episodic_capacity": 1000},
        "emotion": {"valence_range": [-1, 1]},
        "reasoning": {"rule_count": 10},
        "creativity": {"novelty_threshold": 0.3},
        "meta_cognition": {"strategy_history_size": 50}
    },
    "embodiment": {
        "planning": {"horizon": 5},
        "sensors": {"fusion_enabled": True}
    },
    "world_model": {
        "spatial": {"navigation_enabled": True},
        "temporal": {"pattern_detection": True},
        "causal": {"relationship_detection": True}
    },
    "transfer": {
        "strategies": ["structural", "functional", "semantic", "temporal"]
    }
}


async def run_demo():
    """运行演示"""
    print("=== 外部世界-内部心智-交互行动架构演示 ===")
    
    # 创建编排器
    orchestrator = WorldMindInteractionOrchestrator(DEFAULT_CONFIG)
    
    # 显示系统状态
    print(f"初始系统状态: {json.dumps(orchestrator.get_system_status(), indent=2, ensure_ascii=False)}")
    
    # 运行短期交互循环
    print("\n运行5秒交互循环...")
    results = await orchestrator.run_interaction_cycle(5.0)
    print(f"交互循环结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
    
    # 切换世界演示
    print("\n切换到虚拟世界...")
    await orchestrator.switch_world(WorldType.VIRTUAL_WORLD)
    
    # 运行虚拟世界交互
    print("运行3秒虚拟世界交互...")
    virtual_results = await orchestrator.run_interaction_cycle(3.0)
    print(f"虚拟世界交互结果: {json.dumps(virtual_results, indent=2, ensure_ascii=False)}")
    
    # 切换到游戏世界
    print("\n切换到游戏世界...")
    await orchestrator.switch_world(WorldType.GAME_WORLD)
    
    # 运行游戏世界交互
    print("运行3秒游戏世界交互...")
    game_results = await orchestrator.run_interaction_cycle(3.0)
    print(f"游戏世界交互结果: {json.dumps(game_results, indent=2, ensure_ascii=False)}")
    
    # 最终系统状态
    print(f"\n最终系统状态: {json.dumps(orchestrator.get_system_status(), indent=2, ensure_ascii=False)}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行演示
    asyncio.run(run_demo())