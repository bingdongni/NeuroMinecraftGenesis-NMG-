"""
集体记忆系统 (Collective Memory System)
为多智能体部落提供共享记忆存储和知识管理

功能特性：
- 公共危险区域标注
- 资源热点坐标存储
- 有效建造蓝图记录
- 知识版本控制和验证
- 集体学习与知识融合
"""

import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """记忆条目数据结构"""
    id: str
    content: Any
    memory_type: str  # 'danger_zone', 'resource_hotspot', 'blueprint', 'knowledge'
    timestamp: datetime
    reliability_score: float  # 0.0-1.0，可靠性评分
    contributor_id: str  # 贡献者ID
    verification_count: int  # 验证次数
    access_count: int  # 访问次数
    last_accessed: datetime
    tags: Set[str]
    spatial_coords: Optional[Tuple[float, float, float]] = None  # 空间坐标 (x, y, z)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.tags:
            self.tags = set()
        if not self.metadata:
            self.metadata = {}

class CollectiveMemory:
    """
    集体记忆系统核心类
    
    提供多智能体之间的共享记忆存储、检索和管理功能
    支持多种记忆类型的存储和智能检索
    """
    
    def __init__(self, memory_capacity: int = 10000):
        self.memory_capacity = memory_capacity
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.spatial_index: Dict[Tuple[float, float, float], Set[str]] = defaultdict(set)
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.contributor_reliability: Dict[str, float] = defaultdict(lambda: 0.5)
        self.knowledge_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 访问统计
        self.access_patterns: deque = deque(maxlen=1000)
        self.usage_statistics: Dict[str, Dict] = defaultdict(dict)
        
        # 并发控制
        self.lock = threading.RLock()
        
        # 记忆衰减和清理
        self.memory_decay_threshold = 0.1
        self.cleanup_interval = 3600  # 1小时清理一次
        
        logger.info("集体记忆系统初始化完成")
    
    def store_memory(self, memory: MemoryEntry) -> str:
        """
        存储记忆条目
        
        Args:
            memory: 记忆条目对象
            
        Returns:
            str: 记忆条目ID
        """
        with self.lock:
            # 检查容量限制
            if len(self.memory_store) >= self.memory_capacity:
                self._cleanup_memories()
            
            # 生成唯一ID
            if not memory.id:
                memory.id = f"mem_{len(self.memory_store)}_{int(time.time() * 1000)}"
            
            # 存储记忆
            self.memory_store[memory.id] = memory
            
            # 更新索引
            self._update_indices(memory)
            
            # 更新贡献者可靠性
            self._update_contributor_reliability(memory.contributor_id, memory.reliability_score)
            
            logger.info(f"存储记忆条目: {memory.id}, 类型: {memory.memory_type}")
            return memory.id
    
    def _update_indices(self, memory: MemoryEntry):
        """更新各种索引结构"""
        # 类型索引
        self.type_index[memory.memory_type].add(memory.id)
        
        # 标签索引
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        # 空间索引
        if memory.spatial_coords:
            self.spatial_index[memory.spatial_coords].add(memory.id)
        
        # 更新访问统计
        if memory.memory_type not in self.usage_statistics:
            self.usage_statistics[memory.memory_type] = {"count": 0, "total_reliability": 0}
        
        self.usage_statistics[memory.memory_type]["count"] += 1
    
    def _update_contributor_reliability(self, contributor_id: str, reliability: float):
        """更新贡献者的可靠性评分"""
        # 使用指数移动平均更新可靠性
        current = self.contributor_reliability[contributor_id]
        alpha = 0.1  # 学习率
        self.contributor_reliability[contributor_id] = (1 - alpha) * current + alpha * reliability
        
        logger.debug(f"更新贡献者 {contributor_id} 可靠性: {current:.3f} -> {self.contributor_reliability[contributor_id]:.3f}")
    
    def retrieve_memories(self, 
                         memory_type: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         spatial_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                         reliability_threshold: float = 0.0,
                         limit: int = 10) -> List[MemoryEntry]:
        """
        检索记忆条目
        
        Args:
            memory_type: 记忆类型过滤
            tags: 标签过滤
            spatial_range: 空间范围过滤 ((x1,y1,z1), (x2,y2,z2))
            reliability_threshold: 可靠性阈值
            limit: 返回结果限制
            
        Returns:
            List[MemoryEntry]: 记忆条目列表
        """
        with self.lock:
            candidates = set(self.memory_store.keys())
            
            # 类型过滤
            if memory_type:
                candidates &= self.type_index[memory_type]
            
            # 标签过滤
            if tags:
                for tag in tags:
                    if tag in self.tag_index:
                        candidates &= self.tag_index[tag]
                    else:
                        candidates.clear()  # 如果某个标签不存在，清空候选集
                        break
            
            # 空间过滤
            if spatial_range:
                (x1, y1, z1), (x2, y2, z2) = spatial_range
                filtered_ids = set()
                for mem_id in candidates:
                    memory = self.memory_store[mem_id]
                    if memory.spatial_coords:
                        x, y, z = memory.spatial_coords
                        if x1 <= x <= x2 and y1 <= y <= y2 and z1 <= z <= z2:
                            filtered_ids.add(mem_id)
                candidates = filtered_ids
            
            # 可靠性过滤和排序
            valid_memories = []
            for mem_id in candidates:
                memory = self.memory_store[mem_id]
                if memory.reliability_score >= reliability_threshold:
                    valid_memories.append(memory)
            
            # 按可靠性和访问次数排序
            valid_memories.sort(key=lambda m: (
                m.reliability_score * 0.6 + 
                min(m.access_count / 100, 1.0) * 0.4,
                m.timestamp
            ), reverse=True)
            
            # 更新访问统计
            for memory in valid_memories[:limit]:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                self.access_patterns.append((memory.id, datetime.now()))
            
            logger.info(f"检索到 {len(valid_memories[:limit])} 条记忆")
            return valid_memories[:limit]
    
    def verify_memory(self, memory_id: str, verifier_id: str, reliability: float):
        """
        验证记忆条目的可靠性
        
        Args:
            memory_id: 记忆ID
            verifier_id: 验证者ID
            reliability: 新的可靠性评分
        """
        with self.lock:
            if memory_id in self.memory_store:
                memory = self.memory_store[memory_id]
                memory.verification_count += 1
                
                # 使用贝叶斯更新可靠性
                old_score = memory.reliability_score
                alpha = 1.0 / (memory.verification_count + 1)
                memory.reliability_score = (1 - alpha) * old_score + alpha * reliability
                
                logger.info(f"验证记忆 {memory_id}: {old_score:.3f} -> {memory.reliability_score:.3f}")
            else:
                logger.warning(f"尝试验证不存在的记忆: {memory_id}")
    
    def get_resource_hotspots(self, resource_type: str, 
                            limit: int = 20) -> List[Tuple[MemoryEntry, float]]:
        """
        获取资源热点位置
        
        Args:
            resource_type: 资源类型
            limit: 返回数量限制
            
        Returns:
            List[Tuple[MemoryEntry, float]]: (记忆条目, 重要性评分) 列表
        """
        hotspots = []
        
        for memory in self.retrieve_memories(memory_type="resource_hotspot"):
            if resource_type in memory.tags or resource_type == "all":
                # 计算重要性评分
                importance = (
                    memory.reliability_score * 0.4 +
                    min(memory.access_count / 50, 1.0) * 0.3 +
                    memory.verification_count * 0.2 +
                    max(0, 1 - (datetime.now() - memory.timestamp).days / 30) * 0.1
                )
                hotspots.append((memory, importance))
        
        # 按重要性排序
        hotspots.sort(key=lambda x: x[1], reverse=True)
        
        return hotspots[:limit]
    
    def get_danger_zones(self, spatial_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None) -> List[MemoryEntry]:
        """获取危险区域信息"""
        return self.retrieve_memories(
            memory_type="danger_zone",
            reliability_threshold=0.3,
            spatial_range=spatial_range,
            limit=50
        )
    
    def get_building_blueprints(self, 
                              building_type: Optional[str] = None,
                              complexity_level: Optional[str] = None) -> List[MemoryEntry]:
        """
        获取建筑蓝图
        
        Args:
            building_type: 建筑类型
            complexity_level: 复杂程度 ('simple', 'medium', 'complex')
            
        Returns:
            List[MemoryEntry]: 蓝图记忆列表
        """
        tags = []
        if building_type:
            tags.append(building_type)
        if complexity_level:
            tags.append(complexity_level)
        
        return self.retrieve_memories(
            memory_type="blueprint",
            tags=tags,
            reliability_threshold=0.5,
            limit=20
        )
    
    def merge_knowledge(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """
        融合多个记忆条目为统一知识
        
        Args:
            memories: 要融合的记忆列表
            
        Returns:
            MemoryEntry: 融合后的记忆条目
        """
        if not memories:
            return None
        
        # 选择最可靠的记忆作为基础
        base_memory = max(memories, key=lambda m: m.reliability_score)
        
        # 融合内容
        merged_content = self._merge_content([m.content for m in memories])
        
        # 计算融合后的可靠性
        weighted_reliability = sum(
            m.reliability_score * m.reliability_score for m in memories
        ) / sum(m.reliability_score for m in memories)
        
        # 融合标签
        merged_tags = set()
        for memory in memories:
            merged_tags.update(memory.tags)
        
        # 创建融合后的记忆
        merged_memory = MemoryEntry(
            id=f"merged_{int(time.time())}",
            content=merged_content,
            memory_type=base_memory.memory_type,
            timestamp=datetime.now(),
            reliability_score=min(weighted_reliability * 1.2, 1.0),  # 融合可能提高可靠性
            contributor_id="system",
            verification_count=sum(m.verification_count for m in memories),
            access_count=sum(m.access_count for m in memories),
            last_accessed=datetime.now(),
            tags=merged_tags,
            spatial_coords=base_memory.spatial_coords,
            metadata={
                "merged_from": [m.id for m in memories],
                "merge_timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"融合了 {len(memories)} 条记忆为新知识: {merged_memory.id}")
        return merged_memory
    
    def _merge_content(self, contents: List[Any]) -> Any:
        """融合多个内容对象"""
        if not contents:
            return None
        
        if isinstance(contents[0], dict):
            # 字典内容融合
            merged = {}
            for content in contents:
                if isinstance(content, dict):
                    for key, value in content.items():
                        if key not in merged:
                            merged[key] = value
                        elif isinstance(value, list) and isinstance(merged[key], list):
                            merged[key].extend(value)
                        elif isinstance(value, str) and isinstance(merged[key], str):
                            # 文本内容去重合并
                            if value not in merged[key]:
                                merged[key] += " " + value
        else:
            # 简单内容融合
            merged = contents[0]
            for content in contents[1:]:
                if content != merged and content not in ([merged] if isinstance(merged, list) else [merged]):
                    if isinstance(merged, list):
                        merged.append(content)
        
        return merged
    
    def _cleanup_memories(self):
        """清理低可靠性和过期的记忆"""
        current_time = datetime.now()
        memories_to_remove = []
        
        for memory_id, memory in self.memory_store.items():
            # 检查是否需要清理
            should_remove = False
            
            # 可靠性过低
            if memory.reliability_score < self.memory_decay_threshold:
                should_remove = True
            
            # 过期时间（30天未访问且可靠性不高）
            days_since_access = (current_time - memory.last_accessed).days
            if days_since_access > 30 and memory.reliability_score < 0.5:
                should_remove = True
            
            if should_remove:
                memories_to_remove.append(memory_id)
        
        # 执行清理
        for memory_id in memories_to_remove:
            self._remove_memory_from_indices(memory_id)
            del self.memory_store[memory_id]
        
        logger.info(f"清理了 {len(memories_to_remove)} 条过期记忆")
    
    def _remove_memory_from_indices(self, memory_id: str):
        """从索引中移除记忆"""
        if memory_id in self.memory_store:
            memory = self.memory_store[memory_id]
            
            # 从类型索引移除
            self.type_index[memory.memory_type].discard(memory_id)
            
            # 从标签索引移除
            for tag in memory.tags:
                self.tag_index[tag].discard(memory_id)
            
            # 从空间索引移除
            if memory.spatial_coords:
                self.spatial_index[memory.spatial_coords].discard(memory_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆库统计信息"""
        with self.lock:
            total_memories = len(self.memory_store)
            type_counts = {mem_type: len(mem_ids) for mem_type, mem_ids in self.type_index.items()}
            reliability_stats = {
                "mean": np.mean([m.reliability_score for m in self.memory_store.values()]) if self.memory_store else 0,
                "min": np.min([m.reliability_score for m in self.memory_store.values()]) if self.memory_store else 0,
                "max": np.max([m.reliability_score for m in self.memory_store.values()]) if self.memory_store else 0
            }
            
            return {
                "total_memories": total_memories,
                "memory_types": type_counts,
                "total_contributors": len(self.contributor_reliability),
                "reliability_stats": reliability_stats,
                "usage_statistics": dict(self.usage_statistics),
                "memory_capacity_usage": total_memories / self.memory_capacity
            }
    
    def export_memories(self, filename: str, memory_types: Optional[List[str]] = None):
        """导出记忆数据"""
        with self.lock:
            export_data = {}
            
            for memory_id, memory in self.memory_store.items():
                if memory_types is None or memory.memory_type in memory_types:
                    export_data[memory_id] = {
                        **asdict(memory),
                        "timestamp": memory.timestamp.isoformat(),
                        "last_accessed": memory.last_accessed.isoformat(),
                        "tags": list(memory.tags)
                    }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"导出了 {len(export_data)} 条记忆到 {filename}")
    
    def import_memories(self, filename: str, contributor_id: str = "imported"):
        """导入记忆数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            for memory_id, memory_data in import_data.items():
                # 转换时间戳
                memory_data["timestamp"] = datetime.fromisoformat(memory_data["timestamp"])
                memory_data["last_accessed"] = datetime.fromisoformat(memory_data["last_accessed"])
                memory_data["tags"] = set(memory_data["tags"])
                memory_data["contributor_id"] = f"{contributor_id}_{memory_data['contributor_id']}"
                
                # 创建记忆对象
                memory = MemoryEntry(**memory_data)
                
                # 重新分配ID避免冲突
                memory.id = f"imported_{int(time.time())}_{memory_id}"
                
                # 存储记忆
                self.store_memory(memory)
            
            logger.info(f"导入了 {len(import_data)} 条记忆")
            
        except Exception as e:
            logger.error(f"导入记忆失败: {e}")


# 便捷函数
def create_danger_zone_memory(x: float, y: float, z: float, 
                             danger_type: str, description: str,
                             contributor_id: str) -> MemoryEntry:
    """创建危险区域记忆"""
    return MemoryEntry(
        id="",  # 将由系统生成
        content={
            "description": description,
            "severity": "high",  # high, medium, low
            "danger_type": danger_type,
            "position": (x, y, z)
        },
        memory_type="danger_zone",
        timestamp=datetime.now(),
        reliability_score=0.8,
        contributor_id=contributor_id,
        verification_count=0,
        access_count=0,
        last_accessed=datetime.now(),
        tags={danger_type, "hazard"},
        spatial_coords=(x, y, z)
    )

def create_resource_hotspot_memory(x: float, y: float, z: float,
                                  resource_type: str, quantity: str,
                                  quality: float, contributor_id: str) -> MemoryEntry:
    """创建资源热点记忆"""
    return MemoryEntry(
        id="",
        content={
            "resource_type": resource_type,
            "quantity": quantity,  # abundant, moderate, scarce
            "quality": quality,  # 0.0-1.0
            "accessibility": "medium",  # easy, medium, difficult
            "position": (x, y, z)
        },
        memory_type="resource_hotspot",
        timestamp=datetime.now(),
        reliability_score=0.7,
        contributor_id=contributor_id,
        verification_count=0,
        access_count=0,
        last_accessed=datetime.now(),
        tags={resource_type, "resource"},
        spatial_coords=(x, y, z),
        metadata={
            "discovery_method": "exploration",
            "verified": False
        }
    )

def create_blueprint_memory(building_type: str, blueprint_data: Dict,
                           complexity: str, contributor_id: str) -> MemoryEntry:
    """创建建筑蓝图记忆"""
    return MemoryEntry(
        id="",
        content={
            "blueprint_data": blueprint_data,
            "building_type": building_type,
            "materials_required": blueprint_data.get("materials", []),
            "construction_steps": blueprint_data.get("steps", []),
            "estimated_time": blueprint_data.get("time", 0),
            "difficulty": complexity
        },
        memory_type="blueprint",
        timestamp=datetime.now(),
        reliability_score=0.9,
        contributor_id=contributor_id,
        verification_count=0,
        access_count=0,
        last_accessed=datetime.now(),
        tags={building_type, "blueprint", complexity}
    )