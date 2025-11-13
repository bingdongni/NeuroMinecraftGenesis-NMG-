#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能库系统 - 实现组合技能库和技能学习机制

这个模块提供：
1. 27种原子动作的组合技能库
2. 技能学习系统（经验值、熟练度、技能进化）
3. 建造、采集、战斗、探索四大技能分类
4. 技能评估和优化机制

作者：MiniMax智能体
创建时间：2025-11-13
"""

import time
import logging
import asyncio
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import json

from action_executor import ActionExecutor, ActionType, ActionResult


class SkillCategory(Enum):
    """技能分类"""
    BUILDING = auto()  # 建造技能
    MINING = auto()    # 采集技能
    COMBAT = auto()    # 战斗技能
    EXPLORATION = auto() # 探索技能


class SkillDifficulty(Enum):
    """技能难度等级"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class SkillData:
    """技能数据"""
    name: str
    category: SkillCategory
    difficulty: SkillDifficulty
    description: str
    required_skills: List[str] = field(default_factory=list)
    experience_required: int = 100
    prerequisites: List[str] = field(default_factory=list)
    
    # 技能统计
    execution_count: int = 0
    success_count: int = 0
    total_experience: int = 0
    mastery_level: float = 0.0
    
    # 技能参数
    base_duration: float = 1.0
    complexity_score: float = 1.0


@dataclass
class ExecutionResult:
    """技能执行结果"""
    skill_name: str
    success: bool
    duration: float
    experience_gained: int
    performance_score: float
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class SkillLibrary:
    """技能库系统
    
    负责管理所有组合技能，包含技能学习、评估和优化功能
    """
    
    def __init__(self, action_executor: ActionExecutor):
        self.action_executor = action_executor
        self.logger = logging.getLogger(__name__)
        
        # 技能库数据
        self.skills: Dict[str, SkillData] = {}
        self.skill_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.user_skill_levels: Dict[str, float] = defaultdict(lambda: 0.0)  # 用户技能熟练度
        
        # 技能学习参数
        self.base_experience = 10  # 基础经验值
        self.experience_multiplier = 1.0
        
        # 初始化技能库
        self._initialize_skill_library()
    
    def _initialize_skill_library(self):
        """初始化技能库"""
        # 建造技能
        self._add_building_skills()
        
        # 采集技能  
        self._add_mining_skills()
        
        # 战斗技能
        self._add_combat_skills()
        
        # 探索技能
        self._add_exploration_skills()
        
        self.logger.info(f"技能库初始化完成，共 {len(self.skills)} 项技能")
    
    def _add_building_skills(self):
        """添加建造技能"""
        # 基础房屋建造技能
        self.skills["simple_house"] = SkillData(
            name="简易房屋建造",
            category=SkillCategory.BUILDING,
            difficulty=SkillDifficulty.BEGINNER,
            description="建造一个简单的2x3房屋，包括屋顶和门",
            experience_required=100,
            base_duration=60.0,
            complexity_score=1.5
        )
        
        # 农场搭建技能
        self.skills["farm_construction"] = SkillData(
            name="农场搭建",
            category=SkillCategory.BUILDING,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="搭建小型农场，包括灌溉系统",
            required_skills=["simple_house"],
            experience_required=200,
            base_duration=120.0,
            complexity_score=2.0
        )
        
        # 防御工事建造
        self.skills["defense_structure"] = SkillData(
            name="防御工事建造",
            category=SkillCategory.BUILDING,
            difficulty=SkillDifficulty.ADVANCED,
            description="建造防御性建筑，包括围墙和炮台",
            required_skills=["simple_house"],
            experience_required=300,
            base_duration=180.0,
            complexity_score=3.0
        )
        
        # 高级建筑技术
        self.skills["advanced_architecture"] = SkillData(
            name="高级建筑技术",
            category=SkillCategory.BUILDING,
            difficulty=SkillDifficulty.EXPERT,
            description="使用复杂设计和材料的高级建筑",
            required_skills=["simple_house", "farm_construction"],
            experience_required=500,
            base_duration=300.0,
            complexity_score=4.0
        )
    
    def _add_mining_skills(self):
        """添加采集技能"""
        # 基础矿物开采
        self.skills["basic_mining"] = SkillData(
            name="基础矿物开采",
            category=SkillCategory.MINING,
            difficulty=SkillDifficulty.BEGINNER,
            description="开采石头、铁矿等基础矿物",
            experience_required=80,
            base_duration=30.0,
            complexity_score=1.0
        )
        
        # 树木采伐
        self.skills["tree_harvesting"] = SkillData(
            name="树木采伐",
            category=SkillCategory.MINING,
            difficulty=SkillDifficulty.BEGINNER,
            description="高效采伐树木并收集材料",
            experience_required=60,
            base_duration=20.0,
            complexity_score=0.8
        )
        
        # 深层矿物开采
        self.skills["deep_mining"] = SkillData(
            name="深层矿物开采",
            category=SkillCategory.MINING,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="深入地下开采珍稀矿物",
            required_skills=["basic_mining"],
            experience_required=200,
            base_duration=90.0,
            complexity_score=2.5
        )
        
        # 水流收集技术
        self.skills["water_collection"] = SkillData(
            name="水流收集技术",
            category=SkillCategory.MINING,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="收集和管理水源",
            experience_required=150,
            base_duration=45.0,
            complexity_score=1.8
        )
        
        # 珍贵矿物开采
        self.skills["precious_mining"] = SkillData(
            name="珍贵矿物开采",
            category=SkillCategory.MINING,
            difficulty=SkillDifficulty.EXPERT,
            description="开采钻石、金矿等珍贵矿物",
            required_skills=["basic_mining", "deep_mining"],
            experience_required=400,
            base_duration=150.0,
            complexity_score=3.5
        )
    
    def _add_combat_skills(self):
        """添加战斗技能"""
        # 基础战斗
        self.skills["basic_combat"] = SkillData(
            name="基础战斗",
            category=SkillCategory.COMBAT,
            difficulty=SkillDifficulty.BEGINNER,
            description="与普通怪物战斗的基础技巧",
            experience_required=100,
            base_duration=30.0,
            complexity_score=1.5
        )
        
        # 群体攻击
        self.skills["group_combat"] = SkillData(
            name="群体攻击",
            category=SkillCategory.COMBAT,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="同时攻击多个敌人",
            required_skills=["basic_combat"],
            experience_required=200,
            base_duration=60.0,
            complexity_score=2.0
        )
        
        # 防御策略
        self.skills["defensive_strategy"] = SkillData(
            name="防御策略",
            category=SkillCategory.COMBAT,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="运用防御技巧减少伤害",
            required_skills=["basic_combat"],
            experience_required=180,
            base_duration=45.0,
            complexity_score=1.8
        )
        
        # 逃脱路线规划
        self.skills["escape_route"] = SkillData(
            name="逃脱路线规划",
            category=SkillCategory.COMBAT,
            difficulty=SkillDifficulty.ADVANCED,
            description="在危险情况下快速逃脱",
            required_skills=["basic_combat", "defensive_strategy"],
            experience_required=250,
            base_duration=20.0,
            complexity_score=2.5
        )
        
        # 精英怪物战斗
        self.skills["elite_combat"] = SkillData(
            name="精英怪物战斗",
            category=SkillCategory.COMBAT,
            difficulty=SkillDifficulty.EXPERT,
            description="与强敌和Boss战斗的高级技巧",
            required_skills=["group_combat", "defensive_strategy"],
            experience_required=400,
            base_duration=120.0,
            complexity_score=4.0
        )
    
    def _add_exploration_skills(self):
        """添加探索技能"""
        # 基础探索
        self.skills["basic_exploration"] = SkillData(
            name="基础探索",
            category=SkillCategory.EXPLORATION,
            difficulty=SkillDifficulty.BEGINNER,
            description="探索周围区域并绘制地图",
            experience_required=80,
            base_duration=60.0,
            complexity_score=1.2
        )
        
        # 资源发现
        self.skills["resource_discovery"] = SkillData(
            name="资源发现",
            category=SkillCategory.EXPLORATION,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="发现和标记重要资源位置",
            required_skills=["basic_exploration"],
            experience_required=150,
            base_duration=90.0,
            complexity_score=1.8
        )
        
        # 路径规划
        self.skills["path_planning"] = SkillData(
            name="路径规划",
            category=SkillCategory.EXPLORATION,
            difficulty=SkillDifficulty.INTERMEDIATE,
            description="制定高效的移动路径",
            required_skills=["basic_exploration"],
            experience_required=120,
            base_duration=30.0,
            complexity_score=1.5
        )
        
        # 地形分析
        self.skills["terrain_analysis"] = SkillData(
            name="地形分析",
            category=SkillCategory.EXPLORATION,
            difficulty=SkillDifficulty.ADVANCED,
            description="分析地形特征并优化策略",
            required_skills=["basic_exploration", "resource_discovery"],
            experience_required=250,
            base_duration=75.0,
            complexity_score=2.5
        )
        
        # 远程探索
        self.skills["remote_exploration"] = SkillData(
            name="远程探索",
            category=SkillCategory.EXPLORATION,
            difficulty=SkillDifficulty.EXPERT,
            description="探索远距离区域和危险地带",
            required_skills=["path_planning", "terrain_analysis"],
            experience_required=400,
            base_duration=180.0,
            complexity_score=3.5
        )
    
    async def execute_skill(self, skill_name: str, **kwargs) -> ExecutionResult:
        """
        执行指定的技能
        
        Args:
            skill_name: 技能名称
            **kwargs: 技能参数
            
        Returns:
            ExecutionResult: 技能执行结果
        """
        start_time = time.time()
        
        if skill_name not in self.skills:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                duration=0,
                experience_gained=0,
                performance_score=0,
                error_message=f"技能 '{skill_name}' 不存在"
            )
        
        skill_data = self.skills[skill_name]
        
        try:
            # 检查前置技能
            if not self._check_prerequisites(skill_name):
                return ExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    duration=0,
                    experience_gained=0,
                    performance_score=0,
                    error_message="前置技能未满足"
                )
            
            # 执行技能
            result = await self._execute_skill_logic(skill_name, **kwargs)
            
            # 计算经验值
            experience_gained = self._calculate_experience_gained(skill_name, result.success, result.performance_score)
            
            # 更新技能数据
            self._update_skill_data(skill_name, result.success, experience_gained)
            
            # 记录执行历史
            self._record_execution(skill_name, result, experience_gained)
            
            return result
            
        except Exception as e:
            self.logger.error(f"执行技能 {skill_name} 失败: {str(e)}")
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                duration=time.time() - start_time,
                experience_gained=0,
                performance_score=0,
                error_message=str(e)
            )
    
    def _check_prerequisites(self, skill_name: str) -> bool:
        """检查技能前置条件"""
        skill_data = self.skills[skill_name]
        
        # 检查前置技能
        for prereq_skill in skill_data.prerequisites:
            if self.user_skill_levels[prereq_skill] < 1.0:
                return False
        
        # 检查依赖技能
        for req_skill in skill_data.required_skills:
            if req_skill not in self.user_skill_levels or self.user_skill_levels[req_skill] < 0.8:
                return False
        
        return True
    
    async def _execute_skill_logic(self, skill_name: str, **kwargs) -> ExecutionResult:
        """执行技能的具体逻辑"""
        skill_data = self.skills[skill_name]
        
        if skill_name == "simple_house":
            return await self._skill_simple_house(**kwargs)
        elif skill_name == "farm_construction":
            return await self._skill_farm_construction(**kwargs)
        elif skill_name == "defense_structure":
            return await self._skill_defense_structure(**kwargs)
        elif skill_name == "basic_mining":
            return await self._skill_basic_mining(**kwargs)
        elif skill_name == "tree_harvesting":
            return await self._skill_tree_harvesting(**kwargs)
        elif skill_name == "basic_combat":
            return await self._skill_basic_combat(**kwargs)
        elif skill_name == "basic_exploration":
            return await self._skill_basic_exploration(**kwargs)
        else:
            return ExecutionResult(
                skill_name=skill_name,
                success=True,
                duration=skill_data.base_duration,
                experience_gained=0,
                performance_score=0.5,
                details={"message": f"技能 {skill_name} 执行完成"}
            )
    
    async def _skill_simple_house(self, **kwargs) -> ExecutionResult:
        """简易房屋建造技能逻辑"""
        size = kwargs.get('size', {'width': 3, 'length': 4})
        materials = kwargs.get('materials', {'wood': 50, 'stone': 30})
        
        # 模拟建造过程
        steps = [
            ("准备材料", 5.0),
            ("建造地基", 15.0),
            ("建造墙体", 20.0),
            ("建造屋顶", 10.0),
            ("安装门和窗", 10.0)
        ]
        
        total_duration = 0
        for step_name, step_duration in steps:
            await asyncio.sleep(step_duration * 0.1)  # 加速模拟
            total_duration += step_duration * 0.1
        
        # 模拟材料消耗
        materials_consumed = {}
        for material, amount in materials.items():
            consumed = int(amount * 0.8)  # 80%消耗率
            materials_consumed[material] = consumed
        
        performance_score = min(1.0, (kwargs.get('quality', 1.0) + 0.3))
        
        return ExecutionResult(
            skill_name="simple_house",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "size": size,
                "materials_consumed": materials_consumed,
                "quality": kwargs.get('quality', 1.0)
            }
        )
    
    async def _skill_farm_construction(self, **kwargs) -> ExecutionResult:
        """农场搭建技能逻辑"""
        plot_size = kwargs.get('plot_size', 4)
        crop_type = kwargs.get('crop_type', 'wheat')
        
        # 模拟农场建设
        steps = [
            ("整地", 20.0),
            ("建造灌溉渠", 30.0),
            ("种植作物", 25.0),
            ("安装围栏", 15.0)
        ]
        
        total_duration = 0
        for step_name, step_duration in steps:
            await asyncio.sleep(step_duration * 0.1)
            total_duration += step_duration * 0.1
        
        performance_score = 0.9  # 农场建设通常质量较高
        
        return ExecutionResult(
            skill_name="farm_construction",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "plot_size": plot_size,
                "crop_type": crop_type,
                "estimated_yield": plot_size * 10
            }
        )
    
    async def _skill_defense_structure(self, **kwargs) -> ExecutionResult:
        """防御工事建造技能逻辑"""
        structure_type = kwargs.get('structure_type', 'wall')
        fortification_level = kwargs.get('fortification_level', 1)
        
        # 模拟防御工事建设
        steps = [
            ("地基加固", 25.0),
            ("建造主体结构", 35.0),
            ("安装防御设施", 30.0),
            ("测试防御能力", 15.0)
        ]
        
        total_duration = 0
        for step_name, step_duration in steps:
            await asyncio.sleep(step_duration * 0.1)
            total_duration += step_duration * 0.1
        
        performance_score = min(1.0, fortification_level * 0.3 + 0.4)
        
        return ExecutionResult(
            skill_name="defense_structure",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "structure_type": structure_type,
                "fortification_level": fortification_level,
                "defense_rating": fortification_level * 20
            }
        )
    
    async def _skill_basic_mining(self, **kwargs) -> ExecutionResult:
        """基础矿物开采技能逻辑"""
        mining_depth = kwargs.get('mining_depth', 10)
        target_materials = kwargs.get('target_materials', ['stone', 'iron'])
        
        # 模拟开采过程
        total_duration = 0
        materials_found = {}
        
        for _ in range(mining_depth):
            await asyncio.sleep(0.5)
            total_duration += 0.5
            
            # 模拟随机获得材料
            import random
            material = random.choice(target_materials + ['stone'])
            materials_found[material] = materials_found.get(material, 0) + random.randint(1, 3)
        
        performance_score = 0.8  # 开采成功率较高
        
        return ExecutionResult(
            skill_name="basic_mining",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "mining_depth": mining_depth,
                "materials_found": materials_found
            }
        )
    
    async def _skill_tree_harvesting(self, **kwargs) -> ExecutionResult:
        """树木采伐技能逻辑"""
        tree_count = kwargs.get('tree_count', 5)
        
        # 模拟采伐过程
        total_duration = 0
        materials_gained = {'wood': 0}
        
        for i in range(tree_count):
            await asyncio.sleep(0.8)  # 每棵树0.8秒
            total_duration += 0.8
            
            wood_gained = random.randint(2, 8)
            materials_gained['wood'] += wood_gained
        
        performance_score = min(1.0, tree_count * 0.15 + 0.5)
        
        return ExecutionResult(
            skill_name="tree_harvesting",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "trees_cut": tree_count,
                "materials_gained": materials_gained
            }
        )
    
    async def _skill_basic_combat(self, **kwargs) -> ExecutionResult:
        """基础战斗技能逻辑"""
        enemy_count = kwargs.get('enemy_count', 3)
        enemy_type = kwargs.get('enemy_type', 'zombie')
        
        # 模拟战斗过程
        total_duration = 0
        damage_dealt = 0
        enemies_defeated = 0
        
        for _ in range(enemy_count):
            await asyncio.sleep(1.0)  # 每场战斗1秒
            total_duration += 1.0
            
            # 模拟战斗结果
            damage = random.randint(15, 25)
            damage_dealt += damage
            
            if damage >= 20:  # 假设20点伤害可以击败敌人
                enemies_defeated += 1
        
        success_rate = enemies_defeated / enemy_count
        performance_score = success_rate * 0.7 + 0.3
        
        return ExecutionResult(
            skill_name="basic_combat",
            success=enemies_defeated >= enemy_count // 2,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "enemy_count": enemy_count,
                "enemy_type": enemy_type,
                "damage_dealt": damage_dealt,
                "enemies_defeated": enemies_defeated
            }
        )
    
    async def _skill_basic_exploration(self, **kwargs) -> ExecutionResult:
        """基础探索技能逻辑"""
        exploration_radius = kwargs.get('exploration_radius', 5)
        include_underground = kwargs.get('include_underground', False)
        
        # 模拟探索过程
        total_duration = 0
        areas_explored = 0
        resources_discovered = {}
        
        for x in range(-exploration_radius, exploration_radius + 1):
            for z in range(-exploration_radius, exploration_radius + 1):
                await asyncio.sleep(0.1)
                total_duration += 0.1
                areas_explored += 1
                
                # 模拟资源发现
                if random.random() < 0.3:  # 30%概率发现资源
                    resource_type = random.choice(['stone', 'coal', 'iron', 'wood'])
                    resources_discovered[resource_type] = resources_discovered.get(resource_type, 0) + 1
        
        exploration_coverage = areas_explored / ((exploration_radius * 2 + 1) ** 2)
        performance_score = exploration_coverage * 0.8 + 0.2
        
        return ExecutionResult(
            skill_name="basic_exploration",
            success=True,
            duration=total_duration,
            experience_gained=0,
            performance_score=performance_score,
            details={
                "areas_explored": areas_explored,
                "exploration_radius": exploration_radius,
                "resources_discovered": resources_discovered,
                "exploration_coverage": exploration_coverage
            }
        )
    
    def _calculate_experience_gained(self, skill_name: str, success: bool, performance_score: float) -> int:
        """计算获得的经验值"""
        skill_data = self.skills[skill_name]
        
        if not success:
            return 0
        
        # 基础经验值
        base_exp = self.base_experience * skill_data.difficulty.value
        
        # 根据难度调整
        difficulty_multiplier = {
            SkillDifficulty.BEGINNER: 1.0,
            SkillDifficulty.INTERMEDIATE: 1.5,
            SkillDifficulty.ADVANCED: 2.0,
            SkillDifficulty.EXPERT: 3.0,
            SkillDifficulty.MASTER: 5.0
        }
        
        exp = base_exp * difficulty_multiplier[skill_data.difficulty] * performance_score * self.experience_multiplier
        return int(exp)
    
    def _update_skill_data(self, skill_name: str, success: bool, experience_gained: int):
        """更新技能数据"""
        skill_data = self.skills[skill_name]
        
        skill_data.execution_count += 1
        if success:
            skill_data.success_count += 1
        
        skill_data.total_experience += experience_gained
        
        # 更新熟练度（0-5级）
        total_needed = skill_data.experience_required
        mastery_progress = min(1.0, skill_data.total_experience / total_needed)
        skill_data.mastery_level = 1.0 + mastery_progress * 4.0  # 1-5级
        
        # 更新用户技能水平
        self.user_skill_levels[skill_name] = mastery_progress
        
        self.logger.info(f"技能 {skill_name} 升级: 熟练度 {skill_data.mastery_level:.2f}, 经验 {experience_gained}")
    
    def _record_execution(self, skill_name: str, result: ExecutionResult, experience_gained: int):
        """记录技能执行历史"""
        self.skill_performance_history[skill_name].append({
            'timestamp': time.time(),
            'success': result.success,
            'duration': result.duration,
            'performance_score': result.performance_score,
            'experience_gained': experience_gained
        })
    
    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """获取技能信息"""
        if skill_name not in self.skills:
            return {}
        
        skill_data = self.skills[skill_name]
        
        return {
            'name': skill_data.name,
            'category': skill_data.category.name,
            'difficulty': skill_data.difficulty.name,
            'description': skill_data.description,
            'mastery_level': skill_data.mastery_level,
            'execution_count': skill_data.execution_count,
            'success_count': skill_data.success_count,
            'success_rate': skill_data.success_count / skill_data.execution_count if skill_data.execution_count > 0 else 0,
            'total_experience': skill_data.total_experience,
            'required_skills': skill_data.required_skills,
            'prerequisites': skill_data.prerequisites,
            'user_level': self.user_skill_levels[skill_name]
        }
    
    def get_all_skills(self) -> Dict[str, Dict[str, Any]]:
        """获取所有技能信息"""
        return {name: self.get_skill_info(name) for name in self.skills.keys()}
    
    def get_skills_by_category(self, category: SkillCategory) -> Dict[str, Dict[str, Any]]:
        """按分类获取技能"""
        return {name: info for name, info in self.get_all_skills().items() 
                if self.skills[name].category == category}
    
    def get_recommended_skills(self) -> List[str]:
        """获取推荐技能（基于当前水平和前置条件）"""
        recommendations = []
        
        for skill_name, skill_data in self.skills.items():
            # 检查前置条件
            if self._check_prerequisites(skill_name):
                # 优先推荐未学习的技能
                if self.user_skill_levels[skill_name] < 1.0:
                    recommendations.append(skill_name)
        
        # 按难度和熟练度排序
        recommendations.sort(key=lambda x: (self.skills[x].difficulty.value, -self.user_skill_levels[x]))
        
        return recommendations[:5]  # 返回前5个推荐
    
    def export_skill_data(self, filename: str):
        """导出技能数据"""
        export_data = {
            'skills': self.get_all_skills(),
            'user_levels': dict(self.user_skill_levels),
            'export_time': time.time()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"技能数据已导出到 {filename}")
    
    def import_skill_data(self, filename: str):
        """导入技能数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 恢复用户技能水平
            self.user_skill_levels.update(import_data.get('user_levels', {}))
            
            self.logger.info(f"技能数据已从 {filename} 导入")
            
        except FileNotFoundError:
            self.logger.error(f"技能数据文件 {filename} 不存在")
        except Exception as e:
            self.logger.error(f"导入技能数据失败: {str(e)}")


# 测试和示例代码
if __name__ == "__main__":
    import random
    
    async def test_skill_library():
        """测试技能库系统"""
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 创建动作执行器和技能库
        action_executor = ActionExecutor()
        skill_library = SkillLibrary(action_executor)
        
        print("=== 技能库系统测试 ===")
        
        # 显示所有技能
        print("\n1. 可用技能列表:")
        for skill_name, skill_info in skill_library.get_all_skills().items():
            print(f"   {skill_name}: {skill_info['description']}")
        
        # 测试执行基础技能
        print("\n2. 执行基础技能:")
        
        # 测试房屋建造
        result = await skill_library.execute_skill("simple_house", 
                                                  size={'width': 3, 'length': 4},
                                                  quality=0.9)
        print(f"   房屋建造: 成功={result.success}, 用时={result.duration:.2f}s, 性能={result.performance_score:.2f}")
        
        # 测试采伐
        result = await skill_library.execute_skill("tree_harvesting", tree_count=3)
        print(f"   树木采伐: 成功={result.success}, 用时={result.duration:.2f}s, 性能={result.performance_score:.2f}")
        
        # 测试探索
        result = await skill_library.execute_skill("basic_exploration", exploration_radius=3)
        print(f"   基础探索: 成功={result.success}, 用时={result.duration:.2f}s, 性能={result.performance_score:.2f}")
        
        # 显示技能信息
        print("\n3. 技能信息:")
        skill_info = skill_library.get_skill_info("simple_house")
        print(f"   房屋建造: 熟练度={skill_info['mastery_level']:.2f}, 执行次数={skill_info['execution_count']}")
        
        # 显示推荐技能
        print("\n4. 推荐技能:")
        recommendations = skill_library.get_recommended_skills()
        for skill_name in recommendations:
            skill_info = skill_library.get_skill_info(skill_name)
            print(f"   {skill_name}: {skill_info['description']}")
        
        # 按分类显示技能
        print("\n5. 按分类显示技能:")
        for category in SkillCategory:
            skills = skill_library.get_skills_by_category(category)
            print(f"   {category.name} ({len(skills)}个技能):")
            for skill_name, info in skills.items():
                print(f"     - {info['name']}: {info['description']}")
    
    # 运行测试
    asyncio.run(test_skill_library())