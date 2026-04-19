"""
Atari游戏序列管理模块

该模块实现了Atari游戏序列管理功能，包括：
1. 100个不同Atari游戏的序列生成
2. 游戏环境管理和切换
3. 训练步数控制
4. 游戏状态跟踪

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging

# 尝试导入 gym，如果失败则使用模拟环境
try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
import random
import logging
from collections import deque
from dataclasses import dataclass
import json


@dataclass
class GameInfo:
    """游戏信息数据结构"""
    name: str
    game_id: int
    action_space: int
    observation_space: Tuple[int, ...]
    difficulty_level: float
    required_skills: List[str]
    training_complexity: float


class AtariGameSequence:
    """
    Atari游戏序列管理类
    
    该类负责管理100个不同的Atari游戏序列，包括：
    - 游戏序列生成和顺序控制
    - 游戏环境管理和状态跟踪
    - 训练步数控制和数据收集
    - 游戏间切换和状态重置
    """
    
    # 经典Atari游戏列表（简化版本，包含不同类型和难度）
    ATARI_GAMES = [
        # 经典街机游戏
        "Pong", "Breakout", "SpaceInvaders", "Asteroids", "Centipede",
        "Galaxian", "BattleZone", "RiverRaid", "Enduro", "MontezumaRevenge",
        
        # 平台跳跃游戏
        "Adventure", "Pitfall", "JungleHunt", "KeystoneKapers", "IceClimber",
        "DonkeyKong", "DonkeyKongJr", "MarioBros", "Popeye", "Qbert",
        
        # 射击游戏
        "Defender", "Galaga", "Xevious", "Robotron", "Tempest",
        "Stargate", "Phoenix", "Scramble", "LifeForce", "Bubbles",
        
        # 策略和益智游戏
        "Tetris", "Columns", "DrMario", "PuzzleBobble", "Lemmings",
        "ChipChallenge", "OilSpill", "CavemanGames", "DoubleDunk", "基巴丹",
        
        # 竞速和运动游戏
        "Enduro", "GrandPrix", "AlpineSkiing", "Skiing", "IceHockey",
        "Boxing", "Tennis", "Football", "Baseball", "Basketball",
        
        # 迷宫和探索游戏
        "MazeCraze", "RoadRunner", "Berzerk", "RobotTank", "LaserGates",
        "BankHeist", "Amidar", "Frogger", " Frogger", "MsPacman",
        
        # 角色扮演和冒险
        "Adventure", "CloudKingdom", "Atlantis", "Pitfall", "Spelunker",
        "Ghostbusters", "Mickey", "Gradius", "Contra", "MetalGear",
        
        # 经典街机（更多）
        "BurgerTime", "DigDug", "Mappy", "PacMan", "Qbert",
        "Rampage", "SmashTV", "Sinistar", "Vindicators", "WizardOfWor",
        
        # 模拟和经营游戏
        "SimCity", "SimTower", "TransportTycoon", "RailroadTycoon", "ThemePark",
        "RollerCoaster", "ZooKeeper", "FishHunter", "BattleToads", "Gradius2",
        
        # 动作和格斗游戏
        "StreetFighter", "MortalKombat", "Tekken", "VirtuaFighter", "KillerInstinct",
        "SoulCalibur", "DeadOrAlive", "Doom", "DukeNukem", "Wolfenstein",
        
        # 休闲和街机
        "Bejeweled", "PuzzleBobble", "Tetris", "PuyoPuyo", "Columns",
        "Lumines", "BustAMove", "PuzzleFighter", "Daimyo", "Sheriff",
        
        # 更多经典游戏
        "Centipede", "Frogger", "Joust", "LodeRunner", "Rtype",
        "Excitebike", "Exerion", "Fangame", "Zelda", "Metroid",
        
        # 现代风格游戏
        "GeometryWars", "Super Stardust", "Super Stardust HD", "PixelJunk", "Braid",
        "Super Meat Boy", "Terraria", "Minecraft", "Starbound", "Don't Starve",
        
        # 合作和竞技游戏
        "TeamFortress", "CounterStrike", "Overwatch", "LeagueOfLegends", "Dota2",
        "RocketLeague", "FIFA", "PES", "Madden", "NBA2K",
        
        # 创意和实验游戏
        "Minecraft", "GarrysMod", "Roblox", "Fortnite", "AmongUs",
        "FallGuys", "Phasmophobia", "VRChat", "SecondLife", "Habbo",
        
        # 动作冒险游戏
        "Zelda", "Mario", "Sonic", "CrashBandicoot", "Spyro",
        "Rayman", "Kirby", "DonkeyKong", "DKCountry", "DK64",
        
        # 平台游戏
        "SuperMario", "Sonic", "MegaMan", "MegaManX", "Castlevania",
        "Metroid", "Zelda", "Link", "Samus", "Mario64",
        
        # 射击游戏
        "Doom", "DukeNukem", "Wolfenstein", "Quake", "HalfLife",
        "UnrealTournament", "Halo", "CallOfDuty", "Battlefield", "RainbowSix",
        
        # 角色扮演游戏
        "FinalFantasy", "DragonQuest", "Persona", "ChronoTrigger", "SecretOfMana",
        "Tales", "PhantasyStar", "StarOcean", "Xenogears", "ValkyrieProfile"
    ]
    
    def __init__(self, total_games: int = 100, steps_per_game: int = 10000):
        """
        初始化Atari游戏序列
        
        Args:
            total_games: 总游戏数量
            steps_per_game: 每个游戏的训练步数
        """
        self.total_games = total_games
        self.steps_per_game = steps_per_game
        self.current_game_index = 0
        self.game_history = []
        
        # 生成游戏序列
        self.games_sequence = self._generate_games_sequence()
        
        # 游戏信息缓存
        self.games_info = {}
        self._initialize_games_info()
        
        # 当前游戏环境
        self.current_env = None
        self.current_game_name = None
        
        # 设置日志
        self.logger = logging.getLogger("atari_sequence")
        self.logger.info(f"初始化Atari游戏序列，共{len(self.games_sequence)}个游戏")
    
    def _generate_games_sequence(self) -> List[str]:
        """生成游戏序列
        
        确保包含不同类型、难度和技能要求的游戏：
        1. 难度递增：简单→中等→困难
        2. 技能多样化：反应、策略、记忆、规划
        3. 游戏类型平衡：动作、益智、策略、模拟
        """
        if self.total_games > len(self.ATARI_GAMES):
            # 如果需要更多游戏，循环使用或生成变体
            games_sequence = self.ATARI_GAMES * ((self.total_games // len(self.ATARI_GAMES)) + 1)
        else:
            games_sequence = self.ATARI_GAMES[:self.total_games]
        
        # 根据复杂度进行排序和分组
        complex_games = [
            "MontezumaRevenge", "Pitfall", "Adventure", "Robotron", "Tempest",
            "Tennis", "Boxing", "Football", "Tetris", "DrMario",
            "SimCity", "TransportTycoon", "StreetFighter", "MortalKombat",
            "CounterStrike", "Overwatch", "LeagueOfLegends", "FinalFantasy",
            "DragonQuest", "Zelda", "Mario64", "Minecraft", "Terraria"
        ]
        
        medium_games = [
            "Breakout", "Pong", "SpaceInvaders", "Asteroids", "PacMan",
            "Qbert", "DigDug", "DonkeyKong", "Enduro", "Galaga",
            "Defender", "Berzerk", "Frogger", "MsPacman", "Centipede",
            "Columns", "Tetris", "MarioBros", "Contra", "Metroid"
        ]
        
        simple_games = [
            "Blackjack", "BasicMath", "Training", "Round1", "Demo", 
            "Alley", "Airstriker", "RiverRaid", "Joust", "BurgerTime"
        ]
        
        # 构建平衡的序列：简单→中等→复杂循环
        balanced_sequence = []
        game_pool = [simple_games, medium_games, complex_games]
        
        for i in range(self.total_games):
            pool_idx = i % len(game_pool)
            games_in_pool = game_pool[pool_idx]
            
            if games_in_pool:
                game = games_in_pool[i // len(game_pool) % len(games_in_pool)]
                balanced_sequence.append(game)
            else:
                # 如果当前池子为空，使用基础序列
                balanced_sequence.append(games_sequence[i])
        
        # 随机打乱但保持大致平衡
        random.shuffle(balanced_sequence)
        
        return balanced_sequence[:self.total_games]
    
    def _initialize_games_info(self):
        """初始化游戏信息"""
        for i, game_name in enumerate(self.games_sequence):
            # 模拟游戏信息（实际应用中应该从数据库或配置文件中加载）
            game_info = self._create_game_info(game_name, i)
            self.games_info[game_name] = game_info
    
    def _create_game_info(self, game_name: str, game_id: int) -> GameInfo:
        """创建游戏信息
        
        Args:
            game_name: 游戏名称
            game_id: 游戏ID
            
        Returns:
            游戏信息对象
        """
        # 根据游戏名称确定特征
        action_spaces = {
            # 简单动作游戏
            "Pong": 3, "Breakout": 3, "SpaceInvaders": 4, "Asteroids": 6,
            "Galaxian": 4, "Centipede": 4, "DonkeyKong": 4, "PacMan": 4,
            
            # 复杂动作游戏
            "Robotron": 8, "Tempest": 8, "Defender": 8, "Berzerk": 8,
            "Contra": 6, "MegaMan": 8, "StreetFighter": 18, "MortalKombat": 19,
            
            # 策略游戏
            "Tetris": 7, "Columns": 7, "DrMario": 7, "SimCity": 6,
            "Chess": 12, "Checkers": 6,
            
            # 运动游戏
            "Tennis": 6, "Boxing": 18, "Football": 8, "Basketball": 6,
            "Baseball": 8,
            
            # 探索游戏
            "Adventure": 9, "Pitfall": 9, "Minecraft": 8, "Zelda": 12,
            "Mario64": 12
        }
        
        action_space = action_spaces.get(game_name, 6)  # 默认6个动作
        observation_space = (84, 84, 3)  # 标准Atari分辨率
        
        # 技能要求分类
        skill_mapping = {
            "Pong": ["reaction", "timing"],
            "Breakout": ["reaction", "spatial"],
            "SpaceInvaders": ["reaction", "spatial", "timing"],
            "Tetris": ["planning", "spatial", "memory"],
            "PacMan": ["pathfinding", "reaction", "memory"],
            "Adventure": ["exploration", "planning", "memory"],
            "Minecraft": ["creativity", "planning", "spatial"],
            "CounterStrike": ["reaction", "teamwork", "strategy"],
            "LeagueOfLegends": ["strategy", "teamwork", "planning"],
            "StreetFighter": ["reaction", "combo", "timing"]
        }
        
        required_skills = skill_mapping.get(game_name, ["general"])
        
        # 难度分级（基于技能复杂度和动作空间）
        if action_space <= 4:
            difficulty = 0.3  # 简单
        elif action_space <= 8:
            difficulty = 0.6  # 中等
        elif action_space <= 18:
            difficulty = 0.8  # 困难
        else:
            difficulty = 1.0  # 非常困难
        
        # 训练复杂度（基于技能要求）
        skill_weights = {
            "reaction": 0.2,
            "timing": 0.15,
            "spatial": 0.15,
            "memory": 0.15,
            "planning": 0.2,
            "pathfinding": 0.15,
            "exploration": 0.1,
            "strategy": 0.25,
            "teamwork": 0.2,
            "creativity": 0.3,
            "combo": 0.2
        }
        
        training_complexity = sum(skill_weights.get(skill, 0.1) for skill in required_skills)
        training_complexity = min(training_complexity, 1.0)
        
        return GameInfo(
            name=game_name,
            game_id=game_id,
            action_space=action_space,
            observation_space=observation_space,
            difficulty_level=difficulty,
            required_skills=required_skills,
            training_complexity=training_complexity
        )
    
    def get_current_game(self) -> Optional[GameInfo]:
        """获取当前游戏信息
        
        Returns:
            当前游戏信息，如果序列结束则返回None
        """
        if self.current_game_index >= len(self.games_sequence):
            return None
        
        game_name = self.games_sequence[self.current_game_index]
        return self.games_info[game_name]
    
    def next_game(self) -> Optional[GameInfo]:
        """切换到下一个游戏
        
        Returns:
            下一个游戏信息，如果序列结束则返回None
        """
        if self.current_game_index >= len(self.games_sequence):
            self.logger.warning("游戏序列已结束")
            return None
        
        # 记录当前游戏到历史
        current_game_name = self.games_sequence[self.current_game_index]
        self.game_history.append({
            'game_id': self.current_game_index,
            'game_name': current_game_name,
            'completion_time': None  # 可以在实际使用时记录
        })
        
        # 切换到下一个游戏
        self.current_game_index += 1
        
        if self.current_game_index >= len(self.games_sequence):
            self.logger.info("所有游戏已完成")
            return None
        
        next_game_name = self.games_sequence[self.current_game_index]
        self.logger.info(f"切换到游戏 {self.current_game_index + 1}/{len(self.games_sequence)}: {next_game_name}")
        
        return self.games_info[next_game_name]
    
    def get_game_env(self, game_id: int = None) -> Any:
        """获取游戏环境
        
        Args:
            game_id: 游戏ID，如果为None则使用当前游戏
            
        Returns:
            游戏环境对象
        """
        if game_id is None:
            game_id = self.current_game_index
        
        if game_id >= len(self.games_sequence):
            raise ValueError(f"游戏ID {game_id} 超出范围")
        
        game_name = self.games_sequence[game_id]
        
        # 检查gym是否可用
        if not GYM_AVAILABLE:
            self.logger.warning(f"Gym库不可用，为游戏 {game_name} 创建模拟环境")
            env = self._create_mock_environment(game_name)
        else:
            # 尝试使用真实Gym环境
            try:
                gym_game_name = f"{game_name}NoFrameskip-v4"
                env = gym.make(gym_game_name)
                
                # 应用Atari预处理器
                try:
                    from gym.wrappers import AtariPreprocessing
                    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
                except ImportError:
                    self.logger.warning("AtariPreprocessing不可用，使用原始环境")
                    
            except Exception as e:
                self.logger.warning(f"无法创建真实环境 {game_name}: {e}，使用模拟环境")
                env = self._create_mock_environment(game_name)
        
        self.current_env = env
        self.current_game_name = game_name
        
        return env
    
    def _create_mock_environment(self, game_name: str):
        """创建模拟环境用于演示
        
        Args:
            game_name: 游戏名称
            
        Returns:
            模拟环境对象
        """
        class MockAtariEnv:
            """模拟Atari环境"""
            
            def __init__(self, game_name):
                self.game_name = game_name
                self.game_info = self._get_game_info(game_name)
                self.action_space = self.game_info.action_space
                self.observation_space = self.game_info.observation_space
                self.current_step = 0
                self.max_steps = 1000
                
                # 模拟状态
                self.state = np.random.rand(*self.observation_space)
                self.score = 0
                self.lives = 3
                
                self.logger = logging.getLogger(f"mock_env_{game_name}")
            
            def _get_game_info(self, game_name):
                """获取游戏信息"""
                for game_info in self.games_info.values():
                    if game_info.name == game_name:
                        return game_info
                # 默认信息
                return GameInfo(
                    name=game_name,
                    game_id=0,
                    action_space=6,
                    observation_space=(84, 84, 3),
                    difficulty_level=0.5,
                    required_skills=["general"],
                    training_complexity=0.5
                )
            
            def reset(self):
                """重置环境"""
                self.current_step = 0
                self.state = np.random.rand(*self.observation_space)
                self.score = 0
                self.lives = 3
                return self.state
            
            def step(self, action):
                """执行动作"""
                self.current_step += 1
                
                # 模拟奖励
                reward = np.random.normal(0, 1)
                self.score += reward
                
                # 模拟游戏结束条件
                done = self.current_step >= self.max_steps or self.lives <= 0
                
                # 模拟新状态
                self.state = np.random.rand(*self.observation_space)
                
                # 随机生命值变化
                if np.random.random() < 0.1:  # 10%概率损失生命
                    self.lives = max(0, self.lives - 1)
                
                return self.state, reward, done, {
                    'score': self.score,
                    'lives': self.lives,
                    'steps': self.current_step
                }
            
            def close(self):
                """关闭环境"""
                pass
        
        return MockAtariEnv(game_name)
    
    def get_games_list(self) -> List[str]:
        """获取完整游戏列表
        
        Returns:
            游戏名称列表
        """
        return self.games_sequence.copy()
    
    def get_game_info(self, game_name: str) -> Optional[GameInfo]:
        """获取特定游戏信息
        
        Args:
            game_name: 游戏名称
            
        Returns:
            游戏信息对象
        """
        return self.games_info.get(game_name)
    
    def get_sequence_statistics(self) -> Dict[str, Any]:
        """获取序列统计信息
        
        Returns:
            统计信息字典
        """
        total_games = len(self.games_sequence)
        completed_games = len(self.game_history)
        
        # 技能分布统计
        all_skills = []
        difficulty_levels = []
        complexities = []
        
        for game_info in self.games_info.values():
            all_skills.extend(game_info.required_skills)
            difficulty_levels.append(game_info.difficulty_level)
            complexities.append(game_info.training_complexity)
        
        skill_counts = {}
        for skill in set(all_skills):
            skill_counts[skill] = all_skills.count(skill)
        
        return {
            'total_games': total_games,
            'completed_games': completed_games,
            'completion_rate': completed_games / total_games if total_games > 0 else 0,
            'current_game_index': self.current_game_index,
            'skill_distribution': skill_counts,
            'average_difficulty': np.mean(difficulty_levels) if difficulty_levels else 0,
            'average_complexity': np.mean(complexities) if complexities else 0,
            'difficulty_progression': difficulty_levels,
            'complexity_progression': complexities
        }
    
    def save_sequence_info(self, filepath: str):
        """保存序列信息到文件
        
        Args:
            filepath: 文件路径
        """
        sequence_info = {
            'games_sequence': self.games_sequence,
            'games_info': {
                name: {
                    'name': info.name,
                    'game_id': info.game_id,
                    'action_space': info.action_space,
                    'observation_space': info.observation_space,
                    'difficulty_level': info.difficulty_level,
                    'required_skills': info.required_skills,
                    'training_complexity': info.training_complexity
                }
                for name, info in self.games_info.items()
            },
            'game_history': self.game_history,
            'statistics': self.get_sequence_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sequence_info, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"序列信息已保存到: {filepath}")
    
    def load_sequence_info(self, filepath: str):
        """从文件加载序列信息
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sequence_info = json.load(f)
            
            # 恢复游戏序列
            self.games_sequence = sequence_info.get('games_sequence', [])
            
            # 恢复游戏信息
            for name, info_dict in sequence_info.get('games_info', {}).items():
                self.games_info[name] = GameInfo(**info_dict)
            
            # 恢复历史记录
            self.game_history = sequence_info.get('game_history', [])
            
            # 恢复当前进度
            self.current_game_index = len(self.game_history)
            
            self.logger.info(f"序列信息已从 {filepath} 加载")
            
        except Exception as e:
            self.logger.error(f"加载序列信息失败: {e}")
            raise


def create_diverse_game_sequence(total_games: int = 100, difficulty_progression: str = "gradual") -> List[str]:
    """
    创建多样化的游戏序列
    
    Args:
        total_games: 总游戏数
        difficulty_progression: 难度递增方式 ("gradual", "random", "skill_based")
        
    Returns:
        游戏名称列表
    """
    sequence_manager = AtariGameSequence(total_games)
    
    if difficulty_progression == "gradual":
        # 按复杂度排序
        games_with_complexity = [(name, info.training_complexity) 
                               for name, info in sequence_manager.games_info.items()]
        games_with_complexity.sort(key=lambda x: x[1])
        
        return [game[0] for game in games_with_complexity]
    
    elif difficulty_progression == "random":
        # 随机打乱
        games = list(sequence_manager.games_info.keys())
        random.shuffle(games)
        return games
    
    elif difficulty_progression == "skill_based":
        # 按技能类型分组排列
        skill_groups = {}
        for name, info in sequence_manager.games_info.items():
            primary_skill = info.required_skills[0] if info.required_skills else "general"
            if primary_skill not in skill_groups:
                skill_groups[primary_skill] = []
            skill_groups[primary_skill].append(name)
        
        # 按技能类型顺序排列
        skill_order = ["reaction", "timing", "spatial", "memory", "planning", "strategy"]
        organized_sequence = []
        
        for skill in skill_order:
            if skill in skill_groups:
                organized_sequence.extend(skill_groups[skill])
        
        return organized_sequence
    
    else:
        return sequence_manager.games_sequence


def main():
    """主函数 - 演示Atari游戏序列管理"""
    # 创建游戏序列
    sequence = AtariGameSequence(total_games=20, steps_per_game=1000)
    
    # 获取统计信息
    stats = sequence.get_sequence_statistics()
    print("游戏序列统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 演示游戏切换
    print("\n演示游戏切换:")
    for i in range(5):
        game_info = sequence.get_current_game()
        if game_info:
            print(f"当前游戏 {i+1}: {game_info.name} (难度: {game_info.difficulty_level:.2f})")
            sequence.next_game()
        else:
            print("序列结束")
            break
    
    # 保存序列信息
    sequence.save_sequence_info("demo_game_sequence.json")
    print("\n序列信息已保存")


if __name__ == "__main__":
    main()