#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化检查点管理器 - 支持断点续跑功能

实现功能：
- 自动保存最佳个体到检查点
- 支持断点续跑功能
- 管理检查点历史
- 恢复进化状态

Author: NeuroMinecraftGenesis Team
Date: 2025-11-13
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import hashlib
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CheckpointMetadata:
    """检查点元数据类"""
    generation: int
    timestamp: str
    best_fitness: float
    avg_fitness: float
    population_size: int
    genome_length: int
    diversity: float
    species_count: int
    checkpoint_type: str  # "auto", "manual", "best"
    file_hash: str
    file_size: int
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """从字典创建对象"""
        return cls(**data)


class CheckpointManager:
    """
    进化检查点管理器
    
    负责保存和恢复进化状态，支持断点续跑功能
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "models/genomes",
                 auto_save_interval: int = 10,
                 max_checkpoints: int = 100,
                 backup_enabled: bool = True):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            auto_save_interval: 自动保存间隔（代数）
            max_checkpoints: 最大保存的检查点数量
            backup_enabled: 是否启用备份
        """
        self.checkpoint_dir = checkpoint_dir
        self.auto_save_interval = auto_save_interval
        self.max_checkpoints = max_checkpoints
        self.backup_enabled = backup_enabled
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "history"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "best"), exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 检查点信息存储
        self.checkpoint_registry = {}  # generation -> metadata
        self.last_save_generation = 0
        
        # 当前进化状态
        self.current_state = {
            'generation': 0,
            'population': [],
            'fitness_scores': [],
            'best_individual': None,
            'best_fitness': float('-inf'),
            'evolution_params': {},
            'visualizer_data': {}
        }
        
        # 加载现有的检查点信息
        self._load_checkpoint_registry()
        
        self.logger.info(f"检查点管理器初始化完成 - 检查点目录: {checkpoint_dir}")
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def save_checkpoint(self,
                       population: List[np.ndarray],
                       fitness_scores: List[float],
                       generation: int,
                       best_individual: Optional[np.ndarray] = None,
                       checkpoint_type: str = "auto",
                       description: str = "") -> Dict[str, str]:
        """
        保存进化检查点
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数列表
            generation: 当前代数
            best_individual: 最佳个体（可选）
            checkpoint_type: 检查点类型 ("auto", "manual", "best")
            description: 描述信息
            
        Returns:
            保存的检查点文件路径信息
        """
        try:
            # 确定最佳个体
            if best_individual is None and fitness_scores:
                best_idx = np.argmax(fitness_scores)
                best_individual = population[best_idx]
                best_fitness = fitness_scores[best_idx]
            else:
                best_fitness = max(fitness_scores) if fitness_scores else 0.0
            
            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
            
            # 计算遗传多样性
            diversity = self._calculate_diversity(population)
            
            # 估算物种数量
            species_count = self._estimate_species_count(population)
            
            # 生成检查点文件路径
            if checkpoint_type == "best":
                # 保存最佳个体
                subdir = "best"
                filename = f"best_gen{generation:05d}_fitness{best_fitness:.6f}.pkl"
            else:
                # 保存一般检查点（包括auto类型）
                subdir = "history"
                filename = f"checkpoint_gen{generation:05d}.pkl"
            
            filepath = os.path.join(self.checkpoint_dir, subdir, filename)
            
            # 创建检查点数据
            checkpoint_data = {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'population': population,
                'fitness_scores': fitness_scores,
                'best_individual': best_individual,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'species_count': species_count,
                'evolution_params': self.current_state.get('evolution_params', {}),
                'visualizer_data': self.current_state.get('visualizer_data', {})
            }
            
            # 保存检查点文件
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # 计算文件哈希和大小
            file_hash = self._calculate_file_hash(filepath)
            file_size = os.path.getsize(filepath)
            
            # 创建元数据
            metadata = CheckpointMetadata(
                generation=generation,
                timestamp=datetime.now().isoformat(),
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                population_size=len(population),
                genome_length=len(best_individual) if best_individual is not None else 0,
                diversity=diversity,
                species_count=species_count,
                checkpoint_type=checkpoint_type,
                file_hash=file_hash,
                file_size=file_size,
                description=description
            )
            
            # 保存元数据
            metadata_path = filepath.replace('.pkl', '_meta.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 更新注册表
            self.checkpoint_registry[generation] = metadata
            self.last_save_generation = generation
            
            # 维护检查点历史
            self._maintain_checkpoint_history()
            
            self.logger.info(f"检查点保存成功: Gen {generation}, Best fitness: {best_fitness:.6f}")
            
            return {
                'filepath': filepath,
                'metadata_path': metadata_path,
                'generation': generation,
                'best_fitness': best_fitness,
                'type': checkpoint_type
            }
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            raise
    
    def load_checkpoint(self, 
                       generation: Optional[int] = None,
                       checkpoint_type: str = "auto") -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            generation: 指定代数，如果为None则加载最新的
            checkpoint_type: 检查点类型偏好 ("auto", "best", "latest")
            
        Returns:
            恢复的进化状态数据
        """
        try:
            # 确定要加载的检查点
            checkpoint_info = self._select_checkpoint(generation, checkpoint_type)
            if not checkpoint_info:
                raise ValueError("未找到合适的检查点")
            
            filepath = checkpoint_info['filepath']
            self.logger.info(f"加载检查点: {filepath}")
            
            # 加载检查点数据
            with open(filepath, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 验证数据完整性
            self._verify_checkpoint_integrity(checkpoint_data)
            
            # 更新当前状态
            self.current_state.update(checkpoint_data)
            self.current_state['generation'] = checkpoint_data['generation']
            
            # 记录加载的检查点信息
            load_info = {
                'loaded_generation': checkpoint_data['generation'],
                'loaded_fitness': checkpoint_data['best_fitness'],
                'load_timestamp': datetime.now().isoformat(),
                'source_file': filepath,
                'metadata': checkpoint_info['metadata'].to_dict() if checkpoint_info.get('metadata') else None
            }
            
            self.logger.info(f"检查点加载成功: Gen {checkpoint_data['generation']}, "
                           f"Fitness: {checkpoint_data['best_fitness']:.6f}")
            
            return {
                'state': checkpoint_data,
                'load_info': load_info,
                'resume_recommendations': self._generate_resume_recommendations(checkpoint_data)
            }
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
    
    def get_best_historical_fitness(self) -> float:
        """获取历史最佳适应度"""
        if not self.checkpoint_registry:
            return float('-inf')
        return max(meta.best_fitness for meta in self.checkpoint_registry.values())
    
    def list_checkpoints(self, 
                        checkpoint_type: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        列出检查点
        
        Args:
            checkpoint_type: 过滤类型
            limit: 限制返回数量
            
        Returns:
            检查点信息列表
        """
        checkpoints = []
        
        for generation, metadata in self.checkpoint_registry.items():
            if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                continue
            
            checkpoint_info = {
                'generation': generation,
                'timestamp': metadata.timestamp,
                'best_fitness': metadata.best_fitness,
                'avg_fitness': metadata.avg_fitness,
                'diversity': metadata.diversity,
                'species_count': metadata.species_count,
                'type': metadata.checkpoint_type,
                'file_size': metadata.file_size,
                'description': metadata.description
            }
            checkpoints.append(checkpoint_info)
        
        # 按适应度排序
        checkpoints.sort(key=lambda x: x['best_fitness'], reverse=True)
        
        return checkpoints[:limit]
    
    def should_auto_save(self, generation: int) -> bool:
        """
        检查是否需要自动保存
        
        Args:
            generation: 当前代数
            
        Returns:
            是否需要保存
        """
        return (generation - self.last_save_generation >= self.auto_save_interval and
                generation % self.auto_save_interval == 0)
    
    def cleanup_old_checkpoints(self, keep_best: int = 10):
        """
        清理旧的检查点
        
        Args:
            keep_best: 保留最佳适应度的检查点数量
        """
        try:
            checkpoints = self.list_checkpoints()
            
            if len(checkpoints) <= self.max_checkpoints:
                return
            
            # 保留最佳适应度的检查点
            best_checkpoints = checkpoints[:keep_best]
            
            # 删除其他检查点
            for checkpoint in checkpoints[keep_best:]:
                generation = checkpoint['generation']
                metadata = self.checkpoint_registry.get(generation)
                if metadata:
                    self._delete_checkpoint_files(generation)
            
            # 重新加载注册表
            self._load_checkpoint_registry()
            
            self.logger.info(f"清理完成，保留 {keep_best} 个最佳检查点")
            
        except Exception as e:
            self.logger.error(f"清理检查点失败: {e}")
    
    def export_checkpoint(self, 
                         generation: int,
                         export_path: str,
                         include_visualizer_data: bool = True) -> bool:
        """
        导出检查点到指定路径
        
        Args:
            generation: 要导出的代数
            export_path: 导出路径
            include_visualizer_data: 是否包含可视化数据
            
        Returns:
            是否导出成功
        """
        try:
            # 加载检查点
            result = self.load_checkpoint(generation)
            state = result['state']
            
            # 准备导出数据
            export_data = {
                'export_info': {
                    'original_generation': generation,
                    'export_timestamp': datetime.now().isoformat(),
                    'export_version': '1.0'
                },
                'evolution_state': {
                    'generation': state['generation'],
                    'population_size': len(state['population']),
                    'genome_length': len(state['best_individual']) if state['best_individual'] is not None else 0,
                    'best_fitness': state['best_fitness'],
                    'avg_fitness': state['avg_fitness'],
                    'diversity': state['diversity']
                },
                'genome_data': {
                    'best_individual': state['best_individual'].tolist() if state['best_individual'] is not None else None,
                    'population_sample': [ind.tolist() for ind in state['population'][:10]]  # 只导出前10个个体
                }
            }
            
            # 包含可视化数据
            if include_visualizer_data and 'visualizer_data' in state:
                export_data['visualizer_data'] = state['visualizer_data']
            
            # 包含进化参数
            if 'evolution_params' in state:
                export_data['evolution_params'] = state['evolution_params']
            
            # 保存导出文件
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"检查点导出成功: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出检查点失败: {e}")
            return False
    
    def _load_checkpoint_registry(self):
        """加载检查点注册表"""
        self.checkpoint_registry.clear()
        
        # 扫描所有元数据文件
        for root, dirs, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file.endswith('_meta.json'):
                    try:
                        metadata_path = os.path.join(root, file)
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        metadata = CheckpointMetadata.from_dict(data)
                        self.checkpoint_registry[metadata.generation] = metadata
                        
                    except Exception as e:
                        self.logger.warning(f"加载元数据失败 {metadata_path}: {e}")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _estimate_species_count(self, population: List[np.ndarray], threshold: float = 0.5) -> int:
        """估算物种数量"""
        if len(population) < 2:
            return 1
        
        # 简化的聚类方法
        from scipy.spatial.distance import pdist, squareform
        
        if len(population) > 2:
            # 构建距离矩阵
            population_matrix = np.array(population)
            distances = pdist(population_matrix)
            condensed_dist = squareform(distances)
            
            # 简单的聚类
            clusters = 1
            for i in range(len(condensed_dist)):
                for j in range(i + 1, len(condensed_dist)):
                    if condensed_dist[i, j] > threshold * np.max(condensed_dist):
                        clusters += 1
            
            return min(clusters, len(population))
        else:
            return 1
    
    def _select_checkpoint(self, generation: Optional[int], checkpoint_type: str) -> Optional[Dict[str, Any]]:
        """选择要加载的检查点"""
        if generation is not None:
            # 特定代数
            if generation in self.checkpoint_registry:
                metadata = self.checkpoint_registry[generation]
                filepath = os.path.join(self.checkpoint_dir, 
                                      "best" if metadata.checkpoint_type == "best" else "history",
                                      f"checkpoint_gen{generation:05d}.pkl")
                if os.path.exists(filepath):
                    return {'filepath': filepath, 'metadata': metadata}
        
        # 根据类型选择
        if checkpoint_type == "best":
            # 选择最佳适应度的检查点
            checkpoints = self.list_checkpoints("best")
            if checkpoints:
                best = checkpoints[0]
                metadata = self.checkpoint_registry[best['generation']]
                filepath = os.path.join(self.checkpoint_dir, "best", 
                                      f"best_gen{best['generation']:05d}_fitness{best['best_fitness']:.6f}.pkl")
                if os.path.exists(filepath):
                    return {'filepath': filepath, 'metadata': metadata}
        
        # 选择最新的检查点
        latest_generation = max(self.checkpoint_registry.keys()) if self.checkpoint_registry else None
        if latest_generation is not None:
            metadata = self.checkpoint_registry[latest_generation]
            subdir = "best" if metadata.checkpoint_type == "best" else "history"
            filename = f"best_gen{latest_generation:05d}_fitness{metadata.best_fitness:.6f}.pkl" if metadata.checkpoint_type == "best" else f"checkpoint_gen{latest_generation:05d}.pkl"
            
            filepath = os.path.join(self.checkpoint_dir, subdir, filename)
            if os.path.exists(filepath):
                return {'filepath': filepath, 'metadata': metadata}
        
        return None
    
    def _verify_checkpoint_integrity(self, checkpoint_data: Dict[str, Any]):
        """验证检查点数据完整性"""
        required_keys = ['generation', 'population', 'fitness_scores', 'best_fitness']
        
        for key in required_keys:
            if key not in checkpoint_data:
                raise ValueError(f"检查点数据缺少必要字段: {key}")
        
        if len(checkpoint_data['population']) != len(checkpoint_data['fitness_scores']):
            raise ValueError("种群和适应度数据长度不匹配")
        
        if checkpoint_data['best_individual'] is not None:
            if len(checkpoint_data['best_individual']) != len(checkpoint_data['population'][0]):
                raise ValueError("最佳个体基因组长度不匹配")
    
    def _maintain_checkpoint_history(self):
        """维护检查点历史"""
        # 清理超过限制的检查点
        if len(self.checkpoint_registry) > self.max_checkpoints:
            # 保留最佳适应度的检查点
            sorted_checkpoints = sorted(self.checkpoint_registry.items(), 
                                      key=lambda x: x[1].best_fitness, 
                                      reverse=True)
            
            # 删除较旧的检查点
            for generation, metadata in sorted_checkpoints[self.max_checkpoints:]:
                self._delete_checkpoint_files(generation)
    
    def _delete_checkpoint_files(self, generation: int):
        """删除检查点相关文件"""
        metadata = self.checkpoint_registry.get(generation)
        if metadata:
            # 删除数据文件
            if metadata.checkpoint_type == "best":
                filename = f"best_gen{generation:05d}_fitness{metadata.best_fitness:.6f}.pkl"
                subdir = "best"
            else:
                filename = f"checkpoint_gen{generation:05d}.pkl"
                subdir = "history"
            
            data_path = os.path.join(self.checkpoint_dir, subdir, filename)
            metadata_path = os.path.join(self.checkpoint_dir, subdir, filename.replace('.pkl', '_meta.json'))
            
            for path in [data_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # 从注册表中删除
            del self.checkpoint_registry[generation]
    
    def _generate_resume_recommendations(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成继续进化的建议"""
        generation = checkpoint_data['generation']
        diversity = checkpoint_data['diversity']
        best_fitness = checkpoint_data['best_fitness']
        
        recommendations = {
            'next_generation': generation + 1,
            'suggested_mutation_rate': 0.1,  # 默认值
            'suggested_crossover_rate': 0.8,  # 默认值
            'population_analysis': {}
        }
        
        # 基于多样性分析
        if diversity < 0.5:
            recommendations['population_analysis']['diversity_warning'] = "种群多样性较低，建议增加变异率或引入新的基因"
            recommendations['suggested_mutation_rate'] = 0.2
        elif diversity > 2.0:
            recommendations['population_analysis']['diversity_warning'] = "种群多样性过高，可能存在过度分散"
            recommendations['suggested_mutation_rate'] = 0.05
        
        # 基于进化停滞分析
        recommendations['evolution_status'] = self._analyze_evolution_status(checkpoint_data)
        
        return recommendations
    
    def _analyze_evolution_status(self, checkpoint_data: Dict[str, Any]) -> Dict[str, str]:
        """分析进化状态"""
        best_fitness = checkpoint_data['best_fitness']
        diversity = checkpoint_data['diversity']
        
        status = {}
        
        # 适应度水平评估
        if best_fitness > 10.0:
            status['fitness_level'] = "优秀"
        elif best_fitness > 5.0:
            status['fitness_level'] = "良好"
        else:
            status['fitness_level'] = "需要改进"
        
        # 多样性评估
        if diversity < 0.5:
            status['diversity_level'] = "低多样性"
        elif diversity > 2.0:
            status['diversity_level'] = "高多样性"
        else:
            status['diversity_level'] = "适中多样性"
        
        return status


if __name__ == "__main__":
    # 测试代码
    print("CheckpointManager 模块测试")
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="test_checkpoints",
        auto_save_interval=5,
        max_checkpoints=20
    )
    
    # 模拟进化数据
    for gen in range(15):
        # 模拟种群
        population = [np.random.randn(10) for _ in range(50)]
        fitness_scores = [np.sum(individual**2) + np.random.normal(0, 0.1) for individual in population]
        
        # 保存检查点
        if gen % 5 == 0:
            checkpoint_info = checkpoint_manager.save_checkpoint(
                population, fitness_scores, gen,
                checkpoint_type="auto" if gen % 10 != 0 else "best"
            )
            print(f"保存检查点: Gen {gen}")
    
    # 列出检查点
    checkpoints = checkpoint_manager.list_checkpoints(limit=10)
    print(f"\n找到 {len(checkpoints)} 个检查点")
    for cp in checkpoints[:3]:
        print(f"  Gen {cp['generation']}: Fitness={cp['best_fitness']:.3f}")
    
    # 加载最新检查点
    if checkpoints:
        latest_gen = checkpoints[0]['generation']
        result = checkpoint_manager.load_checkpoint(latest_gen)
        print(f"\n加载检查点: Gen {result['state']['generation']}")
        print(f"继续进化建议: {result['resume_recommendations']}")
    
    print("\nCheckpointManager 测试完成")