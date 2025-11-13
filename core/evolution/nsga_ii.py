"""
NSGA-II多目标遗传算法选择算子实现

本模块实现了NSGA-II算法的核心选择算子，用于多目标优化问题的帕累托前沿选择。
NSGA-II是一种高效的多目标进化算法，通过非支配排序和拥挤距离计算来维护种群多样性。

特性:
- 非支配排序 (Non-dominated Sorting)
- 拥挤距离计算 (Crowding Distance)
- 精英保留策略 (Elitism)
- 帕累托前沿维护

作者: NeuroMinecraftGenesis Team
日期: 2025-11-13
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import random
from deap import tools


class NSGA2Selector:
    """
    NSGA-II多目标选择算子
    
    实现了经典的NSGA-II选择算法，通过非支配排序和拥挤距离计算
    来维护帕累托最优解集，并保持种群多样性。
    
    算法流程:
    1. 对所有个体进行非支配排序
    2. 计算拥挤距离
    3. 选择保留的个体数量
    4. 生成新的种群
    """
    
    def __init__(self):
        """初始化NSGA-II选择算子"""
        self.name = "NSGA2-Selector"
        
    def non_dominated_sorting(self, population: List, fitnesses: List[Tuple]) -> List[List]:
        """
        执行非支配排序算法
        
        将种群按照帕累托支配关系分层，每一层包含相互非支配的个体。
        前沿层(F1)包含所有非支配个体，次前沿层(F2)包含去除F1后剩余的非支配个体，以此类推。
        
        Args:
            population: 种群个体列表
            fitnesses: 适应度列表，每个适应度为多目标元组
            
        Returns:
            List[List]: 帕累托前沿层列表，每个层包含该层中的个体索引
        """
        fronts = []  # 存储各个帕累托前沿层
        domination_count = {}  # 支配每个个体的个体数量
        dominated_solutions = {}  # 每个个体支配的个体集合
        rank = {}  # 每个个体的前沿层编号
        
        n = len(population)
        
        # 初始化统计变量
        for i in range(n):
            domination_count[i] = 0
            dominated_solutions[i] = set()
            rank[i] = 0
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(fitnesses[i], fitnesses[j]):
                        # i支配j
                        dominated_solutions[i].add(j)
                    elif self._dominates(fitnesses[j], fitnesses[i]):
                        # j支配i
                        domination_count[i] += 1
            
            # 如果个体不被任何个体支配，它属于第一前沿层
            if domination_count[i] == 0:
                rank[i] = 1
        
        # 构建第一前沿层
        first_front = [i for i in range(n) if rank[i] == 1]
        fronts.append(first_front)
        
        # 构建后续前沿层
        current_front = first_front
        front_id = 1
        
        while current_front:
            next_front = []
            front_id += 1
            
            for i in current_front:
                # 对于被当前前沿层中个体支配的个体
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = front_id
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def _dominates(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        判断适应度1是否帕累托支配适应度2
        
        在多目标优化中，个体A帕累托支配个体B当且仅当：
        A在所有目标上都不劣于B，且A在至少一个目标上优于B
        
        Args:
            fitness1: 第一个个体的适应度元组
            fitness2: 第二个个体的适应度元组
            
        Returns:
            bool: 如果fitness1支配fitness2返回True，否则返回False
        """
        # 对于所有目标，i不能劣于j，且至少在一个目标上优于j
        better_in_all = all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2))
        better_in_one = any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))
        
        return better_in_all and better_in_one
    
    def crowding_distance(self, front_indices: List[int], fitnesses: List[Tuple]) -> Dict[int, float]:
        """
        计算前沿层内个体的拥挤距离
        
        拥挤距离用于衡量个体在目标空间中的分布密度。
        距离越大表示个体越稀疏（多样性越好），越可能被选中。
        
        距离计算方法:
        1. 对每个目标函数排序
        2. 边界个体距离设为无穷大
        3. 中间个体距离基于相邻个体的目标值差异计算
        
        Args:
            front_indices: 前沿层中的个体索引列表
            fitnesses: 所有个体的适应度列表
            
        Returns:
            Dict[int, float]: 个体索引到拥挤距离的映射
        """
        if len(front_indices) <= 2:
            # 前沿层个体数量少于等于2时，所有个体拥挤距离为无穷大
            return {i: float('inf') for i in front_indices}
        
        distance = {i: 0.0 for i in front_indices}
        num_objectives = len(fitnesses[0])
        
        # 为每个目标函数计算拥挤距离
        for obj_idx in range(num_objectives):
            # 按当前目标的适应度值排序
            front_sorted = sorted(front_indices, 
                                key=lambda x: fitnesses[x][obj_idx])
            
            # 边界个体距离设为无穷大
            distance[front_sorted[0]] = float('inf')
            distance[front_sorted[-1]] = float('inf')
            
            # 获取目标值的范围
            obj_values = [fitnesses[i][obj_idx] for i in front_sorted]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue  # 避免除零错误
                
            # 计算中间个体的拥挤距离
            for i in range(1, len(front_sorted) - 1):
                if distance[front_sorted[i]] != float('inf'):
                    # 拥挤距离基于相邻个体在所有目标上的差异累加
                    crowding_increment = (obj_values[i+1] - obj_values[i-1]) / obj_range
                    distance[front_sorted[i]] += crowding_increment
        
        return distance
    
    def select(self, population: List, fitnesses: List[Tuple], 
               k: int, lambda_: float = None) -> List:
        """
        使用NSGA-II选择算子选择个体
        
        结合非支配排序和拥挤距离，从当前种群中选择最优的k个个体。
        算法优先选择较低前沿层（更好解）的个体，如果前沿层容量不足，
        则根据拥挤距离选择密度较小的个体。
        
        Args:
            population: 当前种群
            fitnesses: 种群适应度列表
            k: 需要选择的个体数量
            lambda_: 选择压力参数（可选）
            
        Returns:
            List: 选择后的个体列表
        """
        if k >= len(population):
            return population[:]
        
        # 执行非支配排序
        fronts = self.non_dominated_sorting(population, fitnesses)
        
        selected = []
        
        # 选择个体直到达到目标数量
        for front in fronts:
            if len(selected) + len(front) <= k:
                # 如果当前前沿层个体全部加入后不超过目标数量，则全部加入
                selected.extend(front)
            else:
                # 需要从当前前沿层中选择部分个体
                remaining = k - len(selected)
                if remaining <= 0:
                    break
                
                # 计算当前前沿层的拥挤距离
                distances = self.crowding_distance(front, fitnesses)
                
                # 根据拥挤距离排序，优先选择拥挤距离大的个体
                front_sorted = sorted(front, key=lambda x: distances[x], reverse=True)
                
                # 选择前remaining个个体
                selected.extend(front_sorted[:remaining])
                break
        
        # 返回选择后的个体
        return [population[i] for i in selected]
    
    def get_pareto_fronts(self, population: List, fitnesses: List[Tuple]) -> List[List]:
        """
        获取所有的帕累托前沿层
        
        Args:
            population: 种群
            fitnesses: 种群适应度
            
        Returns:
            List[List]: 帕累托前沿层列表
        """
        return self.non_dominated_sorting(population, fitnesses)
    
    def get_hypervolume(self, fitnesses: List[Tuple], reference_point: Tuple = None) -> float:
        """
        计算超体积指标 (Hypervolume Indicator)
        
        超体积是衡量多目标算法性能的重要指标，表示帕累托前沿
        相对于参考点的覆盖体积。数值越大表示解集质量越好。
        
        Args:
            fitnesses: 适应度列表
            reference_point: 参考点（默认为每个目标的最小值-1）
            
        Returns:
            float: 超体积值
        """
        if not fitnesses:
            return 0.0
        
        if reference_point is None:
            # 使用所有目标最小值-1作为参考点
            reference_point = tuple(min(f[i] for f in fitnesses) - 1 
                                  for i in range(len(fitnesses[0])))
        
        # 简化的超体积计算（实际实现需要更复杂的算法）
        fronts = self.non_dominated_sorting(list(range(len(fitnesses))), fitnesses)
        if not fronts:
            return 0.0
        
        # 计算第一前沿层内个体的超体积
        first_front = fronts[0]
        if len(first_front) <= 1:
            return 1.0
        
        # 简化的体积计算（实际应该使用更精确的算法）
        volume = 0.0
        for i in first_front:
            individual_volume = 1.0
            for j, obj_val in enumerate(fitnesses[i]):
                individual_volume *= max(0, obj_val - reference_point[j] + 1)
            volume += individual_volume
        
        return volume
    
    def diversity_metrics(self, fitnesses: List[Tuple]) -> Dict[str, float]:
        """
        计算多样性指标
        
        Args:
            fitnesses: 适应度列表
            
        Returns:
            Dict[str, float]: 包含各种多样性指标的字典
        """
        if len(fitnesses) < 2:
            return {
                'spread': 0.0,
                'spacing': 0.0,
                'hypervolume': 0.0,
                'num_dominated_solutions': 0
            }
        
        # 计算解的分布范围
        num_objectives = len(fitnesses[0])
        ranges = []
        spreads = []
        
        for obj_idx in range(num_objectives):
            obj_values = [f[obj_idx] for f in fitnesses]
            obj_range = max(obj_values) - min(obj_values)
            ranges.append(obj_range)
        
        # 计算第一前沿层的分布情况
        fronts = self.non_dominated_sorting(list(range(len(fitnesses))), fitnesses)
        first_front = fronts[0] if fronts else []
        
        # 间距指标（Spacing）
        if len(first_front) > 1:
            distances = []
            for i, idx_i in enumerate(first_front):
                min_dist = float('inf')
                for j, idx_j in enumerate(first_front):
                    if i != j:
                        # 计算欧氏距离
                        dist = sum((fitnesses[idx_i][k] - fitnesses[idx_j][k])**2 
                                 for k in range(num_objectives)) ** 0.5
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            mean_dist = np.mean(distances)
            spacing = np.sqrt(np.mean([(d - mean_dist)**2 for d in distances]))
        else:
            spacing = 0.0
        
        # 超体积
        hypervolume = self.get_hypervolume(fitnesses)
        
        return {
            'spread': np.mean(ranges),
            'spacing': spacing,
            'hypervolume': hypervolume,
            'num_dominated_solutions': len(first_front)
        }