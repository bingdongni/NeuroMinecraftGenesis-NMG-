"""
思维力监控模块
=============

专门负责思维力能力的监控、评估和分析。

思维力定义：
- 逻辑思维：基于规则和证据的推理能力
- 批判性思维：分析和评估信息的能力
- 创造性思维：产生新颖想法和解决方案的能力
- 系统性思维：理解复杂系统关系的能力
- 抽象思维：处理抽象概念的能力
- 归纳推理：从具体到一般的推理能力
- 演绎推理：从一般到具体的推理能力

主要功能：
- 实时思维指标监控
- 思维性能评估
- 思维模式分析
- 思维深度测量
- 推理能力测试

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math


class ThinkingMonitor:
    """
    思维力监控器
    
    监控和评估思维力相关的各种认知能力：
    1. 逻辑推理能力和速度
    2. 问题解决策略和方法
    3. 批判性思维和判断力
    4. 抽象思维和概念理解
    5. 系统分析和综合能力
    6. 创新思维和发散思维
    """
    
    def __init__(self):
        """初始化思维力监控器"""
        self.component_name = "思维力监控器"
        self.last_update_time = time.time()
        
        # 思维力的子组件
        self.thinking_components = {
            "logical_reasoning": {
                "name": "逻辑推理",
                "description": "基于规则的推理分析",
                "weight": 0.25
            },
            "critical_thinking": {
                "name": "批判性思维",
                "description": "分析评估和判断能力",
                "weight": 0.20
            },
            "abstract_thinking": {
                "name": "抽象思维",
                "description": "概念理解和抽象处理",
                "weight": 0.20
            },
            "systematic_thinking": {
                "name": "系统性思维",
                "description": "复杂系统分析能力",
                "weight": 0.15
            },
            "creative_thinking": {
                "name": "创造性思维",
                "description": "创新思维和发散思考",
                "weight": 0.20
            }
        }
        
        # 思维性能基准
        self.thinking_benchmarks = {
            "logical_reasoning_speed": {
                "excellent": 3.5,   # 逻辑推理速度（题/分钟）
                "good": 2.8,
                "average": 2.0,
                "poor": 1.2
            },
            "reasoning_accuracy": {
                "excellent": 0.92,  # 推理准确率
                "good": 0.85,
                "average": 0.78,
                "poor": 0.65
            },
            "problem_solving_efficiency": {
                "excellent": 0.90,  # 问题解决效率
                "good": 0.80,
                "average": 0.70,
                "poor": 0.55
            },
            "abstract_concept_grasping": {
                "excellent": 0.88,  # 抽象概念理解
                "good": 0.78,
                "average": 0.68,
                "poor": 0.52
            },
            "system_analysis_depth": {
                "excellent": 0.85,  # 系统分析深度
                "good": 0.75,
                "average": 0.65,
                "poor": 0.50
            }
        }
        
        # 当前思维状态
        self.current_thinking_state = self._initialize_thinking_state()
        
        # 思维历史数据
        self.thinking_history = []
        
        # 思维测试任务
        self.thinking_tasks = [
            "逻辑推理题",
            "数学证明题",
            "概念分类任务",
            "系统分析题",
            "创意解决问题",
            "批判性阅读",
            "抽象模式识别",
            "因果关系推理",
            "归纳总结任务",
            "演绎推理题"
        ]
    
    def _initialize_thinking_state(self) -> Dict[str, Any]:
        """初始化思维状态"""
        current_time = time.time()
        
        # 生成初始思维指标
        logical_score = self._generate_logical_reasoning_score()
        critical_score = self._generate_critical_thinking_score()
        abstract_score = self._generate_abstract_thinking_score()
        systematic_score = self._generate_systematic_thinking_score()
        creative_score = self._generate_creative_thinking_score()
        
        # 计算加权综合得分
        overall_score = (
            logical_score * 0.25 +
            critical_score * 0.20 +
            abstract_score * 0.20 +
            systematic_score * 0.15 +
            creative_score * 0.20
        )
        
        return {
            "timestamp": current_time,
            "overall_score": round(overall_score, 1),
            
            # 逻辑推理指标
            "logical_reasoning_speed": logical_score,
            "logical_reasoning_accuracy": self._generate_reasoning_accuracy(),
            "deductive_reasoning": self._generate_deductive_reasoning(),
            "inductive_reasoning": self._generate_inductive_reasoning(),
            
            # 批判性思维指标
            "critical_analysis_accuracy": critical_score,
            "evidence_evaluation": self._generate_evidence_evaluation(),
            "bias_detection": self._generate_bias_detection(),
            "logical_fallacy_identification": self._generate_fallacy_identification(),
            
            # 抽象思维指标
            "abstract_concept_grasping": abstract_score,
            "symbolic_processing": self._generate_symbolic_processing(),
            "pattern_recognition": self._generate_pattern_recognition(),
            "analogical_reasoning": self._generate_analogical_reasoning(),
            
            # 系统性思维指标
            "system_analysis_depth": systematic_score,
            "holistic_thinking": self._generate_holistic_thinking(),
            "cause_effect_modeling": self._generate_cause_effect_modeling(),
            "complex_system_navigation": self._generate_system_navigation(),
            
            # 创造性思维指标
            "divergent_thinking": creative_score,
            "idea_fluency": self._generate_idea_fluency(),
            "idea_flexibility": self._generate_idea_flexibility(),
            "idea_originality": self._generate_idea_originality(),
            
            # 综合思维指标
            "thinking_depth": self._calculate_thinking_depth(overall_score),
            "thinking_speed": self._generate_thinking_speed(),
            "thinking_flexibility": self._generate_thinking_flexibility(),
            "thinking_persistence": self._generate_thinking_persistence(),
            
            # 思维状态指标
            "cognitive_load": self._generate_cognitive_load(),
            "mental_energy": self._generate_mental_energy(),
            "focus_level": self._generate_focus_level(),
            "mental_fatigue": self._generate_mental_fatigue(),
            
            # 响应时间
            "response_time": self._generate_thinking_response_time(),
            
            # 稳定性
            "stability": self._generate_thinking_stability(),
            
            # 效率
            "efficiency": self._calculate_thinking_efficiency(overall_score),
            
            # 趋势
            "trend": "stable"
        }
    
    def _generate_logical_reasoning_score(self) -> float:
        """生成逻辑推理得分"""
        # 逻辑推理受教育和训练影响较大
        base_score = random.uniform(70, 92)
        
        # 时间因素（早晨思维更清晰）
        hour = datetime.now().hour
        if 7 <= hour <= 11:
            cognitive_boost = 3
        elif 14 <= hour <= 17:
            cognitive_boost = 0
        else:
            cognitive_boost = -4
        
        # 添加随机波动
        random_factor = np.random.normal(0, 2)
        score = base_score + cognitive_boost + random_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_critical_thinking_score(self) -> float:
        """生成批判性思维得分"""
        # 批判性思维需要经验积累
        base_score = random.uniform(65, 88)
        
        # 受教育和经验影响
        education_factor = np.random.uniform(-2, 6)
        experience_factor = np.random.uniform(-3, 4)
        score = base_score + education_factor + experience_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_abstract_thinking_score(self) -> float:
        """生成抽象思维得分"""
        # 抽象思维能力相对稳定
        base_score = random.uniform(68, 90)
        
        # 少量随机波动
        noise = np.random.normal(0, 3)
        score = base_score + noise
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_systematic_thinking_score(self) -> float:
        """生成系统性思维得分"""
        # 系统性思维需要训练
        base_score = random.uniform(60, 85)
        
        # 专业训练影响
        training_factor = np.random.uniform(-1, 8)
        score = base_score + training_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_creative_thinking_score(self) -> float:
        """生成创造性思维得分"""
        # 创造性思维波动较大
        base_score = random.uniform(55, 88)
        
        # 情绪和环境影响
        mood_factor = np.random.choice([-5, 0, 3, 7], p=[0.1, 0.3, 0.4, 0.2])
        environment_factor = np.random.uniform(-2, 4)
        score = base_score + mood_factor + environment_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_reasoning_accuracy(self) -> float:
        """生成推理准确率"""
        return round(random.uniform(0.78, 0.95), 3)
    
    def _generate_deductive_reasoning(self) -> float:
        """生成演绎推理能力"""
        return round(random.uniform(0.72, 0.92), 3)
    
    def _generate_inductive_reasoning(self) -> float:
        """生成归纳推理能力"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_evidence_evaluation(self) -> float:
        """生成证据评估能力"""
        return round(random.uniform(0.68, 0.88), 3)
    
    def _generate_bias_detection(self) -> float:
        """生成偏见检测能力"""
        return round(random.uniform(0.65, 0.85), 3)
    
    def _generate_fallacy_identification(self) -> float:
        """生成谬误识别能力"""
        return round(random.uniform(0.60, 0.82), 3)
    
    def _generate_symbolic_processing(self) -> float:
        """生成符号处理能力"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_pattern_recognition(self) -> float:
        """生成模式识别能力"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_analogical_reasoning(self) -> float:
        """生成类比推理能力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_holistic_thinking(self) -> float:
        """生成整体性思维"""
        return round(random.uniform(0.68, 0.87), 3)
    
    def _generate_cause_effect_modeling(self) -> float:
        """生成因果关系建模"""
        return round(random.uniform(0.70, 0.89), 3)
    
    def _generate_system_navigation(self) -> float:
        """生成系统导航能力"""
        return round(random.uniform(0.65, 0.86), 3)
    
    def _generate_idea_fluency(self) -> float:
        """生成想法流畅性"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_idea_flexibility(self) -> float:
        """生成想法灵活性"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_idea_originality(self) -> float:
        """生成想法原创性"""
        return round(random.uniform(0.55, 0.82), 3)
    
    def _calculate_thinking_depth(self, overall_score: float) -> float:
        """计算思维深度"""
        # 思维深度基于整体得分，但也考虑分析能力
        depth_base = overall_score / 100
        analytical_factor = 0.75  # 默认分析能力值
        depth = (depth_base * 0.7 + analytical_factor * 0.3) * 10  # 转换为0-10量表
        
        return round(depth, 2)
    
    def _generate_thinking_speed(self) -> float:
        """生成思维速度"""
        return round(random.uniform(2.0, 4.0), 2)  # 每分钟处理的思维任务数
    
    def _generate_thinking_flexibility(self) -> float:
        """生成思维灵活性"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_thinking_persistence(self) -> float:
        """生成思维坚持性"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_cognitive_load(self) -> float:
        """生成认知负荷"""
        return round(random.uniform(0.30, 0.80), 3)
    
    def _generate_mental_energy(self) -> float:
        """生成心理能量"""
        return round(random.uniform(0.50, 0.85), 3)
    
    def _generate_focus_level(self) -> float:
        """生成专注水平"""
        return round(random.uniform(0.60, 0.90), 3)
    
    def _generate_mental_fatigue(self) -> float:
        """生成心理疲劳"""
        return round(random.uniform(0.10, 0.65), 3)
    
    def _generate_thinking_response_time(self) -> float:
        """生成思维响应时间"""
        return round(random.uniform(0.8, 3.0), 2)
    
    def _generate_thinking_stability(self) -> float:
        """生成思维稳定性"""
        return round(random.uniform(0.75, 0.93), 3)
    
    def _calculate_thinking_efficiency(self, overall_score: float) -> float:
        """计算思维效率"""
        # 效率 = 准确率 * 速度 * 质量 / 100
        accuracy = 0.85  # 默认准确率
        speed = 2.5  # 默认速度
        quality = overall_score / 100
        
        # 标准化速度因子
        speed_factor = min(speed / 3.0, 1.1)
        
        efficiency = (accuracy * speed_factor * quality * 100)
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前思维力指标"""
        return self.current_thinking_state.copy()
    
    def update_thinking_metrics(self, **kwargs):
        """更新思维指标"""
        current_time = time.time()
        
        # 更新指定指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_thinking_state:
                self.current_thinking_state[metric_name] = value
        
        # 更新时间戳
        self.current_thinking_state["timestamp"] = current_time
        
        # 重新计算综合指标
        self._recalculate_composite_metrics()
        
        # 分析趋势
        self._analyze_thinking_trend()
        
        # 添加到历史记录
        self._add_to_history()
    
    def _recalculate_composite_metrics(self):
        """重新计算综合指标"""
        # 重新计算整体得分
        logical_score = self.current_thinking_state["logical_reasoning_speed"]
        critical_score = self.current_thinking_state["critical_analysis_accuracy"]
        abstract_score = self.current_thinking_state["abstract_concept_grasping"]
        systematic_score = self.current_thinking_state["system_analysis_depth"]
        creative_score = self.current_thinking_state["divergent_thinking"]
        
        overall_score = (
            logical_score * 0.25 +
            critical_score * 0.20 +
            abstract_score * 0.20 +
            systematic_score * 0.15 +
            creative_score * 0.20
        )
        
        self.current_thinking_state["overall_score"] = round(overall_score, 1)
        
        # 重新计算指标
        self.current_thinking_state["thinking_depth"] = self._calculate_thinking_depth(overall_score)
        self.current_thinking_state["efficiency"] = self._calculate_thinking_efficiency(overall_score)
    
    def _analyze_thinking_trend(self):
        """分析思维趋势"""
        if len(self.thinking_history) < 2:
            self.current_thinking_state["trend"] = "stable"
            return
        
        # 获取最近两次整体得分
        recent_scores = []
        for data_point in reversed(self.thinking_history[-10:]):
            if "overall_score" in data_point:
                recent_scores.append(data_point["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 2.5:
                self.current_thinking_state["trend"] = "rising"
            elif change < -2.5:
                self.current_thinking_state["trend"] = "declining"
            else:
                self.current_thinking_state["trend"] = "stable"
    
    def _add_to_history(self):
        """添加当前状态到历史记录"""
        history_point = self.current_thinking_state.copy()
        self.thinking_history.append(history_point)
        
        # 限制历史记录长度
        if len(self.thinking_history) > 100:
            self.thinking_history = self.thinking_history[-100:]
    
    def simulate_thinking_task(self, task_type: str = "random") -> Dict[str, Any]:
        """
        模拟思维任务执行
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        if task_type == "random":
            task_type = random.choice(self.thinking_tasks)
        
        # 根据任务类型调整性能
        task_performance_factors = {
            "逻辑推理题": 0.95,
            "数学证明题": 0.88,
            "概念分类任务": 0.92,
            "系统分析题": 0.85,
            "创意解决问题": 0.90,
            "批判性阅读": 0.87,
            "抽象模式识别": 0.93,
            "因果关系推理": 0.91,
            "归纳总结任务": 0.89,
            "演绎推理题": 0.94
        }
        
        base_performance = task_performance_factors.get(task_type, 1.0)
        
        # 生成任务结果
        accuracy = base_performance * random.uniform(0.70, 0.98)
        response_time = self.current_thinking_state["response_time"] * random.uniform(0.7, 1.4)
        complexity = random.choice(["简单", "中等", "复杂", "高难度"])
        
        task_result = {
            "task_type": task_type,
            "timestamp": time.time(),
            "accuracy": round(accuracy, 3),
            "response_time": round(response_time, 2),
            "complexity": complexity,
            "cognitive_load": random.uniform(0.4, 0.9),
            "thinking_depth": random.uniform(0.6, 0.95),
            "solution_quality": random.uniform(0.65, 0.95)
        }
        
        # 根据任务表现调整思维状态
        performance_impact = (accuracy - 0.80) * 8  # 性能影响因子
        fatigue_increase = random.uniform(0.03, 0.08)  # 疲劳增加
        
        self.update_thinking_metrics(
            overall_score=self.current_thinking_state["overall_score"] + performance_impact,
            mental_fatigue=self.current_thinking_state["mental_fatigue"] + fatigue_increase
        )
        
        return task_result
    
    def get_thinking_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取思维力分析报告
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            思维力分析报告
        """
        if not self.thinking_history:
            return {"error": "暂无历史数据"}
        
        # 过滤时间范围内的数据
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [d for d in self.thinking_history if d["timestamp"] >= cutoff_time]
        
        if not recent_data:
            return {"error": "指定时间范围内无数据"}
        
        # 计算统计数据
        scores = [d["overall_score"] for d in recent_data]
        depths = [d.get("thinking_depth", 5.0) for d in recent_data]
        speeds = [d.get("thinking_speed", 2.5) for d in recent_data]
        
        analysis = {
            "time_range": f"最近{hours}小时",
            "data_points": len(recent_data),
            
            "overall_performance": {
                "mean_score": round(np.mean(scores), 1),
                "std_score": round(np.std(scores), 1),
                "min_score": round(np.min(scores), 1),
                "max_score": round(np.max(scores), 1),
                "trend": self._calculate_trend(scores)
            },
            
            "thinking_characteristics": {
                "mean_depth": round(np.mean(depths), 2),
                "mean_speed": round(np.mean(speeds), 2),
                "depth_consistency": round(1 - np.std(depths)/np.mean(depths), 3),
                "speed_consistency": round(1 - np.std(speeds)/np.mean(speeds), 3)
            },
            
            "cognitive_state": {
                "average_cognitive_load": round(np.mean([d.get("cognitive_load", 0.5) for d in recent_data]), 3),
                "average_mental_energy": round(np.mean([d.get("mental_energy", 0.7) for d in recent_data]), 3),
                "average_fatigue": round(np.mean([d.get("mental_fatigue", 0.3) for d in recent_data]), 3)
            },
            
            "component_analysis": self._analyze_thinking_components(recent_data),
            
            "recommendations": self._generate_thinking_recommendations(recent_data)
        }
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """计算得分趋势"""
        if len(scores) < 2:
            return "stable"
        
        # 线性回归计算趋势
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.3:
            return "improving"
        elif slope < -0.3:
            return "declining"
        else:
            return "stable"
    
    def _analyze_thinking_components(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """分析思维组件性能"""
        component_scores = {
            "logical_reasoning": [],
            "critical_thinking": [],
            "abstract_thinking": [],
            "systematic_thinking": [],
            "creative_thinking": []
        }
        
        for data_point in recent_data:
            component_scores["logical_reasoning"].append(data_point.get("logical_reasoning_speed", 75))
            component_scores["critical_thinking"].append(data_point.get("critical_analysis_accuracy", 75))
            component_scores["abstract_thinking"].append(data_point.get("abstract_concept_grasping", 75))
            component_scores["systematic_thinking"].append(data_point.get("system_analysis_depth", 75))
            component_scores["creative_thinking"].append(data_point.get("divergent_thinking", 75))
        
        component_analysis = {}
        for component, scores in component_scores.items():
            if scores:
                component_analysis[component] = {
                    "mean_score": round(np.mean(scores), 1),
                    "stability": round(1 - np.std(scores)/np.mean(scores), 3),
                    "trend": self._calculate_trend(scores)
                }
        
        return component_analysis
    
    def _generate_thinking_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """生成思维改善建议"""
        recommendations = []
        
        # 分析当前状态
        avg_fatigue = np.mean([d.get("mental_fatigue", 0.3) for d in recent_data])
        avg_energy = np.mean([d.get("mental_energy", 0.7) for d in recent_data])
        avg_load = np.mean([d.get("cognitive_load", 0.5) for d in recent_data])
        
        if avg_fatigue > 0.5:
            recommendations.append("思维疲劳较高，建议适当休息或进行放松活动")
        
        if avg_energy < 0.6:
            recommendations.append("心理能量偏低，建议补充营养或进行轻度运动")
        
        if avg_load > 0.7:
            recommendations.append("认知负荷过重，建议简化任务或分步骤处理")
        
        # 基于时间提供建议
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 11:
            recommendations.append("当前为思维活跃期，适合进行复杂推理任务")
        elif 14 <= current_hour <= 16:
            recommendations.append("下午思维稳定期，适合进行系统性分析工作")
        
        # 基于组件性能提供建议
        current_state = self.current_thinking_state
        
        if current_state["logical_reasoning_speed"] < 70:
            recommendations.append("逻辑推理能力有待提升，建议练习数学和逻辑题")
        
        if current_state["abstract_concept_grasping"] < 75:
            recommendations.append("抽象思维需要加强，建议阅读哲学或理论文章")
        
        if current_state["creative_thinking"] < 70:
            recommendations.append("创造性思维可以培养，建议进行头脑风暴活动")
        
        return recommendations if recommendations else ["当前思维状态良好，继续保持"]
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择要更新的指标
        updateable_metrics = [
            "logical_reasoning_speed",
            "critical_analysis_accuracy", 
            "abstract_concept_grasping",
            "system_analysis_depth",
            "divergent_thinking"
        ]
        
        metric_to_update = random.choice(updateable_metrics)
        
        # 生成新的值
        if metric_to_update == "logical_reasoning_speed":
            new_value = self._generate_logical_reasoning_score()
        elif metric_to_update == "critical_analysis_accuracy":
            new_value = self._generate_critical_thinking_score()
        elif metric_to_update == "abstract_concept_grasping":
            new_value = self._generate_abstract_thinking_score()
        elif metric_to_update == "system_analysis_depth":
            new_value = self._generate_systematic_thinking_score()
        elif metric_to_update == "divergent_thinking":
            new_value = self._generate_creative_thinking_score()
        else:
            new_value = self.current_thinking_state.get(metric_to_update, 75)
        
        # 更新指标
        self.update_thinking_metrics(**{metric_to_update: new_value})
        
        return metric_to_update


# 测试函数
def test_thinking_monitor():
    """测试思维力监控器"""
    print("开始测试思维力监控器...")
    
    # 创建思维力监控器
    thinking_monitor = ThinkingMonitor()
    
    # 测试当前指标
    print("\n1. 当前思维指标:")
    current_metrics = thinking_monitor.get_current_metrics()
    print(f"  整体得分: {current_metrics['overall_score']:.1f}%")
    print(f"  思维深度: {current_metrics['thinking_depth']:.2f}")
    print(f"  逻辑推理: {current_metrics['logical_reasoning_speed']:.1f}")
    print(f"  响应时间: {current_metrics['response_time']:.2f}s")
    
    # 模拟思维任务
    print("\n2. 模拟思维任务:")
    for i in range(3):
        result = thinking_monitor.simulate_thinking_task()
        print(f"  任务{i+1}: {result['task_type']} - 准确率: {result['accuracy']:.1%}")
    
    # 模拟实时更新
    print("\n3. 模拟实时更新 (5次):")
    for i in range(5):
        metric = thinking_monitor.simulate_real_time_update()
        print(f"  更新{i+1}: {metric}")
    
    # 生成分析报告
    print("\n4. 思维力分析:")
    analysis = thinking_monitor.get_thinking_analysis(hours=1)
    if "error" not in analysis:
        print(f"  数据点数量: {analysis['data_points']}")
        print(f"  平均得分: {analysis['overall_performance']['mean_score']:.1f}")
        print(f"  思维深度: {analysis['thinking_characteristics']['mean_depth']:.2f}")
        print(f"  主要建议: {analysis['recommendations'][0]}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_thinking_monitor()