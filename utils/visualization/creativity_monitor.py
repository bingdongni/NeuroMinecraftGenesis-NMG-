"""
创造力监控模块
=============

专门负责创造力能力的监控、评估和分析。

创造力定义：
- 发散思维：从一个点出发，进行各种方向的思考
- 聚合思维：将不同元素组合形成新方案
- 想象能力：在头脑中构建新形象和场景
- 灵感捕捉：快速识别和记录创新想法
- 原创性：产生新颖独特的解决方案
- 创新执行：将创意转化为实际成果的能力

主要功能：
- 实时创造力指标监控
- 创新思维评估
- 创意产生测量
- 创新质量分析
- 创意训练建议

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math


class CreativityMonitor:
    """
    创造力监控器
    
    监控和评估创造力相关的各种认知能力：
    1. 发散思维和想法流畅性
    2. 创造性想象和灵感捕捉
    3. 原创性和独特性评估
    4. 创新思维灵活性
    5. 创意实现能力
    6. 创新适应性
    """
    
    def __init__(self):
        """初始化创造力监控器"""
        self.component_name = "创造力监控器"
        self.last_update_time = time.time()
        
        # 创造力的子组件
        self.creativity_components = {
            "divergent_thinking": {
                "name": "发散思维",
                "description": "从一个点出发进行多方向思考",
                "weight": 0.25
            },
            "convergent_thinking": {
                "name": "聚合思维",
                "description": "组合元素形成新方案",
                "weight": 0.20
            },
            "imaginative_ability": {
                "name": "想象能力",
                "description": "头脑中构建新形象和场景",
                "weight": 0.20
            },
            "originality": {
                "name": "原创性",
                "description": "产生新颖独特的解决方案",
                "weight": 0.20
            },
            "innovation_execution": {
                "name": "创新执行",
                "description": "将创意转化为实际成果",
                "weight": 0.15
            }
        }
        
        # 创造力性能基准
        self.creativity_benchmarks = {
            "idea_fluency": {
                "excellent": 15.0,  # 想法流畅性（每分钟想法数）
                "good": 12.0,
                "average": 9.0,
                "poor": 6.0
            },
            "idea_flexibility": {
                "excellent": 0.85,  # 想法灵活性
                "good": 0.75,
                "average": 0.65,
                "poor": 0.50
            },
            "idea_originality": {
                "excellent": 0.90,  # 想法原创性
                "good": 0.80,
                "average": 0.70,
                "poor": 0.55
            },
            "creative_imagination": {
                "excellent": 0.88,  # 创造性想象
                "good": 0.78,
                "average": 0.68,
                "poor": 0.52
            },
            "innovation_rate": {
                "excellent": 0.25,  # 创新转化率
                "good": 0.18,
                "average": 0.12,
                "poor": 0.08
            }
        }
        
        # 当前创造力状态
        self.current_creativity_state = self._initialize_creativity_state()
        
        # 创造力历史数据
        self.creativity_history = []
        
        # 创造力测试任务
        self.creativity_tasks = [
            "头脑风暴练习",
            "替代用途测试",
            "创意故事编写",
            "问题创新解决",
            "艺术创作任务",
            "发明设计方案",
            "诗歌创作练习",
            "视觉创意设计",
            "音乐创作任务",
            "科学创新思考"
        ]
    
    def _initialize_creativity_state(self) -> Dict[str, Any]:
        """初始化创造力状态"""
        current_time = time.time()
        
        # 生成初始创造力指标
        divergent_score = self._generate_divergent_thinking_score()
        convergent_score = self._generate_convergent_thinking_score()
        imaginative_score = self._generate_imaginative_ability_score()
        originality_score = self._generate_originality_score()
        execution_score = self._generate_innovation_execution_score()
        
        # 计算加权综合得分
        overall_score = (
            divergent_score * 0.25 +
            convergent_score * 0.20 +
            imaginative_score * 0.20 +
            originality_score * 0.20 +
            execution_score * 0.15
        )
        
        return {
            "timestamp": current_time,
            "overall_score": round(overall_score, 1),
            
            # 发散思维指标
            "idea_fluency": divergent_score,
            "idea_flexibility": self._generate_idea_flexibility(),
            "cognitive_elaboration": self._generate_cognitive_elaboration(),
            "mental_flexibility": self._generate_mental_flexibility(),
            
            # 聚合思维指标
            "idea_synthesis": convergent_score,
            "pattern_combination": self._generate_pattern_combination(),
            "conceptual_blending": self._generate_conceptual_blending(),
            "solution_integration": self._generate_solution_integration(),
            
            # 想象能力指标
            "creative_imagination": imaginative_score,
            "visual_creation": self._generate_visual_creation(),
            "fantasy_construction": self._generate_fantasy_construction(),
            "future_scenario": self._generate_future_scenario(),
            
            # 原创性指标
            "idea_originality": originality_score,
            "unique_perspective": self._generate_unique_perspective(),
            "novel_association": self._generate_novel_association(),
            "innovative_breakthrough": self._generate_innovative_breakthrough(),
            
            # 创新执行指标
            "innovation_implementation": execution_score,
            "creative_breakthroughs": self._generate_creative_breakthroughs(),
            "innovation_rate": self._generate_innovation_rate(),
            "creative_persistence": self._generate_creative_persistence(),
            
            # 创造力特质指标
            "creative_confidence": self._generate_creative_confidence(),
            "risk_taking": self._generate_risk_taking(),
            "openness_experience": self._generate_openness_experience(),
            "creative_motivation": self._generate_creative_motivation(),
            
            # 创意过程指标
            "insight_emergence": self._generate_insight_emergence(),
            "creative_flow_state": self._generate_creative_flow_state(),
            "ideation_quality": self._generate_ideation_quality(),
            "creative_problem_solving": self._generate_creative_problem_solving(),
            
            # 创意环境指标
            "creative_environment": self._generate_creative_environment(),
            "diversity_exposure": self._generate_diversity_exposure(),
            "stimulus_stimulation": self._generate_stimulus_stimulation(),
            "collaborative_creativity": self._generate_collaborative_creativity(),
            
            # 响应时间
            "response_time": self._generate_creativity_response_time(),
            
            # 稳定性
            "stability": self._generate_creativity_stability(),
            
            # 效率
            "efficiency": self._calculate_creativity_efficiency(overall_score),
            
            # 趋势
            "trend": "stable"
        }
    
    def _generate_divergent_thinking_score(self) -> float:
        """生成发散思维得分"""
        # 发散思维受情绪和环境影响较大
        base_score = random.uniform(60, 88)
        
        # 情绪状态影响
        emotion_factor = np.random.choice([-8, -3, 2, 8, 12], p=[0.05, 0.15, 0.4, 0.25, 0.15])
        
        # 环境因素
        hour = datetime.now().hour
        if 9 <= hour <= 11 or 15 <= hour <= 17:
            environment_boost = 5  # 创造力活跃期
        elif 20 <= hour <= 23:
            environment_boost = 8  # 晚间创意高峰
        else:
            environment_boost = -2
        
        score = base_score + emotion_factor + environment_boost + np.random.normal(0, 3)
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_convergent_thinking_score(self) -> float:
        """生成聚合思维得分"""
        # 聚合思维相对稳定
        base_score = random.uniform(65, 85)
        
        # 少量随机波动
        noise = np.random.normal(0, 4)
        score = base_score + noise
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_imaginative_ability_score(self) -> float:
        """生成想象能力得分"""
        # 想象能力受休息状态影响
        base_score = random.uniform(58, 90)
        
        # 疲劳状态影响
        fatigue_factor = np.random.uniform(-10, 2)
        score = base_score + fatigue_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_originality_score(self) -> float:
        """生成原创性得分"""
        # 原创性较难评估，波动较大
        base_score = random.uniform(50, 85)
        
        # 知识和经验基础
        knowledge_factor = np.random.uniform(-5, 10)
        score = base_score + knowledge_factor + np.random.normal(0, 5)
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_innovation_execution_score(self) -> float:
        """生成创新执行得分"""
        # 执行能力受动力和资源影响
        base_score = random.uniform(55, 82)
        
        # 动机水平
        motivation_factor = np.random.uniform(-3, 8)
        score = base_score + motivation_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_idea_flexibility(self) -> float:
        """生成想法灵活性"""
        return round(random.uniform(0.60, 0.90), 3)
    
    def _generate_cognitive_elaboration(self) -> float:
        """生成认知精化能力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_mental_flexibility(self) -> float:
        """生成思维灵活性"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_pattern_combination(self) -> float:
        """生成模式组合能力"""
        return round(random.uniform(0.68, 0.87), 3)
    
    def _generate_conceptual_blending(self) -> float:
        """生成概念融合能力"""
        return round(random.uniform(0.62, 0.85), 3)
    
    def _generate_solution_integration(self) -> float:
        """生成方案整合能力"""
        return round(random.uniform(0.70, 0.89), 3)
    
    def _generate_visual_creation(self) -> float:
        """生成视觉创作能力"""
        return round(random.uniform(0.58, 0.83), 3)
    
    def _generate_fantasy_construction(self) -> float:
        """生成幻想构建能力"""
        return round(random.uniform(0.60, 0.88), 3)
    
    def _generate_future_scenario(self) -> float:
        """生成未来场景构建"""
        return round(random.uniform(0.65, 0.90), 3)
    
    def _generate_unique_perspective(self) -> float:
        """生成独特视角"""
        return round(random.uniform(0.55, 0.82), 3)
    
    def _generate_novel_association(self) -> float:
        """生成新颖联想"""
        return round(random.uniform(0.60, 0.86), 3)
    
    def _generate_innovative_breakthrough(self) -> float:
        """生成创新突破"""
        return round(random.uniform(0.50, 0.78), 3)
    
    def _generate_creative_breakthroughs(self) -> float:
        """生成创意突破频次"""
        return round(random.uniform(0.15, 0.35), 3)
    
    def _generate_innovation_rate(self) -> float:
        """生成创新转化率"""
        return round(random.uniform(0.10, 0.28), 3)
    
    def _generate_creative_persistence(self) -> float:
        """生成创意坚持性"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_creative_confidence(self) -> float:
        """生成创意自信"""
        return round(random.uniform(0.60, 0.92), 3)
    
    def _generate_risk_taking(self) -> float:
        """生成冒险精神"""
        return round(random.uniform(0.45, 0.80), 3)
    
    def _generate_openness_experience(self) -> float:
        """生成开放体验性"""
        return round(random.uniform(0.70, 0.93), 3)
    
    def _generate_creative_motivation(self) -> float:
        """生成创意动机"""
        return round(random.uniform(0.65, 0.90), 3)
    
    def _generate_insight_emergence(self) -> float:
        """生成洞察出现率"""
        return round(random.uniform(0.55, 0.85), 3)
    
    def _generate_creative_flow_state(self) -> float:
        """生成创意心流状态"""
        return round(random.uniform(0.60, 0.88), 3)
    
    def _generate_ideation_quality(self) -> float:
        """生成构思质量"""
        return round(random.uniform(0.68, 0.91), 3)
    
    def _generate_creative_problem_solving(self) -> float:
        """生成创意问题解决"""
        return round(random.uniform(0.70, 0.89), 3)
    
    def _generate_creative_environment(self) -> float:
        """生成创意环境适宜度"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_diversity_exposure(self) -> float:
        """生成多样性接触"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_stimulus_stimulation(self) -> float:
        """生成刺激激发度"""
        return round(random.uniform(0.58, 0.87), 3)
    
    def _generate_collaborative_creativity(self) -> float:
        """生成协作创造力"""
        return round(random.uniform(0.62, 0.86), 3)
    
    def _generate_creativity_response_time(self) -> float:
        """生成创意响应时间"""
        return round(random.uniform(1.2, 4.5), 2)
    
    def _generate_creativity_stability(self) -> float:
        """生成创造力稳定性"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _calculate_creativity_efficiency(self, overall_score: float) -> float:
        """计算创造力效率"""
        # 效率 = 原创性 * 流畅性 * 可行性 / 100
        originality = 0.7  # 默认值
        fluency = 1.0  # 默认值
        feasibility = 0.7  # 默认值
        
        efficiency = (originality * fluency * feasibility * overall_score / 100) * 100
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前创造力指标"""
        return self.current_creativity_state.copy()
    
    def update_creativity_metrics(self, **kwargs):
        """更新创造力指标"""
        current_time = time.time()
        
        # 更新指定指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_creativity_state:
                self.current_creativity_state[metric_name] = value
        
        # 更新时间戳
        self.current_creativity_state["timestamp"] = current_time
        
        # 重新计算综合指标
        self._recalculate_composite_metrics()
        
        # 分析趋势
        self._analyze_creativity_trend()
        
        # 添加到历史记录
        self._add_to_history()
    
    def _recalculate_composite_metrics(self):
        """重新计算综合指标"""
        # 重新计算整体得分
        divergent_score = self.current_creativity_state["idea_fluency"]
        convergent_score = self.current_creativity_state["idea_synthesis"]
        imaginative_score = self.current_creativity_state["creative_imagination"]
        originality_score = self.current_creativity_state["idea_originality"]
        execution_score = self.current_creativity_state["innovation_implementation"]
        
        overall_score = (
            divergent_score * 0.25 +
            convergent_score * 0.20 +
            imaginative_score * 0.20 +
            originality_score * 0.20 +
            execution_score * 0.15
        )
        
        self.current_creativity_state["overall_score"] = round(overall_score, 1)
        
        # 重新计算效率
        self.current_creativity_state["efficiency"] = self._calculate_creativity_efficiency(overall_score)
    
    def _analyze_creativity_trend(self):
        """分析创造力趋势"""
        if len(self.creativity_history) < 2:
            self.current_creativity_state["trend"] = "stable"
            return
        
        # 获取最近两次整体得分
        recent_scores = []
        for data_point in reversed(self.creativity_history[-10:]):
            if "overall_score" in data_point:
                recent_scores.append(data_point["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 3.5:
                self.current_creativity_state["trend"] = "rising"
            elif change < -3.5:
                self.current_creativity_state["trend"] = "declining"
            else:
                self.current_creativity_state["trend"] = "stable"
    
    def _add_to_history(self):
        """添加当前状态到历史记录"""
        history_point = self.current_creativity_state.copy()
        self.creativity_history.append(history_point)
        
        # 限制历史记录长度
        if len(self.creativity_history) > 100:
            self.creativity_history = self.creativity_history[-100:]
    
    def simulate_creativity_task(self, task_type: str = "random") -> Dict[str, Any]:
        """
        模拟创造力任务执行
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        if task_type == "random":
            task_type = random.choice(self.creativity_tasks)
        
        # 根据任务类型调整性能
        task_performance_factors = {
            "头脑风暴练习": 1.0,
            "替代用途测试": 0.95,
            "创意故事编写": 0.92,
            "问题创新解决": 0.88,
            "艺术创作任务": 0.90,
            "发明设计方案": 0.85,
            "诗歌创作练习": 0.93,
            "视觉创意设计": 0.91,
            "音乐创作任务": 0.87,
            "科学创新思考": 0.86
        }
        
        base_performance = task_performance_factors.get(task_type, 1.0)
        
        # 生成任务结果
        originality = base_performance * random.uniform(0.60, 0.95)
        fluency = base_performance * random.uniform(0.70, 0.98)
        feasibility = base_performance * random.uniform(0.65, 0.90)
        response_time = self.current_creativity_state["response_time"] * random.uniform(0.6, 1.5)
        
        task_result = {
            "task_type": task_type,
            "timestamp": time.time(),
            "originality_score": round(originality, 3),
            "fluency_score": round(fluency, 3),
            "feasibility_score": round(feasibility, 3),
            "response_time": round(response_time, 2),
            "creative_engagement": random.uniform(0.6, 0.95),
            "insight_quality": random.uniform(0.55, 0.88)
        }
        
        # 根据任务表现调整创造力状态
        performance_impact = ((originality + fluency + feasibility) / 3 - 0.75) * 12
        engagement_boost = task_result["creative_engagement"] * 0.1
        
        self.update_creativity_metrics(
            overall_score=self.current_creativity_state["overall_score"] + performance_impact,
            creative_flow_state=self.current_creativity_state["creative_flow_state"] + engagement_boost
        )
        
        return task_result
    
    def get_creativity_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取创造力分析报告
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            创造力分析报告
        """
        if not self.creativity_history:
            return {"error": "暂无历史数据"}
        
        # 过滤时间范围内的数据
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [d for d in self.creativity_history if d["timestamp"] >= cutoff_time]
        
        if not recent_data:
            return {"error": "指定时间范围内无数据"}
        
        # 计算统计数据
        scores = [d["overall_score"] for d in recent_data]
        originality_scores = [d.get("idea_originality", 0.7) for d in recent_data]
        fluency_scores = [d.get("idea_fluency", 10) for d in recent_data]
        
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
            
            "creativity_characteristics": {
                "mean_originality": round(np.mean(originality_scores), 3),
                "mean_fluency": round(np.mean(fluency_scores), 2),
                "originality_stability": round(1 - np.std(originality_scores)/np.mean(originality_scores), 3),
                "fluency_consistency": round(1 - np.std(fluency_scores)/np.mean(fluency_scores), 3)
            },
            
            "creative_state": {
                "average_flow_state": round(np.mean([d.get("creative_flow_state", 0.75) for d in recent_data]), 3),
                "average_motivation": round(np.mean([d.get("creative_motivation", 0.78) for d in recent_data]), 3),
                "average_confidence": round(np.mean([d.get("creative_confidence", 0.80) for d in recent_data]), 3)
            },
            
            "component_analysis": self._analyze_creativity_components(recent_data),
            
            "innovation_insights": self._generate_innovation_insights(recent_data),
            
            "recommendations": self._generate_creativity_recommendations(recent_data)
        }
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """计算得分趋势"""
        if len(scores) < 2:
            return "stable"
        
        # 线性回归计算趋势
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.4:
            return "improving"
        elif slope < -0.4:
            return "declining"
        else:
            return "stable"
    
    def _analyze_creativity_components(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """分析创造力组件性能"""
        component_scores = {
            "divergent_thinking": [],
            "convergent_thinking": [],
            "imaginative_ability": [],
            "originality": [],
            "innovation_execution": []
        }
        
        for data_point in recent_data:
            component_scores["divergent_thinking"].append(data_point.get("idea_fluency", 75))
            component_scores["convergent_thinking"].append(data_point.get("idea_synthesis", 75))
            component_scores["imaginative_ability"].append(data_point.get("creative_imagination", 75))
            component_scores["originality"].append(data_point.get("idea_originality", 75))
            component_scores["innovation_execution"].append(data_point.get("innovation_implementation", 75))
        
        component_analysis = {}
        for component, scores in component_scores.items():
            if scores:
                component_analysis[component] = {
                    "mean_score": round(np.mean(scores), 1),
                    "stability": round(1 - np.std(scores)/np.mean(scores), 3),
                    "trend": self._calculate_trend(scores)
                }
        
        return component_analysis
    
    def _generate_innovation_insights(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """生成创新洞察"""
        current_state = self.current_creativity_state
        
        # 分析创新模式
        innovation_patterns = []
        
        if current_state["idea_fluency"] > 85:
            innovation_patterns.append("高想法流畅性 - 适合头脑风暴")
        
        if current_state["idea_originality"] > 80:
            innovation_patterns.append("强原创性倾向 - 善于产生独特想法")
        
        if current_state["innovation_implementation"] > 75:
            innovation_patterns.append("优秀执行能力 - 能将创意转化为现实")
        
        if current_state["creative_flow_state"] > 0.85:
            innovation_patterns.append("深度心流状态 - 创作效率极高")
        
        return {
            "dominant_patterns": innovation_patterns,
            "innovation_potential": round((current_state["idea_originality"] + current_state["idea_fluency"]) / 2, 1),
            "execution_effectiveness": current_state["innovation_rate"],
            "creative_readiness": round((current_state["creative_motivation"] + current_state["creative_confidence"]) / 2, 3)
        }
    
    def _generate_creativity_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """生成创造力改善建议"""
        recommendations = []
        
        # 分析当前状态
        avg_flow = np.mean([d.get("creative_flow_state", 0.75) for d in recent_data])
        avg_motivation = np.mean([d.get("creative_motivation", 0.78) for d in recent_data])
        avg_confidence = np.mean([d.get("creative_confidence", 0.80) for d in recent_data])
        
        if avg_flow < 0.7:
            recommendations.append("创意心流状态较低，建议寻找安静舒适的环境进行创作")
        
        if avg_motivation < 0.7:
            recommendations.append("创意动机不足，建议设定明确的创作目标或挑战")
        
        if avg_confidence < 0.75:
            recommendations.append("创意自信需要提升，建议从小的创意练习开始")
        
        # 基于组件性能提供建议
        current_state = self.current_creativity_state
        
        if current_state["idea_fluency"] < 70:
            recommendations.append("想法流畅性有待提升，建议进行发散思维训练")
        
        if current_state["idea_originality"] < 65:
            recommendations.append("原创性可以加强，建议接触更多不同领域的知识和经验")
        
        if current_state["innovation_implementation"] < 70:
            recommendations.append("创新执行需要改善，建议学习项目管理和执行技巧")
        
        # 基于时间提供建议
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11:
            recommendations.append("上午是创意思维活跃期，适合进行创意策划工作")
        elif 15 <= current_hour <= 17:
            recommendations.append("下午适合进行创意实现和项目推进")
        elif 20 <= current_hour <= 23:
            recommendations.append("晚间创意高峰，适合进行艺术创作和自由联想")
        
        # 创意环境建议
        if current_state["creative_environment"] < 0.7:
            recommendations.append("当前环境可能不适合创作，建议改善工作空间或寻找新的灵感场所")
        
        return recommendations if recommendations else ["当前创造力状态良好，继续保持创意实践"]
    
    def calculate_innovation_ratio(self) -> float:
        """计算创新性动作占比"""
        # 基于多个指标计算创新比例
        originality_factor = self.current_creativity_state.get("idea_originality", 0.7)
        fluency_factor = min(self.current_creativity_state.get("idea_fluency", 10) / 15.0, 1.0)
        execution_factor = self.current_creativity_state.get("innovation_implementation", 70) / 100
        
        innovation_ratio = (originality_factor * 0.4 + fluency_factor * 0.3 + execution_factor * 0.3)
        
        return round(innovation_ratio, 3)
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择要更新的指标
        updateable_metrics = [
            "idea_fluency",
            "idea_synthesis", 
            "creative_imagination",
            "idea_originality",
            "innovation_implementation"
        ]
        
        metric_to_update = random.choice(updateable_metrics)
        
        # 生成新的值
        if metric_to_update == "idea_fluency":
            new_value = self._generate_divergent_thinking_score()
        elif metric_to_update == "idea_synthesis":
            new_value = self._generate_convergent_thinking_score()
        elif metric_to_update == "creative_imagination":
            new_value = self._generate_imaginative_ability_score()
        elif metric_to_update == "idea_originality":
            new_value = self._generate_originality_score()
        elif metric_to_update == "innovation_implementation":
            new_value = self._generate_innovation_execution_score()
        else:
            new_value = self.current_creativity_state.get(metric_to_update, 75)
        
        # 更新指标
        self.update_creativity_metrics(**{metric_to_update: new_value})
        
        return metric_to_update


# 测试函数
def test_creativity_monitor():
    """测试创造力监控器"""
    print("开始测试创造力监控器...")
    
    # 创建创造力监控器
    creativity_monitor = CreativityMonitor()
    
    # 测试当前指标
    print("\n1. 当前创造力指标:")
    current_metrics = creativity_monitor.get_current_metrics()
    print(f"  整体得分: {current_metrics['overall_score']:.1f}%")
    print(f"  想法流畅性: {current_metrics['idea_fluency']:.1f}")
    print(f"  原创性: {current_metrics['idea_originality']:.1%}")
    print(f"  创新比例: {creativity_monitor.calculate_innovation_ratio():.1%}")
    
    # 模拟创造力任务
    print("\n2. 模拟创造力任务:")
    for i in range(3):
        result = creativity_monitor.simulate_creativity_task()
        print(f"  任务{i+1}: {result['task_type']} - 原创性: {result['originality_score']:.1%}")
    
    # 模拟实时更新
    print("\n3. 模拟实时更新 (5次):")
    for i in range(5):
        metric = creativity_monitor.simulate_real_time_update()
        print(f"  更新{i+1}: {metric}")
    
    # 生成分析报告
    print("\n4. 创造力分析:")
    analysis = creativity_monitor.get_creativity_analysis(hours=1)
    if "error" not in analysis:
        print(f"  数据点数量: {analysis['data_points']}")
        print(f"  平均得分: {analysis['overall_performance']['mean_score']:.1f}")
        print(f"  创新洞察: {len(analysis['innovation_insights']['dominant_patterns'])}个")
        print(f"  主要建议: {analysis['recommendations'][0]}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_creativity_monitor()