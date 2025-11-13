"""
想象力监控模块
=============

专门负责想象力能力的监控、评估和分析。

想象力定义：
- 创造性想象：在头脑中创造新形象和场景
- 情景模拟：在心中构建和演练未来场景
- 概念构建：形成抽象概念和理论框架
- 象征思维：理解和使用象征性表达
- 虚拟体验：在想象中体验不同情况
- 未来预测：预想和规划未来可能性

主要功能：
- 实时想象力指标监控
- 创意想象评估
- 情景构建能力测量
- 想象灵活性分析
- 想象力训练建议

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math


class ImaginationMonitor:
    """
    想象力监控器
    
    监控和评估想象力相关的各种认知能力：
    1. 创造性想象和图像生成能力
    2. 情景模拟和预测构建能力
    3. 概念抽象和理论构建能力
    4. 象征思维和隐喻理解能力
    5. 虚拟体验和角色扮演能力
    6. 未来想象和可能性探索
    """
    
    def __init__(self):
        """初始化想象力监控器"""
        self.component_name = "想象力监控器"
        self.last_update_time = time.time()
        
        # 想象力的子组件
        self.imagination_components = {
            "creative_imagination": {
                "name": "创造性想象",
                "description": "在头脑中创造新形象和场景",
                "weight": 0.25
            },
            "scenario_construction": {
                "name": "情景构建",
                "description": "在心中构建和演练未来场景",
                "weight": 0.20
            },
            "conceptual_building": {
                "name": "概念构建",
                "description": "形成抽象概念和理论框架",
                "weight": 0.20
            },
            "symbolic_thinking": {
                "name": "象征思维",
                "description": "理解和使用象征性表达",
                "weight": 0.15
            },
            "virtual_experience": {
                "name": "虚拟体验",
                "description": "在想象中体验不同情况",
                "weight": 0.20
            }
        }
        
        # 想象力性能基准
        self.imagination_benchmarks = {
            "imaginative_vividness": {
                "excellent": 0.90,  # 想象生动性
                "good": 0.82,
                "average": 0.74,
                "poor": 0.60
            },
            "scenario_depth": {
                "excellent": 0.88,  # 情景深度
                "good": 0.78,
                "average": 0.68,
                "poor": 0.52
            },
            "conceptual_flexibility": {
                "excellent": 0.85,  # 概念灵活性
                "good": 0.75,
                "average": 0.65,
                "poor": 0.50
            },
            "symbolic_proficiency": {
                "excellent": 0.82,  # 象征熟练度
                "good": 0.72,
                "average": 0.62,
                "poor": 0.48
            },
            "virtual_realism": {
                "excellent": 0.86,  # 虚拟现实感
                "good": 0.76,
                "average": 0.66,
                "poor": 0.52
            }
        }
        
        # 当前想象力状态
        self.current_imagination_state = self._initialize_imagination_state()
        
        # 想象力历史数据
        self.imagination_history = []
        
        # 想象力测试任务
        self.imagination_tasks = [
            "创造性图像生成",
            "未来情景预测",
            "抽象概念构建",
            "象征意义理解",
            "虚拟体验模拟",
            "故事情节编织",
            "空间想象任务",
            "角色扮演练习",
            "隐喻创造训练",
            "可能性探索练习"
        ]
    
    def _initialize_imagination_state(self) -> Dict[str, Any]:
        """初始化想象力状态"""
        current_time = time.time()
        
        # 生成初始想象力指标
        creative_score = self._generate_creative_imagination_score()
        scenario_score = self._generate_scenario_construction_score()
        conceptual_score = self._generate_conceptual_building_score()
        symbolic_score = self._generate_symbolic_thinking_score()
        virtual_score = self._generate_virtual_experience_score()
        
        # 计算加权综合得分
        overall_score = (
            creative_score * 0.25 +
            scenario_score * 0.20 +
            conceptual_score * 0.20 +
            symbolic_score * 0.15 +
            virtual_score * 0.20
        )
        
        return {
            "timestamp": current_time,
            "overall_score": round(overall_score, 1),
            
            # 创造性想象指标
            "imaginative_vividness": creative_score,
            "image_generation_speed": self._generate_image_generation_speed(),
            "visual_creativity": self._generate_visual_creativity(),
            "fantasy_construction": self._generate_fantasy_construction(),
            "mental_image_richness": self._generate_mental_image_richness(),
            
            # 情景构建指标
            "scenario_depth": scenario_score,
            "future_prediction_accuracy": self._generate_future_prediction_accuracy(),
            "situation_modeling": self._generate_situation_modeling(),
            "temporal_simulation": self._generate_temporal_simulation(),
            "contingency_planning": self._generate_contingency_planning(),
            
            # 概念构建指标
            "conceptual_flexibility": conceptual_score,
            "abstract_reasoning": self._generate_abstract_reasoning(),
            "theoretical_building": self._generate_theoretical_building(),
            "knowledge_synthesis": self._generate_knowledge_synthesis(),
            "conceptual_integration": self._generate_conceptual_integration(),
            
            # 象征思维指标
            "symbolic_proficiency": symbolic_score,
            "metaphor_understanding": self._generate_metaphor_understanding(),
            "symbol_interpretation": self._generate_symbol_interpretation(),
            "allegory_comprehension": self._generate_allegory_comprehension(),
            "symbolic_creation": self._generate_symbolic_creation(),
            
            # 虚拟体验指标
            "virtual_realism": virtual_score,
            "mental_time_travel": self._generate_mental_time_travel(),
            "perspective_taking": self._generate_perspective_taking(),
            "empathetic_simulation": self._generate_empathetic_simulation(),
            "immersive_experience": self._generate_immersive_experience(),
            
            # 想象力特质指标
            "imaginative_confidence": self._generate_imaginative_confidence(),
            "creative_courage": self._generate_creative_courage(),
            "fantasy_tolerance": self._generate_fantasy_tolerance(),
            "curiosity_level": self._generate_curiosity_level(),
            
            # 想象力过程指标
            "imagination_flow": self._generate_imagination_flow(),
            "creative_incubation": self._generate_creative_incubation(),
            "insight_emergence_imagination": self._generate_insight_emergence_imagination(),
            "imagination_synthesis": self._generate_imagination_synthesis(),
            
            # 想象力环境指标
            "imagination_environment": self._generate_imagination_environment(),
            "creative_stimulation": self._generate_creative_stimulation(),
            "imaginative_support": self._generate_imaginative_support(),
            "inspiration_access": self._generate_inspiration_access(),
            
            # 响应时间
            "response_time": self._generate_imagination_response_time(),
            
            # 稳定性
            "stability": self._generate_imagination_stability(),
            
            # 效率
            "efficiency": self._calculate_imagination_efficiency(overall_score),
            
            # 趋势
            "trend": "stable"
        }
    
    def _generate_creative_imagination_score(self) -> float:
        """生成创造性想象得分"""
        # 创造性想象受情绪和灵感影响
        base_score = random.uniform(62, 90)
        
        # 灵感波动较大
        inspiration_factor = np.random.choice([-10, -5, 3, 8, 15], p=[0.05, 0.15, 0.3, 0.35, 0.15])
        
        # 时间因素（晚间想象力较活跃）
        hour = datetime.now().hour
        if 20 <= hour <= 23:
            time_boost = 8  # 晚间想象力高峰
        elif 9 <= hour <= 11:
            time_boost = 2  # 上午一般
        else:
            time_boost = -3  # 其他时间较低
        
        score = base_score + inspiration_factor + time_boost + np.random.normal(0, 3)
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_scenario_construction_score(self) -> float:
        """生成情景构建得分"""
        # 情景构建需要逻辑和经验
        base_score = random.uniform(65, 85)
        
        # 经验和逻辑因素
        experience_factor = np.random.uniform(-4, 6)
        logic_factor = np.random.uniform(-3, 4)
        score = base_score + experience_factor + logic_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_conceptual_building_score(self) -> float:
        """生成概念构建得分"""
        # 概念构建受教育背景影响
        base_score = random.uniform(60, 82)
        
        # 教育背景因素
        education_factor = np.random.uniform(-2, 8)
        abstraction_factor = np.random.uniform(-3, 5)
        score = base_score + education_factor + abstraction_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_symbolic_thinking_score(self) -> float:
        """生成象征思维得分"""
        # 象征思维相对稳定
        base_score = random.uniform(68, 88)
        
        # 文化背景影响
        culture_factor = np.random.uniform(-5, 6)
        score = base_score + culture_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_virtual_experience_score(self) -> float:
        """生成虚拟体验得分"""
        # 虚拟体验受同理心和想象力影响
        base_score = random.uniform(58, 88)
        
        # 同理心因素
        empathy_factor = np.random.uniform(-6, 8)
        immersion_factor = np.random.uniform(-4, 6)
        score = base_score + empathy_factor + immersion_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_image_generation_speed(self) -> float:
        """生成图像生成速度"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_visual_creativity(self) -> float:
        """生成视觉创造力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_fantasy_construction(self) -> float:
        """生成幻想构建能力"""
        return round(random.uniform(0.62, 0.87), 3)
    
    def _generate_mental_image_richness(self) -> float:
        """生成心理图像丰富性"""
        return round(random.uniform(0.68, 0.91), 3)
    
    def _generate_future_prediction_accuracy(self) -> float:
        """生成未来预测准确率"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_situation_modeling(self) -> float:
        """生成情景建模能力"""
        return round(random.uniform(0.70, 0.89), 3)
    
    def _generate_temporal_simulation(self) -> float:
        """生成时间模拟能力"""
        return round(random.uniform(0.68, 0.88), 3)
    
    def _generate_contingency_planning(self) -> float:
        """生成应急预案构建"""
        return round(random.uniform(0.65, 0.86), 3)
    
    def _generate_abstract_reasoning(self) -> float:
        """生成抽象推理能力"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_theoretical_building(self) -> float:
        """生成理论构建能力"""
        return round(random.uniform(0.68, 0.87), 3)
    
    def _generate_knowledge_synthesis(self) -> float:
        """生成知识综合能力"""
        return round(random.uniform(0.72, 0.91), 3)
    
    def _generate_conceptual_integration(self) -> float:
        """生成概念整合能力"""
        return round(random.uniform(0.70, 0.89), 3)
    
    def _generate_metaphor_understanding(self) -> float:
        """生成隐喻理解能力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_symbol_interpretation(self) -> float:
        """生成象征解释能力"""
        return round(random.uniform(0.68, 0.89), 3)
    
    def _generate_allegory_comprehension(self) -> float:
        """生成寓言理解能力"""
        return round(random.uniform(0.63, 0.85), 3)
    
    def _generate_symbolic_creation(self) -> float:
        """生成象征创造能力"""
        return round(random.uniform(0.60, 0.84), 3)
    
    def _generate_mental_time_travel(self) -> float:
        """生成心理时间旅行"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_perspective_taking(self) -> float:
        """生成视角转换能力"""
        return round(random.uniform(0.73, 0.91), 3)
    
    def _generate_empathetic_simulation(self) -> float:
        """生成共情模拟能力"""
        return round(random.uniform(0.68, 0.89), 3)
    
    def _generate_immersive_experience(self) -> float:
        """生成沉浸式体验能力"""
        return round(random.uniform(0.72, 0.93), 3)
    
    def _generate_imaginative_confidence(self) -> float:
        """生成想象自信"""
        return round(random.uniform(0.65, 0.90), 3)
    
    def _generate_creative_courage(self) -> float:
        """生成创造勇气"""
        return round(random.uniform(0.58, 0.85), 3)
    
    def _generate_fantasy_tolerance(self) -> float:
        """生成幻想容忍度"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_curiosity_level(self) -> float:
        """生成好奇心水平"""
        return round(random.uniform(0.68, 0.93), 3)
    
    def _generate_imagination_flow(self) -> float:
        """生成想象流状态"""
        return round(random.uniform(0.62, 0.88), 3)
    
    def _generate_creative_incubation(self) -> float:
        """生成创意孵化"""
        return round(random.uniform(0.60, 0.86), 3)
    
    def _generate_insight_emergence_imagination(self) -> float:
        """生成洞察涌现"""
        return round(random.uniform(0.65, 0.89), 3)
    
    def _generate_imagination_synthesis(self) -> float:
        """生成想象综合"""
        return round(random.uniform(0.68, 0.90), 3)
    
    def _generate_imagination_environment(self) -> float:
        """生成想象环境适宜度"""
        return round(random.uniform(0.62, 0.87), 3)
    
    def _generate_creative_stimulation(self) -> float:
        """生成创意激发度"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_imaginative_support(self) -> float:
        """生成想象支持度"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_inspiration_access(self) -> float:
        """生成灵感获取能力"""
        return round(random.uniform(0.58, 0.84), 3)
    
    def _generate_imagination_response_time(self) -> float:
        """生成想象响应时间"""
        return round(random.uniform(1.0, 3.5), 2)
    
    def _generate_imagination_stability(self) -> float:
        """生成想象力稳定性"""
        return round(random.uniform(0.68, 0.90), 3)
    
    def _calculate_imagination_efficiency(self, overall_score: float) -> float:
        """计算想象力效率"""
        # 效率 = 创造性 * 流畅性 * 质量 / 100
        creativity = self.current_imagination_state.get("imaginative_vividness", 0.75)
        speed = min(self.current_imagination_state.get("image_generation_speed", 0.8) * 100, 100) / 100
        quality = overall_score / 100
        
        efficiency = (creativity * speed * quality * overall_score / 100) * 100
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前想象力指标"""
        return self.current_imagination_state.copy()
    
    def update_imagination_metrics(self, **kwargs):
        """更新想象力指标"""
        current_time = time.time()
        
        # 更新指定指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_imagination_state:
                self.current_imagination_state[metric_name] = value
        
        # 更新时间戳
        self.current_imagination_state["timestamp"] = current_time
        
        # 重新计算综合指标
        self._recalculate_composite_metrics()
        
        # 分析趋势
        self._analyze_imagination_trend()
        
        # 添加到历史记录
        self._add_to_history()
    
    def _recalculate_composite_metrics(self):
        """重新计算综合指标"""
        # 重新计算整体得分
        creative_score = self.current_imagination_state["imaginative_vividness"]
        scenario_score = self.current_imagination_state["scenario_depth"]
        conceptual_score = self.current_imagination_state["conceptual_flexibility"]
        symbolic_score = self.current_imagination_state["symbolic_proficiency"]
        virtual_score = self.current_imagination_state["virtual_realism"]
        
        overall_score = (
            creative_score * 0.25 +
            scenario_score * 0.20 +
            conceptual_score * 0.20 +
            symbolic_score * 0.15 +
            virtual_score * 0.20
        )
        
        self.current_imagination_state["overall_score"] = round(overall_score, 1)
        
        # 重新计算效率
        self.current_imagination_state["efficiency"] = self._calculate_imagination_efficiency(overall_score)
    
    def _analyze_imagination_trend(self):
        """分析想象力趋势"""
        if len(self.imagination_history) < 2:
            self.current_imagination_state["trend"] = "stable"
            return
        
        # 获取最近两次整体得分
        recent_scores = []
        for data_point in reversed(self.imagination_history[-10:]):
            if "overall_score" in data_point:
                recent_scores.append(data_point["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 3.2:
                self.current_imagination_state["trend"] = "rising"
            elif change < -3.2:
                self.current_imagination_state["trend"] = "declining"
            else:
                self.current_imagination_state["trend"] = "stable"
    
    def _add_to_history(self):
        """添加当前状态到历史记录"""
        history_point = self.current_imagination_state.copy()
        self.imagination_history.append(history_point)
        
        # 限制历史记录长度
        if len(self.imagination_history) > 100:
            self.imagination_history = self.imagination_history[-100:]
    
    def simulate_imagination_task(self, task_type: str = "random") -> Dict[str, Any]:
        """
        模拟想象力任务执行
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        if task_type == "random":
            task_type = random.choice(self.imagination_tasks)
        
        # 根据任务类型调整性能
        task_performance_factors = {
            "创造性图像生成": 1.0,
            "未来情景预测": 0.92,
            "抽象概念构建": 0.88,
            "象征意义理解": 0.90,
            "虚拟体验模拟": 0.94,
            "故事情节编织": 0.96,
            "空间想象任务": 0.93,
            "角色扮演练习": 0.91,
            "隐喻创造训练": 0.89,
            "可能性探索练习": 0.87
        }
        
        base_performance = task_performance_factors.get(task_type, 1.0)
        
        # 生成任务结果
        vividness = base_performance * random.uniform(0.65, 0.98)
        creativity = base_performance * random.uniform(0.70, 0.96)
        response_time = self.current_imagination_state["response_time"] * random.uniform(0.7, 1.4)
        
        task_result = {
            "task_type": task_type,
            "timestamp": time.time(),
            "vividness": round(vividness, 3),
            "creativity": round(creativity, 3),
            "response_time": round(response_time, 2),
            "imaginative_engagement": random.uniform(0.65, 0.95),
            "novelty_level": random.uniform(0.60, 0.92)
        }
        
        # 根据任务表现调整想象力状态
        performance_impact = ((vividness + creativity) / 2 - 0.75) * 10
        inspiration_boost = task_result["imaginative_engagement"] * 0.08
        
        self.update_imagination_metrics(
            overall_score=self.current_imagination_state["overall_score"] + performance_impact,
            imagination_flow=self.current_imagination_state["imagination_flow"] + inspiration_boost
        )
        
        return task_result
    
    def get_imagination_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取想象力分析报告
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            想象力分析报告
        """
        if not self.imagination_history:
            return {"error": "暂无历史数据"}
        
        # 过滤时间范围内的数据
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [d for d in self.imagination_history if d["timestamp"] >= cutoff_time]
        
        if not recent_data:
            return {"error": "指定时间范围内无数据"}
        
        # 计算统计数据
        scores = [d["overall_score"] for d in recent_data]
        vividness_scores = [d.get("imaginative_vividness", 0.75) for d in recent_data]
        creativity_scores = [d.get("visual_creativity", 0.75) for d in recent_data]
        
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
            
            "imagination_characteristics": {
                "mean_vividness": round(np.mean(vividness_scores), 3),
                "mean_creativity": round(np.mean(creativity_scores), 3),
                "vividness_consistency": round(1 - np.std(vividness_scores)/np.mean(vividness_scores), 3),
                "creativity_stability": round(1 - np.std(creativity_scores)/np.mean(creativity_scores), 3)
            },
            
            "imaginative_state": {
                "average_flow": round(np.mean([d.get("imagination_flow", 0.75) for d in recent_data]), 3),
                "average_confidence": round(np.mean([d.get("imaginative_confidence", 0.78) for d in recent_data]), 3),
                "average_curiosity": round(np.mean([d.get("curiosity_level", 0.80) for d in recent_data]), 3)
            },
            
            "component_analysis": self._analyze_imagination_components(recent_data),
            
            "imagination_insights": self._generate_imagination_insights(recent_data),
            
            "recommendations": self._generate_imagination_recommendations(recent_data)
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
    
    def _analyze_imagination_components(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """分析想象力组件性能"""
        component_scores = {
            "creative_imagination": [],
            "scenario_construction": [],
            "conceptual_building": [],
            "symbolic_thinking": [],
            "virtual_experience": []
        }
        
        for data_point in recent_data:
            component_scores["creative_imagination"].append(data_point.get("imaginative_vividness", 75))
            component_scores["scenario_construction"].append(data_point.get("scenario_depth", 75))
            component_scores["conceptual_building"].append(data_point.get("conceptual_flexibility", 75))
            component_scores["symbolic_thinking"].append(data_point.get("symbolic_proficiency", 75))
            component_scores["virtual_experience"].append(data_point.get("virtual_realism", 75))
        
        component_analysis = {}
        for component, scores in component_scores.items():
            if scores:
                component_analysis[component] = {
                    "mean_score": round(np.mean(scores), 1),
                    "stability": round(1 - np.std(scores)/np.mean(scores), 3),
                    "trend": self._calculate_trend(scores)
                }
        
        return component_analysis
    
    def _generate_imagination_insights(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """生成想象力洞察"""
        current_state = self.current_imagination_state
        
        # 分析想象力模式
        imagination_patterns = []
        
        if current_state["imaginative_vividness"] > 82:
            imagination_patterns.append("高想象生动性 - 能在脑海中生成清晰图像")
        
        if current_state["scenario_depth"] > 80:
            imagination_patterns.append("强情景构建能力 - 善于预测和模拟未来场景")
        
        if current_state["conceptual_flexibility"] > 78:
            imagination_patterns.append("优秀概念灵活性 - 能快速理解和构建抽象概念")
        
        if current_state["virtual_realism"] > 83:
            imagination_patterns.append("强虚拟体验能力 - 能沉浸式地体验想象场景")
        
        return {
            "dominant_patterns": imagination_patterns,
            "imagination_potential": round((current_state["imaginative_vividness"] + current_state["visual_creativity"]) / 2, 1),
            "creative_courage": current_state["creative_courage"],
            "inspiration_access": current_state["inspiration_access"]
        }
    
    def _generate_imagination_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """生成想象力改善建议"""
        recommendations = []
        
        # 分析当前状态
        avg_flow = np.mean([d.get("imagination_flow", 0.75) for d in recent_data])
        avg_confidence = np.mean([d.get("imaginative_confidence", 0.78) for d in recent_data])
        avg_curiosity = np.mean([d.get("curiosity_level", 0.80) for d in recent_data])
        
        if avg_flow < 0.7:
            recommendations.append("想象流状态较低，建议进行放松冥想或自由联想练习")
        
        if avg_confidence < 0.75:
            recommendations.append("想象自信需要提升，建议从小的创意练习开始")
        
        if avg_curiosity < 0.75:
            recommendations.append("好奇心可以激发，建议接触新的体验和知识领域")
        
        # 基于组件性能提供建议
        current_state = self.current_imagination_state
        
        if current_state["imaginative_vividness"] < 70:
            recommendations.append("想象生动性有待提升，建议进行视觉化练习和图像训练")
        
        if current_state["scenario_depth"] < 72:
            recommendations.append("情景构建能力需要加强，建议练习情景模拟和预测练习")
        
        if current_state["conceptual_flexibility"] < 75:
            recommendations.append("概念灵活性可以培养，建议学习哲学和理论思维")
        
        if current_state["virtual_realism"] < 78:
            recommendations.append("虚拟体验能力可提升，建议进行角色扮演和同理心练习")
        
        # 基于时间提供建议
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11:
            recommendations.append("上午想象状态适中，适合进行结构化的想象力训练")
        elif 14 <= current_hour <= 16:
            recommendations.append("下午适合进行概念构建和理论想象练习")
        elif 20 <= current_hour <= 23:
            recommendations.append("晚间为想象黄金期，适合进行自由创作和艺术想象")
        
        # 环境建议
        if current_state["imagination_environment"] < 0.7:
            recommendations.append("想象环境需要改善，建议寻找安静、激发灵感的空间")
        
        if current_state["creative_stimulation"] < 0.65:
            recommendations.append("创意激发不足，建议接触艺术、音乐或其他创造性的内容")
        
        return recommendations if recommendations else ["当前想象力状态良好，继续保持创意实践"]
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择要更新的指标
        updateable_metrics = [
            "imaginative_vividness",
            "scenario_depth", 
            "conceptual_flexibility",
            "symbolic_proficiency",
            "virtual_realism"
        ]
        
        metric_to_update = random.choice(updateable_metrics)
        
        # 生成新的值
        if metric_to_update == "imaginative_vividness":
            new_value = self._generate_creative_imagination_score()
        elif metric_to_update == "scenario_depth":
            new_value = self._generate_scenario_construction_score()
        elif metric_to_update == "conceptual_flexibility":
            new_value = self._generate_conceptual_building_score()
        elif metric_to_update == "symbolic_proficiency":
            new_value = self._generate_symbolic_thinking_score()
        elif metric_to_update == "virtual_realism":
            new_value = self._generate_virtual_experience_score()
        else:
            new_value = self.current_imagination_state.get(metric_to_update, 75)
        
        # 更新指标
        self.update_imagination_metrics(**{metric_to_update: new_value})
        
        return metric_to_update


# 测试函数
def test_imagination_monitor():
    """测试想象力监控器"""
    print("开始测试想象力监控器...")
    
    # 创建想象力监控器
    imagination_monitor = ImaginationMonitor()
    
    # 测试当前指标
    print("\n1. 当前想象力指标:")
    current_metrics = imagination_monitor.get_current_metrics()
    print(f"  整体得分: {current_metrics['overall_score']:.1f}%")
    print(f"  想象生动性: {current_metrics['imaginative_vividness']:.1%}")
    print(f"  情景深度: {current_metrics['scenario_depth']:.1%}")
    print(f"  虚拟现实感: {current_metrics['virtual_realism']:.1%}")
    
    # 模拟想象力任务
    print("\n2. 模拟想象力任务:")
    for i in range(3):
        result = imagination_monitor.simulate_imagination_task()
        print(f"  任务{i+1}: {result['task_type']} - 生动性: {result['vividness']:.1%}")
    
    # 模拟实时更新
    print("\n3. 模拟实时更新 (5次):")
    for i in range(5):
        metric = imagination_monitor.simulate_real_time_update()
        print(f"  更新{i+1}: {metric}")
    
    # 生成分析报告
    print("\n4. 想象力分析:")
    analysis = imagination_monitor.get_imagination_analysis(hours=1)
    if "error" not in analysis:
        print(f"  数据点数量: {analysis['data_points']}")
        print(f"  平均得分: {analysis['overall_performance']['mean_score']:.1f}")
        print(f"  想象洞察: {len(analysis['imagination_insights']['dominant_patterns'])}个")
        print(f"  主要建议: {analysis['recommendations'][0]}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_imagination_monitor()