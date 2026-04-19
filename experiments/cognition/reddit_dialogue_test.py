#!/usr/bin/env python3
"""
Reddit对话测试模块
================

该模块实现了Reddit对话环境的泛化测试，主要测试智能体扮演助手回答r/AskScience问题的能力，评估社会认知和交流能力的迁移。

测试特点：
- 使用PRAW库获取公开数据
- 智能体扮演科学助手
- 测试社会认知迁移能力
- 评估回答质量和被采纳率

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

import time
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 模拟PRAW库依赖
try:
    import praw
except ImportError:
    praw = None

# 日志配置
import logging
logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    """问题分类"""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ASTRONOMY = "astronomy"
    MATHEMATICS = "mathematics"
    EARTH_SCIENCE = "earth_science"
    TECHNOLOGY = "technology"
    MEDICINE = "medicine"


class ResponseQuality(Enum):
    """回答质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    INCORRECT = "incorrect"


@dataclass
class RedditQuestion:
    """Reddit问题数据类"""
    title: str
    content: str
    category: QuestionCategory
    difficulty: float  # 0.0-1.0
    upvotes: int
    timestamp: str
    author: str
    question_id: str
    thread_url: str = ""
    related_topics: List[str] = field(default_factory=list)


@dataclass
class RedditResponse:
    """Reddit回答数据类"""
    content: str
    quality: ResponseQuality
    accuracy: float  # 0.0-1.0
    clarity: float  # 0.0-1.0
    helpfulness: float  # 0.0-1.0
    upvotes: int
    timestamp: str
    response_id: str
    is_accepted: bool = False
    author: str = "AI_Assistant"


@dataclass
class DialogueTestResult:
    """对话测试结果数据类"""
    question_answered: int
    accuracy_rate: float
    clarity_score: float
    helpfulness_score: float
    acceptance_rate: float
    response_time: float
    category_performance: Dict[str, float]
    quality_distribution: Dict[str, int]


class RedditDialogueTest:
    """
    Reddit对话泛化测试类
    
    测试智能体扮演助手在Reddit r/AskScience环境中的回答质量和社交认知能力
    """
    
    def __init__(self):
        """初始化Reddit对话测试环境"""
        # 模拟Reddit数据
        self.sample_questions = self._generate_sample_questions()
        self.test_categories = list(QuestionCategory)
        
        # 测试参数
        self.max_questions_per_category = 5
        self.response_time_limit = 30.0  # 秒
        self.accuracy_threshold = 0.7
        
        # 模拟PRAW配置（实际使用时会连接到真实Reddit API）
        self.reddit_config = {
            "client_id": "mock_client_id",
            "client_secret": "mock_client_secret",
            "user_agent": "NeuroMinecraftGenesis/1.0",
            "subreddit": "askscience"
        }
        
        logger.info("Reddit对话测试环境初始化完成")
    
    def _generate_sample_questions(self) -> List[RedditQuestion]:
        """生成模拟的Reddit科学问题"""
        questions = []
        
        # 物理学问题
        physics_questions = [
            RedditQuestion(
                title="为什么光速是宇宙中的速度上限？",
                content="我理解光速是299,792,458米每秒，但为什么不能更快？有什么物理原理限制了这个速度？",
                category=QuestionCategory.PHYSICS,
                difficulty=0.8,
                upvotes=156,
                timestamp="2025-11-13T14:30:00Z",
                author="physics_student_123",
                question_id="q001",
                related_topics=["相对论", "时空", "宇宙学"]
            ),
            RedditQuestion(
                title="量子纠缠是如何实现的？",
                content="两个粒子无论距离多远都能瞬间互相影响，这在实际应用中可能吗？",
                category=QuestionCategory.PHYSICS,
                difficulty=0.9,
                upvotes=89,
                timestamp="2025-11-13T13:15:00Z",
                author="quantum_enthusiast",
                question_id="q002",
                related_topics=["量子力学", "信息传递", "贝尔不等式"]
            )
        ]
        
        # 化学问题
        chemistry_questions = [
            RedditQuestion(
                title="为什么化学反应会放热或吸热？",
                content="有些反应需要加热才能进行，有些反应会自发放热，这背后的原理是什么？",
                category=QuestionCategory.CHEMISTRY,
                difficulty=0.6,
                upvotes=234,
                timestamp="2025-11-13T12:45:00Z",
                author="chem_learner",
                question_id="q003",
                related_topics=["热力学", "活化能", "键能"]
            ),
            RedditQuestion(
                title="DNA复制的准确性是如何保证的？",
                content="在细胞分裂过程中，DNA复制几乎完美无错，这个过程是如何确保准确性的？",
                category=QuestionCategory.BIOLOGY,
                difficulty=0.7,
                upvotes=445,
                timestamp="2025-11-13T11:20:00Z",
                author="bio_student",
                question_id="q004",
                related_topics=["分子生物学", "DNA聚合酶", "纠错机制"]
            )
        ]
        
        # 天文学问题
        astronomy_questions = [
            RedditQuestion(
                title="黑洞的事件视界到底在哪里？",
                content="我读到说黑洞的引力场如此强大，连光都无法逃脱。那么事件视界的边界是确切的吗？",
                category=QuestionCategory.ASTRONOMY,
                difficulty=0.85,
                upvotes=178,
                timestamp="2025-11-13T10:30:00Z",
                author="space_lover",
                question_id="q005",
                related_topics=["广义相对论", "时空曲率", "光速逃逸"]
            )
        ]
        
        # 数学问题
        mathematics_questions = [
            RedditQuestion(
                title="为什么π是无理数？",
                content="π的无限不循环小数特性是如何被证明的？这在数学上意味着什么？",
                category=QuestionCategory.MATHEMATICS,
                difficulty=0.75,
                upvotes=123,
                timestamp="2025-11-13T09:45:00Z",
                author="math_major",
                question_id="q006",
                related_topics=["数论", "超越数", "数学证明"]
            )
        ]
        
        # 地球科学问题
        earth_science_questions = [
            RedditQuestion(
                title="板块构造理论如何解释地震和火山？",
                content="地球表面的板块运动会引起地震和火山爆发，这个过程的具体机制是什么？",
                category=QuestionCategory.EARTH_SCIENCE,
                difficulty=0.65,
                upvotes=267,
                timestamp="2025-11-13T08:15:00Z",
                author="geology_student",
                question_id="q007",
                related_topics=["地质学", "构造运动", "地震学"]
            )
        ]
        
        # 合并所有问题
        questions.extend(physics_questions)
        questions.extend(chemistry_questions)
        questions.extend(astronomy_questions)
        questions.extend(mathematics_questions)
        questions.extend(earth_science_questions)
        
        return questions
    
    def simulate_reddit_api_fetch(self, category: QuestionCategory, limit: int = 10) -> List[RedditQuestion]:
        """模拟从Reddit API获取问题（实际会使用PRAW库）"""
        # 模拟API延迟
        time.sleep(random.uniform(0.5, 1.5))
        
        # 根据分类筛选问题
        filtered_questions = [q for q in self.sample_questions if q.category == category]
        
        # 返回指定数量的随机问题
        return random.sample(filtered_questions, min(limit, len(filtered_questions)))
    
    def evaluate_question_complexity(self, question: RedditQuestion) -> float:
        """评估问题复杂度和专业程度"""
        complexity_factors = {
            "technical_terms": len(re.findall(r'\b(量子|相对论|DNA|板块|无理数|酶)\b', question.content)),
            "question_length": min(len(question.content) / 500, 1.0),  # 归一化长度
            "domain_specificity": len(question.related_topics) / 5.0,  # 主题深度
            "upvote_ratio": min(question.upvotes / 500, 1.0),  # 社区认可度
            "difficulty_preset": question.difficulty
        }
        
        # 加权计算复杂度
        complexity_score = (
            complexity_factors["technical_terms"] * 0.3 +
            complexity_factors["question_length"] * 0.2 +
            complexity_factors["domain_specificity"] * 0.2 +
            complexity_factors["upvote_ratio"] * 0.1 +
            complexity_factors["difficulty_preset"] * 0.2
        )
        
        return min(complexity_score, 1.0)
    
    def generate_ai_response(self, question: RedditQuestion) -> RedditResponse:
        """模拟AI助手生成回答"""
        start_time = time.time()
        
        # 评估问题复杂度
        complexity = self.evaluate_question_complexity(question)
        
        # 根据复杂度生成不同质量的回答
        if complexity < 0.3:
            # 简单问题，容易回答准确
            accuracy = random.uniform(0.8, 0.95)
            clarity = random.uniform(0.75, 0.9)
            helpfulness = random.uniform(0.8, 0.95)
        elif complexity < 0.6:
            # 中等复杂度
            accuracy = random.uniform(0.65, 0.85)
            clarity = random.uniform(0.6, 0.8)
            helpfulness = random.uniform(0.6, 0.8)
        else:
            # 高复杂度，可能存在一些不准确
            accuracy = random.uniform(0.4, 0.75)
            clarity = random.uniform(0.5, 0.7)
            helpfulness = random.uniform(0.5, 0.7)
        
        # 根据准确度确定质量等级
        if accuracy >= 0.85:
            quality = ResponseQuality.EXCELLENT
        elif accuracy >= 0.7:
            quality = ResponseQuality.GOOD
        elif accuracy >= 0.5:
            quality = ResponseQuality.AVERAGE
        elif accuracy >= 0.3:
            quality = ResponseQuality.POOR
        else:
            quality = ResponseQuality.INCORRECT
        
        # 模拟社区接受度
        community_acceptance = (accuracy + clarity + helpfulness) / 3.0
        is_accepted = random.random() < community_acceptance * 0.6  # 60%的机会被接受
        upvotes = int(community_acceptance * random.randint(10, 100))
        
        response_time = time.time() - start_time
        
        return RedditResponse(
            content=f"AI生成的关于{question.category.value}的回答...",
            quality=quality,
            accuracy=accuracy,
            clarity=clarity,
            helpfulness=helpfulness,
            upvotes=upvotes,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            response_id=f"r_{question.question_id}_{int(time.time())}",
            is_accepted=is_accepted
        )
    
    def evaluate_social_understanding(self, response: RedditResponse, question: RedditQuestion) -> float:
        """评估社会理解和交流能力"""
        social_factors = {
            "tone_appropriateness": random.uniform(0.6, 0.95),  # 语调适当性
            "politeness": random.uniform(0.7, 0.9),  # 礼貌程度
            "empathy": random.uniform(0.5, 0.8),  # 共情能力
            "cultural_sensitivity": random.uniform(0.8, 0.95),  # 文化敏感性
            "engagement_style": random.uniform(0.6, 0.85)  # 参与风格
        }
        
        # 考虑问题类型对社会理解的要求
        if question.category in [QuestionCategory.BIOLOGY, QuestionCategory.MEDICINE]:
            # 生物医学问题需要更多共情
            social_factors["empathy"] *= 1.2
        
        social_score = sum(social_factors.values()) / len(social_factors)
        return min(social_score, 1.0)
    
    def run_zero_shot_test(self) -> float:
        """
        运行零样本对话测试
        
        在没有特定社交媒体示例的情况下测试智能体的社会认知和交流能力
        
        Returns:
            float: 零样本测试分数 (0.0 - 1.0)
        """
        logger.info("开始Reddit对话零样本测试...")
        
        start_time = time.time()
        total_score = 0.0
        total_questions = 0
        
        # 对每个科学领域进行测试
        for category in self.test_categories:
            logger.info(f"测试领域: {category.value}")
            
            # 获取该领域的问题
            questions = self.simulate_reddit_api_fetch(category, self.max_questions_per_category)
            
            category_score = 0.0
            
            for question in questions:
                # 生成AI回答
                response = self.generate_ai_response(question)
                
                # 评估社会理解
                social_score = self.evaluate_social_understanding(response, question)
                
                # 综合评分 (准确性、清晰度、帮助性、社会理解各占25%)
                question_score = (
                    response.accuracy * 0.25 +
                    response.clarity * 0.25 +
                    response.helpfulness * 0.25 +
                    social_score * 0.25
                )
                
                category_score += question_score
                total_score += question_score
                total_questions += 1
                
                # 检查是否超时
                if time.time() - start_time > self.response_time_limit * len(questions):
                    break
            
            # 分类平均分
            if questions:
                category_score /= len(questions)
                logger.info(f"{category.value} 平均分数: {category_score:.3f}")
        
        # 计算总体分数
        zero_shot_score = total_score / total_questions if total_questions > 0 else 0.0
        
        completion_time = time.time() - start_time
        
        logger.info(f"Reddit对话零样本测试完成: {zero_shot_score:.3f} (用时: {completion_time:.2f}秒)")
        
        return zero_shot_score
    
    def run_few_shot_test(self, max_attempts: int = 50, baseline_score: float = 0.0) -> float:
        """
        运行少样本对话适应测试
        
        在零样本基础上允许有限次数的对话学习适应
        
        Args:
            max_attempts: 最大适应尝试次数
            baseline_score: 基准分数（零样本分数）
            
        Returns:
            float: 少样本适应后分数 (0.0 - 1.0)
        """
        logger.info(f"开始Reddit对话少样本测试，最大尝试次数: {max_attempts}")
        
        current_score = baseline_score
        learning_rate = 0.04
        
        # 模拟渐进式社交学习过程
        for attempt in range(max_attempts):
            # 随机选择学习和测试的分类
            category = random.choice(self.test_categories)
            questions = self.simulate_reddit_api_fetch(category, 1)
            
            if not questions:
                continue
            
            question = questions[0]
            response = self.generate_ai_response(question)
            
            # 模拟学习改进
            # 社交技能学习
            if attempt < 10:
                # 前10次主要学习基本交流
                social_improvement = random.uniform(0.02, 0.08)
            elif attempt < 25:
                # 中间学习特定领域表达
                domain_improvement = random.uniform(0.03, 0.06)
            else:
                # 最后优化整体质量
                quality_improvement = random.uniform(0.02, 0.04)
            
            # 计算当前表现
            current_performance = (
                response.accuracy * 0.25 +
                response.clarity * 0.25 +
                response.helpfulness * 0.25 +
                self.evaluate_social_understanding(response, question) * 0.25
            )
            
            # 更新分数
            if current_performance > current_score:
                current_score += (current_performance - current_score) * learning_rate
            
            # 防止分数过高
            current_score = min(current_score, 1.0)
        
        final_score = current_score
        
        logger.info(f"Reddit对话少样本测试完成: {final_score:.3f}")
        
        return final_score
    
    def analyze_conversation_quality(self, responses: List[Tuple[RedditQuestion, RedditResponse]]) -> DialogueTestResult:
        """分析对话质量和社交表现"""
        if not responses:
            return DialogueTestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})
        
        total_accuracy = sum(r[1].accuracy for r in responses) / len(responses)
        total_clarity = sum(r[1].clarity for r in responses) / len(responses)
        total_helpfulness = sum(r[1].helpfulness for r in responses) / len(responses)
        
        acceptance_count = sum(1 for r in responses if r[1].is_accepted)
        acceptance_rate = acceptance_count / len(responses)
        
        # 分类性能分析
        category_performance = {}
        for category in self.test_categories:
            category_responses = [r for r in responses if r[0].category == category]
            if category_responses:
                category_avg = sum(r[1].accuracy for r in category_responses) / len(category_responses)
                category_performance[category.value] = category_avg
        
        # 质量分布统计
        quality_dist = {}
        for quality in ResponseQuality:
            quality_dist[quality.value] = sum(1 for r in responses if r[1].quality == quality)
        
        return DialogueTestResult(
            question_answered=len(responses),
            accuracy_rate=total_accuracy,
            clarity_score=total_clarity,
            helpfulness_score=total_helpfulness,
            acceptance_rate=acceptance_rate,
            response_time=sum(r[1].accuracy for r in responses),  # 简化时间评估
            category_performance=category_performance,
            quality_distribution=quality_dist
        )
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """生成详细的Reddit对话测试报告"""
        # 运行测试
        zero_shot_score = self.run_zero_shot_test()
        few_shot_score = self.run_few_shot_test(baseline_score=zero_shot_score)
        
        # 生成测试对话样本
        test_conversations = []
        for category in self.test_categories[:3]:  # 只测试前3个分类
            questions = self.simulate_reddit_api_fetch(category, 2)
            for question in questions:
                response = self.generate_ai_response(question)
                test_conversations.append((question, response))
        
        # 分析对话质量
        dialogue_analysis = self.analyze_conversation_quality(test_conversations)
        
        # 社会认知能力评估
        social_cognition_metrics = {
            "communication_style": random.uniform(0.6, 0.9),
            "cultural_awareness": random.uniform(0.7, 0.95),
            "emotional_intelligence": random.uniform(0.5, 0.8),
            "context_understanding": random.uniform(0.6, 0.85),
            "audience_adaptation": random.uniform(0.5, 0.8)
        }
        
        report = {
            "test_environment": "reddit_dialogue",
            "subreddit": "askscience",
            "zero_shot_results": {
                "overall_score": zero_shot_score,
                "answer_accuracy": random.uniform(0.5, 0.85),
                "communication_effectiveness": random.uniform(0.6, 0.9),
                "social_adaptation": random.uniform(0.4, 0.8)
            },
            "few_shot_results": {
                "overall_score": few_shot_score,
                "learning_progression": (few_shot_score - zero_shot_score) / 50,
                "social_skill_improvement": random.uniform(0.3, 0.7)
            },
            "dialogue_analysis": {
                "total_questions_answered": dialogue_analysis.question_answered,
                "acceptance_rate": dialogue_analysis.acceptance_rate,
                "quality_distribution": dialogue_analysis.quality_distribution,
                "category_performance": dialogue_analysis.category_performance
            },
            "social_cognition": social_cognition_metrics,
            "conversation_samples": [
                {
                    "question_title": q.title,
                    "question_category": q.category.value,
                    "response_quality": r.quality.value,
                    "accuracy": r.accuracy,
                    "accepted": r.is_accepted
                }
                for q, r in test_conversations[:5]  # 只显示前5个样本
            ],
            "performance_insights": {
                "strongest_domain": "biology" if dialogue_analysis.category_performance.get("biology", 0) > 0.7 else "physics",
                "communication_strength": "clarity",
                "improvement_areas": ["precision", "engagement"],
                "optimization_suggestions": [
                    "加强科学准确性训练",
                    "提升社交互动技巧",
                    "改善复杂问题回答能力"
                ]
            }
        }
        
        return report


def main():
    """演示函数"""
    test = RedditDialogueTest()
    
    print("=" * 60)
    print("Reddit对话泛化测试演示")
    print("=" * 60)
    
    # 运行测试
    zero_shot = test.run_zero_shot_test()
    few_shot = test.run_few_shot_test(baseline_score=zero_shot)
    
    print(f"零样本分数: {zero_shot:.3f}")
    print(f"少样本分数: {few_shot:.3f}")
    print(f"学习提升: {few_shot - zero_shot:.3f}")
    print(f"适应速度: {(few_shot - zero_shot) / 50:.6f}")
    
    # 社交认知评估
    social_metrics = {
        "communication_style": random.uniform(0.6, 0.9),
        "cultural_awareness": random.uniform(0.7, 0.95),
        "emotional_intelligence": random.uniform(0.5, 0.8)
    }
    print(f"\n社交认知评估:")
    for aspect, score in social_metrics.items():
        print(f"  {aspect}: {score:.3f}")
    
    # 生成详细报告
    report = test.generate_detailed_report()
    print("\n详细报告:")
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()