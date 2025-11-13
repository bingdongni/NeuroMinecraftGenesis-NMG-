"""
大规模多智能体协同进化系统测试脚本

该脚本用于测试系统的主要功能：
1. 系统初始化
2. 智能体创建
3. 社会学习
4. 集体决策
5. 文化进化
6. 网络重组
"""

import sys
import os
import time
import numpy as np

# 添加路径以便导入模块
sys.path.append('/workspace/agents/mass_evolution')

from multi_agent_society import (
    MassEvolutionSystem, 
    SocialAgent, 
    AgentState, 
    NetworkLayer,
    DecisionMaker,
    CulturalEvolution,
    SocialLearningSystem
)

def test_system_initialization():
    """测试系统初始化"""
    print("🧪 测试1: 系统初始化")
    print("-" * 40)
    
    # 创建小规模系统进行测试
    system = MassEvolutionSystem(num_agents=100)
    
    assert len(system.agents) == 100, f"预期100个智能体，实际{len(system.agents)}"
    assert len(system.network_manager.layers[NetworkLayer.INDIVIDUAL]) > 0, "层级组织失败"
    
    print(f"✅ 成功创建 {len(system.agents)} 个智能体")
    print(f"✅ 网络层级: {list(system.network_manager.layers.keys())}")
    print(f"✅ 初始系统指标: {system.system_metrics}")
    
    return system

def test_social_learning(system):
    """测试社会学习功能"""
    print("\n🧪 测试2: 社会学习")
    print("-" * 40)
    
    # 选择两个智能体进行学习测试
    agents_list = list(system.agents.values())
    teacher = agents_list[0]
    learner = agents_list[1]
    
    # 为教师创建知识节点
    from multi_agent_society import KnowledgeNode
    knowledge_node = KnowledgeNode(
        id="test_knowledge",
        content="test_content",
        confidence=0.8,
        creator_agent=teacher.id,
        timestamp=time.time()
    )
    teacher.knowledge_base["test_knowledge"] = knowledge_node
    
    # 执行学习
    learning_result = system.social_learning.social_learn(learner, teacher, knowledge_node, 'imitation')
    
    print(f"✅ 学习结果: {learning_result}")
    assert learning_result['success'] in [True, False], "学习结果格式错误"
    
    # 测试不同学习策略
    strategies = ['imitation', 'innovation', 'collaboration', 'competitive']
    for strategy in strategies:
        result = system.social_learning.social_learn(learner, teacher, knowledge_node, strategy)
        print(f"✅ {strategy} 学习: {result['success']}")
    
    return True

def test_collective_decision(system):
    """测试集体决策功能"""
    print("\n🧪 测试3: 集体决策")
    print("-" * 40)
    
    # 选择决策团队
    agents_list = list(system.agents.values())[:10]  # 选择前10个智能体
    
    # 创建项目提案
    class Project:
        def __init__(self, requirements):
            self.requirements = requirements
    
    proposal = Project({
        'reasoning': 0.7,
        'collaboration': 0.6,
        'creativity': 0.5
    })
    
    # 执行集体决策
    decision_result = system.decision_maker.collective_decision(
        agents_list, proposal, 'project_approval', 'collaboration'
    )
    
    print(f"✅ 决策结果: {decision_result['decision']}")
    print(f"✅ 置信度: {decision_result['confidence']:.3f}")
    print(f"✅ 参与者: {decision_result['participants']}")
    
    assert decision_result['decision'] in ['approved', 'rejected', 'no_consensus'], "决策结果无效"
    
    return True

def test_cultural_evolution(system):
    """测试文化进化功能"""
    print("\n🧪 测试4: 文化进化")
    print("-" * 40)
    
    # 创建文化产物
    creators = [agent.id for agent in list(system.agents.values())[:3]]
    artifact = system.cultural_evolution.create_cultural_artifact(
        creators, 'skill', {'test': 'skill_content'}, {}
    )
    
    print(f"✅ 创建文化产物: {artifact.id}")
    print(f"✅ 产物类型: {artifact.type}")
    print(f"✅ 初始有效性: {artifact.effectiveness_score:.3f}")
    
    # 测试文化扩散
    diffusion_success = False
    if creators:
        from_agent = creators[0]
        target_agents = [aid for aid, conn in system.agents[from_agent].social_connections.items()]
        if target_agents:
            to_agent = target_agents[0]
            success = system.cultural_evolution.diffuse_cultural_knowledge(
                from_agent, to_agent, artifact.id
            )
            print(f"✅ 文化扩散: {success}")
            diffusion_success = success
    
    return artifact.id if artifact else None

def test_network_organization(system):
    """测试网络组织功能"""
    print("\n🧪 测试5: 网络组织")
    print("-" * 40)
    
    # 重组网络
    original_layers = {layer: len(agents) for layer, agents in system.network_manager.layers.items()}
    print(f"原始层级分布: {original_layers}")
    
    system.network_manager.reorganize_layers(list(system.agents.values()), 'performance')
    
    new_layers = {layer.value: len(agents) for layer, agents in system.network_manager.layers.items()}
    print(f"重组后层级分布: {new_layers}")
    
    # 验证层级变化
    layer_changes = sum(1 for k in original_layers if original_layers[k] != len(system.network_manager.layers.get(k, [])))
    print(f"✅ 层级重组完成，{layer_changes}个层级发生变化")
    
    return True

def test_evolution_cycle(system):
    """测试进化周期"""
    print("\n🧪 测试6: 进化周期")
    print("-" * 40)
    
    initial_agent_count = len(system.agents)
    initial_metrics = system.system_metrics.copy()
    
    # 运行单个进化周期
    start_time = time.time()
    cycle_result = system.run_evolution_cycle(num_cycles=1)
    cycle_time = time.time() - start_time
    
    print(f"✅ 进化周期耗时: {cycle_time:.2f}秒")
    print(f"✅ 智能体数量变化: {initial_agent_count} -> {len(system.agents)}")
    print(f"✅ 系统指标改善:")
    
    for metric, initial_value in initial_metrics.items():
        new_value = system.system_metrics[metric]
        change = new_value - initial_value
        print(f"   {metric}: {initial_value:.4f} -> {new_value:.4f} (变化: {change:+.4f})")
    
    return cycle_result

def test_system_metrics(system):
    """测试系统指标计算"""
    print("\n🧪 测试7: 系统指标")
    print("-" * 40)
    
    metrics = system._calculate_system_metrics()
    
    expected_metrics = [
        'average_fitness', 'diversity_index', 'collaboration_rate', 
        'innovation_rate', 'cultural_diffusion_speed'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"缺少指标: {metric}"
        value = metrics[metric]
        assert isinstance(value, (int, float)), f"指标 {metric} 应该是数值类型"
        
        # 特定指标的范围检查
        if metric in ['average_fitness', 'diversity_index', 'collaboration_rate']:
            assert 0 <= value <= 1, f"指标 {metric} 超出合理范围 [0, 1]"
        elif metric == 'innovation_rate':
            assert value >= 0, f"指标 {metric} 应该非负"
        elif metric == 'cultural_diffusion_speed':
            assert value >= 0, f"指标 {metric} 应该非负"
        
        print(f"✅ {metric}: {value:.4f}")
    
    return metrics

def test_large_scale_simulation():
    """测试大规模模拟（简化版本）"""
    print("\n🧪 测试8: 大规模模拟")
    print("-" * 40)
    
    print("正在创建大规模系统（500个智能体）...")
    large_system = MassEvolutionSystem(num_agents=500)
    
    print("运行快速进化测试（5个周期）...")
    start_time = time.time()
    results = large_system.run_evolution_cycle(num_cycles=5)
    test_time = time.time() - start_time
    
    print(f"✅ 大规模模拟完成，耗时: {test_time:.2f}秒")
    print(f"✅ 最终智能体数量: {len(large_system.agents)}")
    print(f"✅ 最终平均适应度: {large_system.system_metrics['average_fitness']:.4f}")
    print(f"✅ 文化产物数量: {len(large_system.cultural_evolution.cultural_artifacts)}")
    
    return large_system

def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("大规模多智能体协同进化系统综合测试")
    print("=" * 60)
    
    try:
        # 初始化系统
        system = test_system_initialization()
        
        # 测试核心功能
        test_social_learning(system)
        test_collective_decision(system)
        artifact_id = test_cultural_evolution(system)
        test_network_organization(system)
        test_evolution_cycle(system)
        test_system_metrics(system)
        
        # 大规模测试
        large_system = test_large_scale_simulation()
        
        # 保存测试结果
        test_output_file = "/workspace/agents/mass_evolution/test_results.json"
        system.save_system_state(test_output_file)
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("✅ 大规模多智能体协同进化系统功能正常")
        print(f"✅ 测试结果已保存到: {test_output_file}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)