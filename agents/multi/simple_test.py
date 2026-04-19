"""
简化版多智能体系统测试
快速验证核心功能
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加路径
sys.path.append('/workspace/NeuroMinecraftGenesis/agents/multi')

def test_basic_functionality():
    """测试基本功能"""
    print("🤖 开始简化测试")
    
    try:
        # 1. 测试集体记忆系统
        print("📝 测试集体记忆系统...")
        from collective_memory import CollectiveMemory, create_danger_zone_memory
        
        memory_system = CollectiveMemory()
        
        # 创建记忆
        memory = create_danger_zone_memory(100, 0, 50, "test", "测试危险区域", "test_agent")
        memory_id = memory_system.store_memory(memory)
        
        # 检索记忆
        retrieved = memory_system.retrieve_memories(memory_type="danger_zone")
        print(f"✅ 集体记忆: 存储 {memory_id}, 检索到 {len(retrieved)} 条")
        
        # 2. 测试社会认知系统
        print("🧠 测试社会认知系统...")
        from social_cognition import SocialCognitionSystem, SocialAction
        
        social_system = SocialCognitionSystem(agent_count=4)
        
        # 记录行为
        action = SocialAction(
            actor_id="agent_0",
            action_type="help",
            target_id="agent_1", 
            timestamp=datetime.now(),
            success=True,
            impact_score=0.7,
            description="测试帮助",
            context={}
        )
        social_system.record_social_action(action)
        
        # 分析意图
        intentions = social_system.analyze_intentions("agent_0")
        print(f"✅ 社会认知: 记录行为, 分析 {len(intentions)} 种意图")
        
        # 3. 测试协作协议
        print("🤝 测试协作协议...")
        from collaboration_protocol import CollaborationProtocol, Task, TaskType, TaskPriority, Resource, ResourceType
        
        collab_system = CollaborationProtocol(agent_count=4)
        
        # 创建资源
        resource = Resource(
            id="",
            resource_type=ResourceType.MATERIAL,
            name="测试材料",
            quantity=100,
            quality=0.8,
            shared=True,
            accessibility="medium"
        )
        resource_id = collab_system.create_resource(resource)
        
        # 创建任务
        task = Task(
            id="",
            task_type=TaskType.EXPLORATION,
            title="测试任务",
            description="用于测试",
            priority=TaskPriority.MEDIUM,
            estimated_duration=4,
            required_skills=["exploration"],
            required_resources={ResourceType.MATERIAL: 5},
            created_by="test_system"
        )
        task_id = collab_system.create_task(task)
        
        # 分配任务
        assignment = collab_system.assign_task(task_id, "agent_0")
        print(f"✅ 协作协议: 创建资源 {resource_id}, 任务 {task_id}, 分配{'成功' if assignment else '失败'}")
        
        # 4. 测试部落系统
        print("🏛️ 测试部落系统...")
        from tribal_society import TribalSociety
        
        tribe = TribalSociety(agent_count=4)
        
        # 执行模拟步骤
        tribe.is_running = True
        tribe._execute_simulation_step()
        tribe.is_running = False
        
        # 获取状态
        status = tribe.get_tribal_status()
        print(f"✅ 部落系统: 初始化 {len(tribe.agents)} 个智能体, 执行 {tribe.simulation_step} 步")
        
        print("\n🎉 所有核心功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_demo():
    """运行简单演示"""
    print("\n🎬 运行演示...")
    
    try:
        from tribal_society import TribalSociety
        from collaboration_protocol import create_simple_task
        
        # 创建演示部落
        tribe = TribalSociety(agent_count=6)
        
        print(f"创建了包含 {len(tribe.agents)} 个智能体的部落")
        
        # 显示智能体信息
        for agent_id, agent in list(tribe.agents.items())[:3]:
            print(f"  🤖 {agent_id}: {agent.personality.value} (能量: {agent.energy_level:.2f})")
        
        # 运行5步模拟
        tribe.is_running = True
        for step in range(5):
            tribe._execute_simulation_step()
            tribe.simulation_step += 1
            tribe.simulation_time += __import__('datetime').timedelta(hours=1)
            
            if step % 2 == 1:  # 每2步显示一次状态
                active_agents = len([a for a in tribe.agents.values() if a.current_state.value != 'idle'])
                print(f"  📊 步骤 {step+1}: {active_agents} 个活跃智能体")
        
        tribe.is_running = False
        
        # 显示最终结果
        metrics = tribe.get_tribal_status()
        print(f"\n🏆 演示结果:")
        print(f"  ⏱️  模拟步数: {tribe.simulation_step}")
        print(f"  🧠 集体记忆: {metrics['memory_analysis']['total_memories']} 条")
        print(f"  🤝 社交行为: {metrics['social_analysis']['total_social_actions']} 次")
        print(f"  📋 任务数量: {metrics['system_status']['tasks']['total']} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 多智能体社会系统快速测试")
    print("=" * 50)
    
    # 基本功能测试
    basic_success = test_basic_functionality()
    
    if basic_success:
        # 演示
        demo_success = run_simple_demo()
        
        if demo_success:
            print("\n✨ 所有测试和演示成功完成！")
            return True
    
    print("\n⚠️  测试过程中遇到问题")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)