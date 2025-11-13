"""
多智能体社会系统测试和演示脚本

运行完整的多智能体社会系统测试，验证所有功能模块
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入系统模块
try:
    # 使用绝对导入
    sys.path.append('/workspace/NeuroMinecraftGenesis/agents/multi')
    from collective_memory import CollectiveMemory, create_danger_zone_memory, create_resource_hotspot_memory, create_blueprint_memory
    from social_cognition import SocialCognitionSystem, IntentionType, TrustLevel, SocialAction, create_intention
    from collaboration_protocol import CollaborationProtocol, Task, TaskType, TaskPriority, Resource, ResourceType, Conflict, DecisionProposal
    from tribal_society import TribalSociety, AgentPersonality, AgentState, create_tribal_society_with_config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'multi_agent_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class MultiAgentSystemTester:
    """多智能体系统测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始多智能体社会系统全面测试")
        
        test_suites = [
            ("集体记忆系统测试", self.test_collective_memory),
            ("社会认知系统测试", self.test_social_cognition),
            ("协作协议系统测试", self.test_collaboration_protocol),
            ("部落社会系统测试", self.test_tribal_society),
            ("集成功能测试", self.test_integration),
            ("性能压力测试", self.test_performance),
            ("演示模式", self.run_demo)
        ]
        
        for test_name, test_func in test_suites:
            try:
                logger.info(f"执行测试: {test_name}")
                start_time = time.time()
                result = test_func()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    "success": True,
                    "duration": end_time - start_time,
                    "details": result
                }
                logger.info(f"✅ {test_name} 完成 ({end_time - start_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ {test_name} 失败: {e}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time
                }
        
        # 生成测试报告
        self.generate_test_report()
        
        return self.test_results
    
    def test_collective_memory(self) -> dict:
        """测试集体记忆系统"""
        logger.info("测试集体记忆系统...")
        
        # 创建记忆系统
        memory_system = CollectiveMemory(memory_capacity=100)
        
        # 测试基本功能
        results = {}
        
        # 1. 创建和存储记忆
        danger_memory = create_danger_zone_memory(100, 0, 50, "creeper", "苦力怕密集区域", "test_agent")
        memory_id = memory_system.store_memory(danger_memory)
        results["memory_storage"] = memory_id is not None
        
        # 2. 检索记忆
        retrieved_memories = memory_system.retrieve_memories(memory_type="danger_zone", limit=10)
        results["memory_retrieval"] = len(retrieved_memories) > 0
        
        # 3. 验证记忆
        memory_system.verify_memory(memory_id, "test_agent", 0.8)
        retrieved = memory_system.memory_store.get(memory_id)
        results["memory_verification"] = retrieved is not None and retrieved.reliability_score > 0.5
        
        # 4. 资源热点功能
        resource_memory = create_resource_hotspot_memory(0, 5, 0, "iron", "abundant", 0.9, "test_agent")
        memory_system.store_memory(resource_memory)
        
        hotspots = memory_system.get_resource_hotspots("iron", limit=5)
        results["resource_hotspots"] = len(hotspots) > 0
        
        # 5. 记忆融合
        memories_to_merge = [danger_memory, resource_memory]
        merged_memory = memory_system.merge_knowledge(memories_to_merge)
        results["memory_merging"] = merged_memory is not None
        
        # 6. 统计信息
        stats = memory_system.get_memory_statistics()
        results["statistics"] = stats["total_memories"] > 0
        
        logger.info(f"集体记忆系统测试完成: {sum(results.values())}/{len(results)} 项通过")
        return results
    
    def test_social_cognition(self) -> dict:
        """测试社会认知系统"""
        logger.info("测试社会认知系统...")
        
        # 创建社会认知系统
        social_system = SocialCognitionSystem(agent_count=8)
        
        results = {}
        
        # 1. 记录社会行为
        action1 = SocialAction(
            actor_id="agent_0",
            action_type="help",
            target_id="agent_1",
            timestamp=datetime.now(),
            success=True,
            impact_score=0.7,
            description="提供帮助",
            context={}
        )
        social_system.record_social_action(action1)
        results["action_recording"] = len(social_system.social_actions) > 0
        
        # 2. 意图分析
        intentions = social_system.analyze_intentions("agent_0", observation_window=10)
        results["intention_analysis"] = len(intentions) == len(IntentionType)
        
        # 3. 信任模型
        trust_model = social_system.build_trust_model("agent_0")
        results["trust_model"] = isinstance(trust_model, dict)
        
        # 4. 智能体推荐
        recommendations = social_system.get_social_recommendations("agent_0", "collaboration")
        results["social_recommendations"] = isinstance(recommendations, list)
        
        # 5. 社会学习
        learning_success = social_system.initiate_social_learning("agent_0", "agent_1", "exploration", "observation")
        results["social_learning"] = isinstance(learning_success, bool)
        
        # 6. 领导选举
        leader = social_system.elect_leader("balanced")
        results["leadership_election"] = leader.startswith("agent_")
        
        # 7. 社交档案
        profile = social_system.get_agent_social_profile("agent_0")
        results["social_profile"] = "agent_id" in profile and "social_activity" in profile
        
        # 8. 网络分析
        network_analysis = social_system.analyze_social_network()
        results["network_analysis"] = "network_density" in network_analysis
        
        logger.info(f"社会认知系统测试完成: {sum(results.values())}/{len(results)} 项通过")
        return results
    
    def test_collaboration_protocol(self) -> dict:
        """测试协作协议系统"""
        logger.info("测试协作协议系统...")
        
        # 创建协作协议系统
        collab_system = CollaborationProtocol(agent_count=8)
        
        results = {}
        
        # 1. 创建任务
        task = Task(
            id="",
            task_type=TaskType.EXPLORATION,
            title="测试探索任务",
            description="用于测试的探索任务",
            priority=TaskPriority.HIGH,
            estimated_duration=6,
            required_skills=["exploration"],
            required_resources={ResourceType.MATERIAL: 5},
            created_by="test_system"
        )
        
        task_id = collab_system.create_task(task)
        results["task_creation"] = task_id is not None and task_id in collab_system.tasks
        
        # 2. 任务分配
        assignment_success = collab_system.assign_task(task_id, "agent_0")
        results["task_assignment"] = isinstance(assignment_success, bool)
        
        # 3. 任务推荐
        recommendations = collab_system.get_task_recommendations("agent_0", limit=5)
        results["task_recommendations"] = isinstance(recommendations, list)
        
        # 4. 创建资源
        resource = Resource(
            id="",
            resource_type=ResourceType.MATERIAL,
            name="测试材料",
            quantity=50,
            quality=0.8,
            shared=True,
            accessibility="medium"
        )
        
        resource_id = collab_system.create_resource(resource)
        results["resource_creation"] = resource_id is not None and resource_id in collab_system.resources
        
        # 5. 资源分享
        share_success = collab_system.share_resource(resource_id, "agent_0", "agent_1", 10)
        results["resource_sharing"] = isinstance(share_success, bool)
        
        # 6. 冲突解决
        conflict_data = {
            "id": "",
            "conflict_type": "resource_competition",
            "description": "资源竞争冲突",
            "involved_agents": ["agent_0", "agent_1"],
            "timestamp": datetime.now(),
            "severity": 3
        }
        
        from collaboration_protocol import Conflict
        conflict = Conflict(**conflict_data)
        conflict_id = collab_system.resolve_conflict(conflict)
        results["conflict_resolution"] = conflict_id in collab_system.conflicts
        
        # 7. 决策提案
        decision_data = {
            "id": "",
            "decision_type": "resource_allocation",
            "title": "资源分配决策",
            "description": "测试资源分配决策",
            "proposer_id": "agent_0",
            "timestamp": datetime.now(),
            "arguments": {"agent_0": ["优化资源使用"]},
            "voting_deadline": datetime.now() + timedelta(hours=1),
            "required_quorum": 3,
            "decision_threshold": 0.6
        }
        
        from collaboration_protocol import DecisionProposal
        decision = DecisionProposal(**decision_data)
        decision_id = collab_system.propose_decision(decision)
        results["decision_proposal"] = decision_id in collab_system.decisions
        
        # 8. 投票
        vote_success = collab_system.cast_vote(decision_id, "agent_1", 1)
        results["voting"] = isinstance(vote_success, bool)
        
        # 9. 协作指标
        metrics = collab_system.get_collaboration_metrics()
        results["collaboration_metrics"] = "task_completion_rate" in metrics
        
        logger.info(f"协作协议系统测试完成: {sum(results.values())}/{len(results)} 项通过")
        return results
    
    def test_tribal_society(self) -> dict:
        """测试部落社会系统"""
        logger.info("测试部落社会系统...")
        
        # 创建小型部落进行测试
        tribe = TribalSociety(agent_count=4)
        
        results = {}
        
        # 1. 部落初始化
        results["tribe_initialization"] = len(tribe.agents) == 4
        
        # 2. 智能体特征
        agent = list(tribe.agents.values())[0]
        results["agent_characteristics"] = (
            hasattr(agent, 'personality') and 
            hasattr(agent, 'energy_level') and 
            hasattr(agent, 'personal_goals')
        )
        
        # 3. 社交网络
        results["social_network"] = len(tribe.social_network) > 0
        
        # 4. 集体记忆初始化
        memory_stats = tribe.collective_memory.get_memory_statistics()
        results["collective_memory_init"] = memory_stats["total_memories"] > 0
        
        # 5. 模拟一步执行
        tribe.is_running = True
        tribe._execute_simulation_step()
        tribe.is_running = False
        results["simulation_step"] = tribe.simulation_step > 0
        
        # 6. 部落状态检查
        status = tribe.get_tribal_status()
        results["tribal_status"] = (
            "basic_info" in status and 
            "collective_metrics" in status and 
            "agent_overview" in status
        )
        
        # 7. 协作集成
        collab_status = tribe.collaboration_protocol.get_system_status()
        results["collaboration_integration"] = "tasks" in collab_status
        
        # 8. 社会认知集成
        social_stats = tribe.social_cognition.get_system_statistics()
        results["social_integration"] = "total_social_actions" in social_stats
        
        # 9. 集体智能计算
        results["collective_intelligence"] = len(tribe.collective_intelligence_metrics) > 0
        
        # 10. 数据导出
        try:
            export_path = f"/tmp/test_tribe_{int(time.time())}.json"
            tribe.export_simulation_data(export_path)
            results["data_export"] = os.path.exists(export_path)
            if os.path.exists(export_path):
                os.remove(export_path)
        except Exception as e:
            logger.warning(f"数据导出测试失败: {e}")
            results["data_export"] = False
        
        logger.info(f"部落社会系统测试完成: {sum(results.values())}/{len(results)} 项通过")
        return results
    
    def test_integration(self) -> dict:
        """测试系统集成"""
        logger.info("测试系统集成...")
        
        # 创建完整系统
        tribe = TribalSociety(agent_count=6)
        
        results = {}
        
        # 1. 系统组件集成
        results["component_integration"] = (
            tribe.collective_memory is not None and
            tribe.social_cognition is not None and
            tribe.collaboration_protocol is not None
        )
        
        # 2. 模拟多步执行
        tribe.is_running = True
        for i in range(5):
            tribe._execute_simulation_step()
            tribe.simulation_step += 1
            tribe.simulation_time += timedelta(hours=1)
        
        tribe.is_running = False
        results["multi_step_simulation"] = tribe.simulation_step >= 5
        
        # 3. 跨系统数据流
        # 检查集体记忆是否有新数据
        memory_before = len(tribe.collective_memory.memory_store)
        
        # 执行更多模拟步骤
        for i in range(3):
            tribe._execute_simulation_step()
            tribe.simulation_step += 1
            tribe.simulation_time += timedelta(hours=1)
        
        memory_after = len(tribe.collective_memory.memory_store)
        results["memory_data_flow"] = memory_after >= memory_before
        
        # 4. 社会交互影响
        actions_before = len(tribe.social_cognition.social_actions)
        
        # 触发社会交互
        tribe._process_social_interactions()
        
        actions_after = len(tribe.social_cognition.social_actions)
        results["social_interaction_flow"] = actions_after >= actions_before
        
        # 5. 任务执行影响
        tasks_before = len(tribe.collaboration_protocol.tasks)
        
        # 处理任务执行
        tribe._process_task_execution()
        
        tasks_after = len(tribe.collaboration_protocol.tasks)
        results["task_execution_flow"] = tasks_after >= tasks_before
        
        # 6. 集体智能演化
        intelligence_before = tribe.collective_intelligence_metrics.get("collective_intelligence", 0)
        
        # 执行更多步骤计算集体智能
        tribe._calculate_collective_intelligence()
        
        intelligence_after = tribe.collective_intelligence_metrics.get("collective_intelligence", 0)
        results["intelligence_evolution"] = isinstance(intelligence_after, float)
        
        # 7. 系统状态一致性
        status = tribe.get_tribal_status()
        results["system_consistency"] = (
            status["basic_info"]["agent_count"] == 6 and
            status["basic_info"]["simulation_step"] >= 8
        )
        
        logger.info(f"系统集成测试完成: {sum(results.values())}/{len(results)} 项通过")
        return results
    
    def test_performance(self) -> dict:
        """测试系统性能"""
        logger.info("测试系统性能...")
        
        results = {}
        
        # 性能测试参数
        agent_counts = [8, 16, 24]
        performance_results = {}
        
        for count in agent_counts:
            logger.info(f"测试 {count} 个智能体的性能")
            
            # 创建系统
            tribe = TribalSociety(agent_count=count)
            
            # 测量初始化时间
            init_start = time.time()
            # 部落已经在初始化中，这里测量总时间
            
            # 测量模拟执行时间
            simulation_start = time.time()
            tribe.is_running = True
            tribe._execute_simulation_step()
            tribe.is_running = False
            simulation_end = time.time()
            
            simulation_time = simulation_end - simulation_start
            
            performance_results[f"{count}_agents"] = {
                "simulation_step_time": simulation_time,
                "memory_entries": len(tribe.collective_memory.memory_store),
                "social_actions": len(tribe.social_cognition.social_actions),
                "tasks": len(tribe.collaboration_protocol.tasks)
            }
            
            logger.info(f"{count} 智能体 - 模拟步执行时间: {simulation_time:.4f}s")
        
        # 分析性能结果
        results["performance_test"] = performance_results
        
        # 检查性能是否在合理范围内
        base_time = performance_results["8_agents"]["simulation_step_time"]
        scaling_factor = performance_results["16_agents"]["simulation_step_time"] / base_time
        results["reasonable_scaling"] = scaling_factor < 3.0  # 16个智能体的时间不应超过8个的3倍
        
        logger.info(f"性能测试完成: {results['reasonable_scaling']}")
        return results
    
    def run_demo(self) -> dict:
        """运行演示"""
        logger.info("运行演示模式...")
        
        # 创建演示部落
        tribe = TribalSociety(agent_count=8)
        
        # 运行短期演示
        demo_duration = 10  # 10个模拟步骤
        logger.info(f"运行 {demo_duration} 步演示模拟")
        
        tribe.is_running = True
        for step in range(demo_duration):
            logger.info(f"执行模拟步骤 {step + 1}/{demo_duration}")
            tribe._execute_simulation_step()
            tribe.simulation_step += 1
            tribe.simulation_time += timedelta(hours=1)
            
            # 每3步输出一次状态
            if (step + 1) % 3 == 0:
                status = tribe.get_tribal_status()
                logger.info(f"步骤 {step + 1} 状态 - 智能体活跃: {status['agent_overview']}")
        
        tribe.is_running = False
        
        # 生成演示结果
        demo_results = {
            "simulation_steps": demo_duration,
            "final_collective_intelligence": tribe.collective_intelligence_metrics.get("collective_intelligence", 0),
            "final_metrics": tribe.collaboration_protocol.get_collaboration_metrics(),
            "social_network_status": tribe.social_cognition.get_system_statistics(),
            "memory_utilization": tribe.collective_memory.get_memory_statistics()
        }
        
        # 导出演示数据
        demo_path = f"/tmp/demo_results_{int(time.time())}.json"
        tribe.export_simulation_data(demo_path)
        
        # 生成演示报告
        report_path = tribe.generate_analysis_report(demo_path)
        
        demo_results["data_exported"] = os.path.exists(demo_path)
        demo_results["report_generated"] = os.path.exists(report_path)
        
        logger.info(f"演示完成 - 最终集体智能: {demo_results['final_collective_intelligence']:.3f}")
        return demo_results
    
    def generate_test_report(self):
        """生成测试报告"""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        # 生成报告内容
        report = f"""# 多智能体社会系统测试报告

## 测试概况
- 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {total_duration.total_seconds():.2f} 秒
- 总测试数: {total_tests}
- 通过: {passed_tests} ✅
- 失败: {failed_tests} ❌
- 成功率: {passed_tests/total_tests*100:.1f}%

## 详细测试结果

"""
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result["success"] else "❌ 失败"
            duration = result.get("duration", 0)
            
            report += f"### {test_name}\n"
            report += f"- 状态: {status}\n"
            report += f"- 耗时: {duration:.2f} 秒\n"
            
            if not result["success"]:
                report += f"- 错误: {result.get('error', '未知错误')}\n"
            elif "details" in result:
                if isinstance(result["details"], dict):
                    if all(isinstance(v, bool) for v in result["details"].values()):
                        # 布尔值详细结果
                        passed = sum(result["details"].values())
                        total = len(result["details"])
                        report += f"- 子测试: {passed}/{total} 通过\n"
                    else:
                        # 其他详细结果
                        report += f"- 详情: {result['details']}\n"
            
            report += "\n"
        
        # 保存报告
        report_path = f"/workspace/NeuroMinecraftGenesis/agents/multi/test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON格式结果
        json_path = f"/workspace/NeuroMinecraftGenesis/agents/multi/test_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📊 测试报告已生成:")
        print(f"   📄 详细报告: {report_path}")
        print(f"   📊 JSON数据: {json_path}")
        
        return report_path


def main():
    """主函数"""
    print("🤖 多智能体社会系统测试")
    print("=" * 50)
    
    # 创建测试器
    tester = MultiAgentSystemTester()
    
    # 运行所有测试
    try:
        results = tester.run_all_tests()
        
        # 显示总结
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["success"])
        
        print(f"\n🎯 测试总结:")
        print(f"   总计: {total_tests} 个测试")
        print(f"   通过: {passed_tests} ✅")
        print(f"   失败: {total_tests - passed_tests} ❌")
        print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！多智能体社会系统运行正常。")
            return True
        else:
            print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查日志。")
            return False
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)