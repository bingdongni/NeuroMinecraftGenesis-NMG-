#!/usr/bin/env python3
"""
海马体记忆系统演示脚本
展示完整的功能特性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'brain'))

from hippocampus import *

def demonstrate_hippocampus_system():
    """演示海马体记忆系统的完整功能"""
    
    print("🧠 海马体记忆系统功能演示")
    print("=" * 60)
    
    # 创建系统实例
    memory_system = HippocampusMemorySystem(
        max_memory_size=500,
        embedding_dim=64,
        consolidation_hour=22
    )
    
    # 1. 存储多样化的记忆
    print("\n1. 📝 存储多样化记忆")
    
    memories_data = [
        ("学会了Python编程", "semantic", 0.8, 0.6, False),
        ("完成了一个重要项目", "episodic", 0.9, 0.7, True),
        ("和朋友们聚餐很开心", "episodic", 0.7, 0.8, False),
        ("发现了新的算法", "creative", 0.8, 0.6, True),
        ("工作中遇到挫折", "episodic", -0.3, -0.2, False),
        ("掌握了机器学习", "semantic", 0.9, 0.7, False),
        ("创造了有趣的应用", "creative", 0.9, 0.8, True),
        ("团队协作成功", "episodic", 0.8, 0.7, False),
        ("学习了深度学习", "semantic", 0.8, 0.6, False),
        ("解决了难题很兴奋", "creative", 0.8, 0.9, True)
    ]
    
    stored_ids = []
    for i, (content, mem_type, reward, emotion, creativity) in enumerate(memories_data):
        memory_id = memory_system.store_memory(
            content=content,
            memory_type=mem_type,
            reward_value=reward,
            emotional_valence=emotion,
            creativity_flag=creativity
        )
        stored_ids.append(memory_id)
        print(f"   {i+1:2d}. {content}")
    
    # 等待异步处理
    time.sleep(2)
    
    # 2. 概念形成演示
    print(f"\n2. 🧠 概念形成演示")
    
    # 触发概念形成
    programming_ids = stored_ids[:3]  # 选择编程相关记忆
    concepts = memory_system.form_concepts_from_memories(programming_ids)
    
    print(f"   从 {len(programming_ids)} 个编程记忆形成概念:")
    for concept_id in concepts:
        concept = memory_system.concepts[concept_id]
        print(f"   - {concept.name}")
        print(f"     定义: {concept.definition}")
        print(f"     属性: {', '.join(list(concept.attributes)[:3])}")
        print(f"     置信度: {concept.confidence_score:.3f}")
    
    # 3. 知识蒸馏演示
    print(f"\n3. 🎯 知识蒸馏演示")
    
    # 选择多个记忆进行蒸馏
    multiple_ids = stored_ids[:5]
    knowledge_id = memory_system.distill_knowledge(multiple_ids)
    
    if knowledge_id:
        knowledge = memory_system.distilled_knowledge[knowledge_id]
        print(f"   蒸馏知识成功:")
        print(f"     ID: {knowledge.knowledge_id[:8]}...")
        print(f"     压缩比: {knowledge.compression_ratio:.2f}")
        print(f"     保真度: {knowledge.fidelity_score:.3f}")
        print(f"     质量分数: {knowledge.quality_score:.3f}")
        print(f"     关键特征数: {len(knowledge.key_features)}")
        
        # 显示关键特征
        print("     主要特征:")
        for feature, value in list(knowledge.key_features.items())[:3]:
            print(f"       - {feature}: {value:.3f}")
    else:
        print("   知识蒸馏失败")
    
    # 4. 语义网络演示
    print(f"\n4. 🌐 语义网络演示")
    
    # 构建语义网络
    memory_system.build_semantic_network()
    
    print(f"   语义网络统计:")
    print(f"     概念数: {len(memory_system.concepts)}")
    print(f"     关联数: {sum(len(connections) for connections in memory_system.semantic_network.edges.values()) // 2}")
    
    # 展示概念关系
    if memory_system.concepts:
        first_concept = list(memory_system.concepts.values())[0]
        relationships = memory_system.find_semantic_relationships(first_concept.concept_id)
        
        print(f"   概念 '{first_concept.name}' 的语义关系:")
        for rel in relationships[:3]:
            print(f"     - {rel['description']}")
    
    # 5. 记忆检索演示
    print(f"\n5. 🔍 记忆检索演示")
    
    # 精确检索
    query_results = memory_system.retrieve_memories("编程学习", top_k=3)
    print(f"   精确检索 '编程学习': {len(query_results)} 个结果")
    for i, result in enumerate(query_results):
        memory = result['memory']
        print(f"     {i+1}. {memory.content}")
        print(f"        相似度: {result['similarity_score']:.3f}")
        print(f"        关联记忆数: {len(result['related_memories'])}")
    
    # 6. 长期巩固演示
    print(f"\n6. 💾 长期巩固演示")
    
    # 强制执行巩固
    consolidation_result = memory_system.consolidate_memories(force=True)
    
    print(f"   巩固结果:")
    print(f"     状态: {consolidation_result['status']}")
    print(f"     巩固记忆数: {consolidation_result['consolidated_memories']}")
    print(f"     遗忘记忆数: {consolidation_result['forgotten_memories']}")
    print(f"     新概念数: {consolidation_result['new_concepts']}")
    print(f"     新蒸馏知识数: {consolidation_result['new_distilled_knowledge']}")
    print(f"     处理时间: {consolidation_result['processing_time']:.3f}秒")
    
    # 7. 系统统计演示
    print(f"\n7. 📊 系统统计演示")
    
    stats = memory_system.get_memory_statistics()
    
    print(f"   记忆概览:")
    print(f"     总记忆数: {stats['memory_overview']['total_memories']}")
    print(f"     容量使用率: {stats['memory_overview']['memory_capacity_usage']:.1%}")
    print(f"     工作记忆大小: {stats['memory_overview']['working_memory_size']}")
    
    print(f"   记忆分布:")
    for mem_type, count in stats['memory_distribution']['by_type'].items():
        print(f"     {mem_type}: {count}")
    
    print(f"   概念统计:")
    print(f"     概念数: {stats['conceptual_stats']['total_concepts']}")
    print(f"     形成概念数: {stats['conceptual_stats']['concepts_formed']}")
    print(f"     语义关联数: {stats['conceptual_stats']['semantic_network']['total_associations']}")
    
    print(f"   知识蒸馏统计:")
    print(f"     蒸馏知识数: {stats['knowledge_stats']['total_distilled_knowledge']}")
    print(f"     平均压缩比: {stats['knowledge_stats']['avg_compression_ratio']:.2f}")
    print(f"     平均质量分: {stats['knowledge_stats']['avg_quality_score']:.3f}")
    
    print(f"   性能统计:")
    print(f"     检索准确率: {stats['performance_stats']['retrieval_accuracy']:.1%}")
    print(f"     成功检索: {stats['performance_stats']['successful_retrievals']}")
    print(f"     失败检索: {stats['performance_stats']['failed_retrievals']}")
    
    # 8. 数据持久化演示
    print(f"\n8. 💾 数据持久化演示")
    
    # 导出记忆状态
    export_file = "memory_state.json"
    memory_system.export_memory_state(export_file)
    print(f"   记忆状态已导出到: {export_file}")
    
    # 检查文件大小
    if os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"   文件大小: {file_size} 字节")
    
    # 清理资源
    memory_system.cleanup()
    print(f"   资源清理完成")
    
    print(f"\n" + "=" * 60)
    print(f"✅ 海马体记忆系统功能演示完成!")
    print(f"")
    print(f"🎯 核心功能验证:")
    print(f"   ✅ 概念形成和抽象化机制")
    print(f"   ✅ 知识蒸馏和压缩存储")
    print(f"   ✅ 语义记忆网络")
    print(f"   ✅ 记忆提取和关联")
    print(f"   ✅ 长期记忆巩固")
    print(f"")
    print(f"🚀 系统已准备好集成到更大的AI应用中!")
    print(f"=" * 60)
    
    return memory_system

if __name__ == "__main__":
    demonstrate_hippocampus_system()