#!/usr/bin/env python3
"""
多智能体系统完整单元测试
"""

import pytest
import numpy as np
import asyncio


class TestCollaborationProtocol:
    """协作协议测试类"""

    def test_initialization(self):
        """测试协作协议初始化"""
        from agents.multi.collaboration_protocol import CollaborationProtocol

        protocol = CollaborationProtocol()

        assert protocol is not None

    def test_message_creation(self):
        """测试消息创建"""
        from agents.multi.collaboration_protocol import Message

        message = Message(
            sender="agent_1",
            receiver="agent_2",
            content="Hello",
            msg_type="request"
        )

        assert message is not None
        assert message.sender == "agent_1"

    def test_protocol_handshake(self):
        """测试协议握手"""
        from agents.multi.collaboration_protocol import CollaborationProtocol

        protocol = CollaborationProtocol()

        result = asyncio.run(protocol.perform_handshake("agent_1", "agent_2"))

        assert result is not None


class TestCollectiveMemory:
    """集体记忆测试类"""

    def test_initialization(self):
        """测试集体记忆初始化"""
        from agents.multi.collective_memory import CollectiveMemory

        memory = CollectiveMemory()

        assert memory is not None

    def test_memory_storage(self):
        """测试记忆存储"""
        from agents.multi.collective_memory import CollectiveMemory

        memory = CollectiveMemory()

        # 存储
        key = memory.store(
            agent_id="agent_1",
            content="Shared knowledge"
        )

        assert key is not None

    def test_memory_retrieval(self):
        """测试记忆检索"""
        from agents.multi.collective_memory import CollectiveMemory

        memory = CollectiveMemory()

        # 存储
        memory.store(agent_id="agent_1", content="Test knowledge")

        # 检索
        results = memory.retrieve("test")

        assert results is not None


class TestSocialCognition:
    """社交认知测试类"""

    def test_initialization(self):
        """测试社交认知初始化"""
        from agents.multi.social_cognition import SocialCognition

        cognition = SocialCognition()

        assert cognition is not None

    def test_belief_attribution(self):
        """测试信念归因"""
        from agents.multi.social_cognition import SocialCognition

        cognition = SocialCognition()

        belief = cognition.attribute_belief(
            agent_id="agent_1",
            observation={"action": "helping"}
        )

        assert belief is not None

    def test_intention_recognition(self):
        """测试意图识别"""
        from agents.multi.social_cognition import SocialCognition

        cognition = SocialCognition()

        intention = cognition.recognize_intention(
            agent_id="agent_1",
            actions=["gather", "build", "share"]
        )

        assert intention is not None


class TestMultiAgentSystem:
    """多智能体系统测试类"""

    def test_initialization(self):
        """测试多智能体系统初始化"""
        from agents.multi.tribal_society import TribalSociety

        system = TribalSociety(num_agents=5)

        assert system is not None
        assert system.num_agents == 5

    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """测试智能体创建"""
        from agents.multi.tribal_society import TribalSociety

        system = TribalSociety(num_agents=3)

        # 创建智能体
        system.create_agents()

        assert len(system.agents) == 3

    @pytest.mark.asyncio
    async def test_inter_agent_communication(self):
        """测试智能体间通信"""
        from agents.multi.tribal_society import TribalSociety

        system = TribalSociety(num_agents=3)
        system.create_agents()

        # 发送消息
        await system.send_message(
            sender="agent_0",
            receiver="agent_1",
            message="Hello"
        )

        # 验证消息发送
        assert system.message_log is not None


class TestTribalSociety:
    """部落社会测试类"""

    def test_initialization(self):
        """测试部落社会初始化"""
        from agents.multi.tribal_society import TribalSociety

        society = TribalSociety(num_agents=10)

        assert society is not None
        assert society.num_agents == 10

    def test_culture_formation(self):
        """测试文化形成"""
        from agents.multi.tribal_society import TribalSociety

        society = TribalSociety(num_agents=10)
        society.create_agents()

        # 形成文化
        culture = society.form_culture(iterations=5)

        assert culture is not None

    def test_knowledge_sharing(self):
        """测试知识共享"""
        from agents.multi.tribal_society import TribalSociety

        society = TribalSociety(num_agents=5)
        society.create_agents()

        # 共享知识
        result = society.share_knowledge()

        assert result is not None

    def test_social_hierarchy(self):
        """测试社会层级"""
        from agents.multi.tribal_society import TribalSociety

        society = TribalSociety(num_agents=10)
        society.create_agents()

        # 建立层级
        hierarchy = society.establish_hierarchy()

        assert hierarchy is not None


class TestMassEvolution:
    """大规模进化测试类"""

    def test_initialization(self):
        """测试大规模进化初始化"""
        from agents.mass_evolution.multi_agent_society import MultiAgentSociety

        society = MultiAgentSociety(num_agents=100)

        assert society is not None
        assert society.num_agents == 100

    def test_population_dynamics(self):
        """测试种群动态"""
        from agents.mass_evolution.multi_agent_society import MultiAgentSociety

        society = MultiAgentSociety(num_agents=50)

        # 运行动态
        society.evolve(generations=3)

        assert society.generation > 0

    def test_competition_mechanism(self):
        """测试竞争机制"""
        from agents.mass_evolution.multi_agent_society import MultiAgentSociety

        society = MultiAgentSociety(num_agents=30)

        # 运行竞争
        survivors = society.run_competition()

        assert survivors is not None
        assert len(survivors) < 30

    def test_evolution_tracking(self):
        """测试进化追踪"""
        from agents.mass_evolution.multi_agent_society import MultiAgentSociety

        society = MultiAgentSociety(num_agents=20)

        # 运行几代
        for _ in range(3):
            society.evolve(generations=1)

        # 获取历史
        history = society.get_evolution_history()

        assert history is not None
        assert len(history) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
