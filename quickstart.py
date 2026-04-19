"""快速开始示例"""

from NeuroMinecraftGenesis import SingleAgent, MinecraftWorld

def main():
    # 创建Minecraft世界
    world = MinecraftWorld()
    
    # 创建AI代理
    agent = SingleAgent()
    
    # 运行代理
    agent.run_in_world(world)

if __name__ == "__main__":
    main()
