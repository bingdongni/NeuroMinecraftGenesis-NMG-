# Worlds模块

环境世界模块，负责创建和管理AI代理所处的各种环境和世界。

## 子模块

### minecraft/
Minecraft世界集成
- 与Minecraft服务器通信
- 方块世界操作
- 游戏机制集成

#### server/
Minecraft服务器管理
- 服务器启动/停止
- 世界配置文件管理
- 插件系统

### procgen/
程序化生成世界
- 自动地形生成
- 结构自动构建
- 参数化环境创建

### real/
现实世界模拟
- 现实环境映射
- 传感器数据处理
- 真实世界交互接口

## 使用方法

```python
from .minecraft import MinecraftWorld
from .procgen import ProceduralWorld
from .real import RealWorld

# 创建Minecraft世界
minecraft = MinecraftWorld(server_config="server.properties")

# 创建程序化世界
procgen = ProceduralWorld(seed=42, size=1000)
```