# Data模块

数据存储模块，管理项目产生的所有数据文件。

## 子模块

### minecraft_episodes/
Minecraft游戏记录
- 代理游戏过程的完整记录
- 游戏状态数据
- 行为轨迹文件

### brain_scans/
脑扫描数据
- 神经网络可视化数据
- 脑活动模式记录
- 认知状态快照

### evolution_logs/
进化日志
- 进化过程详细记录
- 适者生存日志
- 变异历史追踪

## 数据格式

```python
# 数据访问示例
from .minecraft_episodes import EpisodeReader
from .brain_scans import BrainScanData
from .evolution_logs import EvolutionLogger

# 读取游戏记录
episodes = EpisodeReader.load_episode("episode_001.json")

# 访问脑扫描数据
scan = BrainScanData.load("brain_scan_001.h5")
```