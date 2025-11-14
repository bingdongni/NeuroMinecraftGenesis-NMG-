#!/usr/bin/env python3
"""
NeuroMinecraft Genesis环境检查脚本
检查所有必需的依赖和配置文件

NeuroMinecraft Genesis环境检查脚本
检查所有必需的依赖和配置文件
"""

import os
import sys
import subprocess
import importlib.util
import platform
import shutil
from pathlib import Path
import json
from typing import List, Dict, Tuple

class Colors:
    """控制台颜色常量"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE) -> None:
    """打印彩色文本"""
    print(f"{color}{text}{Colors.END}")

def print_header(text: str) -> None:
    """打印标题"""
    print_colored("\n" + "="*60, Colors.BLUE)
    print_colored(f"🔍 {text}", Colors.BOLD + Colors.BLUE)
    print_colored("="*60, Colors.BLUE)

def print_success(text: str) -> None:
    """打印成功信息"""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_error(text: str) -> None:
    """打印错误信息"""
    print_colored(f"❌ {text}", Colors.RED)

def print_warning(text: str) -> None:
    """打印警告信息"""
    print_colored(f"⚠️  {text}", Colors.YELLOW)

def print_info(text: str) -> None:
    """打印信息"""
    print_colored(f"ℹ️  {text}", Colors.CYAN)

class EnvironmentChecker:
    """环境检查器"""
    
    def __init__(self):
        self.results = []
        self.project_root = Path(__file__).parent.absolute()
        
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print_header("Python版本检查")
        
        version = sys.version_info
        print_colored(f"🐍 当前Python版本: {version.major}.{version.minor}.{version.micro}", Colors.WHITE)
        
        if version < (3, 8):
            print_error("Python版本过低，需要Python 3.8+")
            return False
        elif version >= (3, 12):
            print_warning("Python版本较新，某些包可能不完全兼容")
        else:
            print_success("Python版本符合要求")
            
        # 检查是否为虚拟环境
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print_success("检测到虚拟环境")
        else:
            print_warning("建议使用虚拟环境安装依赖")
            
        return True
    
    def check_project_structure(self) -> bool:
        """检查项目结构"""
        print_header("项目结构检查")
        
        required_files = [
            ("requirements.txt", "Python依赖配置"),
            ("quickstart.py", "快速启动脚本"),
            ("utils/brain_engine/six_dimension_brain.py", "六维认知引擎"),
            ("agents/evolution/disco_rl_agent.py", "进化AI智能体"),
            ("utils/quantum_simulator/quantum_brain.py", "量子计算模拟器"),
            ("worlds/integrated_environment.py", "三世界集成系统"),
            ("utils/visualization/streamlit_dashboard.py", "可视化仪表板"),
        ]
        
        optional_files = [
            ("worlds/minecraft/server/paper.jar", "Minecraft服务器核心"),
            (".github/workflows/ci.yml", "GitHub Actions配置"),
            (".github/dependabot.yml", "自动依赖更新"),
        ]
        
        missing_required = []
        missing_optional = []
        
        for file_path, description in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"{description}: {file_path}")
            else:
                print_error(f"缺少必需文件: {description} ({file_path})")
                missing_required.append(file_path)
        
        for file_path, description in optional_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"{description}: {file_path}")
            else:
                print_warning(f"可选文件缺失: {description} ({file_path})")
                missing_optional.append(file_path)
        
        if missing_optional:
            print_info(f"可选文件建议: 考虑添加 {len(missing_optional)} 个文件以获得完整功能")
            
        return len(missing_required) == 0
    
    def check_minecraft_server(self) -> bool:
        """检查Minecraft服务器文件"""
        print_header("Minecraft服务器检查")
        
        server_dir = self.project_root / "worlds" / "minecraft" / "server"
        paper_jar = server_dir / "paper.jar"
        eula_file = server_dir / "eula.txt"
        server_props = server_dir / "server.properties"
        
        if not paper_jar.exists():
            print_error("缺少 paper.jar 文件")
            print_info("📥 下载地址: https://papermc.io/")
            print_info("💡 下载命令:")
            print_colored(
                f'curl -L -o worlds/minecraft/server/paper.jar "'
                'https://api.papermc.io/v2/projects/paper/versions/1.20.1/'
                'builds/latest/downloads/paper-1.20.1-latest.jar"',
                Colors.CYAN
            )
            return False
        else:
            file_size = paper_jar.stat().st_size
            print_success(f"Minecraft服务器文件存在 (大小: {file_size // (1024*1024)} MB)")
        
        if eula_file.exists():
            print_success("EULA协议文件存在")
        else:
            print_warning("EULA文件不存在，服务端可能无法启动")
            
        if server_props.exists():
            print_success("服务器配置文件存在")
        else:
            print_warning("服务器配置文件不存在")
            
        return True
    
    def check_dependencies(self) -> bool:
        """检查Python依赖"""
        print_header("Python依赖检查")
        
        # 核心依赖包列表
        core_packages = [
            'torch', 'numpy', 'pandas', 'matplotlib',
            'streamlit', 'plotly', 'requests', 'websocket',
            'pyyaml', 'click', 'tqdm', 'rich'
        ]
        
        # 高级依赖包列表
        advanced_packages = [
            'qiskit', 'nengo', 'networkx', 'scikit-learn',
            'transformers', 'seaborn', 'bokeh', 'dash',
            'fastapi', 'uvicorn', 'flask'
        ]
        
        # 工具依赖包列表
        utility_packages = [
            'colorama', 'psutil', 'sympy', 'scipy',
            'pillow', 'opencv-python', 'jupyter'
        ]
        
        def check_package_group(packages: List[str], group_name: str) -> Tuple[int, int]:
            missing = []
            available = []
            
            for package in packages:
                try:
                    spec = importlib.util.find_spec(package.replace('-', '_'))
                    if spec is None:
                        missing.append(package)
                        print_error(f"  缺失: {package}")
                    else:
                        available.append(package)
                        print_success(f"  可用: {package}")
                except Exception as e:
                    missing.append(package)
                    print_error(f"  错误: {package} ({str(e)[:50]}...)")
            
            return len(available), len(missing)
        
        print_colored("🔧 检查核心依赖包:", Colors.BLUE)
        core_available, core_missing = check_package_group(core_packages, "核心")
        
        print_colored("\n🧠 检查高级依赖包:", Colors.BLUE)
        advanced_available, advanced_missing = check_package_group(advanced_packages, "高级")
        
        print_colored("\n🛠️ 检查工具依赖包:", Colors.BLUE)
        utility_available, utility_missing = check_package_group(utility_packages, "工具")
        
        total_available = core_available + advanced_available + utility_available
        total_missing = core_missing + advanced_missing + utility_missing
        
        print_header("依赖检查总结")
        print_success(f"已安装: {total_available} 个包")
        if total_missing > 0:
            print_error(f"缺失: {total_missing} 个包")
            print_info(f"💡 安装命令: pip install -r requirements.txt")
            return False
        else:
            print_success("所有依赖包都已安装")
            return True
    
    def check_system_resources(self) -> bool:
        """检查系统资源"""
        print_header("系统资源检查")
        
        # 检查内存
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            print_colored(f"💾 系统内存: {memory_gb:.1f} GB", Colors.WHITE)
            
            if memory_gb >= 8:
                print_success("内存充足，支持完整功能")
            elif memory_gb >= 4:
                print_warning("内存适中，部分功能可能受限")
            else:
                print_error("内存不足，建议至少4GB")
                return False
                
        except ImportError:
            print_warning("无法检查内存 (需要psutil)")
        
        # 检查磁盘空间
        try:
            disk = shutil.disk_usage(self.project_root)
            free_gb = disk.free / (1024**3)
            
            print_colored(f"💿 可用磁盘空间: {free_gb:.1f} GB", Colors.WHITE)
            
            if free_gb >= 10:
                print_success("磁盘空间充足")
            elif free_gb >= 5:
                print_warning("磁盘空间适中")
            else:
                print_error("磁盘空间不足")
                return False
                
        except Exception as e:
            print_warning(f"无法检查磁盘空间: {str(e)}")
        
        # 检查Java (用于Minecraft服务器)
        java_available = shutil.which('java') is not None
        if java_available:
            try:
                result = subprocess.run(['java', '-version'], 
                                      capture_output=True, text=True)
                java_version = result.stderr.split('\n')[0]
                print_success(f"Java已安装: {java_version}")
            except:
                print_warning("Java可能未正确安装")
        else:
            print_warning("Java未安装 (Minecraft服务器需要)")
        
        return True
    
    def check_network_connectivity(self) -> bool:
        """检查网络连接"""
        print_header("网络连接检查")
        
        test_urls = [
            ("https://pypi.org", "PyPI包索引"),
            ("https://github.com", "GitHub"),
            ("https://papermc.io", "PaperMC"),
        ]
        
        for url, name in test_urls:
            try:
                import requests
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print_success(f"{name}: 连接正常")
                else:
                    print_warning(f"{name}: 状态码 {response.status_code}")
            except Exception as e:
                print_error(f"{name}: 连接失败 - {str(e)[:50]}...")
        
        return True
    
    def generate_report(self) -> Dict:
        """生成检查报告"""
        print_header("生成检查报告")
        
        report = {
            "timestamp": subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], 
                                      capture_output=True, text=True).stdout.strip(),
            "system": {
                "platform": platform.platform(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "architecture": platform.machine(),
            },
            "checks": {
                "python_version": "pass" if self.check_python_version() else "fail",
                "project_structure": "pass" if self.check_project_structure() else "fail",
                "minecraft_server": "pass" if self.check_minecraft_server() else "fail",
                "dependencies": "pass" if self.check_dependencies() else "fail",
                "system_resources": "pass" if self.check_system_resources() else "fail",
                "network_connectivity": "pass" if self.check_network_connectivity() else "fail",
            }
        }
        
        # 保存报告
        report_file = self.project_root / "environment_check_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print_success(f"检查报告已保存到: {report_file}")
        except Exception as e:
            print_error(f"无法保存报告: {str(e)}")
        
        return report
    
    def run_all_checks(self) -> bool:
        """运行所有检查"""
        print_colored(
            "\n🚀 NeuroMinecraft Genesis环境检查",
            Colors.BOLD + Colors.BLUE
        )
        print_colored(
            f"项目路径: {self.project_root}",
            Colors.CYAN
        )
        
        checks = [
            self.check_python_version,
            self.check_project_structure,
            self.check_minecraft_server,
            self.check_dependencies,
            self.check_system_resources,
            self.check_network_connectivity,
        ]
        
        failed_checks = []
        for check in checks:
            try:
                if not check():
                    failed_checks.append(check.__name__)
            except Exception as e:
                print_error(f"检查 {check.__name__} 时出错: {str(e)}")
                failed_checks.append(check.__name__)
        
        # 生成报告
        report = self.generate_report()
        
        # 最终总结
        print_header("检查总结")
        
        passed = len(checks) - len(failed_checks)
        total = len(checks)
        
        print_colored(f"检查项目: {total}", Colors.WHITE)
        print_success(f"通过检查: {passed}")
        
        if failed_checks:
            print_error(f"失败检查: {len(failed_checks)}")
            for check_name in failed_checks:
                print_colored(f"  - {check_name}", Colors.RED)
            return False
        else:
            print_success("🎉 所有检查通过！项目可以正常运行")
            return True

def main():
    """主函数"""
    try:
        checker = EnvironmentChecker()
        success = checker.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_colored("\n\n⚠️ 检查被用户中断", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ 检查过程中发生错误: {str(e)}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()