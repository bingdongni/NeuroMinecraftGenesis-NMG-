#!/bin/bash
# NeuroMinecraft Genesis 自动GitHub上传脚本
# 开发者：bingdongni

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置信息 (请修改这些值)
GITHUB_USERNAME="your-username"
REPOSITORY_NAME="NeuroMinecraft-Genesis"
GITHUB_TOKEN="your-github-token"  # 如果需要的话

echo -e "${BLUE}"
echo "████████████████████████████████████████████████████████████████"
echo "█                                                              █"
echo "█         NeuroMinecraft Genesis 自动上传脚本                █"
echo "█                    开发者: bingdongni                        █"
echo "█                                                              █"
echo "████████████████████████████████████████████████████████████████"
echo -e "${NC}"

echo ""
echo -e "${BLUE}🚀 开始自动上传到 GitHub...${NC}"
echo ""

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git 未安装，请先安装 Git${NC}"
    exit 1
fi

# 检查是否为git仓库
if [ ! -d ".git" ]; then
    echo -e "${BLUE}📦 初始化Git仓库...${NC}"
    git init
else
    echo -e "${GREEN}✅ Git仓库已存在${NC}"
fi

# 添加GitHub远程仓库
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPOSITORY_NAME}.git"

# 检查是否已经添加远程仓库
if git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}⚠️  远程仓库已配置${NC}"
    echo -e "${BLUE}📝 更新远程仓库地址...${NC}"
    git remote set-url origin "${REMOTE_URL}"
else
    echo -e "${BLUE}🔗 添加远程仓库...${NC}"
    git remote add origin "${REMOTE_URL}"
fi

echo ""
echo -e "${BLUE}📋 准备上传文件...${NC}"

# 检查重要文件
important_files=("README.md" "LICENSE" "requirements.txt" "quickstart.py")
for file in "${important_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "   ✅ $file"
    else
        echo -e "   ❌ $file 不存在"
    fi
done

echo ""
echo -e "${BLUE}🧹 清理临时文件...${NC}"

# 删除Python缓存
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 删除系统文件
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

echo ""
echo -e "${BLUE}📊 检查文件状态...${NC}"

# 显示将要上传的文件
echo "📁 文件清单:"
git status --porcelain

echo ""
echo -e "${BLUE}🎯 添加文件到Git...${NC}"

# 添加所有文件
git add .

# 检查是否有变更
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  没有新的变更需要提交${NC}"
else
    echo -e "${GREEN}✅ 文件已添加到暂存区${NC}"
fi

echo ""
echo -e "${BLUE}💬 创建提交...${NC}"

# 创建提交
COMMIT_MESSAGE="Initial commit: NeuroMinecraft Genesis v1.0 - AGI自主进化系统

✨ 特性:
- 🧠 六维认知引擎
- 🔬 DiscoRL算法发现
- ⚛️ 量子-类脑融合
- 🌍 三世界集成
- 🤝 多智能体协同
- 📚 终身学习系统

🚀 开发者: bingdongni
📅 发布日期: $(date +%Y-%m-%d)"

git commit -m "${COMMIT_MESSAGE}"

echo ""
echo -e "${BLUE}🌿 设置主分支...${NC}"

# 重命名为main分支
git branch -M main

echo ""
echo -e "${BLUE}🚀 推送到GitHub...${NC}"

# 检查是否需要认证
if [ -n "$GITHUB_TOKEN" ]; then
    echo -e "${GREEN}🔑 使用GitHub Token认证${NC}"
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPOSITORY_NAME}.git"
fi

# 推送到GitHub
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}✅ 上传成功！${NC}"
    echo ""
    echo -e "${YELLOW}🎉 您的项目已经成功上传到 GitHub！${NC}"
    echo ""
    echo -e "${BLUE}📝 仓库地址:${NC}"
    echo "   https://github.com/${GITHUB_USERNAME}/${REPOSITORY_NAME}"
    echo ""
    echo -e "${BLUE}📋 后续建议:${NC}"
    echo "   1. 设置仓库描述和标签"
    echo "   2. 完善 README.md 内容"
    echo "   3. 创建 GitHub Pages (可选)"
    echo "   4. 添加贡献指南 (CONTRIBUTING.md)"
    echo "   5. 设置Issues模板"
    echo "   6. 创建第一个Release"
    echo ""
    echo -e "${BLUE}📚 更多详细信息请查看:${NC}"
    echo "   - GitHub上传详细指南.md"
    echo ""
else
    echo -e "${RED}❌ 上传失败${NC}"
    echo ""
    echo -e "${YELLOW}🔧 可能的解决方案:${NC}"
    echo "   1. 检查网络连接"
    echo "   2. 确认GitHub仓库名称正确"
    echo "   3. 检查GitHub认证信息"
    echo "   4. 尝试手动上传:"
    echo "      git push origin main"
    echo ""
fi

echo ""
echo -e "${GREEN}🏁 脚本执行完成${NC}"
echo ""
