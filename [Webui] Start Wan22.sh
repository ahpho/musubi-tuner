#!/bin/bash

# -------------------------------
# 1. 检查 Python 版本（官方要求 3.10+）
# -------------------------------
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
if [ -z "$PYVER" ]; then
    echo "[错误] 未检测到 Python，请先安装 Python 3.10 或更高版本。"
    read -p "按任意键退出..."
    exit 1
fi

PYMAJOR=$(echo $PYVER | cut -d. -f1)
PYMINOR=$(echo $PYVER | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]; }; then
    echo "[错误] Python 版本需 3.10+，当前版本为 $PYVER。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 检测到 Python $PYVER（符合官方要求）"

# -------------------------------
# 2. 激活虚拟环境
# -------------------------------
if [ ! -f "venv/bin/activate" ]; then
    echo "[错误] 未找到虚拟环境，请先运行安装脚本创建虚拟环境"
    read -p "按任意键退出..."
    exit 1
fi

source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[错误] 虚拟环境激活失败，请手动执行：source venv/bin/activate"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 虚拟环境已激活"

# -------------------------------
# 3. 运行 Wan22 WebUI
# -------------------------------
echo ""
echo "================================================="
echo "[启动] 开始运行 Wan2.2 训练工具"
echo "================================================="
echo "⚠️ Wan2.2训练我这里是直接同时训练高低噪声，效果更佳，但显存占用较大！"
echo "⚠️ 显存优化可以参考 .md 文档，进行调整"
echo "******************************************************************************"

python wan22_webui.py

read -p "按任意键退出..."
