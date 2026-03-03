#!/bin/bash
set -e  # 遇到错误立即退出

echo "================================================="
echo "Musubi Tuner 纯净安装脚本（CUDA 12.4）"
echo "================================================="

# -------------------------------
# 函数定义区域（必须在调用前定义）
# -------------------------------

# 验证安装结果
verify_installation() {
    echo ""
    echo "================================================="
    echo "[6] 安装结果验证"
    echo "================================================="
    python -c "import torch; print(f'PyTorch 版本：{torch.__version__}（需 ≥2.5.1）')"
    python -c "import torch; print(f'CUDA 支持：{torch.cuda.is_available()}（需为 True）')"
    python -c "import torch; print(f'CUDA 版本：{torch.version.cuda if torch.cuda.is_available() else \"未检测到\"}')"
    python -c "import torchvision; print(f'torchvision 版本：{torchvision.__version__}')"
    
    echo ""
    echo "================================================="
    echo "是否安装 SageAttention？(Y/N，默认Y)"
    echo "SageAttention 可以显著提升注意力机制的性能"
    echo "注意：Linux 需要编译安装，确保系统已安装 CUDA 开发工具"
    echo "即使不安装，依然可以使用本工具，只是性能会下降"
    echo "================================================="
    read -p "请选择 (Y/N): " install_sage
    install_sage=${install_sage:-Y}
    
    if [ "${install_sage^^}" = "Y" ]; then
        install_sage_attention
    else
        echo "跳过 SageAttention 安装"
        final_complete
    fi
}

# 安装 SageAttention
install_sage_attention() {
    echo ""
    echo "================================================="
    echo "[7] 安装 SageAttention（Linux 从源码编译）"
    echo "================================================="
    
    echo "正在安装 SageAttention 依赖..."
    pip install ninja packaging
    
    echo "正在从 PyPI 安装 SageAttention..."
    pip install sageattention
    
    if [ $? -ne 0 ]; then
        echo "[错误] SageAttention 安装失败"
        echo "可能需要："
        echo "1. 确保 CUDA Toolkit 已正确安装（nvcc 可用）"
        echo "2. 确保系统有编译环境（gcc/g++）"
        exit 1
    fi
    
    echo "验证安装结果..."
    echo "检查 SageAttention 模块..."
    python -c "import sageattention; print('✅ SageAttention 导入成功')"
    
    if [ $? -ne 0 ]; then
        echo "[警告] SageAttention 导入失败，可能安装不完整"
    else
        echo "✅ SageAttention 验证通过"
    fi
    
    final_complete
}

# 最终完成
final_complete() {
    echo ""
    echo "================================================="
    echo "✅ 所有安装步骤完成！"
    echo "提示：运行时请确保虚拟环境已激活（source venv/bin/activate）"
    echo "================================================="
}

# -------------------------------
# 1. 检查 Python 版本（官方要求 3.10+）
# -------------------------------
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
if [ -z "$PYVER" ]; then
    echo "[错误] 未检测到 Python，请先安装 Python 3.10 或更高版本。"
    exit 1
fi

PYMAJOR=$(echo $PYVER | cut -d. -f1)
PYMINOR=$(echo $PYVER | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]; }; then
    echo "[错误] Python 版本需 3.10+，当前版本为 $PYVER。"
    exit 1
fi
echo "✅ 检测到 Python $PYVER（符合官方要求）"

# -------------------------------
# 2. 处理虚拟环境（默认不删除，询问用户）
# -------------------------------
echo ""
echo "[1] 虚拟环境处理"
DELETE_VENV="N"
read -p "是否删除现有虚拟环境并重建？(Y/N，默认N) " USERINPUT
USERINPUT=${USERINPUT:-N}
if [ "${USERINPUT^^}" = "Y" ]; then
    DELETE_VENV="Y"
fi

# 根据用户选择处理虚拟环境
if [ "$DELETE_VENV" = "Y" ]; then
    echo "正在删除旧虚拟环境..."
    rm -rf venv
    echo "正在创建新虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境已重建"
else
    if [ ! -d "venv" ]; then
        echo "未检测到虚拟环境，正在创建..."
        python3 -m venv venv
        echo "✅ 虚拟环境已创建"
    else
        echo "✅ 保留现有虚拟环境"
        echo "检测到现有环境，跳转到安装结果验证..."
        source venv/bin/activate
        if [ $? -ne 0 ]; then
            echo "[错误] 虚拟环境激活失败，请手动执行：source venv/bin/activate"
            exit 1
        fi
        # 跳转到验证函数
        verify_installation
        exit 0
    fi
fi

# 激活虚拟环境
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[错误] 虚拟环境激活失败，请手动执行：source venv/bin/activate"
    exit 1
fi
echo "✅ 虚拟环境已激活"

# -------------------------------
# 3. 清理虚拟环境内的旧依赖（仅保留旧环境时需要）
# -------------------------------
echo "[2] 清理环境依赖..."
if [ "$DELETE_VENV" = "N" ]; then
    # 只有保留旧环境时，才需要卸载里面的旧包
    echo "正在卸载虚拟环境内的旧版本依赖..."
    pip uninstall -y torch torchvision torchaudio xformers 2>/dev/null || true
else
    # 新环境无需卸载（本身就是空的）
    echo "新环境无需清理旧依赖"
fi
echo "✅ 依赖清理完成"

# -------------------------------
# 4. 升级 pip
# -------------------------------
echo "[3] 升级 pip 到最新版本..."
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
echo "✅ pip 升级完成"

# -------------------------------
# 5. 安装 PyTorch（官方推荐）
# -------------------------------
echo "[4] 安装 PyTorch + torchvision（CUDA 12.4 版本）..."
echo "指定版本：torch==2.5.1+cu124, torchvision==0.20.1+cu124"
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
if [ $? -ne 0 ]; then
    echo "[错误] PyTorch 安装失败，建议手动执行："
    echo "pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi
echo "✅ PyTorch + torchvision 安装完成"

# -------------------------------
# 6. 安装 Musubi Tuner 核心依赖
# -------------------------------
echo "[5] 安装 Musubi Tuner 本地依赖..."
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
if [ $? -ne 0 ]; then
    echo "[错误] 核心依赖安装失败，请确保当前目录包含 setup.py"
    exit 1
fi
echo "✅ 核心依赖安装完成"

echo "安装 TensorBoard（用于训练日志可视化）..."
pip install tensorboard -i https://mirrors.aliyun.com/pypi/simple/
pip install flask -i https://mirrors.aliyun.com/pypi/simple/
pip install flask-socketio -i https://mirrors.aliyun.com/pypi/simple/

if [ $? -ne 0 ]; then
    echo "[警告] TensorBoard 安装失败，可能影响日志可视化功能"
else
    echo "✅ TensorBoard 安装完成"
fi

# -------------------------------
# 7. 执行验证流程
# -------------------------------
# 执行验证流程
verify_installation
