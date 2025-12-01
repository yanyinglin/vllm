#!/bin/bash
# 临时修复构建环境问题的脚本
# 使用方法: source fix_build_env.sh 然后运行你的构建命令

# 获取系统 torch 的路径
TORCH_PREFIX=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
export CMAKE_PREFIX_PATH="${TORCH_PREFIX}:${CMAKE_PREFIX_PATH}"

# 过滤掉临时的构建隔离路径
export PYTHONPATH=$(python -c "import sys; paths = [p for p in sys.path if not (p.startswith('/tmp/pip-build-env-') or p.startswith('/tmp/pip-req-build-'))]; print(':'.join(paths))")

echo "已设置 CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
echo "已设置 PYTHONPATH (已过滤临时路径)"
echo ""
echo "现在可以使用以下命令构建:"
echo "  pip install . --no-build-isolation"
echo "或者"
echo "  python setup.py bdist_wheel"

