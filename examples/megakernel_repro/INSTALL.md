# 算子安装与编译

配套 [实验 GUIDE](GUIDE.md)。以下命令针对报告使用的 H200／SM90、Python 3.12、CUDA Toolkit 12.8 环境。先准备能运行 XTuner 原生 BF16 训练的基础环境，再构建这里的额外算子。构建可在 CPU 阶段完成；分布式测试和训练按实验 README 的 GPU 授权约定执行。

## 1. 固定源码和环境

| 组件 | 本次使用的版本／源码 | 用途 |
|---|---|---|
| XTuner | 本仓库 `megakernel-moe-adapters-20261009` 分支；训练实现来自 `cb005e65` | 已包含 UniEP／Liger 接入和 W13 选项 |
| Liger Kernel／CuTe native | `linkedin/Liger-Kernel`，`bad6c3ec4269fc20fe70ab0bf882ac659b7cbd92`，0.8.3 | Python wrapper 和原生库分别打包 |
| CUTLASS | `NVIDIA/cutlass`，`4370102f9dacab813282e1d67722fceb0b90a019` | Liger 编译使用的 4.4.1 源码 |
| DeepEP | `deepseek-ai/DeepEP`，`3fcf25ed1b50f8abab6bac71e792432f7158fc26` | 四组已发布复现的通信依赖 |
| Transformer Engine | `NVIDIA/TransformerEngine`，`c188b533cc3721ca9c6bbfd26148f5cf60108c25`，2.11 | 仅 Serial 单层基线需要 |
| UniEP／Triton-distributed | `ByteDance-Seed/Triton-distributed`，`7908e4ea9010bb2238f9f05b28c904fd4b71c60a` ＋下文补丁 | 独立的 UniEP 环境 |
| UniEP 的 Triton | 子模块 `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`，3.7.1＋tritondist | 与 distributed plugin 同源编译 |

Liger／Serial 环境使用 PyTorch **2.9.1+cu128**、Triton **3.5.1**、NVSHMEM **3.6.5**、TVM FFI **0.1.11**、cuDNN runtime **9.15.1.9**、cuDNN frontend **1.10.0**。FA2 的环境记录为 `flash-attn==2.8.3`，FA3 为 `flash-attn-3==3.0.0b1`；2K 使用 FA2，16K 使用 FA3。这些 attention 依赖在基础训练环境中准备，实际选中的函数由复现脚本记录。

UniEP 的定制 Triton 单独安装，保留 Liger 环境的 Triton 3.5.1。历史 UniEP 环境还使用过 DeepEP `1.2.1+9af0e0d`；该环境的旧结果与这里四组复现的来源分别记录。

在已克隆的 XTuner 仓库根目录设置路径。`BUILD_ROOT` 选工作区内一个新的专用目录，源码、wheel 和缓存都留在该目录：

```bash
export REPRO_ROOT="$PWD"
export BUILD_ROOT=/path/to/workspace/megakernel-deps
mkdir -p "$BUILD_ROOT/src" "$BUILD_ROOT/wheels" "$BUILD_ROOT/cache"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
export XDG_CACHE_HOME="$BUILD_ROOT/cache/xdg"
export TRITON_CACHE_DIR="$BUILD_ROOT/cache/triton"
export CUDA_CACHE_PATH="$BUILD_ROOT/cache/cuda"
export PIP_CACHE_DIR="$BUILD_ROOT/cache/pip"
```

以下虚拟环境继承已准备好的基础训练依赖，与原实验的安装方式相同。确认继承的 Torch、CUDA 和 attention 版本后继续：

```bash
python3.12 -m venv --system-site-packages "$BUILD_ROOT/venv-liger"
source "$BUILD_ROOT/venv-liger/bin/activate"
python -m pip install 'cmake>=3.24' ninja setuptools setuptools-scm wheel hatchling packaging pybind11 \
  'apache-tvm-ffi==0.1.11' 'nvidia-nvshmem-cu12==3.6.5'
python -m pip install --no-deps -e "$REPRO_ROOT"
python -m pip freeze > "$BUILD_ROOT/liger-environment.txt"
nvcc --version
```

cuDNN runtime 覆盖在其他依赖安装完成后设置，随后冻结环境：

```bash
python -m pip install --no-deps 'nvidia-cudnn-cu12==9.15.1.9' \
  'nvidia-cudnn-frontend==1.10.0'
```

Torch wheel 的 metadata 仍可能要求 cuDNN 9.10.2.21，因此 `pip check` 会报告这项已知版本冲突。保持这里的显式覆盖，并核对实际导入版本。

## 2. DeepEP

```bash
git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_ROOT/src/DeepEP"
git -C "$BUILD_ROOT/src/DeepEP" checkout --detach 3fcf25ed1b50f8abab6bac71e792432f7158fc26
export TORCH_CUDA_ARCH_LIST=9.0
env -u NVSHMEM_DIR python -m pip wheel "$BUILD_ROOT/src/DeepEP" \
  --no-deps --no-build-isolation -w "$BUILD_ROOT/wheels/deepep"
python -m pip install --no-deps "$BUILD_ROOT"/wheels/deepep/deep_ep-*.whl
```

该提交能自动发现 PyPI NVSHMEM，并链接 `libnvshmem_host.so.3`。命令清除 `NVSHMEM_DIR`，使用这条发现路径；显式指定原生 NVSHMEM 安装目录时，该目录需同时提供头文件、device 静态库和未版本化的 `libnvshmem_host.so`。构建日志应包含 NVSHMEM，不能出现关闭其功能的提示。

按当前 Python 自动取得动态库目录：

```bash
export NVSHMEM_HOME="$(python -c 'import importlib.util; s=importlib.util.find_spec("nvidia.nvshmem"); print(next(iter(s.submodule_search_locations)))')"
export NVSHMEM_LIBDIR="$NVSHMEM_HOME/lib"
export LD_LIBRARY_PATH="$NVSHMEM_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$NVSHMEM_LIBDIR/libnvshmem_host.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
```

DeepEP 在以上固定提交上无需本目录额外补丁。多机传输及历史其他提交的修复不属于这四组单机复现。

## 3. Liger Python wrapper 与 CuTe 原生库

```bash
git clone https://github.com/linkedin/Liger-Kernel.git "$BUILD_ROOT/src/Liger-Kernel"
git -C "$BUILD_ROOT/src/Liger-Kernel" checkout --detach bad6c3ec4269fc20fe70ab0bf882ac659b7cbd92
git clone https://github.com/NVIDIA/cutlass.git "$BUILD_ROOT/src/cutlass"
git -C "$BUILD_ROOT/src/cutlass" checkout --detach 4370102f9dacab813282e1d67722fceb0b90a019
export CUTLASS_HOME="$BUILD_ROOT/src/cutlass"

python -m pip wheel "$BUILD_ROOT/src/Liger-Kernel" \
  --no-deps --no-build-isolation -w "$BUILD_ROOT/wheels/liger-python"
python -m pip install --no-deps "$BUILD_ROOT"/wheels/liger-python/liger_kernel-*.whl

export LIGER_CUTE_CUDA_ARCH=90a
export LIGER_CUTE_ENABLE_SM90_NONRDC_MOE=1
export LIGER_CUTE_SM90_NONRDC_ALL_CONFIGS=1
export LIGER_CUTE_BUILD_JOBS=4
python -m pip wheel "$BUILD_ROOT/src/Liger-Kernel/liger_cute_kernels" \
  --no-deps --no-build-isolation -w "$BUILD_ROOT/wheels/liger-native"
python -m pip install --no-deps "$BUILD_ROOT"/wheels/liger-native/liger_cute_kernels-*.whl
```

原生 wheel 应包含 `libliger_cute_kernels.so` 和 `liger_moe_sm90_nonrdc.cubin`。`ALL_CONFIGS=1` 用于编译完整现成模板菜单，包括 Qwen 使用的模板；只编默认 Mixtral 模板不足以完成本目录的 Qwen 筛选。CUTLASS 3.9.2 与该源码所需接口不匹配，使用上表固定版本。

若已有同版本 core 和 cubin，可设置 `LIGER_CUTE_CORE_DIR=/path/to/core-and-cubin` 后执行原生 wheel 命令，复用编译产物。保留上述 non-RDC 构建开关，并将 `.so` 与 `.cubin` 一同打包。

**运行配置由各实验入口设置。** 单层 Serial／Liger 历史复现使用 RDC 和自动模板；Qwen 训练使用 non-RDC 和指定模板。16K 的菜单在 [template_inventory.json](qwen_16k_liger_templates/template_inventory.json)。该固定 Liger 提交已提供强制模板接口，无需将离线 tuner 修复或固定 split-K 诊断补丁应用到性能库。

Serial 所需 TE 可在同一环境独立构建：

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git "$BUILD_ROOT/src/TransformerEngine"
git -C "$BUILD_ROOT/src/TransformerEngine" checkout --detach c188b533cc3721ca9c6bbfd26148f5cf60108c25
git -C "$BUILD_ROOT/src/TransformerEngine" submodule update --init --recursive
NVTE_FRAMEWORK=pytorch python -m pip wheel "$BUILD_ROOT/src/TransformerEngine" \
  --no-deps --no-build-isolation -w "$BUILD_ROOT/wheels/te"
python -m pip install --no-deps "$BUILD_ROOT"/wheels/te/*.whl
```

## 4. UniEP：单独环境与补丁

四个已发布复现实验使用 Liger／Serial；本节供分支中的 UniEP 接入准备依赖，尚未新增 UniEP 实验入口。

```bash
python3.12 -m venv --system-site-packages "$BUILD_ROOT/venv-uniep"
source "$BUILD_ROOT/venv-uniep/bin/activate"
python -m pip install 'cmake>=3.24' ninja setuptools setuptools-scm wheel packaging pybind11 \
  'cuda.core==1.0.1' 'cuda-python>=12.0' 'nvshmem4py-cu12==0.3.0' \
  'nvidia-nvshmem-cu12==3.6.5' 'Cython>=0.29.24'
git clone https://github.com/ByteDance-Seed/Triton-distributed.git "$BUILD_ROOT/src/Triton-distributed"
git -C "$BUILD_ROOT/src/Triton-distributed" checkout --detach 7908e4ea9010bb2238f9f05b28c904fd4b71c60a
git -C "$BUILD_ROOT/src/Triton-distributed" submodule update --init 3rdparty/triton
```

按 [补丁说明](patches/README.md) 顺序应用三个 UniEP 补丁，然后构建匹配的 Triton 与 plugin：

```bash
cd "$BUILD_ROOT/src/Triton-distributed"
git apply --check "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-swiglu-backward.patch"
git apply "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-swiglu-backward.patch"
git apply --check "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-async-retain.patch"
git apply "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-async-retain.patch"
git apply --check "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-package-tma.patch"
git apply "$REPRO_ROOT/examples/megakernel_repro/patches/uniep-package-tma.patch"

export TRITON_HOME="$BUILD_ROOT/cache/triton-build"
TRITON_DIST_TRITON_INSTALL_MODE=wheel \
  TRITON_DIST_TRITON_WHEEL_OUT="$BUILD_ROOT/wheels/triton-dist-triton" \
  MAX_JOBS=4 bash scripts/build_triton.sh
python -m pip install --no-deps "$BUILD_ROOT"/wheels/triton-dist-triton/triton-*.whl
export TRITON_SOURCE_DIR="$PWD/3rdparty/triton"
export TRITON_DIST_SKIP_TRITON_BUILD=1
export TRITON_DIST_TRITON_REQUIREMENT="$(python -c 'from importlib.metadata import version; print("triton==" + version("triton"))')"
python -m pip wheel ./python --no-deps --no-build-isolation \
  -w "$BUILD_ROOT/wheels/triton-dist"
python -m pip install --no-deps "$BUILD_ROOT"/wheels/triton-dist/triton_dist-*.whl
python -m pip install --no-deps -e "$REPRO_ROOT"
cd "$REPRO_ROOT"
```

`build_triton.sh` 自带两项 plugin hook 补丁、`TRITON_EXT_ENABLED=1` 和 libstdc++ 符号隔离，stock Triton 3.7.1 wheel 不能替代这个构建。构建版本可能带有 Git hash，plugin 的依赖要求按刚安装的 wheel metadata 设置。上述 wheel 模式保留基础环境，绕开该脚本默认 editable 模式的包清理和 Torch metadata 改写。plugin 通过 `python/setup.py` 构建，链接同一子模块及其 LLVM／MLIR；若自动定位不到 LLVM，设置 `LLVM_INSTALL_DIR` 为该次 Triton 构建实际使用的目录。

UniEP 环境与 Torch 原有的 Triton 3.5.1 dependency pin 存在已知差异。后续依赖安装使用冻结环境和 `--no-deps`，并检查实际导入的 Triton 及 distributed plugin 来源。

## 5. 安装后检查

回到选定环境，重新设置第 2 节的 NVSHMEM 动态库路径。记录版本和导入路径：

```bash
python - <<'PY'
from importlib import metadata, util
for name in ('torch', 'triton', 'deep-ep', 'liger-kernel', 'liger-cute-kernels',
             'apache-tvm-ffi', 'nvidia-nvshmem-cu12', 'triton-dist'):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, 'not installed')
for name in ('xtuner', 'deep_ep', 'liger_kernel', 'liger_cute_kernels', 'triton', 'triton_dist'):
    spec = util.find_spec(name)
    print(name, spec.origin if spec else None)
PY
python -m pip freeze > "$BUILD_ROOT/final-environment.txt"
```

Liger 环境继续执行 [GUIDE 的配置检查](GUIDE.md#查看计划和检查配置)，随后按各实验 README 运行。实际导入、模板、attention 和 W13 开关以运行产物为准。构建成功与实验数值验收分别记录。

本次指南和补丁完成源码／命令选项及 CPU 检查；没有重新编译整套依赖或执行 GPU 实验。历史 wheel、模型、数据和原始 tracker 不随仓库上传。
