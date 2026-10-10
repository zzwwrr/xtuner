# MegaKernel 预研实验复现

这份目录配套 XTuner 的 UniEP / Liger CuTe EP 接入分支，包含报告中的四组实验。入口已整理为仓库相对路径，不依赖原来的 `Missions/`、本地授权文件或多层 runner。原始实验执行过；这次发布只完成 CPU 检查，没有重新运行 GPU 实验。

| 目录 | 测什么 | 原始结果 |
|---|---|---|
| [liger_vs_serial](liger_vs_serial/README.md) | 合成输入下，Liger 对 DeepEP＋Transformer Engine 的单层前反向 | 前向 2.148×；前反向合计 1.002× |
| [qwen_2k_rc0](qwen_2k_rc0/README.md) | 完整 Qwen，2K、关闭主层重算，仅替换 routed MoE 后端 | 吞吐 +19.14% |
| [qwen_2k_rc1](qwen_2k_rc1/README.md) | 完整 Qwen，2K、开启主层重算，仅替换 routed MoE 后端 | 吞吐 +29.49% |
| [qwen_16k_liger_templates](qwen_16k_liger_templates/README.md) | 固定 16K Liger 训练配置，筛选现成模板并确认候选 | 7970.69 → 8078.82 有效 token/s/卡；+1.36% |

`Serial` 是 DeepEP＋Transformer Engine 的隔离层基线；`native` 是 XTuner 的 DeepEP＋Triton grouped GEMM 路径。16K 的 A/B 都是 Liger，比较模板，不计算相对 native 的后端收益。数值检查和性能分别报告，历史训练对照存在部分 grad_norm 超差。

## 环境和输入

首次准备环境请按 [安装与编译指南](INSTALL.md) 操作；UniEP 依赖修复、Liger 离线 tuner 修复和分支内 W13 改动说明在 [补丁指南](patches/README.md)。

原实验使用单机八张 H200、SM90、NVLink，训练源码基线为 `cb005e65fa54f9ee56a55778aeffa4bba9eed2af`。本目录提交只新增复现文件，沿用该接入实现。Qwen 训练固定 48 层、H2048/I768/E128/K8、BF16、EP8/FSDP1、GB32、g1。

训练环境记录：Python 3.12、PyTorch 2.9.1+cu128、Triton 3.5.1、cuDNN runtime 9.15.1.9、cuDNN frontend 1.10.0，安装 DeepEP、NVSHMEM、Liger Kernel 及匹配的 Liger CuTe native 包。2K 使用 FA2；16K 使用 FA3。单层 Serial 还需要 Transformer Engine（原环境 2.11）。这些是原环境版本记录；安装时还需按各项目构建说明匹配 CUDA 和 native ABI。

Liger 训练使用 SM90 non-RDC binary，并要求 `LIGER_MOE_FORCE_CONFIG` / `LIGER_MOE_BWD_FORCE_CONFIG` 选择接口。16K 菜单在该目录的 `template_inventory.json`，运行前确认安装包编译了这些 tuple。候选菜单和开关来自预研时安装的 native 包；仅安装普通 Python 包不代表所有 native 模板已就绪。复现脚本使用当前环境，不自动安装、升级或编译依赖。

在仓库根目录设置实际输入和运行 Python：

```bash
export QWEN3_MOE_PATH=/path/to/Qwen3-30B-A3B
export ALPACA_PATH=/path/to/alpaca-jsonl
export PYTHON=/path/to/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

`ALPACA_PATH` 是原 FTDP/SFT JSONL 数据目录，使用原 `internlm2` chat template 和数据顺序。不同数据或 tokenizer 会改变有效 token 数，不能要求得到相同绝对 TGS。模型和数据只读，数据缓存、编译缓存及日志写入实验输出目录。

NVSHMEM native 库必须能被动态加载。按实际安装位置设置，例如：

```bash
export NVSHMEM_LIBDIR=/path/to/site-packages/nvidia/nvshmem/lib
export LD_LIBRARY_PATH="$NVSHMEM_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$NVSHMEM_LIBDIR/libnvshmem_host.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
```

若 Torch、TVM FFI 等库在另一安装前缀，还需将其实际库目录加入 `LD_LIBRARY_PATH`。完整运行会在每个 arm 的 `runtime-rank*.json` 记录实际导入来源、版本和模板开关。

原容器的 Liger Python 包位于虚拟环境、Torch 等依赖位于系统 site-packages；这种混合环境还需要将两处实际 site-packages 加入 `PYTHONPATH`。通常的完整虚拟环境无需这一步。选择 Python 后先确认 `import torch, deep_ep, liger_cute_kernels` 能导入。

## 查看计划和检查配置

三个 Qwen 入口默认打印计划，不启动 GPU job：

```bash
"$PYTHON" examples/megakernel_repro/qwen_2k_rc0/run.py
"$PYTHON" examples/megakernel_repro/qwen_2k_rc1/run.py
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py --sweep
```

加 `--check-config` 会构造 A/B 的训练配置并保存实际配置及依赖来源；不执行训练。依赖导入可能进行设备发现。单层 benchmark 用 `bash -n` 检查启动脚本，`compare.py --help` 查看固定形状参数。

## 执行和结果读取

取得对应 GPU 执行授权、确认八卡独占且依赖已准备后，设置 `GPU_AUTHORIZATION=GRANTED_BY_USER`。Qwen 入口用 `--execute --block-start-utc <实际 allocation 起点>`；默认完整 ABBA，失败立即停止并保留日志，不补齐失败结果。两个小时资源块的最后十五分钟用于收尾，脚本在启动前预留完整实验的预算。2296 执行仍应由 2999 发布已授权 assignment。

每个目录的 README 有完整命令。一次只运行一组实验。默认输出为该目录 `runs/<UTC>/`，也可用 `--out` 指定工作区内一个尚不存在的专用目录。保持 `--python` 指向准备好的环境。无需给训练加 profiler，profile 数据不进入本组吞吐结果。

Qwen 固定原来的 80 步 LR schedule，短 run 通过截断迭代器停在 2/30/50 步。80 步统计 11–79；50 步统计 11–49；30 步统计 11–29。有效 TGS 为全部 rank 的 `text_tokens` 总和 ÷（8 × 逐步最慢 rank 步时之和），按 token 有效数统计；每个 run 必须有完整八个 rank 的 tracker。`common/metrics.py` 同时给出 A0/B0、A1/B1 配对及双方自重复的数值检查。

输出包括命令和配置、八 rank tracker、运行日志、退出码以及 `summary.json`。`memory/max_memory_GB` / `reserved_memory_GB` 字段实际以 GiB 计，不包含全部 NVSHMEM 等框架外设备内存。独立复现结果应使用其自身的来源和环境记录。

## 文件怎么组织

各实验目录包含入口和 README；训练目录还包含 `train_config.py`。`common/` 是三个训练实验实际消费的共用配置、训练入口、顺序运行和 tracker 汇总代码。16K 目录另有真实输入采集、单层菜单筛选和冻结模板菜单，可通过 `--sweep` 执行完整搜索过程。

原始入口对应预研资料中的 `1727-prep/perf-debug/run.sh`、`local-2999-f1-rc0-liger/run.py`、`local-2999-liger-gb32-rc1-20261007/run.py`、`local-2999-qwen-liger16k-template-sweep-20261010/run.py`；这些来源用于追溯，本仓库入口已消除相应本地目录依赖。

## 本次发布的检查

Python 语法、shell 语法、三个默认计划及十份前向/十五份反向菜单检查通过。三个 Qwen 实验各构造 A/B 两份实际配置，除输入、输出、缓存路径外，与历史 resolved config 完整一致。CPU 汇总原始八 rank tracker 得到 2K rc0 的 1.191410×、rc1 的 1.294897×，以及 16K 候选的 8078.82 有效 token/s/卡；单层原始日志复算前向 2.147507×、合计 1.001638×。GPU 复跑应在调用者取得授权后执行。
