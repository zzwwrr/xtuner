# 补丁与接入改动

安装顺序见 [INSTALL](../INSTALL.md)。补丁文件采用仓库相对路径，在表中固定提交的干净依赖 checkout 根目录应用；每次先执行 `git apply --check`。若 reverse check 已通过，说明该补丁已应用，不重复执行。

## UniEP 的依赖补丁

基线：Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a`。按表中顺序应用：

| 补丁 | 修复内容 | 是否重新编译 |
|---|---|---|
| [uniep-swiglu-backward.patch](uniep-swiglu-backward.patch) | SwiGLU backward 的 dA／dB 输出指针缺少行偏移；补齐起始行与循环递增 | Python／JIT kernel；重新安装打包即可，首次调用重新编译 kernel |
| [uniep-async-retain.patch](uniep-async-retain.patch) | 将 combine 前向的 host stream synchronize 替换为 CUDA event＋pinned count tensor 保留，防止 checkpoint 无梯度前向结束后缓冲过早复用 | Python 层；重新安装即可 |
| [uniep-package-tma.patch](uniep-package-tma.patch) | 将 `triton_dist/language/tma` 纳入安装包，避免 wheel 安装遗漏 TMA 模块 | 重新打包 distributed wheel |

第二项只移除对应 combine 前向位置的 host 等待，并保留 count tensor 的生命周期；没有移除 `ep_barrier_all`，也没有取消反向／dispatch 的其他同步。仅删除 synchronize 而不保留 tensor 会产生异步读取风险。

SwiGLU 和 async-retain 两项来自预研成功使用的修复；TMA 打包项来自当时依赖 checkout 的构建修正。它们的适用基线与性能候选参数分开记录。`plugin/patches/0001-*`、`0002-*` 是该依赖自身携带的 Triton 扩展补丁，由其 `scripts/build_triton.sh` 应用，无需另复制或重复应用。

## Liger：性能复现与离线 tuner

本目录四组性能复现使用 `bad6c3ec4269fc20fe70ab0bf882ac659b7cbd92` 的 native 接口和已编译菜单。训练需要启用 SM90 non-RDC／ALL_CONFIGS；这些是构建参数，不是新增源码补丁。

[liger-offline-tuner-build.patch](liger-offline-tuner-build.patch) 另供源码级离线 tuner 使用：拆分反向模板编译单元、补齐 host wrapper 显式实例化，并补上 TP reduce、remote all-reduce 和 CE workspace／reduce 的链接源文件。它不属于报告的性能 wheel，也不用于已发布的 `single_sweep.py` 菜单筛选。

需使用独立 Liger checkout 应用这份补丁，保留性能库的原源码与产物。应用后可编译 tuner，编译过程不运行 GPU workload：

```bash
# 在独立的 Liger checkout 根目录，且 HEAD 为上面的固定提交
git apply --check "$REPRO_ROOT/examples/megakernel_repro/patches/liger-offline-tuner-build.patch"
git apply "$REPRO_ROOT/examples/megakernel_repro/patches/liger-offline-tuner-build.patch"
cmake -S liger_cute_kernels/csrc/core/src/moe/tune -B build/tuner \
  -DCMAKE_BUILD_TYPE=Release -DLIGER_CUTE_CUDA_ARCH=90a -GNinja
cmake --build build/tuner --target tune_moe_fwd_bwd -j 4
```

沿用 INSTALL 中的 `CUTLASS_HOME`、`NVSHMEM_HOME`、CUDA 和 TVM FFI 环境；可执行程序运行还需要独立的多卡任务授权。此补丁会改变 tuner 专用的反向模板分组及 dispatch 注册，不能拿它生成的 core 替代四组实验的性能库。

固定 split-K／`LIGER_DIAG_NO_K_SPLIT`、反向重算计时插桩，以及缓存反向原型属于独立诊断或候选，未纳入这四组复现。性能入口使用原生 split-K 选择。

## XTuner 分支中已经包含的改动

| 提交 | 内容 | 使用方式 |
|---|---|---|
| `938549c1` | UniEP／Liger routed MoE 接入，保持 shared experts 等外围训练逻辑 | 选择对应 `moe_backend` |
| `309a6d8f` | packed W13 一次拆分复制，反向以 cat 形成 dW13 | `XTUNER_LIGER_W13_SPLIT_CAT=1` |
| `cb005e65` | 无梯度前向使用 W13 strided view，减少连续化复制 | `XTUNER_LIGER_NOGRAD_STRIDED=1` |

这些改动已在当前分支，不再应用外部 patch。两个 W13 开关可以并存，默认均关闭；各复现入口设置自己对应的值，16K 模板实验为 split/cat 开启、no-grad strided 关闭。启用某项候选与证明其完整训练数值通过是两件事，结果按对应实验记录。

## 本次打包检查

三个 UniEP 补丁已在固定提交的源文件上顺序通过 apply／reverse check；Liger tuner 补丁在固定提交上通过同样检查，包含新增的 `moe_bwd_part1.cu`。检查只涉及 CPU 和文件，未追加依赖全量编译或 GPU 验证。
