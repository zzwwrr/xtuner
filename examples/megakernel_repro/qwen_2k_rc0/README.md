# Qwen 2K：关闭主层重算的后端对照

环境准备：[安装／编译](../INSTALL.md) · [补丁说明](../patches/README.md)。

完整 Qwen3-30B-A3B，48 层、BF16、EP8/FSDP1、GB32、2K pack、g1、rc0、eager、FA2、eager CE。AdamW foreach 保持 `None`，不更改原优化器选择。只将 routed MoE 从 native 换成 Liger，router、attention、loss 和训练框架保持同配置。

顺序为 native → Liger → Liger → native，各 80 步，从同一模型初始化。两种后端使用同一数据和 seed0。计时窗口 11–79 步，完整八 rank tracker 用于有效 TGS 和数值检查。

先按 [GUIDE](../GUIDE.md) 设置输入、Python 和 NVSHMEM 库路径，然后：

```bash
"$PYTHON" examples/megakernel_repro/qwen_2k_rc0/run.py --check-config
export GPU_AUTHORIZATION=GRANTED_BY_USER
"$PYTHON" examples/megakernel_repro/qwen_2k_rc0/run.py --execute
```

入口默认只打印计划；完成后 `runs/<UTC>/summary.json` 汇总四次运行。共享脚本在 `../common/`。

Liger 固定 non-RDC，W13 split/cat 开启、no-grad strided 关闭，split-K 沿用原生选择。前向 tuple 为 `128,64,4,64,128,64,4,64,4,4,128`；反向为 `16,128,64,4,256,128,64,2,32,64,64,3,128`。

历史 native / Liger 平均步时为 2.350492 / 1.972866 s，Liger 有效 TGS 为 4121.40 token/s/卡，步时比 1.191410×、吞吐约 +19.14%。部分 grad_norm 检查超差，性能记录与数值状态分别保留。

配置来自 `local-2999-f1-rc0-liger` 的四臂原始配置；新入口将原 B0 runner 的配置与顺序启动整理为仓库相对路径。没有复用本地任务授权或历史 CPU_READY 状态。
