# Qwen 2K：开启主层重算的后端对照

环境准备：[安装／编译](../INSTALL.md) · [补丁说明](../patches/README.md)。

完整 Qwen3-30B-A3B，48 层、BF16、EP8/FSDP1、GB32、2K pack、g1、rc1、eager、FA2、eager CE。优化器、数据、seed0 与 rc0 实验保持原设置。当前源码最后一层排除 checkpoint，因此 rc1 对前 47 个主层重算。

顺序为 native → Liger → Liger → native，各 80 步，从同一模型初始化。只替换 routed MoE 后端。计时窗口 11–79 步，原 80 步 LR schedule 完整保留。

先按 [GUIDE](../GUIDE.md) 设置环境，再执行：

```bash
"$PYTHON" examples/megakernel_repro/qwen_2k_rc1/run.py --check-config
export GPU_AUTHORIZATION=GRANTED_BY_USER
"$PYTHON" examples/megakernel_repro/qwen_2k_rc1/run.py --execute \
  --block-start-utc 2026-10-10T00:00:00+00:00
```

时间参数替换为实际 allocation 起点。入口默认打印计划；四臂结果和数值检查写入 `runs/<UTC>/summary.json`。

Liger tuple、non-RDC 及 W13 标志与 [rc0 实验](../qwen_2k_rc0/README.md) 相同。主层 checkpoint 会重放层内 attention、router、专家及 MoE 通信；Liger backward 内部激活重建是另一层机制。

历史 native / Liger 平均步时 2.987198 / 2.306899 s，有效 TGS 为 2721.94 / 3524.63 token/s/卡，步时比 1.294897×、吞吐约 +29.49%。跨后端后段 grad_norm 有严格超差，双方自重复通过；不称为完整训练数值验收通过。

来源为 `local-2999-liger-gb32-rc1-20261007` 的成功 ABBA 与配置。脚本消费本目录配置及 `../common/`，不依赖原本机 runner 目录。
