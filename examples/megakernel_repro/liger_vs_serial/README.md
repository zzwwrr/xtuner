# Liger 对 Serial 的隔离单层测试

环境准备：[安装／编译](../INSTALL.md) · [补丁说明](../patches/README.md)。

测试整个 `dispatch → experts → combine` 的前向和反向。Serial 使用 DeepEP＋Transformer Engine；输入是固定随机种子生成的激活、权重、路由和上游梯度，独立于完整 XTuner 训练。

固定单机 EP8，每 rank T8192（8K token）、H2048/I768/E128/K8、BF16、seed42；各卡随机种子为 seed＋rank。DeepEP 使用 24 SM。先以 FP32 reference 核对 Y、dX、路由概率梯度和专家权重梯度，再做 5 次 warmup、20 次计时。前向和反向 CUDA Events 分别计时，逐次取八 rank 最大值再求均值。没有外层 checkpoint。

环境和 GPU 授权约定在 [GUIDE](../GUIDE.md)。在仓库根目录运行：

```bash
export GPU_AUTHORIZATION=GRANTED_BY_USER
bash examples/megakernel_repro/liger_vs_serial/run.sh serial
bash examples/megakernel_repro/liger_vs_serial/run.sh liger
```

脚本打印两次运行的 `RUN=` 路径。完成后用 CPU 汇总：

```bash
"$PYTHON" examples/megakernel_repro/liger_vs_serial/summarize.py /path/to/serial-run /path/to/liger-run
```

| 实现 | 前向 ms | 反向 ms | 合计 ms |
|---|---:|---:|---:|
| Serial：DeepEP＋TE | 5.717 | 6.016 | 11.733 |
| Liger | 2.662 | 9.052 | 11.714 |

历史前向加速 2.148×，反向约 0.665×，合计约 1.002×。这组结果不能直接替代 XTuner native 或完整训练吞吐对照。

`compare.py`、`serial_core.py`、`serial_te.py` 来自 2026-09-23 成功运行的归档源码；只调整同目录导入，并移除本次不发布的 UniEP 分支。启动脚本改为使用调用者提供的 Python/依赖环境。原始 Liger timing 使用自动模板选择和 RDC 路径，未强制 split-K。
