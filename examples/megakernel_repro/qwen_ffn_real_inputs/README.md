# Qwen 真实输入：XTuner native / Liger 单层 FFN 正反向

环境准备：[安装／编译](../INSTALL.md) · [补丁说明](../patches/README.md) · [总 GUIDE](../GUIDE.md)。

这组脚本复现报告中的同配置单层 FFN 表。使用 XTuner 的实际接入实现：native 调用 `tests/module/moe_backend/bench_moe_layer.py::_native_segment`，包含 DeepEP dispatch、`MoEBlock` 的 Triton grouped GEMM 和 combine；Liger 调用 `build_fused_ep_forward('liger')` 返回的 `LigerCuteSession`。不含 router、attention、共享专家、FSDP 参数搬运、loss 或 optimizer。

## 固定配置与步骤

实采使用 Qwen3-30B-A3B 原48层、单机8×H200/NVLink、BF16、EP8/FSDP1、GB32/g1/rc1/eager、FA3、chunk CE1024、seed0。保留原80步 LR schedule，只截断训练迭代器到2步。输入模型、tokenizer 和数据均只读。GB32/g1 每卡每步4个pack；回放测量其中一次 routed FFN 调用，每卡输入8192或16384个token。

顺序为 `8K capture → native → Liger → 16K capture → native → Liger`，共六个串行进程。实采位于 `step1/layer23` 的首次有梯度调用，即 checkpoint 重放时的一次调用，保存同次输入、路由、专家权重和上游梯度dY，避免四个pack之间配错。随后两后端共享该份输入进行独立单层回放；回放也使用 reentrant checkpoint，首次无梯度前向F0与反向重放Fr分别记录。

两档均使用原 Liger SM90 non-RDC 包、既有 W13 split_cat 优化及固定模板，不使用缓存反向原型或诊断构建：

```text
FWD: 128,64,4,64,128,64,4,64,4,2,128
BWD: 8,128,64,4,128,256,64,2,32,64,64,3,128
```

这两组 tuple 必须已经编译进安装包。native 固定默认 Triton grouped GEMM；前向 no-grad strided 优化关闭。每档只做一组A/B，每臂10次预热、50次计时，不调参、不加 profiler。

## 使用

按总 GUIDE 设置 `QWEN3_MOE_PATH`、`ALPACA_PATH`、`PYTHON`、`PYTHONPATH` 和 NVSHMEM 动态库路径。在仓库根目录查看计划、检查配置：

```bash
"$PYTHON" examples/megakernel_repro/qwen_ffn_real_inputs/run.py
"$PYTHON" examples/megakernel_repro/qwen_ffn_real_inputs/run.py --check-config \
  --out /path/to/workspace/ffn-config-check
```

默认只打印计划；`--check-config` 构造每档 native/Liger 配置和实际导入版本，不训练，依赖导入可能进行设备发现。输出目录必须尚不存在，不能位于模型、数据或只读源码目录中。

取得 GPU 授权、准备好八卡独占环境后执行：

```bash
export GPU_AUTHORIZATION=GRANTED_BY_USER
"$PYTHON" examples/megakernel_repro/qwen_ffn_real_inputs/run.py --execute \
  --out /path/to/workspace/ffn-real-8k16k
```

`--seq-lens 8192` 或 `--seq-lens 16384` 可只复现一档。`--python` 指定 worker 的 Python，默认当前解释器；`--timeout` 默认每进程900秒。临时目录在 `--tmp-root` 下单独创建，路径应短于60字节，避免 Python multiprocessing 的 AF_UNIX 限制；默认使用存在的 `/root/workspace`，否则当前目录。失败立即停止，不自动重试或换模板。这个入口不要求固定两小时资源块或 allocation 起点。

## 计时与数值检查

CUDA Events 包住 routed FFN 调用：F0为首次前向，Fr为反向重放；反向时间包含Fr与内部激活重建，总计独立包住首次前向及反向。分配、输入克隆和迭代前 barrier 在计时外。每次先取8个rank的最大耗时，再对50次求平均；加速比为 native 相应聚合耗时除以 Liger 相应聚合耗时。各阶段独立取最慢rank，不强求分项相加等于总计。

计时前检查Y、dX、dprob、dW13、dW2，对照FP32纯Torch EP reference；Liger还与同输入native比较。Y/dX的atol/rtol为0.03，其他为0.05，relative L2≤0.02、有限且超差元素为0。任一rank失败就停止。这是固定输入BF16筛查，不能代替整训数值或收敛验收。

输出保留 `source.json` / `source.diff`、配置、命令、实际运行版本、八rank原始计时与数值gate、capture SHA及receipt、实采训练日志和tracker。成功后自动精确清理已消费的实采张量及本次work/cache/tmp，保留清理回执；失败产物保留供调用者诊断。没有checkpoint/HF导出。

完成后查看 `summary.md` 的七列表格和 `summary.json` 的完整数据，也可以CPU重新汇总：

```bash
"$PYTHON" examples/megakernel_repro/qwen_ffn_real_inputs/summarize.py \
  --out /path/to/workspace/ffn-real-8k16k
```

## 原始结果与范围

原测试源码为 `cb005e65fa54f9ee56a55778aeffa4bba9eed2af`，环境沿用总 GUIDE，2026-10-12实测结果如下。单位ms，反向含checkpoint重放：

| seq_len | 方法（native or liger） | 前向时间 | 前向加速比 | 反向时间 | 反向加速比 | 总体加速比 |
|---|---|---:|---:|---:|---:|---:|
| 8K | native | 5.531 | 1× | 12.785 | 1× | 1× |
| 8K | liger | 3.156 | 1.753× | 15.388 | 0.831× | 0.988× |
| 16K | native | 10.592 | 1× | 24.758 | 1× | 1× |
| 16K | liger | 5.852 | 1.810× | 27.030 | 0.916× | 1.075× |

合计独立计时分别为8K native/Liger 18.423/18.650ms、16K 35.462/32.992ms。原测试32份数值screening通过，1600条rank计时已独立复算。本次脚本发布只做CPU检查，没有重新执行GPU。

该结果对应某一层的一份固定真实输入，并非48层平均、整模型前向或整训TGS；与报告中的GB32完整训练吞吐分别统计。数据、路由分布、依赖、硬件或源码变化会改变绝对耗时。
