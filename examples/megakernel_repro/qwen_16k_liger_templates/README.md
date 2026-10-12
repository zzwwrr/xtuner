# Qwen 16K：Liger 已选模板和现成菜单筛选

环境准备：[安装／编译](../INSTALL.md) · [补丁说明](../patches/README.md)。

完整 Qwen3-30B-A3B，48 层、BF16、EP8/FSDP1、GB32、16K pack、g1、rc1、compile、FA3、chunk CE1024。non-RDC、W13 split/cat 开启、no-grad strided 关闭，split-K 原生选择。A/B 都使用 Liger，仅模板改变。

| 配置 | 前向 tuple | 反向 tuple |
|---|---|---|
| A：当前 CS2 | `128,64,4,64,128,64,4,64,4,2,128` | `16,128,64,4,256,128,64,2,32,64,64,3,128` |
| B：已选候选 | 同 A | `8,128,64,4,128,256,64,2,32,64,64,3,128` |

候选保留前向 CS2，调整反向工作划分和 dW GEMM 分块。没有改通信协议、精度或融合范围。

## 复现已选候选

环境设置见 [GUIDE](../GUIDE.md)。默认计划为 A/B/B/A，各 50 步；原 80 步 LR schedule 保留，截断训练迭代器到 50 步，统计窗口为 11–49：

```bash
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py --check-config
export GPU_AUTHORIZATION=GRANTED_BY_USER
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py --execute
```

`summary.json` 汇总完整四臂的有效 TGS、步时、PyTorch 显存峰值和配对/自重复数值检查。

## 重新筛选菜单

`--sweep` 执行原搜索过程：两步完整训练实采 step1/layer23 的同次 MoE 输入、权重和 dY → 单层筛选已编译菜单 → 当前配置及最多两份不同候选各 30 步 → 选吞吐最高的不同候选做四臂各 50 步 ABBA。

```bash
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py --sweep
"$PYTHON" examples/megakernel_repro/qwen_16k_liger_templates/run.py --sweep --execute
```

`capture_hook.py` 采集 checkpoint 重放中第一份有梯度调用，并用该调用的输出 hook 保存对应 dY。`single_sweep.py` 先试十份前向，再试十五份反向，最后试前两份/后三份的组合，5 次 warmup、20 次计时；每轮完成首次前向、重放和反向后才换模板。数值筛选参照同输入的当前 Liger 模板，属于 screening。安装 binary 必须覆盖 `template_inventory.json` 的菜单；该菜单不是全参数笛卡尔积。

真实采样张量只用于本次单层消费者；完成后在 run 专属目录精确清理。GPU 日志、tracker、配置和汇总保留。

## 历史结果

| 配置 | 有效 token/s/卡 | 平均步时 s |
|---|---:|---:|
| A，双 50 步合并 | 7970.69 | 8.112777 |
| B，双 50 步合并 | 8078.82 | 8.004200 |

有效吞吐增加 1.356%，平均每步减少约 108.58 ms。部分单步 grad_norm 超 5% 阈值；loss 差小于 0.01，数值状态单独记录。这组 +1.36% 是 Liger 模板收益；8079 不是相对 native 的配对加速结果。

来源为 `local-2999-qwen-liger16k-template-sweep-20261010`。采集与单层脚本沿用原实现，调整源码/输出/菜单为仓库相对路径；配置和顺序启动在共用 runner 中展开，避免历史多层 runner 依赖。
