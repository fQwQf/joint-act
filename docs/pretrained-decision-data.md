# 预训练模型改造与机器人数据来源

检索与源码核对日期：2026-10-08。本文区分源码事实、公开数据元信息和方法建议；没有把社区模型的演示或自报成绩当作我们的实验结果。

## 结论

可以采用 Open JEV 类项目的做法：保留预训练模型主干，把自回归答案生成改为直接决策读出。对低层机器人控制，首选已有机器人预训练的 OpenVLA，而不是纯文本 LLM。公开 LIBERO 演示已经包含第一阶段所需的图像、指令、本体状态和连续动作，不需要重新人工标注一套“Jev 机器人数据”。

权重继承解决预训练成本，演示数据解决动作监督。输出机制改变之后仍需要适配训练。现有证据支持工程路线可行，尚不能证明我们的联合动作表示优于连续回归或 OpenVLA-OFT。

## 已核对的实现

| 项目与源码版本 | 实际做法 | 数据从哪里来 | 可以迁移的部分 |
| --- | --- | --- | --- |
| [zhihz/openjev](https://github.com/zhihz/openjev/tree/ff94f61ec3b2d6a71b4c2b01e76104dfe37da4e6) | 冻结 Qwen3-4B-Instruct-2507；读下一位置的候选字母 logits，做 restricted softmax；当前逐问题执行 | 部署路径不需要额外训练数据，评测仍需要真值 | 建立不生成解释的预训练基线，检查候选 token 和提示偏差 |
| [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev/tree/bd4118882f733574a3250a4b65fe4d884130c08b) | 保留 Qwen 语言主干；最后有效位置接标量评分头；LoRA 适配；每候选一个文本分支 | 公共标签转换、可执行游戏/规则、合成工作流等；不同发布版的训练混合不同 | 复用主干、从旧输出层初始化评分头、区分硬标签与软标签、独立校准 |
| [OmniJev/OneJev](https://github.com/OmniJev/OneJev/tree/7e3007d7829c7f59f5af15c60fb8454a11e37170) | 保留多模态基座及原词表；在答案读出位置选择候选字母 logits；公开配方全参数微调语言部分，冻结视觉塔 | 原训练 99,193 道题；公开 94,707 道题，17.7 GB；大部分来自公开数据原标签或实际轨迹结果，部分规则数据用教师标注 | 从 VLM 直接转为一次前向的决策模型；保留已有视觉能力；复用数据标签而非统一依赖蒸馏 |
| [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9ecdca45a9bcef17991a07d3041a87685a) | Qwen3-0.6B 主干接候选评分头；Choice 可加集合注意力；完整候选路径批处理 | 游戏状态与程序解、已有专家策略/轨迹、明确区分的教师分布和真实结果标签 | 从专家行为学习动作选择；整条轨迹划分；区分行为监督与执行结果监督 |

这四个项目是社区系统/工具源码，不能据此推断闭源 Jev 的内部架构或 RLCD 细节。

### Open-Jev 的关键代码

[jev/model.py](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/model.py) 使用原语言模型输出矩阵的两行初始化新评分头：

```text
w_score = W_lm[Yes] - W_lm[No]
s(x,c) = w_score · h(x, question, candidate)
p(c | x, candidate_set) = softmax_c(s(x,c))
```

在相同隐藏表示下，二分类 sigmoid 与原 Yes/No 两项归一化的概率相同。多候选 softmax 则是另一种建模选择，不能把这种初始化等价性推广为多候选准确率保证。其后训练 LoRA 和评分头，使用交叉熵与 Brier 项。

该实现提取的是语言主干，视觉塔被丢弃。不能把它加载自多模态模型这一事实解释为其部署路径支持图像。机器人模型必须保留视觉塔和视觉投影。

逐候选路径可以在一次 batch 前向里运行，但默认路径重复编码上下文。一次 `forward()` 调用并不意味着计算量与候选数无关。我们的固定动作原型直接由一个上下文特征输出 K 个 logits，避免让主干分别阅读 K 个数值动作描述。

### OneJev 的关键代码

[train/common.py](https://github.com/OmniJev/OneJev/blob/7e3007d7829c7f59f5af15c60fb8454a11e37170/train/common.py) 的 `readout_logits` 与 `slot_view` 分别读取答案位置和选择允许的字母；[train/sft.py](https://github.com/OmniJev/OneJev/blob/7e3007d7829c7f59f5af15c60fb8454a11e37170/train/sft.py) 的 `slot_loss` 使用 restricted-softmax CE+Brier。答案本身不作为已生成前缀送入该读出位置。

[4B 配方](https://github.com/OmniJev/OneJev/blob/7e3007d7829c7f59f5af15c60fb8454a11e37170/train/configs/onejev_4b_full.yaml) 明确设置 `multimodal: true`、`full_finetune: true`、学习率 `5e-6`、一轮训练和 `brier_weight: 1.0`。这是一条微调预训练模型的路线，不是从随机权重训练决策模型。

[公开数据卡](https://huggingface.co/datasets/OmniJev/OneJev-Data) 列出 GUI、图像、短视频、程序视频、文本和规则来源。其数据可供学习转换方式，但不包含我们需要的连续机械臂控制监督；不能直接替代机器人演示。

## 机器人数据的具体入口

首选 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)，与原 OpenVLA 和 OpenVLA-OFT 的 LIBERO 微调数据来源对齐。核对时仓库 HEAD 为 `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`。

以下数量为该版本 `dataset_info.json` 中 **train split 的 shardLengths 之和**，没有把切出的滑窗当作独立轨迹：

| 套件目录 | 轨迹数 | 元数据 numBytes，十进制 GB | 原始元数据 |
| --- | ---: | ---: | --- |
| libero_spatial_no_noops | 432 | 1.915 | [Spatial](https://huggingface.co/datasets/openvla/modified_libero_rlds/blob/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_spatial_no_noops/1.0.0/dataset_info.json) |
| libero_object_no_noops | 454 | 2.817 | [Object](https://huggingface.co/datasets/openvla/modified_libero_rlds/blob/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_object_no_noops/1.0.0/dataset_info.json) |
| libero_goal_no_noops | 428 | 1.842 | [Goal](https://huggingface.co/datasets/openvla/modified_libero_rlds/blob/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0/dataset_info.json) |
| libero_10_no_noops | 379 | 3.657 | [Long](https://huggingface.co/datasets/openvla/modified_libero_rlds/blob/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_10_no_noops/1.0.0/dataset_info.json) |
| 合计 | 1,693 | 10.231 | 元数据相加；磁盘占用另含缓存和转换副本 |

OpenVLA-OFT 的 [LIBERO 配方](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md) 给出这四个目录，说明过滤了近零动作，并公开套件独立微调和联合微调路径。原始 LIBERO 的“每任务约 50 条演示”不能直接替代修改版的实际轨迹计数。

每个 timestep 提供第三人称 RGB、腕部 RGB、8D 本体状态、7D 连续末端控制动作及语言指令。图像朝向、旋转表示、夹爪极性和归一化需遵守 [本工程的数据契约](data.md)，不能只对齐 tensor 形状。

另一入口是 [官方 LIBERO HDF5 演示](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets)。它包含本工程可直接导入的图像和动作，但处理条件与 modified 数据不同：例如 no-op 过滤。主结果统一采用同一来源和预处理，以免把数据差异计入方法收益。

原 OpenVLA 基座已经利用 [970K 机器人轨迹预训练](https://github.com/openvla/openvla#pretrained-vlas)。复用基座时无需重新下载、重做这一预训练；如后续需要广泛跨场景适配，再考虑 BridgeData V2 或 Open X-Embodiment 子集。不同机器人控制坐标、频率和夹爪规范须分别转换。

## 如何把演示变成我们的训练标签

先按完整父轨迹划分 train/validation/test，再从训练轨迹拟合动作归一化和原型。对时间 t：

```text
input  = (I_agent,t, I_wrist,t, instruction, proprio_t)
target = A_t = [a_t, ..., a_(t+H-1)] ∈ R^(H×7)
k*     = nearest_prototype(normalize(A_t))
delta* = normalize(A_t) - C[k*]
```

`H=8, K=64` 是现有工程的起始配置，不是已验证的最优值。原型通过整个 `H×7` 向量联合聚类得到，残差保留细粒度精度。训练时没有把 ground-truth 动作输入主干；部署时先由模型预测 k，再计算对应的连续修正。

一个完整动作块同时监督模式分类和连续动作重建。因此第一阶段的额外人工标注数为零，额外教师调用数为零。切窗数量为 `sum_episode max(T-H+1, 0)`；相邻窗口相关，不能宣称获得了同样多的独立演示。

数据只能表明某状态下专家执行了什么。其他原型没有出现在这条演示中，不能因此标成执行失败；分类分布解释为行为模式分布。若日后要预测成功概率，必须在仿真中执行候选动作并记录明确后续策略下的结果，另建结果数据。

## 权重改造的适用边界

原自回归 OpenVLA 学习的是：

```text
p(a_1,...,a_D | o,l) = ∏_d p(a_d | o,l,a_<d)
```

保留第一个动作位置的 logits，可以得到第一个动作 token 的预训练分布。把同一位置的输出复制七次，或把原来依赖真实前缀的位置改成占位符，不能保持后续坐标的条件分布。增加时间块后，时间相关性也需要适配。

可执行路线分三档：

1. **冻结主干＋训练读出头**：低成本测量原特征是否足够，作为诊断。
2. **预训练 OpenVLA＋LoRA＋联合模式/残差头**：主线。保留视觉塔、投影和因果语言主干，通过一个 joint query 直接读出完整动作块。
3. **更大范围微调或机器人教师蒸馏**：在前两档显示明确瓶颈后使用。教师必须具备对应机器人域的动作能力；通用 LLM 自报的关节动作不能作为可靠控制真值。

Open-Jev 的 Yes/No 初始化可以在语义一致的评分器里直接复用。新的 K 个动作原型没有现成的 K 行语言词表语义，不能套用同样的初始化等价性。原 OpenVLA 的动作-bin 输出行可以用于保留旧动作 token 的对照，但这仍不能绕过从串行到并行的训练分布变化。

采用冻结视觉塔与 LoRA 仍是端到端策略：推理直接从观测到低层控制，训练目标直接监督动作，并通过可训练模块反传。端到端不要求所有参数全量更新。

## 要突出和验证的优势

改造不减少 7B 主干参数，视觉编码和输入 prefill 仍有成本。LoRA 主要降低训练参数及优化器开销；非自回归读出减少串行生成。用一个 batch 评分 K 个完整候选路径则会增加主干计算，这两种实现不可混为一谈。

[OpenVLA-OFT，RSS 2025](https://arxiv.org/abs/2502.19645) 已研究并行解码、动作块和连续回归。因此“借鉴 Jev，把 OpenVLA 改成一次前向”有充分实现先例，但自身不能构成独创性。当前待验证的主张是：在相同预训练权重、演示数据、相机、本体状态和执行 horizon 下，联合模式＋残差是否比直接连续回归更好保留多模态动作选择，且保持接近的端到端延迟。

动作离散化配合连续修正也已有 [BeT，2022](https://arxiv.org/abs/2206.11251) 的先例；[VQ-BeT 官方实现，ICML 2024](https://github.com/jayLEE0301/vq_bet_official) 提供了残差向量量化的训练路线。这部分机制不能归为 Jev 同期才出现的创新。当前工程是可复现的研究候选；论文还需要说明相对这些已有方法的实质区别，并给出相应证据。

第一轮使用 Spatial 公共演示：冻结特征读出、LoRA 联合头、同主干连续回归各一组；在同一仿真初始状态测任务成功率和实际决策延迟。模式命中率、量化误差和 L1 只用于诊断，不能替代闭环结果。数据比例实验应按父轨迹抽样并与对照匹配，而不是抽相邻帧后声称低数据需求。

截至本文核对，公开数据和源码已确认；本工程在真实预训练权重及真实演示上的性能，仍以 [验证记录](validation.md) 和之后实际闭环实验为准。
