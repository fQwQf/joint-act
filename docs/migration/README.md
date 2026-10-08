# 迁移到新训练服务器

仓库：<https://github.com/fQwQf/joint-act>（私有）。
Release：`migration-2026-10-08`。这是研究与迁移快照，尚无新增方法的闭环效果结论。

## 包含的内容

| 位置 | 内容 |
| --- | --- |
| Git | 源码、测试、固定上游版本、实验配置、诊断记录、服务器申请报告及可编辑源文件 |
| `canonical-spatial.json` + 同名 `.part-*` | 全部 432 个已转换 Spatial episode、固定划分、归一化统计、H=8/K=64 码本、原始来源记录 |
| `experiments.json` + 同名 `.part-*` | joint/regression 的 100/1000 步检查点、优化器/随机状态、原始指标、8 步新增目标检查、最终五组消融计划、旧版源码 |
| `SHA256SUMS` | 每个附件的 SHA-256；分卷清单还记录顺序和字节数 |
| 官方 Hugging Face | 固定 revision 的 OpenVLA 7B 基础模型，由下载脚本获取并逐文件校验 |

数据和模型文件不进入 Git 历史。两个压缩包均分为不超过 512 MiB 的附件。
不复制原机器的虚拟环境、账户凭据或缓存；在新服务器安装依赖。
数据沿用上游许可和来源，详见 `THIRD_PARTY.md`，本仓库不改变其许可。

## 1. 下载并校验

建议在持久化大容量磁盘准备至少 80 GB 空间；同时保留压缩包和解包数据会占用更多空间。
需要 `git`、`gh`、`zstd` 和 Python 3.10。使用有仓库读取权限的 GitHub 账户登录。

```bash
gh auth login
gh repo clone fQwQf/joint-act /data/jointact/repository
cd /data/jointact/repository
gh release download migration-2026-10-08 --repo fQwQf/joint-act --dir /data/jointact-download
cd /data/jointact-download
sha256sum -c SHA256SUMS
cd /data/jointact/repository
python3 scripts/migration.py restore \
  --manifest /data/jointact-download/canonical-spatial.json \
  --destination /data/jointact-data-stage
python3 scripts/migration.py restore \
  --manifest /data/jointact-download/experiments.json \
  --destination /data/jointact-experiment-stage
```

恢复脚本先校验每个分卷，再流式解包；拒绝已存在的恢复目录、越界路径及链接。
每个恢复目录都应是新目录。完整数据验证完成前保留原包。

## 2. 保留检查点内的路径语义

旧检查点、训练数据和消融计划使用 `/tmp/jointact-1019` 作为逻辑根。
在新主机上让这个路径指向持久化磁盘，便可保持配置及其哈希原样。
**先确认该路径不存在或就是本项目；不要覆盖其他目录或链接。**

```bash
mkdir -p /data/jointact
# 以下目标子目录在首次恢复前应不存在。
mv /data/jointact-data-stage/libero-spatial-full /data/jointact/
mv /data/jointact-experiment-stage/study-spatial /data/jointact/
mv /data/jointact-experiment-stage/study-alignment /data/jointact/
mv /data/jointact-experiment-stage/study-alignment-v2 /data/jointact/
mv /data/jointact-experiment-stage/repo /data/jointact/repo-original
ln -s /data/jointact /tmp/jointact-1019
```

如果 `/tmp/jointact-1019` 已存在并属于其他工作，请另选主机/隔离环境，或显式设计
路径重定位并重新建立配置来源记录，不要批量改写已归档的实验配置与 manifest。
一些集群会清理 `/tmp`，重启后需要重新建立这个链接；真实数据仍在 `/data`。
`repo-original` 保留早期实现供复查；新实验使用刚克隆的 `repository`。

## 3. 安装运行环境与基础模型

3090 上实际验证的是 Python 3.10、PyTorch 2.9.1+cu128、torchvision 0.24.1、
Transformers 4.40.1、PEFT 0.11.1、timm 0.9.10。精选依赖版本见
`runtime-constraints.txt`，完整环境清单仅用于审计，包含原主机不相关的包，不应整份安装。
下面是按该组合重建的安装步骤；新服务器、驱动及 GPU 组合仍需现场验证。

```bash
python3.10 -m venv /data/jointact/venv
source /data/jointact/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
cd /data/jointact/repository
python -m pip install -c docs/migration/runtime-constraints.txt -e '.[openvla,dev,serve,video]'
export HF_HOME=/data/jointact/hf
python scripts/download_base.py --output /data/jointact/pretrained-openvla7b
jointact validate-data --root /data/jointact/libero-spatial-full
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q
jointact doctor --path /data/jointact
```

基础模型固定为 `openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f`，
约 15 GB，下载后校验三片权重以及配置、tokenizer 和模型实现文件。
已通过其他方式传入同一模型时，可加 `--verify-only` 只做校验。
不需要重新下载 RLDS 或重新转换数据；使用迁移包中的 canonical 数据即可。

Pro100 的确切型号与驱动尚未提供。若上述 PyTorch wheel 不支持其架构，应先选择支持
该 GPU 的版本并记录环境变更，不能直接声称获得了原 3090 环境的逐位复现。

## 4. 恢复与扩展实验

两个 1000 步父检查点均带 `training.pt`，可延续优化器、学习率调度与数据位置。
最终计划 `study-alignment-v2` 的五组实验尚未执行。`study-alignment` 中仅有组合损失
1000→1008 的工程检查；它不是正式消融结果。普通 resume 不允许改变新损失设置。

```bash
# 使用明确分配给本项目且空闲的 GPU；保留原 world_size=1 和 accumulation=16。
CUDA_VISIBLE_DEVICES=0 jointact train \
  --config /tmp/jointact-1019/study-alignment-v2/configs/aligned_cost-seed42.yaml \
  --stop-after 1100

# 后续从这个新分支自己的检查点继续；1100/2000 是绝对优化器步数。
CUDA_VISIBLE_DEVICES=0 jointact train \
  --config /tmp/jointact-1019/study-alignment-v2/configs/aligned_cost-seed42.yaml \
  --resume /tmp/jointact-1019/study-alignment-v2/runs/aligned_cost-seed42/checkpoints \
  --stop-after 2000
```

其余配置为 `joint`、`aligned`、`cost`、`regression`，均位于同一 `configs/` 目录。
多个空闲 GPU 可各跑一个独立变体，使用 `CUDA_VISIBLE_DEVICES=N`；这样无需改变原单卡
父检查点的世界大小。把单卡检查点直接改成 8 卡 DDP 精确续训会被拒绝。
若要 8 卡 DDP，需从固定基础模型重新生成实验，统一有效批量与预算并单独报告。

`jointact run-study --plan /tmp/jointact-1019/study-alignment-v2/study.json --device cuda:0`
会运行完整 20,000 步计划并评测完整 val/test，不是短检查命令；运行前按分配的算力安排预算。

## 5. 仿真评测

LIBERO、MuJoCo、robosuite 和渲染设置见 `docs/libero.md`、`docs/remote-workspace.md`。
固定 LIBERO revision 记录在 `upstream.lock.json`，无需重新采集示范。
新主机应重建 LIBERO 路径配置；不要复制旧主机的 OSMesa 系统库或虚拟环境。
可用 GPU 无头 EGL 或安装当前系统的 OSMesa。先检查环境与完整试验协议，再执行
同初始状态、同执行 horizon 的配对评测。数据验证/训练损失不等于闭环成功率。
