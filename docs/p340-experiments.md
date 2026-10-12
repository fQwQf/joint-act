# P340 真机实验：摄像头＋夹爪

实现状态：数据、训练和策略接口已接通；机械臂尚未连接，标定、相机新鲜度、控制器回读、夹爪开合和运动完成标志仍需实机验收。CPU 合成数据只用于检查工程链路，不属于实验结果。

## 当前协议

单固定 RGB 相机，固定末端朝向。动作是机器人基座坐标系下的 `[dx_mm,dy_mm,dz_mm,gripper_open01]`，状态是控制器回读的 XYZ 毫米坐标加最近一次夹爪开合命令。后者不是夹爪实测位置，控制器坐标也不视作外部测量。夹爪 1 表示开、0 表示闭，通过现场标定的两个 SDK 数值换算，兼容反向开合。

模型预测 8 步，每次只执行第 1 步，等控制器报告完成后重新观察。所有基线使用相同执行接口和执行长度。当前为逐步、事件驱动控制，不宣称固定 Hz；改变动作执行长度或引入异步控制须另建协议。每步限制为校准工作区内的短距离移动，超界直接拒绝，不静默修改动作。

模型端使用单相机、四维动作和四维状态；在 P340 训练集重新拟合原型库和归一化，夹爪保持 0–1。现有 LIBERO 七维检查点不能直接控制该机械臂；从预训练 OpenVLA 初始化 P340 训练。新模块改变源码指纹，应重新 prepare-study 到新目录，保留旧仿真配置、检查点和源代码。

## 接入与示教

在连接电脑安装项目及相机/SDK 依赖，记录实际安装版本用于迁移：

```bash
pip install -e .
pip install opencv-python pymycobot
pip freeze > p340-environment.txt
cp configs/p340_limits.example.json p340-limits.json
```

示例限位是占位值，不能直接用于真机。先在厂商工具中核对基座坐标、实际可达且无障碍的工作区、夹爪开闭端点、相机固定位置及初始夹爪状态，再修改全部相关值和 calibration_id。程序不会自动回零或释放电机。不要将未验证的 is_moving_end 当成可靠反馈。官方 SDK 的部分调用可能等待，独立子进程负责超时退出；这只会停止继续发指令，**不会撤销固件中已有运动**，现场应能使用实体停止/断电措施。

摄像头后台仅保留一帧，记录的是主机接收时间，不能当作曝光时间。首次用可见变化检查相机内部缓存和延迟。observation_max_age_s 同时约束状态读取开始时间和帧接收时间；正式评估前固定这个阈值。

```bash
python scripts/p340.py collect --port /dev/ttyUSB0 --camera 0 \
  --limits p340-limits.json --output data/p340-recordings \
  --episode-id pick-001 --group-id session01-layoutA \
  --instruction 'put the red block in the left tray' --enable-motion
```

终端输入 `dx dy dz g`，例如通过校准后的小步输入 `1 0 0 1`；输入 q 结束，再明确标记 yes/no。相同采集批次和布局家族应使用相同 group-id，不要按帧或随机给每条轨迹分组。首轮每任务 20 条验证采集质量，再扩至每任务 50–100 条。当前录制原图逐帧写入 HDF5，不把完整视频留在内存。异常保留 partial 文件和事件日志；不自动当作成功示教。

## 构建数据与训练

```bash
python scripts/p340.py dataset --recordings data/p340-recordings --output data/p340
jointact validate-data --root data/p340
jointact prepare --root data/p340 --output data/p340/artifacts --horizon 8 --num-modes 64
jointact prepare-study --config configs/p340_joint.yaml --output runs/p340-study \
  --seeds 42 --variants joint aligned cost aligned_cost regression
```

转换仅接纳成功且完成的 demonstration，排除 evaluation、失败示教与 partial 文件；评估失败仍保存在事件账本中。按 group-id 稳定分配 train/val/test，生成后检查每个集合和每项任务覆盖，不靠试换 seed 选择效果更好的划分。数据、原型和训练配置一旦冻结，后续新采集单独生成版本。

训练使用 [120 GB 整机限制入口](memory-safe-running.md)，workers=0，batch=1，累积=16，单任务运行。`configs/p340_joint.yaml` 是起始实验配置，额外目标仍由 prepare-study 的各变体设置；各组共同决定步数与选择节点。控制电脑不加载 7B 权重，推理由服务器承担。

## 策略执行与结果记录

服务器上的 serve 也要放入内存受限队列；下方第一行是队列中需要执行的服务命令。控制端命令在机械臂连接电脑运行。

```bash
jointact serve --checkpoint /path/to/p340-checkpoint --device cuda:0 --precision bf16 --host 127.0.0.1 --port 8000
# 控制电脑通过 SSH 转发访问服务器的 localhost:8000。
python scripts/p340.py policy --server http://127.0.0.1:8000 \
  --port /dev/ttyUSB0 --camera 0 --limits p340-limits.json \
  --output data/p340-trials --episode-id joint-layoutA-01 --group-id heldout-layoutA \
  --instruction 'put the red block in the left tray' --max-steps 100 --enable-motion
python scripts/p340.py summarize --recordings data/p340-trials > p340-results.json
```

服务端声明的动作语义必须为 P340；日志包含权重身份、限位、相机编号、指令、步数上限、执行长度、图像接收、状态读取、推理往返、命令发出和完成时间。请求动作在执行前记录，故拒绝动作和中断不会悄悄从账本消失。汇总按模型和协议分组，列出所有成功、失败、中断、未完成试次；未完成试次存在时不给最终成功率。记录式汇总不替代配对统计或跨种子分析。

推理耗时包含 PNG 编码、网络和服务器计算；另报图像接收到命令发出、命令到完成的延迟。当前汇总保留首个动作，不删除预热，且延迟统计仅覆盖已完成动作。正式性能对比另用相同观测集、固定预热的 runtime benchmark，避免把网络耗时或机械执行时间解释成模型性能。

先比较搬运、精确放置、推物三个任务。开发阶段三方法×三任务×20次=180次；正式轮次按首轮波动确定，计划三方法×三任务×三种子×20次=540次。固定成功判据、配对布局、步数预算，并交错运行不同方法；完整五组机制消融以仿真为主。

实现依据：厂商 [SDK 源码](https://github.com/elephantrobotics/pymycobot/blob/main/pymycobot/ultraArmP340.py) 与 [API 文档](https://github.com/elephantrobotics/pymycobot/blob/main/docs/ultraArm_P340_en.md)，查阅于 2026-10-12。新电脑安装后应冻结实际 SDK 版本并执行上述实机验收。

## 工程验证记录

2026-10-12，远端 CPU 全套 51 项测试通过，Ruff 检查通过。新增路径包含毫米与夹爪极性、越界/陈旧观测拒绝、控制器故障锁定、逐帧写入与按组划分、四维模型训练及重新加载推理、错误动作语义拒绝，以及失败/未完成试次汇总。同时覆盖慢速模式设置后再次检查观测新鲜度，以及推理服务权重身份改变时拒绝执行。硬件通信和相机驱动仍待实际设备验证。
