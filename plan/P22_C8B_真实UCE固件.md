# P22 · C8-B 真实 UCE 固件与控制接入

状态：未开始
对应原方案：§4.5（固件调度）、§5.4（内存与真实 UCE 接入）、§7.2 C8-B、§8.1b Q09
验证层级：R（Rocket 固件实际取指执行 + 联仿对拍）；板级复现归 P24。

## 1 目标与不做

目标：

1. 真实固件（RISC-V C/汇编，Rocket 上取指执行）执行 Tile Program：调度循环、四类 context 状态（READY/WAIT_EVENT/WAIT_AM/WAIT_ENGINE_QUEUE）、EVT_STATUS 轮询一次选 context、EVT_TAKE 取 lease、PC 越过 ALLOC、绝不重发 `AM_REQ`。
2. admitted capsule 与控制边界：compiler 新增 emitter（拟新增）把已验证 typed capsule 变成固件镜像（descriptor 表 + step 流）；Group admission 经 ControlBackend 等价边界（拟新增）提交 capsule，真实 UCE 负责 L1 view、MFE 请求、AM_REQ/BOA_SUBMIT、等待与错误分支、phase signal 和 return。
3. 同 Task 互斥：同一 Task 不能同时由 Python UCE 与 Rocket 解释；不支持的 Matmul 指令闭包明确拒绝，不回退 Python 执行（§5.4）。
4. Q09 与错误收尾、8 contexts 配置、FINISHING 退休。

不做：新源级 CFG/TRY/await_any（IR13 延期维持，总则 §3.4/§6）；通用 DAG 执行器、共享内存 atomic；RoCC interrupt 优化；板级 F 项（P24）；前端微架构（P21 已交付）。

## 2 前置与输入

- P20：`validator-cosim` 通过（Python UCE→RTL 对拍基线、事件批/写回报通路）。
- P21：Rocket+BoaFrontend 系统可构建，指令编码 header、mailbox/位图地址、resp bound 可用。
- 输入：P06 cutover 后 typed artifact（版本按届时规则）；P03 oracle bytes；`examples/` typed matmul workload；已冻结的后处理/数值 profile（P03/P06）。

## 3 当前证据、必读与修改边界

- 现状：validator 的 context 调度在 `tile.py` Python UCE（`_select_context/_issue_context`，`tile.py:319 step`）；AM 一次申请/挂起/唤醒语义在 `runtime/shared_am.py:206 request`（PARKED 不占 slot/quota/generation，wake 一次）。固件必须复刻该语义而不是重设计；Q 项软件证据（`tests/test_boa_am.py:938-1097` 等）只是模型层，硬件层须按 §8.1b 在 R 复现。
- emitter/ControlBackend 均不存在：`pipeline_validator/compiler/` 与 `runtime/group_port.py:59 GroupPortAdapter` 是现有 admission 通路参照；新边界为拟新增，命名与落点在 evidence 定稿。
- 修改 `pipeline_validator/compiler/` 的capsule emitter、`runtime/` 与实际Group admission调用点，以及 `hardware/uce/firmware/` 的固件/linker；复用P21系统，不另建hardware/rocket或源CFG。
- 必读：§4.5 指令表与调度约束、§5.4 两条控制路径比较规则、§8.1 C8 相关 T 项（T01/T08/T14/T17/T19）、Q01–Q07/Q09 硬件层列。

## 4 接口与不变量

- 通用调度固件ELF与任务数据分离：capsule包含artifact派生的身份/资源/事件表、typed views、descriptors和有限指令序列，不为每任务重编一个固件。TCM/linker映射、ELF和capsule分别hash。
- 调度不变量：每 owner 至多一条挂起请求；WAIT_AM context 只在唤醒位置位后可运行，取 lease 后 PC 越过 ALLOC；`BOA_SUBMIT` FULL → WAIT_ENGINE_QUEUE，恢复后重试**同一步**（同 seq、同一 descriptor 快照，descriptor immutable）；完成先进入 UCE-local，等待仅由位图驱动；读位图不驱动释放（granted/drain/slot_free 由设备自主）。
- 生命周期：phase signal（Group `input_released` 语义 §5.1）、context return；return 后有 FINAL drain → FINISHING，缺 FINAL 是错误；FINISHING 后释放/退休通知不依赖已返回 context 再 poll。
- 最小固件闭包须完整覆盖L1 allocation/view/free、MFE load/store及完成、AM allocation、typed Matmul四模式、await、phase signal、return及故障收尾；列出对应现有executable opcode。支持集外操作在admission拒绝。执行选择control_backend=rocket与boa_backend=rtl_cosim分离，同Task只能绑定一条ControlBackend。
- 错误：第三条未受理 FINAL 链的 AM_REQ 是 ERROR/fault（T01 运行时注入分支）；错误 terminal 不伪装成功；旧 generation 响应不得污染新 owner（T17）；故障/取消后计数入独立分母。

## 5 实施步骤

1. 从新版compiled artifact提取完整闭包，生成独立capsule，不省略MFE/L1后让host代做；复算bundle/事件数、检查views/output绑定、schema/fingerprint、opcode及初始化信息。
2. ControlBackend绑定既有Group admission/phase/retirement，经受保护TCM路径提交capsule。新增 `--sim-override control_backend=rocket`、默认python，同Task双绑定明确拒绝；加载不隐式推进cycle。
3. 固件内核：调度循环（每轮一次 EVT_STATUS）、WAIT_AM/WAIT_EVENT/WAIT_ENGINE_QUEUE 转移、EVT_TAKE、同 step FULL 重试、phase/return/FINISHING、CTX_CANCEL 错误分支。
4. 按硬件capability/admission容量配置context表，默认4并验1/2/4/8，不写死成不可配置8；八contexts各bundle1的L1/事件容量也须足够。
5. 联仿对拍：同 artifact 在 Python-UCE（P20）与 Rocket 固件两控制路径运行；比较 C bytes、所有权与合法事件偏序；周期不要求相同，取指/调度/轮询开销计入并记录（§5.4）。
6. Q09/错误回归：Q09 序列在固件层复现；T01 注入、T14 FULL、T17 换代、T19 FINISHING/缺 FINAL 各一组。
7. 注册真正固件级rocket-uce cases，与P21信号级区分；reproduce重新启动Rocket RTL执行ELF/capsule，不能仅重放Python部分或读旧日志。
8. 文档同步：原方案 §4.5 中暂定指令名与固件描述按 P21/P22 冻结结果更新，保留历史表述标签。

## 6 验收表

| Case ID | 构造 | 必须观测 | 失败特征 |
| --- | --- | --- | --- |
| fw-matmul-bytes | typed matmul capsule 固件执行 | C bytes 与 oracle、与 Python-UCE 路径逐 bit 一致 | 任一路径单独通过 |
| fw-event-order | 同上 | 事件偏序合法且与 Python 路径一致；取指/调度开销计入 report | 忽略固件开销 |
| fw-wait-am-once | 槽满迫使 PARKED | `AM_REQ` 恰好一次；wake 位图置位后才运行；无重发（重发计数=0） | 轮询重发 AM_REQ |
| fw-full-retry | mailbox 满返回 FULL | 同 seq、同 descriptor 重试同一步，无 seq 跳变、无 descriptor 变异 | 重新快照或推进 seq |
| fw-q09 | FINAL(slot0)→ALLOC(slot0)→await(旧事件)，无中间 reap | 旧链自主 slot_free 后新申请授予，旧事件仍可 await，不挂死（§8.1b Q09） | 等待事件退休才授予 |
| fw-bundle-error | 注入第三条未受理 FINAL 链申请 | ERROR/fault、无 AM 副作用、独立分母记账 | 挂起等待 |
| fw-finishing | return 时存在 FINAL drain；缺 FINAL 注入 | 前者 FINISHING 自主收尾；后者错误终态；释放不依赖返回 context | 伪成功或挂死 |
| fw-8ctx | `--context-mode 8` capsule | 8 context 各持一槽全部推进完成，互不串扰 | 表溢出或互踩 |
| fw-reject-closure | capsule 含固件不支持操作（如 Gather/Scatter） | admission 明确拒绝，不回退 Python UCE | 静默回退 |
| fw-mutex | 同 Task 双路径同时启用尝试 | 互斥生效，第二路径被拒 | 双解释同 Task |
| fw-stale-gen | 换代后旧响应到达 | 丢弃计数，新 owner 无污染 | 旧字节入新 lease |

## 7 可执行命令

```bash
# 本任务runner按配置生成artifact/capsule、完整A/B/C bindings和golden，
# 分别执行Python UCE与真实Rocket ELF，不要求预造/tmp中间文件。
hardware/scripts/check rocket-uce --config hardware/configs/single_array.yaml \
  --out /tmp/p22/q09 --seed 5 --case fw-q09
hardware/scripts/check rocket-uce --config hardware/configs/single_array.yaml \
  --out /tmp/p22/ctx8 --seed 8 --case fw-8ctx
# 单独CLI复现时消费manifest完整配置/bindings/input，并同时指定
# --hw-override boa_backend=rtl_cosim 与 --sim-override control_backend=rocket。
# 只选RTL BOA仍然是Python UCE，不是本任务的真实固件验收。
```

预期：`summary.json` 标 R 层；板级复现按 P24 的 `--program-manifest/--serial/--jtag-device` 接口另行执行，不在本任务宣称。

## 8 产物、交接与停止条件

产物：emitter、ControlBackend 边界、固件源+linker map+hash、`rocket-uce` 固件 cases、`plan/evidence/P22.md`、原方案 §4.5 文档同步。
交接：P23 用本任务的 Rocket+BOA config 与固件镜像做单阵列综合；P26 依赖本任务互斥/多 context 行为。
停止条件：Q09 或互斥任一不可证；capsule 闭包白名单无法从 artifact 判定；阻塞须列明缺失输入。
