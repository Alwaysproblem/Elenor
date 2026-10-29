# P21 · C8-A Rocket / RoCC 硬件边界

状态：未开始
对应原方案：§4.5（Rocket / RoCC UCE）、§7.2 C8-A、§7.4.2/§7.4.6（模块划分与 RoCC 接口）
验证层级：R（真实 Rocket 构建 + Verilator 联仿中的前端合同）；信号级驱动仅是本任务 harness 证据。

## 1 目标与不做

目标：

1. 真实 Rocket 系统构建：按 §1.4 候选 pin 实际构建 Chipyard harness（RocketTile + 本任务 BoaFrontend），锁定 transitive dependencies 后纳入 P02 工具链；干净 elaborate + Verilator 仿真可运行。
2. BoaFrontend：实现 §4.5 六条自定义指令到 P18/P20 已验收的 allocator/event/mailbox 端口的翻译；resp 有限周期返回，busy 不得让 hart 等整条链；descriptor/mailbox 走 TCM uncached 路径并有 release/acquire 顺序；不假定 Rocket D-cache 与 BOA L1 SPM 自动一致。
3. 指令编码唯一来源：opcode/funct/字段由 P10 权威 schema 生成 Scala 常量（schema 若未含 RoCC 编码，本任务扩展 schema 并凭评审冻结，不手写三套编号）。
4. 明确边界：仅 RoCC 拉信号（testbench 直接驱动 cmd/resp）不算 P22 真实 UCE；本任务只证前端合同。

不做：固件与 Tile Program 执行（P22）；Python UCE 互斥（P22）；FPGA/板级（P23+）；前端内建调度器（禁止第二套 scheduler，总则 §4.1 P22 为接线 owner）；BOA 微架构修改（P11–P18 范围）。

## 2 前置与输入

- P10：wire/descriptor schema 与 codec；若需扩展 RoCC 指令编码字段，按 P10 的冻结流程补录后再生成。
- P20：`validator-cosim` 通过——前端必须在已联仿的单阵列系统上驱动，不能对空壳 top 拉信号。
- P02：锁定工具链（Scala/Chisel/RISC-V/Verilator）；`--case ID` 公共分派已存在，本任务注册 `rocket-uce` target 与信号级 case 清单。
- 输入：§1.4 候选 pin——Chipyard `e02074414158c85059c1dcb231ff2f8941254b9a`、Gemmini `25809f78323a729ef76fb68f3cedd8a24da2942b`、Rocket `c7aabd2823d3d001a7a9e8a982bb65d6cb7c9bdf`、Scala 2.13.12/Chisel 6.5.0。必须真实构建后才能称为已验证；不兼容时带证据调整整套 pin，不切 latest。

## 3 当前证据、必读与修改边界

- 仓库当前无 `hardware/`、无 Chipyard 集成；`runtime/firmware.py::FirmwareRuntime` 只是 Driver-Firmware 3.3 的取指/校验延迟模型，与本任务无关，不得当作真实 UCE 证据。
- 前端落点为 `hardware/uce/src/main/scala/`：BoaFrontend、ConfigFragment与TCM/uncached接口，按原方案唯一目录，不另建hardware/rocket。
- 修改边界：不改 P18 top 内部；`BoaFrontend` 只消费其 allocator/event/mailbox 端口；validator 侧无改动。
- 必读：§4.5 全节（指令表、固件调度约束、resp 合同）、§7.4.2 结构图、P18/P20 evidence 交接。

## 4 接口与不变量

- 六条指令语义与 resp（§4.5 表）：`AM_REQ`→GRANTED(lease)/PARKED(req)/ERROR；`BOA_SUBMIT`→ACCEPTED(token)/FULL/ERROR；`EVT_STATUS`→非破坏性电平位图；`EVT_TAKE`→record；`REC_REAP`→ok/ERROR（不影响 AM）；`CTX_CANCEL`→ok。resp 只带状态与短 token，完整 descriptor/record 在受保护 mailbox。
- AM_REQ/BOA_SUBMIT短响应须有有限上界，区分resp.ready可用时内部响应界和外部持续背压。CTX_CANCEL先受理取消/停止提交，旧访问quiesce与回收异步结束后才发失败终态，不得等整链结束长期占RoCC。FULL不消费seq，EVT_STATUS非破坏性，TCM发布release/消费acquire。
- 最小 cmd/resp 并发：resp 未取走期间前端不得丢弃该响应；同 context 的下一条依赖指令由固件排序（V1 无中断，读位图只是观察）。

## 5 实施步骤

1. 扩展 P10 schema（若缺 RoCC 编码）→ 生成 Scala 常量与 C header；记录冻结依据。
2. Chipyard harness：BOA RoCC ConfigFragment 叠加进候选基线；真实构建，锁依赖；输出 elaborated RTL 与 firrtl 报告。
3. `BoaFrontend` 实现：cmd 译码 → 对 allocator（AM_REQ）、step mailbox（SUBMIT）、EventTracker（EVT_*/REC_REAP）、TileAmAllocator（CTX_CANCEL）的端口事务；resp 打包。
4. TCM/uncached：descriptor/mailbox 地址窗口、release/acquire 栅栏映射、访存端口证据（波形可见 uncached 通路）。
5. 与 P20 backend 对接的联仿 harness：Rocket Verilator top 内 BoaFrontend 驱动 BOA；testbench 信号级驱动 cmd/resp 供前端合同测试。
6. 注册 `rocket-uce` target（信号级 stage）；固件级 case 由 P22 续接同一 target。
7. 由有限队列、仲裁和TCM服务假设推导响应界，再用极端背压联仿核验；测得最大值不是上界证明。记录外部ready/内存最终服务前提，旧链drain不进入AM_REQ短响应依赖。

## 6 验收表

| Case ID | 构造 | 必须观测 | 失败特征 |
| --- | --- | --- | --- |
| rocket-build | 干净目录构建+elaborate | 构建/elaborate 成功，依赖 hash 入 manifest | 仅打印版本 |
| fe-amreq-granted | 信号级 AM_REQ（空闲 allocator） | GRANTED(lease) 在 bound 内返回，slot/quota/mailbox 一次性取得 | 半授予（占槽缺元数据） |
| fe-amreq-parked | 信号级 AM_REQ（不可授予） | PARKED 返回且无 slot/quota/generation 副作用；slot_free 后位图置位 | 前端自行重发或占资源 |
| fe-submit-full | mailbox 填满后 BOA_SUBMIT | FULL 在 bound 内返回，seq 不消费，descriptor 不变 | 静默吞步骤 |
| fe-evt-status | 制造就绪/未就绪事件 | 位图电平正确，连续读状态不变 | 破坏性读 |
| fe-ctx-cancel | 挂起请求+live lease 时 CTX_CANCEL | 请求撤销、lease 回收、bundle 归还；被取消请求永不唤醒 | 唤醒僵尸请求 |
| fe-tcm-uncached | descriptor 写+读 | 波形显示 TCM/uncached 通路与 release/acquire 顺序 | 走 D-cache 缓存行 |
| fe-illegal-instr | 非法 funct/opcode | ERROR resp，无 BOA 状态变化 | 静默忽略 |

## 7 可执行命令

```bash
# 构建与前端合同（均为本任务交付入口，拟新增；容器由 P02 env 提供）：
hardware/scripts/check rocket-uce --out /tmp/p21 --seed 1 --case rocket-build
hardware/scripts/check rocket-uce --out /tmp/p21 --seed 3 --case fe-amreq-parked
# 联仿载体复用 P20 backend 运行同一 artifact（信号级 stage 仅到前端合同，不做固件断言）
```

预期：`summary.json` 标 validation_layer=R；resp bound 测量值与冻结配置入 manifest。

## 8 产物、交接与停止条件

产物：可构建的 Rocket+BoaFrontend 系统、schema 扩展与生成常量、TCM 映射、`rocket-uce` 信号级 cases、resp bound 配置、`plan/evidence/P21.md`。
交接 P22：指令编码 header、mailbox/事件位图地址、bound 配置、harness 入口；交接 P23：可综合 ConfigFragment 与 elaboration 报告。
停止条件：候选 pin 无法构建且整套调整仍失败；前端任一合同 case 不可证。信号级通过不得写成真实 UCE 通过。
