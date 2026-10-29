# P16 · C5-H：credit、AM RMW 与有限服务

| 项   | 值                     |
| ---- | ---------------------- |
| ID   | P16（对应原方案 C5-H） |
| 前置 | P12、P14、P15          |
| 状态 | 未开始                 |

## 1. 目标与不做

**目标**：实现有限周期资源层：每真实 FIFO 的 O+R≤Q credit（发射按最小不可暂停微块预留、到达仅 R→O、下游有保障后 dequeue 才释放）、`pending_am_updates` 与 step_committed 判定、ResultCapture/AmRouter、AM group 的 bank/adder 独立服务与 16-lane FP32 RMW 地址互锁、bound-engine ready 仲裁、drain 公平端口服务、以及 §6.2 的阻塞原因分解计数。

**不做**：不实现 epilogue 后处理与 FINAL 自动释放次序本体（P17，但本层提供 credit 基础设施与 L1 写端口约定）；不重复 P14 的授予逻辑；不做微块级周期与软件 step 级模型的周期等值声称（§3.5：只要求边界顺序与不变量一致）；不建 UCE/固件。

## 2. 前置与输入

- P12 通过：`min_nonstoppable_outputs` 合同数字与原始 compute_done 接口。
- P14 通过：ChainTable 游标、mailbox、bound_engine；P15 通过：事件发射时机约束（chain_done 在 slot_free 后）。
- 原方案必读：§3.4（ready 仲裁与 FULL/WAIT_ENGINE_QUEUE）、§3.5（O/R credit、pending_am_updates、commit 公式、服务资格与全成全败）、§3.6（阶段次序）、§6.2（观测分解表）、§7.2 C5、§7.4.3（group 端口）、§8.1 T09–T12/T20/T22。
- 现有仓库证据（step 粒度软件模型，边界参照）：`pipeline_validator/runtime/boa_result.py:72 BoaResultService`（launch 原子预留 1 FIFO 项+2 credit，compute_done→commit→output→drain→release→event）；`hardware_config.yaml:104-112`（result_fifo_depth=4、commit_cycles=2、output_drain_cycles=1、am_credit_capacity=8、l1_output_credits=1）。硬件按微块预留，数值不必相同。

## 3. 必读文件与修改 ownership

- 必读：本文件、总则 §3.2/§5、`plan/evidence/P12.md`、`plan/evidence/P14.md`、`plan/evidence/P15.md`。
- 本任务新建：`hardware/boa/src/main/scala/boa/service/ResultCapture.scala`、`AmRouter.scala`、`AmGroup.scala`（2 slot URAM 映射+16-lane FP32 RMW 流水）、`CreditTracker.scala`、`ReadyArbiter.scala`（均拟新增）；测试 `hardware/boa/src/test/scala/boa/service/`；配置 `hardware/configs/am-service-baseline.yaml`（P05 格式，拟新增：bank/adder/FIFO 深度，供 PPA exploration 冻结前的评审值，明确标注非既成事实）。
- 注册并实现 `hardware/scripts/check` target `boa-service`（`--case` 用 P02 统一分派）。
- 不修改：P14/P15 已交付模块；`pipeline_validator/**`。

## 4. 输入输出、接口与不变量

- **O/R credit（每个真实 FIFO 独立）**：结果 FIFO、mailbox、AM group 请求队列、drain 队列各自独立 O+R≤Q，不同层容量不相加。发射：R += 最坏不可暂停输出量（用 P12 合同数字，覆盖所有在途）；到达：R−=到达、O+=到达（到达不释放总 credit）；安全出队：O−=出队，仅当下游已有独立接收保障。微块超保证容量→合法边界再分块或拒绝发射；微块数 O=6/R=2/Q=8 时不得再发射（T10）。
- **commit 判定**：`step_committed = end_of_step_seen AND all_expected_results_received AND pending_am_updates == 0`；`pending_am_updates` 覆盖 result capture、route、queue、RMW、待写回全部在途，与 FIFO 的 O/R 含义不同，不得混同；阵列仍在产出时计数暂为零不得触发 commit。compute_done 可先到，commit/output 不提前（T09）。
- **RMW 与 hazard**：`partial → AmRouter → group read old C（ADD）→ FP32 adder → writeback`；旧 C 不回送阵列。同一物理行 read→adder→writeback 保持互锁或等价 forwarding，控制位变化不绕过；INIT 跳过对应位置旧值读取。bank 吞吐≠adder 吞吐，分别计量（T11/T12）。
- **服务公平**：group 服务资格按微块获取，不为等输入的整条链锁 group；多资源获取全成或全败；drain 读端口按年龄/轮转，持续 ADD 不得饿死 drain（T22 服务面）。ready 仲裁只在 bound_engine 的 ready 候选内公平轮转/按年龄，优先可释放状态（drain）与已有链，不空等未 ready 高优先级候选；每 handle 同时最多一个 issued 未 committed 步骤。
- 观测覆盖credit、bank/adder、mailbox及租约驻留分解，trace关联完整身份。4engine/8slot均衡时每engine≤2仅是策略分布；1/2engine可更多，metadata不设2handle上限。

## 5. 实施步骤

1. CreditTracker 通用组件（参数化 Q）+ 各 FIFO 实例化；发射/到达/出队三端口与不变量断言（SVA/chiseltest）。
2. AmGroup：URAM 2 slot 映射（§7.4.3）、16-lane FP32 adder 流水、地址互锁窗口（含 URAM 读延迟）、INIT 跳读。
3. AmRouter：E 路 512 bit→4 group、同 group 冲突仲裁、输出打拍。
4. ReadyArbiter：bound_engine 内候选公平选择、mailbox 配额对接、FULL→WAIT_ENGINE_QUEUE 语义（游标不动）。
5. 定向与场景测试：T09（延迟 FIFO/RMW/writeback 注入）、T10（Q=8/O=6/R=2 拒发；到达仅 R→O）、T11（多 group 热点 vs 低冲突，吞吐守独立能力）、T12（同地址 RMW/WAR/控制位变化）、T20（单 context 两 engine 上限；两 context 各两链覆盖——单阵列基线只验 1-engine 形态，两 engine 形态用 two-engine.yaml）、T22 服务面（背压+持续 ADD 公平性）。
6. 当前DUT直接验证compute_done/commit；output/free/chain_done接口用有界端口harness检查旧AM访问归零条件，不把尚未实现P17的drain结果当本任务前置。完整五时刻由P17/P18真实集成验证。
7. 注册 `boa-service` target；更新 evidence（不变量断言结果、阻塞原因分解样本 trace）。

## 6. 验收表

| case        | 构造                              | 必须观测                                               | 失败特征 / 层               |
| ----------- | --------------------------------- | ------------------------------------------------------ | --------------------------- |
| PS01 (T09)  | 故意延迟 FIFO/RMW/writeback       | compute_done 先到；commit/output 不提前；五时刻单调    | 边界合并（C/R）             |
| PS02 (T10)  | O=6/R=2/Q=8 继续发射请求          | 拒发；到达仅 R→O；出队后容量才释放                     | 到达即放行（C/R）           |
| PS03 (T11)  | 热点 group+空闲 group             | 热点吞吐≤bank/adder 独立能力；等待方公平可观测         | bank 数当 adder 吞吐（C/R） |
| PS04 (T12)  | 同地址 RMW、写后读、控制位变化    | 互锁/forwarding 生效，无旧值覆盖新值                   | hazard 旁路（C/R）          |
| PS05 (T22a) | drain 与持续 ADD 竞争读端口       | drain 按年龄/轮转取得端口，无永久饥饿                  | ADD 饿死 drain（C/R）       |
| PS06 (T20)  | two-engine.yaml 两 context 各两链 | 最多两 engine 同时执行；单 context 不超两 engine       | 并发越界（C/R）             |
| PS07        | commit 公式断言                   | pending_am_updates 归零前禁止 commit；三个与项缺一不可 | 提前 commit（R）            |
| PS08        | 全成全败多资源获取                | 部分获取即整体回退，无悬挂资格                         | 半持有（R）                 |

反例边界：无限 FIFO、零延迟 AM、额外永久空槽、以加 bank 声称加 adder 吞吐，全部禁止；WAIT_ANY/等待登记端到端归 P15/P22，本层不强加源级控制流。

## 7. 可执行命令（契约；target 由本任务实现）

```bash
hardware/scripts/check boa-service \
  --config hardware/configs/am-service-baseline.yaml --out artifacts/p16 --seed 1
hardware/scripts/check boa-service \
  --config hardware/configs/two-engine.yaml --out artifacts/p16-t20 --seed 1 --case PS06
hardware/scripts/reproduce artifacts/p16/manifest.json --out artifacts/p16-repro --offline
```

预期：manifest 含 PS01–PS08 断言、O/R 高水位与冲突计数、样本 trace（分解原因字段齐全）；周期数为实测新测量，不沿用软件旧数。

## 8. 产物、交接与停止条件

- 产物：ResultCapture/AmRouter/AmGroup/CreditTracker/ReadyArbiter、`am-service-baseline.yaml`、`boa-service` target、evidence。
- 交接：P17 消费 credit 基础设施、drain 端口仲裁与 AM 读接口；P18 集成门核对 PS01/PS07；P25/P26 消费 group/路由参数化边界；PPA 任务消费容量评审值。
- 停止条件：O/R 不变量在任意注入序列下破裂；pending_am_updates 无法覆盖全部在途且以缩角定义「解决」；公平仲裁需要无限缓冲。

## 9. 版本、cutover 与文档同步

- `am-service-baseline.yaml` 的容量/延迟值为本任务评审值，标注「非既成事实」；PPA exploration 冻结物理参数后更新并重跑受影响 case，不改历史记录。
- 软件 `hardware_config.yaml:104-112` 的 step 粒度默认值不因硬件微块粒度改动；两者对应关系（§3.5）在 evidence 中显式说明，防止后序对拍误读。
- 若集成期（P18）发现 pending_am_updates 口径需扩（新增在途类别），先更新本文件口径定义并全量重验 PS07，不在 top 内临时放宽。
