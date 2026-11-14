# C028 → W1/W7：发射接受集与观测时刻

## 接受集

在 W1 的 P1 相位快照有效的 `valid && ready`，P3 只提交这一接受集。
同一拍至多一个 Context，接受 lane 必须是程序序连续前缀；
首个未接受 lane 之后不得接受更年轻 lane。
RR 指针仅在实际接受至少一条指令时推进到该 Context + 1；仅有 offer 不推进。

每条原生指令记录原发起 `ContextKey + owner_op_seq`。操作序号、已接受指令计数和
next-PC 仅按接受前缀推进；未接受后缀的指令身份和 payload 保持稳定，直至接受或定向取消。
branch 重定向使用原发起 owner；响应拍的新 grant 不是 branch、fetch 返回或写回的 owner。

## 与 Python 模型的映射

| 观测点                 | RTL 语义                                          | 模型锚点                                      | 禁止混淆                                     |
| ---------------------- | ------------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| head probe             | 对每个 Context 头部做依赖、资源、CSR、region 探测 | `pipeline_validator/tile.py::_probe_head`     | 不把静态 READY 位当作完整 eligibility        |
| RR 服务                | 选择一个 eligible Context，并以实际接受推进指针   | `pipeline_validator/tile.py::_select_context` | 不要求跨 Context 的原生指令交错逐周期相等    |
| `uce_issue`            | 记录实际接受的原生指令前缀及其 owner              | UCE 的发射边界                                | 不是所有指令都对应 engine launch             |
| `uce_launch_attempt`   | 仅对应真正的 engine/helper launch 尝试            | `_issue_engine_launch`（L1016）               | 不能用任意 RV32 grant 充当 launch 尝试       |
| `uce_launch_accepted`  | engine/frontend 队列实际接受，携带原 owner        | `_enqueue_engine_launch`（L1046）             | 不等价 engine 完成或 Task 退休               |
| `task_retirement_wait` | 已停止新发射，但本 Task 本地完成义务未闭合        | `_complete_context`（L1298–1305）             | park、ROB 暂空、输入早释放均不能代替安全退休 |

模型比较对象是 abstract contract 节点和跨 Context 依赖偏序，不是原生 PC／指令序。
原生 PC、指令和寄存器结果只与独立 ISA reference 或定向程序的明确结果比较。

## PMU 与停顿归因

实现输出与唯一归因优先序登记在 `rtl/scalar/doc/perctx_issue_impl.yaml`。
`WAIT_REGION`、`WAIT_CSR`、`WAIT_FETCH` 必须可区分；同一 Context 同一拍只能记入
一个主停顿原因。engine 接受／完成、退休、过期 fetch 拒收和信用返还分别计数。

C010 的 `rtl/scalar/doc/perctx_issue.yaml` 是已封印的合同输入，本卡不改写。
W7 消费实现投影，不把这些私有计数器补进历史 seal，也不因观测而改变调度状态。
C012 的 region 投影只用于头部阻塞／唤醒和 E1/E2 边界；此交接不宣称 VME 已实现。
