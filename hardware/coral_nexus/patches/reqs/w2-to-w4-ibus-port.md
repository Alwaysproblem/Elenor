# C028 → C030：定向取指端口合同

## 责任边界

W2 管理每 Context 的取指 PC、epoch、请求身份、FRB／指令缓冲和信用。
W4 管理指令存储物理端口、安装／取指仲裁与真实存储延迟。
本请求不引入 VME，不将 C028 前端验证当作 C030 物理端口验收。

## 交接接口

`PerCtxFetchMux` 的 `request`／`response` 是 Decoupled 通道。
`CtxFetchRequest` 携带对齐的取指行地址和 `CtxFetchKey`：
`ctx_slot + ctx_generation + fetch_epoch + fetch_reqid`。
Tile 身份由绑定的 Tile 端口恢复；取指请求尚未分配父指令 `owner_op_seq`，
不得用当前 issue Context 或 PC 推断响应归属。
`CtxFetchResponse` 原样返回完整 key、取指行数据及 fault。

每 Context 在途容量唯一取自已重新冻结的 C019
`constraints.fetch_outstanding_per_context`。`C028TargetParams.scala` 是其生成消费者；
生成／检查入口为 `scripts/lib/gen_frontend_params.py`，没有缺参时的默认值。
C1/C2 的物理容量和指纹必须与生成消费者一致。

## 握手与生命周期

- 只有 accepted Task 且程序 READY 才能发起首取指；hold 不授权取指。
- valid 且未 ready 时 request 的地址和完整 key 保持稳定。重定向不能给已呈现的请求换代；
  已过期的请求仍可完成握手，再按原身份结算。
- 响应可以延迟或乱序。旧 epoch／旧 Task 结果只能被丢弃并结算其原请求，
  不能写新指令缓冲或释放新请求的信用。重复／未知 key 不能凭空增加信用。
- branch 仅推进本 Context 的 fetch epoch，不推进 Task generation；
  cancel 清除本 Context 未接受指令，已接受取指继续排空；park 不清缓冲或 RF／ROB。
- A 的请求等待返回时，B 的请求仍应参与服务。W2 的公平性按 accepted 请求机会计，
  不将该界冒称物理存储响应的周期延迟界。

## 现有 Core 适配与 C030 验证

现有 `IBusIO` 是固定下一拍返回数据的接口。C028 的 Core 接线保存请求接受拍的完整 key，
按这一既有相位恢复响应；保留现有 LSU 优先的物理接口选择，不新增 C030 仲裁策略。
前端独立测试直接驱动 tagged response，覆盖长延迟和乱序，不能由固定拍距 Core smoke 代替。

C030 应使用真实端口对照 C019 容量检查信用守恒、同拍 redirect/response、乱序旧包和 B 的服务界。
失败必须修复实现或产生新 target／新 seal，不能修改历史通过证据。
