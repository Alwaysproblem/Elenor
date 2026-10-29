# P19 · C7-B Verilator C ABI 与单步桥

状态：未开始
对应原方案：§5.3（功能模型、周期模型与 RTL 联仿）、§7.2 C7-B、§7.4.2（复位约定）
验证层级：本任务交付 R 层桥本体与 F 层 ABI 合同测试；不含 validator 内存接线（P20）与 Rocket（P21/P22）。

## 1 目标与不做

目标：

1. 建立进程内 C ABI（C++/Verilator shared library + Python 薄封装），让 Python 侧逐 edge 驱动 P18 单阵列 top，请求/完成按 beat 批量交换；禁止逐命令起子进程、逐 byte Python 回调、跑完整矩阵后回填时序（总则 §3.4）。
2. 版本/能力协商、实例创建/reset/销毁、稳定输入布局与 manifest（RTL/source/config hash、精度、array shape、context/group/slot 配置）。
3. 落实NoSecondAM：ABI只暴露硬件AM观测，不提供Python回写AM的旁路；Python backend factory在绑定时拒绝两个AM权威。C库不能自行发现宿主Python对象，不以配置标志代替实际互斥。
4. `cosim-abi` check target 与本任务 case 清单注册（公共 `--case ID` 分派由 P02 提供，本任务只注册）。

不做：validator 内存服务接线、ByteStore 读服务取样与写完成回报（P20）；Rocket/RoCC 前端（P21）；固件（P22）；矩形（P25）；多阵列（P26）；FPGA 综合（P23+）；数值 oracle 与向量生成（P03）；wire schema 编码冻结（P10 已冻结，本任务消费）。

## 2 前置与输入

- P10：`hardware/boa/protocol/boa_protocol_v0.json` 权威 schema 及生成的 C header/codec；descriptor/事件记录字段与位宽以 P10 冻结为准，本任务不得手改字段或另写第三套常量。
- P18：单阵列 top（`hardware/gemmini/` 适配 + allocator/event/AM service/epilogue 集成）及其 Verilator 可 elaborate 模块名、时钟/复位端口约定。
- P02：锁定工具链镜像、`hardware/scripts/check` 分派器与 `--case ID` 基础分派、manifest 基础结构。
- 输入数据：P03 golden vectors 或 P18 既有功能向量（仅作 ABI 层驱动数据，不新增数值断言）。

## 3 当前证据、必读与修改边界

- 仓库当前没有 `hardware/` 实现（总则 §1 只读核对）；本任务全部硬件路径为**拟新增**，不得在 evidence 中写成已存在。
- Python 侧当前无 cosim 模块；`pipeline_validator/` 下仅 cycle 后端（`engines.py::BOAEngine`、`tile.py::ComputeTile.step` 的 `engines.tick → uce.step` 次序）。本任务不改这些文件；Python 封装落点建议 `pipeline_validator/cosim/`（拟新增），动 `pipeline_validator/` 的 backend 选择是 P20 范围。
- 必读：`design/proposal/BOA_Gemmini_Integration_Design.md` §5.3 全表（权威/禁止事项矩阵）、P10/P18 的 evidence 交接、总则 §4.2 check 约定。

## 4 接口与不变量

C ABI 函数集（导出名以实现为准，语义冻结如下）：

- `api_version()` / `create(config*) / destroy(handle)`：config 含 array shape、context 数、group/slot 数、精度 profile、beat 宽度；任一字段与 RTL elaboration 参数不符必须拒绝创建。
- `capability(handle, out*)`：回读实际 shape/context/slot/beat 配置，供 wrapper 与 manifest 校验。
- `reset(handle)` 登记同步复位输入，由后续 `edge` 在调用者指定cycle执行，不能内部推进时钟；`reset_complete` 返回实例epoch，各slot generation按自身原值换代，不混成一个全局generation。P20以实例epoch和owner/slot generation共同过滤旧响应。
- `load_beats(handle, beats*, n)` 只稳定/排队下一edge的输入，不推进时间；`edge(handle, outputs*, cap)` 恰好推进一个有效RTL时钟edge，批量返回memory requests、completion/events和握手状态。批容量不足采用有界保留/显式读取协议，不能丢事件或重跑同一edge。
- NoSecondAM由Python唯一backend绑定实现；ABI状态仅由RTL更新，不接受Python partial/lease镜像写入。

不变量：从相同初始身份状态重放相同beat序列，事件批确定；连续reset会改变epoch/generation，比较须核对规定的身份映射而非要求token原始字节不变。`edge`不吞拍不跳拍；旧访问安全结清；结构体布局显式固定，有限批不丢事件，错误不留半初始化句柄。

## 5 实施步骤

1. 由 P10 schema 生成 C header 与 codec（含事件/请求记录布局）；对生成物跑布局静态断言。
2. Verilator包装：明确clock低/高、组合settle与reset的eval序列。每个ABI edge只含一个有效时钟edge，允许为组合稳定多次eval；不能误写成“一周期只能eval一次”，也不能内部跑到完成。
3. ABI库落在 `hardware/boa/sim/`，按原方案唯一落点扩展；实现版本/create/capability/reset/load_beats/edge/destroy及有界批。地址、mask、req_id、owner/generation、data/status按P10 schema传输。
4. Python 薄封装（`pipeline_validator/cosim/` 拟新增）：ctypes/cffi 绑定、版本握手、beat 批量收发、manifest 写入（复用 P02 manifest 结构）。
5. 实例epoch与各slot generation分别建模；reset期间保持有界旧请求账本，完成后输出reset_complete。过期响应引用结清由P20集成，不以清空数组代替安全回收。
6. `hardware/scripts/check` 注册 `cosim-abi` target 与 case 清单；未支持 case 非零报错。
7. 在 `hardware/boa/sim/` 既有测试布局增加ABI错误/批容量边界回归，实际加载Verilated P18 top运行bytes和事件smoke；host接口检查标F、真实RTL执行标R，不能用fake library替代R。

## 6 验收表

| Case ID | 构造 | 必须观测 | 失败特征 |
| --- | --- | --- | --- |
| abi-version-mismatch | wrapper 以旧 API 版本握手 | create 返回版本错误，无崩溃、无半初始化句柄 | 默默降级成功 |
| abi-shape-refuse | config shape ≠ RTL elaboration 参数 | 创建被拒，错误指明字段 | 创建成功但事件错位 |
| abi-edge-determinism | 两个相同初始身份实例重放；另测连续reset换代 | 前者事件一致；后者按规定epoch/generation映射变化，数值/偏序一致 | 忽略换代或顺序漂移 |
| abi-batch-equivalence | 同edge输入分多次load_beats或一次批量，再各调用一次edge | beat、事件和数值等价，load_beats不走时钟 | 每次load_beats也推进一拍 |
| abi-edge-strictness | 计数 `edge` 调用次数与 RTL cycle 计数 | 1:1，无自动多拍 | 内部跑到完成 |
| abi-reset-generation | 链中途排入reset并由edge执行 | reset_complete、实例epoch、各slot换代与旧访问账本正确 | reset暗跑时钟或旧事件混新代 |
| abi-no-second-am | factory尝试给同Tile注册cycle/RTL两个AM权威 | 绑定被拒，正常RTL只有硬件更新AM | 只检查flag却仍创建两manager |
| abi-capability | 读取 capability 并与 config 比对 | 完全一致 | 静默截断 |
| abi-batch-capacity | 返回批小于当拍request/event量，host延迟取出 | 有界保留、握手/显式读取，不丢事件不重复edge | 覆盖事件或重跑时钟补数据 |

## 7 可执行命令

```bash
# 锁定容器内（P02 env 分派），路径与 target 均为本任务交付，拟新增：
hardware/scripts/check cosim-abi --out /tmp/p19 --seed 7 --case abi-edge-determinism
hardware/scripts/check cosim-abi --out /tmp/p19 --seed 7        # 全部 case
```

预期：`summary.json` 记录 validation_layer=R（ABI 合同项标 F）、case 级结果、manifest（RTL/source/config hash、shape、beat 宽度）。无 P02 环境/无 Verilator 时按总则标阻塞，不得 echo PASS。

## 8 产物、交接与停止条件

产物：ABI header/库、Verilator wrapper、Python 封装、manifest、case 注册、`plan/evidence/P19.md`。
交接 P20：`edge` 语义、generation 令牌、事件批格式、capability 字段、backend 未接入（P20 负责）。
停止条件：任一验收 case 失败且无法在本任务内修复；P10/P18 交接缺失关键接口；阻塞须写明缺失输入与解锁条件，不得标通过。
