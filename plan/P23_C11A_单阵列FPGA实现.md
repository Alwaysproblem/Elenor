# P23 · C11-A：VCU118 单阵列 FPGA 实现与时序 gate

- 状态：未开始。
- 前置：**P22** 通过；其交接应包含 P02 锁定工具链、P18 BOA RTL、P20 联仿、P21/P22 真实 Rocket 系统及固件证据。
- 依据：原方案 §1.4、§7.3、§7.4.1–7.4.7、§8.1c F01、§8.2。
- 公共规则：[执行总则](00_执行总则与共享合同.md)；任务位置：[总索引](README.md)。

## 1. 目标与边界

把已通过真实 UCE 联仿的单 Tile、单 16×16 阵列系统接到 Chipyard 的 pinned VCU118 harness，完成可复现的综合、布局布线与 bitstream。**本任务不烧板，不声称 F02–F08 已通过。** 输出是 P24 可烧写、具有时序与来源证据的真实 bitstream，不是只 elaborate 的 RTL。

保留一个 Rocket hart、默认4个逻辑 contexts、每 Tile 4 AM groups×2 slots。单阵列不把 AM 缩成两个槽；支持8-context构建/运行验收配置，不把 context 数等同 hart 数。逻辑 tile_k 保持8。

不实现多 Tile、Group L2/MFE 全功能、PCIe/XDMA、HBM、异步 BOA 时钟或 partial reconfiguration。首版以 Tile DMA 经 TileLink 搬运 DDR↔L1，BOA 只访问 L1。

## 2. 输入与开工检查

1. 读取 `plan/evidence/P22.md` 及其 firmware/RTL/config manifests，确认真实 Rocket 取指执行而非 Python 控制或直接 RoCC 驱动。
2. 读取 P02 的 toolchain.lock、dependency workspace/build overlay 和只读 Vivado 挂载方式；核对 pinned Chipyard `fpga/src/main/scala/vcu118/Configs.scala` 中 `WithVCU118Tweaks` 的实际构造方式。
3. 检查用户提供的 Vivado 安装目录、器件支持、license 可用性；只读取/调用，不能安装或修改宿主环境。探测输出与版本进入 manifest。
4. 确认 P10 的地址/descriptor/事件容量和 P22 linker/TCM 映射已冻结；本任务不能另定义板级 ABI。
5. 输入矩阵、golden、artifact 均来自 P03/P09/P22 同一合同；板上 firmware loader 不得跳过 fingerprint/capability 检查。

缺 Vivado/license 时先完成可达的 harness/config/约束和仿真检查，记录阻塞；没有真实 synth/place/route 报告不能通过本任务。

## 3. 修改范围与交付路径

| 路径（除前序交付外均拟新增） | 责任 |
| --- | --- |
| `hardware/fpga/vcu118/` | 叠加配置、约束、存储绑定、观测寄存器、构建 Tcl/脚本 |
| `hardware/configs/vcu118_single_array.yaml` | 1 Tile、1 array、16×16、16×16×8、4 groups/8 slots、100 MHz |
| `hardware/configs/vcu118_single_array_8ctx.yaml` | 同一硬件边界的8-context验收构建；若基线容量已8则可共用bitstream并显式证明 |
| `hardware/uce/src/main/scala/` | 在既有系统接入 board harness、Tile DMA、可编程背压观测 |
| `hardware/scripts/check` 的 `fpga-synth` target | 调用真实 Vivado flow，收集并检查报告 |
| `hardware/toolchain/toolchain.lock` | 写入本次验证的 Vivado/器件/IP/约束依赖 hash，不写授权密钥 |

复用前序 AM/array/事件实现；如 FPGA 内存 wrapper 必须改变读延迟，修改其唯一实现并重跑依赖 gate，不复制仿真版和板级版两套协议状态机。遵守同一 Gemmini 源码只编译一次。

## 4. 具体实施步骤

1. **配置叠加。** 从 pinned VCU118 harness 叠加 BoaRoCC 配置，保留 UART、SPI SD、DDR backing memory、JTAG；用 build overlay 引用权威 BOA/UCE 源。生成并保存完整 elaboration 参数，核对没有另一份 Gemmini/Rocket/Chisel。
2. **时钟与复位。** 首版同一个100 MHz域。定义同步 Tile reset，取消挂起、清 metadata、generation换代；不清空整个URAM来掩盖INIT错误。约束覆盖实际时钟与合法异步输入；不得通过 false_path 隐藏 BOA 内部违例。
3. **存储推断/实例绑定。** AM每slot 256 KiB目标按原方案8块URAM并排、每group两slot级联实现；保留4组独立端口和RMW服务。L1按已冻结profile映射BRAM/URAM。结果、mailbox、事件与trace容量独立，不拿剩余URAM补一个无限FIFO。
4. **延迟对齐。** 从综合wrapper参数确定URAM读延迟、输出寄存、router打拍。将延迟反馈到RMW互锁、credit、tag和drain；使用P16/P18同一接口重新验证，禁止仅改时间常量使trace好看。
5. **DDR路径。** 接入Tile DMA，证明命令接受、DDR请求、L1写入、producer完成、BOA读以及C的DMA回写按真正响应推进；不能以提交DMA就发布input_ready/output_ready。
6. **观测。** 实现有限BRAM trace ring、PMU、UART导出接口及L1 output背压注入寄存器。记录ctx/epoch/handle/generation/seq/engine/group/mcycle，ring满记录overflow，不能无声覆盖后称顺序验收通过。
7. **布局。** 将本Tile阵列/router/AM固定于同一SLR，依据实际netlist检查，不把整个板级DDR/harness都错误限制在该SLR。检查URAM可用列、宽总线寄存与布线拥塞。
8. **真实实现。** 从干净输出目录执行elaboration、synthesis、place/route、timing和bitstream。导出利用率、WNS/TNS、时钟、未约束路径、CDC/DRC/critical warnings；逐条处置关键告警。
9. **时序整改。** 若WNS<0，不交付为可验收版本。先修组合长路径/placement；改PE流水要重跑P11/P12/P18/P20/P22相关gate。若只能降频，记录目标偏离及影响，不将降频结果标100 MHz达成；更新manifest和后续验收条件。
10. **复现。** 在第二个干净目录、已锁依赖且关闭网络的条件重跑入口。仿真数值/周期要求确定；Vivado实现不强求所有bitstream字节在非确定工具行为下相同，但工具、约束、配置、结果与差异都必须可解释并分别hash。

## 5. 验收

| Case | 构造 | 必须观察 | 失败条件 |
| --- | --- | --- | --- |
| F01-A | 锁定单阵列配置完整实现 | 真bitstream、post-route WNS≥0、无未处理关键DRC/未约束关键路径 | 仅有RTL、负WNS、忽略关键时序违例 |
| MAP-A | 检查综合网表与hierarchy | AM为Tile-local4组8槽、URAM及RMW端口符合冻结profile | AM变private、推成大量寄存器却未报告、端口数假设错误 |
| RESET-A | FPGA memory wrapper RTL reset注入 | 无旧代写新owner，INIT覆盖垃圾，无需物理清零 | reset后旧响应仍可写新region |
| DMA-A | DDR→L1→BOA→L1→DDR仿真 | bytes及phase顺序正确，真实DMA流量 | 绕过SPM、用host直接拷贝C代替DMA |
| OBS-A | 小ring强制overflow/背压寄存器 | overflow可见，FINAL保持lease直至恢复 | trace缺失不报警、背压只延迟事件不阻止释放 |
| CTX8-A | 8-context elaboration及真实固件回归 | 容量与admission匹配，非固定4context硬编码 | 超出4即越界或拒绝合法配置 |

实际资源数量以报告为准：64 URAM等是目标映射推导，不能把它直接作为实测值填入结果。WNS≥0是必要条件，不替代DRC/功能检查。

## 6. 执行命令合同

以下 `fpga-synth` 由本任务实现，P02提供分派器；在仓库根执行，脚本内部进入锁定容器。Vivado通过P02冻结的环境变量/只读挂载输入，禁止脚本下载安装。

```bash
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_single_array.yaml \
  --out /tmp/boa-p23/single
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_single_array_8ctx.yaml \
  --out /tmp/boa-p23/ctx8
hardware/scripts/reproduce /tmp/boa-p23/single/manifest.json \
  --out /tmp/boa-p23/reproduce --offline
```

如两配置共用一bitstream，第二命令须真实验证配置等价及容量上界，输出明确artifact引用和8-context运行证据，不能空跑。

预期产物：`manifest.json`、`summary.json`、实际生成的bitstream及hash、完整工具日志、post-route timing/utilization/DRC报告、解析配置、RTL/固件/约束hash、SLR占用报告、DDR/L1仿真actual/golden。脚本不满足时序/检查条件必须非零退出。

## 7. 完成与交接

更新FPGA使用说明、锁文件和P23 evidence，列出bitstream精确路径、器件、频率、复位/背压/trace寄存器与firmware兼容版本，交给P24。不得写“板上通过”，不得自动开始烧写。

停止条件：Vivado/license不可用、实际器件不符、不可解释DRC/时序违例、硬件pipeline改动未重新对拍、8-context容量不能表达。保留失败日志/最小配置，明确需要的外部输入或上游修复。
