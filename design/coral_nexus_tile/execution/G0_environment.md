# CN-G0：来源、环境与原核基线（C001–C008）

入口与公共验收规则见[执行总指南](../execution_guide.md)。本页是**未来实施卡片**，不是已执行记录。`H`、`V`、`P0`、`EVIDENCE`、`cn()` 的定义见总指南；每卡开始时把 `EVIDENCE` 设为该卡的新运行目录，不覆盖以前的证据。C001/C002 不依赖尚未存在的 `cn.sh`。

<a id="c001"></a>

## C001 — 来源锁、pristine submodule 与可复现物化

- **任务**：W0-01
- **前置**：无
- **建议 commit**：`build(coral): pin upstream sources and materialize isolated worktrees`
- **输入/先读**：[W0 计划](../implementation/W0_Baseline_Environment_Plan.md) 来源锁任务；总体计划 §3 文件 owner。使用已固定的 Coral `f79522157fc0f1d7311f44e209fc36c9c8b7a039`、compiler `d715556c3853719e414182c0bf5f4a10f248f1d9`，不跟踪浮动分支。
- **修改边界**：新增仓库根 `.gitmodules`、`H/upstream/` 两个 gitlink、`H/env/{sources.lock,overlay.yaml}`、`H/patches/{series,file_owners.yaml}`、`H/scripts/lib/{materialize.sh,verify_sources.sh}`、`H/verification/environment/test_sources.py`。不改上游来源树；实现补丁只在派生树应用。
- **实现步骤**：
  1. 读取固定提交的所有依赖声明，将 URL、完整 SHA、归档摘要、许可证路径与 patch 顺序写入锁；compiler 内嵌 Coral pin 单独记录，不能静默对齐。
  2. 实现 `materialize.sh`：按源内容、overlay、补丁序列摘要在 `out/coral_nexus/work/` 创建隔离构建根，保留上游 Bazel 根；每个补丁严格检查，不 fuzzy apply，不写 pristine submodule。
  3. 实现真正的来源校验：检查 gitlink、文件内容、依赖锁和文件 owner 冲突。派生树标识取内容摘要，不靠本地提交时间或用户 Git 身份产生“可复现 SHA”。负例只使用临时目录的来源副本。
- **验证**：以下入口由**本卡实现**，脚本失败必须非零，不能仅打印锁内容。

```bash
bash "$H/scripts/lib/verify_sources.sh" --lock "$H/env/sources.lock" --out "$EVIDENCE/sources"
conda run -n elenor-validator python -m pytest "$H/verification/environment/test_sources.py" -q
```

- **通过/交接**：同一输入物化两次得到相同源码内容摘要；改 source SHA、补丁上下文、overlay 覆盖已有文件、双 owner 各被对应检查拒绝。来源、许可证与实际派生树能互相追溯。W0-01 可关闭；C002/C003 消费可运行脚本，而非空目录或未来 handler。

<a id="c002"></a>

## C002 — 锁定容器与 bootstrap 构建/校验

- **任务**：W0-02
- **前置**：C001
- **建议 commit**：`build(coral): add pinned rootless toolchain container`
- **输入/先读**：[W0 计划](../implementation/W0_Baseline_Environment_Plan.md) 容器任务；C001 来源锁。先真实解析 Debian trixie digest、snapshot 包版本和工具归档 SHA，不写占位 digest。
- **修改边界**：新增 `H/env/{Dockerfile,tools.lock,image.lock,requirements.lock}`、`H/env/snapshot/`、`H/scripts/handlers/env.sh`、`H/verification/environment/test_environment.py`。镜像构建可在 Dockerfile 内安装锁定依赖；宿主不安装依赖或改 daemon。
- **实现步骤**：
  1. 锁定 Bazel 9.1.0、实际 Verilator/Chisel 依赖与 RV32 GCC 归档；validator Python 3.11 环境与硬件工具环境隔离，不要求完整 IREE 后端先完成。
  2. 实现可独立调用的 `env.sh build/verify`：宿主已有 Docker 构建镜像，镜像内验证工具版本、编译器实际 ISA/ABI、依赖 hash；不要在容器里调用宿主 Docker socket。
  3. 运行用户采用调用者 UID/GID，限制挂载到源码和输出目录，不使用 privileged、宿主 HOME 或运行期 sudo；编译一个使用锁定工具链的真实小程序，验证产物类型与预期输出。
- **验证**：此时仍使用**本卡的 bootstrap handler**，没有 `cn.sh` 前置。

```bash
bash "$H/scripts/handlers/env.sh" build
bash "$H/scripts/handlers/env.sh" verify
conda run -n elenor-validator python -m pytest "$H/verification/environment/test_environment.py" -q
```

- **通过/交接**：实际 Docker build/run 成功，输出由非 root 用户拥有，hash/工具链失配明确失败；镜像 ID/归档身份分别登记，不能把 tag、OCI digest 与本地 image ID 混为同一个字段。C003 复用本 handler；W0-02 尚待离线和 devcontainer 验收。

<a id="c003"></a>

## C003 — 统一 cn.sh 入口与实际已实现命令

- **任务**：W0-02
- **前置**：C002
- **建议 commit**：`feat(coral-cli): dispatch implemented commands in the locked container`
- **输入/先读**：总指南命令注册规则；C001/C002 已验证的 handlers。
- **修改边界**：新增 `H/scripts/{cn.sh,cn,commands.yaml}`、`H/scripts/lib/{docker_run.sh,run_manifest.sh}`、`H/verification/environment/test_dispatch.py`；调用 C001/C002，不复制其执行逻辑。
- **实现步骤**：
  1. `cn.sh` 只负责参数传递、路径规范化和容器启动；容器内 dispatcher 只注册已存在且可执行的 `env build/verify`、`sources verify`、`unit --suite environment`。
  2. 参数按 argv 传递，支持含空格路径；未知命令/缺 handler 非零退出且不列为可用能力。**不提交一排返回成功或固定 69 的未来 handler 占位文件**。
  3. 所有执行写 run metadata：命令、输入/工具/源码内容摘要、退出原因、输出路径；失败记录保持真实状态，不将预期负例和运行成功混淆。
- **验证**：`cn()` 从本卡开始可用；environment suite 必须真正执行来源与容器测试。

```bash
cn sources verify
cn env verify
cn unit --suite environment --out "$EVIDENCE/unit"
```

- **通过/交接**：真实 handler 结果及错误码能穿透 wrapper；含空格参数、错误挂载、未知命令的负例不会执行额外 shell 命令。后续卡原子加入功能实现与 registry，不另建 CLI 或 CMake 后备入口。

<a id="c004"></a>

## C004 — 离线种子、同镜像 devcontainer 与许可索引

- **任务**：W0-02、W0-05
- **前置**：C003
- **建议 commit**：`build(coral): seal offline seeds and share the development image`
- **输入/先读**：[W0 计划](../implementation/W0_Baseline_Environment_Plan.md) 双复现、来源许可与环境规则。P0 核心目标用 `_verilator` 后缀；传递依赖不得仅登记未归档。
- **修改边界**：`H/env/{seed_targets.json,image.lock,notices.md}`、`H/scripts/handlers/env.sh`、新增 `.devcontainer/coral_nexus/devcontainer.json`、`H/MAINTENANCE.md`、environment suite。
- **实现步骤**：
  1. 填充并封存选定上游目标的 Bazel vendor/distdir/repository cache、Python wheels、GCC、后续需要的独立 ISA 参考及 trace processor 来源。各工具有锁定来源与许可，不能依赖忽略目录中偶然已有的二进制。
  2. 增加真实 `env export`；使用独立空的可丢构建缓存、断网启动来重建选定目标。删除一个必要归档后必须失败，不能偷偷联网补齐。
  3. devcontainer 引用同一锁定镜像、非 root 身份和受限挂载，不建第二套工具链。实际启动 devcontainer 后运行相同版本/权限/编译探针；若缺启动工具则记录阻断，不能把静态 JSON 检查当全部环境已完成。
- **验证**：本卡扩展 C003 命令；`env verify` 须检查实际网络、身份、挂载和离线编译行为。

```bash
cn env export
cn env verify
cn unit --suite environment --out "$EVIDENCE/unit"
```

- **通过/交接**：新缓存断网构建成功、缺种子精确失败，Docker 与 devcontainer 使用同一来源锁及编译环境。W0-02 可关闭；W0-05 此时仅有环境/许可基础，最终完整 CS1 双复现留 C073。

<a id="c005"></a>

## C005 — 原核 scalar ELF 的真实端口运行

- **任务**：W0-03
- **前置**：C004
- **建议 commit**：`test(coral-baseline): run upstream scalar ELF with known outputs`
- **输入/先读**：C001 固定源码；[W0 计划](../implementation/W0_Baseline_Environment_Plan.md) P0 任务。P0 不接受 Nexus helper，也不等待最终 `cs1_c1/c2` 配置。
- **修改边界**：新增 `H/configs/{p0_coral.yaml,p0_workloads.yaml}`、`H/kernels/p0/`、`H/nexus_p0/BUILD.bazel`、`H/scripts/handlers/baseline.sh`；扩展 `commands.yaml` 和 baseline suite。
- **实现步骤**：
  1. 定义 `kind: p0_upstream`，从实际 Bazel query/cquery 得到生成目标和 ELF 路径，不使用旧 README 的裸 cocotb label。
  2. 运行未改造原核的 scalar/load-store ELF，输入采用可独立手算的字节样本；通过正式装载/运行接口采集实际输出和结束状态，不能只观察 stdout 非空。
  3. 登记 P0 的 full RetirementBuffer 模式、RTL/tool/config hash 与初始化时钟边界。若现有 harness 未暴露所需 full 目标，增加引用上游生成模块的真实 harness，不修改原核算法。
- **验证**：`baseline` 与 `unit --suite baseline` 由本卡实现/注册。

```bash
cn baseline --config "$P0" --out "$EVIDENCE/p0"
cn unit --suite baseline --config "$P0" --out "$EVIDENCE/unit"
```

- **通过/交接**：固定输入得到精确预期字节，加载错误/越界和 `kind: cs1_target` 配置被拒；P0 来源树仍 pristine。C006 扩展基线，不把 P0 hash 冒充改造后的 CS1 hash。

<a id="c006"></a>

## C006 — 原核 BF16 基线与已知精确输入捕获

- **任务**：W0-03
- **前置**：C005
- **建议 commit**：`test(coral-baseline): capture upstream BF16 execution deterministically`
- **输入/先读**：W0 P0 数值证据边界、[W3 设计 §6](../components/W3_RVV_VME.md)。此卡仅验证已知精确向量，不替代 C047 的完整 bit oracle。
- **修改边界**：`H/kernels/p0/` BF16 kernel、`H/nexus_p0/cocotb/p0_bf16_matmul.py`、P0 workload registry/manifest、baseline suite。
- **实现步骤**：
  1. 以实际 `VmeCoreMiniAxi` 目标运行 BF16 原生矩阵指令；原核 suite 使用 `vme_test_verilator`，不能用不含 VME 的 RVV C++ sim。
  2. 使用零 seed、单位矩阵或小整数等独立可计算输入，捕获 input/actual bits、flags、cycle_count、ELF/hash。保持 fixture 与核执行时钟边界一致，不插任意 rdcycle 指令后混比未插桩程序。
  3. 同配置两次执行记录在不同目录，比对数值与确定性周期；故意改一个 expected bit 必须给出具体失配。ISA-reference 对特殊值的已知差异留待独立 oracle，不改 DUT 以凑数。
- **验证**：复用 C005 的完整 baseline handler。

```bash
cn baseline --config "$P0" --out "$EVIDENCE/p0-a"
cn baseline --config "$P0" --out "$EVIDENCE/p0-b"
cn unit --suite baseline --config "$P0" --out "$EVIDENCE/unit"
```

- **通过/交接**：scalar 与本卡 BF16 精确向量全部通过，异常退出不会成为 PASS；W0-03 可关闭。完整数值支持和 P0/P1/PN 性能资格仍需 C047/C072。

<a id="c007"></a>

## C007 — 证据 schema 与 CN-G0 可执行门

- **任务**：W7-01、W7-06
- **前置**：C006
- **建议 commit**：`feat(coral-verify): validate evidence identity and the baseline gate`
- **输入/先读**：[W7 计划](../implementation/W7_Verification_Trace_Plan.md) 证据/manifest 规则；C001–C006 实际输出。
- **修改边界**：新增 `H/verification/{schemas/evidence.py,gates.py,tests/test_evidence.py}`、`H/verification/gates/G0.yaml`；扩展 C003 dispatcher 的 `verify`、`gate --stage G0`。
- **实现步骤**：
  1. 把 `run outcome`、case 期望结果、证据完整性与 task 完成状态拆开；每份证据包含 producer commit ID、代码内容摘要、输入/config/tool hashes、命令与可取回文件索引。
  2. 实现证据重算和已注册 G0 的条件执行，不仅检查路径存在。source 内容摘要不包含引用自身的 manifest 或最终 commit SHA；提交 SHA 由后置 receipt 关联。
  3. 注入错误产物、陈旧 source hash、缺 expected、崩溃无结束记录；检查器应具体拒绝。尚未实现的 G1–G5 明确不支持，不能空条件 PASS。
- **验证**：命令由**本卡实现**；G0 输入目录和证据选择记录在 `$EVIDENCE/gate`，不可默认取“最新文件”。

```bash
cn verify --manifest "$EVIDENCE/p0-manifest.json"
cn gate --stage G0 --out "$EVIDENCE/gate"
```

- **通过/交接**：先把 C006 产物索引写入本卡 `p0-manifest.json`，其中引用可获取的实际归档；正例验证通过，陈旧/伪造产物拒绝。CN-G0 关闭；W7-06 仅完成 G0 evaluator，绝非整个 CS1 门已完成。

<a id="c008"></a>

## C008 — 目标约束和参数来源 schema

- **任务**：W0-04
- **前置**：C007
- **建议 commit**：`feat(coral-config): validate sourced target constraints before sealing`
- **输入/先读**：总计划 §2.2、当前 `V/hardware_config.yaml`、上游真实生成参数与 C001 来源锁。模型值、上游默认、实测派生值必须分列。
- **修改边界**：新增 `H/configs/{target.schema.json,params.d/}`、`H/scripts/lib/validate_constraints.py`、`H/verification/contracts/test_target_constraints.py`；注册 `unit --suite contracts`。
- **实现步骤**：
  1. 定义 source 参数 provenance、C1/C2、full retirement、地址/队列上界以及包/协议/数值各版本域；不给未测容量填零或假“冻结”值。
  2. contracts suite 支持**无 `--config`**的纯数据校验，执行该提交已注册的全部测试，不依赖未来 RTL 顶层或 final target。
  3. 实际校验 C1 位宽退化、单位、保留区、静态上界与可编码性；runtime 峰值只能作旁证。产物此时为 draft constraints，不生成 `PROTOCOL_FREEZE.json` 或支持声明。
- **验证**：runner 由 C003 提供，本卡增加 contracts suite 与测试。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：同一参数多来源冲突、0-width/越界、容量单位不一致被拒；合法 C1/C2 约束可解析且保持不同版本域。C009–C019 消费 schema；W0-04 的最终配置与实际生成证明在 C019/C020/C040/C065 闭合。
