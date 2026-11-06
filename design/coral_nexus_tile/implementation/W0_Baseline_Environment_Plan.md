# W0 基线环境 实施计划（Baseline Environment）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 状态：本轮只写实施计划；未运行其中任何【未来命令】，未创建其中任何【待创建文件】。
> 基线：总体实施计划 [../plan.md](../plan.md)（§2 参数冻结任务表、§3 owner 与共享文件、§4 CN-G0…CN-G5、§7 命令合同）、组件设计 [../components/W0_Baseline_Environment.md](../components/W0_Baseline_Environment.md)、需求 r5 [../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。
> 上游事实均来自已核对固定 SHA 原文（见 §2.2 证据表）；本地命令均为「当前可执行」核实结果。标记规则与组件设计一致：【待创建，本包 X-xx 实现】与【未来命令，待创建】。
> 需求条目以 [../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md) 为基准（PAR/PAR_PREF/PERF/VER/B/CMP/CNT/RES 各族）。

## 1. 已选方案摘要、非目标与 owner

### 1.1 方案摘要（引用 ../plan.md 未实测细节不伪称冻结）

| 维度           | 落定（推荐方案）                                                                                                                                                                                   | 说明                                                                                                                                                                                                                                 |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| workspace 形态 | `hardware/coral_nexus/` 是 Nexus 源文件与配置的权威目录；Bazel 运行根是按 lock 实例化并在 `out/coral_nexus/work/<id>/` 目录按内容寻址物化的工作树                                                  | 上游宏用仓库相对标签（如 `//platforms:coralnpu_v2`、`//toolchain/crt`，coralnpu@f795221 rules/coralnpu_v2.bzl:24-36、:266），若上游仅作依赖模块，这些标签会指到 Nexus 根或报缺失，故运行根必须保留上游 MODULE.bazel；偏差已报 parent |
| 上游来源       | pristine submodule `upstream/coralnpu`（gitlink `f7952215…`）与 `upstream/coralnpu-compiler`（gitlink `d715556c…`）；工作树复制到 `out/` 后再应用编号补丁                                          | 设计 §7 方案 A。compiler 嵌套子模块（iree `2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d`、llvm-project `6cce5bca3b901ec32e250fb43e88eb0fe511af93`、coralnpu `782675ac1af7cefd7612c5bdae5445796b62a7f6`）只登记 SHA，不初始化检出         |
| 上游锁定       | coralnpu `f79522157fc0f1d7311f44e209fc36c9c8b7a039`（组件设计 f795221…）；compiler 嵌套 coralnpu = `782675ac1af7cefd7612c5bdae5445796b62a7f6`，与本包 pin 不同，属已知差异，如实记录，不静默改 pin | 偏斜处理见 §T3 参数来源表；不设「静默对齐」路径                                                                                                                                                                                      |
| 基础镜像       | `debian:trixie` 在 S0/冻结时解析 digest；apt 包取 dated snapshot；Bazel 9.1.0（= 固定 `.bazelversion`，README 8.6.0 是已知口径差异）                                                               | 基础 digest 与 snapshot 时刻在冻结步骤用真实查询得到；镜像构建期可联网，构建后执行/重放默认断网                                                                                                                                      |
| RV32 工具链    | 预构建 GCC 归档（GCS，SHA256 `de06690c…` 在 `rules/repo_defs.bzl:37`）；Rocky 源码重建路径只作为审计文档，不做任何隐式 fallback                                                                    | 归档缺种子或校验失败时直接非零退出；不会切换 Rocky 镜像重跑                                                                                                                                                                          |
| 容器策略       | 单一 Dockerfile，构建期一次性安装 apt snapshot 包、bazel 9.1.0 deb、固定 SHA 的 buildifier/verible/scalafmt、锁定 Python 3.11 与 pip wheel                                                         | 运行期全部走已填充的离线种子（distdir/repository cache/vendor），不再拉网                                                                                                                                                            |
| CLI            | 单一宿主入口 `hardware/coral_nexus/scripts/cn.sh` <command>，仅 host wrapper + docker run + 标准化 exit code/manifest 框架                                                                         | handler 属各 owner（见 §C）；父计划 §7.2 口径                                                                                                                                                                                        |
| Python 3.11    | `env/requirements.lock`（pip，逐包带 SHA256）安装到运行根 `nexus/env/`；构建期一次性生成，之后全部断网复用                                                                                         | 环境 yml 与 requirements 锁的口径一致，由 W0-02 维护                                                                                                                                                                                 |
| devcontainer   | `.devcontainer/coral_nexus/devcontainer.json` 引用同一 image tag，不声明 `build`、`privileged`、docker socket                                                                                      | image tag 与 lock 一致性由 verify 校验                                                                                                                                                                                               |
| 双复现         | (a) sealed image + seed 离线复现；(b) 清缓存重建                                                                                                                                                   | 不比 digest 相等，而是比 artifact 内容与 manifest 输出                                                                                                                                                                               |

### 1.2 本包非目标

不做：RTL/协议语义、消息/状态机；CS2/CS3；性能对比与 PPA/加速比声明；完整 coralnpu compiler 构建（不是 CS1 门槛）；自动 fallback 或静默对齐 pin；自动 retry 或第二套环境定义；P1/PN 性能结论（数据来源由 W7-05 汇合）。

### 1.3 唯一 owner 表（冲突裁决用）

| 资产                                                                                                               | 唯一 owner                               | 说明                                                                                          |
| ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `hardware/coral_nexus/env/*`（sources.lock、tools.lock、image.lock、requirements.lock、seed_targets.json、env.sh） | W0-01/02                                 | 其他包通过 verify 侧读取，不得写入                                                            |
| `hardware/coral_nexus/scripts/`（cn.sh、cn、commands.yaml、lib/\*）                                                | W0-02                                    | 分派文件按 commands.yaml 声明；新增子命令由实施包实现，但框架/exit code/manifest 约定归 W0-02 |
| `hardware/coral_nexus/patches/*` 编号与 series 顺序、ownership table                                               | W0-01                                    | patch 内容归其所属包（W2/W3/W4/W6 等），W0 只管应用顺序和来源                                 |
| `hardware/coral_nexus/configs/{p0_coral.yaml,cs1_c1.yaml,cs1_c2.yaml}` 与参数来源表                                | W0-01/02（`p0_coral`）＋W0-04（cs1\*\*） | W0-03 对 workload 内容提出修改意见，参数值仍归 W0-04                                          |
| `hardware/coral_nexus/kernels/p0/*` P0 原生 kernel                                                                 | W0-03                                    | 其余 `kernels/` 归 W5                                                                         |
| manifest 生成框架（`sources_manifest.json`/`manifest.json` 的 schema/字段命名/消费者标注约定）                     | W0                                       | 各包用同一 schema 产出自己的 manifest，字段标注遵守 PAR-08                                    |
| `upstream/` submodule gitlink、`.gitmodules`                                                                       | W0-01                                    | 改 pin 需 lock 变更，见 §2.4 变更流程                                                         |
| `nexus_p0/`（P0 附加包，加文件、不动上游 .bazelrc/上游目标）                                                       | W0-03                                    | W0-01 物化时创建该路径位；P0 后续阶段复用同一物化根                                           |

## 2. 当前文件证据与待创建清单

### 2.1 已核实的当前事实（host）

| 事实                                                                                                                             | 证据                                                              | 对 W0 的输入                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 本机 Docker CLI/daemon 可用                                                                                                      | `docker version` 28.3.0                                           | 仅证明可用；不据此断言 buildx/其他版本行为                                                           |
| 本机 git、patch、python3 存在                                                                                                    | `which git patch python3`                                         | host bootstrap（`sources verify` 的非镜像路径）可用；镜像同检查覆盖                                  |
| repo 无 `hardware/` 目录，无 `.gitmodules`，无 `.devcontainer/`                                                                  | `ls`/`find` 空结果                                                | 所有 hardware/coral_nexus 与 devcontainer 路径为【待创建】；`git submodule add` 会新建 `.gitmodules` |
| `out/` 在本仓库被忽略                                                                                                            | `.gitignore` 有 `out/` 条目；`.git/info/exclude` 亦有             | 产物目录天然不入 git                                                                                 |
| validator 现命令面（`python -m pipeline_validator`、`--compile-only`/`--compiled-file`/`--hw-config` 等）                        | `pipeline_validator/cli.py`、`README.md`                          | 用于镜像内 Python3.11 与 host 一致性检查                                                             |
| conda env `elenor-validator`（python 3.11、xdsl 0.69.0 等）                                                                      | `pipeline_validator/environment.yml`                              | 生成 `env/requirements.lock` 的输入                                                                  |
| HardwareConfig YAML（grouped，`schema_version: 2`，逐字段 `_HW_*` 映射；`HardwareConfig.from_yaml` 只读 `hardware_config.yaml`） | `pipeline_validator/config.py`、`hardware_config.yaml`            | W0-04 感知 target manifest 中「从 validator 侧可映射字段」的来源                                     |
| 编译产物 ABI v3                                                                                                                  | `pipeline_validator/compiled_program.py`、`execution_verifier.py` | W0 不改它，只在引用时校验                                                                            |

### 2.2 固定上游证据（均按固定 SHA/commit URL 原文读取，未运行）

| #   | 已核对事实与行号                                                                                                                                                                                                                                                                                                                                                 | 用途                                                                                                           |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| E01 | `.bazelversion` = `9.1.0`（上游 `.bazelversion:1`）；README 写 8.6.0（README.md:35）                                                                                                                                                                                                                                                                             | lock 内固定 Bazel 版本来源；双口径差异进 manifest 已知差异                                                     |
| E02 | `MODULE.bazel:1-97`：`module(name="coralnpu_hw")`、rules_scala 7.2.6、rules_python 1.9.0（python_version 3.11.6:89）、Maven lock `//third_party:maven_install.json`（:67）、`register_toolchains("//toolchain/host_clang:host_clang_toolchain_def","//toolchain:cc_coralnpu_v2_toolchain","//toolchain:cc_coralnpu_v2_semihosting_toolchain")`（:78-83）         | W0-01 lock 字段与 W0-04 目标生成入口                                                                           |
| E03 | `rules/repo_defs.bzl:23-40`：`define_coralnpu_toolchain()` `http_archive` → `toolchain_coralnpu_v2`，URL `https://storage.googleapis.com/shodan-public-artifacts/toolchain_coralnpu_v2-2026-06-29.tar.xz`，sha256 `de06690c2da5cd783d76b2998208bd4db4dcdc22dec146c7b0a5ee1af40d3db7`                                                                             | W0-01 tools.lock 的 GCC 工具链归档锁                                                                           |
| E04 | `rules/repos.bzl:40-51`：Verilator v5.052 tar.gz + patch，sha256 `8c8d2e11…`                                                                                                                                                                                                                                                                                     | W1 共享库 C ABI 底座、tools 锁                                                                                 |
| E05 | `tests/verilator_sim/BUILD:199-206` `core_mini_axi_sim`；`tests/verilator_verilator_sim/BUILD` 中无 vme 目标（文件 302 行、0 个 vme 匹配）。`tests/verilator_sim/coralnpu/core_mini_axi_sim.cc:36-42` flags（`--binary --cycles --trace --trace_file --instr_trace --debug_axi --backdoor_load`）；`:110` 缺 `--binary` 报错；`:114` 成功/失败 exit 0/1          | P0 scalar 目标与退出码判据                                                                                     |
| E06 | `examples/BUILD` 内 `coralnpu_v2_hello_world_add_floats`，README 输出路径 `bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/coralnpu_v2_hello_world_add_floats.elf`（README.md:53-63）                                                                                                                                                                        | P0 真实 ELF 目标；cquery 取实际 out 路径                                                                       |
| E07 | `tests/cocotb/BUILD`：`core_mini_axi_model` :70-79、`vme_core_mini_axi_model` :131-143（ZVT_ON/ZVTBF16FMM_ON :134-135）、core_mini suite :290-302、vme_rvv_load_store :706-716、verilator 模型 `verilator_cocotb_model`                                                                                                                                          | P0/VME 目标（已按 parent 同步为 `_verilator` 后缀口径）                                                        |
| E08 | `tests/cocotb/vme_test/BUILD:19-37` `VME_TEST_TESTCASES` 16 条；`:38-58` coralflow_test `vme_test`（非 testcases，由 py 内 `@coralflow_test` 枚举）；vme_test_utils.h 全文 223 行，mset 助记符 `.insn` 封装                                                                                                                                                      | P0 VME 附加 fixture 的调用点                                                                                   |
| E09 | `tests/cocotb/vme_test/vme_test_utils.h:19,23-40`：Zvt mset 助记符用 `.insn r` 写死；`:175-186` `ZvtMatmulFpWord/ZvtVtzeroWord/ZvtWord`                                                                                                                                                                                                                          | W0-03 kernel 助记符复用（不重造编码）                                                                          |
| E10 | `hdl/chisel/src/coralnpu/Core.scala:81-135`：EmitCore CLI 参数表（`--enableAxiInstructionFetch/--xlen/--moduleName/--fetchDataBits/--enableRvv/--enableVme/--enableFloat/--enableZfbfmin/--enableVectorBf16/--fpuPipeRegs/--enableVerification/--exposeDebugPorts/--lsuDataBits/--itcmSizeKBytes/--dtcmSizeKBytes/--useAxi/--useTlul`，未识别参数转发给 Chisel） | W0-04 EmitCore CLI 参数枚举与生成参数表                                                                        |
| E11 | `hdl/chisel/src/coralnpu/Parameters.scala`：`instructionLanes=4:77`、`rvvVlen=128:100`、`vmeTeRatio=8:109`（`vmeTe=rvvVlen/vmeTeRatio` :110）、`vmeSubtileBytes=16:111`、`retirementBuffer/Lanes :141-142`、`fpuPipeRegs:126`、`itcm/dtcmSizeKBytes :176-177`、`EmitParametersHeader :222`                                                                       | 无法 CLI override 的「上游常量」类参数；变化须走 patch 路径                                                    |
| E12 | `hdl/chisel/src/coralnpu/rvv/RvvCore.scala:657-690`：Chisel 端 `GenerateBackendConfig` 生成 `rvv_config.svh`，`enableVectorBf16→\`define ZVFBFWMA_ON\``（:670/672），`enableVme→\`define ZVT_ON\` + \`define ZVTBF16FMM_ON\``（:676/679）                                                                                                                        | W0-04 生成参数来源（Chisel 生成文件也是 SV define 的真实来源之一）                                             |
| E13 | `toolchain/cc_toolchain_config.bzl:164-173`：rv32 march/mabi/mcmodel medany/nostdlib                                                                                                                                                                                                                                                                             | Manifest ISA/ABI 字段（实际命令与产出 ELF，不读 README 字串）                                                  |
| E14 | `rules/linker.bzl:26-33`：默认 ITCM 8k/DTCM 32k；DTcm origin 0x10000（默认）/0x100000（非默认）；默认 stack 128 字节                                                                                                                                                                                                                                             | W0-04 链接器参数来源（itcm/dtcm/stack 默认值）                                                                 |
| E15 | `platforms/BUILD.bazel:14-25`、`rules/coralnpu_v2.bzl:20-36`、`:266`、`:219-252`：`//platforms:coralnpu_v2`、`//platforms:coralnpu_v2_semihosting` 为仓库相对标签；`//toolchain/crt` 依赖；`coralnpu_v2_binary` 产出 `<name>.elf` filegroup（:112、:255）                                                                                                        | 决定 workspace 形态必须保留上游 Bazel 根（§1.1）                                                               |
| E16 | `.bazelrc:36-38`（build/action_env clang-19 等）、`:46/61`（tag filters：build/test tag filters 排除 vcs/synthesis/power/spyglass）、`:39-41`（remotejdk_11）、`:148`（toolchain resolution）、`:.gitignore:1-2`（`bazel-*`、`.bazelrc.user`）                                                                                                                   | W0-01 保留上游 `.bazelrc`；W0-03 选择非 vcs/synth/power 目标                                                   |
| E17 | `utils/coralnpu.dockerfile:10`（`FROM debian:trixie`）、`:44`（libmpfr-dev）、`:86`（`apt-get install bazel bazel-8.6.0 bazel-9.1.0`）、`:103-107`（builder 用户 sudoers 白名单）、`:122`（libmpfr.so.4 symlink）、README.md:35（README 写 8.6.0）                                                                                                               | W0-02 Dockerfile 复刻清单、`libmpfr.so.4` symlink 必要性、sudo 拒绝依据、Bazel/apt 双口径差异                  |
| E18 | `.github/workflows/ci.yml:75/113/169/233`（docker run `--ipc=host`）、`:175/239`（bazel test tag filters）                                                                                                                                                                                                                                                       | W0-02 运行 flag 不复制上游 CI 做法以外的默认，无 privileged/socket                                             |
| E19 | `utils/workspace_status.sh`、`rules/autogen.bzl`（`autogen_scm_info`）、`utils/scm_info.py:55`（默认 40Hex `'f'`）                                                                                                                                                                                                                                               | W0-01 物化根须自带 git 历史或显式 stamp override 方案                                                          |
| E20 | `tests/cocotb/vme_test/vme_matmul_test.cc:17-22,39-41,54,168-176,220-224,267-294`                                                                                                                                                                                                                                                                                | P0 kernel vtfmm.b16（mt0/mt8）参考（真实编码与形状）                                                           |
| E21 | `tests/cocotb/vme_test/vme_test_bench.py:39-41,44-53,422-462`（`_fp_matmul_ref` :374-381；`vme_matmul_bf16_test` :422-462 用 `default_rng(2026)` 精确可复现 inputs，quarter-integer 输入 exact-in-fp32）                                                                                                                                                         | P0 基线数值闭环参考（上游自带 expected 逻辑，可在 fixture 内复用）                                             |
| E22 | `coralnpu_test_utils/coralflow/base_fixture.py:25-115` 与 `verilator test fixture`：`load_elf_and_lookup_symbols/write/write_word/read/read_word/timeout_cycles/run_to_halt/get_cycle_count`（`cycle_count` 由 verilator fixture `wait_for_halted` 返回值给出，base_fixture.py:29、:102-104；verilator_test_fixture.py:124-135）                                 | P0 capture 时钟来源（不需要 RTL 内 rdcycle 即可得 end-to-end cycle；不因加 rdcycle 采样改变结果）              |
| E23 | Csr.scala `mcycle 0xB00 / mcycleh 0xB80`（`:83,403,531,685-689`）                                                                                                                                                                                                                                                                                                | 仅作为「如需 in-program per-phase 计数可用」的备选记录，本任务默认采用 E22 fixture 端计数，不使用 rdcycle 采样 |
| E24 | compiler 子模块 `third_party/coralnpu@d715556c… tree` 内 `coralnpu`、`iree`、`llvm-project` gitlink = `782675ac… / 2b05c5db… / 6cce5bca…`（GitHub contents API at d715556）                                                                                                                                                                                      | W0-01 `sources.lock` 三条 pin；compiler 侧不做 CS1 构建                                                        |
| E25 | 固定 `.gitignore:1`（`bazel-*`）、`.gitignore:2`（`.bazelrc.user`）与上游 `README.md:53-63`（旧无后缀 cocotb 命令）                                                                                                                                                                                                                                              | W0-01 不得直接跑 README 旧命令；以 `core_mini_axi_sim_cocotb_verilator` 等带后缀目标为准（parent 已核实）      |

> 证据边界：上述均未运行（无 bazel、无 docker build、无 verilator 编译）；W0-03 起的真实构建在【未来命令，待创建】入口下进行。

### 2.3 待创建文件清单（owner 标注）

【待创建，本包 W0-01 实现】

```
hardware/coral_nexus/
  upstream/coralnpu                # gitlink f7952215…（pristine，禁止本地修改）
  upstream/coralnpu-compiler       # gitlink d715556…（pristine）
  .gitmodules                      # 由 git submodule add 生成
  patches/series                   # 空起始序列（无 W0 之外 patch 时保持空）
  patches/file_owners.yaml         # 可变文件 owner 表（初始为空，禁止双 owner）
  patches/apply.sh                 # out 工作树补丁应用器（host POSIX shell）
  env/sources.lock                 # upstream/patch/lock/toolchain 归档锁
  env/overlay.yaml                 # Nexus 专用目录 → worktree 拷贝表
  scripts/lib/materialize.sh       # 物化 out/coral_nexus/work/<id> + stamp
  scripts/lib/verify_sources.sh    # sources verify 复核器
```

【待创建，本包 W0-02 实现】

```
hardware/coral_nexus/env/Dockerfile       # 复刻上游 apt 集合 + clang-19 + bazel 9.1.0 + linters
hardware/coral_nexus/env/image.lock       # base digest、OCI labels、seed SHA、verify 规则
hardware/coral_nexus/env/snapshot/…       # apt 包清单 + sources.list 片段（或 lock 引用）
hardware/coral_nexus/env/requirements.lock # pip 锁（Python3.11，逐包 sha256）
hardware/coral_nexus/env/seed_targets.json # 断网种子应包含的目标 label 清单
hardware/coral_nexus/scripts/cn.sh        # 宿主 wrapper（唯一入口）
hardware/coral_nexus/scripts/cn           # 容器内 dispatcher（python3，标准库）
hardware/coral_nexus/scripts/commands.yaml # 命令注册表（owner/status/网络策略/exit code 约定）
hardware/coral_nexus/scripts/lib/docker_run.sh   # run flags 汇编（--user/--network/…）
hardware/coral_nexus/scripts/handlers/{env,sources,baseline}.sh
hardware/coral_nexus/scripts/lib/run_manifest.sh # 每次运行 run.json + 退出码约定
.devcontainer/coral_nexus/devcontainer.json       # 引用同一 image tag，无 build/特权
```

【待创建，本包 W0-03 实现】

```
hardware/coral_nexus/configs/p0_coral.yaml        # kind: p0_upstream（W0-01/02 提供骨架）
hardware/coral_nexus/configs/p0_workloads.yaml    # 工作量/数据形状/seed/expected 来源（W0-03）
hardware/coral_nexus/kernels/p0/…                 # 原核 kernel & fixture
hardware/coral_nexus/nexus_p0/BUILD.bazel         # 新增目标（不修改上游文件）
hardware/coral_nexus/nexus_p0/cocotb/p0_bf16_matmul.py
hardware/coral_nexus/scripts/handlers/baseline.sh # baseline 命令实现
```

【待创建，本包 W0-04 实现】

```
hardware/coral_nexus/configs/cs1_c1.yaml
hardware/coral_nexus/configs/cs1_c2.yaml
configs/params.d/params.base.yaml                 # 各包参数贡献（由各包写入）
hardware/coral_nexus/rtl/targets.bzl              # 生成 target 定义（由本包维护）
hardware/coral_nexus/scripts/handlers/build_config_stage.sh # build --config 的生成阶段
```

【待创建，本包 W0-05 实现】

```
hardware/coral_nexus/MAINTENANCE.md               # 维护文档初版
third_party_notices.md（或 hardware/coral_nexus/env/notices.md；命名以 parent 裁定为准）
```

> 【待创建，本包 W0-05 实现】为 parent 在 plan.md §3.1「【新】…」的文件；本表仅列本包职责内路径。不创建任何实现脚手架或空目录（`hardware/coral_nexus/{env,configs,scripts,upstream,patches}` 与 `.devcontainer/` 目录是这些待创建路径的容器目录，由具体待创建文件落地时一并出现）。

## 3. 任务依赖表（与 CN-G 阶段)

| 任务  | 硬前置                                                                                                                                                                                  | 并行窗口                                                               | 输入 lock/合同                                                                 | 输出消费者                                                                                            |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| W0-01 | 无（仅读取当前仓与固定上游 URL）                                                                                                                                                        | 与 W0-02 的 Dockerfile/工具锁定同时进行                                | 本仓 AGENTS.md、固定上游 URL（E01-E25）                                        | W0-02（tools.lock 中 GCC/Verilator 锁、image 构建）、W1-W7（patches/file_owners/sources.lock）、W0-03 |
| W0-02 | 独立于 W0-01 的 Dockerfile/基础镜像/apt lock/CLI/wrapper 可先行                                                                                                                         | 与 W0-01 并行；镜像构建不需要上游源                                    | sources.lock（GCC/Verilator SHA）、image.lock（W0-05 消费）、requirements.lock | W0-03（镜像 + 离线种子）、W0-04（build_config 命令链）、全包（cn.sh）                                 |
| W0-03 | W0-01（sources verify、materialize）、W0-02（image + seed）                                                                                                                             | 与 W1-W7 的 `-01` 并行                                                 | p0_coral.yaml、p0_workloads.yaml、image.lock                                   | CN-G0 判定、W0-05（运行基线对拍）、W7-05（P0 capture 用于 P1/PN 比较）                                |
| W0-04 | (a) sources.lock/overlay（W0-01）；(b) 联合冻结产物（../plan.md §3.2 `PROTOCOL_FREEZE`）＋`cs1_*` 参数输入冻结记录；**不以其他 -01 DONE 为前置，以「各 -01 产物落盘 + hash 记录」为准** | 与 W0-03 并行（config schema/生成逻辑先行，最终 cs1\_\* 在冻结后落盘） | upstream EmitCore CLI 参数（E10）、固定上游 Sources Lock、各包 params.d/\*     | cn.sh build/config stage（W1-W7 的 `unit --suite wN` 消费）、W0-05（cs1 build 复现）                  |
| W0-05 | (a) 执行复现：W0-03 + W0-02（image/seed）；(b) 构建复现：W0-01 + W0-02 + W0-04 + 现有 validator 环境                                                                                    | 与各包 P1/PN 并行准备，最终判定在 CN-G5                                | image.lock、seed_targets、MAINTENANCE.md、P0 manifest、cs1 manifest            | CN-G5 判定、W7-06（gate --stage G5）                                                                  |

> 联合冻结不是循环依赖：W0-04 只读取冻结产物落盘后的 hash 与参数值（../plan.md §3.2），不要求任何其他包先 DONE；W0-03 与 W0-04 的 cs1\_\* 产出互不依赖（P0 用 p0_coral 固定，cs1 用冻结后参数）。

## C. cn.sh 命令注册表（唯一 CLI 契约；任务写法见 §T0-T4）

> 所有命令【未来命令，待创建，W0-02 建 wrapper；本表列出 owner 与真实退出码；不存在任何「未实现时静默通过」路径】：任何 handler 缺失 → 退出 69，stderr 输出 `handler for <cmd> not implemented; owner <owner-task>`。

| 命令                                  | 实现 owner                                                                      | 容器网络                          | 输入→输出                                                                              | 非零退出码语义                                                                       | 不实现时行为                                     |
| ------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `env build`                           | W0-02                                                                           | 允许构建期联网（仅锁定源）        | env/Dockerfile + lock 文件 → 镜像 + 离线种子 + `sources.lock` 校验通过                 | 20=Dockerfile 失败；21=lock 与实际解不匹配；22=宿主违规（需 sudo/宿主网络变动等）    | 69                                               |
| `env verify`                          | W0-02                                                                           | 断网                              | image.lock + Dockerfile + seed_targets.json → 逐项断言 + `env verify` verdict manifest | 22=镜像/宿主/路径/挂载违规；23=seed 缺失或 hash 不符                                 | 69                                               |
| `env export`                          | W0-02（W0-05 最终调用）                                                         | 断网                              | image + seed → `out/coral_nexus/<run_id>/env-export/` 产物                             | 22=image/seed 不一致；23=seed 不完整                                                 | 69                                               |
| `sources verify`                      | W0-01 脚本，由 W0-02 的 cn.sh 分派                                              | 断网                              | sources.lock → materialize + patch + overlay 一致性 verdict                            | 10=submodule/lock 不匹配；12=patch 不 apply；13=ownership 冲突；14=overlay collision | 69（在 W0-02 dispatch 内注册，脚本本体归 W0-01） |
| `baseline --config --out`             | W0-03                                                                           | 断网（构建期已在 env build 拉网） | p0_coral.yaml → Bazel 构建原核 + 运行 + capture + manifest                             | 见 §T2「独立验证」的 exit-code 语义段（30-59）                                       | 69                                               |
| `build --config --out`                | 生成链 W0-04；共享库 W1-03                                                      | 断网                              | cs1_c\*.yaml → targets_gen.bzl + cs1 build 产物 + target manifest                      | 由 W0-04/各 RTL owner 定（W0-04 固定 stage 划分）                                    | W0-04 落地前：69                                 |
| `build --prepare-only --config --out` | W0-04（build 的旗标分支，不新增子命令；退出码沿用 build 区段 70+）              | 断网                              | cs1_c\*.yaml → 仅生成 targets_gen.bzl/gen_flags + 预检 verdict，不执行构建             | 同 build 区段（70+）；错误码须先登记进 commands.yaml                                 | W0-04 落地前：69                                 |
| `unit --suite wN`                     | 各功能 owner（W7-06 提供统一报告 schema）；W0 仅注册                            | 断网                              | 各包 suites → 统一 run.json                                                            | owner 定                                                                             | 69                                               |
| `package`                             | W5-01/05                                                                        | 断网                              | binding + config → sealed package + manifest                                           | W5 定                                                                                | 69                                               |
| `cosim`                               | W1-03（单 Tile）/ W1-05（final）                                                | 断网                              | scenario + config → 逐 tick 日志 + manifest                                            | W1 定                                                                                | 69                                               |
| `verify --manifest`                   | W7-03/06                                                                        | 断网                              | manifest → checker verdict                                                             | W7 定                                                                                | 69                                               |
| `replay`                              | W7-05                                                                           | 断网                              | bundle → 重放报告                                                                      | W7 定                                                                                | 69                                               |
| `bench`                               | W7-05                                                                           | 断网                              | perf matrix → P0/P1/PN 比较报告                                                        | W7 定                                                                                | 69                                               |
| `gate --stage Gk`                     | W7-06（读证据）；W0-05 提供 G0 的 evidence 定义（本包产出 `cn_g0_evidence` 段） | 断网                              | 各阶段 verdict → gate report                                                           | 20=某 producer verdict 非 PASS；21=evidence 缺失                                     | 69                                               |

> 容器退出码总表：0=PASS；1=校验失败（verdict FAIL）；2=运行中断（外部信号）；64=配置 kind/路径不接受；69=handler 未实现；20-29=env/sources/build 阶段失败（W0 拥有定义权，见各任务）；30-59=baseline 判据（W0-03）；70+=build/config 阶段（W0-04）；由各 owner 填充的区间须以 10 为步长递增，冲突须在 commands.yaml 一致（W0-02 管注册）。**总则（parent 裁定）**：本区间划分经 W0-02 在 commands.yaml 定义后即冻结；其他包不得自造区间，新增错误码必须先登记进 commands.yaml 再使用。

## T0/W0-01 来源锁、pristine submodule 与 out 工作树补丁（命令注册表见 §C）

**优先级** P0；**硬前置**：无；**并行窗口**：与 W0-02 的镜像构建互不等待。

**输入**

1. 固定上游事实（§2.2 E01-E25）。
2. 现有 AGENTS.md / .gitignore（仓库根）；`out/` 已忽略。
3. 无任何已存在 hardware/、.gitmodules——全部为创建。

**输出**

1. `hardware/coral_nexus/env/sources.lock`【待创建】字段：`coralnpu: {repo, commit: f7952215…, tree_sha, submodules:[none], note: pristine}`、`coralnpu-compiler: {repo, commit: d715556c…, nested: {coralnpu: 782675ac…, iree: 2b05c5db…, llvm-project: 6cce5bca…}, note: not-built-in-CS1}`、`toolchain_archive: {url(same as E03), sha256(同 E03), name: toolchain_coralnpu_v2-2026-06-29.tar.xz}`、`verilator: {url(v5.052), sha256(8c8d2e11…)}`、`bazel: {version: 9.1.0}`。
2. `hardware/coral_nexus/upstream/{coralnpu,coralnpu-compiler}` gitlink 与 `.gitmodules`【待创建】——由 host `git submodule add` 实现（**唯一一次动根目录**；submodule 一经创建保持 pristine，后续永远从 out 工作树应用补丁）。
3. `hardware/coral_nexus/patches/{series,file_owners.yaml,apply.sh}`【待创建】。`series` 起始为**空**（P0 无 patch）；后续由各 owner（W2/W3/W4/W6）提交带前缀编号文件名（如 `0002-w2-scalar.patch`）——W0 只管顺序与 apply，不写内容。
4. `hardware/coral_nexus/env/overlay.yaml`、`scripts/lib/{materialize.sh,verify_sources.sh}`【待创建】。

**机械步骤**

1. 用 git（host，仅 submodule add；本步骤**不**运行在容器内，因 W0-02 镜像尚未建——这是 W0-01 先行于 W0-02 镜像构建的唯一窗口，后续全部 verify 由 cn.sh 入口）`git submodule add -b main` 两个上游（`-b main` 与组件设计一致），随后将本地 gitlink 重置/`checkout` 至固定 SHA（若 default branch tip 早于/晚于锁定 SHA，用 `git fetch` 后 `git checkout <sha>`，并在 sources.lock 记录）；**禁止 checkout 非 lock SHA**；submodule 一经添加即 `--` 后不再切换。
2. 生成 `sources.lock`，每项带来源 URL（与 §2.2 同一固定 commit 的文件路径，供复核者打开原文比对），并逐项标明「与组件设计 §3.1/§5.1 一致」。
3. 编写 `patches/file_owners.yaml`：初始仅登记上游路径规则；**禁止**两个包对同一上游路径并发写入；冲突时按共享合同（../plan.md §3.1）交 owner 协调并回填本表；W0 不预判内容，仅维护表结构。
4. 编写 `overlay.yaml` / `materialize.sh`：从 `upstream/coralnpu` 物化出 `out/coral_nexus/work/<id>/`；id = sha256(内容清单)（无 patch 时 P0 也落位同一 id，保证幂等与可重复）；在物化出的树内应用 `patches/series`（顺序 = series 文件顺序；每个 patch 前先 `git apply --check`；失败即终止，**不做** fuzzy/`--3way`，不打印「已跳过」）；再按 overlay.yaml 拷入 Nexus 目录（如 `nexus/env`、`nexus/kernels/p0`、`nexus/rtl/targets.bzl`）；worktree 内执行一次 `git init + git add + git commit` 产生确定 `scm_info` stamp（W0-02 stamp override 的先决条件）。
5. 编写 `verify_sources.sh`：对物化树执行 `git diff --stat`==空、`.bazelversion` 内容与 lock 一致、submodule gitlink == lock、workspace 根 `MODULE.bazel` 内容与 lock 中 `f7952215` 的 tree hash 一致；失败列出差异行后非零退出。
6. 依据 parent 裁定（Main 2026-11-05 IRC）：W0-01 产物中**不创建**任何 `.bazelrc.user`；stamp override 逻辑放在 `materialize.sh` 生成的 `work/nexus_env/workspace_status.sh`（物化树内部，随 id 固定）；不写死 stamp 值（值本身由物化 id 推导）。

**消费者与不变量**

- 所有构建必须在 `out/coral_nexus/work/<id>` 运行根下进行；`hardware/coral_nexus/` 下**永不**出现 `.bazelrc`、`MODULE.bazel`、`MODULE.bazel.lock` 等 Bazel 堆文件（E15/E16：上游宏的平台/工具链标签是仓库相对路径，需要上游为运行根）；这是「pristine submodule + out 工作树补丁」落定的机械实现。
- `kscm` 树内 stamp 由 W0-02 的 `workspace_status.sh` 输出（E19 是上游路径，不改动它；在物化树内覆盖 `--workspace_status_command=…/workspace_status.sh`）。
- `sources.lock` 变更（任一字段改变）须经 parent/组件设计 review 后重走（../plan.md §2.1 变更即锁定）。

**独立验证（本任务验收判据；全部为【未来命令，待创建】）**

- 正例：`bash hardware/coral_nexus/scripts/cn.sh sources verify --config hardware/coral_nexus/env/sources.lock --out out/coral_nexus/sources`（镜像内运行，W0-02 dispatch；镜像未建前 host 侧 `python3 scripts/lib/verify_sources.sh` 亦可跑，仅用于 W0-01 自验）→ exit 0，`sources.lock` 各条与物化树一致，`git submodule status` 全部为 ` <sha> path`（前导空格，非 `-`/`+`）。
- 负例 a（pinned SHA 不匹配）：临时改 `upstream/coralnpu` gitlink（`git checkout HEAD~1` 后回 `verify`）→ exit 10，且 verdict 列出 `expected=f7952215… got=<actual>`。
- 负例 b（patch 无法 apply）：构造 `patches/9999-selftest.patch` 修改 `README.md` 一行使其与真实内容冲突 → exit 12。
- 负例 c（ownership 冲突）：物化后向已登记 owner 的上游文件追加第二个包前缀的 patch → exit 13。
- 负例 d（overlay collision）：`overlay.yaml` 中把 Nexus 输出写入上游已存在路径（如 `hdl/chisel/src/coralnpu/Core.scala`）→ exit 14。
- 正例 b（幂等）：同一 lock 重复运行 → worktree id / 文件内容一致（byte-identical，验证 materialize 可重复）。

**退出条件**：全部正例 1/6 通过且负例 a/b/c/d 各命中各自非零码且 verdict 输出差异行；`hardware/coral_nexus/` 下不出现构建产物或 `bazel-*` 软链。

## T1/W0-02 容器/镜像/离线种子/CLI 入口

**优先级** P0；**硬前置**：Dockerfile/基础镜像可先行；`env build` 的离线种子阶段依赖 W0-01（tools.lock 的 GCC/Verilator 项与 sources.lock 物化 id）；cn.sh 的 dispatch/exit code 框架无前置。

**输入**

1. E17（上游 Dockerfile），E05/E07（目标），E04（Verilator 源归档），E03（GCC 归档）。
2. E18（CI 运行方式），E16（.bazelrc，不修改）。

**输出**（全部【待创建，本包 W0-02 实现】）

1. `env/Dockerfile`：复刻上游 apt 集合（E17），并满足：
   - `FROM debian:trixie@sha256:<digest>`（构建时 parent/用户核实后填入 lock；本计划不写假 SHA）。
   - 保留上游 builder sudo 白名单（E17:103-107），仅用于 apt/mkdir/chown/ln——**不需要更宽**，且 `--cap-drop=ALL` 的运行容器内 sudo 默认无效；保留 `_UID/_GID` 构建参数并按 CI 做法（E18 `--ipc=host`）提供共享 shm 可选项；**不**改 `ENTRYPOINT`。
   - 加装 `bazel=9.1.0`（E17 用 apt 安装；本包镜像同），并设置 `/usr/local/bin/bazel --version` 为 9.1.0 供 lock 校验（README 的 8.6.0 是已知口径差异，不做第二套安装）。
   - 保留 `libmpfr.so.4` symlink workaround（E17:122），其目标文件必须存在，否则 build 失败（不静默跳过）。
2. `env/requirements.lock`（Python 3.11，逐包 sha256，含 host `environment.yml` 的四个包）+ `env/seed_targets.json`（初始 `//examples:coralnpu_v2_hello_world_add_floats`、`//tests/verilator_sim:core_mini_axi_sim`、Chisel 生成目标（E07 的 vme/core_mini cc_library）与 cocotb 目标；W0-03/04 落地时追加）。
3. `env/image.lock`：base digest（待实现时真实解析）、image tag、labels（sources.lock/toolchain sha/bazel 版本/uid:gid）、seed manifest sha。
4. `scripts/cn.sh`（host wrapper）＋ `scripts/cn`（容器内 python3 dispatcher）＋ `scripts/commands.yaml`（§C 表）＋ `scripts/lib/docker_run.sh`：
   - 宿主零安装/零 sudo/零全局 pip：wrapper 仅依赖 docker + git + patch（E17 复刻环境外不需要）。
   - 运行 flags：`docker run --rm --init --runtime=runc --ipc=host`（E18 同源）`--network=none`（除 env build）、`--user "$(id -u):$(id -g)"`、仓库 bind mount `:ro`、`out/coral_nexus` bind mount（rw）、`--cap-drop=ALL`、`--pids-limit`；**禁止** `--privileged`、`--device`、`--cap-add`，禁止挂载 `docker.sock`、宿主 `/var/run`、宿主 HOME。
   - 标准化退出码 + run manifest 框架（`run.json` 含 command、argv、image digest、seeds sha、exit、start/end UTC）。
5. `.devcontainer/coral_nexus/devcontainer.json`【待创建】：引用与 Dockerfile 同一 image **tag**（而非 digest），声明 `remoteUser: "builder"`、"features": {}；禁止 `build`、`privileged`、`mounts` 内含 docker.sock；`env verify`（W0-05）校验该 JSON 只包含 image/features 两个 key。

**机械步骤**

1. 从上游 Dockerfile 复刻（E17），生成 `env/requirements.lock`（host 侧 python3 --resolve-only 或镜像内 pip download --require-hashes 后写入；**不在 host 安装**）。
2. `env build` 流程（镜像内）：build base stage → apt 一次性安装（使用 snapshot 锁）→ bazel 9.1.0 → linters（buildifier/verible/scalafmt，按 E17:88-95 同版本）→ 生成 `out/coral_nexus/seed/`（bazel `--repo_env=CORALNPU_OFFLINE_SEED=1` + `bazel vendor` 到 seed_targets.json 的闭包）→ 写 image.lock（含 manifest sha 校验）。
3. `env verify`（容器内、断网）：a) image ID 与 lock digest 匹配；b) `bazel --version` == 9.1.0；c) `libmpfr.so.4` 目标存在；d) `--user` 生效（非 root）；e) 挂载仅 repo(:ro)+out；f) 无 docker.sock/privileged；g) 断网下 seed_targets.json 的 build 能从 seed 完成（`bazel fetch=false` 口径），失败即 23 并列出缺失 label。
4. `sources verify` 由本包 dispatch（W0-01 脚本本体），本包只保证 `env` 阶段产物（image/lock/seed）就绪。

**独立验证（【未来命令，待创建】）**

- 正例：上表 1-4 全部通过，且重复运行（`env build` 幂等）：image ID 一致、seed 文件 set 一致且 sha 一致、`env verify` 重复 exit 0。
- 负例 a：seed 中删除 GCC 归档文件 → `env verify` exit 23 列出具体文件。
- 负例 b（断网构建缺 label）：`env/seed_targets.json` 临时移除 `//tests/verilator_sim:core_mini_axi_sim` → `env build` 离线重放 exit 23（失败列表含该 label）——**证明**离线重放不会默默联网。
- 负例 c（mount 违规）：构造 `cn.sh env --network=none` 时在 wrapper 中强制加 `--cap-add=SYS_ADMIN` 参数 → `docker_run.sh` 自检 exit 22 拒绝启动。
- 正例 b（非 root）：镜像内 `whoami` != root，且 `sudo apt-get update` 因 sudoers 白名单限定而失败（E17:103-107 白名单不含全量）。

**退出条件**：全部正例 exit 0 且三个负例各自命中预期非零码；`hardware/coral_nexus/` 下无 root-owned 文件；`docker images` 只出现 `coral-nexus-cs1:*` 一个镜像族（W0-05 口径）。

## T2/W0-03 原核 P0 基线（无 Nexus helper）

**优先级** P0；**硬前置**：W0-01（sources verify 过）、W0-02（image + seed）；**并行窗口**：与 W1-W7 `-01` 并行；**本任务无任何 CS1 数值宣称出口**（oracle 待 W7-02，见 ../plan.md §3.2）。

**输入**

- `configs/p0_coral.yaml`【待创建，kind: p0_upstream】（W0-01/02 提供骨架，本任务补 target/manifest 项）。
- 固定上游目标（E05-E08、E20-E22）。
- p0 附加 kernel/fixture（真实 ELF，非后门），与上游 `vme_matmul_test.cc` / `vme_test_utils.h` 同构，不改上游文件。

**输出**

1. `configs/p0_coral.yaml`：kind 为 `p0_upstream`；`baseline` 命令**仅接受**该 kind（cs1_target 类直接非零退出）；manifest 段落明确「P0 不支持的新控制语义由 P1/PN 说明」，不写任何支持性宣称。
2. `configs/p0_workloads.yaml`：小向量精确可计算数据集（如 `hello_world`：input1=input2=1.0 → 期望 2.0；`vme_matmul`（bf16，tm=16, tn=16, tk=2, init=0）：a/b ∈ {-4..4} 步长 0.25 精确 bf16 → `_fp_matmul_ref` 期望值可手工算出）。所有 expected 明确为「上游参考 + 手工可验证小向量」级，不写「oracle」。
3. `kernels/p0/` 与 `nexus_p0/BUILD.bazel`：**新增**文件，不动上游（overlay 路径），提供 `p0_bf16_matmul` cocotb fixture，读 `vme_matmul_test` ELF 的符号，从 fixture 侧注入输入（E22 `write`）、`run_to_halt`、`read` 结果 tile，并 `get_cycle_count`（E22：cycle_count 来自 `wait_for_halted` 返回值）。**不**在 kernel 内插 `rtcycle` 采样（避免改变时序；若后续 P1/PN 需要 per-phase 计数则由 W7-05 任务明确方法与本 capture 兼容说明；不插桩的计数口径对齐：同一输入下「无插入计数」与「有 mcycle-sampling 计数」的差异须在 manifest 中单列一行 `instrumentation_overhead_cycles`）。
4. `scripts/handlers/baseline.sh` + run manifest：每项 capture 记录 inputs/expected/actual/cycle_count/pass-or-FAIL（expected 来自 p0_workloads；无 expected 的项只标注 `not-evaluated`，不标 PASS）。

**机械步骤**

1. `materialize.sh` 物化纯 upstream 根（无 patch）；用 `bazel query` 验证目标存在（`//tests/cocotb:core_mini_axi_sim_cocotb_verilator`、`//tests/cocotb/vme_test:vme_test_verilator`、`//tests/verilator_sim:core_mini_axi_sim`、`//examples:coralnpu_v2_hello_world_add_floats`，与 parent 已同步的 `_verilator` 后缀口径一致）；对不存在的旧无后缀名做负例（query 失败）。
2. 构建/运行真实核：
   - `bazel build //examples:coralnpu_v2_hello_world_add_floats`；`cquery --output=files` 获 ELF 实际路径（E06 说明 bazel-out 含配置 hash，不能写死）。
   - `bazel build //tests/verilator_sim:core_mini_axi_sim`；`core_mini_axi_sim --binary <elf> --cycles 1000000 --instr_trace`（E05 flags）→ exit 0。
   - `bazel test //tests/cocotb:core_mini_axi_sim_cocotb_verilator`（E16 tag filter 默认排除 vcs/synth/power，与 E18 CI 一致）。
   - `bazel test //tests/cocotb/vme_test:vme_test_verilator` → 16 case（E08），逐 case 在 JUnit/test.xml 有 pass。
   - 注：`rvv_core_mini_axi_sim_cocotb`（28 case，size enormous，E07）与 `vme_rvv_load_store_test`（E07 :706）**不在**本任务 G0 要求内（运行成本高）；其结果作为后续 W5-02/03 P0 捕获时的既有上游参考记录，不改变本任务判定。
3. 在物化树内 `git apply --check`（空 series → 即空 diff）确认物化根仍为 pristine；再按 overlay.yaml 拷入 `nexus_p0/`（只增文件，无上游改动）。
4. `bazel test //nexus_p0:p0_bf16_matmul`（cocotb fixture；写 mm_a/mm_b/mm_c_init/mm_impl（E20 符号）→ run_to_halt → read mm_out → 对照 p0_workloads.yaml 期望）；记录 end-to-end cycle_count（E22）。
5. 写 manifest + verdict（不写数值 PASS；对非 expected 项明确 `not-evaluated`）。

**独立验证（【未来命令，待创建】）**

- exit-code 语义：`baseline` 使用 §C 总表的 30-59 段（30=构建失败；31=捕获失败；32=capture expected 不匹配；33=断网下缺输入；其他 30-59 值本任务不使用）。
- `cn.sh baseline --config configs/p0_coral.yaml --out out/coral_nexus/<run_id>` 的 manifest 内：
  - hello_world：actual output == {2.0}（手工可算，来自本计划，不需要 W7 oracle）。
  - p0_bf16_matmul（tm=16, tn=16, tk=2, init=0, a/b ∈ 精确 bf16 小数集）：actual == 手算 expected（如 a=[[1]]/[[2]] 小向量直接手工累加，或沿用 E21 `_fp_matmul_ref`），cycle_count 是正整数。
  - 上游 16 case：test.xml 中 16 个 testcase 全 pass；失败时按 E18 的 tag filter 与上游 README 区分（不静默归因于环境）。
- 双复现预检（W0-05 接口）：同一 run_id 下再次运行（第二次执行用相同种子与输入）→ capture 中的 cycle_count/actual 完全一致（fixture 固定 seed，E22/E21 无时间源干扰）。
- 负例 a：`baseline --config configs/cs1_c1.yaml` → 拒绝（kind 不符）非零退出。
- 负例 b：构造 `p0_workloads.yaml` 中 expected 非法值（如 bf16 输入含非精确值），运行后 actual != expected → capture 标 FAIL（不是 PASS），并记录 mismatch bytes——**证明** PASS 仅来自真实比较，不是「有输出即过」。
- 负例 c：断网下运行（W0-02 默认 network=none）→ 全流程不联网也能完成（Bazel fetch=false 生效）。

**退出条件**：上述正例全部记录 PASS 且负例命中；manifest 中不存在「oracle/数值宣称」字段；`nexus_p0/` 只新增文件、上游文件 diff 为空（W0-01 verify 已覆盖）。

## T3/W0-04 支持 target／配置生成（分两段：前置自由段＋冻结后段）

**优先级** P1（生成逻辑）＋P0 依赖冻结门；**硬前置**：W0-01（sources.lock/overlay）；**冻结前置**：../plan.md §3.2 联合冻结产物（本任务**不要求**其他 `-01` DONE，只要求其产物落盘 + hash 登记）；**并行窗口**：schema/参数来源表可在 W0-01 后自由推进，`cs1_*` 最终文件在冻结后写入。

**输入**

- E10（EmitCore CLI 参数）、E11（Parameters.scala 固定常量）、E12（Chisel 生成 define）、E13/E14（toolchain/linker 默认值）、E15/E16（上游 bazel 目标）；各包 `configs/params.d/*.yaml` 的参数贡献。
- validator 当前事实（§2.1）：`target_fingerprint` 的 `_HW_TARGET_FIELDS`（execution_verifier.py:73-120）与 `_SIM_TARGET_FIELDS:107`、`HardwareConfig.from_yaml`（config.py:388-390）。

**输出**

1. `configs/cs1_c1.yaml`、`configs/cs1_c2.yaml`【待创建，仅冻结门后】：kind: cs1_target；字段含 `context_count`（c1=1 / c2=2）、`vme_instance_count=1`、`tile_mask`（最终 0xf）、每字段 `source`（upstream-scalar / validator-model-baseline / derived / measured）、`consumer`、`consumed: {rtl|validator|both|none}`。**P1 与 PN 的比较口径**（Main 2026-11-05 裁定）：P1=改造后同 RTL、仅 1 Context 活跃；PN=同 RTL、多 Context 活跃；**两者共用 cs1_c2.yaml（同 RTL hash），仅 workload 侧 active-context 数不同**；cs1_c1.yaml 仅作为退化配置的功能回归（C1 退化闭环，../plan.md §7 表），不进入 P1/PN 比较。
2. `rtl/targets.bzl`【待创建】：E10 的宏封装 `nexus_target(...)`，从 cs1\_\*.yaml 输入生成上游 `chisel_cc_library` 调用（gen_flags/defines/vopts 全由 yaml 驱动，不硬编码）。
3. `configs/params.d/params.base.yaml`：W0-04 维护的参数来源表（schema：`{id, kind, source{file,line,value}, consumers[], freeze_status, freeze_owner}`）。
4. `scripts/handlers/build_config_stage.sh`：`build --config` 中先生成物化根内的 `targets_gen.bzl`/`gen_flags`，再执行 Bazel，并写 `target_manifest.json`。

**参数来源表（真实生成参数与冻结过程；不臆造）**

| 参数                                                                                                                     | 来源（file:line，固定 SHA）                                                           | 消费者                               | 冻结途径（owner、方法）                                                                                                 | 现状态                                                              |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `itcmSizeKBytes=8 / dtcmSizeKBytes=32`（core-mini）                                                                      | E10（EmitCore CLI）/ `Parameters.scala:59-60`                                         | chisel gen；linker（E14）            | RTL 前：与 W4-01/05-04 联合（../plan.md §2.2 行 70、71）；W0-04 只登记与 cs1\_\*.yaml 默认值                            | 上游默认值，随 cs1_c1/c2 落库                                       |
| `fetchDataBits=128 / lsuDataBits=128`                                                                                    | `flags.bzl:18-19`（RVV_COMMON_GEN_FLAGS）；Core.scala `--fetchDataBits/--lsuDataBits` | emit 参数表                          | 已上游锁定值（E10 枚举）；cs1\_\* 落库                                                                                  | 已可 CLI override，上游默认 256（Core.scala），flags.bzl 覆盖为 128 |
| `enableRvv/enableFloat/enableZfbfmin/enableVectorBf16/enableVme/useAxi`                                                  | `flags.bzl:17-29`                                                                     | emit 参数表                          | 已上游锁定；cs1\_\* 落库                                                                                                | 已可 CLI override                                                   |
| `enableVerification=True`                                                                                                | `Core.scala --enableVerification`（E10）；verilator model cflags（E07）               | emit 参数表                          | 用于 verification 变体目标；不用于性能基线（PAR_PREF-05 口径）                                                          | cs1_c1/c2 生成 `*_verification` 变体（仅功能/记录用）               |
| VME 几何常量（`rvvVlen=128`、`vmeTeRatio=8`→`vmeTe=16`、`vmeSubtileBytes=16`、`vmeMaxTileWrites=4`、retirement 16/8 等） | `Parameters.scala:100,109-117,141-142`（E11）                                         | emit 参数表（KP\_\*）                | **不可 CLI override**；变化须走 patch（W3 owner，../plan.md §3.1）；W0-04 登记为 `source=upstream_frozen_constant`      | 上游常量；cs1\_\* 落库为 `consumed: rtl`                            |
| `DISPATCH3 / ARBITER_ON / UNMK_USCS_LOAD_NOHANDSHAKE`                                                                    | `RvvCore.scala:657-690`（E12 生成段）                                                 | SV define 路径（生成文件，而非 CLI） | 上游生成逻辑；cs1\_\* 落库为 `source=generated`                                                                         | 由 chisel 生成，非 CLI                                              |
| `rvv_config.svh` 中 `ZVT_ON/ZVTBF16FMM_ON/ZVFBFWMA_ON`                                                                   | E12（`GenerateBackendConfig`）                                                        | SV define                            | 与 E12 同源；登记为 generated（不重复 CLI 传入）                                                                        | generated                                                           |
| `-march/mabi/mcmodel medany/nostdlib`                                                                                    | `cc_toolchain_config.bzl:164-173`（E13）                                              | toolchain                            | 上游固定；cs1\_\* 落库 ISA/ABI 为 `consumed: both`                                                                      | 上游固定                                                            |
| ITCM/DTCM linker origin/stack 默认（8k/32k→origin 0x10000；非默认→0x100000；stack 128 B）                                | `rules/linker.bzl:26-33`（E14）                                                       | linker script                        | cs1\_\* 落库；W5-04 在 slot 表冻结时消费（../plan.md §2.2 行 70）                                                       | 上游默认；随 itcm/dtcm 值改变                                       |
| Verilator 版本 v5.052 & its patch                                                                                        | `rules/repos.bzl:40-51`（E04）                                                        | verilator build                      | 上游锁定（E04 sha）；W0-02 tools.lock 校验                                                                              | 上游锁定                                                            |
| `uce_dispatch_per_cycle`、`stream_token_overhead_cycles`、`stream_fence_cycles`                                          | validator `hardware_config.yaml` 现值（schema2）                                      | validator 指纹（PAR-06）             | 现值是 validator 模型值，RTL 侧不存在对应；manifest 标 `consumed: validator`＋`known_limitation`（PAR-08 不冒充已接线） | 已声明未接线（PAR-08）                                              |
| 程序 slot/stack/scratch、L1/cache/MSHR/Profile 参数                                                                      | `hardware_config.yaml`（§2.1）                                                        | W4/W5 外侧                           | ../plan.md §2.2 各冻结任务（W4-01＆W5-04＆W0-04 共同）；W0-04 不先定值                                                  | 标 `unfrozen`（待外部冻结）                                         |

> 全部参数值**从实际文件落库**：W0-04 的 `params.base.yaml` 生成时逐条 `file:line` 引用；「`consumed` 字段」按 PAR-08 分别标 `rtl|validator|both|none`，缺标注即本表错误。**compiler 偏斜**（W3 嵌套 `coralnpu = 782675ac…` ≠ 硬件 pin `f7952215…`）：两侧 pin 均保持、不加 alias；若某 kernel 需 compiler 路径，按 ../plan.md §2.2 行 64 的实验性路径独立走，不静默改 pin，也不把「完整 compiler 构建」设为 CS1 前置。

**机械步骤（P1-生成链，冻结门后段）**

1. 读取 sources.lock + 各包 `params.d/*`；对每条参数记录做 `file:line` 复核（脚本断言行内内容含期望字串）；任何「来源行不存在/被 patch 改动」→ 落库失败并指出 diff。
2. 生成 `configs/cs1_c1.yaml`（仅 1 Context 退化功能回归；字段全部带 `kind: cs1_target`、`context_count: 1`）与 `configs/cs1_c2.yaml`（`context_count: 2`）。
3. 生成 `rtl/targets_gen.bzl`（由 `nexus_target` 宏展开，输入全部来自 yaml，禁止手写数值）；`scripts/handlers/build_config_stage.sh` 消费 yaml → 实际构建。
4. 对 `cs1_c2` 生成并核对：a) emit 参数头与 flags.bzl 落库值一致（`grep` KP\_\*/`-D`）；b) `//nexus:rtl/targets_gen.bzl` 可被 `build --config` 正确执行；c) manifest 指纹与 `pipeline_validator.execution_verifier.target_fingerprint` 对同字段输入一致（W0 桥接：validator 现有字段与 manifest 的映射表写进本包 `configs/cs1_c1.yaml` 的 `manifest_bridge` 段）。

**独立验证（【未来命令，待创建】）**

- 正例：`cn.sh build --config configs/cs1_c1.yaml --out …/build-c1` → exit 0，产物含 `target_manifest.json` 且指纹字段满足 PAR-01/PAR-08（无「已声明未接线」被写为支持）；`cs1_c2` 同。
- 负例 a（kind 不符）：`baseline --config configs/cs1_c1.yaml` → 非零退出（W0-03 侧已覆盖；本包 build 同样只接受 cs1_target）。
- 负例 b（参数来源丢失）：params.base.yaml 将一条 `source.file:line` 指向在 patch 后不存在的行号 → 生成失败并列 diff。
- 负例 c（consumed 误标）：把 PAR-08 的已声明未接线参数写为 `consumed: rtl` → manifest bridge 校验失败。
- 正例 b（cs1_c1 退化）：以 cs1_c1 生成的 target 在 c1 下 P0 等价工作量的 manifest 中 cycle/资源数据与 cs1_c2 P1 口径可区分（c1=1 Context、c2=2 Context；P1/PN 比较 W7-05 用 cs1_c2 同 RTL hash，与 Main 裁定一致）。

**退出条件**：两个 yaml + targets_gen + bridge 全产出；负例 a/b/c 命中；P1/PN 与 c1 的用途差异在 yaml 字段中显式（`profile.p1/pn.kind` 注明 active-context 口径）。

## T4/W0-05 双复现与许可维护

**优先级** P0 初期准备（schema/脚本）、最终判定在 CN-G5；**硬前置**：执行复现需 W0-02＋W0-03；构建复现需 W0-01＋W0-02＋W0-04；**并行**：schema/负例可先行。

**输入**：image.lock、seed_targets.json、P0 manifest、cs1 manifest、`pipeline_validator` 现命令（§2.1）、E17（buildifier/verible/scalafmt 版本）、E24（三 license 文件存在性）。

**输出**

1. `MAINTENANCE.md`【待创建】：升级流程（../plan.md §2.1 S0→S1→S2 重走）、锁替换规则（新 hash → 新 image tag → gate 重跑）。
2. 第三方 notice 清单【待创建，命名按 parent 裁定】：E24 的三个 LICENSE（coralnpu/coralnpu-compiler/及嵌套）；apt 包级 license（从镜像 `/usr/share/doc/*/copyright` 汇总）；pip 包 license（requirements.lock 附 `--no-deps` 收集）。
3. `cn_g0_evidence` 段（W0-03 baseline manifest 产出的字段；本包在 gate G0 的 evidence reader 引用它（../plan.md §7 表）：**gate 读证据**（逐条 assert），不检查文件存在。

**双复现定义（../plan.md §7.2）**

- **执行复现**：sealed image + 离线种子（W0-02 `env export`）在断网下重放 P0 与 cs1 目标。判据是**内容比对**而非 digest 等值：
  - `sources_manifest.json`、`target_manifest.json` 指纹字段/值序列一致（PAR-06 字段域）。
  - 产物 artifact（ELF、chisel 生成文件）**byte-identical**：Bazel 缓存清空后在非网络容器内重新构建，对比 ELF bytes（E05/E06）。若因 debug/环境 metadata 不同导致差异，须 diff 出「唯一差异字节来源」并写入 MAINTENANCE.md（如 `-fPIC` 或链接器时间戳）；无法解释的差异 → 双复现不通过。
  - 容器内 validator：`python -m pipeline_validator --compile-only -w pow`（§2.1 README 命令面，**仅校验运行不报错+产物 hash 与 host 侧一致**），非 CS1 判据。
- **构建复现**：清基础镜像缓存重建（image build 无 cache），重放 apt snapshot 解析 + GCC/Verilator 归档下载，验证 apt 包清单/锁内 SHA/seed 内容完全一致；产物同上 byte-identical。

**机械步骤（复现命令，全部【未来命令，待创建】）**

1. `cn.sh env export` → `out/coral_nexus/<run_id>/env-export/`（image seed tar + seed manifest）。
2. 新建一个空白 daemon 专用 workspace（不共享 docker 缓存）→ `docker load` 后 `cn.sh env verify` → exit 0。
3. `cn.sh baseline --config configs/p0_coral.yaml --out …/repro-exec`（离线）→ 与 P0 产物 byte-identical。
4. `cn.sh build --config configs/cs1_c2.yaml --out …/repro-build-c2`（离线）→ 同上。
5. 构建端：`cn.sh env build`（重复两次，第二次用独立 docker context / `--no-cache`）→ lock 值一致 + 离线复现流程重跑后产物 byte-identical。
6. `MAINTENANCE.md` 内容：升级流程、双入口（cn.sh + devcontainer）同 image 校验、third-party 增删（锁变更路径）。

**独立验证（全部【未来命令，待创建】）**

- 负例 a（镜像篡改）：修改 image.lock 内任一 SHA → `env verify` exit 22。
- 负例 b（seed 缺文件）：删除 seed 内一个归档 → 离线重放 exit 23（列出缺失 label）；**不得**自动重试或回落到联网。
- 负例 c（devcontainer 违规）：为 devcontainer.json 加 `build`/`privileged`/`mounts.docker.sock` → `env verify` exit 22。
- 负例 d（未接线宣称）：将 PAR-08 未接线参数在 target_manifest 中标 `consumed: rtl` → manifest bridge 校验失败（PAR-08 负例）。
- 正例：上述 1-5 全部 byte-identical 且 manifest 字段域一致。

**退出条件**：四条负例命中 + 五条正例 byte-identical；MAINTENANCE 与 notice 清单覆盖 E17/E24 锁内每一项。

## 5. 能力有限模型手册（低级模型执行序）

1. **先读**：本文件 §2（证据表）→ ../plan.md §2/§3（任务/owner）→ 本文件 §C/T0-T4 对应自己负责的任务。
2. **禁止**：直接运行任何【未来命令】（当前不存在）；改 `hardware/coral_nexus/` 外的任何文件（.gitignore/.gitmodules 除外，且只由 W0-01 改）；读/写 `upstream/` 内任何文件；在 `hardware/coral_nexus/` 下保留空目录（由落盘文件携带）。
3. **参数与锁改动**：遇到 `params.d` 缺字段或 source 行冲突时，**不**静默补值；产出最小失败证据（diff/行号）并报告 owner；「完整 compiler 构建」不是 CS1 前置；compiler 偏斜（W3 的 `coralnpu=782675ac…`）不静默改 pin，仅按 ../plan.md §2.2 行 64 的实验路径另立任务。
4. **正负例顺序**：先跑负例（a/b/c/d）确认「目标合同被拒/报错」，再跑正例；负例失败信息若因缺输入/解析失败**不**算通过。
5. **何时不标 done**：任一负例未命中预期码、任何 `consumed` 未标注、「P0 capture 有 expected 却未实际比较」。
6. **何时标 done**：本文件 §T0-T4 各任务的「退出条件」逐字满足、且 run manifest 已落盘。

## 6. 需求映射与风险处理

| 需求/阶段                                | W0 落地（含需求侧 T-ID：T-33/T-35/T-37/T-53 见 ../plan.md §2.2 的共同出口） |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| G0（环境就绪/可重放）                    | T0 + T1（W0-01/02）；判据见 §T0/§T1                                         |
| P0（原核构建可稳定运行）、基线 manifest  | T2（W0-03）                                                                 |
| P1/PN 入口（同 RTL 双 Context/退化比较） | T3 cs1_c2 + cs1_c1 退化回归；真实性能比较 W7-05                             |
| 参数/产物 manifest                       | §T3 参数来源表（PAR-01/06/08 消费标注）                                     |
| 单一锁定环境/宿主零安装                  | §T1（Dockerfile 复刻上游 + 非宿主侧安装），E17 证据                         |
| ISA/ABI/指令集核对（不读 README 字串）   | T2 的 manifest 段落（E13 落库，E25 排除旧无后缀命令）                       |
| 双复现、维护说明初版                     | T4                                                                          |
| 源码/补丁管理（设计 §7 方案 A）          | T0 pristine + 编号补丁 + ownership 冲突拒绝                                 |
| CN-G0…G5 vs 需求 G0…G6 映射              | ../plan.md §4（parent 维护）；本包输出由 CN-G0/G1 消费                      |

风险处理（对应 ../plan.md §2 与组件设计 §9）：「完整 compiler 不在 CS1 前置」→ T3 参数来源表明确不落 compiler 构建产物；「compiler 偏斜不静默改 pin」→ E24 记录 + §T3 落库路径只走实验任务；「P0 数值宣称前置 oracle」→ T2 不写数值 PASS，oracle 在 W7-02，对比在 W7-05；「参数值必须真实生成参数来源」→ §T3 source{file:line} 强制复核。
