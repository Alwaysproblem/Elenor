# hardware/scripts — 统一硬件入口（P02 / P03 / P06）

`env` 管理镜像构建/进入；`check` / `reproduce` 是所有 target 的统一分派器，
`container_check.py` 是锁定镜像内的 case 执行器，`common.py` 是宿主侧共享库
（仅 stdlib，宿主零安装约束）。P03 的数值 case 位于 `numeric_cases.py`，
P06 的 numeric-profile case 位于 `numeric_profile_cases.py`。

```text
hardware/scripts/check      TARGET [BOARD] --out DIR [--config PATH] [--seed N] [--case ID]
hardware/scripts/reproduce  MANIFEST --out DIR --offline
```

所有真实执行发生在锁定容器内；宿主脚本只做分派、身份校验和 manifest 对比，
绝不安装、绝不写仓库（仓库以 read-only 方式挂载）。

## 退出码（check / reproduce / 容器内一致）

| code | 含义                                                                                         |
| ---- | -------------------------------------------------------------------------------------------- |
| 0    | 全部请求 case 通过（PASS 只能由显式断言产生）                                                |
| 2    | 用法错误：未知 target/case、参数缺失/冲突、lock 缺失或镜像不存在、`--out` 非空               |
| 3    | target 已注册但未实现（输出 `not implemented (owner Pnn)`）                                  |
| 4    | 能力阻塞：无 Vivado 时 `check fpga-synth vcu118`（输出 blocked + owner P23，`--out` 可省略） |
| 5    | docker 本身失败（守护进程、超时被强制移除）                                                  |
| 6    | 配置被 schema/target/section 校验拒绝                                                        |
| 7    | 至少一个 case 断言失败（仍写出 manifest/summary 供诊断）                                     |
| 8    | 容器内环境/身份校验失败（lock/build_inputs/镜像身份/必需路径/工具探测）                      |
| 9    | reproduce：manifest 缺失/非法/非 PASS，或 source/lock/镜像身份漂移                           |
| 10   | reproduce：确定性产物 hash 或 metrics 与首跑不一致                                           |

## Target 注册表与后续任务注册约定

`targets.json`（schema 1）是 target/owner/case 的唯一权威：

```json
"toolchain": {
  "owner": "P02", "implemented": true, "board": null,
  "cases": ["elaborate", "softfloat", "riscv", "python-env"],
  "config": {"required_target": "toolchain", "consumed_sections": ["validator"],
             "default_config": "hardware/configs/toolchain-minimal.yaml"}
}
```

后续任务注册一个 target 的完整步骤（顺序执行，缺一即被视为未实现）：

1. 在 `targets.json` 增加 target 条目：`owner`（Pnn）、`implemented: false`、
   `board`（仅板级 target，如 `vcu118`）、`cases`（真实 case ID，不得预填占位）、
   `config.required_target`（与 target 同名）、`config.consumed_sections`（该 target
   消费的配置 section，见 hardware/configs/README.md）、`config.default_config`。
2. 在 `container_check.py` 的 `CASE_RUNNERS` 增加 `{case: (runner, layer)}`；runner
   必须执行真实程序并对实际产物断言，禁止 log/version/存在性即 PASS。registry 中
   已注册但没有 runner 的 case 会在容器内以 exit 8 拒绝（漂移保护）。
3. 若 target 消费新的配置 section，把该 section 加入 `consumed_sections`，否则配置
   校验会拒绝非空 section（保留 section：topology/numeric/board）。
4. `implemented` 置 true 之前，`check <target>` 一律 exit 3（owner 提示）；
   板级 target 在能力缺失时的阻塞语义见退出码 4（当前仅 `fpga-synth`）。

约束：未知 target / 未知 case 一律 exit 2，绝无隐式 all 兜底；`--case` 只在已注册
case 内过滤，不改变 case 集合语义。

## 配置

统一 YAML 入口见 `hardware/configs/README.md`（schema 1）。配置内容以文本形式
序列化进 `<out>/request.json`，由容器内 `/opt/validator/bin/python` 以
`yaml.safe_load` 解析并校验；宿主不解析 YAML。

## lock 消费约定（runner 视角）

runner 只消费 P02 生成的 `hardware/toolchain/toolchain.lock`，绝不写它：

- 必需 `schema == 1`；`image.id` 必须是精确 Docker image-config ID
  （`sha256:<64hex>`），可变 tag 一律拒绝。宿主用该 ID 直接 `docker run`
  （先 `docker image inspect` 核实本地存在且 ID 一致）。
- `build_inputs`（relative_path → sha256）在容器内对 read-only 挂载逐文件复核，
  任一漂移 exit 8。这是 "reproduce 校验 source 身份" 的机检基础。
- manifest `lock.content` 内嵌完整 lock JSON + 其 sha256；reproduce 先比对当前
  lock sha256 与 manifest 记录，不一致 exit 9。

## 容器调用契约

宿主每次执行（check 与 reproduce 相同）：

```text
docker run --rm --name nexus-<mode>-<pid> \
  --network none --cap-drop ALL --security-opt no-new-privileges \
  --user <hostUID>:<hostGID> -e HOME=/tmp/nexus-home \
  --entrypoint /opt/validator/bin/python -w /workspace \
  -v <abs-repo>:/workspace:ro -v <abs-out>:/out \
  <lock image.id> \
  /workspace/hardware/scripts/container_check.py \
  --request /out/request.json --repo /workspace --out /out \
  --expect-lock-sha256 <sha256(lock)> --expect-image-id <lock image.id>
```

镜像侧假设（由 P02 Dockerfile/ENV 满足；ENTRYPOINT/CMD 被绕过）：

- `/opt/validator/bin/python`（Python 3.11 + xdsl + PyYAML）是解释器与 validator 环境。
- ENV PATH 解析：`verilator`、`riscv64-unknown-elf-gcc/-readelf/-objdump`、`java`
  （兜底 `/opt/native/bin/java`）、verilator `--build` 所需 `g++`/`make`（CC/CXX 包装）。
- firtool 解析顺序：runner 注入 `FIRTOOL_BIN` → `/opt/circt/bin/firtool` → PATH。
- `/opt/bin/sbt`（离线 wrapper，读 `SBT_BOOT_DIRECTORY`/`COURSIER_CACHE`/
  `SBT_IVY_HOME`）与 `/opt/sbt-bundle/{boot,coursier,ivy}`。
- `/opt/deps/chipyard`（`NEXUS_DEPS`）、`NEXUS_GEMMINI` 或
  `/workspace/hardware/gemmini/upstream`、`/opt/softfloat/lib/libsoftfloat.so` +
  `/opt/softfloat/include`、`/opt/toolchain-build.json`。
- 容器以任意 UID:GID 运行（无 /etc/passwd 条目），`HOME` 重定向至 `/tmp/nexus-home`，
  由 runner 创建。无 socket/privileged/device 挂载，无网络，宿主零污染。
- 单次容器硬超时 4 小时（`common.HOST_TIMEOUT_SECONDS`），超时强制 `docker rm -f`。

## `<out>` 目录布局

```text
request.json            宿主写入的完整请求（含 config 全文、seed、case 选择、身份期望）
manifest.json           容器写出的完整运行记录（结构见下）
summary.json            容器写出的汇总（PASS 仅由断言产生）
reproduce_report.json   仅 reproduce：逐项对比结果
logs/docker-*.log       docker 客户端输出
logs/<case>/NN-*.log    每步命令完整输出（argv/cwd/env+exit）
artifacts/<case>/       确定性产物（SV、JSON、ELF、反汇编、report、C bytes 等）
work/                   每次全新构建的临时工作区（PASS 后删除，FAIL 保留诊断）
```

## request.json

`schema, mode(check|reproduce), task_id, target, board, cases[], seed,
config_path, config_content, lock_path, expect{lock_sha256,image_id},
source{host 身份}, created_utc[, reproduce_of]`。容器以 request 为唯一输入，
不存在 mount 歧义。

## manifest.json

`schema, task_id, mode, created_utc, status, request, config(完整解析后配置),
lock{path, sha256, content(完整 lock JSON)}, image.id,
source{host{head,dirty,diff_sha256,untracked_sha256,untracked_files}, fixtures},
probes{tools, binaries(sha256), python{packages}},
environment{uid,gid,network,work,home}, cases{<case>:{layer,status,assertions[],
commands[],artifacts[{path,sha256,bytes,deterministic}],metrics}},
resource_cleanup, counts`。

## summary.json

`schema, task_id, status, validation_layer{case: F|R}, cases[], failures[],
counts{cases,passed,failed,assertions,assertions_failed,commands},
resource_cleanup, manifest`。`PASS` 仅当全部请求 case 的全部断言通过。

## toolchain target 的四个 case（owner P02）

| case         | layer       | 证明内容（核心断言）                                                                                                                                                                                              |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `elaborate`  | R           | 全新 `/out/work`：拷贝锁定依赖 bundle 与 overlay，生成 HardFloat SV；Verilator 编译并运行 harness，逐向量比较完整 a/b/c/out/flags；保存 SV、仿真二进制与 actual/golden JSON 的确定性 hash                         |
| `softfloat`  | F           | 强制运行已编译的 C ABI probe，再 ctypes 加载 PIC `libsoftfloat.so`；真实调用 f32_add/f32_mulAdd，与独立已知位型逐向量比较，包括完整输出 bits 与异常 flags                                                         |
| `riscv`      | F（工具链） | 显式生成固定路径 `smoke.o` 后链接 rv64gc_zicsr_zifencei/lp64d ELF，避免随机临时对象名进入符号表；readelf 断言 ELF64/RISC-V/EXEC、entry==\_start、ISA 与 ABI；objdump 核对链接后的指令和符号。不声称执行了 CPU/RTL |
| `python-env` | F           | 重建的 validator 真实运行 boa-am-matmul（report 全部 checks pass、completed）；再以 P01 流程重放编译产物于全新 ByteStore，输出 C 字节 SHA-256 必须等于 P01 独立整数 oracle golden `ebd15557…`（周期只记录不钉死） |

actual/golden 的 `kind` 和 ABI 诊断 metadata 可不同；比较的是全部向量数据，不将 metadata 的 JSON 字节差异当成数值失败。离线复现分别核对两个完整文件的 hash。

toolchain 的 seed 语义：manifest/request 完整记录并传给 python-env replay 的
SimConfig.seed；validator CLI 本身无 seed 输入，toolchain 各 case 使用固定向量。

## numeric target（owner P03）

独立精确数值 reference + SoftFloat 3e + 实际 HardFloat primitive 的逐 bit/flags
对照；不是完整 BOA/Gemmini RTL，也不修改生产 validator 数值。合同、Python API、
golden 格式见 [numeric_freeze.md](../boa/reference/numeric_freeze.md)。

```bash
hardware/scripts/check numeric --out /tmp/boa-p03/all --seed 20261028
hardware/scripts/check numeric --case probe-nan --out /tmp/boa-p03/nan --seed 20261028
hardware/scripts/reproduce /tmp/boa-p03/all/manifest.json --out /tmp/boa-p03/offline --offline
```

| case         | layer | 实际检查                                                                                                                          |
| ------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------- |
| `selftest`   | F     | 独立 Fraction/整数原语、Matmul/AM/mask/K=0、后处理/INT8、直接 Conv/流量和 binary64 反例                                           |
| `crosscheck` | R     | 从 pinned HardFloat 源生成 SV、Verilator 编译，再将 FMA/add/mul/BF16/i32 的所有 bits/flags 与精确 oracle、真实 SoftFloat 调用对拍 |
| `probe-nan`  | R     | 完整特殊值笛卡尔矩阵、定向边界和 BF16 转换边界；保留原始 NaN payload 输入与实际输出，不做 NaN 归一化后再比较                      |
| `vectors`    | F     | 同 seed 两次生成全部 JSON，精确比较；另用 committed baseline 的 seed/tile_k 重生成并核对仓库 golden                               |

`numeric.yaml` 的 `random_cases` 是 crosscheck 每种 primitive 的随机向量数，默认 1024；
`probe-nan` 不添加随机行。`tile_k` 作用于 selftest/向量默认几何，4/8 定向反例保留
自身几何。seed 真正参与输入生成，不只写 manifest。

每次 check/reproduce 从镜像依赖 bundle 复制全新构建目录。单次 all 运行中
crosscheck 和 probe-nan 共享本次刚构建的 probe，不跨运行复用二进制。SoftFloat
无原生 BF16 API，BF16 使用独立的 SoftFloat 整数 RNE 适配器，并与实际
`RecFNToRecFN` 对拍；该限制明确写入 report。错误时保留 first_failure 和三方原始结果。

## numeric-profile target（owner P06）

生产路径数值 gate：`pipeline_validator.tests.test_boa_numeric_profile` 在锁定
容器内执行真实 `parse -> compile -> load -> Simulator`（full_memory ByteStore），
把观测到的 HBM 输出 bytes 与 committed golden 逐 bit 对拍。golden 权威是
P03 `vectors/matmul.json`（不变）与 P06 定向 fixture
`pipeline_validator/tests/data/numeric_profile_directed_golden.json`（由未改动的
P03 reference 在独立进程生成，verbatim 拷贝进测试数据）；模块本身不 import
`hardware/boa/reference`，也不自行推导期望 bytes。

```bash
hardware/scripts/check numeric-profile --out /tmp/boa-p06/all --seed 20261028
hardware/scripts/check numeric-profile --case vector-parity --out /tmp/boa-p06/parity --seed 20261028
hardware/scripts/check numeric-profile --case zero-k --out /tmp/boa-p06/zero --seed 20261028
hardware/scripts/check numeric-profile --case cutover-reject --out /tmp/boa-p06/cut --seed 20261028
hardware/scripts/reproduce /tmp/boa-p06/all/manifest.json --out /tmp/boa-p06/offline --offline
```

| case             | layer | 实际检查                                                                                                                                                                                                                                                   |
| ---------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vector-parity`  | F     | 30 个合法 P03 Matmul：16 个完整 f32/BF16 输出、14 个 P08 病例的原始 f32 AM（含 INT8 病例，明确不声明 FINAL）；17 个非法向量真实经过 config、parse/compile/load、数值 API 或输入 word 边界拒绝；另含 3 个定向 case 与 8-vs-16 指纹/bytes、binary64 反例检查 |
| `zero-k`         | F     | N-P03-11 零 K/零维组：ADD K=0 逐 bit 保留（-0）、INIT K=0 得 +0、m=0/n=0 空 bytes 正常完成                                                                                                                                                                 |
| `cutover-reject` | F     | 旧 v4 artifact（verbatim 拷贝，从不 reseal）在 parse 阶段按版本拒绝（绝不到 fingerprint）；当前格式 serialize->parse->load->run 复现 golden；同一新 artifact 换 target（numeric tile_k）被 `target fingerprint mismatch` 独立拒绝                          |

配置：`numeric-profile.yaml` 只有空 section。committed 向量固定了每条 case 自身的
`boa_numeric_tile_k`（2/4/8/16），因此 `validator.hw_overrides` 必须为空——容器内
配置校验对非空 overrides 直接 exit 6，绝不静默丢弃或回显接受。每条执行向量的
report 记录其实际执行的 `executed_tile_k` 与 `target_fingerprint`（k8/k16 两行
fingerprint 必须不同）。`--seed` 记录并进 SimConfig；golden 固定，seed 不改变期望
bytes。每 case 产物：`{case}.report.json`、`{case}.vectors.json`（含
actual/golden hex）、`sources/*.mlir`（deterministic）与 `{case}.timing.json`
（`deterministic: false`，wall-clock 永不进入离线对比；cycles 为确定性指标）。

## soft-conv target（owner P08）

软件 im2col Conv 与 FINAL epilogue 的生产路径 gate：
`pipeline_validator.tests.test_soft_conv` 在锁定容器内执行真实
`parse -> compile -> serialize -> parse -> load -> Simulator`（full_memory
ByteStore），把观测 bytes 与 committed golden 逐 bit 对拍。golden 权威是
P03 `vectors/{conv,epilogue,matmul}.json`（不变）与生成 case 的
`examples/workloads/soft_conv_inputs/pad1_stride1/golden.json`（由未改动的
P03 reference 在生成器进程中产出）；模块不 import
`hardware/boa/reference`，也不自行推导期望 bytes。

```bash
hardware/scripts/check soft-conv --out /tmp/boa-p08/all --seed 20261028
hardware/scripts/check soft-conv --case conv-pad-stride-dilation --out /tmp/boa-p08/conv --seed 20261028
hardware/scripts/reproduce /tmp/boa-p08/all/manifest.json --out /tmp/boa-p08/offline --offline
```

| case                       | layer | 实际检查                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `int8-chain`               | F     | i8 输出 AM 链全程对拍：matmul.json i8 链（含 2 步链）、conv-i8-out 经 gather-im2col、以及每个链可达的 epilogue.json i8 向量（A/B 由 golden AM 精确因式分解合成、两个因子都保留）；每条合成链跑两次：无 epilogue 的 f32 FINAL 先逐 byte 对拍 committed golden AM（量化绝不掩盖错误操作数），再跑向量自身的 INIT+ADD(k=0,FINAL) 链对拍 output_hex；golden AM 含链不可达字的向量逐条记 blocked 及证明类别 |
| `param-reject`             | F     | epilogue.json 全部非法参数向量在真实边界拒绝（source parse，或 source 文法无法表达的字段走 v6 artifact 解码），另加 codec tamper（bias 长度、zero_point bool）                                                                                                                                                                                                                                         |
| `stage-matrix`             | F     | epilogue.json 的 bias/scale/relu/clip/inv_scale 全组合 × f32/bf16/i8 输出，在拥有该合同的边界（生产 `boa_numeric.apply_epilogue`，即 runtime FINAL 路径调用的函数）对 committed AM 做 full bytes 对拍；另有 Simulator 全链 wiring 向量                                                                                                                                                                 |
| `conv-pad-stride-dilation` | F     | 8 个 conv fixture（含 asymmetric 与 pad1-stride1）：C bytes 与捕获的 im2col A 操作数 bytes 均等于 golden                                                                                                                                                                                                                                                                                               |
| `conv-traffic`             | F     | 逐阶段流量：transfer/gather 计数器（issued_by_op、byte_counters、gather PMU）与独立解析计数逐 dict 相等（模块 docstring 的 counter<->analytic 映射表）                                                                                                                                                                                                                                                 |

架构事实（记入 report 的 blocked 类别）：生产 AM 只承载 bf16 FMA 链计算出的
值，没有 AM 注入 op。numeric_freeze §2 将算术 NaN 输出统一 canonical 化
（sNaN/payload NaN 不可达）、零符号规则使 −0 不可进入 AM、有限字必须是精确
bf16 乘积。因此 epilogue.json 中 golden AM 含 sNaN/−0/非 bf16 乘积尾数的
i8 向量（本轮 21 个：17 sNaN/payload、3 尾数、1 −0）以显式 blocked 记录
（含证明类别与首坏字），绝不静默跳过；counts 分列 reached/blocked。

配置：`soft-conv.yaml` 只有空 section。committed 向量固定了每条链自身的
`boa_numeric_tile_k`，因此 `validator.hw_overrides` 必须为空——容器内配置
校验对非空 overrides 直接 exit 6。每 case 产物与 numeric-profile 相同
（report/vectors/sources/timing，timing 为 deterministic: false）。

## reproduce 算法

1. 读取 manifest，要求 schema 1 / status PASS / 结构完整，且 task_id 与注册表中已实现 target 的 owner 一致（否则 exit 9）。
2. 宿主重算 source 身份（scope：`hardware/ pipeline_validator/ examples/` 的 HEAD、
   tracked diff、untracked 文件逐文件 hash），与 manifest 不一致 exit 9。
3. 当前 `toolchain.lock` sha256 与 manifest 记录比对、镜像 ID 比对（exit 9）。
4. 从 manifest 的完整解析后 config、seed、cases 重建 request，写入全新 `--out`，
   以同一镜像、`--network none`、全新 `/out/work` 再执行同一 engine；镜像内
   build_inputs 逐文件复核（无暖缓存、无网络、不重用任何旧类/旧二进制）。
5. 对比两份 manifest：确定性 artifact（`deterministic: true`）逐文件 sha256、
   metrics.cycles、工具身份 probes、lock sha256；任何差异 exit 10 并写
   `reproduce_report.json`。对比对象是实际输出字节，绝非日志。

SBT/Coursier 依赖 bundle 是镜像内 pinned 输入（`/opt/sbt-bundle` → 每次
`/out/work/dependencies` 全新拷贝），不是暖缓存；sbt wrapper 默认离线。
