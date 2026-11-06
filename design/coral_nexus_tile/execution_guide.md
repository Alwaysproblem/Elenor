# Coral Nexus CS1：按 commit 拆分的执行 guide

> 版本：执行拆分 v1。**76 个计划 commit，覆盖原计划全部 42 个任务；当前均未实施、未创建 Git commit。**  
> 用途：把本文件及一个 commit 卡交给实现模型，按明确依赖完成代码、验证与交接。架构范围仍以用户已批准的[总体设计](Coral_Nexus_Tile_Design.md)与[总计划](plan.md)为准；本 guide 只细化施工顺序、文件边界与 commit 退出条件。

## 1. 从哪里开始

1. 读本文件 §2～§6；按 §7 选择**前置已合入且证据有效**的最小编号 commit。
2. 打开对应阶段页的完整卡片，阅读它引用的包计划/设计段与实际源码。不能只复制 commit 标题就开始写代码。
3. 单次执行一个卡片。既有 source/API 行号会变化，开工时用当前源码核对；未来路径不存在是待创建目标，不是可直接执行的命令。
4. 按 §3 的提交规则完成实现、实际 smoke、回归、证据归档。未授权实际 Git commit 时只交验证通过的变更，状态为 `VERIFIED_NOT_COMMITTED`，不能声称已提交。
5. 按 §8 的 42 任务覆盖表更新完成状态：一张卡通过只关闭其范围，不自动关闭整个 Wn-xx。

| 阶段页                                         | commit 区间 | 阶段出口                                                      |
| ---------------------------------------------- | ----------- | ------------------------------------------------------------- |
| [G0 环境与基线](execution/G0_environment.md)   | C001–C008   | 固定来源、可运行容器、原核精确基线与证据门；没有 CS1 能力宣称 |
| [G1 合同与联合封印](execution/G1_contracts.md) | C009–C020   | 全部合同、codec、容量证明与实验配置共同封印                   |
| [G2 真实标量闭环](execution/G2_scalar.md)      | C021–C041   | 真实 LSU/L1、安装、事件、C ABI、C1/C2 原生 scalar 输出与退休  |
| [G3 RVV/VME 与计算库](execution/G3_compute.md) | C042–C054   | RVV/VME、区域/累加链、独立 bit oracle、全部计算族与尾块验证   |
| [G4 完整系统](execution/G4_system.md)          | C055–C068   | indexed/授权/Profile/取消/debug、四 RTL Tile、完整适用 T 矩阵 |
| [G5 重放与交付](execution/G5_acceptance.md)    | C069–C076   | 可信重放、预登记性能、双复现、完整证据和 CS1 最终出口         |

CN-G0～CN-G5 是本项目执行阶段，不等于根需求 §20 的 G0～G6。C008 的配置 schema 可在 G0 基线完成后准备；正式 CN-G1 到 C020 才关闭。阶段页分组不是隐含的整阶段等待：**硬依赖以每卡“前置”字段为准**；编号升序始终是一条合法的串行执行路线。

## 2. 交给其他模型的执行提示

复制以下模板，只替换本次 commit ID、基线与交付目录；不要把模板占位内容提交进实现源码：

```text
目标：执行 design/coral_nexus_tile/execution_guide.md 中的 Cxxx。
先读：总指南 §2–§6、该卡完整正文、卡片指定的包计划/设计段。
基线：集成分支的具体 commit SHA；前置卡 SHA 与证据 manifest 已提供。
范围：仅该卡列出的文件与同步调用方；共享文件由指定集成 owner 统一修改。
步骤：核对当前 API/配置 → 实现完整行为 → 正负例 → 实际 smoke → 提交门。
约束：不缩小强制范围；不造 fallback/no-op/mock 完成；不修改宿主环境；
      不为通过测试改 oracle/掩盖错误；不把单元端点当真实 CS1。
验证：执行卡片命令及总指南提交门，保存输入/实际输出/退出原因/资源闭包。
交接：commit SHA（若本次获准提交）、前置/代码/配置 hash、证据归档位置、
      已完成卡/仍未关闭原任务/具体阻断；不要只回复“已完成”。
本次权限：按用户授权决定是否执行 git commit；本模板本身不是推送授权。
```

### 2.1 首次执行会话初始化

从仓库根目录运行，变量只影响当前 shell。每张卡设置自己的 ID 与新尝试号；未形成的配置变量只是路径名，不能在 producer commit 之前拿来运行命令。

```bash
set -euo pipefail
H=hardware/coral_nexus
V=pipeline_validator
P0="$H/configs/p0_coral.yaml"
C1="$H/configs/cs1_c1.yaml"
C2="$H/configs/cs1_c2.yaml"
COMMIT_ID=C001
ATTEMPT=attempt-001
EVIDENCE="out/coral_nexus/$COMMIT_ID/$ATTEMPT"
mkdir -p "$EVIDENCE"
cn() { bash hardware/coral_nexus/scripts/cn.sh "$@"; }
```

`cn()` 仅是命令缩写，**C003 前不能调用**；C001/C002 用各自同卡交付的 bootstrap handler。每次重跑使用新 ATTEMPT，不覆盖曾被 receipt 引用的结果。路径内 `H/`、`V/` 在卡片散文中是本节缩写；上游修改在派生构建树中完成，最终提交的是 `H/patches/` 下可审补丁，不提交 build tree。

## 3. 什么才算一个可提交的 commit

- **一个可审核的完整行为单元**，不是固定行数配额。实现＋必要 consumer 迁移＋有意义的回归＋当前文档说明在同一 commit；不留“下一 commit 修编译”的破碎中间态。
- 契约 commit 可以先于消费者实现，但必须带实际 schema/静态校验工具与拒绝边界，并明确仍为 draft/experimental；不能只增加空目录、接口 stub、常量成功或若干 `TODO`。
- 独立 RTL 模块先在真实 module tb 验证，再接顶层合法；此时证据只能标 `unit_rtl`。未接入的功能不注册为可用 CS1，不用假模块填满新顶层后声称成功。
- API/端口改变涉及多个 owner 时，由一个集成 owner 在**同一个原子 commit**修改 producer、全部 consumers、类型/生成物和 tests。其他模型提交接口请求/分支变更，不抢写共享文件。
- 每卡“建议 commit”是可直接采用的主题；补充正文记录行为、不变量、关联任务、验证与证据。**不要把失败测试先提交进主线，再靠下一 commit 修绿。** failing-before/passing-after 记录保存在临时变体或测试证据，最终主线测试应通过。
- 代码变更使后续已完成卡的 source/config/ABI hash 失效时，重新执行受影响闭包；旧 PASS 不继承。不 amend 已发布 seal 来掩盖版本变化，不要求无依赖地单独 revert 任一底层 commit。
- 不强制每个包独立 PR、固定 diff 行数或特定主分支名称；只要求给定前置时可构建/验证、可审核、证据可定位。

### 3.1 Git 合入与并行规则

默认按 C001→C076 串行合入，最容易复现。允许文件不重叠且前置都已存在的卡并行开发：每模型独立 worktree/短分支；不要在同一工作树里多模型执行 Git 操作。

集成 owner 在目标分支应用一个卡的完整变更后跑其提交门，再合入下一卡；基线变动后必须重新验证。任何 branch 名/rebase/merge 操作遵守当前用户授权，不 force push、不 reset 用户改动、不将其他模型未验证的改动顺手提交。

共享路径仍沿[总计划 §3](plan.md)的唯一 owner：W1 管 Python 边界/调度接线，W2 管 Core/SCore/Parameters/CoreAxi 实体，W3 管 RVV/Zvt，W4 管 LSU/内存，W5 管包/驻留/kernel，W6 管事件与 gate_ctl 逻辑，W7 管 report/checker。跨 owner commit 的卡必须指定集成责任，不用两份副本避免冲突。

### 3.2 建议提交正文与证据 receipt

```text
feat(coral-context): route delayed writebacks by context and operation identity

Commit-Guide: C026
Tasks: W2-02
Requires: <直接前置卡及其实际提交SHA>
Invariant: completion matches owner + operation generation, never PC alone
Validation: <实际执行命令、结果与负例判据>
Evidence: <可取回的归档URI或交付相对路径>#sha256=<真实摘要>
```

证据根 `out/` 是本地生成目录，不天然可被别的机器读取。交接必须提供可取回的归档/共享位置与摘要、重跑命令；不能只写 `/tmp/...`、绝对本机路径或一个不存在的链接。

run manifest 记录被测源码/生成物内容摘要，**不引用尚未产生的最终 commit SHA**。代码提交后在归档 receipt 关联 SHA 与这些摘要；receipt 不再写回本 commit 自身，避免自引用 hash。hooks 修改代码后需重跑受影响检查并重新形成证据。

## 4. 验证阶梯与命令可用时间

### 4.1 所有实现 commit 的提交门

每卡的专用验证命令为必要条件，还须通过既有仓库门。单元/构建不能代替卡片要求的真实 CLI、C ABI 或 RTL 程序运行。

```bash
conda run -n elenor-validator pre-commit run -a
conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v
```

以上是**仓库已有命令**，不是本轮执行记录；硬件/容器完成后可在相同锁定环境中执行等价命令。新 schema/RTL/kernel tests 仍须跑所属卡的 suite，不因 pytest 全库绿就认为硬件已验证。纯文档导航变更运行仓库 Prettier 与链接/结构检查；不声称测试了未改的 RTL。

- 格式化 hook 产生改动时检查并只接受本卡范围的改动，随后重跑；不能提交无关格式化噪声。
- 既有故障已由用户报告时不为“确认”再跑一次；完成可达工作并记录准确阻断。新失败不得用 disable/skip/常量 expected 粉饰。
- 永久回归验证消费者行为、边界、错误、状态转换与守恒；不得用 grep 次数、源码文本、文件存在、非空、mock echo 作完成证明。
- 正常场景应成功且资源闭合；预期 fault 场景的 **case 可 PASS，但 DUT outcome 必须为对应失败**。缺陷 mutation 则应使 checker FAIL，二者不同。
- 上限到达只触发取消/排空；无法安全隔离保持 poison/未闭合，不把超时或残余 pending 当成功。

### 4.2 工具 producer 表（防止验证依赖未来工具）

| 工具/入口                                                    | 第一次具备的 commit | 可用范围与限制                                                           |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------------------ |
| 独立 `verify_sources.sh`                                     | C001                | 无容器/无 cn.sh 的来源校验与物化自验                                     |
| 独立 `handlers/env.sh build/verify`                          | C002                | bootstrap Docker 环境；不挂 Docker socket                                |
| `cn env`、`cn sources verify`、`cn unit --suite environment` | C003                | 只注册已有 handler；未知功能必须非零且不显示为 supported                 |
| `cn env export`                                              | C004                | 实际离线种子/镜像归档，非空 tar 不算复现                                 |
| `cn baseline --config "$P0"`                                 | C005                | 真实未改造原核；拒绝 cs1_target 配置                                     |
| `cn verify`、`cn gate --stage G0`                            | C007                | 完整实现已注册阶段；不对未实现 G1–G5 空通过                              |
| `cn unit --suite contracts`                                  | C008                | 纯合同，无 `--config`；每卡原子新增 validator/cases                      |
| numeric-bootstrap、kernel-bootstrap suites                   | C011、C018          | 真实构建/单元探针；没有 CS1 支持声明                                     |
| protocol/CSR 生成与校验                                      | C017                | 草案生成可供测量；正式联合 seal 在 C020                                  |
| `cn build --prepare-only`、按 target 的 `cn unit --suite wN` | C019                | 只构建本卡已注册的真实模块，不要求未来完整 top；不存在的 suite/case 非零 |
| `cn gate --stage G1`                                         | C020                | 全合同一次 seal，缺 producer 不能推测补齐                                |
| `libcn_cosim.so` C ABI driver 的模块验证载体                 | C025                | 实际 C ABI＋专用单元端点；stub 证据永不成为 CS1                          |
| `cn package`                                                 | C037                | 真正 build/seal/validate；首次正式 scalar 包在 C038                      |
| `cn build` 完整 RTL target、`cn cosim --backend rtl`         | C040                | 首个真实 oneTile 标量闭环；不能 C025 以后就把 stub 当正式 Tile           |
| `cn gate --stage G2`、G3                                     | C041、C054          | 各阶段专属实跑证据，不抢先标全部能力 supported                           |
| 四 RTL Tile 的 cosim、G4 门                                  | C065、C068          | mixed 仅调试，G4 拒绝替代证据                                            |
| `cn replay`                                                  | C069                | 完整窗口，实际请求匹配后才注入响应；缺口拒绝                             |
| `cn bench`                                                   | C072                | C071 先提交预登记矩阵，事后不能改门限追认通过                            |
| `cn gate --stage G5`                                         | C074                | C075 再验证拒绝边界，C076 关闭最终交接                                   |

每个新 handler 的 README/`--help` 同卡同步实现范围；不能 C003 一次性公布 76 张卡的未来功能。CLI 非零表示失败/阻断的细分错误码以 `commands.yaml` 为单一入口映射，业务原因保存在 report；不要在各包重新发明相互冲突的数值区间。

### 4.3 场景与 gate 输入不能隐式取“最新产物”

卡片中每条 `cn cosim` 所用场景 YAML 必须由该卡生成/提交，并显式绑定 package manifest、实际 input bindings/bytes、target、oracle、seed、背压/延迟与期望 outcome。`cn package` 只构建与封印；将 image bytes 放入 Python 外侧 program source 的动作发生在 cosim 准备阶段并单独记时，不在 testbench 直写 SRAM。

`cn gate --stage Gk --out "$EVIDENCE/gate"` 执行前由本卡把前置 run manifests 的路径、内容摘要及检验类型列入该目录的 `inputs.json`；gate 读取该显式输入清单并重验，不按 mtime 扫 `out/` 猜最近一次通过。原核、bootstrap、unit、oneTile、fourTile 与 mixed 的 evidence tier 必须可区分。

只有准备好的 `inputs.json` 而无执行产物的目录是合法启动状态；运行器不得因这个准备文件拒绝 gate。已有运行 manifest/封窗结果则不得覆盖，须换新 attempt。失败即停止后续命令；预期负例由 suite 显式检查其失败原因，不通过忽略退出码继续。

单次 run 的 `--out` 目录统一产出 `manifest.json`；多个 run 的汇总文件为 `cs1_manifest.json`。后续卡从前置 receipt 取得具体归档并校验摘要，不凭本机历史目录名猜路径。C1–C7 是判据分类，不表示它们从第一张验证卡起就全部实现：C041 提供 scalar 所需判据，C047/C054 接数值，C064 补齐系统判据；C6 的真实行为重放到 C069 才完成，之前只能检验窗口/包完整性，不能标为“已重放”。

G5 产品门与实施流程关闭分开：C074 必须检查届时已齐备的全部适用 CS1 产品证据；不能因尚缺某项产品证据就缩成“已到达子集通过”。C075 的拒绝边界验证和 C076 的交接审计不作为 C074 的前置提交。最后以独立 receipt 关闭 W7-06 与 42 项流程台账，不让门读取它自身未来 commit 的 SHA。

## 5. 联合冻结与启动顺序的具体解环

| 常见施工环                                     | 本 guide 的执行解法                                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 所有 -01 等彼此 DONE 才产出合同                | C009–C016 分别产出完整的可校验 draft；最终 C020 一次集成 seal。draft 可读不等于正式运行获授权                                                         |
| slot 容量等真实 ELF，真实 ELF 又等封印 target  | C017 生成 draft helper，C018 编译**不运行正式 CS1**的 bootstrap ELF做静态资源测量，C019 生成实验 target，C020 联合 seal；C038/C052 实际库逐个再次核验 |
| oracle 要先 supported，首次实跑包又要 oracle   | experimental profile 可封印测试；C021/C047 独立 oracle，实际结果通过后才提升对应能力支持状态                                                          |
| 标量闭环先于真实 LSU/安装/事件                 | C031 LSU、C032 MFE、C033/C034 事件/终态、C039 安装先于 C040/C041                                                                                      |
| W1-05 等各包 -05，而各 -05 又等四 Tile harness | oneTile 实际入口 C040；各包用它自验，C065 才汇合四 Tile，C067 最终矩阵                                                                                |
| DRAIN 等所有 PENDING 事件先终态                | 只等真实 accepted/传输义务；随后 MARK_EVENTS 授权 RTL 写本地 RESET，Group 仅写 Group 表                                                               |
| RETIRE_READY 后等父 R/pin 先归零               | RETIRE 授权本地回收→可靠 TASK_COMPLETION→剩余父 pin/R/Route 结算；早释放另走方向信号路径                                                              |
| E1 申请等目标 accepted，accepted 又等 grant    | E1 在 Context 头部 valid、pre-queue 登记/授予；E2 只在 grant 后 accepted 时消费声明                                                                   |

后续计算库若超过当前 target 资源上界，允许回到构建/配置阶段产生新 target/profile/package 身份并复验所有消费者，**不允许原地改旧 seal 或用自动扩容 fallback**。容量实测任务不是下一模型可以跳过的 TBD。

## 6. 实现时必须保持的合同

1. 唯一执行 owner：RTL Tile 内不存在第二套 Python L1/engine 完成路径；Python-only 模型作为独立后端保留。`send` 不偷偷 step，P4 不递归产生同 tick response。
2. 八函数 C ABI 与五相位保持总计划命名；steady-state bridge 预分配，不重复复制整个容量。program epoch 与 run generation 分域，warm 本地取指合法，Tile 外文本字节为零。
3. 四 lane 同 Context 连续前缀、RR 跳过阻塞头；完整 Context/op/generation 匹配迟到写回。CSR flags 从所属 ROB 条目退休合并，不能要求 resultUpdate 和 retire 同拍。
4. E1/E2、区域链和 mode table 按最新 W3 勘误；参考 seed/drain 16417 不是16386。`CLEARED` 恢复合法，尾块模板不能被三行全局清单误拒；无 grant 不占共享队列。
5. VME 内部 ROD/有限宽/jam、乘溢出 Inf/累加溢出 max-normal、输出 RNE；KD-01 是参考已知差异，不降容差。mask 无效 lane 不进入 special/flags，不能以乘零代排除。
6. Group `all_tasks` 后可提前解纯读 pin；物理回收还需 owner/全部 claim/view/pin/in-flight 闭合。早释放不还 R/L1/Frame/UCE/program 引用，Task 退休不双重 unpin。
7. scalar/RVV 仅访问已授权本地窗口；外部地址只由合法 MFE descriptor/binding 生成并做全宽检查。页授权 acquire/revoke 与真实引用结算线性化，不能假 grant 或静默 double release。
8. 数值、故障、背压、取消和 stale/unknown 行为均在变更涉及的真实路径闭合。缺必需能力不得靠 unsupported、skip 或无声模型 fallback 改小验收范围。
9. 固定 kernel 库覆盖原计划支持族；scan/recurrence 要有明确方程与状态初值/输出，不把 KV 字节拷贝当状态计算。近似算法/系数/舍入进封印 profile，独立参考不从 DUT输出反拟合。
10. P1/PN 同 `cs1_c2`、同 RTL hash、同总工作量，仅活跃 Context 数不同；P0 是不同原核 hash但 full/mini 模式需对齐。makespan 取最早开始到最晚承诺完成，不将各任务时长相加。

## 7. 依赖、领用与状态

**每卡“前置”是 commit-ID DAG 的直接边**。编号小不是已合入证明；领用者必须核对前置的实际 Git SHA、验证 receipt 和 frozen input hash。默认顺序执行可以直接按阶段页向下走；如并行选择，所有前置合入且文件所有权无冲突才能领取。

只维护一份实现状态记录（由集成 owner 保存），不要让八个包各自宣称同一个共享任务已完成。建议记录字段：`commit_id, state, base_sha, dependency_shas, implementer, source_digest, evidence_uri, evidence_sha256, committed_sha, blocked_reason`；本轮不生成虚假的状态文件或 PASS 标记。

状态：`NOT_STARTED → IN_PROGRESS → VERIFIED_NOT_COMMITTED → COMMITTED`。检查失败返回当前卡修复；工具/外部依赖确实不可达则 `BLOCKED`，记录缺失产物与可复现证据，其他无依赖卡可推进。未知环境故障不自动改协议/精度/目标。

- 未合入卡可继续修正成一个完整 commit，不提交破损中间态。
- 已合入卡发现缺陷：在独立修复 commit 中复现→修正→验证，关联原 `Commit-Guide` ID和所有受影响依赖；不要为隐藏失败重写已发布历史。
- 若一张卡必须再拆，只允许拆成同样可编译可验证的子单元：先更新本 guide 的 ID/依赖/覆盖记录，确保 downstream 不消费半成品。不能机械按行数拆接口 producer/consumer。
- 回退底层卡须同时回退/迁移其消费者闭包；本计划不承诺有依赖的任意 commit 都可孤立 revert。

## 8. 原 42 个任务到 commit 的关闭映射

“最早可关闭”是消费者证据必须到达的点，不是强制所有实现动作等到此点才可并行。尤其 `-01` 的 draft producer 不互等 DONE；共同 seal 消费的是文件/hash。

| 原任务 | 实现 commit                                                                                                                                                                                                                                                                                                                 | 最早可关闭                               | 额外退出条件                                                                        |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| W0-01  | [C001](execution/G0_environment.md#c001)                                                                                                                                                                                                                                                                                    | [C001](execution/G0_environment.md#c001) | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W0-02  | [C002](execution/G0_environment.md#c002), [C003](execution/G0_environment.md#c003), [C004](execution/G0_environment.md#c004)                                                                                                                                                                                                | [C004](execution/G0_environment.md#c004) | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W0-03  | [C005](execution/G0_environment.md#c005), [C006](execution/G0_environment.md#c006)                                                                                                                                                                                                                                          | [C006](execution/G0_environment.md#c006) | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W0-04  | [C008](execution/G0_environment.md#c008), [C019](execution/G1_contracts.md#c019), [C020](execution/G1_contracts.md#c020)                                                                                                                                                                                                    | [C065](execution/G4_system.md#c065)      | 静态生成通过后，还需实际 C1/C2 顶层消费相同参数；不能 prepare-only 就标 supported。 |
| W0-05  | [C004](execution/G0_environment.md#c004), [C073](execution/G5_acceptance.md#c073)                                                                                                                                                                                                                                           | [C073](execution/G5_acceptance.md#c073)  | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W1-01  | [C009](execution/G1_contracts.md#c009), [C017](execution/G1_contracts.md#c017), [C020](execution/G1_contracts.md#c020)                                                                                                                                                                                                      | [C020](execution/G1_contracts.md#c020)   | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W1-02  | [C022](execution/G2_scalar.md#c022), [C024](execution/G2_scalar.md#c024)                                                                                                                                                                                                                                                    | [C040](execution/G2_scalar.md#c040)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W1-03  | [C025](execution/G2_scalar.md#c025), [C040](execution/G2_scalar.md#c040)                                                                                                                                                                                                                                                    | [C040](execution/G2_scalar.md#c040)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W1-04  | [C035](execution/G2_scalar.md#c035), [C036](execution/G2_scalar.md#c036)                                                                                                                                                                                                                                                    | [C066](execution/G4_system.md#c066)      | 早释放、gate 双唤醒、剩余 pin 与 R 退休须在真实系统场景成立。                       |
| W1-05  | [C065](execution/G4_system.md#c065)                                                                                                                                                                                                                                                                                         | [C068](execution/G4_system.md#c068)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W2-01  | [C010](execution/G1_contracts.md#c010)                                                                                                                                                                                                                                                                                      | [C020](execution/G1_contracts.md#c020)   | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W2-02  | [C026](execution/G2_scalar.md#c026)                                                                                                                                                                                                                                                                                         | [C041](execution/G2_scalar.md#c041)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W2-03  | [C028](execution/G2_scalar.md#c028)                                                                                                                                                                                                                                                                                         | [C041](execution/G2_scalar.md#c041)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W2-04  | [C027](execution/G2_scalar.md#c027), [C040](execution/G2_scalar.md#c040)                                                                                                                                                                                                                                                    | [C054](execution/G3_compute.md#c054)     | 包含 RVV/VME 返回的 flags 与配置顺序，不仅 scalar 独测。                            |
| W2-05  | [C041](execution/G2_scalar.md#c041)                                                                                                                                                                                                                                                                                         | [C041](execution/G2_scalar.md#c041)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W3-01  | [C011](execution/G1_contracts.md#c011), [C012](execution/G1_contracts.md#c012)                                                                                                                                                                                                                                              | [C020](execution/G1_contracts.md#c020)   | 合同可在联合 seal 前发布；关闭点等待统一 hash，不等 DUT supported。                 |
| W3-02  | [C042](execution/G3_compute.md#c042), [C043](execution/G3_compute.md#c043)                                                                                                                                                                                                                                                  | [C053](execution/G3_compute.md#c053)     | RVV owner、配置、访存和 flags 在真实 kernel 中端到端验证，不仅模块 tb 通过。        |
| W3-03  | [C044](execution/G3_compute.md#c044), [C045](execution/G3_compute.md#c045)                                                                                                                                                                                                                                                  | [C053](execution/G3_compute.md#c053)     | E1 未 grant 不入共享队列；E2 一次消费与取消闭合真机可观察。                         |
| W3-04  | [C046](execution/G3_compute.md#c046)                                                                                                                                                                                                                                                                                        | [C053](execution/G3_compute.md#c053)     | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W3-05  | [C053](execution/G3_compute.md#c053)                                                                                                                                                                                                                                                                                        | [C054](execution/G3_compute.md#c054)     | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W4-01  | [C013](execution/G1_contracts.md#c013)                                                                                                                                                                                                                                                                                      | [C020](execution/G1_contracts.md#c020)   | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W4-02  | [C023](execution/G2_scalar.md#c023), [C029](execution/G2_scalar.md#c029), [C030](execution/G2_scalar.md#c030)                                                                                                                                                                                                               | [C059](execution/G4_system.md#c059)      | 完整 cache/MSHR/维护及 Python 外侧切分均有实际消费者。                              |
| W4-03  | [C031](execution/G2_scalar.md#c031)                                                                                                                                                                                                                                                                                         | [C041](execution/G2_scalar.md#c041)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W4-04  | [C032](execution/G2_scalar.md#c032), [C055](execution/G4_system.md#c055), [C056](execution/G4_system.md#c056), [C057](execution/G4_system.md#c057)                                                                                                                                                                          | [C066](execution/G4_system.md#c066)      | 直搬、indexed、mask、页授权均通过；不能 C032 后整任务 DONE。                        |
| W4-05  | [C029](execution/G2_scalar.md#c029), [C058](execution/G4_system.md#c058)                                                                                                                                                                                                                                                    | [C066](execution/G4_system.md#c066)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W4-06  | [C032](execution/G2_scalar.md#c032), [C059](execution/G4_system.md#c059)                                                                                                                                                                                                                                                    | [C062](execution/G4_system.md#c062)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W5-01  | [C014](execution/G1_contracts.md#c014), [C037](execution/G2_scalar.md#c037)                                                                                                                                                                                                                                                 | [C037](execution/G2_scalar.md#c037)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W5-02  | [C018](execution/G1_contracts.md#c018), [C038](execution/G2_scalar.md#c038), [C052](execution/G3_compute.md#c052)                                                                                                                                                                                                           | [C054](execution/G3_compute.md#c054)     | bootstrap 测量不等真实映像执行；全库资源复核后闭合。                                |
| W5-03  | [C048](execution/G3_compute.md#c048), [C049](execution/G3_compute.md#c049), [C050](execution/G3_compute.md#c050), [C051](execution/G3_compute.md#c051), [C052](execution/G3_compute.md#c052)                                                                                                                                | [C054](execution/G3_compute.md#c054)     | 每个已承诺计算族真实数值通过，不以包/族名存在代执行。                               |
| W5-04  | [C039](execution/G2_scalar.md#c039), [C063](execution/G4_system.md#c063)                                                                                                                                                                                                                                                    | [C066](execution/G4_system.md#c066)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W5-05  | [C063](execution/G4_system.md#c063)                                                                                                                                                                                                                                                                                         | [C067](execution/G4_system.md#c067)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W6-01  | [C015](execution/G1_contracts.md#c015)                                                                                                                                                                                                                                                                                      | [C020](execution/G1_contracts.md#c020)   | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W6-02  | [C033](execution/G2_scalar.md#c033)                                                                                                                                                                                                                                                                                         | [C041](execution/G2_scalar.md#c041)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W6-03  | [C034](execution/G2_scalar.md#c034), [C059](execution/G4_system.md#c059)                                                                                                                                                                                                                                                    | [C062](execution/G4_system.md#c062)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W6-04  | [C060](execution/G4_system.md#c060), [C061](execution/G4_system.md#c061)                                                                                                                                                                                                                                                    | [C062](execution/G4_system.md#c062)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W6-05  | [C062](execution/G4_system.md#c062)                                                                                                                                                                                                                                                                                         | [C067](execution/G4_system.md#c067)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W7-01  | [C007](execution/G0_environment.md#c007), [C016](execution/G1_contracts.md#c016)                                                                                                                                                                                                                                            | [C020](execution/G1_contracts.md#c020)   | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W7-02  | [C021](execution/G2_scalar.md#c021), [C038](execution/G2_scalar.md#c038), [C047](execution/G3_compute.md#c047), [C054](execution/G3_compute.md#c054)                                                                                                                                                                        | [C054](execution/G3_compute.md#c054)     | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W7-03  | [C041](execution/G2_scalar.md#c041), [C054](execution/G3_compute.md#c054), [C064](execution/G4_system.md#c064)                                                                                                                                                                                                              | [C067](execution/G4_system.md#c067)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W7-04  | [C066](execution/G4_system.md#c066), [C067](execution/G4_system.md#c067)                                                                                                                                                                                                                                                    | [C068](execution/G4_system.md#c068)      | 对应卡全部 PASS，相关原计划验收无缺项；产物 hash 与消费者一致。                     |
| W7-05  | [C069](execution/G5_acceptance.md#c069), [C070](execution/G5_acceptance.md#c070), [C071](execution/G5_acceptance.md#c071), [C072](execution/G5_acceptance.md#c072)                                                                                                                                                          | [C072](execution/G5_acceptance.md#c072)  | 预登记、实际测量、重放与单位审计全部通过。                                          |
| W7-06  | [C007](execution/G0_environment.md#c007), [C020](execution/G1_contracts.md#c020), [C041](execution/G2_scalar.md#c041), [C054](execution/G3_compute.md#c054), [C068](execution/G4_system.md#c068), [C074](execution/G5_acceptance.md#c074), [C075](execution/G5_acceptance.md#c075), [C076](execution/G5_acceptance.md#c076) | [C076](execution/G5_acceptance.md#c076)  | 最后关闭整个 CS1 出口，不因为早期 G0/G1 evaluator 已存在就 DONE。                   |

## 9. 每张卡的交接输出

交接正文至少包含以下字段（可以直接复制填写真实值）：

```text
Commit-Guide: Cxxx
State: COMMITTED | VERIFIED_NOT_COMMITTED | BLOCKED
Base-SHA:
Dependency-SHAs:
Changed-files:
Behavior-and-invariants:
Commands-and-exit-results:
Positive-observation:
Negative-observation:
Evidence-archive-and-sha256:
Committed-SHA: 未提交时明确写“未提交”
Closed-W-tasks:
Still-open-W-tasks:
Next-ready-commit-IDs:
```

源码/依赖锁、配置、机器码和真实输出都要能定位；不要输出自述“全绿”而没有运行信息。环境 bootstrap、协议单元、ISA oracle、真实 RTL、模型调度、板卡/PPA 的证据层级分开。当前没有 FPGA 不阻止 CS1，但不生成物理面积/功耗/Fmax 数字。

## 10. 本轮文档验证范围

本轮只交付执行 guide 与计划导航，未创建 hardware 项目、未运行未来 `cn` 命令、未创建 Git commit。文档检查覆盖 76 卡／42 任务映射、依赖无环、命令 producer 前置、相对链接、表格、围栏及 shell 语法。

实际运行了 C022 所列既有 `l2_admission_wait` 命令：`full_memory`、双 Group/Device Context，21686 cycles，`task_completed`、`credit_invariant`、`arena_zero_leak` 三项通过。这仅证明现有 Python 基线和该命令可用，不是未来 CS1/数值/RTL 证据。
