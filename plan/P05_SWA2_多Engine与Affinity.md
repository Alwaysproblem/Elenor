# P05 · SW-A2 多 engine 与固定 affinity

状态：未开始
前置：P04 通过（per-Tile manager 与 per-Tile 报告是本任务的底座；P02 只影响 target 注册时机，不阻塞软件实现）。
对应原方案：SW-A2（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §1.3、§3.1、§4.1、§7.2 SW-A 表）；T04/T05/T06（§8.1）；总则 §3.1。

## 1. 目标与不做

目标：引入每 Tile 多 BOA engine（`arrays_per_tile`），engine 独立于 group 按 “live handle 数最少、平局最小 ID” 在授予时一次性选定；affinity 自授予保持到 slot_free，忙时等原 engine，禁止迁移。新配置键与 fingerprint 的唯一软件落点在本任务；**本任务自身完成必要 target identity 升级并拒绝旧拓扑 artifact**（不等 P06）。

不改数值/终态/申请协议，不实现RTL；测试不篡改bound_engine制造非法正常状态。合法estimate计算保留，不把它当成应删除的旧operandless Matmul。

## 2. 前置与输入

- P04 evidence（per-Tile manager、`shared_am_per_tile` 报告、`boa_am_topology` 身份字段）。
- 设计 §4.1 授予原子性与 engine 选择规则；§3.4（bound_engine 可接收是 STEP 发射条件）；§8.1 T04/T05/T06/T20；§8.1c F05（板级由 P27/P28 复验，本任务交出软件层证据与配置拼写）。
- 总则 §3.1（“单 engine 可服务超过两个 live handles；每 engine 同时最多一个 executing step”）与 §4.1（本任务交接最终配置拼写）。

## 3. 当前证据与修改文件

现状（符号定位）：

- `pipeline_validator/runtime/shared_am.py::_allocate_lease`：`bound_engine=slot.group`（affinity 只是元数据且与 group 绑定）；`_engine_handles` 按 bound_engine 计数；`reset()` 清空。
- `pipeline_validator/tile.py`：`ComputeTile.__init__` 只建 `self.boa = BOAEngine(cfg, tile_id, tracer)` 并挂 tile 级 `BoaResultService`；`TileUCE._engine_for_queue(queue_key="BOA")` 返回 `tile.boa`；launch 构建处 `engine = self._engine_for_queue(tile, queue_key)`。
- `pipeline_validator/config.py`：`"engines.boa.*": "boa_*"` 映射表与 `HardwareConfig` 字段；`hardware_config.yaml` `engines.boa` 段。
- `pipeline_validator/execution_verifier.py`：`_HW_TARGET_FIELDS`（P04 已加 `boa_am_topology`）。

修改config/yaml、SharedAmManager、tile.py、engines.py、PMU/trace/report聚合、fingerprint、全部调用方与相关fixtures/tests；新增1/2/4engine配置。导出符号先查references，迁移tick/busy/drain/reset/cancel/summary，不能只改launch。

## 4. 输入输出、接口与不变量

- 新配置（唯一软件落点，拼写以本任务 evidence 交接为准）：yaml `engines.boa.arrays_per_tile: int`（≥1，默认 1）→ 扁平 `boa_arrays_per_tile`；映射表 + `HardwareConfig` 字段 + `_HW_TARGET_FIELDS` 各加一条。配置越界（0/负数/超大）在配置加载时报错。engine id 域 `0..arrays_per_tile-1`，group id 域仍是 `0..3`，两者独立。
- 几何从旧OPA专用命名干净迁移到 `engines.boa.array_rows/array_cols`→`boa_array_rows/boa_array_cols`，独立 `tile_m/tile_n`→`boa_tile_m/boa_tile_n` 默认16/16并进入target identity；它们控制实际分块/时序而非纯描述。K数值块独立命名 `engines.boa.numeric_tile_k`→`boa_numeric_tile_k`（默认8，由P06落地），不混同WorkloadConfig.tile_k。迁移所有旧geometry消费者，不保留alias；已有estimate路径也使用明确的新几何。
- target identity：`boa_arrays_per_tile` 进入 fingerprint → 所有 P05 前编译 artifact（含 P04 后重编译的）因 target_hash 变化被拒；artifact 序列化格式不变，故 schema/ABI 不升（版本决策与证据写入 evidence；若发现格式耦合则按严格规则 bump）。
- 不变量：授予原子取得 slot/quota/ChainTable/mailbox + engine 绑定，不占槽后等元数据；`bound_engine` 在 lease 生命周期内恒定；忙 engine 上该 context 等（`WAIT_ENGINE_QUEUE` 语义），其他 engine 空闲也不迁移；每 engine 同时至多一个 executing step（每实例 pipeline_depth=1 不变）；“从空启动 + least-live 均衡下每 engine ≤2 handles” 只是分布结果，不是硬件隐限。

## 5. 实施步骤

1. 配置与指纹：按 §4 加键/字段/映射/yaml 注释；更新 `hardware_config.yaml`（`arrays_per_tile: 1` 默认注释说明 V1 目标 4）。构造非法值负例。
2. `SharedAmManager.__init__` 增加 `engine_count` 参数（来自 `cfg.boa_arrays_per_tile`，由 `ComputeTile` 传入）；`_allocate_lease` 改为 `bound_engine = argmin(_engine_handles[e] for e in range(engine_count)), 平局取最小 e`，删除 `=slot.group`；trace `am_allocate.bound_engine` 字段语义变为 engine id（保留字段名）。
3. 构造boa_engines池，给每实例独立engine_id、busy/tick/stall/trace身份；共享Tile allocator与有界结果服务。删除tile.boa旧接口，迁移全部调用方。typed按lease.bound_engine路由；estimate无AM，保留合法时序执行，单阵列使用engine0，多阵列用明确记录的确定性ready仲裁，不获取AM、不冒充数值。
4. 按bound_engine分开候选/队列服务，不能一个忙engine队头挡住其他engine合法ready工作；同context程序序仍保持，同链单issued约束不变。共享BoaResultService每Tile每cycle只推进一次，不能因E个engine而重复tick/free/event；fault/reset作用域和每engine结果归属准确。
5. 测试（全部用合法 IR+配置构造，不改内部状态）：
   - T04：同 context 两链经 least-live 落到不同 engine，trace 中每链全部 step 的 bound_engine 恒定，两条链独立推进；
   - T05：构造 bound engine 忙（长 step）+ 其他空闲，断言不迁移、等待原因可见；
   - T06：`arrays_per_tile=1` 配置下两 context × 2 链 → 单 engine 服务 4 个 live handles（>2），step 公平交替；`arrays_per_tile=2` 下从空启动 4 handles → 2/2 均衡，断言这是分布而非硬上限（注释引用总则 §3.1）；
   - T20 雏形：单 context 两独立输出在 ≥2 engines 时并发；
   - 授予原子性：授予时刻 engine 绑定与 slot/quota 同 trace 周期出现。
6. 配置实例（`hardware/configs/`，P02 格式）：`boa-4engine.yaml`（默认拓扑基线）、`boa-1engine.yaml`、`boa-2engine.yaml`（T06 证据与 P27/P28 板级 2-engine F05 用例的软件参照）。
7. 重跑typed与estimate混合、split-K/multicontext，记录新周期和PMU；删除仅固定旧实现数字的测试，保留可解释边界/资源/并发回归，不重新钉新周期常量。
8. 交付独立测试/smoke与配置；two-slot-progress统一target由P09注册并消费本任务case，避免P05隐式依赖P02。
9. 全量回归 `pre-commit` + `pytest -q -n 4`。

## 6. 验收表

| Case ID                        | 构造                                    | 必须观测                                                                   | 失败特征                       |
| ------------------------------ | --------------------------------------- | -------------------------------------------------------------------------- | ------------------------------ |
| T04-P05-1 affinity 恒定        | 两链不同 engine 全程 trace              | 每链所有 step bound_engine 不变                                            | 出现迁移字段变化               |
| T05-P05-2 忙不迁移             | bound 忙 + 邻 engine 空闲               | 等待原因为 engine 忙；不换 engine；解除后继续                              | 静默换 engine                  |
| T06-P05-3 单 engine >2 handles | `boa-1engine.yaml`，4 live handles      | engine 正常服务 4 handles，公平交替；无 quota 违规（quota 仍按 context≤2） | 出现 “>2 即拒绝” 的隐限        |
| T06-P05-4 2-engine 均衡        | `boa-2engine.yaml`，4 handles           | 2/2 分布；文档注明分布非上限                                               | 把 2/2 写成硬约束断言          |
| ID-P05-5 旧 artifact 拒绝      | P05 前 artifact 加载                    | `target fingerprint mismatch`，非零                                        | 旧 hash 静默通过               |
| ID-P05-6 新键生效              | `--hw-override`/yaml 改 arrays_per_tile | fingerprint 变化；engine 数随之；非法值配置报错                            | 键存在但不进指纹/不生效        |
| GRANT-P05-7 原子授予           | 授予 trace                              | slot/quota/mailbox/engine 绑定同一周期完成                                 | 先占槽后补元数据               |
| REG-P05-8 全量 gate            | 步骤 9                                  | exit 0                                                                     | 漏迁移 `tile.boa` 引用导致失败 |

## 7. 可执行命令

- `conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_am.py -q`（子集先行）再全量。
- 真CLI：`bash examples/run.sh matmul-splitk-pipeline --hw-override boa_arrays_per_tile=4 --json --report /tmp/p05/splitk.json`，确认配置真实影响engine并发而非仅改变指纹。
- P09注册后运行 `hardware/scripts/check two-slot-progress --out /tmp/p09/progress --config hardware/configs/boa-1engine.yaml --case t06-one-engine-many-handles`；多engine case另用匹配2/4配置，不用4engine配置验证单engine>2句。

## 8. 产物、交接与停止条件

交付engine池/调度/trace/配置、1/2/4engine实例和P05 evidence，交接新array_rows/cols、tile_m/n以及P06的numeric_tile_k/profile命名。typed/estimate均保持真实执行；不能因estimate无lease就拒绝整个已有时序功能。统一runner由P09集成。
