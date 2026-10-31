# hardware/configs — 统一硬件配置格式（schema 1）

所有 target 的 `--config` 入口。P02 定义格式，P03 注册 reference numeric 实例，
P06 注册生产 `numeric-profile`；各任务通过 `config.consumed_sections` 声明消费者。

## Schema 1

```yaml
schema: 1 # 必填，恒为 1
target: toolchain # 必填，必须与 targets.json 中该 target 的 required_target 一致
topology: {} # 保留，显式注册 topology section 消费者前必须为空 mapping 或缺省
numeric: {} # toolchain 不消费；numeric target 的非空格式见下
validator: # toolchain 消费的 section；numeric target 必须为空
  hw_overrides: {} # 可选；键值对，逐项映射为 validator CLI 的 --hw-override k=v
board: {} # 保留（板级消费者注册前必须为空）
```

## 校验规则（container_check.py::validate_config，exit 6 拒绝）

1. 顶层键只能是 `schema/target/topology/numeric/validator/board`，未知键拒绝。
2. `schema` 必须为 1；`target` 必填，且必须同时等于请求的 target 与 registry 的
   `required_target`（错 target 的配置一律拒绝，无 fallback）。
3. 每个 section 若出现必须是 mapping。
4. 不在 `consumed_sections` 中的 section 必须为空 `{}` —— 保留 section 在 owner
   注册消费者之前不接受任何内容（fail closed）。
5. `validator` 内仅允许 `hw_overrides`。
6. `hw_overrides` 键须匹配 `[A-Za-z0-9_][A-Za-z0-9_.-]*`；值须是 bool / number /
   匹配 `[A-Za-z0-9_][A-Za-z0-9_.+-]*` 的字符串（保证可安全序列化为 CLI 参数）。
   执行时按键排序逐项展开为 `--hw-override <k>=<v>`（bool 转 `true|false`），
   施加到真实 validator 运行（通用消费者：python-env case；numeric-profile 的限制见下）。

## hw_overrides 与既有 CLI 的关系

`validator.hw_overrides` 只映射 `pipeline_validator` 的 `--hw-override`（扁平键，
如 `boa_am_slot_capacity_bytes: 65536` → `--hw-override boa_am_slot_capacity_bytes=65536`）。
`--hw-config`（分组 YAML）不走本字段：如需整文件配置，后续 owner 在自己的
section/字段中显式注册映射，不得隐式复用。

## 注册新 section / 新实例的约定

- 新 section：owner 在 `targets.json` 对应 target 的 `config.consumed_sections`
  中加入 section 名，并在自己的 case runner 中解析（schema 仍由 P02 基础校验：
  未知顶层键、错 target、类型错误一律先拒）。未注册即写内容 = 配置被拒。
- 新实例：`hardware/configs/<target-or-purpose>.yaml`。除下述 P05 独立软件
  smoke 配置外，`target:` 必须指向已实现 target；容器分派器仍拒绝未实现 target。
- manifest 保存完整解析后的配置对象（不是路径），reproduce 从 manifest 重建。

## numeric section（P03）

`hardware/configs/numeric.yaml` 为 numeric target 的默认配置：

```yaml
schema: 1
target: numeric
topology: {}
numeric:
  profile: bf16_fp32_mac_rne_v1
  tile_k: 8
  random_cases: 1024
validator: {}
board: {}
```

三个字段可缺省，采用上述默认值。profile 必须精确匹配；tile_k 为正整数；
random_cases 为非负整数，表示 crosscheck 每种 primitive 的随机输入数量；
bool 不视作整数，未知 numeric 字段拒绝。probe-nan 只跑定向/边界/特殊值矩阵，
不消费随机行。tile_k 控制参考模型/向量默认贡献块大小，不更改有意固定 4/8 的反例。

numeric target 不消费 topology/validator/board，必须为空或省略。完整 resolved
config 保存于 manifest，reproduce 使用其值重建请求，不依赖原 YAML 路径仍然存在。

## 现有实例

- `toolchain-minimal.yaml`：P02 toolchain target 的最小实例（四个 case 全跑，
  无 hw_overrides）。`check toolchain --config hardware/configs/toolchain-minimal.yaml ...`
  亦可在省略 `--config` 时由 registry 默认提供。
- `numeric.yaml`：P03 独立 reference/SoftFloat/HardFloat 数值对照默认实例；
  `check numeric --config hardware/configs/numeric.yaml ...` 可显式选择，也可省略 `--config`。
- `numeric-profile.yaml`：P06 生产 full_memory 数值/零工作/v5 cutover gate；
  `check numeric-profile --config hardware/configs/numeric-profile.yaml ...`。
- `shared-am-functional.yaml` / `shared-am-cycle.yaml`：P09 的两个固定场景
  target（Q08/Q09/T01-Q06 与 P07 终态偏序 + trace-audit.sql）；场景钉死硬件
  配置，`validator.hw_overrides` 必须为空，非空被 exit 6 拒绝。

## 生产 numeric-profile 配置（P06）

`numeric-profile.yaml` 是 schema 1 runner wrapper，不是原生 `--hw-config`
（schema 2）。向量自带权威 numeric K（2/4/8/16），因此该 target 要求
`validator.hw_overrides` 为空，其余 section 也为空；非空覆盖明确拒绝而非忽略。
每条向量记录实际执行的 numeric K 和 target fingerprint。

普通 workload 使用 schema 2 的 `engines.boa.numeric_profile` /
`engines.boa.numeric_tile_k`，或 `--hw-override boa_numeric_tile_k=16`；
profile 唯一为 `bf16_fp32_mac_rne_v1`，numeric K 默认8。两者属于数值 target，
不能将新配置应用到旧 artifact；必须从源重编译 v5。

## BOA 1/2/4 engine 软件实例（P05）

`boa-1engine.yaml`、`boa-2engine.yaml`、`boa-4engine.yaml` 使用 schema 1，
目标为 P09 已注册实现的 `two-slot-progress`（registry 默认实例为
`boa-1engine.yaml`）。真实字段在 `validator.hw_overrides`：
`boa_arrays_per_tile`、`boa_array_rows/cols`、`boa_tile_m/n`；其余 section 为空。
容器 runner 把同一文件透传给真实测试模块 CLI 消费；P05 的独立软件消费者
保持可用：

```bash
conda run -n elenor-validator python -m pipeline_validator.tests.test_boa_multi_engine \
  --config hardware/configs/boa-1engine.yaml
hardware/scripts/check two-slot-progress --config hardware/configs/boa-4engine.yaml \
  --case engine-pool-progress --out OUT --seed 20261028
```

`t06-one-engine-many-handles` 只在 `boa-1engine.yaml` 下合法（fail closed）；
多 engine case 用匹配的 2/4 配置。

这些 wrapper 不传给 validator 的 `--hw-config`（后者为 schema 2）。
直接运行工作负载时用 `--hw-override boa_arrays_per_tile=4` 等扁平字段，或
schema-2 YAML 的 `engines.boa.arrays_per_tile/array_rows/array_cols/tile_m/tile_n`。
旧 `engines.boa.opa.*`/`boa_num_opa`/`boa_opa_rows/cols` 已删除，无 alias。
数组数量范围 1–8 是软件实验上界，默认 1，V1 多阵列基线 4；几何默认16×16。
2-engine 下4 handles 的2/2分布是 least-live 的结果，不是每 engine 两个 handle 上限。
