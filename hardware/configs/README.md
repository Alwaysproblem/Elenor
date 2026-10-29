# hardware/configs — 统一硬件配置格式（schema 1）

所有 target 的 `--config` 入口。P02 定义格式与校验规则；P05 起的实例落在本目录，
`Pnn` 通过 `targets.json` 的 `config.consumed_sections` 注册自己消费的 section。

## Schema 1

```yaml
schema: 1 # 必填，恒为 1
target: toolchain # 必填，必须与 targets.json 中该 target 的 required_target 一致
topology: {} # 保留（P05 拓扑/几何消费者注册前必须为空 mapping 或缺省）
numeric: {} # 保留（数值消费者注册前必须为空）
validator: # toolchain target 当前唯一消费的 section
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
   施加到真实 validator 运行（当前消费者：python-env case）。

## hw_overrides 与既有 CLI 的关系

`validator.hw_overrides` 只映射 `pipeline_validator` 的 `--hw-override`（扁平键，
如 `boa_am_slot_capacity_bytes: 65536` → `--hw-override boa_am_slot_capacity_bytes=65536`）。
`--hw-config`（分组 YAML）不走本字段：如需整文件配置，后续 owner 在自己的
section/字段中显式注册映射，不得隐式复用。

## 注册新 section / 新实例的约定

- 新 section：owner 在 `targets.json` 对应 target 的 `config.consumed_sections`
  中加入 section 名，并在自己的 case runner 中解析（schema 仍由 P02 基础校验：
  未知顶层键、错 target、类型错误一律先拒）。未注册即写内容 = 配置被拒。
- 新实例：`hardware/configs/<target-or-purpose>.yaml`，`target:` 字段必须指向
  已注册且已实现的 target；给未实现 target 引用的配置由分派器拒绝。
- manifest 保存完整解析后的配置对象（不是路径），reproduce 从 manifest 重建。

## 现有实例

- `toolchain-minimal.yaml`：P02 toolchain target 的最小实例（四个 case 全跑，
  无 hw_overrides）。`check toolchain --config hardware/configs/toolchain-minimal.yaml ...`
  亦可在省略 `--config` 时由 registry 默认提供。
