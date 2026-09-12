// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// task.range from=0 to=0 违反 from<to；这是本文件唯一故意违反的范围合同。
// case: {
//       "id": "N08",
//       "name": "n08_invalid_zero",
//       "nodes": ["InvalidZero"],
//       "edges": {},
//       "context_partition": [
//         ["InvalidZero"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "placement": 1,
//         "expected_exit": 2
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "task.range from=0 to=0 违反 from<to；除此之外 program/signature/view/actual/release/store 均合法",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "预期 verifier 拒绝并返回 exit 2",
//         "Scheduling Quality": "不执行调度质量判断"
//       },
//       "forbidden_dependencies": "不得用 ParseError、越界 view、缺失 release 或错误 actual 列表替代指定 verifier 错误",
//       "tensors": {
//         "Input": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 8192,
//           "arena_interval_bytes": [0, 8192],
//           "role": "external_input"
//         },
//         "Output": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 8192,
//           "arena_interval_bytes": [262144, 270336],
//           "role": "output"
//         }
//       },
//       "phase": {
//         "InvalidZero": ["pref_Input"],
//         "invalid_invariant": "task.range from=0 to=0 违反 from<to"
//       },
//       "pins": {
//         "InvalidZero": "device-slot"
//       },
//       "node_programs": {
//         "InvalidZero": {
//           "program": "prog_InvalidZero",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         }
//       }
//     }
builtin.module {
  tile.program @prog_InvalidZero(
    %task: !nest.task, %i0: !nest.l2_buffer<1x64x64xbf16>, %out: !nest.l2_buffer<1x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 1 {
    %b_input = nest.alloc slot = "input" role = "in" shape = [1, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<1x64x64xbf16>
    %b_output = nest.alloc slot = "output" role = "inout" shape = [1, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<1x64x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [4096] strides = [1]
      : !nest.global_view<4096xbf16>
    %h_output = nest.subview %arena offsets = [131072] sizes = [4096] strides = [1]
      : !nest.global_view<4096xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Input">
    %tasks = nest.task.range from = 0 to = 0 : !nest.task_range
    %grid_InvalidZero, %read_InvalidZero, %ready_InvalidZero =
      nest.dispatch.tasks.async @prog_InvalidZero tasks(%tasks) globals()
      bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input) : (
        !nest.event<"grid_InvalidZero">, !nest.event<"read_InvalidZero">,
        !nest.event<"ready_InvalidZero">)
    nest.release %b_input depends_on(%read_InvalidZero, %pref_input)
    %store_Output = nest.dma.store.async %b_output into %h_output depends_on(%ready_InvalidZero)
      : !nest.event<"store_Output">
    nest.release %b_output depends_on(%store_Output)
    nest.await %grid_InvalidZero, %store_Output
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
