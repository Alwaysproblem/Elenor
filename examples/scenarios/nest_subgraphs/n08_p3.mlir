// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// participant count=3；L2 shape、task.range 与 placement popcount 精确一致。
// case: {
//       "id": "N08",
//       "name": "n08_p3",
//       "nodes": ["Grid3"],
//       "edges": {},
//       "context_partition": [
//         ["Grid3"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "placement": 7
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "placement=7、task.range=0..3；input/output L2 首维均为 3，aggregate expected/seen=3",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "所有已接纳 Context 完成",
//         "Scheduling Quality": "仅记录实际 service/等待，不声称最优调度"
//       },
//       "forbidden_dependencies": "不得从固定 4-task 类型读取尾 participant，也不得等待掩码外任务",
//       "tensors": {
//         "Input": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 24576,
//           "arena_interval_bytes": [0, 24576],
//           "role": "external_input"
//         },
//         "Output": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 24576,
//           "arena_interval_bytes": [262144, 286720],
//           "role": "output"
//         }
//       },
//       "phase": {
//         "Grid3": ["pref_Input"],
//         "aggregate": {
//           "expected": 3,
//           "task_ids": [0, 1, 2]
//         }
//       },
//       "pins": {
//         "Grid3": "device-slot"
//       },
//       "node_programs": {
//         "Grid3": {
//           "program": "prog_Count3",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         }
//       }
//     }
builtin.module {
  tile.program @prog_Count3(
    %task: !nest.task, %i0: !nest.l2_buffer<3x64x64xbf16>, %out: !nest.l2_buffer<3x64x64xbf16>) {
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
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 7 {
    %b_input = nest.alloc slot = "input" role = "in" shape = [3, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<3x64x64xbf16>
    %b_output = nest.alloc slot = "output" role = "inout" shape = [3, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<3x64x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [12288] strides = [1]
      : !nest.global_view<12288xbf16>
    %h_output = nest.subview %arena offsets = [131072] sizes = [12288] strides = [1]
      : !nest.global_view<12288xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_input">
    %tasks = nest.task.range from = 0 to = 3 : !nest.task_range
    %grid_Grid3, %read_Grid3, %ready_Grid3 = nest.dispatch.tasks.async @prog_Count3 tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Grid3">, !nest.event<"read_Grid3">, !nest.event<"ready_Grid3">)
    nest.release %b_input depends_on(%read_Grid3, %pref_input)
    %store_Output = nest.dma.store.async %b_output into %h_output depends_on(%ready_Grid3)
      : !nest.event<"store_Output">
    nest.release %b_output depends_on(%store_Output)
    nest.await %grid_Grid3, %store_Output
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
