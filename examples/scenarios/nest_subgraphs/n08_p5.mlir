// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// participant count=5：以 placement15/range0..4 与 placement1/range0..1 两个 Context 同时提交。
// 第五项 output 使用独立 arena tensor index=3，不与前四项别名。
// case: {
//       "id": "N08",
//       "name": "n08_p5",
//       "nodes": ["Grid4", "Grid1"],
//       "edges": {},
//       "context_partition": [
//         ["Grid4"],
//         ["Grid1"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "placement": [15, 1]
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "count5 拆为两个合法 grid：task ids 0..3 与 0；两个 Context 同时 submit，aggregate expected 分别为4和1",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "所有已接纳 Context 完成",
//         "Scheduling Quality": "仅记录实际 service/等待，不声称最优调度"
//       },
//       "forbidden_dependencies": "不得把总需求5解释为5个物理 UCE；不得把第五项写入前四项 output HBM 区间",
//       "tensors": {
//         "Input_0_3": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 32768,
//           "arena_interval_bytes": [0, 32768],
//           "role": "external_input"
//         },
//         "Input_4": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 8192,
//           "arena_interval_bytes": [262144, 270336],
//           "role": "external_input"
//         },
//         "Output_0_3": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768,
//           "arena_interval_bytes": [524288, 557056],
//           "role": "output",
//           "logical_participants": [0, 1, 2, 3]
//         },
//         "Output_4": {
//           "index": 3,
//           "offset_elements": 393216,
//           "bytes": 8192,
//           "arena_interval_bytes": [786432, 794624],
//           "role": "output",
//           "logical_participants": [4]
//         }
//       },
//       "phase": {
//         "Grid4": ["pref_Input_0_3"],
//         "Grid1": ["pref_Input_4"],
//         "aggregate": {
//           "Grid4": 4,
//           "Grid1": 1
//         }
//       },
//       "pins": {
//         "Grid4": "device-slot",
//         "Grid1": "device-slot"
//       },
//       "node_programs": {
//         "Grid4": {
//           "program": "prog_Count4",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         },
//         "Grid1": {
//           "program": "prog_Count1",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": "device-slot"
//         }
//       }
//     }
builtin.module {
  tile.program @prog_Count4(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_Count1(
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
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "input_0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_output = nest.alloc slot = "output_0" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_output = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Input_0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Grid4, %read_Grid4, %ready_Grid4 = nest.dispatch.tasks.async @prog_Count4 tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Grid4">, !nest.event<"read_Grid4">, !nest.event<"ready_Grid4">)
    nest.release %b_input depends_on(%read_Grid4, %pref_input)
    %store_Output = nest.dma.store.async %b_output into %h_output depends_on(%ready_Grid4)
      : !nest.event<"store_Output_0">
    nest.release %b_output depends_on(%store_Output)
    nest.await %grid_Grid4, %store_Output
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 1 {
    %b_input = nest.alloc slot = "input_1" role = "in" shape = [1, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<1x64x64xbf16>
    %b_output = nest.alloc slot = "output_1" role = "inout" shape = [1, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<1x64x64xbf16>
    %h_input = nest.subview %arena offsets = [131072] sizes = [4096] strides = [1]
      : !nest.global_view<4096xbf16>
    %h_output = nest.subview %arena offsets = [393216] sizes = [4096] strides = [1]
      : !nest.global_view<4096xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_Input_1">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid_Grid1, %read_Grid1, %ready_Grid1 = nest.dispatch.tasks.async @prog_Count1 tasks(%tasks)
      globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input)
      : (!nest.event<"grid_Grid1">, !nest.event<"read_Grid1">, !nest.event<"ready_Grid1">)
    nest.release %b_input depends_on(%read_Grid1, %pref_input)
    %store_Output = nest.dma.store.async %b_output into %h_output depends_on(%ready_Grid1)
      : !nest.event<"store_Output_1">
    nest.release %b_output depends_on(%store_Output)
    nest.await %grid_Grid1, %store_Output
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    nexus.await %done_0, %done_1
    nexus.return
  }
}
