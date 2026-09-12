// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// A output 与 B input 是两个独立 L2 allocation，但映射同一 HBM tensor index=1。
// 关键顺序为 store_done → 显式 nest.await → 无 depends_on 的真实 prefetch；用 PC 顺序表达可见性。
// case: {
//       "id": "N10",
//       "name": "n10_local_visibility",
//       "nodes": ["A", "B"],
//       "edges": {
//         "B": ["A"]
//       },
//       "context_partition": [
//         ["A", "B"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device_contexts": 4,
//         "fidelity": "full_memory",
//         "num_dma_channels": 2,
//         "arena_binding": "arena=0x1000000:8388608:rw",
//         "placement": 15
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "同 Context 中 A global Store 完成事件被显式 nest.await；只有越过该 PC await 后才从同一 HBM 地址 prefetch 到独立 B_input L2",
//         "数值": "未建模；engine descriptor 与未初始化的合成输出只用于时序",
//         "Liveness": "所有已接纳 Context 完成",
//         "Scheduling Quality": "仅记录实际 service/等待，不声称最优调度"
//       },
//       "forbidden_dependencies": "禁止给 prefetch 捏造 depends_on；禁止直接把 A 的 output_ready 当作 HBM 可见",
//       "tensors": {
//         "A_input": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 32768,
//           "arena_interval_bytes": [0, 32768],
//           "role": "external_input"
//         },
//         "A_output/B_input": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 32768,
//           "arena_interval_bytes": [262144, 294912],
//           "role": "producer_output_and_consumer_input",
//           "alias": true
//         },
//         "B_output": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768,
//           "arena_interval_bytes": [524288, 557056],
//           "role": "output"
//         }
//       },
//       "phase": {
//         "A": ["pref_A_input"],
//         "B": [
//           "explicit nest.await store_done",
//           "pref_A_output_without_depends_on"
//         ]
//       },
//       "pins": {
//         "A": 0,
//         "B": 1
//       },
//       "node_programs": {
//         "A": {
//           "program": "prog_A",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": 0
//         },
//         "B": {
//           "program": "prog_B",
//           "engine": "evu:relu",
//           "repeat": 1,
//           "pin": 1
//         }
//       }
//     }
builtin.module {
  tile.program @prog_A(
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
  tile.program @prog_B(
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
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A_input = nest.alloc slot = "A_input" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_A_output = nest.alloc slot = "A_output" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B_input = nest.alloc slot = "B_input_from_A" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B_output = nest.alloc slot = "B_output" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A_input = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_A_output = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B_output = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_A_input = nest.dma.prefetch.async %h_A_input into %b_A_input : !nest.event<"pref_A_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A, %read_A, %ready_A = nest.dispatch.tasks.async @prog_A context = 0 tasks(%tasks)
      globals() bindings(%b_A_input, %b_A_output) ins(%b_A_input) outs(%b_A_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A_input)
      : (!nest.event<"grid_A">, !nest.event<"read_A">, !nest.event<"ready_A">)
    nest.release %b_A_input depends_on(%read_A, %pref_A_input)
    %store_done = nest.dma.store.async %b_A_output into %h_A_output depends_on(%ready_A)
      : !nest.event<"store_done">
    nest.release %b_A_output depends_on(%store_done)
    nest.await %store_done
    %pref_A_output = nest.dma.prefetch.async %h_A_output into %b_B_input
      : !nest.event<"pref_A_output">
    %grid_B, %read_B, %ready_B = nest.dispatch.tasks.async @prog_B context = 1 tasks(%tasks)
      globals() bindings(%b_B_input, %b_B_output) ins(%b_B_input) outs(%b_B_output)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A_output)
      : (!nest.event<"grid_B">, !nest.event<"read_B">, !nest.event<"ready_B">)
    nest.release %b_B_input depends_on(%read_B, %pref_A_output)
    %store_B = nest.dma.store.async %b_B_output into %h_B_output depends_on(%ready_B)
      : !nest.event<"store_B">
    nest.release %b_B_output depends_on(%store_B)
    nest.await %grid_A, %grid_B, %store_B
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
