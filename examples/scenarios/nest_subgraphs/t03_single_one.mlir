// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "t03",
//       "name": "t03_single_one",
//       "nodes": ["Step0"],
//       "edges": {},
//       "context_partition": [
//         ["Step0"]
//       ],
//       "resource_config": {
//         "uce_context_mode": 4,
//         "device_context_mode": 1,
//         "placement": 15,
//         "fidelity": "full_memory",
//         "arena_binding": "arena=0x1000000:8388608:rw"
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序",
//         "数值": "未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明",
//         "Liveness": "应完成；以同次运行 trace 和退出状态确认",
//         "Scheduling Quality": "1 次静态展开；活动 grid 数应为 1，不得解释为运行时 early-exit"
//       },
//       "forbidden_dependencies": "只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性",
//       "tensors": {
//         "State0": {
//           "formal": "arena",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "Input0": {
//           "formal": "arena",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "State1": {
//           "formal": "arena",
//           "index": 5,
//           "offset_elements": 655360,
//           "interval_bytes": [1310720, 1343488],
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "Step0": {
//           "program": "prog_Step0",
//           "pin": 0,
//           "engine": "EVU:add",
//           "repeat": 1
//         }
//       },
//       "consumer_ready_granularity": {
//         "State0->Step0": "prefetch 完成",
//         "Input0->Step0": "prefetch 完成"
//       },
//       "notes": {
//         "iteration_count": 1,
//         "event_tags": ["iteration_0"],
//         "zero_case": null,
//         "stable_tensor_indices": {
//           "State0": 0,
//           "Input0": 1,
//           "Input1": 2,
//           "Input2": 3,
//           "Input3": 4,
//           "State1": 5,
//           "State2": 6,
//           "State3": 7,
//           "State4": 8
//         }
//       }
//     }
builtin.module {
  tile.program @prog_Step0(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>,
    %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_State0 = nest.alloc slot = "State0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_Input0 = nest.alloc slot = "Input0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Input0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_State1 = nest.alloc slot = "State1" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State1 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_State0 = nest.dma.prefetch.async %h_State0 into %b_State0 : !nest.event<"pref_State0">
    %pref_Input0 = nest.dma.prefetch.async %h_Input0 into %b_Input0 : !nest.event<"pref_Input0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Step0, %read_Step0, %ready_Step0 = nest.dispatch.tasks.async @prog_Step0 context = 0
      tasks(%tasks) globals() bindings(%b_State0, %b_Input0, %b_State1) ins(%b_State0, %b_Input0)
      outs(%b_State1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_State0, %pref_Input0)
      : (!nest.event<"grid_Step0">, !nest.event<"read_Step0">, !nest.event<"ready_Step0">)
    nest.release %b_State0 depends_on(%read_Step0, %pref_State0)
    nest.release %b_Input0 depends_on(%read_Step0, %pref_Input0)
    %store_State1 = nest.dma.store.async %b_State1 into %h_State1 depends_on(%ready_Step0)
      : !nest.event<"store_State1">
    nest.release %b_State1 depends_on(%store_State1)
    nest.await %grid_Step0, %store_State1
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
