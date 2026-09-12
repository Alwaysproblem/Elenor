// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "t03",
//       "name": "t03_stage",
//       "nodes": ["Step0", "Step1", "Step2", "Step3"],
//       "edges": {
//         "Step1": ["Step0"],
//         "Step2": ["Step1"],
//         "Step3": ["Step2"]
//       },
//       "context_partition": [
//         ["Step0", "Step1"],
//         ["Step2", "Step3"]
//       ],
//       "resource_config": {
//         "uce_context_mode": 4,
//         "device_context_mode": 4,
//         "placement": 15,
//         "fidelity": "full_memory",
//         "arena_binding": "arena=0x1000000:8388608:rw"
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序",
//         "数值": "未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明",
//         "Liveness": "应完成；以同次运行 trace 和退出状态确认",
//         "Scheduling Quality": "4 次静态展开；活动 grid 数应为 4，不得解释为运行时 early-exit"
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
//         "Input1": {
//           "formal": "arena",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "Input2": {
//           "formal": "arena",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "Input3": {
//           "formal": "arena",
//           "index": 4,
//           "offset_elements": 524288,
//           "interval_bytes": [1048576, 1081344],
//           "bytes": 32768
//         },
//         "State1": {
//           "formal": "arena",
//           "index": 5,
//           "offset_elements": 655360,
//           "interval_bytes": [1310720, 1343488],
//           "bytes": 32768
//         },
//         "State2": {
//           "formal": "arena",
//           "index": 6,
//           "offset_elements": 786432,
//           "interval_bytes": [1572864, 1605632],
//           "bytes": 32768
//         },
//         "State3": {
//           "formal": "arena",
//           "index": 7,
//           "offset_elements": 917504,
//           "interval_bytes": [1835008, 1867776],
//           "bytes": 32768
//         },
//         "State4": {
//           "formal": "arena",
//           "index": 8,
//           "offset_elements": 1048576,
//           "interval_bytes": [2097152, 2129920],
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "Step0": {
//           "program": "prog_Step0",
//           "pin": 0,
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "Step1": {
//           "program": "prog_Step1",
//           "pin": 1,
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "Step2": {
//           "program": "prog_Step2",
//           "pin": 2,
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "Step3": {
//           "program": "prog_Step3",
//           "pin": 3,
//           "engine": "EVU:add",
//           "repeat": 1
//         }
//       },
//       "consumer_ready_granularity": {
//         "State0->Step0": "prefetch 完成",
//         "Input0->Step0": "prefetch 完成",
//         "State1->Step1": "output_ready(all_tasks)",
//         "Input1->Step1": "prefetch 完成",
//         "State2->Step2": "producer context_done 后从 State_i HBM 区间 prefetch",
//         "Input2->Step2": "prefetch 完成",
//         "State3->Step3": "output_ready(all_tasks)",
//         "Input3->Step3": "prefetch 完成"
//       },
//       "notes": {
//         "iteration_count": 4,
//         "event_tags": ["iteration_0", "iteration_1", "iteration_2", "iteration_3"],
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
  tile.program @prog_Step1(
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
  tile.program @prog_Step2(
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
  tile.program @prog_Step3(
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
    %b_Input1 = nest.alloc slot = "Input1" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Input1 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_State2 = nest.alloc slot = "State2" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State2 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_State0 = nest.dma.prefetch.async %h_State0 into %b_State0 : !nest.event<"pref_State0">
    %pref_Input0 = nest.dma.prefetch.async %h_Input0 into %b_Input0 : !nest.event<"pref_Input0">
    %pref_Input1 = nest.dma.prefetch.async %h_Input1 into %b_Input1 : !nest.event<"pref_Input1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Step0, %read_Step0, %ready_Step0 = nest.dispatch.tasks.async @prog_Step0 context = 0
      tasks(%tasks) globals() bindings(%b_State0, %b_Input0, %b_State1) ins(%b_State0, %b_Input0)
      outs(%b_State1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_State0, %pref_Input0)
      : (!nest.event<"grid_Step0">, !nest.event<"read_Step0">, !nest.event<"ready_Step0">)
    %grid_Step1, %read_Step1, %ready_Step1 = nest.dispatch.tasks.async @prog_Step1 context = 1
      tasks(%tasks) globals() bindings(%b_State1, %b_Input1, %b_State2) ins(%b_State1, %b_Input1)
      outs(%b_State2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_Step0, %pref_Input1)
      : (!nest.event<"grid_Step1">, !nest.event<"read_Step1">, !nest.event<"ready_Step1">)
    nest.release %b_State0 depends_on(%read_Step0, %pref_State0)
    nest.release %b_Input0 depends_on(%read_Step0, %pref_Input0)
    nest.release %b_Input1 depends_on(%read_Step1, %pref_Input1)
    %store_State1 = nest.dma.store.async %b_State1 into %h_State1 depends_on(%ready_Step0)
      : !nest.event<"store_State1">
    nest.release %b_State1 depends_on(%read_Step1, %store_State1)
    %store_State2 = nest.dma.store.async %b_State2 into %h_State2 depends_on(%ready_Step1)
      : !nest.event<"store_State2">
    nest.release %b_State2 depends_on(%store_State2)
    nest.await %grid_Step0, %grid_Step1, %store_State1, %store_State2
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_State2 = nest.alloc slot = "State2" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State2 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_Input2 = nest.alloc slot = "Input2" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Input2 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_State3 = nest.alloc slot = "State3" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State3 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_Input3 = nest.alloc slot = "Input3" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Input3 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_State4 = nest.alloc slot = "State4" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_State4 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_State2 = nest.dma.prefetch.async %h_State2 into %b_State2 : !nest.event<"pref_State2">
    %pref_Input2 = nest.dma.prefetch.async %h_Input2 into %b_Input2 : !nest.event<"pref_Input2">
    %pref_Input3 = nest.dma.prefetch.async %h_Input3 into %b_Input3 : !nest.event<"pref_Input3">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Step2, %read_Step2, %ready_Step2 = nest.dispatch.tasks.async @prog_Step2 context = 2
      tasks(%tasks) globals() bindings(%b_State2, %b_Input2, %b_State3) ins(%b_State2, %b_Input2)
      outs(%b_State3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_State2, %pref_Input2)
      : (!nest.event<"grid_Step2">, !nest.event<"read_Step2">, !nest.event<"ready_Step2">)
    %grid_Step3, %read_Step3, %ready_Step3 = nest.dispatch.tasks.async @prog_Step3 context = 3
      tasks(%tasks) globals() bindings(%b_State3, %b_Input3, %b_State4) ins(%b_State3, %b_Input3)
      outs(%b_State4)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_Step2, %pref_Input3)
      : (!nest.event<"grid_Step3">, !nest.event<"read_Step3">, !nest.event<"ready_Step3">)
    nest.release %b_State2 depends_on(%read_Step2, %pref_State2)
    nest.release %b_Input2 depends_on(%read_Step2, %pref_Input2)
    nest.release %b_Input3 depends_on(%read_Step3, %pref_Input3)
    %store_State3 = nest.dma.store.async %b_State3 into %h_State3 depends_on(%ready_Step2)
      : !nest.event<"store_State3">
    nest.release %b_State3 depends_on(%read_Step3, %store_State3)
    %store_State4 = nest.dma.store.async %b_State4 into %h_State4 depends_on(%ready_Step3)
      : !nest.event<"store_State4">
    nest.release %b_State4 depends_on(%store_State4)
    nest.await %grid_Step2, %grid_Step3, %store_State3, %store_State4
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    nexus.await %done_0, %done_1
    nexus.return
  }
}
