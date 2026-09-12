// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "t04",
//       "name": "t04_stage_uniform",
//       "nodes": [
//         "A0",
//         "B0",
//         "C0",
//         "A1",
//         "B1",
//         "C1",
//         "A2",
//         "B2",
//         "C2",
//         "A3",
//         "B3",
//         "C3"
//       ],
//       "edges": {
//         "B0": ["A0"],
//         "B1": ["A1"],
//         "B2": ["A2"],
//         "B3": ["A3"],
//         "C0": ["B0"],
//         "C1": ["B1"],
//         "C2": ["B2"],
//         "C3": ["B3"]
//       },
//       "context_partition": [
//         ["A0", "B0", "C0"],
//         ["A1", "B1", "C1"],
//         ["A2", "B2", "C2"],
//         ["A3", "B3", "C3"]
//       ],
//       "resource_config": {
//         "uce_context_mode": 4,
//         "device_context_mode": 4,
//         "placement": 15,
//         "fidelity": "full_memory",
//         "bindings": {
//           "R0": "R0=0x1000000:8388608:rw",
//           "R1": "R1=0x2000000:8388608:rw",
//           "R2": "R2=0x3000000:8388608:rw",
//           "R3": "R3=0x4000000:8388608:rw",
//           "W": "W=0x5000000:32768:r"
//         }
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查真实 HBM/L2 数据边、input_released/output_ready、最终 Store 与释放顺序",
//         "数值": "未建模；仅使用真实搬运和合成 engine service，不作为 tensor 数值证明",
//         "Liveness": "应完成；以同次运行 trace 和退出状态确认",
//         "Scheduling Quality": "uniform 对照：R0 也恢复短 EVU；仍在 R1 context_done 后提交 R3，但不强求 R3 复用 R1 的 slot"
//       },
//       "forbidden_dependencies": "只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性",
//       "tensors": {
//         "R0.input": {
//           "formal": "R0",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "R0.A": {
//           "formal": "R0",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "R0.B": {
//           "formal": "R0",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "R0.C": {
//           "formal": "R0",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "R1.input": {
//           "formal": "R1",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "R1.A": {
//           "formal": "R1",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "R1.B": {
//           "formal": "R1",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "R1.C": {
//           "formal": "R1",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "R2.input": {
//           "formal": "R2",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "R2.A": {
//           "formal": "R2",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "R2.B": {
//           "formal": "R2",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "R2.C": {
//           "formal": "R2",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "R3.input": {
//           "formal": "R3",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "R3.A": {
//           "formal": "R3",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "R3.B": {
//           "formal": "R3",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "R3.C": {
//           "formal": "R3",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "W": {
//           "formal": "W",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768,
//           "readonly": true
//         }
//       },
//       "node_programs": {
//         "A0": {
//           "program": "prog_A_fast",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "B0": {
//           "program": "prog_B",
//           "pin": "跟随 device slot",
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "C0": {
//           "program": "prog_C",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "A1": {
//           "program": "prog_A_fast",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "B1": {
//           "program": "prog_B",
//           "pin": "跟随 device slot",
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "C1": {
//           "program": "prog_C",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "A2": {
//           "program": "prog_A_fast",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "B2": {
//           "program": "prog_B",
//           "pin": "跟随 device slot",
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "C2": {
//           "program": "prog_C",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "A3": {
//           "program": "prog_A_fast",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         },
//         "B3": {
//           "program": "prog_B",
//           "pin": "跟随 device slot",
//           "engine": "EVU:add",
//           "repeat": 1
//         },
//         "C3": {
//           "program": "prog_C",
//           "pin": "跟随 device slot",
//           "engine": "EVU:relu",
//           "repeat": 1
//         }
//       },
//       "consumer_ready_granularity": {
//         "A0->B0": "output_ready(all_tasks)",
//         "B0->C0": "output_ready(all_tasks)",
//         "A1->B1": "output_ready(all_tasks)",
//         "B1->C1": "output_ready(all_tasks)",
//         "A2->B2": "output_ready(all_tasks)",
//         "B2->C2": "output_ready(all_tasks)",
//         "A3->B3": "output_ready(all_tasks)",
//         "B3->C3": "output_ready(all_tasks)"
//       },
//       "notes": {
//         "request_intervals": "每个 Rr formal 内 interval 0/1/2/3 分别为 input/A/B/C；请求间不别名",
//         "shared_weight": "W 是 4x64x64xbf16 只读 formal，所有 A 实际读取",
//         "reuse_identity": "stage 的 R1/R2/R3（uniform 还包括 R0）重复提交同一 ctx_request_fast；局部 event tag 相同，nexus done_Rr 唯一",
//         "generation_scope": "只证明正常重复 submit 的 namespace/slot generation 隔离；不伪造旧通知注入"
//       }
//     }
builtin.module {
  tile.program @prog_A_fast(
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
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
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
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_C(
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
  nest.context @ctx_request_fast(
    %Req: !nest.global_memref<4194304xbf16>, %W: !nest.global_memref<4x64x64xbf16>)
    placement = 15 {
    %b_input = nest.alloc slot = "input" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_W = nest.alloc slot = "W" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %b_A = nest.alloc slot = "A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %b_B = nest.alloc slot = "B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %b_C = nest.alloc slot = "C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_input = nest.subview %Req offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_W = nest.subview %W offsets = [0, 0, 0] sizes = [4, 64, 64] strides = [1, 1, 1]
      : !nest.global_view<4x64x64xbf16>
    %h_A = nest.subview %Req offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B = nest.subview %Req offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_C = nest.subview %Req offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_input">
    %pref_W = nest.dma.prefetch.async %h_W into %b_W : !nest.event<"pref_W">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A, %read_A, %ready_A = nest.dispatch.tasks.async @prog_A_fast tasks(%tasks) globals()
      bindings(%b_input, %b_W, %b_A) ins(%b_input, %b_W) outs(%b_A)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input, %pref_W)
      : (!nest.event<"grid_A">, !nest.event<"read_A">, !nest.event<"ready_A">)
    %grid_B, %read_B, %ready_B = nest.dispatch.tasks.async @prog_B tasks(%tasks) globals()
      bindings(%b_A, %b_B) ins(%b_A) outs(%b_B)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A)
      : (!nest.event<"grid_B">, !nest.event<"read_B">, !nest.event<"ready_B">)
    %grid_C, %read_C, %ready_C = nest.dispatch.tasks.async @prog_C tasks(%tasks) globals()
      bindings(%b_B, %b_C) ins(%b_B) outs(%b_C)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_B)
      : (!nest.event<"grid_C">, !nest.event<"read_C">, !nest.event<"ready_C">)
    nest.release %b_input depends_on(%read_A, %pref_input)
    nest.release %b_W depends_on(%read_A, %pref_W)
    %store_A = nest.dma.store.async %b_A into %h_A depends_on(%ready_A) : !nest.event<"store_A">
    nest.release %b_A depends_on(%read_B, %store_A)
    %store_B = nest.dma.store.async %b_B into %h_B depends_on(%ready_B) : !nest.event<"store_B">
    nest.release %b_B depends_on(%read_C, %store_B)
    %store_C = nest.dma.store.async %b_C into %h_C depends_on(%ready_C) : !nest.event<"store_C">
    nest.release %b_C depends_on(%store_C)
    nest.await %grid_A, %grid_B, %grid_C, %store_A, %store_B, %store_C
    nest.return
  }
  nexus.program @run(
    %R0: !nest.global_memref<4194304xbf16>, %R1: !nest.global_memref<4194304xbf16>,
    %R2: !nest.global_memref<4194304xbf16>, %R3: !nest.global_memref<4194304xbf16>,
    %W: !nest.global_memref<4x64x64xbf16>) {
    %done_R0 = nexus.submit_context.async @ctx_request_fast(%R0, %W) : !nexus.event<"done_R0">
    %done_R1 = nexus.submit_context.async @ctx_request_fast(%R1, %W) : !nexus.event<"done_R1">
    %done_R2 = nexus.submit_context.async @ctx_request_fast(%R2, %W) : !nexus.event<"done_R2">
    nexus.await %done_R1
    %done_R3 = nexus.submit_context.async @ctx_request_fast(%R3, %W) : !nexus.event<"done_R3">
    nexus.await %done_R0, %done_R1, %done_R2, %done_R3
    nexus.return
  }
}
