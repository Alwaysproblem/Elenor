// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "s09",
//       "name": "s09_single_short",
//       "nodes": ["A", "B", "C", "D"],
//       "edges": {
//         "B": ["A"],
//         "C": ["A", "B"],
//         "D": ["A", "B", "C"]
//       },
//       "context_partition": [
//         ["A", "B", "C", "D"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device": 1,
//         "placement": 15,
//         "fidelity": "full_memory",
//         "group_sram_bytes": 196352
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成",
//         "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描",
//         "Liveness": "预期运行期 L2 永久容量 fault，非成功完成",
//         "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"
//       },
//       "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性",
//       "tensors": {
//         "input_A": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 32768
//         },
//         "W_A": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 32768
//         },
//         "A": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768
//         },
//         "B": {
//           "index": 3,
//           "offset_elements": 393216,
//           "bytes": 32768
//         },
//         "C": {
//           "index": 4,
//           "offset_elements": 524288,
//           "bytes": 32768
//         },
//         "D": {
//           "index": 5,
//           "offset_elements": 655360,
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "A": {
//           "program": "prog_A",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "B": {
//           "program": "prog_B",
//           "pin": 1,
//           "engine": "evu",
//           "op": "add",
//           "repeat": 1,
//           "loads": 1
//         },
//         "C": {
//           "program": "prog_C",
//           "pin": 2,
//           "engine": "evu",
//           "op": "add",
//           "repeat": 1,
//           "loads": 1
//         },
//         "D": {
//           "program": "prog_D",
//           "pin": 3,
//           "engine": "evu",
//           "op": "add",
//           "repeat": 1,
//           "loads": 1
//         }
//       },
//       "consumer_ready": {
//         "A": ["input_A.prefetch", "W_A.prefetch"],
//         "B": ["A.output_ready"],
//         "C": ["A.output_ready", "B.output_ready"],
//         "D": ["A.output_ready", "B.output_ready", "C.output_ready"]
//       },
//       "说明": "完整 bundle 为 6U=196608 B；配置容量 196352 B。short 应为 admission capacity fault，不是等待。"
//     }
builtin.module {
  tile.program @prog_A(
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
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
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
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_C(
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
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_D(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>,
    %i2: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v2 = tile.subview %i2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l2 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    %load_0_2 = tile.load.async %v2 into %l2 : !tile.event<"load_0_2">
    tile.await %load_0_0, %load_0_1, %load_0_2
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "add" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l1
    tile.free %l2
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_A = nest.alloc slot = "input_A" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_A = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_A = nest.alloc slot = "W_A" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_W_A = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_A = nest.alloc slot = "A" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_A = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_B = nest.alloc slot = "B" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_B = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_C = nest.alloc slot = "C" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_C = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_D = nest.alloc slot = "D" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_D = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input_A = nest.dma.prefetch.async %h_input_A into %b_input_A : !nest.event<"pref_input_A">
    %pref_W_A = nest.dma.prefetch.async %h_W_A into %b_W_A : !nest.event<"pref_W_A">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A, %read_A, %ready_A = nest.dispatch.tasks.async @prog_A context = 0 tasks(%tasks)
      globals() bindings(%b_input_A, %b_W_A, %b_A) ins(%b_input_A, %b_W_A) outs(%b_A)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input_A, %pref_W_A)
      : (!nest.event<"grid_A">, !nest.event<"read_A">, !nest.event<"ready_A">)
    %grid_B, %read_B, %ready_B = nest.dispatch.tasks.async @prog_B context = 1 tasks(%tasks)
      globals() bindings(%b_A, %b_B) ins(%b_A) outs(%b_B)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A)
      : (!nest.event<"grid_B">, !nest.event<"read_B">, !nest.event<"ready_B">)
    %grid_C, %read_C, %ready_C = nest.dispatch.tasks.async @prog_C context = 2 tasks(%tasks)
      globals() bindings(%b_A, %b_B, %b_C) ins(%b_A, %b_B) outs(%b_C)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A, %ready_B)
      : (!nest.event<"grid_C">, !nest.event<"read_C">, !nest.event<"ready_C">)
    %grid_D, %read_D, %ready_D = nest.dispatch.tasks.async @prog_D context = 3 tasks(%tasks)
      globals() bindings(%b_A, %b_B, %b_C, %b_D) ins(%b_A, %b_B, %b_C) outs(%b_D)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_A, %ready_B, %ready_C)
      : (!nest.event<"grid_D">, !nest.event<"read_D">, !nest.event<"ready_D">)
    nest.release %b_input_A depends_on(%read_A, %pref_input_A)
    nest.release %b_W_A depends_on(%read_A, %pref_W_A)
    %store_A_0 = nest.dma.store.async %b_A into %h_A depends_on(%ready_A) : !nest.event<"store_A_0">
    nest.release %b_A depends_on(%read_B, %read_C, %read_D, %store_A_0)
    %store_B_0 = nest.dma.store.async %b_B into %h_B depends_on(%ready_B) : !nest.event<"store_B_0">
    nest.release %b_B depends_on(%read_C, %read_D, %store_B_0)
    %store_C_0 = nest.dma.store.async %b_C into %h_C depends_on(%ready_C) : !nest.event<"store_C_0">
    nest.release %b_C depends_on(%read_D, %store_C_0)
    %store_D_0 = nest.dma.store.async %b_D into %h_D depends_on(%ready_D) : !nest.event<"store_D_0">
    nest.release %b_D depends_on(%store_D_0)
    nest.await %grid_A, %grid_B, %grid_C, %grid_D, %store_A_0, %store_B_0, %store_C_0, %store_D_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
