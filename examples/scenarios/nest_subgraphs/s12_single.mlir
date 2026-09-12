// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "s12",
//       "name": "s12_single",
//       "nodes": ["X", "Q", "K", "V", "QK", "SM", "PV", "O"],
//       "edges": {
//         "Q": ["X"],
//         "K": ["X"],
//         "V": ["X"],
//         "QK": ["Q", "K"],
//         "SM": ["QK"],
//         "PV": ["SM", "V"],
//         "O": ["PV"]
//       },
//       "context_partition": [
//         ["X", "Q", "K", "V", "QK", "SM", "PV", "O"]
//       ],
//       "resource_config": {
//         "uce": 4,
//         "device": 1,
//         "placement": 15,
//         "fidelity": "full_memory"
//       },
//       "expected": {
//         "时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成",
//         "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描",
//         "Liveness": "预期完成",
//         "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"
//       },
//       "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性",
//       "tensors": {
//         "input_X": {
//           "index": 0,
//           "offset_elements": 0,
//           "bytes": 32768
//         },
//         "W_Q": {
//           "index": 1,
//           "offset_elements": 131072,
//           "bytes": 32768
//         },
//         "W_K": {
//           "index": 2,
//           "offset_elements": 262144,
//           "bytes": 32768
//         },
//         "W_V": {
//           "index": 3,
//           "offset_elements": 393216,
//           "bytes": 32768
//         },
//         "X": {
//           "index": 4,
//           "offset_elements": 524288,
//           "bytes": 32768
//         },
//         "Q": {
//           "index": 5,
//           "offset_elements": 655360,
//           "bytes": 32768
//         },
//         "K": {
//           "index": 6,
//           "offset_elements": 786432,
//           "bytes": 32768
//         },
//         "V": {
//           "index": 7,
//           "offset_elements": 917504,
//           "bytes": 32768
//         },
//         "QK": {
//           "index": 8,
//           "offset_elements": 1048576,
//           "bytes": 32768
//         },
//         "SM": {
//           "index": 9,
//           "offset_elements": 1179648,
//           "bytes": 32768
//         },
//         "PV": {
//           "index": 10,
//           "offset_elements": 1310720,
//           "bytes": 32768
//         },
//         "O": {
//           "index": 11,
//           "offset_elements": 1441792,
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "X": {
//           "program": "prog_X",
//           "pin": 0,
//           "engine": "copy",
//           "op": "copy",
//           "repeat": 1,
//           "loads": 1
//         },
//         "Q": {
//           "program": "prog_Q",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "K": {
//           "program": "prog_K",
//           "pin": 1,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "V": {
//           "program": "prog_V",
//           "pin": 2,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "QK": {
//           "program": "prog_QK",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "SM": {
//           "program": "prog_SM",
//           "pin": 1,
//           "engine": "evu",
//           "op": "softmax",
//           "repeat": 1,
//           "loads": 1
//         },
//         "PV": {
//           "program": "prog_PV",
//           "pin": 0,
//           "engine": "boa",
//           "op": "matmul",
//           "repeat": 1,
//           "loads": 1
//         },
//         "O": {
//           "program": "prog_O",
//           "pin": 0,
//           "engine": "copy",
//           "op": "copy",
//           "repeat": 1,
//           "loads": 1
//         }
//       },
//       "consumer_ready": {
//         "X": ["input_X.prefetch"],
//         "Q": ["X.output_ready", "W_Q.prefetch"],
//         "K": ["X.output_ready", "W_K.prefetch"],
//         "V": ["X.output_ready", "W_V.prefetch"],
//         "QK": ["Q.output_ready", "K.output_ready"],
//         "SM": ["QK.output_ready"],
//         "PV": ["SM.output_ready", "V.output_ready"],
//         "O": ["PV.output_ready"]
//       }
//     }
builtin.module {
  tile.program @prog_X(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Q(
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
  tile.program @prog_K(
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
  tile.program @prog_V(
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
  tile.program @prog_QK(
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
  tile.program @prog_SM(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "softmax" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_PV(
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
  tile.program @prog_O(
    %task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_X = nest.alloc slot = "input_X" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_X = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_X = nest.alloc slot = "X" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_Q = nest.alloc slot = "W_Q" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_W_Q = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_Q = nest.alloc slot = "Q" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_Q = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_K = nest.alloc slot = "W_K" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_W_K = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_K = nest.alloc slot = "K" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_K = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_W_V = nest.alloc slot = "W_V" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_W_V = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_V = nest.alloc slot = "V" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_V = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_QK = nest.alloc slot = "QK" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_QK = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_SM = nest.alloc slot = "SM" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_SM = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_PV = nest.alloc slot = "PV" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_PV = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %b_O = nest.alloc slot = "O" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<4x64x64xbf16>
    %h_O = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %pref_input_X = nest.dma.prefetch.async %h_input_X into %b_input_X : !nest.event<"pref_input_X">
    %pref_W_Q = nest.dma.prefetch.async %h_W_Q into %b_W_Q : !nest.event<"pref_W_Q">
    %pref_W_K = nest.dma.prefetch.async %h_W_K into %b_W_K : !nest.event<"pref_W_K">
    %pref_W_V = nest.dma.prefetch.async %h_W_V into %b_W_V : !nest.event<"pref_W_V">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_X, %read_X, %ready_X = nest.dispatch.tasks.async @prog_X context = 0 tasks(%tasks)
      globals() bindings(%b_input_X, %b_X) ins(%b_input_X) outs(%b_X)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_input_X)
      : (!nest.event<"grid_X">, !nest.event<"read_X">, !nest.event<"ready_X">)
    %grid_Q, %read_Q, %ready_Q = nest.dispatch.tasks.async @prog_Q context = 0 tasks(%tasks)
      globals() bindings(%b_X, %b_W_Q, %b_Q) ins(%b_X, %b_W_Q) outs(%b_Q)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_X, %pref_W_Q)
      : (!nest.event<"grid_Q">, !nest.event<"read_Q">, !nest.event<"ready_Q">)
    %grid_K, %read_K, %ready_K = nest.dispatch.tasks.async @prog_K context = 1 tasks(%tasks)
      globals() bindings(%b_X, %b_W_K, %b_K) ins(%b_X, %b_W_K) outs(%b_K)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_X, %pref_W_K)
      : (!nest.event<"grid_K">, !nest.event<"read_K">, !nest.event<"ready_K">)
    %grid_V, %read_V, %ready_V = nest.dispatch.tasks.async @prog_V context = 2 tasks(%tasks)
      globals() bindings(%b_X, %b_W_V, %b_V) ins(%b_X, %b_W_V) outs(%b_V)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_X, %pref_W_V)
      : (!nest.event<"grid_V">, !nest.event<"read_V">, !nest.event<"ready_V">)
    %grid_QK, %read_QK, %ready_QK = nest.dispatch.tasks.async @prog_QK context = 0 tasks(%tasks)
      globals() bindings(%b_Q, %b_K, %b_QK) ins(%b_Q, %b_K) outs(%b_QK)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_Q, %ready_K)
      : (!nest.event<"grid_QK">, !nest.event<"read_QK">, !nest.event<"ready_QK">)
    %grid_SM, %read_SM, %ready_SM = nest.dispatch.tasks.async @prog_SM context = 1 tasks(%tasks)
      globals() bindings(%b_QK, %b_SM) ins(%b_QK) outs(%b_SM)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_QK)
      : (!nest.event<"grid_SM">, !nest.event<"read_SM">, !nest.event<"ready_SM">)
    %grid_PV, %read_PV, %ready_PV = nest.dispatch.tasks.async @prog_PV context = 0 tasks(%tasks)
      globals() bindings(%b_SM, %b_V, %b_PV) ins(%b_SM, %b_V) outs(%b_PV)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_SM, %ready_V)
      : (!nest.event<"grid_PV">, !nest.event<"read_PV">, !nest.event<"ready_PV">)
    %grid_O, %read_O, %ready_O = nest.dispatch.tasks.async @prog_O context = 0 tasks(%tasks)
      globals() bindings(%b_PV, %b_O) ins(%b_PV) outs(%b_O)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ready_PV)
      : (!nest.event<"grid_O">, !nest.event<"read_O">, !nest.event<"ready_O">)
    nest.release %b_input_X depends_on(%read_X, %pref_input_X)
    nest.release %b_W_Q depends_on(%read_Q, %pref_W_Q)
    nest.release %b_W_K depends_on(%read_K, %pref_W_K)
    nest.release %b_W_V depends_on(%read_V, %pref_W_V)
    %store_X_0 = nest.dma.store.async %b_X into %h_X depends_on(%ready_X) : !nest.event<"store_X_0">
    nest.release %b_X depends_on(%read_Q, %read_K, %read_V, %store_X_0)
    %store_Q_0 = nest.dma.store.async %b_Q into %h_Q depends_on(%ready_Q) : !nest.event<"store_Q_0">
    nest.release %b_Q depends_on(%read_QK, %store_Q_0)
    %store_K_0 = nest.dma.store.async %b_K into %h_K depends_on(%ready_K) : !nest.event<"store_K_0">
    nest.release %b_K depends_on(%read_QK, %store_K_0)
    %store_V_0 = nest.dma.store.async %b_V into %h_V depends_on(%ready_V) : !nest.event<"store_V_0">
    nest.release %b_V depends_on(%read_PV, %store_V_0)
    %store_QK_0 = nest.dma.store.async %b_QK into %h_QK depends_on(%ready_QK)
      : !nest.event<"store_QK_0">
    nest.release %b_QK depends_on(%read_SM, %store_QK_0)
    %store_SM_0 = nest.dma.store.async %b_SM into %h_SM depends_on(%ready_SM)
      : !nest.event<"store_SM_0">
    nest.release %b_SM depends_on(%read_PV, %store_SM_0)
    %store_PV_0 = nest.dma.store.async %b_PV into %h_PV depends_on(%ready_PV)
      : !nest.event<"store_PV_0">
    nest.release %b_PV depends_on(%read_O, %store_PV_0)
    %store_O_0 = nest.dma.store.async %b_O into %h_O depends_on(%ready_O) : !nest.event<"store_O_0">
    nest.release %b_O depends_on(%store_O_0)
    nest.await %grid_X, %grid_Q, %grid_K, %grid_V, %grid_QK, %grid_SM, %grid_PV, %grid_O, %store_X_0
      , %store_Q_0, %store_K_0, %store_V_0, %store_QK_0, %store_SM_0, %store_PV_0, %store_O_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
