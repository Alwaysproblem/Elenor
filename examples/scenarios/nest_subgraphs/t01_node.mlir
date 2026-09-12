// 可复现 NEST 时间子图；头部 case JSON 仅供人工阅读，不是可执行 schema。
// case: {
//       "id": "t01",
//       "name": "t01_node",
//       "nodes": [
//         "Load_A0",
//         "Load_B0",
//         "Compute0",
//         "Store0",
//         "Load_A1",
//         "Load_B1",
//         "Compute1",
//         "Store1",
//         "Load_A2",
//         "Load_B2",
//         "Compute2",
//         "Store2",
//         "Load_A3",
//         "Load_B3",
//         "Compute3",
//         "Store3"
//       ],
//       "edges": {
//         "Compute0": ["Load_A0", "Load_B0"],
//         "Store0": ["Compute0"],
//         "Compute1": ["Load_A1", "Load_B1"],
//         "Store1": ["Compute1"],
//         "Compute2": ["Load_A2", "Load_B2"],
//         "Store2": ["Compute2"],
//         "Compute3": ["Load_A3", "Load_B3"],
//         "Store3": ["Compute3"]
//       },
//       "context_partition": [
//         ["Load_A0", "Load_B0", "Compute0", "Store0"],
//         ["Load_A1", "Load_B1", "Compute1", "Store1"],
//         ["Load_A2", "Load_B2", "Compute2", "Store2"],
//         ["Load_A3", "Load_B3", "Compute3", "Store3"]
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
//         "Scheduling Quality": "双 buffer 流水：检查不同 chunk 的真实 load/BOA/store 区间重叠；不以 issue attempt 代替实际 service"
//       },
//       "forbidden_dependencies": "只允许表列真数据边；UCE 共享 pin 争用不是数据依赖；跨 Context 仅使用当前 IR 可见的 context_done/HBM 可见性",
//       "tensors": {
//         "A0": {
//           "formal": "arena",
//           "index": 0,
//           "offset_elements": 0,
//           "interval_bytes": [0, 32768],
//           "bytes": 32768
//         },
//         "B0": {
//           "formal": "arena",
//           "index": 1,
//           "offset_elements": 131072,
//           "interval_bytes": [262144, 294912],
//           "bytes": 32768
//         },
//         "A1": {
//           "formal": "arena",
//           "index": 2,
//           "offset_elements": 262144,
//           "interval_bytes": [524288, 557056],
//           "bytes": 32768
//         },
//         "B1": {
//           "formal": "arena",
//           "index": 3,
//           "offset_elements": 393216,
//           "interval_bytes": [786432, 819200],
//           "bytes": 32768
//         },
//         "A2": {
//           "formal": "arena",
//           "index": 4,
//           "offset_elements": 524288,
//           "interval_bytes": [1048576, 1081344],
//           "bytes": 32768
//         },
//         "B2": {
//           "formal": "arena",
//           "index": 5,
//           "offset_elements": 655360,
//           "interval_bytes": [1310720, 1343488],
//           "bytes": 32768
//         },
//         "A3": {
//           "formal": "arena",
//           "index": 6,
//           "offset_elements": 786432,
//           "interval_bytes": [1572864, 1605632],
//           "bytes": 32768
//         },
//         "B3": {
//           "formal": "arena",
//           "index": 7,
//           "offset_elements": 917504,
//           "interval_bytes": [1835008, 1867776],
//           "bytes": 32768
//         },
//         "O0": {
//           "formal": "arena",
//           "index": 8,
//           "offset_elements": 1048576,
//           "interval_bytes": [2097152, 2129920],
//           "bytes": 32768
//         },
//         "O1": {
//           "formal": "arena",
//           "index": 9,
//           "offset_elements": 1179648,
//           "interval_bytes": [2359296, 2392064],
//           "bytes": 32768
//         },
//         "O2": {
//           "formal": "arena",
//           "index": 10,
//           "offset_elements": 1310720,
//           "interval_bytes": [2621440, 2654208],
//           "bytes": 32768
//         },
//         "O3": {
//           "formal": "arena",
//           "index": 11,
//           "offset_elements": 1441792,
//           "interval_bytes": [2883584, 2916352],
//           "bytes": 32768
//         }
//       },
//       "node_programs": {
//         "Load_A0": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Load_B0": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Compute0": {
//           "program": "prog_Compute0",
//           "pin": "跟随 device slot",
//           "engine": "BOA:matmul",
//           "repeat": 10
//         },
//         "Store0": {
//           "program": "nest.dma.store.async",
//           "pin": null,
//           "engine": "DMA store",
//           "repeat": 1
//         },
//         "Load_A1": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Load_B1": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Compute1": {
//           "program": "prog_Compute1",
//           "pin": "跟随 device slot",
//           "engine": "BOA:matmul",
//           "repeat": 10
//         },
//         "Store1": {
//           "program": "nest.dma.store.async",
//           "pin": null,
//           "engine": "DMA store",
//           "repeat": 1
//         },
//         "Load_A2": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Load_B2": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Compute2": {
//           "program": "prog_Compute2",
//           "pin": "跟随 device slot",
//           "engine": "BOA:matmul",
//           "repeat": 10
//         },
//         "Store2": {
//           "program": "nest.dma.store.async",
//           "pin": null,
//           "engine": "DMA store",
//           "repeat": 1
//         },
//         "Load_A3": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Load_B3": {
//           "program": "nest.dma.prefetch.async",
//           "pin": null,
//           "engine": "DMA load",
//           "repeat": 1
//         },
//         "Compute3": {
//           "program": "prog_Compute3",
//           "pin": "跟随 device slot",
//           "engine": "BOA:matmul",
//           "repeat": 10
//         },
//         "Store3": {
//           "program": "nest.dma.store.async",
//           "pin": null,
//           "engine": "DMA store",
//           "repeat": 1
//         }
//       },
//       "consumer_ready_granularity": {
//         "Load_A0->Compute0": "该 chunk A prefetch 完成",
//         "Load_B0->Compute0": "该 chunk B prefetch 完成",
//         "Compute0->Store0": "output_ready(all_tasks)；复用集合的最终 Store depends_on 列全历次 producer output_ready",
//         "Load_A1->Compute1": "该 chunk A prefetch 完成",
//         "Load_B1->Compute1": "该 chunk B prefetch 完成",
//         "Compute1->Store1": "output_ready(all_tasks)；复用集合的最终 Store depends_on 列全历次 producer output_ready",
//         "Load_A2->Compute2": "该 chunk A prefetch 完成",
//         "Load_B2->Compute2": "该 chunk B prefetch 完成",
//         "Compute2->Store2": "output_ready(all_tasks)；复用集合的最终 Store depends_on 列全历次 producer output_ready",
//         "Load_A3->Compute3": "该 chunk A prefetch 完成",
//         "Load_B3->Compute3": "该 chunk B prefetch 完成",
//         "Compute3->Store3": "output_ready(all_tasks)；复用集合的最终 Store depends_on 列全历次 producer output_ready"
//       },
//       "notes": {
//         "allocations": "node 每 chunk 3 个；stage 偶/奇 Context 各 3 个",
//         "reuse": "chunk 0/2 复用 set0，chunk 1/3 复用 set1；输入覆盖前 await 上次 input_released，输出覆盖前 await 上次 global Store",
//         "load_store_nodes": "Load/Store 是真实 DMA 活动，不伪造 tile Copy/数值 compute"
//       }
//     }
builtin.module {
  tile.program @prog_Compute0(
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
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_9">
    tile.await %compute_9
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Compute1(
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
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_9">
    tile.await %compute_9
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Compute2(
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
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_9">
    tile.await %compute_9
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_Compute3(
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
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    %load_1 = tile.load.async %v1 into %l1 : !tile.event<"load_1">
    tile.await %load_0, %load_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288
      : !tile.event<"compute_0">
    tile.await %compute_0
    %compute_1 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_1">
    tile.await %compute_1
    %compute_2 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_2">
    tile.await %compute_2
    %compute_3 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_3">
    tile.await %compute_3
    %compute_4 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_4">
    tile.await %compute_4
    %compute_5 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_5">
    tile.await %compute_5
    %compute_6 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_6">
    tile.await %compute_6
    %compute_7 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_7">
    tile.await %compute_7
    %compute_8 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_8">
    tile.await %compute_8
    %compute_9 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 accumulate
      : !tile.event<"compute_9">
    tile.await %compute_9
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_chunk0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A0 = nest.alloc slot = "A_set0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B0 = nest.alloc slot = "B_set0" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_O0 = nest.alloc slot = "O_set0" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_O0 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %pref_A0 = nest.dma.prefetch.async %h_A0 into %b_A0 : !nest.event<"pref_A0">
    %pref_B0 = nest.dma.prefetch.async %h_B0 into %b_B0 : !nest.event<"pref_B0">
    %grid_Compute0, %read_Compute0, %ready_Compute0 = nest.dispatch.tasks.async @prog_Compute0
      context = 0 tasks(%tasks) globals() bindings(%b_A0, %b_B0, %b_O0) ins(%b_A0, %b_B0)
      outs(%b_O0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A0, %pref_B0)
      : (!nest.event<"grid_Compute0">, !nest.event<"read_Compute0">, !nest.event<"ready_Compute0">)
    %store_chunk0 = nest.dma.store.async %b_O0 into %h_O0 depends_on(%ready_Compute0)
      : !nest.event<"store_chunk0">
    nest.release %b_A0 depends_on(%read_Compute0, %pref_A0)
    nest.release %b_B0 depends_on(%read_Compute0, %pref_B0)
    nest.release %b_O0 depends_on(%store_chunk0)
    nest.await %grid_Compute0, %store_chunk0
    nest.return
  }
  nest.context @ctx_chunk1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A1 = nest.alloc slot = "A_set1" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B1 = nest.alloc slot = "B_set1" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_O1 = nest.alloc slot = "O_set1" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A1 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B1 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_O1 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %pref_A1 = nest.dma.prefetch.async %h_A1 into %b_A1 : !nest.event<"pref_A1">
    %pref_B1 = nest.dma.prefetch.async %h_B1 into %b_B1 : !nest.event<"pref_B1">
    %grid_Compute1, %read_Compute1, %ready_Compute1 = nest.dispatch.tasks.async @prog_Compute1
      context = 1 tasks(%tasks) globals() bindings(%b_A1, %b_B1, %b_O1) ins(%b_A1, %b_B1)
      outs(%b_O1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A1, %pref_B1)
      : (!nest.event<"grid_Compute1">, !nest.event<"read_Compute1">, !nest.event<"ready_Compute1">)
    %store_chunk1 = nest.dma.store.async %b_O1 into %h_O1 depends_on(%ready_Compute1)
      : !nest.event<"store_chunk1">
    nest.release %b_A1 depends_on(%read_Compute1, %pref_A1)
    nest.release %b_B1 depends_on(%read_Compute1, %pref_B1)
    nest.release %b_O1 depends_on(%store_chunk1)
    nest.await %grid_Compute1, %store_chunk1
    nest.return
  }
  nest.context @ctx_chunk2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A2 = nest.alloc slot = "A_set2" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B2 = nest.alloc slot = "B_set2" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_O2 = nest.alloc slot = "O_set2" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A2 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B2 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_O2 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %pref_A2 = nest.dma.prefetch.async %h_A2 into %b_A2 : !nest.event<"pref_A2">
    %pref_B2 = nest.dma.prefetch.async %h_B2 into %b_B2 : !nest.event<"pref_B2">
    %grid_Compute2, %read_Compute2, %ready_Compute2 = nest.dispatch.tasks.async @prog_Compute2
      context = 0 tasks(%tasks) globals() bindings(%b_A2, %b_B2, %b_O2) ins(%b_A2, %b_B2)
      outs(%b_O2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A2, %pref_B2)
      : (!nest.event<"grid_Compute2">, !nest.event<"read_Compute2">, !nest.event<"ready_Compute2">)
    %store_chunk2 = nest.dma.store.async %b_O2 into %h_O2 depends_on(%ready_Compute2)
      : !nest.event<"store_chunk2">
    nest.release %b_A2 depends_on(%read_Compute2, %pref_A2)
    nest.release %b_B2 depends_on(%read_Compute2, %pref_B2)
    nest.release %b_O2 depends_on(%store_chunk2)
    nest.await %grid_Compute2, %store_chunk2
    nest.return
  }
  nest.context @ctx_chunk3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_A3 = nest.alloc slot = "A_set3" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_B3 = nest.alloc slot = "B_set3" role = "in" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_O3 = nest.alloc slot = "O_set3" role = "inout" shape = [4, 64, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A3 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_B3 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %h_O3 = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1]
      : !nest.global_view<16384xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %pref_A3 = nest.dma.prefetch.async %h_A3 into %b_A3 : !nest.event<"pref_A3">
    %pref_B3 = nest.dma.prefetch.async %h_B3 into %b_B3 : !nest.event<"pref_B3">
    %grid_Compute3, %read_Compute3, %ready_Compute3 = nest.dispatch.tasks.async @prog_Compute3
      context = 1 tasks(%tasks) globals() bindings(%b_A3, %b_B3, %b_O3) ins(%b_A3, %b_B3)
      outs(%b_O3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pref_A3, %pref_B3)
      : (!nest.event<"grid_Compute3">, !nest.event<"read_Compute3">, !nest.event<"ready_Compute3">)
    %store_chunk3 = nest.dma.store.async %b_O3 into %h_O3 depends_on(%ready_Compute3)
      : !nest.event<"store_chunk3">
    nest.release %b_A3 depends_on(%read_Compute3, %pref_A3)
    nest.release %b_B3 depends_on(%read_Compute3, %pref_B3)
    nest.release %b_O3 depends_on(%store_chunk3)
    nest.await %grid_Compute3, %store_chunk3
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_chunk0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_chunk1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_chunk2(%arena) : !nexus.event<"done_2">
    %done_3 = nexus.submit_context.async @ctx_chunk3(%arena) : !nexus.event<"done_3">
    nexus.await %done_0, %done_1, %done_2, %done_3
    nexus.return
  }
}
