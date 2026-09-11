// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s11", "name": "s11_node_p3", "nodes": ["P0", "P1", "P2", "R0", "R", "N0", "N1", "N2", "N3"], "edges": {"R0": ["P0", "P1"], "R": ["R0", "P2"], "N0": ["R"], "N1": ["R"], "N2": ["R"], "N3": ["R"]}, "context_partition": [["P0"], ["P1"], ["P2"], ["R0"], ["R"], ["N0"], ["N1"], ["N2"], ["N3"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_P0": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_P0": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "input_P1": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "W_P1": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "input_P2": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "W_P2": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "P0": {"index": 8, "offset_elements": 1048576, "bytes": 32768}, "P1": {"index": 9, "offset_elements": 1179648, "bytes": 32768}, "P2": {"index": 10, "offset_elements": 1310720, "bytes": 32768}, "R0": {"index": 12, "offset_elements": 1572864, "bytes": 32768}, "R": {"index": 14, "offset_elements": 1835008, "bytes": 32768}, "N0": {"index": 15, "offset_elements": 1966080, "bytes": 32768}, "N1": {"index": 16, "offset_elements": 2097152, "bytes": 32768}, "N2": {"index": 17, "offset_elements": 2228224, "bytes": 32768}, "N3": {"index": 18, "offset_elements": 2359296, "bytes": 32768}}, "node_programs": {"P0": {"program": "prog_P0", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "P1": {"program": "prog_P1", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "P2": {"program": "prog_P2", "pin": "随 device slot", "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "R0": {"program": "prog_R0", "pin": "随 device slot", "engine": "evu", "op": "reduce", "repeat": 1, "loads": 1}, "R": {"program": "prog_R", "pin": "随 device slot", "engine": "evu", "op": "reduce", "repeat": 1, "loads": 1}, "N0": {"program": "prog_N0", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N1": {"program": "prog_N1", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N2": {"program": "prog_N2", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "N3": {"program": "prog_N3", "pin": "随 device slot", "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}}, "consumer_ready": {"P0": ["input_P0.prefetch", "W_P0.prefetch"], "P1": ["input_P1.prefetch", "W_P1.prefetch"], "P2": ["input_P2.prefetch", "W_P2.prefetch"], "R0": ["P0.context_done 后从同一 HBM 区间 prefetch", "P1.context_done 后从同一 HBM 区间 prefetch"], "R": ["R0.context_done 后从同一 HBM 区间 prefetch", "P2.context_done 后从同一 HBM 区间 prefetch"], "N0": ["R.context_done 后从同一 HBM 区间 prefetch"], "N1": ["R.context_done 后从同一 HBM 区间 prefetch"], "N2": ["R.context_done 后从同一 HBM 区间 prefetch"], "N3": ["R.context_done 后从同一 HBM 区间 prefetch"]}, "说明": "3 个 participant；相邻项逐层 reduce，奇数尾项直达下一层，最终真实 Copy fanout 到 N0..N3。"}
builtin.module {
  tile.program @prog_P0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_P1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_P2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.boa.async "matmul" m = 64 n = 64 k = 64 ops = 524288 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_R0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "reduce" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_R (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    %load_0_1 = tile.load.async %v1 into %l1 : !tile.event<"load_0_1">
    tile.await %load_0_0, %load_0_1
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "reduce" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_N3 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P0 = nest.alloc slot = "input_P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P0 = nest.alloc slot = "W_P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P0 = nest.alloc slot = "P0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P0 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P0 = nest.dma.prefetch.async %h_input_P0 into %b_input_P0 : !nest.event<"pref_input_P0">
    %pref_W_P0 = nest.dma.prefetch.async %h_W_P0 into %b_W_P0 : !nest.event<"pref_W_P0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P0, %read_P0, %ready_P0 = nest.dispatch.tasks.async @prog_P0 tasks(%tasks) globals() bindings(%b_input_P0, %b_W_P0, %b_P0) ins(%b_input_P0, %b_W_P0) outs(%b_P0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P0, %pref_W_P0) : (!nest.event<"grid_P0">, !nest.event<"read_P0">, !nest.event<"ready_P0">)
    nest.release %b_input_P0 depends_on(%read_P0, %pref_input_P0)
    nest.release %b_W_P0 depends_on(%read_P0, %pref_W_P0)
    %store_P0_0 = nest.dma.store.async %b_P0 into %h_P0 depends_on(%ready_P0) : !nest.event<"store_P0_0">
    nest.release %b_P0 depends_on(%store_P0_0)
    nest.await %grid_P0, %store_P0_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P1 = nest.alloc slot = "input_P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P1 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P1 = nest.alloc slot = "W_P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P1 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P1 = nest.alloc slot = "P1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P1 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P1 = nest.dma.prefetch.async %h_input_P1 into %b_input_P1 : !nest.event<"pref_input_P1">
    %pref_W_P1 = nest.dma.prefetch.async %h_W_P1 into %b_W_P1 : !nest.event<"pref_W_P1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P1, %read_P1, %ready_P1 = nest.dispatch.tasks.async @prog_P1 tasks(%tasks) globals() bindings(%b_input_P1, %b_W_P1, %b_P1) ins(%b_input_P1, %b_W_P1) outs(%b_P1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P1, %pref_W_P1) : (!nest.event<"grid_P1">, !nest.event<"read_P1">, !nest.event<"ready_P1">)
    nest.release %b_input_P1 depends_on(%read_P1, %pref_input_P1)
    nest.release %b_W_P1 depends_on(%read_P1, %pref_W_P1)
    %store_P1_0 = nest.dma.store.async %b_P1 into %h_P1 depends_on(%ready_P1) : !nest.event<"store_P1_0">
    nest.release %b_P1 depends_on(%store_P1_0)
    nest.await %grid_P1, %store_P1_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_P2 = nest.alloc slot = "input_P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_P2 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_P2 = nest.alloc slot = "W_P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_P2 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P2 = nest.alloc slot = "P2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P2 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_P2 = nest.dma.prefetch.async %h_input_P2 into %b_input_P2 : !nest.event<"pref_input_P2">
    %pref_W_P2 = nest.dma.prefetch.async %h_W_P2 into %b_W_P2 : !nest.event<"pref_W_P2">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_P2, %read_P2, %ready_P2 = nest.dispatch.tasks.async @prog_P2 tasks(%tasks) globals() bindings(%b_input_P2, %b_W_P2, %b_P2) ins(%b_input_P2, %b_W_P2) outs(%b_P2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_P2, %pref_W_P2) : (!nest.event<"grid_P2">, !nest.event<"read_P2">, !nest.event<"ready_P2">)
    nest.release %b_input_P2 depends_on(%read_P2, %pref_input_P2)
    nest.release %b_W_P2 depends_on(%read_P2, %pref_W_P2)
    %store_P2_0 = nest.dma.store.async %b_P2 into %h_P2 depends_on(%ready_P2) : !nest.event<"store_P2_0">
    nest.release %b_P2 depends_on(%store_P2_0)
    nest.await %grid_P2, %store_P2_0
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_P0 = nest.alloc slot = "P0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P0 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P1 = nest.alloc slot = "P1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P1 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R0 = nest.alloc slot = "R0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_P0 = nest.dma.prefetch.async %h_P0 into %b_P0 : !nest.event<"pref_P0">
    %pref_P1 = nest.dma.prefetch.async %h_P1 into %b_P1 : !nest.event<"pref_P1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_R0, %read_R0, %ready_R0 = nest.dispatch.tasks.async @prog_R0 tasks(%tasks) globals() bindings(%b_P0, %b_P1, %b_R0) ins(%b_P0, %b_P1) outs(%b_R0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_P0, %pref_P1) : (!nest.event<"grid_R0">, !nest.event<"read_R0">, !nest.event<"ready_R0">)
    nest.release %b_P0 depends_on(%read_R0, %pref_P0)
    nest.release %b_P1 depends_on(%read_R0, %pref_P1)
    %store_R0_0 = nest.dma.store.async %b_R0 into %h_R0 depends_on(%ready_R0) : !nest.event<"store_R0_0">
    nest.release %b_R0 depends_on(%store_R0_0)
    nest.await %grid_R0, %store_R0_0
    nest.return
  }
  nest.context @ctx_4 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R0 = nest.alloc slot = "R0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R0 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_P2 = nest.alloc slot = "P2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_P2 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_R = nest.alloc slot = "R" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R0 = nest.dma.prefetch.async %h_R0 into %b_R0 : !nest.event<"pref_R0">
    %pref_P2 = nest.dma.prefetch.async %h_P2 into %b_P2 : !nest.event<"pref_P2">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_R, %read_R, %ready_R = nest.dispatch.tasks.async @prog_R tasks(%tasks) globals() bindings(%b_R0, %b_P2, %b_R) ins(%b_R0, %b_P2) outs(%b_R) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R0, %pref_P2) : (!nest.event<"grid_R">, !nest.event<"read_R">, !nest.event<"ready_R">)
    nest.release %b_R0 depends_on(%read_R, %pref_R0)
    nest.release %b_P2 depends_on(%read_R, %pref_P2)
    %store_R_0 = nest.dma.store.async %b_R into %h_R depends_on(%ready_R) : !nest.event<"store_R_0">
    nest.release %b_R depends_on(%store_R_0)
    nest.await %grid_R, %store_R_0
    nest.return
  }
  nest.context @ctx_5 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N0 = nest.alloc slot = "N0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N0 = nest.subview %arena offsets = [1966080] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N0, %read_N0, %ready_N0 = nest.dispatch.tasks.async @prog_N0 tasks(%tasks) globals() bindings(%b_R, %b_N0) ins(%b_R) outs(%b_N0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N0">, !nest.event<"read_N0">, !nest.event<"ready_N0">)
    nest.release %b_R depends_on(%read_N0, %pref_R)
    %store_N0_0 = nest.dma.store.async %b_N0 into %h_N0 depends_on(%ready_N0) : !nest.event<"store_N0_0">
    nest.release %b_N0 depends_on(%store_N0_0)
    nest.await %grid_N0, %store_N0_0
    nest.return
  }
  nest.context @ctx_6 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N1 = nest.alloc slot = "N1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N1 = nest.subview %arena offsets = [2097152] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N1, %read_N1, %ready_N1 = nest.dispatch.tasks.async @prog_N1 tasks(%tasks) globals() bindings(%b_R, %b_N1) ins(%b_R) outs(%b_N1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N1">, !nest.event<"read_N1">, !nest.event<"ready_N1">)
    nest.release %b_R depends_on(%read_N1, %pref_R)
    %store_N1_0 = nest.dma.store.async %b_N1 into %h_N1 depends_on(%ready_N1) : !nest.event<"store_N1_0">
    nest.release %b_N1 depends_on(%store_N1_0)
    nest.await %grid_N1, %store_N1_0
    nest.return
  }
  nest.context @ctx_7 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N2 = nest.alloc slot = "N2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N2 = nest.subview %arena offsets = [2228224] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N2, %read_N2, %ready_N2 = nest.dispatch.tasks.async @prog_N2 tasks(%tasks) globals() bindings(%b_R, %b_N2) ins(%b_R) outs(%b_N2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N2">, !nest.event<"read_N2">, !nest.event<"ready_N2">)
    nest.release %b_R depends_on(%read_N2, %pref_R)
    %store_N2_0 = nest.dma.store.async %b_N2 into %h_N2 depends_on(%ready_N2) : !nest.event<"store_N2_0">
    nest.release %b_N2 depends_on(%store_N2_0)
    nest.await %grid_N2, %store_N2_0
    nest.return
  }
  nest.context @ctx_8 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_R = nest.alloc slot = "R" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_R = nest.subview %arena offsets = [1835008] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_N3 = nest.alloc slot = "N3" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_N3 = nest.subview %arena offsets = [2359296] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_R = nest.dma.prefetch.async %h_R into %b_R : !nest.event<"pref_R">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_N3, %read_N3, %ready_N3 = nest.dispatch.tasks.async @prog_N3 tasks(%tasks) globals() bindings(%b_R, %b_N3) ins(%b_R) outs(%b_N3) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_R) : (!nest.event<"grid_N3">, !nest.event<"read_N3">, !nest.event<"ready_N3">)
    nest.release %b_R depends_on(%read_N3, %pref_R)
    %store_N3_0 = nest.dma.store.async %b_N3 into %h_N3 depends_on(%ready_N3) : !nest.event<"store_N3_0">
    nest.release %b_N3 depends_on(%store_N3_0)
    nest.await %grid_N3, %store_N3_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_0, %done_1
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_3, %done_2
    %done_4 = nexus.submit_context.async @ctx_4(%arena) : !nexus.event<"done_4">
    nexus.await %done_4
    %done_5 = nexus.submit_context.async @ctx_5(%arena) : !nexus.event<"done_5">
    nexus.await %done_4
    %done_6 = nexus.submit_context.async @ctx_6(%arena) : !nexus.event<"done_6">
    nexus.await %done_4
    %done_7 = nexus.submit_context.async @ctx_7(%arena) : !nexus.event<"done_7">
    nexus.await %done_4
    %done_8 = nexus.submit_context.async @ctx_8(%arena) : !nexus.event<"done_8">
    nexus.await %done_0, %done_1, %done_2, %done_3, %done_4, %done_5, %done_6, %done_7, %done_8
    nexus.return
  }
}
