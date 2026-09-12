// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s10", "name": "s10_single", "nodes": ["E0", "E1", "E2", "D2", "D1", "D0"], "edges": {"D2": ["E2"], "D1": ["D2", "E1"], "D0": ["D1", "E0"]}, "context_partition": [["E0", "E1", "E2", "D2", "D1", "D0"]], "resource_config": {"uce": 4, "device": 1, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_E0": {"index": 0, "offset_elements": 0, "bytes": 32768}, "input_E1": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "input_E2": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "W_E2": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "E0": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "E1": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "E2": {"index": 6, "offset_elements": 786432, "bytes": 32768}, "D2": {"index": 7, "offset_elements": 917504, "bytes": 32768}, "D1": {"index": 8, "offset_elements": 1048576, "bytes": 32768}, "D0": {"index": 9, "offset_elements": 1179648, "bytes": 32768}}, "node_programs": {"E0": {"program": "prog_E0", "pin": 0, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "E1": {"program": "prog_E1", "pin": 1, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "E2": {"program": "prog_E2", "pin": 2, "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "D2": {"program": "prog_D2", "pin": 2, "engine": "evu", "op": "add", "repeat": 1, "loads": 1}, "D1": {"program": "prog_D1", "pin": 1, "engine": "evu", "op": "add", "repeat": 1, "loads": 1}, "D0": {"program": "prog_D0", "pin": 0, "engine": "evu", "op": "add", "repeat": 1, "loads": 1}}, "consumer_ready": {"E0": ["input_E0.prefetch"], "E1": ["input_E1.prefetch"], "E2": ["input_E2.prefetch", "W_E2.prefetch"], "D2": ["E2.output_ready"], "D1": ["D2.output_ready", "E1.output_ready"], "D0": ["D1.output_ready", "E0.output_ready"]}}
builtin.module {
  tile.program @prog_E0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_E1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %load_0_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0_0">
    tile.await %load_0_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "relu" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %l0 into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_E2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    tile.free %l0
    tile.free %l1
    %stored = tile.store.async %acc into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_D2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_D1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  tile.program @prog_D0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %v1 = tile.subview %i1 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
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
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_E0 = nest.alloc slot = "input_E0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_E0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_E0 = nest.alloc slot = "E0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_E0 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_input_E1 = nest.alloc slot = "input_E1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_E1 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_E1 = nest.alloc slot = "E1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_E1 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_input_E2 = nest.alloc slot = "input_E2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_E2 = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_E2 = nest.alloc slot = "W_E2" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_E2 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_E2 = nest.alloc slot = "E2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_E2 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_D2 = nest.alloc slot = "D2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_D2 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_D1 = nest.alloc slot = "D1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_D1 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_D0 = nest.alloc slot = "D0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_D0 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_E0 = nest.dma.prefetch.async %h_input_E0 into %b_input_E0 : !nest.event<"pref_input_E0">
    %pref_input_E1 = nest.dma.prefetch.async %h_input_E1 into %b_input_E1 : !nest.event<"pref_input_E1">
    %pref_input_E2 = nest.dma.prefetch.async %h_input_E2 into %b_input_E2 : !nest.event<"pref_input_E2">
    %pref_W_E2 = nest.dma.prefetch.async %h_W_E2 into %b_W_E2 : !nest.event<"pref_W_E2">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_E0, %read_E0, %ready_E0 = nest.dispatch.tasks.async @prog_E0 context = 0 tasks(%tasks) globals() bindings(%b_input_E0, %b_E0) ins(%b_input_E0) outs(%b_E0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_E0) : (!nest.event<"grid_E0">, !nest.event<"read_E0">, !nest.event<"ready_E0">)
    %grid_E1, %read_E1, %ready_E1 = nest.dispatch.tasks.async @prog_E1 context = 1 tasks(%tasks) globals() bindings(%b_input_E1, %b_E1) ins(%b_input_E1) outs(%b_E1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_E1) : (!nest.event<"grid_E1">, !nest.event<"read_E1">, !nest.event<"ready_E1">)
    %grid_E2, %read_E2, %ready_E2 = nest.dispatch.tasks.async @prog_E2 context = 2 tasks(%tasks) globals() bindings(%b_input_E2, %b_W_E2, %b_E2) ins(%b_input_E2, %b_W_E2) outs(%b_E2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_E2, %pref_W_E2) : (!nest.event<"grid_E2">, !nest.event<"read_E2">, !nest.event<"ready_E2">)
    %grid_D2, %read_D2, %ready_D2 = nest.dispatch.tasks.async @prog_D2 context = 2 tasks(%tasks) globals() bindings(%b_E2, %b_D2) ins(%b_E2) outs(%b_D2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_E2) : (!nest.event<"grid_D2">, !nest.event<"read_D2">, !nest.event<"ready_D2">)
    %grid_D1, %read_D1, %ready_D1 = nest.dispatch.tasks.async @prog_D1 context = 1 tasks(%tasks) globals() bindings(%b_D2, %b_E1, %b_D1) ins(%b_D2, %b_E1) outs(%b_D1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_D2, %ready_E1) : (!nest.event<"grid_D1">, !nest.event<"read_D1">, !nest.event<"ready_D1">)
    %grid_D0, %read_D0, %ready_D0 = nest.dispatch.tasks.async @prog_D0 context = 0 tasks(%tasks) globals() bindings(%b_D1, %b_E0, %b_D0) ins(%b_D1, %b_E0) outs(%b_D0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_D1, %ready_E0) : (!nest.event<"grid_D0">, !nest.event<"read_D0">, !nest.event<"ready_D0">)
    nest.release %b_input_E0 depends_on(%read_E0, %pref_input_E0)
    nest.release %b_input_E1 depends_on(%read_E1, %pref_input_E1)
    nest.release %b_input_E2 depends_on(%read_E2, %pref_input_E2)
    nest.release %b_W_E2 depends_on(%read_E2, %pref_W_E2)
    %store_E0_0 = nest.dma.store.async %b_E0 into %h_E0 depends_on(%ready_E0) : !nest.event<"store_E0_0">
    nest.release %b_E0 depends_on(%read_D0, %store_E0_0)
    %store_E1_0 = nest.dma.store.async %b_E1 into %h_E1 depends_on(%ready_E1) : !nest.event<"store_E1_0">
    nest.release %b_E1 depends_on(%read_D1, %store_E1_0)
    %store_E2_0 = nest.dma.store.async %b_E2 into %h_E2 depends_on(%ready_E2) : !nest.event<"store_E2_0">
    nest.release %b_E2 depends_on(%read_D2, %store_E2_0)
    %store_D2_0 = nest.dma.store.async %b_D2 into %h_D2 depends_on(%ready_D2) : !nest.event<"store_D2_0">
    nest.release %b_D2 depends_on(%read_D1, %store_D2_0)
    %store_D1_0 = nest.dma.store.async %b_D1 into %h_D1 depends_on(%ready_D1) : !nest.event<"store_D1_0">
    nest.release %b_D1 depends_on(%read_D0, %store_D1_0)
    %store_D0_0 = nest.dma.store.async %b_D0 into %h_D0 depends_on(%ready_D0) : !nest.event<"store_D0_0">
    nest.release %b_D0 depends_on(%store_D0_0)
    nest.await %grid_E0, %grid_E1, %grid_E2, %grid_D2, %grid_D1, %grid_D0, %store_E0_0, %store_E1_0, %store_E2_0, %store_D2_0, %store_D1_0, %store_D0_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
