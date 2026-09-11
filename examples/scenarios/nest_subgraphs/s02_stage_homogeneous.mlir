// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s02", "name": "s02_stage_homogeneous", "nodes": ["A0", "A1", "A2", "B0", "B1", "C0", "C1", "C2"], "edges": {"A1": ["A0"], "A2": ["A1"], "B1": ["B0"], "C1": ["C0"], "C2": ["C1"]}, "context_partition": [["A0", "A1", "A2"], ["B0", "B1"], ["C0", "C1", "C2"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_A0": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_A0": {"index": 1, "offset_elements": 131072, "bytes": 32768, "reserved_only": true}, "W_A1": {"index": 2, "offset_elements": 262144, "bytes": 32768, "reserved_only": true}, "W_A2": {"index": 3, "offset_elements": 393216, "bytes": 32768, "reserved_only": true}, "input_B0": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "input_C0": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "A0": {"index": 6, "offset_elements": 786432, "bytes": 32768}, "A1": {"index": 7, "offset_elements": 917504, "bytes": 32768}, "A2": {"index": 8, "offset_elements": 1048576, "bytes": 32768}, "B0": {"index": 9, "offset_elements": 1179648, "bytes": 32768}, "B1": {"index": 10, "offset_elements": 1310720, "bytes": 32768}, "C0": {"index": 11, "offset_elements": 1441792, "bytes": 32768}, "C1": {"index": 12, "offset_elements": 1572864, "bytes": 32768}, "C2": {"index": 13, "offset_elements": 1703936, "bytes": 32768}}, "node_programs": {"A0": {"program": "prog_A0", "pin": 0, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "A1": {"program": "prog_A1", "pin": 0, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "A2": {"program": "prog_A2", "pin": 0, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "B0": {"program": "prog_B0", "pin": 1, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "B1": {"program": "prog_B1", "pin": 1, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "C0": {"program": "prog_C0", "pin": 2, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "C1": {"program": "prog_C1", "pin": 2, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "C2": {"program": "prog_C2", "pin": 2, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}}, "consumer_ready": {"A0": ["input_A0.prefetch"], "A1": ["A0.output_ready"], "A2": ["A1.output_ready"], "B0": ["input_B0.prefetch"], "B1": ["B0.output_ready"], "C0": ["input_C0.prefetch"], "C1": ["C0.output_ready"], "C2": ["C1.output_ready"]}, "说明": "全链改为相同 EVU 成本；保留基础版本 weight 的 arena 编号为空洞，确保所有节点输出 tensor index 不变。"}
builtin.module {
  tile.program @prog_A0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_A1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_A2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_B0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_B1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_C0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_C1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_C2 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_A0 = nest.alloc slot = "input_A0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_A0 = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_A0 = nest.alloc slot = "A0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A0 = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_A1 = nest.alloc slot = "A1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A1 = nest.subview %arena offsets = [917504] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_A2 = nest.alloc slot = "A2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_A2 = nest.subview %arena offsets = [1048576] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_A0 = nest.dma.prefetch.async %h_input_A0 into %b_input_A0 : !nest.event<"pref_input_A0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_A0, %read_A0, %ready_A0 = nest.dispatch.tasks.async @prog_A0 context = 0 tasks(%tasks) globals() bindings(%b_input_A0, %b_A0) ins(%b_input_A0) outs(%b_A0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_A0) : (!nest.event<"grid_A0">, !nest.event<"read_A0">, !nest.event<"ready_A0">)
    %grid_A1, %read_A1, %ready_A1 = nest.dispatch.tasks.async @prog_A1 context = 0 tasks(%tasks) globals() bindings(%b_A0, %b_A1) ins(%b_A0) outs(%b_A1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A0) : (!nest.event<"grid_A1">, !nest.event<"read_A1">, !nest.event<"ready_A1">)
    %grid_A2, %read_A2, %ready_A2 = nest.dispatch.tasks.async @prog_A2 context = 0 tasks(%tasks) globals() bindings(%b_A1, %b_A2) ins(%b_A1) outs(%b_A2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_A1) : (!nest.event<"grid_A2">, !nest.event<"read_A2">, !nest.event<"ready_A2">)
    nest.release %b_input_A0 depends_on(%read_A0, %pref_input_A0)
    %store_A0_0 = nest.dma.store.async %b_A0 into %h_A0 depends_on(%ready_A0) : !nest.event<"store_A0_0">
    nest.release %b_A0 depends_on(%read_A1, %store_A0_0)
    %store_A1_0 = nest.dma.store.async %b_A1 into %h_A1 depends_on(%ready_A1) : !nest.event<"store_A1_0">
    nest.release %b_A1 depends_on(%read_A2, %store_A1_0)
    %store_A2_0 = nest.dma.store.async %b_A2 into %h_A2 depends_on(%ready_A2) : !nest.event<"store_A2_0">
    nest.release %b_A2 depends_on(%store_A2_0)
    nest.await %grid_A0, %grid_A1, %grid_A2, %store_A0_0, %store_A1_0, %store_A2_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_B0 = nest.alloc slot = "input_B0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_B0 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_B0 = nest.alloc slot = "B0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_B0 = nest.subview %arena offsets = [1179648] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_B1 = nest.alloc slot = "B1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_B1 = nest.subview %arena offsets = [1310720] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_B0 = nest.dma.prefetch.async %h_input_B0 into %b_input_B0 : !nest.event<"pref_input_B0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_B0, %read_B0, %ready_B0 = nest.dispatch.tasks.async @prog_B0 context = 1 tasks(%tasks) globals() bindings(%b_input_B0, %b_B0) ins(%b_input_B0) outs(%b_B0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_B0) : (!nest.event<"grid_B0">, !nest.event<"read_B0">, !nest.event<"ready_B0">)
    %grid_B1, %read_B1, %ready_B1 = nest.dispatch.tasks.async @prog_B1 context = 1 tasks(%tasks) globals() bindings(%b_B0, %b_B1) ins(%b_B0) outs(%b_B1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_B0) : (!nest.event<"grid_B1">, !nest.event<"read_B1">, !nest.event<"ready_B1">)
    nest.release %b_input_B0 depends_on(%read_B0, %pref_input_B0)
    %store_B0_0 = nest.dma.store.async %b_B0 into %h_B0 depends_on(%ready_B0) : !nest.event<"store_B0_0">
    nest.release %b_B0 depends_on(%read_B1, %store_B0_0)
    %store_B1_0 = nest.dma.store.async %b_B1 into %h_B1 depends_on(%ready_B1) : !nest.event<"store_B1_0">
    nest.release %b_B1 depends_on(%store_B1_0)
    nest.await %grid_B0, %grid_B1, %store_B0_0, %store_B1_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input_C0 = nest.alloc slot = "input_C0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_C0 = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_C0 = nest.alloc slot = "C0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_C0 = nest.subview %arena offsets = [1441792] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_C1 = nest.alloc slot = "C1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_C1 = nest.subview %arena offsets = [1572864] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_C2 = nest.alloc slot = "C2" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_C2 = nest.subview %arena offsets = [1703936] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_C0 = nest.dma.prefetch.async %h_input_C0 into %b_input_C0 : !nest.event<"pref_input_C0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_C0, %read_C0, %ready_C0 = nest.dispatch.tasks.async @prog_C0 context = 2 tasks(%tasks) globals() bindings(%b_input_C0, %b_C0) ins(%b_input_C0) outs(%b_C0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_C0) : (!nest.event<"grid_C0">, !nest.event<"read_C0">, !nest.event<"ready_C0">)
    %grid_C1, %read_C1, %ready_C1 = nest.dispatch.tasks.async @prog_C1 context = 2 tasks(%tasks) globals() bindings(%b_C0, %b_C1) ins(%b_C0) outs(%b_C1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_C0) : (!nest.event<"grid_C1">, !nest.event<"read_C1">, !nest.event<"ready_C1">)
    %grid_C2, %read_C2, %ready_C2 = nest.dispatch.tasks.async @prog_C2 context = 2 tasks(%tasks) globals() bindings(%b_C1, %b_C2) ins(%b_C1) outs(%b_C2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_C1) : (!nest.event<"grid_C2">, !nest.event<"read_C2">, !nest.event<"ready_C2">)
    nest.release %b_input_C0 depends_on(%read_C0, %pref_input_C0)
    %store_C0_0 = nest.dma.store.async %b_C0 into %h_C0 depends_on(%ready_C0) : !nest.event<"store_C0_0">
    nest.release %b_C0 depends_on(%read_C1, %store_C0_0)
    %store_C1_0 = nest.dma.store.async %b_C1 into %h_C1 depends_on(%ready_C1) : !nest.event<"store_C1_0">
    nest.release %b_C1 depends_on(%read_C2, %store_C1_0)
    %store_C2_0 = nest.dma.store.async %b_C2 into %h_C2 depends_on(%ready_C2) : !nest.event<"store_C2_0">
    nest.release %b_C2 depends_on(%store_C2_0)
    nest.await %grid_C0, %grid_C1, %grid_C2, %store_C0_0, %store_C1_0, %store_C2_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_0, %done_1, %done_2
    nexus.return
  }
}
