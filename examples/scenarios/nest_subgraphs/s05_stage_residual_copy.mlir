// 可复现 NEST 子图；中文元数据仅供人工阅读，不是可执行 schema。
// case: {"id": "s05", "name": "s05_stage_residual_copy", "nodes": ["X", "F0", "F1", "Copy", "Add"], "edges": {"F0": ["X"], "F1": ["F0"], "Copy": ["X"], "Add": ["Copy", "F1"]}, "context_partition": [["X"], ["F0", "F1"], ["Copy"], ["Add"]], "resource_config": {"uce": 4, "device": 4, "placement": 15, "fidelity": "full_memory"}, "expected": {"时序/生命周期 Correctness": "检查全部真实输入边、完整 Store/release 依赖和 Context 完成", "数值": "未建模；隐式 L1 output timing，仅作合成 compute-cost 扫描", "Liveness": "预期完成", "Scheduling Quality": "以实际 service 判定；不把 issue/slot 占用冒充有效重叠"}, "forbidden_dependencies": "仅允许 edges 与外部输入；跨 Context 使用当前 IR 的保守 context_done 可见性", "tensors": {"input_X": {"index": 0, "offset_elements": 0, "bytes": 32768}, "W_F0": {"index": 1, "offset_elements": 131072, "bytes": 32768}, "X": {"index": 2, "offset_elements": 262144, "bytes": 32768}, "F0": {"index": 3, "offset_elements": 393216, "bytes": 32768}, "F1": {"index": 4, "offset_elements": 524288, "bytes": 32768}, "Copy": {"index": 5, "offset_elements": 655360, "bytes": 32768}, "Add": {"index": 6, "offset_elements": 786432, "bytes": 32768}}, "node_programs": {"X": {"program": "prog_X", "pin": 0, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "F0": {"program": "prog_F0", "pin": 1, "engine": "boa", "op": "matmul", "repeat": 1, "loads": 1}, "F1": {"program": "prog_F1", "pin": 1, "engine": "evu", "op": "relu", "repeat": 1, "loads": 1}, "Copy": {"program": "prog_Copy", "pin": 2, "engine": "copy", "op": "copy", "repeat": 1, "loads": 1}, "Add": {"program": "prog_Add", "pin": 3, "engine": "evu", "op": "add", "repeat": 1, "loads": 1}}, "consumer_ready": {"X": ["input_X.prefetch"], "F0": ["X.context_done 后从同一 HBM 区间 prefetch", "W_F0.prefetch"], "F1": ["F0.output_ready"], "Copy": ["X.context_done 后从同一 HBM 区间 prefetch"], "Add": ["Copy.context_done 后从同一 HBM 区间 prefetch", "F1.context_done 后从同一 HBM 区间 prefetch"]}, "说明": "Copy 真实 load X 并 Store 到独立 Skip/Copy arena 区间；Add 读取 Copy 和 F1。"}
builtin.module {
  tile.program @prog_X (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_F0 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_F1 (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_Copy (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
  tile.program @prog_Add (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %i1: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x64x64xbf16>) {
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
    %b_input_X = nest.alloc slot = "input_X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_input_X = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_X = nest.alloc slot = "X" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_input_X = nest.dma.prefetch.async %h_input_X into %b_input_X : !nest.event<"pref_input_X">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_X, %read_X, %ready_X = nest.dispatch.tasks.async @prog_X context = 0 tasks(%tasks) globals() bindings(%b_input_X, %b_X) ins(%b_input_X) outs(%b_X) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input_X) : (!nest.event<"grid_X">, !nest.event<"read_X">, !nest.event<"ready_X">)
    nest.release %b_input_X depends_on(%read_X, %pref_input_X)
    %store_X_0 = nest.dma.store.async %b_X into %h_X depends_on(%ready_X) : !nest.event<"store_X_0">
    nest.release %b_X depends_on(%store_X_0)
    nest.await %grid_X, %store_X_0
    nest.return
  }
  nest.context @ctx_1 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_W_F0 = nest.alloc slot = "W_F0" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_W_F0 = nest.subview %arena offsets = [131072] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_F0 = nest.alloc slot = "F0" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_F0 = nest.subview %arena offsets = [393216] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_F1 = nest.alloc slot = "F1" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_F1 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %pref_W_F0 = nest.dma.prefetch.async %h_W_F0 into %b_W_F0 : !nest.event<"pref_W_F0">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_F0, %read_F0, %ready_F0 = nest.dispatch.tasks.async @prog_F0 context = 1 tasks(%tasks) globals() bindings(%b_X, %b_W_F0, %b_F0) ins(%b_X, %b_W_F0) outs(%b_F0) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X, %pref_W_F0) : (!nest.event<"grid_F0">, !nest.event<"read_F0">, !nest.event<"ready_F0">)
    %grid_F1, %read_F1, %ready_F1 = nest.dispatch.tasks.async @prog_F1 context = 1 tasks(%tasks) globals() bindings(%b_F0, %b_F1) ins(%b_F0) outs(%b_F1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ready_F0) : (!nest.event<"grid_F1">, !nest.event<"read_F1">, !nest.event<"ready_F1">)
    nest.release %b_X depends_on(%read_F0, %pref_X)
    nest.release %b_W_F0 depends_on(%read_F0, %pref_W_F0)
    %store_F0_0 = nest.dma.store.async %b_F0 into %h_F0 depends_on(%ready_F0) : !nest.event<"store_F0_0">
    nest.release %b_F0 depends_on(%read_F1, %store_F0_0)
    %store_F1_0 = nest.dma.store.async %b_F1 into %h_F1 depends_on(%ready_F1) : !nest.event<"store_F1_0">
    nest.release %b_F1 depends_on(%store_F1_0)
    nest.await %grid_F0, %grid_F1, %store_F0_0, %store_F1_0
    nest.return
  }
  nest.context @ctx_2 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_X = nest.alloc slot = "X" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_X = nest.subview %arena offsets = [262144] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_Copy = nest.alloc slot = "Copy" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Copy = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_X = nest.dma.prefetch.async %h_X into %b_X : !nest.event<"pref_X">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Copy, %read_Copy, %ready_Copy = nest.dispatch.tasks.async @prog_Copy context = 2 tasks(%tasks) globals() bindings(%b_X, %b_Copy) ins(%b_X) outs(%b_Copy) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_X) : (!nest.event<"grid_Copy">, !nest.event<"read_Copy">, !nest.event<"ready_Copy">)
    nest.release %b_X depends_on(%read_Copy, %pref_X)
    %store_Copy_0 = nest.dma.store.async %b_Copy into %h_Copy depends_on(%ready_Copy) : !nest.event<"store_Copy_0">
    nest.release %b_Copy depends_on(%store_Copy_0)
    nest.await %grid_Copy, %store_Copy_0
    nest.return
  }
  nest.context @ctx_3 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_Copy = nest.alloc slot = "Copy" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Copy = nest.subview %arena offsets = [655360] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_F1 = nest.alloc slot = "F1" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_F1 = nest.subview %arena offsets = [524288] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %b_Add = nest.alloc slot = "Add" role = "inout" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %h_Add = nest.subview %arena offsets = [786432] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %pref_Copy = nest.dma.prefetch.async %h_Copy into %b_Copy : !nest.event<"pref_Copy">
    %pref_F1 = nest.dma.prefetch.async %h_F1 into %b_F1 : !nest.event<"pref_F1">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Add, %read_Add, %ready_Add = nest.dispatch.tasks.async @prog_Add context = 3 tasks(%tasks) globals() bindings(%b_Copy, %b_F1, %b_Add) ins(%b_Copy, %b_F1) outs(%b_Add) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_Copy, %pref_F1) : (!nest.event<"grid_Add">, !nest.event<"read_Add">, !nest.event<"ready_Add">)
    nest.release %b_Copy depends_on(%read_Add, %pref_Copy)
    nest.release %b_F1 depends_on(%read_Add, %pref_F1)
    %store_Add_0 = nest.dma.store.async %b_Add into %h_Add depends_on(%ready_Add) : !nest.event<"store_Add_0">
    nest.release %b_Add depends_on(%store_Add_0)
    nest.await %grid_Add, %store_Add_0
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    %done_1 = nexus.submit_context.async @ctx_1(%arena) : !nexus.event<"done_1">
    nexus.await %done_0
    %done_2 = nexus.submit_context.async @ctx_2(%arena) : !nexus.event<"done_2">
    nexus.await %done_2, %done_1
    %done_3 = nexus.submit_context.async @ctx_3(%arena) : !nexus.event<"done_3">
    nexus.await %done_0, %done_1, %done_2, %done_3
    nexus.return
  }
}
