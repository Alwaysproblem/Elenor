// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// 容量必须以 group_sram_bytes=65536 运行；单 Context bundle=98304，属于永久不可能而非暂时等待。
// case: {"id":"N09","name":"n09_impossible","nodes":["Impossible"],"edges":{},"context_partition":[["Impossible"]],"resource_config":{"uce":4,"device_contexts":4,"fidelity":"full_memory","num_dma_channels":2,"arena_binding":"arena=0x1000000:8388608:rw","group_sram_bytes":65536,"placement":15,"expected_runtime_fault":"L2 capacity fault during context admission"},"expected":{"时序/生命周期 Correctness":"单 Context 原子需求 input U + output 2U = 98304，大于 65536；不得进入 WAIT_CAPACITY","数值":"未建模；engine descriptor 与未初始化的合成输出只用于时序","Liveness":"预期立即容量 fault、非零退出，且不是 max-cycle 超时","Scheduling Quality":"应有零成功 engine service"},"forbidden_dependencies":"禁止把永久不足排入 admission wait；禁止部分分配后启动 DMA/engine","tensors":{"Input":{"index":0,"offset_elements":0,"bytes":32768,"arena_interval_bytes":[0,32768],"role":"external_input"},"Output":{"index":1,"offset_elements":131072,"bytes":65536,"arena_interval_bytes":[262144,327680],"role":"output"}},"phase":{"Impossible":["context_admission_only"],"expected_wait_count":0},"pins":{"Impossible":0},"node_programs":{"Impossible":{"program":"prog_Impossible","engine":"evu:expand","repeat":1,"pin":0}}}
builtin.module {
  tile.program @prog_Impossible (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x128x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %lout = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    tile.free %l0
    %stored = tile.store.async %lout into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_output = nest.alloc slot = "output" role = "inout" shape = [4, 128, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_output = nest.subview %arena offsets = [131072] sizes = [32768] strides = [1] : !nest.global_view<32768xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Impossible, %read_Impossible, %ready_Impossible = nest.dispatch.tasks.async @prog_Impossible context = 0 tasks(%tasks) globals() bindings(%b_input, %b_output) ins(%b_input) outs(%b_output) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input) : (!nest.event<"grid_Impossible">, !nest.event<"read_Impossible">, !nest.event<"ready_Impossible">)
    nest.release %b_input depends_on(%read_Impossible, %pref_input)
    %store_Output = nest.dma.store.async %b_output into %h_output depends_on(%ready_Impossible) : !nest.event<"store_Output">
    nest.release %b_output depends_on(%store_Output)
    nest.await %grid_Impossible, %store_Output
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
