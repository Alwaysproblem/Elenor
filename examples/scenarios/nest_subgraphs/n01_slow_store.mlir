// 可复现 NEST 边界子图；中文元数据是人类可读说明，不是可执行 schema。
// 快 EVU 后顺序执行 10 次同址 HBM Store；只在最后一次 Store 后释放 output。
// case: {"id":"N01","name":"n01_slow_store","nodes":["Compute"],"edges":{},"context_partition":[["Compute"]],"resource_config":{"uce":4,"device_contexts":4,"fidelity":"full_memory","num_dma_channels":2,"arena_binding":"arena=0x1000000:8388608:rw","placement":15},"expected":{"时序/生命周期 Correctness":"output_ready 不早于 tile.store；context_done 不早于第 10 次 global Store 完成","数值":"未建模；engine descriptor 与未初始化的合成输出只用于时序","Liveness":"所有已接纳 Context 完成","Scheduling Quality":"仅记录实际 service/等待，不声称最优调度"},"forbidden_dependencies":"不得用 output_ready 代替 HBM 可见性，也不得用 device run 结束代替 tile_done","tensors":{"Input":{"index":0,"offset_elements":0,"bytes":32768,"arena_interval_bytes":[0,32768],"role":"external_input"},"Output":{"index":1,"offset_elements":131072,"bytes":131072,"arena_interval_bytes":[262144,393216],"role":"output"}},"phase":{"Compute":["pref_Input"],"context_done":["grid_Compute","store_Output_9"]},"pins":{"Compute":0},"node_programs":{"Compute":{"program":"prog_Compute","engine":"evu:expand","repeat":1,"pin":0}}}
builtin.module {
  tile.program @prog_Compute (%task: !nest.task, %i0: !nest.l2_buffer<4x64x64xbf16>, %out: !nest.l2_buffer<4x256x64xbf16>) {
    %v0 = tile.subview %i0 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %l0 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %vo = tile.subview %out task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 64] strides = [1, 1, 1] : !nest.l2_view<1x256x64xbf16>
    %lout = tile.alloc shape = [256, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<256x64xbf16>
    %load_0 = tile.load.async %v0 into %l0 : !tile.event<"load_0">
    tile.await %load_0
    tile.signal input_released(%task)
    %compute_0 = tile.evu.async "expand" ops = 16448 : !tile.event<"compute_0">
    tile.await %compute_0
    %stored = tile.store.async %lout into %vo : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx_0 (%arena: !nest.global_memref<4194304xbf16>) placement = 15 {
    %b_input = nest.alloc slot = "input" role = "in" shape = [4, 64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %b_output = nest.alloc slot = "output" role = "inout" shape = [4, 256, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x256x64xbf16>
    %h_input = nest.subview %arena offsets = [0] sizes = [16384] strides = [1] : !nest.global_view<16384xbf16>
    %h_output = nest.subview %arena offsets = [131072] sizes = [65536] strides = [1] : !nest.global_view<65536xbf16>
    %pref_input = nest.dma.prefetch.async %h_input into %b_input : !nest.event<"pref_input">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_Compute, %read_Compute, %ready_Compute = nest.dispatch.tasks.async @prog_Compute context = 0 tasks(%tasks) globals() ins(%b_input, %b_output) outs(%b_input, %b_output) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%pref_input) : (!nest.event<"grid_Compute">, !nest.event<"read_Compute">, !nest.event<"ready_Compute">)
    nest.release %b_input depends_on(%read_Compute)
    %store_Output_0 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_0">
    nest.await %store_Output_0
    %store_Output_1 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_1">
    nest.await %store_Output_1
    %store_Output_2 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_2">
    nest.await %store_Output_2
    %store_Output_3 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_3">
    nest.await %store_Output_3
    %store_Output_4 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_4">
    nest.await %store_Output_4
    %store_Output_5 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_5">
    nest.await %store_Output_5
    %store_Output_6 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_6">
    nest.await %store_Output_6
    %store_Output_7 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_7">
    nest.await %store_Output_7
    %store_Output_8 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_8">
    nest.await %store_Output_8
    %store_Output_9 = nest.dma.store.async %b_output into %h_output depends_on(%ready_Compute) : !nest.event<"store_Output_9">
    nest.release %b_output depends_on(%store_Output_9)
    nest.await %grid_Compute, %store_Output_9
    nest.return
  }
  nexus.program @run (%arena: !nest.global_memref<4194304xbf16>) {
    %done_0 = nexus.submit_context.async @ctx_0(%arena) : !nexus.event<"done_0">
    nexus.await %done_0
    nexus.return
  }
}
