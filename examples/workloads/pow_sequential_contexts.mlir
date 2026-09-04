builtin.module {
  tile.program @pow_tile (%task: !nest.task, %l2_buf: !nest.l2_buffer<4x128x128xbf16>) {
    %0 = tile.subview %l2_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %1 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %e_load = tile.load.async %0 into %1 : !tile.event<"e_load">
    tile.await %e_load
    tile.signal input_released(%task)
    %e_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536 : !tile.event<"e_pow">
    tile.await %e_pow
    %e_store = tile.store.async %1 into %0 : !tile.event<"e_store">
    tile.await %e_store
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @pow_ctx0 (%Y: !nest.global_memref<4x128x128xbf16>) placement = 15 {
    %l2_buf_1 = nest.alloc slot = "l2_buf" role = "inout" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %2 = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev_in = nest.dma.prefetch.async %2 into %l2_buf_1 : !nest.event<"ev_in">
    %3 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid, %ev_inrel, %ev_outready = nest.dispatch.tasks.async @pow_tile tasks(%3) globals() ins(%l2_buf_1) outs(%l2_buf_1) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ev_in) : (!nest.event<"ev_grid">, !nest.event<"ev_inrel">, !nest.event<"ev_outready">)
    %ev_out = nest.dma.store.async %l2_buf_1 into %2 depends_on(%ev_outready) : !nest.event<"ev_out">
    nest.release %l2_buf_1 depends_on(%ev_out)
    nest.await %ev_grid, %ev_out
    nest.return
  }
  nest.context @pow_ctx1 (%Y_1: !nest.global_memref<4x128x128xbf16>) placement = 15 {
    %l2_buf_2 = nest.alloc slot = "l2_buf" role = "inout" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %4 = nest.subview %Y_1 offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev_in_1 = nest.dma.prefetch.async %4 into %l2_buf_2 : !nest.event<"ev_in">
    %5 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid_1, %ev_inrel_1, %ev_outready_1 = nest.dispatch.tasks.async @pow_tile tasks(%5) globals() ins(%l2_buf_2) outs(%l2_buf_2) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ev_in_1) : (!nest.event<"ev_grid_1">, !nest.event<"ev_inrel_1">, !nest.event<"ev_outready_1">)
    %ev_out_1 = nest.dma.store.async %l2_buf_2 into %4 depends_on(%ev_outready_1) : !nest.event<"ev_out_1">
    nest.release %l2_buf_2 depends_on(%ev_out_1)
    nest.await %ev_grid_1, %ev_out_1
    nest.return
  }
  nexus.program @run_pow (%Y0: !nest.global_memref<4x128x128xbf16>, %Y1: !nest.global_memref<4x128x128xbf16>) {
    %done0 = nexus.submit_context.async @pow_ctx0(%Y0) : !nexus.event<"done0">
    nexus.await %done0
    %done1 = nexus.submit_context.async @pow_ctx1(%Y1) : !nexus.event<"done1">
    nexus.await %done1
    nexus.return
  }
}
