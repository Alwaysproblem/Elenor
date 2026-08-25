builtin.module {
  tile.program @pow_4k_tile (%task : !nest.task, %l2_buf : !nest.l2_buffer<4x128x128xbf16>) {
    %l2_tile = tile.subview %l2_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %l1 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %e_load = tile.load.async %l2_tile into %l1 : !tile.event<"e_load">
    tile.await %e_load
    tile.signal input_released
    %e_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536 : !tile.event<"e_pow">
    tile.await %e_pow
    %e_store = tile.store.async %l1 into %l2_tile : !tile.event<"e_store">
    tile.await %e_store
    tile.signal output_ready
    tile.return
  }
  tile.program @pow_4k_tile_1 (%task : !nest.task, %l2_buf : !nest.l2_buffer<4x128x256xbf16>) {
    %l2_tile = tile.subview %l2_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %l1 = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x256xbf16>
    %e_load = tile.load.async %l2_tile into %l1 : !tile.event<"e_load">
    tile.await %e_load
    tile.signal input_released
    %e_pow = tile.pow.async bytes = 65536 exponent = 2 pow_ops = 65536 : !tile.event<"e_pow">
    tile.await %e_pow
    %e_store = tile.store.async %l1 into %l2_tile : !tile.event<"e_store">
    tile.await %e_store
    tile.signal output_ready
    tile.return
  }
  nest.context @pow_task (%Y : !nest.global_memref<4x128x128xbf16>) placement = 15 context = 0 {
    %l2_buf_pow0 = nest.alloc slot = "l2_buf_pow0" role = "inout" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %src = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev_dma_pow_in0 = nest.dma.prefetch.async %src into %l2_buf_pow0 : !nest.event<"ev_dma_pow_in0">
    %0 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_role_pow0, %ev_inrel_pow0, %ev_outready_pow0 = nest.dispatch.tasks.async @pow_4k_tile context = 0 tasks(%0) ins(%l2_buf_pow0) outs(%l2_buf_pow0) depends_on(%ev_dma_pow_in0) : (!nest.event<"ev_role_pow0">, !nest.event<"ev_inrel_pow0">, !nest.event<"ev_outready_pow0">)
    %ev_dma_pow_out0 = nest.dma.store.async %l2_buf_pow0 into %src depends_on(%ev_outready_pow0) : !nest.event<"ev_dma_pow_out0">
    nest.release %l2_buf_pow0 depends_on(%ev_dma_pow_out0)
    nest.await %ev_role_pow0, %ev_dma_pow_out0
    nest.return
  }
  nest.context @pow_task_1 (%Y : !nest.global_memref<4x128x256xbf16>) placement = 15 context = 1 {
    %l2_buf_pow0 = nest.alloc slot = "l2_buf_pow0_1" role = "inout" shape = [4, 128, 256] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %src = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 256] strides = [1, 1, 1] : !nest.global_view<4x128x256xbf16>
    %ev_dma_pow_in0 = nest.dma.prefetch.async %src into %l2_buf_pow0 : !nest.event<"ev_dma_pow_in0">
    %0 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_role_pow0, %ev_inrel_pow0, %ev_outready_pow0 = nest.dispatch.tasks.async @pow_4k_tile_1 context = 1 tasks(%0) ins(%l2_buf_pow0) outs(%l2_buf_pow0) depends_on(%ev_dma_pow_in0) : (!nest.event<"ev_role_pow0">, !nest.event<"ev_inrel_pow0">, !nest.event<"ev_outready_pow0">)
    %ev_dma_pow_out0 = nest.dma.store.async %l2_buf_pow0 into %src depends_on(%ev_outready_pow0) : !nest.event<"ev_dma_pow_out0">
    nest.release %l2_buf_pow0 depends_on(%ev_dma_pow_out0)
    nest.await %ev_role_pow0, %ev_dma_pow_out0
    nest.return
  }
  nexus.program @run_pow (%Y0 : !nest.global_memref<4x128x128xbf16>, %Y1 : !nest.global_memref<4x128x256xbf16>) {
    %context_done = nexus.submit_context.async @pow_task(%Y0) : !nexus.event<"context_done">
    %context_done_1 = nexus.submit_context.async @pow_task_1(%Y1) : !nexus.event<"context_done_1">
    nexus.await %context_done
    nexus.await %context_done_1
    nexus.return
  }
}
