// PR 3.5 acceptance example: L2 admission wait + release-driven wakeup.
//
// Exact-capacity overlap proof:
//   * group_sram_bytes = 262144 = 2 x 131072-byte L2 buffers.
//   * ctx_a atomically admits a_input (role="in") + a_output (role="inout")
//     at submit time, filling L2 completely.
//   * ctx_b is submitted in the same device-PC cycle with no intermediate
//     await; its b_input (role="in") transiently misses and enters
//     ADMISSION_WAIT (not a fault, no partial allocation).
//   * When A's 4/4 input_released aggregate completes and the legal
//     nest.release final-frees a_input, B's FIFO retry admits b_input in
//     the same cycle; B issues its first group action the next cycle,
//     prefetches, dispatches, loads, signals input_released and releases
//     its buffer while A is still computing pow (before A's final DMA
//     store / context completion).
//
// Run:
//   bash examples/run.sh l2-admission-wait \
//     --trace-json /tmp/l2-admission-wait-trace.json \
//     --report /tmp/l2-admission-wait-report.json \
//     --json
//
// NOTE: the admission wait only triggers when L2 capacity is exactly
// exhausted. The required --hw-override group_sram_bytes=262144 (2 x
// 131072) and --sim-override fidelity=full_memory are load-bearing;
// with a larger default L2 or runtime-only fidelity, both contexts
// admit immediately and run concurrently without any wait.
builtin.module {
  tile.program @prog_a (%task: !nest.task, %in_buf: !nest.l2_buffer<4x128x128xbf16>, %out_buf: !nest.l2_buffer<4x128x128xbf16>) {
    %0 = tile.subview %in_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %1 = tile.subview %out_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %2 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %e_load = tile.load.async %0 into %2 : !tile.event<"e_load">
    tile.await %e_load
    tile.signal input_released(%task)
    %e_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 1048576 : !tile.event<"e_pow">
    tile.await %e_pow
    %e_store = tile.store.async %2 into %1 : !tile.event<"e_store">
    tile.await %e_store
    tile.signal output_ready(%task)
    tile.return
  }
  tile.program @prog_b (%task_1: !nest.task, %buf: !nest.l2_buffer<4x128x128xbf16>) {
    %3 = tile.subview %buf task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %4 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %e_load_1 = tile.load.async %3 into %4 : !tile.event<"e_load">
    tile.await %e_load_1
    tile.signal input_released(%task_1)
    tile.return
  }
  nest.context @ctx_a (%A_IN: !nest.global_memref<4x128x128xbf16>, %A_OUT: !nest.global_memref<4x128x128xbf16>) placement = 15 {
    %in_buf_a = nest.alloc slot = "a_input" role = "in" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %out_buf_a = nest.alloc slot = "a_output" role = "inout" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %5 = nest.subview %A_IN offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %6 = nest.subview %A_OUT offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev_pref_in = nest.dma.prefetch.async %5 into %in_buf_a : !nest.event<"ev_pref_in">
    %ev_pref_out = nest.dma.prefetch.async %6 into %out_buf_a : !nest.event<"ev_pref_out">
    %7 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid_a, %ev_inrel_a, %ev_outready_a = nest.dispatch.tasks.async @prog_a tasks(%7) globals() ins(%in_buf_a, %out_buf_a) outs(%in_buf_a, %out_buf_a) signal_policy { input_released = #nest.aggregate<all_tasks>, output_ready = #nest.aggregate<all_tasks> } depends_on(%ev_pref_in, %ev_pref_out) : (!nest.event<"ev_grid_a">, !nest.event<"ev_inrel_a">, !nest.event<"ev_outready_a">)
    nest.release %in_buf_a depends_on(%ev_inrel_a)
    %ev_store_a = nest.dma.store.async %out_buf_a into %6 depends_on(%ev_outready_a) : !nest.event<"ev_store_a">
    nest.release %out_buf_a depends_on(%ev_store_a)
    nest.await %ev_grid_a, %ev_store_a
    nest.return
  }
  nest.context @ctx_b (%B_IN: !nest.global_memref<4x128x128xbf16>) placement = 15 {
    %in_buf_b = nest.alloc slot = "b_input" role = "in" shape = [4, 128, 128] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %8 = nest.subview %B_IN offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev_pref_b = nest.dma.prefetch.async %8 into %in_buf_b : !nest.event<"ev_pref_b">
    %9 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid_b, %ev_inrel_b, %10 = nest.dispatch.tasks.async @prog_b tasks(%9) globals() ins(%in_buf_b) outs(%in_buf_b) signal_policy { input_released = #nest.aggregate<all_tasks> } depends_on(%ev_pref_b) : (!nest.event<"ev_grid_b">, !nest.event<"ev_inrel_b">, !nest.event<"">)
    nest.release %in_buf_b depends_on(%ev_inrel_b)
    nest.await %ev_grid_b
    nest.return
  }
  nexus.program @run (%A_IN: !nest.global_memref<4x128x128xbf16>, %A_OUT: !nest.global_memref<4x128x128xbf16>, %B_IN: !nest.global_memref<4x128x128xbf16>) {
    %done_a = nexus.submit_context.async @ctx_a(%A_IN, %A_OUT) : !nexus.event<"done_a">
    %done_b = nexus.submit_context.async @ctx_b(%B_IN) : !nexus.event<"done_b">
    nexus.await %done_a, %done_b
    nexus.return
  }
}
