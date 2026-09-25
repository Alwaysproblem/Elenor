// L2 profile switch load-ordering example.
//
// Semantics demonstrated:
//   * ctx_a (L2 mode0) releases its input EARLY (batch-I private early
//     release: a_input's physical extent returns while A is still computing
//     pow, long before A's HBM store / context completion).
//   * ctx_b runs under a DIFFERENT L2 profile (mode1).  Its admission and
//     first HBM->L2 load must therefore NOT be pulled forward to a_input's
//     release cycle: the compiler generates the L2 0->1 profile command
//     (frontier = done_a, i.e. after A's store and completion) and only
//     after that command completes does ctx_b admit and issue its
//     prefetch.  ctx_b is submitted consecutively with NO source-level
//     await, so the ordering shown in the trace is produced by the
//     profile-frontier machinery itself.
//
// Trace ordering proven by
//   pipeline_validator/tests/test_runtime.py::TestL2ProfileSwitchOrdering
// in BOTH fidelities:
//   a_input l2_extent_release  <  ctx_a completion
//   <  L2 profile_command end  <=  ctx_b port active_cycle
//   <=  ctx_b first prefetch leg.
//
// Run:
//   bash examples/run.sh l2-profile-switch-load-ordering \
//     --sim-override fidelity=full_memory \
//     --trace-json /tmp/l2-switch-trace.json \
//     --report /tmp/l2-switch-report.json --json

builtin.module {
  tile.program @prog_a(
    %task: !nest.task, %in_buf: !nest.l2_buffer<4x128x128xbf16>,
    %out_buf : !nest.l2_buffer<4x128x128xbf16>)
            resource_contract = #tile.resources<
        allowed_profiles = [0, 1, 2],
        tile_l1_spm_bytes_per_context = 32768> {
    %0 = tile.subview %in_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128]
      strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %1 = tile.subview %out_buf task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128]
      strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %2 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x128xbf16>
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
  tile.program @prog_b (%task_1: !nest.task, %buf: !nest.l2_buffer<4x128x128xbf16>)
        resource_contract = #tile.resources<
        allowed_profiles = [0, 1, 2],
        tile_l1_spm_bytes_per_context = 32768> {
    %3 = tile.subview %buf task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 128]
      strides = [1, 1, 1] : !nest.l2_view<1x128x128xbf16>
    %4 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x128xbf16>
    %e_load_1 = tile.load.async %3 into %4 : !tile.event<"e_load">
    tile.await %e_load_1
    tile.signal input_released(%task_1)
    tile.return
  }
  nest.context @ctx_a(
    %A_IN: !nest.global_memref<4x128x128xbf16>, %A_OUT: !nest.global_memref<4x128x128xbf16>)
    placement = 15
            resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
        logical_tasks = 4, l2_spm_bytes = 262144, requested_contexts_per_tile = 1> {
    %in_buf_a = nest.alloc slot = "a_input" role = "in" shape = [4, 128, 128] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %out_buf_a = nest.alloc slot = "a_output" role = "inout" shape = [4, 128, 128] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %5 = nest.subview %A_IN offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
      : !nest.global_view<4x128x128xbf16>
    %6 = nest.subview %A_OUT offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
      : !nest.global_view<4x128x128xbf16>
    %ev_pref_in = nest.dma.prefetch.async %5 into %in_buf_a : !nest.event<"ev_pref_in">
    %ev_pref_out = nest.dma.prefetch.async %6 into %out_buf_a : !nest.event<"ev_pref_out">
    %7 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid_a, %ev_inrel_a, %ev_outready_a = nest.dispatch.tasks.async @prog_a l1_mode = 0
      tasks(%7) globals()
      bindings(%in_buf_a, %out_buf_a) ins(%in_buf_a) outs(%out_buf_a)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%ev_pref_in, %ev_pref_out)
      : (!nest.event<"ev_grid_a">, !nest.event<"ev_inrel_a">, !nest.event<"ev_outready_a">)
    nest.release %in_buf_a depends_on(%ev_inrel_a, %ev_pref_in)
    %ev_store_a = nest.dma.store.async %out_buf_a into %6 depends_on(%ev_outready_a)
      : !nest.event<"ev_store_a">
    nest.release %out_buf_a depends_on(%ev_pref_out, %ev_store_a)
    nest.await %ev_grid_a, %ev_store_a
    nest.return
  }
  nest.context @ctx_b(
    %B_IN: !nest.global_memref<4x128x128xbf16>)
    placement = 15
            resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1],
        logical_tasks = 4, l2_spm_bytes = 131072, requested_contexts_per_tile = 1> {
    %in_buf_b = nest.alloc slot = "b_input" role = "in" shape = [4, 128, 128] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
    %8 = nest.subview %B_IN offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
      : !nest.global_view<4x128x128xbf16>
    %ev_pref_b = nest.dma.prefetch.async %8 into %in_buf_b : !nest.event<"ev_pref_b">
    %9 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_grid_b, %ev_inrel_b, %10 = nest.dispatch.tasks.async @prog_b l1_mode = 0
      tasks(%9) globals()
      bindings(%in_buf_b) ins(%in_buf_b) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%ev_pref_b)
      : (!nest.event<"ev_grid_b">, !nest.event<"ev_inrel_b">, !nest.event<"">)
    nest.release %in_buf_b depends_on(%ev_inrel_b, %ev_pref_b)
    nest.await %ev_grid_b
    nest.return
  }
  nexus.program @l2_switch_ordering(
    %A_IN: !nest.global_memref<4x128x128xbf16>, %A_OUT: !nest.global_memref<4x128x128xbf16>,
    %B_IN: !nest.global_memref<4x128x128xbf16>) {
    // ctx_b is submitted CONSECUTIVELY with no source-level await: the
    // ordering must come entirely from the compiler-generated L2 profile
    // command, whose frontier is done_a (A's HBM store and completion).
    // With batch-I early release a same-profile successor would admit at
    // a_input's release cycle; the mode1 requirement instead holds ctx_b
    // until the completed switch.
    %done_a = nexus.submit_context.async @ctx_a(%A_IN, %A_OUT) : !nexus.event<"done_a">
    %done_b = nexus.submit_context.async @ctx_b(%B_IN) : !nexus.event<"done_b">
    nexus.await %done_a, %done_b
    nexus.return
  }
}
