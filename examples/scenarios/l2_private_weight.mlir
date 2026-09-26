// Private-weight comparison fixture for l2-shared-weight (plan/03 §1.5).
//
// Same tensor shapes and tile program family as l2_shared_weight.mlir, but W
// is NOT shared: every @reader prefetches its own 8192 B private copy from the
// same HBM binding. This is the measurement control only — it adds no runtime
// mode. Expected W HBM->L2 input traffic: private = 2 x 8192 = 16384 B versus
// shared = 8192 B; per-tile L2->L1 loads and output writes are identical and
// are not part of the comparison.
//
// Run:
//   bash examples/run.sh l2-private-weight \
//     --sim-override fidelity=full_memory \
//     --memory-trace --trace-json OUT/pw.trace.json \
//     --report OUT/pw.report.json --json
builtin.module {
  tile.program @copy_weight(%task : !nest.task, %w : !nest.l2_buffer<64x64xbf16>,
      %out_buf : !nest.l2_buffer<4x64x64xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %src = tile.subview %w offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.l2_view<64x64xbf16>
    %dst = tile.subview %out_buf task = %task task_dim = 0 offsets = [0, 0, 0]
        sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %local = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
        : !tile.l1_buffer<64x64xbf16>
    %loaded = tile.load.async %src into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %dst : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @private_reader(
      %W : !nest.global_memref<64x64xbf16>,
      %OUT : !nest.global_memref<4x64x64xbf16>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 40960, requested_contexts_per_tile = 1> {
    %weight = nest.alloc slot = "weight" role = "in" shape = [64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<64x64xbf16>
    %result = nest.alloc slot = "result" role = "out" shape = [4, 64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %source = nest.subview %W offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.global_view<64x64xbf16>
    %output = nest.subview %OUT offsets = [0, 0, 0] sizes = [4, 64, 64] strides = [1, 1, 1]
        : !nest.global_view<4x64x64xbf16>
    %prefetched = nest.dma.prefetch.async %source into %weight : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @copy_weight l1_mode = 0
        tasks(%tasks) globals() bindings(%weight, %result) ins(%weight) outs(%result)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> } depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %weight depends_on(%input_done, %prefetched)
    %written = nest.dma.store.async %result into %output depends_on(%ready)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid, %written
    nest.return
  }
  nexus.program @private_copy(
      %W : !nest.global_memref<64x64xbf16>,
      %B_OUT : !nest.global_memref<4x64x64xbf16>,
      %C_OUT : !nest.global_memref<4x64x64xbf16>) {
    %b_done = nexus.submit_context.async @private_reader(%W, %B_OUT)
        : !nexus.event<"b_done">
    nexus.await %b_done
    %c_done = nexus.submit_context.async @private_reader(%W, %C_OUT)
        : !nexus.event<"c_done">
    nexus.await %c_done
    nexus.return
  }
}
