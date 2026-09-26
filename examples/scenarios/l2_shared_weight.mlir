// Batch III acceptance: cross-context readonly L2 weight sharing.
//
// load_weight prefetches W ([64,64] bf16 = 8192 B) from HBM exactly once,
// publishes the backing, releases its own view and returns. Each @reader
// receives the readonly capability as a context L2 formal, forwards it through
// dispatch bindings/ins to the tile program's own L2 formal, and performs four
// real L2->L1 W loads (one per tile). The only eliminated traffic is the
// second HBM->L2 W prefetch; per-tile L2->L1 reads all still happen.
//
// C is submitted only after B completes (nexus.await %b_done), so B's release
// and the producer's retirement cannot free W: the same backing stays live
// with C's DECLARED claim until C's final safe release, which produces the
// single physical l2_extent_release for W.
//
// Run:
//   bash examples/run.sh l2-shared-weight \
//     --sim-override fidelity=full_memory \
//     --memory-trace --trace-json OUT/sw.trace.json \
//     --report OUT/sw.report.json --json
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
  nest.context @load_weight(%W_IN : !nest.global_memref<64x64xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 0, l2_spm_bytes = 8192, requested_contexts_per_tile = 1> {
    %w = nest.alloc slot = "W" role = "in" sharing = "readonly" shape = [64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<64x64xbf16>
    %source = nest.subview %W_IN offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.global_view<64x64xbf16>
    %prefetched = nest.dma.prefetch.async %source into %w : !nest.event<"prefetched">
    %published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">
    nest.release %w depends_on(%prefetched, %published)
    nest.return
  }
  nest.context @reader(
      %OUT : !nest.global_memref<4x64x64xbf16>,
      %weight : !nest.l2_buffer<64x64xbf16>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 32768, requested_contexts_per_tile = 1> {
    %result = nest.alloc slot = "result" role = "out" shape = [4, 64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %output = nest.subview %OUT offsets = [0, 0, 0] sizes = [4, 64, 64] strides = [1, 1, 1]
        : !nest.global_view<4x64x64xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @copy_weight l1_mode = 0
        tasks(%tasks) globals() bindings(%weight, %result) ins(%weight) outs(%result)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %weight depends_on(%input_done)
    %written = nest.dma.store.async %result into %output depends_on(%ready)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid, %written
    nest.return
  }
  nexus.program @share(
      %W : !nest.global_memref<64x64xbf16>,
      %B_OUT : !nest.global_memref<4x64x64xbf16>,
      %C_OUT : !nest.global_memref<4x64x64xbf16>) {
    %loaded = nexus.submit_context.async @load_weight(%W) : !nexus.event<"loaded">
    %shared_w = nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<64x64xbf16>
    %b_done = nexus.submit_context.async @reader(%B_OUT, %shared_w) depends_on(%loaded)
        : !nexus.event<"b_done">
    nexus.await %b_done
    %c_done = nexus.submit_context.async @reader(%C_OUT, %shared_w) depends_on(%loaded)
        : !nexus.event<"c_done">
    nexus.await %c_done
    nexus.return
  }
}
