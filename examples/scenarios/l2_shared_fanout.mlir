// Batch III acceptance: L2 intermediate-result fanout without an HBM detour.
//
// Producer @make_x prefetches A_IN (8192 B) and creates X ([64,64] bf16,
// readonly export) with a real tile.load -> tile.store inside L2, then
// publishes X after the writer output_ready. The module defines NO HBM
// formal, binding or address for X: no GLOBAL_STORE / HBM prefetch may
// reference it. B/C receive the same readonly capability through
// nexus.shared.ref -> context L2 formal -> dispatch bindings/ins -> tile
// program L2 formal and each performs four real L2->L1 X loads.
//
// C submits only after B completes, so producer retirement and B's release
// cannot free X; the single physical free happens after C's final release.
//
// Run:
//   bash examples/run.sh l2-shared-fanout \
//     --sim-override fidelity=full_memory \
//     --memory-trace --trace-json OUT/sf.trace.json \
//     --report OUT/sf.report.json --json
builtin.module {
  tile.program @make_x(%task : !nest.task, %input : !nest.l2_buffer<64x64xbf16>,
      %x : !nest.l2_buffer<64x64xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %src = tile.subview %input offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.l2_view<64x64xbf16>
    %dst = tile.subview %x offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.l2_view<64x64xbf16>
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
  tile.program @copy_shared(%task : !nest.task, %x : !nest.l2_buffer<64x64xbf16>,
      %out_buf : !nest.l2_buffer<4x64x64xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %src = tile.subview %x offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
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
  nest.context @producer(%A_IN : !nest.global_memref<64x64xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 1, l2_spm_bytes = 16384, requested_contexts_per_tile = 1> {
    %input = nest.alloc slot = "input" role = "in" shape = [64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<64x64xbf16>
    %w = nest.alloc slot = "X" role = "out" sharing = "readonly" shape = [64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<64x64xbf16>
    %source = nest.subview %A_IN offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.global_view<64x64xbf16>
    %prefetched = nest.dma.prefetch.async %source into %input : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @make_x l1_mode = 0
        tasks(%tasks) globals() bindings(%input, %w) ins(%input) outs(%w)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> } depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %input depends_on(%input_done, %prefetched)
    %published = nest.publish %w depends_on(%ready) : !nest.event<"published">
    nest.release %w depends_on(%published)
    nest.await %grid
    nest.return
  }
  nest.context @reader(
      %OUT : !nest.global_memref<4x64x64xbf16>,
      %x : !nest.l2_buffer<64x64xbf16>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 32768, requested_contexts_per_tile = 1> {
    %result = nest.alloc slot = "result" role = "out" shape = [4, 64, 64] dtype = "bf16"
        alignment = 256 : !nest.l2_buffer<4x64x64xbf16>
    %output = nest.subview %OUT offsets = [0, 0, 0] sizes = [4, 64, 64] strides = [1, 1, 1]
        : !nest.global_view<4x64x64xbf16>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @copy_shared l1_mode = 0
        tasks(%tasks) globals() bindings(%x, %result) ins(%x) outs(%result)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %x depends_on(%input_done)
    %written = nest.dma.store.async %result into %output depends_on(%ready)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid, %written
    nest.return
  }
  nexus.program @fanout(
      %A_IN : !nest.global_memref<64x64xbf16>,
      %B_OUT : !nest.global_memref<4x64x64xbf16>,
      %C_OUT : !nest.global_memref<4x64x64xbf16>) {
    %a_done = nexus.submit_context.async @producer(%A_IN) : !nexus.event<"a_done">
    %shared_x = nexus.shared.ref %a_done slot = "X" : !nest.l2_buffer<64x64xbf16>
    %b_done = nexus.submit_context.async @reader(%B_OUT, %shared_x) depends_on(%a_done)
        : !nexus.event<"b_done">
    nexus.await %b_done
    %c_done = nexus.submit_context.async @reader(%C_OUT, %shared_x) depends_on(%a_done)
        : !nexus.event<"c_done">
    nexus.await %c_done
    nexus.return
  }
}
