// CPU submits A, then dependent C, then independent B without blocking on A.
// C reads A's real HBM output; B uses a separate destination.
// Bind src, a, b, c to distinct 2048-byte global regions.
builtin.module {
  tile.program @worker_tile(
    %task: !nest.task, %input: !nest.l2_buffer<1x32x32xbf16>,
    %output : !nest.l2_buffer<1x32x32xbf16>)
            resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
        tile_l1_spm_bytes_per_context = 4096> {
    %iv = tile.subview %input task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %ov = tile.subview %output task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 32, 32]
      strides = [1, 1, 1] : !nest.l2_view<1x32x32xbf16>
    %local = tile.alloc shape = [32, 32] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<32x32xbf16>
    %load = tile.load.async %iv into %local : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %compute = tile.evu.async "work" ops = 262144 : !tile.event<"compute">
    tile.await %compute
    %store = tile.store.async %local into %ov : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @worker(
    %input: !nest.global_memref<1024xbf16>, %output: !nest.global_memref<1024xbf16>)
    placement = 1
            resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
        logical_tasks = 1, l2_spm_bytes = 8192, requested_contexts_per_tile = 1> {
    %i = nest.alloc slot = "input" role = "in" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %o = nest.alloc slot = "output" role = "out" shape = [1, 32, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32x32xbf16>
    %hi = nest.subview %input offsets = [0] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %ho = nest.subview %output offsets = [0] sizes = [1024] strides = [1]
      : !nest.global_view<1024xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %prefetch = nest.dma.prefetch.async %hi into %i : !nest.event<"prefetch">
    %grid, %read, %ready = nest.dispatch.tasks.async @worker_tile l1_mode = 0 tasks(%tasks)
      globals()
      bindings(%i, %o) ins(%i) outs(%o)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%prefetch) : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    nest.release %i depends_on(%prefetch, %read)
    %stored = nest.dma.store.async %o into %ho depends_on(%ready) : !nest.event<"stored">
    nest.release %o depends_on(%stored)
    nest.return
  }
  nexus.program @run(
    %src: !nest.global_memref<1024xbf16>, %a: !nest.global_memref<1024xbf16>,
    %b: !nest.global_memref<1024xbf16>, %c: !nest.global_memref<1024xbf16>) {
    %a_done = nexus.submit_context.async @worker(%src, %a) : !nexus.event<"a_done">
    %c_done = nexus.submit_context.async @worker(%a, %c) depends_on(%a_done)
      : !nexus.event<"c_done">
    %b_done = nexus.submit_context.async @worker(%src, %b) : !nexus.event<"b_done">
    nexus.await %c_done, %b_done
    nexus.return
  }
}
