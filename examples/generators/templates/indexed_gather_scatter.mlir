builtin.module {
  tile.program @indexed_mem_tile(
      %task : !nest.task,
      %data : !nest.global_view<16xi32>,
      %out : !nest.global_view<16xi32>,
      %gather_idx_l2 : !nest.l2_buffer<3xi32>,
      %scatter_idx_l2 : !nest.l2_buffer<3xi32>)
                resource_contract = #tile.resources<allowed_profiles = [1, 2],
          tile_l1_spm_bytes_per_context = 3072,
          l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 65536},
          l2_cache = {
            required = false, access = "read", bypass = "allowed", target_bytes = 65536}
            > {
    %gather_idx_view = tile.subview %gather_idx_l2
        offsets = [0] sizes = [3] strides = [1]
        : !nest.l2_view<3xi32>
    %scatter_idx_view = tile.subview %scatter_idx_l2
        offsets = [0] sizes = [3] strides = [1]
        : !nest.l2_view<3xi32>
    %gather_idx_l1 = tile.alloc shape = [3] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<3xi32>
    %scatter_idx_l1 = tile.alloc shape = [3] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<3xi32>
    %gather_dst = tile.alloc shape = [12] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<12xi32>
    %gather_idx_ready = tile.load.async %gather_idx_view into %gather_idx_l1
        : !tile.event<"gather_idx_ready">
    %scatter_idx_ready = tile.load.async %scatter_idx_view into %scatter_idx_l1
        : !tile.event<"scatter_idx_ready">
    tile.await %gather_idx_ready, %scatter_idx_ready
    tile.signal input_released(%task)
    %gathered = tile.gather.global.async %data
        indices(%gather_idx_l1) into %gather_dst
        map = #tile.indexed_map<index_scale = 4 offset = 0 task_stride = 0 repeat = 1 stride = 0 segment = 4>
        window_entries = 3 : !tile.event<"gathered">
    tile.await %gathered
    %scattered = tile.scatter.global.async %gather_dst
        indices(%scatter_idx_l1) into %out
        map = #tile.indexed_map<index_scale = 4 offset = 0 task_stride = 0 repeat = 1 stride = 0 segment = 4>
        window_entries = 3 : !tile.event<"scattered">
    tile.await %scattered
    tile.free %gather_idx_l1
    tile.free %scatter_idx_l1
    tile.free %gather_dst
    tile.return
  }

  nest.context @indexed_memory_context(
      %DATA : !nest.global_memref<16xi32>,
      %OUT : !nest.global_memref<16xi32>,
      %GATHER_IDX : !nest.global_memref<3xi32>,
      %SCATTER_IDX : !nest.global_memref<3xi32>) placement = 1
                resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1, 2],
          logical_tasks = 1, l2_spm_bytes = 8192, requested_contexts_per_tile = 1,
          l2_cache = {
            required = false, access = "read", bypass = "allowed", target_bytes = 65536}
            > {
    %DATA_view = nest.subview %DATA offsets = [0] sizes = [16]
        strides = [1] : !nest.global_view<16xi32>
    %OUT_view = nest.subview %OUT offsets = [0] sizes = [16]
        strides = [1] : !nest.global_view<16xi32>
    %GATHER_IDX_view = nest.subview %GATHER_IDX offsets = [0] sizes = [3]
        strides = [1] : !nest.global_view<3xi32>
    %SCATTER_IDX_view = nest.subview %SCATTER_IDX offsets = [0] sizes = [3]
        strides = [1] : !nest.global_view<3xi32>
    %GATHER_IDX_l2 = nest.alloc slot = "im_gather_idx" role = "in"
        shape = [3] dtype = "i32" alignment = 64
        : !nest.l2_buffer<3xi32>
    %SCATTER_IDX_l2 = nest.alloc slot = "im_scatter_idx" role = "in"
        shape = [3] dtype = "i32" alignment = 64
        : !nest.l2_buffer<3xi32>
    %GATHER_IDX_prefetched = nest.dma.prefetch.async %GATHER_IDX_view into %GATHER_IDX_l2
        : !nest.event<"gather_idx_prefetched">
    %SCATTER_IDX_prefetched = nest.dma.prefetch.async %SCATTER_IDX_view into %SCATTER_IDX_l2
        : !nest.event<"scatter_idx_prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @indexed_mem_tile l1_mode = 1
        tasks(%tasks) globals(%DATA_view, %OUT_view)
        bindings(%GATHER_IDX_l2, %SCATTER_IDX_l2)
        ins(%GATHER_IDX_l2, %SCATTER_IDX_l2)
        outs()
        signal_policy {
          input_released = #nest.aggregate<all_tasks>
        }
        depends_on(%GATHER_IDX_prefetched, %SCATTER_IDX_prefetched)
        : (!nest.event<"im_grid_done">, !nest.event<"im_input_released">,
           !nest.event<"">)
    nest.release %GATHER_IDX_l2 depends_on(%input_released, %GATHER_IDX_prefetched)
    nest.release %SCATTER_IDX_l2 depends_on(%input_released, %SCATTER_IDX_prefetched)
    nest.await %grid_done
    nest.return
  }

  nexus.program @run_indexed_memory(
      %DATA : !nest.global_memref<16xi32>,
      %OUT : !nest.global_memref<16xi32>,
      %GATHER_IDX : !nest.global_memref<3xi32>,
      %SCATTER_IDX : !nest.global_memref<3xi32>) {
    %done = nexus.submit_context.async
        @indexed_memory_context(%DATA, %OUT, %GATHER_IDX, %SCATTER_IDX)
        : !nexus.event<"indexed_memory_done">
    nexus.await %done
    nexus.return
  }
}
