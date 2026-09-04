builtin.module {
  tile.program @gather_tile(
      %task : !nest.task,
      %table : !nest.global_view<8388608xi8>,
      %indices_l2 : !nest.l2_buffer<1024xi32>,
      %output_l2 : !nest.l2_buffer<1x256xi8>) {
    %indices_view = tile.subview %indices_l2
        offsets = [0] sizes = [16] strides = [1]
        : !nest.l2_view<16xi32>
    %output_view = tile.subview %output_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 256] strides = [1, 1]
        : !nest.l2_view<1x256xi8>
    %indices_l1 = tile.alloc shape = [16] dtype = "i32"
        : !tile.l1_buffer<16xi32>
    %gather_dst = tile.alloc shape = [256] dtype = "i8"
        : !tile.l1_buffer<256xi8>
    %indices_ready = tile.load.async %indices_view into %indices_l1
        : !tile.event<"indices_ready">
    tile.await %indices_ready
    tile.signal input_released(%task)
    %gather_done = tile.gather.global.async %table
        indices(%indices_l1) into %gather_dst
        result_bytes = 256 cache_min_bytes = 16384
        cache_target_bytes = 65536 l1_mshr_hint = 16 {
      tile.profiled.access id = "r0" outcome = "L1_HIT"
          bytes = 64 line = "line0"
      tile.profiled.access id = "r1" outcome = "L2_HIT"
          bytes = 64 line = "line1"
      tile.profiled.access id = "r2" outcome = "HBM_MISS"
          bytes = 64 line = "line42" merge = "line42"
      tile.profiled.access id = "r3" outcome = "HBM_MISS"
          bytes = 64 line = "line42" merge = "line42"
    } : !tile.event<"gather_done">
    tile.await %gather_done
    %l2_store_done = tile.store.async %gather_dst into %output_view
        : !tile.event<"l2_store_done">
    tile.await %l2_store_done
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @gather_context(
      %table : !nest.global_memref<8388608xi8>,
      %indices : !nest.global_memref<1024xi32>,
      %output : !nest.global_memref<1x256xi8>) placement = 1 {
    %table_view = nest.subview %table offsets = [0] sizes = [8388608]
        strides = [1] : !nest.global_view<8388608xi8>
    %indices_view = nest.subview %indices offsets = [0] sizes = [1024]
        strides = [1] : !nest.global_view<1024xi32>
    %output_global = nest.subview %output
        offsets = [0, 0] sizes = [1, 256] strides = [1, 1]
        : !nest.global_view<1x256xi8>
    %indices_l2 = nest.alloc slot = "gather_indices" role = "in"
        shape = [1024] dtype = "i32" alignment = 256
        : !nest.l2_buffer<1024xi32>
    %output_l2 = nest.alloc slot = "gather_output" role = "out"
        shape = [1, 256] dtype = "i8" alignment = 256
        : !nest.l2_buffer<1x256xi8>
    %indices_prefetched = nest.dma.prefetch.async %indices_view into %indices_l2
        : !nest.event<"indices_prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @gather_tile
        tasks(%tasks) globals(%table_view)
        ins(%indices_l2, %output_l2) outs(%indices_l2, %output_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%indices_prefetched)
        : (!nest.event<"grid_done">, !nest.event<"input_released">,
           !nest.event<"output_ready">)
    nest.release %indices_l2 depends_on(%input_released)
    %hbm_store_done = nest.dma.store.async %output_l2 into %output_global
        depends_on(%output_ready) : !nest.event<"hbm_store_done">
    nest.release %output_l2 depends_on(%hbm_store_done)
    nest.await %grid_done, %hbm_store_done
    nest.return
  }

  nexus.program @run_gather(
      %table : !nest.global_memref<8388608xi8>,
      %indices : !nest.global_memref<1024xi32>,
      %output : !nest.global_memref<1x256xi8>) {
    %done = nexus.submit_context.async
        @gather_context(%table, %indices, %output)
        : !nexus.event<"gather_context_done">
    nexus.await %done
    nexus.return
  }
}
