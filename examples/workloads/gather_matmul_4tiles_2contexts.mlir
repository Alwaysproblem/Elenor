builtin.module {
  tile.program @gather_matmul_4tile(
      %task : !nest.task,
      %table : !nest.global_view<8388608xi8>,
      %lhs_l2 : !nest.l2_buffer<4x128x64xbf16>,
      %rhs_l2 : !nest.l2_buffer<4x64x128xbf16>,
      %indices_l2 : !nest.l2_buffer<4x16xi32>,
      %output_l2 : !nest.l2_buffer<4x128x128xbf16>) {
    %lhs_view = tile.subview %lhs_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 128, 64] strides = [1, 1, 1]
        : !nest.l2_view<1x128x64xbf16>
    %rhs_view = tile.subview %rhs_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 128] strides = [1, 1, 1]
        : !nest.l2_view<1x64x128xbf16>
    %indices_view = tile.subview %indices_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 16] strides = [1, 1]
        : !nest.l2_view<1x16xi32>
    %output_view = tile.subview %output_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1]
        : !nest.l2_view<1x128x128xbf16>
    %lhs_l1 = tile.alloc shape = [128, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %rhs_l1 = tile.alloc shape = [64, 128] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x128xbf16>
    %indices_l1 = tile.alloc shape = [16] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<16xi32>
    %gather_dst = tile.alloc shape = [256] dtype = "i8"
        alignment = 64 : !tile.l1_buffer<256xi8>
    %matmul_dst = tile.alloc shape = [128, 128] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %lhs_ready = tile.load.async %lhs_view into %lhs_l1
        : !tile.event<"lhs_ready">
    %rhs_ready = tile.load.async %rhs_view into %rhs_l1
        : !tile.event<"rhs_ready">
    %indices_ready = tile.load.async %indices_view into %indices_l1
        : !tile.event<"indices_ready">
    tile.await %lhs_ready, %rhs_ready, %indices_ready
    tile.signal input_released(%task)

    %gather_done = tile.gather.global.async %table
        indices(%indices_l1) into %gather_dst
        result_bytes = 256 cache_min_bytes = 16384
        cache_target_bytes = 65536 l1_mshr_hint = 16 {
      tile.profiled.access id = "r0" outcome = "HBM_MISS"
          bytes = 64 line = "shared_slow_line"
      tile.profiled.access id = "r1" outcome = "L1_HIT"
          bytes = 64 line = "shared_hot_line"
      tile.profiled.access id = "r2" outcome = "L2_HIT"
          bytes = 64 line = "shared_warm_line"
      tile.profiled.access id = "r3" outcome = "HBM_MISS"
          bytes = 64 line = "shared_cold_line"
    } : !tile.event<"gather_done">
    tile.await %gather_done

    %matmul_done = tile.boa.async "matmul"
        m = 128 n = 128 k = 64 ops = 2097152
        : !tile.event<"matmul_done">
    tile.await %matmul_done
    %l2_store_done = tile.store.async %matmul_dst into %output_view
        : !tile.event<"l2_store_done">
    tile.await %l2_store_done
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @gather_matmul_ctx0(
      %lhs : !nest.global_memref<4x128x64xbf16>,
      %rhs : !nest.global_memref<4x64x128xbf16>,
      %table : !nest.global_memref<8388608xi8>,
      %indices : !nest.global_memref<4x16xi32>,
      %output : !nest.global_memref<4x128x128xbf16>)
      placement = 15 context = 0 {
    %lhs_global = nest.subview %lhs
        offsets = [0, 0, 0] sizes = [4, 128, 64] strides = [1, 1, 1]
        : !nest.global_view<4x128x64xbf16>
    %rhs_global = nest.subview %rhs
        offsets = [0, 0, 0] sizes = [4, 64, 128] strides = [1, 1, 1]
        : !nest.global_view<4x64x128xbf16>
    %table_global = nest.subview %table
        offsets = [0] sizes = [8388608] strides = [1]
        : !nest.global_view<8388608xi8>
    %indices_global = nest.subview %indices
        offsets = [0, 0] sizes = [4, 16] strides = [1, 1]
        : !nest.global_view<4x16xi32>
    %output_global = nest.subview %output
        offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
        : !nest.global_view<4x128x128xbf16>
    %lhs_buffer = nest.alloc slot = "gm_ctx0_lhs" role = "in"
        shape = [4, 128, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x128x64xbf16>
    %rhs_buffer = nest.alloc slot = "gm_ctx0_rhs" role = "in"
        shape = [4, 64, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x64x128xbf16>
    %indices_buffer = nest.alloc slot = "gm_ctx0_indices" role = "in"
        shape = [4, 16] dtype = "i32" alignment = 256
        : !nest.l2_buffer<4x16xi32>
    %output_buffer = nest.alloc slot = "gm_ctx0_output" role = "out"
        shape = [4, 128, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x128x128xbf16>
    %lhs_prefetched = nest.dma.prefetch.async %lhs_global into %lhs_buffer
        : !nest.event<"lhs_prefetched">
    %rhs_prefetched = nest.dma.prefetch.async %rhs_global into %rhs_buffer
        : !nest.event<"rhs_prefetched">
    %indices_prefetched = nest.dma.prefetch.async %indices_global into %indices_buffer
        : !nest.event<"indices_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @gather_matmul_4tile context = 0
        tasks(%tasks) globals(%table_global)
        ins(%lhs_buffer, %rhs_buffer, %indices_buffer, %output_buffer)
        outs(%lhs_buffer, %rhs_buffer, %indices_buffer, %output_buffer)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%lhs_prefetched, %rhs_prefetched, %indices_prefetched)
        : (!nest.event<"grid_done">, !nest.event<"input_released">,
           !nest.event<"output_ready">)
    nest.release %lhs_buffer depends_on(%input_released)
    nest.release %rhs_buffer depends_on(%input_released)
    nest.release %indices_buffer depends_on(%input_released)
    %hbm_store_done = nest.dma.store.async %output_buffer into %output_global
        depends_on(%output_ready) : !nest.event<"hbm_store_done">
    nest.release %output_buffer depends_on(%hbm_store_done)
    nest.await %grid_done, %hbm_store_done
    nest.return
  }

  nest.context @gather_matmul_ctx1(
      %lhs : !nest.global_memref<4x128x64xbf16>,
      %rhs : !nest.global_memref<4x64x128xbf16>,
      %table : !nest.global_memref<8388608xi8>,
      %indices : !nest.global_memref<4x16xi32>,
      %output : !nest.global_memref<4x128x128xbf16>)
      placement = 15 context = 1 {
    %lhs_global = nest.subview %lhs
        offsets = [0, 0, 0] sizes = [4, 128, 64] strides = [1, 1, 1]
        : !nest.global_view<4x128x64xbf16>
    %rhs_global = nest.subview %rhs
        offsets = [0, 0, 0] sizes = [4, 64, 128] strides = [1, 1, 1]
        : !nest.global_view<4x64x128xbf16>
    %table_global = nest.subview %table
        offsets = [0] sizes = [8388608] strides = [1]
        : !nest.global_view<8388608xi8>
    %indices_global = nest.subview %indices
        offsets = [0, 0] sizes = [4, 16] strides = [1, 1]
        : !nest.global_view<4x16xi32>
    %output_global = nest.subview %output
        offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
        : !nest.global_view<4x128x128xbf16>
    %lhs_buffer = nest.alloc slot = "gm_ctx1_lhs" role = "in"
        shape = [4, 128, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x128x64xbf16>
    %rhs_buffer = nest.alloc slot = "gm_ctx1_rhs" role = "in"
        shape = [4, 64, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x64x128xbf16>
    %indices_buffer = nest.alloc slot = "gm_ctx1_indices" role = "in"
        shape = [4, 16] dtype = "i32" alignment = 256
        : !nest.l2_buffer<4x16xi32>
    %output_buffer = nest.alloc slot = "gm_ctx1_output" role = "out"
        shape = [4, 128, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x128x128xbf16>
    %lhs_prefetched = nest.dma.prefetch.async %lhs_global into %lhs_buffer
        : !nest.event<"lhs_prefetched">
    %rhs_prefetched = nest.dma.prefetch.async %rhs_global into %rhs_buffer
        : !nest.event<"rhs_prefetched">
    %indices_prefetched = nest.dma.prefetch.async %indices_global into %indices_buffer
        : !nest.event<"indices_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @gather_matmul_4tile context = 1
        tasks(%tasks) globals(%table_global)
        ins(%lhs_buffer, %rhs_buffer, %indices_buffer, %output_buffer)
        outs(%lhs_buffer, %rhs_buffer, %indices_buffer, %output_buffer)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%lhs_prefetched, %rhs_prefetched, %indices_prefetched)
        : (!nest.event<"grid_done">, !nest.event<"input_released">,
           !nest.event<"output_ready">)
    nest.release %lhs_buffer depends_on(%input_released)
    nest.release %rhs_buffer depends_on(%input_released)
    nest.release %indices_buffer depends_on(%input_released)
    %hbm_store_done = nest.dma.store.async %output_buffer into %output_global
        depends_on(%output_ready) : !nest.event<"hbm_store_done">
    nest.release %output_buffer depends_on(%hbm_store_done)
    nest.await %grid_done, %hbm_store_done
    nest.return
  }

  nexus.program @run_gather_matmul_4tiles_2contexts(
      %table : !nest.global_memref<8388608xi8>,
      %lhs0 : !nest.global_memref<4x128x64xbf16>,
      %rhs0 : !nest.global_memref<4x64x128xbf16>,
      %indices0 : !nest.global_memref<4x16xi32>,
      %output0 : !nest.global_memref<4x128x128xbf16>,
      %lhs1 : !nest.global_memref<4x128x64xbf16>,
      %rhs1 : !nest.global_memref<4x64x128xbf16>,
      %indices1 : !nest.global_memref<4x16xi32>,
      %output1 : !nest.global_memref<4x128x128xbf16>) {
    %done0 = nexus.submit_context.async
        @gather_matmul_ctx0(%lhs0, %rhs0, %table, %indices0, %output0)
        : !nexus.event<"gather_matmul_ctx0_done">
    %done1 = nexus.submit_context.async
        @gather_matmul_ctx1(%lhs1, %rhs1, %table, %indices1, %output1)
        : !nexus.event<"gather_matmul_ctx1_done">
    nexus.await %done0, %done1
    nexus.return
  }
}
