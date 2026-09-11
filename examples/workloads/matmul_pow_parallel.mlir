// 调度验证 1：matmul（BOA）与 pow（EVU）无 data 依赖，可以完全并行。
//
// 结构（基于 matmul_2048x512x64_boa256x256x32.mlir 改造）：
//   - matmul 缩为 M=1024 半边：2 个 nest.context（@mm_m0n0 / @mm_m0n1，
//     n_sup 网格 1x2），各 4 task、placement = 15，dispatch 固定 UCE
//     context 0/1，device slot pin 0/1。
//   - 另外 2 个 nest.context（@pow_y0 / @pow_y1）对独立输入 Y0 / Y1 做
//     EVU pow（每个 context 4 task x 2 个 128x128 chunk，placement = 15，
//     UCE context 2/3，device slot pin 2/3）。
//   - pow 的输入 Y 与 matmul 的 A/B/C 完全不相交：无 data 依赖。
//   - nexus.program 里 4 个 context 连续 submit、末尾统一 await：
//     4 个 device slot 从 t=0 起全部并发。
//
// 验证点（trace）：
//   - EVU:pow 的时间窗与 BOA:matmul 的时间窗重叠（并行执行）；
//   - 4 个 Slot:0..3 的 context:run 生命周期互相重叠；
//   - pow context 不等待任何 matmul context 完成。
//
// 运行：bash examples/run.sh matmul-pow-parallel
builtin.module {

  // ---- matmul tile program：与 matmul_2048x512x64 基线完全相同 ----
  // 每 task 一个 256x256 输出块，K=64 在 tile 内展开为 2 x BOA 256x256x32。
  tile.program @mm_ktiled_256x256x64(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x2x256x32xbf16>,
      %b_l2 : !nest.l2_buffer<2x32x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x256x256xbf16>) {
    %a_k0 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %a_k1 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %b_k0 = tile.subview %b_l2
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xbf16>
    %b_k1 = tile.subview %b_l2
        offsets = [1, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xbf16>
    %c_tile = tile.subview %c_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 256, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x256x256xbf16>
    %a_k0_l1 = tile.alloc shape = [256, 32] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<256x32xbf16>
    %a_k1_l1 = tile.alloc shape = [256, 32] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<256x32xbf16>
    %b_k0_l1 = tile.alloc shape = [32, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x256xbf16>
    %b_k1_l1 = tile.alloc shape = [32, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x256xbf16>
    %acc = tile.alloc shape = [256, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<256x256xbf16>
    %a_k0_loaded = tile.load.async %a_k0 into %a_k0_l1
        : !tile.event<"a_k0_loaded">
    %b_k0_loaded = tile.load.async %b_k0 into %b_k0_l1
        : !tile.event<"b_k0_loaded">
    %a_k1_loaded = tile.load.async %a_k1 into %a_k1_l1
        : !tile.event<"a_k1_loaded">
    %b_k1_loaded = tile.load.async %b_k1 into %b_k1_l1
        : !tile.event<"b_k1_loaded">
    tile.await %a_k0_loaded, %b_k0_loaded
    %boa_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa_k0_done">
    tile.await %a_k1_loaded, %b_k1_loaded
    tile.signal input_released(%task)
    tile.await %boa_k0_done
    %boa_k1_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304 accumulate
        : !tile.event<"boa_k1_done">
    tile.await %boa_k1_done
    %c_stored = tile.store.async %acc into %c_tile
        : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- pow tile program：每 task 2 个 128x128 chunk，load → pow → store x2 ----
  tile.program @pow_y_pair_128(
      %task : !nest.task,
      %y_l2 : !nest.l2_buffer<4x2x128x128xbf16>) {
    %y0 = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 128, 128] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x128x128xbf16>
    %y1 = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 128, 128] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x128x128xbf16>
    %l1 = tile.alloc shape = [128, 128] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<128x128xbf16>
    %y0_loaded = tile.load.async %y0 into %l1
        : !tile.event<"y0_loaded">
    tile.await %y0_loaded
    %y0_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536
        : !tile.event<"y0_pow">
    tile.await %y0_pow
    %y0_stored = tile.store.async %l1 into %y0
        : !tile.event<"y0_stored">
    tile.await %y0_stored
    %y1_loaded = tile.load.async %y1 into %l1
        : !tile.event<"y1_loaded">
    tile.await %y1_loaded
    tile.signal input_released(%task)
    %y1_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536
        : !tile.event<"y1_pow">
    tile.await %y1_pow
    %y1_stored = tile.store.async %l1 into %y1
        : !tile.event<"y1_stored">
    tile.await %y1_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- matmul contexts：M=1024 半边，n_sup 网格 1x2，placement = 15 ----
  nest.context @mm_m0n0(
      %A : !nest.global_memref<1x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<1x2x4x256x256xbf16>)
      placement = 15 context = 0 {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0, 0] sizes = [1, 4, 2, 256, 32] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0, 0] sizes = [1, 2, 32, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x1x4x256x256xbf16>
    %a_l2 = nest.alloc slot = "mmp_m0n0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mmp_m0n0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mmp_m0n0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mmp_m0n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mmp_m0n0_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @mm_ktiled_256x256x64 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c_l2) ins(%a_l2, %b_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mmp_m0n0_grid_done">, !nest.event<"mmp_m0n0_inrel">,
           !nest.event<"mmp_m0n0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mmp_m0n0_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_m0n1(
      %A : !nest.global_memref<1x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<1x2x4x256x256xbf16>)
      placement = 15 context = 1 {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0, 0] sizes = [1, 4, 2, 256, 32] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [1, 0, 0, 0] sizes = [1, 2, 32, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x1x4x256x256xbf16>
    %a_l2 = nest.alloc slot = "mmp_m0n1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mmp_m0n1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mmp_m0n1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mmp_m0n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mmp_m0n1_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @mm_ktiled_256x256x64 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c_l2) ins(%a_l2, %b_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mmp_m0n1_grid_done">, !nest.event<"mmp_m0n1_inrel">,
           !nest.event<"mmp_m0n1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mmp_m0n1_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // ---- pow contexts：独立输入 Y0/Y1，无 data 依赖，与 matmul 并发 ----
  nest.context @pow_y0(
      %Y0 : !nest.global_memref<4x2x128x128xbf16>)
      placement = 15 context = 2 {
    %y_blk = nest.subview %Y0
        offsets = [0, 0, 0, 0] sizes = [4, 2, 128, 128] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x128x128xbf16>
    %y_l2 = nest.alloc slot = "mmp_pow_y0" role = "inout"
        shape = [4, 2, 128, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x128x128xbf16>
    %y_prefetched = nest.dma.prefetch.async %y_blk into %y_l2
        : !nest.event<"mmp_pow_y0_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @pow_y_pair_128 context = 2
        tasks(%tasks) globals()
        bindings(%y_l2) ins(%y_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%y_prefetched)
        : (!nest.event<"mmp_pow_y0_grid_done">, !nest.event<"mmp_pow_y0_inrel">,
           !nest.event<"mmp_pow_y0_out_ready">)
    %y_store_done = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%output_ready) : !nest.event<"mmp_pow_y0_store_done">
    nest.release %y_l2 depends_on(%input_released, %y_prefetched, %y_store_done)
    nest.await %grid_done, %y_store_done
    nest.return
  }

  nest.context @pow_y1(
      %Y1 : !nest.global_memref<4x2x128x128xbf16>)
      placement = 15 context = 3 {
    %y_blk = nest.subview %Y1
        offsets = [0, 0, 0, 0] sizes = [4, 2, 128, 128] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x128x128xbf16>
    %y_l2 = nest.alloc slot = "mmp_pow_y1" role = "inout"
        shape = [4, 2, 128, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x128x128xbf16>
    %y_prefetched = nest.dma.prefetch.async %y_blk into %y_l2
        : !nest.event<"mmp_pow_y1_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @pow_y_pair_128 context = 3
        tasks(%tasks) globals()
        bindings(%y_l2) ins(%y_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%y_prefetched)
        : (!nest.event<"mmp_pow_y1_grid_done">, !nest.event<"mmp_pow_y1_inrel">,
           !nest.event<"mmp_pow_y1_out_ready">)
    %y_store_done = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%output_ready) : !nest.event<"mmp_pow_y1_store_done">
    nest.release %y_l2 depends_on(%input_released, %y_prefetched, %y_store_done)
    nest.await %grid_done, %y_store_done
    nest.return
  }

  // 4 个 context 连续 submit：2 个 matmul slot + 2 个 pow slot 从 t=0 并发。
  nexus.program @matmul_pow_parallel(
      %A : !nest.global_memref<1x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<1x2x4x256x256xbf16>,
      %Y0 : !nest.global_memref<4x2x128x128xbf16>,
      %Y1 : !nest.global_memref<4x2x128x128xbf16>) {
    %mm_n0_done = nexus.submit_context.async @mm_m0n0(%A, %B, %C)
        : !nexus.event<"mm_m0n0_done">
    %mm_n1_done = nexus.submit_context.async @mm_m0n1(%A, %B, %C)
        : !nexus.event<"mm_m0n1_done">
    %pow_y0_done = nexus.submit_context.async @pow_y0(%Y0)
        : !nexus.event<"pow_y0_done">
    %pow_y1_done = nexus.submit_context.async @pow_y1(%Y1)
        : !nexus.event<"pow_y1_done">
    nexus.await %mm_n0_done, %mm_n1_done, %pow_y0_done, %pow_y1_done
    nexus.return
  }
}
