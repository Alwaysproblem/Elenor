// 2048x512x64 bf16 Matmul，按 BOA tile shape 256x256x32 切分，无循环全展开：
//
//   BOA 块: 256x256x32；K: 64 = 2 x 32 步，全部在单个 tile program 内静态展开
//   输出块: (2048/256) x (512/256) = 8 x 2 = 16 个 256x256 块
//     = 4 个 nest.context 超块 (m_sup x n_sup = 2 x 2) x 每 context 4 个 task
//   M/N 维度 tiling 跨 context：context 网格 2x2（@mm_m0n0..@mm_m1n1）
//   每个 context 的 task.range 0..4 沿 M 把超块再切成 4 x 256 行
//   （placement = 15 强制 task 数 = popcount = 4，单 task_dim 只能沿一维细分）
//
// 每个 nest.context 占用一个 device slot（context = 0..3）并 dispatch 到
// 全组 4 tiles（placement = 15），UCE context pin = slot 0..3：每 tile 4 个
// 硬件 context 并行承载 4 个 slot 的 task（--context-mode 4，超出 V1.x
// 每 tile 2 context 上限，作为 validator what-if 探索，见 config.py 注释）。
//
// HBM 全局输入使用 block-packed（预切）布局，tiling 维都在前导维上，
// 所有 subview / DMA 均为连续 row-major 区间（validator V1 物理约束）：
//
//   A_tiled [2, 4, 2, 256, 32]  = (m_sup, m_task, k_step, BM, BK)
//   B_tiled [2, 2, 32, 256]     = (n_sup, k_step, BK, BN)
//   C_tiled [2, 2, 4, 256, 256] = (m_sup, n_sup, m_task, BM, BN)
//
// K tiling 完全发生在 tile 内：两个 K-step 各占独立 L2/L1 buffer，
// 展开为 2 条 tile.boa.async（m=256 n=256 k=32），第二条 accumulate 进同一累加器。
//
// 运行：bash examples/run.sh matmul-2048x512-boa256
builtin.module {

  // SPMD tile program：4 个 context 共用同一份程序；task 区分超块内的
  // 4 个 M 行块（task_dim = 0），B 块在同一 context 内被 4 个 task 共享。
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
    // 两个 K-step 的输入 load 一次性发射，让 MFE load 与 BOA compute 流水重叠。
    %a_k0_loaded = tile.load.async %a_k0 into %a_k0_l1
        : !tile.event<"a_k0_loaded">
    %b_k0_loaded = tile.load.async %b_k0 into %b_k0_l1
        : !tile.event<"b_k0_loaded">
    %a_k1_loaded = tile.load.async %a_k1 into %a_k1_l1
        : !tile.event<"a_k1_loaded">
    %b_k1_loaded = tile.load.async %b_k1 into %b_k1_l1
        : !tile.event<"b_k1_loaded">
    tile.await %a_k0_loaded, %b_k0_loaded
    // K-step 0：BOA 256x256x32，首步覆写累加器（无 accumulate）。
    %boa_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa_k0_done">
    tile.await %a_k1_loaded, %b_k1_loaded
    tile.signal input_released(%task)
    tile.await %boa_k0_done
    // K-step 1：BOA 256x256x32 accumulate，结果累加进 acc。
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

  // ---- M/N tiling：context 网格 2x2（m_sup, n_sup），全部 placement = 15 ----
  // context (m_sup, n_sup) 拥有输出超块 C[m_sup, n_sup]（4 x 256 行 x 256 列），
  // 由 4 个 task（m_task 0..3，1:1 映射到 4 tiles）各算一个 256x256 块。

  nest.context @mm_m0n0(
      %A : !nest.global_memref<2x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
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
    %a_l2 = nest.alloc slot = "mm_m0n0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mm_m0n0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mm_m0n0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_m0n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_m0n0_b_prefetched">
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
        : (!nest.event<"mm_m0n0_grid_done">, !nest.event<"mm_m0n0_inrel">,
           !nest.event<"mm_m0n0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mm_m0n0_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_m0n1(
      %A : !nest.global_memref<2x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
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
    %a_l2 = nest.alloc slot = "mm_m0n1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mm_m0n1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mm_m0n1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_m0n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_m0n1_b_prefetched">
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
        : (!nest.event<"mm_m0n1_grid_done">, !nest.event<"mm_m0n1_inrel">,
           !nest.event<"mm_m0n1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mm_m0n1_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_m1n0(
      %A : !nest.global_memref<2x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
      placement = 15 context = 2 {
    %a_blk = nest.subview %A
        offsets = [1, 0, 0, 0, 0] sizes = [1, 4, 2, 256, 32] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0, 0] sizes = [1, 2, 32, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x1x4x256x256xbf16>
    %a_l2 = nest.alloc slot = "mm_m1n0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mm_m1n0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mm_m1n0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_m1n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_m1n0_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @mm_ktiled_256x256x64 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c_l2) ins(%a_l2, %b_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_m1n0_grid_done">, !nest.event<"mm_m1n0_inrel">,
           !nest.event<"mm_m1n0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mm_m1n0_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_m1n1(
      %A : !nest.global_memref<2x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
      placement = 15 context = 3 {
    %a_blk = nest.subview %A
        offsets = [1, 0, 0, 0, 0] sizes = [1, 4, 2, 256, 32] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [1, 0, 0, 0] sizes = [1, 2, 32, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x1x4x256x256xbf16>
    %a_l2 = nest.alloc slot = "mm_m1n1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mm_m1n1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mm_m1n1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_m1n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_m1n1_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @mm_ktiled_256x256x64 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c_l2) ins(%a_l2, %b_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_m1n1_grid_done">, !nest.event<"mm_m1n1_inrel">,
           !nest.event<"mm_m1n1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mm_m1n1_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // 4 个 context 连续 submit（无中间 await），4 个 device slot 并发；
  // 每个 slot 的 4-task grid 铺满全组 4 tiles（placement = 15），
  // 每 tile 用 UCE context 0..3 分别承载 4 个 slot 的 task。
  nexus.program @matmul_2048x512x64(
      %A : !nest.global_memref<2x4x2x256x32xbf16>,
      %B : !nest.global_memref<2x2x32x256xbf16>,
      %C : !nest.global_memref<2x2x4x256x256xbf16>) {
    %m0n0_done = nexus.submit_context.async @mm_m0n0(%A, %B, %C)
        : !nexus.event<"mm_m0n0_done">
    %m0n1_done = nexus.submit_context.async @mm_m0n1(%A, %B, %C)
        : !nexus.event<"mm_m0n1_done">
    %m1n0_done = nexus.submit_context.async @mm_m1n0(%A, %B, %C)
        : !nexus.event<"mm_m1n0_done">
    %m1n1_done = nexus.submit_context.async @mm_m1n1(%A, %B, %C)
        : !nexus.event<"mm_m1n1_done">
    nexus.await %m0n0_done, %m0n1_done, %m1n0_done, %m1n1_done
    nexus.return
  }
}
