// 调度验证 3：pow 算子对 matmul 输出 C 有 data 依赖 —— pow 的输入就是
// matmul 写回 HBM 的 C。此时 pow context 必须等它的数据生产者（产出对应
// C 半边的 matmul context）完成后才能启动，不能提前调度；但不必等与它
// 无关的 matmul context。
//
// 结构（基于 matmul_2048x512x64_boa256x256x32.mlir 改造）：
//   - 4 个 matmul nest.context（@mm_m0n0..@mm_m1n1，与基线完全相同）：
//     各 4 task、placement = 15、UCE context pin 0..3、device slot pin 0..3。
//   - 2 个 pow nest.context（@pow_np_c0 / @pow_np_c1）：placement = 15，
//     各 4 task，每 task 2 个 256x256 chunk（load → pow → store x2）。
//     @pow_np_c0 消费 C[m_sup=0] 半边（由 @mm_m0n0 + @mm_m0n1 产出），
//     @pow_np_c1 消费 C[m_sup=1] 半边（由 @mm_m1n0 + @mm_m1n1 产出）。
//   - data 依赖通过 nexus.await 表达（跨 context 的生产者-消费者顺序，
//     C 为 :rw binding：matmul 写、pow 读后再原地写回）。
//
// 验证点（trace）：
//   - @pow_np_c0 的首个 engine 活动 > max(@mm_m0n0, @mm_m0n1 完成时间)
//     （绝不早于生产者）；@pow_np_c1 同理对 m1 行；
//   - @pow_np_c0 只等 m0 行，可与仍在运行的 m1 行 matmul 重叠
//     （依赖感知：不串行化无关工作）；
//   - 与 matmul_pow_free_slot.mlir（无依赖版）对比：本例 pow 启动被
//     生产者完成时间约束，而非 slot 空闲时间。
//
// 运行：bash examples/run.sh matmul-pow-data-dep
builtin.module {

  // ---- matmul tile program：与基线完全相同 ----
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
    tile.free %a_k0_l1
    tile.free %a_k1_l1
    tile.free %b_k0_l1
    tile.free %b_k1_l1
    %c_stored = tile.store.async %acc into %c_tile
        : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- pow tile program：每 task 2 个 256x256 chunk，load → pow → store x2，
  // ---- 作用在 matmul 输出 C 上（l2 视图 [2,4,256,256] = C 的一个 m_sup 半边，
  // ---- task_dim = 1 按 m_task 取行，两个 subview 分别取 n_sup 0/1 块）。
  tile.program @pow_np_pair_256(
      %task : !nest.task,
      %c_l2 : !nest.l2_buffer<2x4x256x256xbf16>) {
    %np0 = tile.subview %c_l2 task = %task task_dim = 1
        offsets = [0, 0, 0, 0] sizes = [1, 1, 256, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x256xbf16>
    %np1 = tile.subview %c_l2 task = %task task_dim = 1
        offsets = [1, 0, 0, 0] sizes = [1, 1, 256, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x256xbf16>
    %l1 = tile.alloc shape = [256, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<256x256xbf16>
    %np0_loaded = tile.load.async %np0 into %l1
        : !tile.event<"np0_loaded">
    tile.await %np0_loaded
    %np0_pow = tile.pow.async bytes = 131072 exponent = 2 pow_ops = 262144
        : !tile.event<"np0_pow">
    tile.await %np0_pow
    %np0_stored = tile.store.async %l1 into %np0
        : !tile.event<"np0_stored">
    tile.await %np0_stored
    %np1_loaded = tile.load.async %np1 into %l1
        : !tile.event<"np1_loaded">
    tile.await %np1_loaded
    tile.signal input_released(%task)
    %np1_pow = tile.pow.async bytes = 131072 exponent = 2 pow_ops = 262144
        : !tile.event<"np1_pow">
    tile.await %np1_pow
    %np1_stored = tile.store.async %l1 into %np1
        : !tile.event<"np1_stored">
    tile.await %np1_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 4 个 matmul context：与基线 matmul_2048x512x64 完全一致 ----
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
    %a_l2 = nest.alloc slot = "mpd_m0n0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mpd_m0n0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_m0n0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mpd_m0n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mpd_m0n0_b_prefetched">
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
        : (!nest.event<"mpd_m0n0_grid_done">, !nest.event<"mpd_m0n0_inrel">,
           !nest.event<"mpd_m0n0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_m0n0_c_store_done">
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
    %a_l2 = nest.alloc slot = "mpd_m0n1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mpd_m0n1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_m0n1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mpd_m0n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mpd_m0n1_b_prefetched">
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
        : (!nest.event<"mpd_m0n1_grid_done">, !nest.event<"mpd_m0n1_inrel">,
           !nest.event<"mpd_m0n1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_m0n1_c_store_done">
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
    %a_l2 = nest.alloc slot = "mpd_m1n0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mpd_m1n0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_m1n0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mpd_m1n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mpd_m1n0_b_prefetched">
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
        : (!nest.event<"mpd_m1n0_grid_done">, !nest.event<"mpd_m1n0_inrel">,
           !nest.event<"mpd_m1n0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_m1n0_c_store_done">
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
    %a_l2 = nest.alloc slot = "mpd_m1n1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "mpd_m1n1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_m1n1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mpd_m1n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mpd_m1n1_b_prefetched">
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
        : (!nest.event<"mpd_m1n1_grid_done">, !nest.event<"mpd_m1n1_inrel">,
           !nest.event<"mpd_m1n1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_m1n1_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // ---- pow contexts：消费 matmul 输出 C 的对应半边（data 依赖） ----
  // pin 到生产者释放的 slot：@pow_np_c0 → slot 0（m0 行完成后空闲），
  // @pow_np_c1 → slot 1（m1 行完成时 slot 1 早已空闲）。
  nest.context @pow_np_c0(
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
      placement = 15 context = 0 {
    %c_blk = nest.subview %C
        offsets = [0, 0, 0, 0, 0] sizes = [1, 2, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x2x4x256x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_pow_c0" role = "inout"
        shape = [2, 4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x4x256x256xbf16>
    %c_prefetched = nest.dma.prefetch.async %c_blk into %c_l2
        : !nest.event<"mpd_pow_c0_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @pow_np_pair_256 context = 0
        tasks(%tasks) globals()
        bindings(%c_l2) ins(%c_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%c_prefetched)
        : (!nest.event<"mpd_pow_c0_grid_done">, !nest.event<"mpd_pow_c0_inrel">,
           !nest.event<"mpd_pow_c0_out_ready">)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_pow_c0_store_done">
    nest.release %c_l2 depends_on(%input_released, %c_prefetched, %c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @pow_np_c1(
      %C : !nest.global_memref<2x2x4x256x256xbf16>)
      placement = 15 context = 1 {
    %c_blk = nest.subview %C
        offsets = [1, 0, 0, 0, 0] sizes = [1, 2, 4, 256, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x2x4x256x256xbf16>
    %c_l2 = nest.alloc slot = "mpd_pow_c1" role = "inout"
        shape = [2, 4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x4x256x256xbf16>
    %c_prefetched = nest.dma.prefetch.async %c_blk into %c_l2
        : !nest.event<"mpd_pow_c1_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @pow_np_pair_256 context = 1
        tasks(%tasks) globals()
        bindings(%c_l2) ins(%c_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%c_prefetched)
        : (!nest.event<"mpd_pow_c1_grid_done">, !nest.event<"mpd_pow_c1_inrel">,
           !nest.event<"mpd_pow_c1_out_ready">)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"mpd_pow_c1_store_done">
    nest.release %c_l2 depends_on(%input_released, %c_prefetched, %c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // data 依赖链：pow 半边只 await 产出它的两个 matmul context。
  // @pow_np_c0 在 m0 行完成后即可启动（可与仍在运行的 m1 行重叠）；
  // @pow_np_c1 在 m1 行完成后启动。
  nexus.program @matmul_pow_data_dep(
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
    nexus.await %m0n0_done, %m0n1_done
    %pow_c0_done = nexus.submit_context.async @pow_np_c0(%C)
        : !nexus.event<"pow_np_c0_done">
    nexus.await %m1n0_done, %m1n1_done
    %pow_c1_done = nexus.submit_context.async @pow_np_c1(%C)
        : !nexus.event<"pow_np_c1_done">
    nexus.await %pow_c0_done, %pow_c1_done
    nexus.return
  }
}
