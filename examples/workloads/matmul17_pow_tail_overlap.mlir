// 调度验证 4：tiling 需要的 tile context 数不是 16 的倍数（本例 17），
// 以及 placement != 15 的 context 独占单 tile 时，剩余 tile context 能否
// 提前喂给 data 依赖的 pow 算子。
//
// matmul 为 M=5120（20 x 256 行）、N=256、K=64，输出 20 个 256x256 块，
// 按 17 个 tile context 切分（16 + 1，不是 16 的倍数）：
//   - 4 个 placement = 15 的 context（@mm_b0..@mm_b3，slot/UCE pin 0..3），
//     各 4 task → 16 个 tile context，每 task 1 块，覆盖 C 块 0..15；
//   - 1 个 placement = 1 的 context（@mm_tail，slot/UCE pin 4），
//     1 task → 只占 tile 0 的 1 个 tile context（第 17 个），单 task 内
//     顺序展开剩余 4 块（C 块 16..19，8 个 BOA 步）。
//   - 调度上 5 个 context 分开派发、并发在飞：满 placement 的 4 个铺满
//     Tile0..3，尾 context 只落 Tile0（--context-mode 5 /
//     --device-context-mode 5，tile 0 上 5 个 UCE context 并存）。
//
// pow（data 依赖，消费 matmul 输出 C 的前 16 块）：
//   - @pow_np_lo 消费 C 块 0..7（生产者 @mm_b0 + @mm_b1），
//     @pow_np_hi 消费 C 块 8..15（生产者 @mm_b2 + @mm_b3）；
//   - 各 placement = 15、4 task、每 task 2 个 256x256 chunk；
//   - 4 个满 placement context（含 C 写回）先于尾 context 完成：当
//     @pow_np_lo 启动时，@mm_tail 仍在 tile 0 的 UCE context 4 上算
//     剩余块 —— 此时 tile 0 的其余 UCE context 与 tile 1..3 全部空闲，
//     pow 用 placement = 15 抢占这些资源提前运行，不等 @mm_tail。
//
// 验证点（trace）：
//   - 5 个 matmul context 从 t=0 并发在飞；@mm_tail 的所有 engine 活动
//     只出现在 Tile0（placement = 1），其余 context 铺满 Tile0..3；
//   - @pow_np_lo 的 EVU:pow 时间窗与 @mm_tail 的运行窗口重叠
//     （tail 未结束时 pow 已提前运行）；
//   - pow 的首个活动晚于其生产者（@mm_b0/@mm_b1 等）完成时间
//     （data 依赖正确，不提前读 C）。
//
// 运行：bash examples/run.sh matmul17-pow-tail-overlap
builtin.module {

  // ---- 满 placement matmul tile program：每 task 一个 256x256 块 ----
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

  // ---- 尾 context tile program：单 task 顺序展开 4 个 256x256 块 ----
  // （placement = 1 → 只有 1 个 task；tiling 余量在 task 内部串行处理，
  //   这就是第 17 个 tile context。L1 buffer 跨块复用。）
  tile.program @mm_tail_4blk_256x256x64(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x2x256x32xbf16>,
      %b_l2 : !nest.l2_buffer<2x32x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x256x256xbf16>) {
    %b_k0 = tile.subview %b_l2
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xbf16>
    %b_k1 = tile.subview %b_l2
        offsets = [1, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xbf16>
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
    %b_k0_loaded = tile.load.async %b_k0 into %b_k0_l1
        : !tile.event<"b_k0_loaded">
    %b_k1_loaded = tile.load.async %b_k1 into %b_k1_l1
        : !tile.event<"b_k1_loaded">

    // ---- 块 0 ----
    %a0_k0 = tile.subview %a_l2
        offsets = [0, 0, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %a0_k1 = tile.subview %a_l2
        offsets = [0, 1, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %c0_tile = tile.subview %c_l2
        offsets = [0, 0, 0] sizes = [1, 256, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x256x256xbf16>
    %a0_k0_loaded = tile.load.async %a0_k0 into %a_k0_l1
        : !tile.event<"a0_k0_loaded">
    %a0_k1_loaded = tile.load.async %a0_k1 into %a_k1_l1
        : !tile.event<"a0_k1_loaded">
    tile.await %a0_k0_loaded, %b_k0_loaded
    %boa0_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa0_k0_done">
    tile.await %a0_k1_loaded, %b_k1_loaded
    tile.await %boa0_k0_done
    %boa0_k1_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304 accumulate
        : !tile.event<"boa0_k1_done">
    tile.await %boa0_k1_done
    %c0_stored = tile.store.async %acc into %c0_tile
        : !tile.event<"c0_stored">
    tile.await %c0_stored

    // ---- 块 1 ----
    %a1_k0 = tile.subview %a_l2
        offsets = [1, 0, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %a1_k1 = tile.subview %a_l2
        offsets = [1, 1, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %c1_tile = tile.subview %c_l2
        offsets = [1, 0, 0] sizes = [1, 256, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x256x256xbf16>
    %a1_k0_loaded = tile.load.async %a1_k0 into %a_k0_l1
        : !tile.event<"a1_k0_loaded">
    %a1_k1_loaded = tile.load.async %a1_k1 into %a_k1_l1
        : !tile.event<"a1_k1_loaded">
    tile.await %a1_k0_loaded
    %boa1_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa1_k0_done">
    tile.await %a1_k1_loaded
    tile.await %boa1_k0_done
    %boa1_k1_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304 accumulate
        : !tile.event<"boa1_k1_done">
    tile.await %boa1_k1_done
    %c1_stored = tile.store.async %acc into %c1_tile
        : !tile.event<"c1_stored">
    tile.await %c1_stored

    // ---- 块 2 ----
    %a2_k0 = tile.subview %a_l2
        offsets = [2, 0, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %a2_k1 = tile.subview %a_l2
        offsets = [2, 1, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %c2_tile = tile.subview %c_l2
        offsets = [2, 0, 0] sizes = [1, 256, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x256x256xbf16>
    %a2_k0_loaded = tile.load.async %a2_k0 into %a_k0_l1
        : !tile.event<"a2_k0_loaded">
    %a2_k1_loaded = tile.load.async %a2_k1 into %a_k1_l1
        : !tile.event<"a2_k1_loaded">
    tile.await %a2_k0_loaded
    %boa2_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa2_k0_done">
    tile.await %a2_k1_loaded
    tile.await %boa2_k0_done
    %boa2_k1_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304 accumulate
        : !tile.event<"boa2_k1_done">
    tile.await %boa2_k1_done
    %c2_stored = tile.store.async %acc into %c2_tile
        : !tile.event<"c2_stored">
    tile.await %c2_stored

    // ---- 块 3 ----
    %a3_k0 = tile.subview %a_l2
        offsets = [3, 0, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %a3_k1 = tile.subview %a_l2
        offsets = [3, 1, 0, 0] sizes = [1, 1, 256, 32] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x256x32xbf16>
    %c3_tile = tile.subview %c_l2
        offsets = [3, 0, 0] sizes = [1, 256, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x256x256xbf16>
    %a3_k0_loaded = tile.load.async %a3_k0 into %a_k0_l1
        : !tile.event<"a3_k0_loaded">
    %a3_k1_loaded = tile.load.async %a3_k1 into %a_k1_l1
        : !tile.event<"a3_k1_loaded">
    tile.await %a3_k0_loaded
    %boa3_k0_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304
        : !tile.event<"boa3_k0_done">
    tile.await %a3_k1_loaded
    tile.signal input_released(%task)
    tile.await %boa3_k0_done
    %boa3_k1_done = tile.boa.async "matmul"
        m = 256 n = 256 k = 32 ops = 4194304 accumulate
        : !tile.event<"boa3_k1_done">
    tile.await %boa3_k1_done
    %c3_stored = tile.store.async %acc into %c3_tile
        : !tile.event<"c3_stored">
    tile.await %c3_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- pow tile program：每 task 2 个 256x256 chunk，load → pow → store x2 ----
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

  // ---- 4 个满 placement matmul context：C 块 0..15（16 个 tile context） ----
  nest.context @mm_b0(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 0 {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0] sizes = [4, 2, 256, 32] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [2, 32, 256] strides = [1, 1, 1]
        : !nest.global_view<2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [0, 0, 0] sizes = [4, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<4x256x256xbf16>
    %a_l2 = nest.alloc slot = "m17_b0_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "m17_b0_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "m17_b0_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"m17_b0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"m17_b0_b_prefetched">
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
        : (!nest.event<"m17_b0_grid_done">, !nest.event<"m17_b0_inrel">,
           !nest.event<"m17_b0_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_b0_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_b1(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 1 {
    %a_blk = nest.subview %A
        offsets = [4, 0, 0, 0] sizes = [4, 2, 256, 32] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [2, 32, 256] strides = [1, 1, 1]
        : !nest.global_view<2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [4, 0, 0] sizes = [4, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<4x256x256xbf16>
    %a_l2 = nest.alloc slot = "m17_b1_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "m17_b1_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "m17_b1_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"m17_b1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"m17_b1_b_prefetched">
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
        : (!nest.event<"m17_b1_grid_done">, !nest.event<"m17_b1_inrel">,
           !nest.event<"m17_b1_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_b1_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_b2(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 2 {
    %a_blk = nest.subview %A
        offsets = [8, 0, 0, 0] sizes = [4, 2, 256, 32] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [2, 32, 256] strides = [1, 1, 1]
        : !nest.global_view<2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [8, 0, 0] sizes = [4, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<4x256x256xbf16>
    %a_l2 = nest.alloc slot = "m17_b2_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "m17_b2_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "m17_b2_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"m17_b2_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"m17_b2_b_prefetched">
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
        : (!nest.event<"m17_b2_grid_done">, !nest.event<"m17_b2_inrel">,
           !nest.event<"m17_b2_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_b2_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @mm_b3(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 3 {
    %a_blk = nest.subview %A
        offsets = [12, 0, 0, 0] sizes = [4, 2, 256, 32] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [2, 32, 256] strides = [1, 1, 1]
        : !nest.global_view<2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [12, 0, 0] sizes = [4, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<4x256x256xbf16>
    %a_l2 = nest.alloc slot = "m17_b3_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "m17_b3_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "m17_b3_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"m17_b3_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"m17_b3_b_prefetched">
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
        : (!nest.event<"m17_b3_grid_done">, !nest.event<"m17_b3_inrel">,
           !nest.event<"m17_b3_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_b3_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // ---- 尾 context：placement = 1（只用 tile 0 的 1 个 tile context），
  // ---- slot/UCE pin 4，1 task 内串行算完剩余 4 块（C 块 16..19）。
  nest.context @mm_tail(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 1 context = 4 {
    %a_blk = nest.subview %A
        offsets = [16, 0, 0, 0] sizes = [4, 2, 256, 32] strides = [1, 1, 1, 1]
        : !nest.global_view<4x2x256x32xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [2, 32, 256] strides = [1, 1, 1]
        : !nest.global_view<2x32x256xbf16>
    %c_blk = nest.subview %C
        offsets = [16, 0, 0] sizes = [4, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<4x256x256xbf16>
    %a_l2 = nest.alloc slot = "m17_tail_a" role = "in"
        shape = [4, 2, 256, 32] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x2x256x32xbf16>
    %b_l2 = nest.alloc slot = "m17_tail_b" role = "in"
        shape = [2, 32, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x32x256xbf16>
    %c_l2 = nest.alloc slot = "m17_tail_c" role = "out"
        shape = [4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x256x256xbf16>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"m17_tail_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"m17_tail_b_prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid_done, %input_released, %output_ready =
        nest.dispatch.tasks.async @mm_tail_4blk_256x256x64 context = 4
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c_l2) ins(%a_l2, %b_l2) outs(%c_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"m17_tail_grid_done">, !nest.event<"m17_tail_inrel">,
           !nest.event<"m17_tail_out_ready">)
    nest.release %a_l2 depends_on(%input_released, %a_prefetched)
    nest.release %b_l2 depends_on(%input_released, %b_prefetched)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_tail_c_store_done">
    nest.release %c_l2 depends_on(%c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // ---- pow contexts：data 依赖于前 16 块的 matmul 输出 C ----
  // @pow_np_lo 消费块 0..7（等 @mm_b0 + @mm_b1），pin slot 0；
  // @pow_np_hi 消费块 8..15（等 @mm_b2 + @mm_b3），pin slot 1。
  // 两个 pow 都不等 @mm_tail：当它们启动时 @mm_tail 通常仍在 tile 0 的
  // UCE context 4 上算剩余块，pow 抢占其余空闲 tile context 提前运行。
  nest.context @pow_np_lo(
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 0 {
    %c_blk = nest.subview %C
        offsets = [0, 0, 0] sizes = [8, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<8x256x256xbf16>
    %c_l2 = nest.alloc slot = "m17_pow_lo" role = "inout"
        shape = [2, 4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x4x256x256xbf16>
    %c_prefetched = nest.dma.prefetch.async %c_blk into %c_l2
        : !nest.event<"m17_pow_lo_prefetched">
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
        : (!nest.event<"m17_pow_lo_grid_done">, !nest.event<"m17_pow_lo_inrel">,
           !nest.event<"m17_pow_lo_out_ready">)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_pow_lo_store_done">
    nest.release %c_l2 depends_on(%input_released, %c_prefetched, %c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  nest.context @pow_np_hi(
      %C : !nest.global_memref<20x256x256xbf16>)
      placement = 15 context = 1 {
    %c_blk = nest.subview %C
        offsets = [8, 0, 0] sizes = [8, 256, 256] strides = [1, 1, 1]
        : !nest.global_view<8x256x256xbf16>
    %c_l2 = nest.alloc slot = "m17_pow_hi" role = "inout"
        shape = [2, 4, 256, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<2x4x256x256xbf16>
    %c_prefetched = nest.dma.prefetch.async %c_blk into %c_l2
        : !nest.event<"m17_pow_hi_prefetched">
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
        : (!nest.event<"m17_pow_hi_grid_done">, !nest.event<"m17_pow_hi_inrel">,
           !nest.event<"m17_pow_hi_out_ready">)
    %c_store_done = nest.dma.store.async %c_l2 into %c_blk
        depends_on(%output_ready) : !nest.event<"m17_pow_hi_store_done">
    nest.release %c_l2 depends_on(%input_released, %c_prefetched, %c_store_done)
    nest.await %grid_done, %c_store_done
    nest.return
  }

  // 5 个 matmul context 先连续 submit（17 个 tile context = 4 x 4 + 1），
  // 随后按 data 依赖 await 对应生产者再 submit pow：@pow_np_lo 只等
  // @mm_b0/@mm_b1，@pow_np_hi 只等 @mm_b2/@mm_b3 —— 二者均不等 @mm_tail。
  nexus.program @matmul17_pow_tail_overlap(
      %A : !nest.global_memref<20x2x256x32xbf16>,
      %B : !nest.global_memref<2x32x256xbf16>,
      %C : !nest.global_memref<20x256x256xbf16>) {
    %b0_done = nexus.submit_context.async @mm_b0(%A, %B, %C)
        : !nexus.event<"mm_b0_done">
    %b1_done = nexus.submit_context.async @mm_b1(%A, %B, %C)
        : !nexus.event<"mm_b1_done">
    %b2_done = nexus.submit_context.async @mm_b2(%A, %B, %C)
        : !nexus.event<"mm_b2_done">
    %b3_done = nexus.submit_context.async @mm_b3(%A, %B, %C)
        : !nexus.event<"mm_b3_done">
    %tail_done = nexus.submit_context.async @mm_tail(%A, %B, %C)
        : !nexus.event<"mm_tail_done">
    nexus.await %b0_done, %b1_done
    %pow_lo_done = nexus.submit_context.async @pow_np_lo(%C)
        : !nexus.event<"pow_np_lo_done">
    nexus.await %b2_done, %b3_done
    %pow_hi_done = nexus.submit_context.async @pow_np_hi(%C)
        : !nexus.event<"pow_np_hi_done">
    nexus.await %tail_done, %pow_lo_done, %pow_hi_done
    nexus.return
  }
}
