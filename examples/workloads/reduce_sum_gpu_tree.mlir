// 256x4096 bf16 Reduce-Sum（沿 reduce axis K 规约），GPU reduce-tree 算法
// 优化的 MLIR 建模：4 个 "block"（UCE context dispatch）各规约一个 K 配对
// 分区，partial 再按二叉树两级合并——对应 CUDA 两段式规约（block 内
// 部分和 + 跨 block 树状合并）：
//
//   总问题:  Y[256] = sum_k X[256, 4096]            X = 2 MiB bf16
//
//   第 1 层 (leaf, reduce-axis tiling):  X_tiled [4, 8, 64, 512] =
//     (m_task, k_step, R, KC)。4 个 leaf dispatch（placement = 15，每 dispatch
//     4 task 按 m_task 分工到 4 tiles；UCE pin = 0..3，即每 tile 4 个 UCE
//     context 并发，--context-mode 4）各负责一个 K 配对分区：leaf d 规约
//     k_step {2d, 2d+1} 两块 chunk（每 task 2 x 64 KiB，ops = 65536 + 64 =
//     65600），产出该行块的 K-对 partial p_d[64]。8 个 k_step 恰被 4 个
//     leaf 覆盖一次。
//
//   第 2 层 (跨 dispatch 合并树, CUDA 两段式 global handoff):  leaf 把
//     partial p_d[4, 64] 经 HBM store 写入 scratch S[d]（对应 GPU 把 block
//     partial 写回 global memory）；合并 dispatch 在 store 完成后从 scratch
//     prefetch 读回——C0 = S[0]+S[1]（pin 0）、C1 = S[2]+S[3]（pin 2）、
//     Y = C0+C1（pin 0），每 task 合并自己行块的 [64] 切片
//     （ops = 2 x 64 + 64 = 192），最后单条 HBM store 写 Y。
//     （ELENOR 优化方向：in-context L2 驻留可省去这次 HBM 往返，本例
//     按 GPU 原始算法建模。）
//
//   X_tiled [4, 8, 64, 512] = (m_task, k_step, R, KC)     2 MiB bf16
//   Y       [4, 64]         = (m_task, R)                 1 KiB f32
//
// 数值边界：本 validator 是时间模型。`tile.evu.async` 不携带 buffer operand、
// 不执行任何数值运算、也没有 accumulate 语义；pair/树合并的命名只是算法
// 结构标注。输入 chunk 为 bf16，partial 与输出为 f32（BF16 reduction 使用
// FP32 accumulate，见 ELENOR_EVU_Design），仅为类型意图，无数值检查。本例
// 验证树形调度的依赖结构、搬运字节数、生命周期与时序，不证明 tensor 数值
// 正确性。
//
// 验证点（trace / report）：
//   - EVU 切片 28 = 4 leaf x 4 task x 1 pair（ops = 65600）
//     + 3 个合并 dispatch x 4 task（ops = 192）；
//   - MFE_LD0 切片 56 = 4 leaf x 4 task x 2 chunk（65536 B）
//     + 3 个合并 x 4 task x 2（256 B f32）；
//   - 依赖树不出界：C0 只等 leaf0/1（经 scratch store->prefetch），C1 只等
//     leaf2/3，final 只等 C0/C1；leaf 4 路并发（4 个 UCE context / tile）；
//   - HBM->L2 prefetch 7 条（X 2097152 B + 6 x 1 KiB scratch 读回）；
//     L2->HBM store 7 条（6 x 1 KiB partial/中间结果 + 1 KiB Y）；
//   - 单 nest.context 常驻，active_context_peak = 1。
//
// 运行：bash examples/run.sh reduce-sum-gpu-tree
builtin.module {

  // leaf d：本行块对 k_step {2d, 2d+1} 的配对规约（4 个静态变体）。
  // ---- leaf 变体 0（k_step 0+1）----
  tile.program @rs_leaf_p0(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x512xbf16>,
      %p_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 135168> {
    %xa = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %xb = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %p_v = tile.subview %p_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %buf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %buf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %xa_l = tile.load.async %xa into %buf0 : !tile.event<"xa_l">
    %xb_l = tile.load.async %xb into %buf1 : !tile.event<"xb_l">
    tile.await %xa_l, %xb_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %buf0
    tile.free %buf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 1（k_step 2+3）----
  tile.program @rs_leaf_p1(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x512xbf16>,
      %p_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 135168> {
    %xa = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 2, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %xb = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 3, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %p_v = tile.subview %p_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %buf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %buf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %xa_l = tile.load.async %xa into %buf0 : !tile.event<"xa_l">
    %xb_l = tile.load.async %xb into %buf1 : !tile.event<"xb_l">
    tile.await %xa_l, %xb_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %buf0
    tile.free %buf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 2（k_step 4+5）----
  tile.program @rs_leaf_p2(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x512xbf16>,
      %p_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 135168> {
    %xa = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 4, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %xb = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 5, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %p_v = tile.subview %p_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %buf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %buf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %xa_l = tile.load.async %xa into %buf0 : !tile.event<"xa_l">
    %xb_l = tile.load.async %xb into %buf1 : !tile.event<"xb_l">
    tile.await %xa_l, %xb_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %buf0
    tile.free %buf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 3（k_step 6+7）----
  tile.program @rs_leaf_p3(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x512xbf16>,
      %p_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 135168> {
    %xa = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 6, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %xb = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 7, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %p_v = tile.subview %p_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %buf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %buf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %xa_l = tile.load.async %xa into %buf0 : !tile.event<"xa_l">
    %xb_l = tile.load.async %xb into %buf1 : !tile.event<"xb_l">
    tile.await %xa_l, %xb_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %buf0
    tile.free %buf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // 跨 dispatch 合并节点：两份 [4, 64] f32 partial 按行块逐 [1, 64] 合并
  // （树的同一层形态，两级复用同一程序）。
  tile.program @rs_pair_reduce(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x64xf32>,
      %b_l2 : !nest.l2_buffer<4x64xf32>,
      %o_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 4096> {
    %a_v = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %b_v = tile.subview %b_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %o_v = tile.subview %o_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %pa = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %pb = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %a_l = tile.load.async %a_v into %pa : !tile.event<"a_l">
    %b_l = tile.load.async %b_v into %pb : !tile.event<"b_l">
    tile.await %a_l, %b_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 192 : !tile.event<"cc">
    tile.await %cc
    tile.free %pa
    tile.free %pb
    %o_stored = tile.store.async %acc into %o_v : !tile.event<"o_stored">
    tile.await %o_stored
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @rs_tree_ctx(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>,
      %S : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 28, l2_spm_bytes = 2113536,
          requested_contexts_per_tile = 4> {
    %x_blk = nest.subview %X
        offsets = [0, 0, 0, 0] sizes = [4, 8, 64, 512] strides = [1, 1, 1, 1]
        : !nest.global_view<4x8x64x512xbf16>
    %y_blk = nest.subview %Y
        offsets = [0, 0] sizes = [4, 64] strides = [1, 1]
        : !nest.global_view<4x64xf32>
    %s0_blk = nest.subview %S
        offsets = [0, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %s1_blk = nest.subview %S
        offsets = [1, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %s2_blk = nest.subview %S
        offsets = [2, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %s3_blk = nest.subview %S
        offsets = [3, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %s4_blk = nest.subview %S
        offsets = [4, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %s5_blk = nest.subview %S
        offsets = [5, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_tree_x" role = "in"
        shape = [4, 8, 64, 512] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x512xbf16>
    %p0_l2 = nest.alloc slot = "rs_tree_p0" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p1_l2 = nest.alloc slot = "rs_tree_p1" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p2_l2 = nest.alloc slot = "rs_tree_p2" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p3_l2 = nest.alloc slot = "rs_tree_p3" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %q0a_l2 = nest.alloc slot = "rs_tree_q0a" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %q0b_l2 = nest.alloc slot = "rs_tree_q0b" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %q1a_l2 = nest.alloc slot = "rs_tree_q1a" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %q1b_l2 = nest.alloc slot = "rs_tree_q1b" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %c0_l2 = nest.alloc slot = "rs_tree_c0" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %c1_l2 = nest.alloc slot = "rs_tree_c1" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %f0_l2 = nest.alloc slot = "rs_tree_f0" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %f1_l2 = nest.alloc slot = "rs_tree_f1" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %y_l2 = nest.alloc slot = "rs_tree_y" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    // 一次性把整个 X 装进 L2。
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_tree_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    // ---- 第 1 层：4 个 leaf 并发（K 配对分区，UCE pin 0..3）----
    %grid_l0, %read_l0, %ready_l0 =
        nest.dispatch.tasks.async @rs_leaf_p0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %p0_l2) ins(%x_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_tree_grid_l0">, !nest.event<"rs_tree_read_l0">,
           !nest.event<"rs_tree_ready_l0">)
    %grid_l1, %read_l1, %ready_l1 =
        nest.dispatch.tasks.async @rs_leaf_p1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%x_l2, %p1_l2) ins(%x_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_tree_grid_l1">, !nest.event<"rs_tree_read_l1">,
           !nest.event<"rs_tree_ready_l1">)
    %grid_l2, %read_l2, %ready_l2 =
        nest.dispatch.tasks.async @rs_leaf_p2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%x_l2, %p2_l2) ins(%x_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_tree_grid_l2">, !nest.event<"rs_tree_read_l2">,
           !nest.event<"rs_tree_ready_l2">)
    %grid_l3, %read_l3, %ready_l3 =
        nest.dispatch.tasks.async @rs_leaf_p3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%x_l2, %p3_l2) ins(%x_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_tree_grid_l3">, !nest.event<"rs_tree_read_l3">,
           !nest.event<"rs_tree_ready_l3">)
    // ---- 第 1.5 层：leaf partial 写回 scratch（对应 GPU 把 block partial
    //      写回 global memory）----
    %store_p0 = nest.dma.store.async %p0_l2 into %s0_blk
        depends_on(%ready_l0) : !nest.event<"rs_tree_store_p0">
    %store_p1 = nest.dma.store.async %p1_l2 into %s1_blk
        depends_on(%ready_l1) : !nest.event<"rs_tree_store_p1">
    %store_p2 = nest.dma.store.async %p2_l2 into %s2_blk
        depends_on(%ready_l2) : !nest.event<"rs_tree_store_p2">
    %store_p3 = nest.dma.store.async %p3_l2 into %s3_blk
        depends_on(%ready_l3) : !nest.event<"rs_tree_store_p3">
    // ---- 第 2 层：两两合并（store -> scratch prefetch -> dispatch，CUDA
    //      两段式 global handoff；依赖树不出界：C0 只等 leaf0/1，C1 只等
    //      leaf2/3）----
    %pref_c0a = nest.dma.prefetch.async %s0_blk into %q0a_l2
        depends_on(%store_p0) : !nest.event<"rs_tree_pref_c0a">
    %pref_c0b = nest.dma.prefetch.async %s1_blk into %q0b_l2
        depends_on(%store_p1) : !nest.event<"rs_tree_pref_c0b">
    %pref_c1a = nest.dma.prefetch.async %s2_blk into %q1a_l2
        depends_on(%store_p2) : !nest.event<"rs_tree_pref_c1a">
    %pref_c1b = nest.dma.prefetch.async %s3_blk into %q1b_l2
        depends_on(%store_p3) : !nest.event<"rs_tree_pref_c1b">
    %grid_c0, %read_c0, %ready_c0 =
        nest.dispatch.tasks.async @rs_pair_reduce l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%q0a_l2, %q0b_l2, %c0_l2) ins(%q0a_l2, %q0b_l2) outs(%c0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%pref_c0a, %pref_c0b)
        : (!nest.event<"rs_tree_grid_c0">, !nest.event<"rs_tree_read_c0">,
           !nest.event<"rs_tree_ready_c0">)
    %grid_c1, %read_c1, %ready_c1 =
        nest.dispatch.tasks.async @rs_pair_reduce l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%q1a_l2, %q1b_l2, %c1_l2) ins(%q1a_l2, %q1b_l2) outs(%c1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%pref_c1a, %pref_c1b)
        : (!nest.event<"rs_tree_grid_c1">, !nest.event<"rs_tree_read_c1">,
           !nest.event<"rs_tree_ready_c1">)
    // ---- 第 2.5 层：合并结果写回 scratch ----
    %store_c0 = nest.dma.store.async %c0_l2 into %s4_blk
        depends_on(%ready_c0) : !nest.event<"rs_tree_store_c0">
    %store_c1 = nest.dma.store.async %c1_l2 into %s5_blk
        depends_on(%ready_c1) : !nest.event<"rs_tree_store_c1">
    // ---- 第 3 层：根合并 -> 最终 Y ----
    %pref_fa = nest.dma.prefetch.async %s4_blk into %f0_l2
        depends_on(%store_c0) : !nest.event<"rs_tree_pref_fa">
    %pref_fb = nest.dma.prefetch.async %s5_blk into %f1_l2
        depends_on(%store_c1) : !nest.event<"rs_tree_pref_fb">
    %grid_f, %read_f, %ready_f =
        nest.dispatch.tasks.async @rs_pair_reduce l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%f0_l2, %f1_l2, %y_l2) ins(%f0_l2, %f1_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%pref_fa, %pref_fb)
        : (!nest.event<"rs_tree_grid_f">, !nest.event<"rs_tree_read_f">,
           !nest.event<"rs_tree_ready_f">)
    // ---- 生命周期：release 全部位于最后一次使用之后 ----
    nest.release %x_l2 depends_on(%read_l0, %read_l1, %read_l2, %read_l3,
                                  %x_prefetched)
    nest.release %p0_l2 depends_on(%store_p0)
    nest.release %p1_l2 depends_on(%store_p1)
    nest.release %p2_l2 depends_on(%store_p2)
    nest.release %p3_l2 depends_on(%store_p3)
    nest.release %q0a_l2 depends_on(%read_c0, %pref_c0a)
    nest.release %q0b_l2 depends_on(%read_c0, %pref_c0b)
    nest.release %q1a_l2 depends_on(%read_c1, %pref_c1a)
    nest.release %q1b_l2 depends_on(%read_c1, %pref_c1b)
    nest.release %c0_l2 depends_on(%store_c0)
    nest.release %c1_l2 depends_on(%store_c1)
    nest.release %f0_l2 depends_on(%read_f, %pref_fa)
    nest.release %f1_l2 depends_on(%read_f, %pref_fb)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_f) : !nest.event<"rs_tree_y_store_done">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_l0, %grid_l1, %grid_l2, %grid_l3
    nest.await %grid_c0, %grid_c1, %grid_f, %y_store
    nest.return
  }

  nexus.program @reduce_sum_gpu_tree(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>,
      %S : !nest.global_memref<8x4x64xf32>) {
    %done = nexus.submit_context.async @rs_tree_ctx(%X, %Y, %S)
        : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
