// 256x4096 bf16 Reduce-Sum（沿 reduce axis K 规约），一次 L2 驻留、
// 4 个 UCE hardware context 沿 REDUCE AXIS 切分且全部 4 tile 参与执行：
//
//   总问题:  Y[256] = sum_k X[256, 4096]            X = 2 MiB bf16
//   终点:    扁平 4 路合并后的最终 Y[4, 64] = Y[256]（partial 全程 L2 驻留，
//   经 context-local 直接交付，免 HBM 中转）
//
// 与 reduce_sum_multiuce 的差别：multiuce 的 4 个 UCE context 沿 m_task
// （非 reduce axis）分工，各规约完整 K=4096；本例的 4 个 UCE context 沿
// reduce axis 分工——dispatch d 规约 k_step 配对 {2d, 2d+1}（K = 8 x 512
// = 4096，每 context 覆盖 1024 列的全部 256 行），8 个 k_step 恰被覆盖
// 一次。X 整体只做一次 HBM->L2 prefetch（2 MiB 驻留 L2）。
//
//   X_tiled [4, 8, 64, 512] = (m_task, k_step, R, KC)     2 MiB bf16
//   Y       [4, 64]         = (m_task, R)                 1 KiB f32
//
// 跨 dispatch 交付使用 context-local L2：leaf partial 由 combine 在同一
// context 直接读取（ins 绑定同一 L2 buffer，等待 leaf output_ready），
// release 等待 writer output_ready 与 reader input_released。partial 不经
// HBM（对比 gpu-tree 的 scratch HBM 交接）。
//
// context placement = 15（4 个 tile 全部参与），4 个 leaf dispatch 的
// UCE context pin = 0..3：运行期每个 tile 同时驻留 4 个 UCE context
// （--context-mode 4, requested_contexts_per_tile = 4）。每个 dispatch
// 4 task（task_dim = 0）沿 m_task 切分行块，k_step 配对由 4 个静态程序
// 变体 @rs_sk_uce_q0..q3 的固定 view offsets 区分：变体 d 的 task t 读
// X[t, {2d, 2d+1}, :, :]（2 个 64 KiB chunk），pair-reduce 出 [64] f32
// partial 写入 p_d 的行切片（逻辑 payload 2 x 64 KiB chunk + 256 B acc =
// 132096 B；每 buffer 各占整 stripe round，Arena contract = 135168 B）。
// input_released 在最后一次 chunk load await 之后才发；输出 4 个独立
// [4, 64] f32 context-local L2 buffer，由 combine 同 context 直读。
//
// 数值边界：本 validator 是时间模型。`tile.evu.async` 不携带 buffer operand、
// 不执行任何数值运算、也没有 accumulate 语义；首步命名 `reduce_sum`、后续
// `reduce_sum_accumulate` 只是 IR 上的循环携带依赖意图。输入 chunk 为 bf16，
// 累加器与输出为 f32（BF16 reduction 使用 FP32 accumulate，见
// ELENOR_EVU_Design），仅为类型意图，无数值检查。本例验证 tiling 结构、
// 搬运字节数、依赖/生命周期与时序，不证明 tensor 数值正确性。
//
// 验证点（trace / report）：
//   - EVU 切片 20 = 4 leaf dispatch x 4 task x 1 pair-reduce（ops = 65600，
//     2 x 32768 输入 + 64 输出）+ 1 个合并 dispatch x 4 task（ops = 320，
//     4 x 64 输入 + 64 输出）；
//   - MFE_LD0 切片 48 = 4 leaf x 4 task x 2 chunk（65536 B）
//     + 合并 x 4 task x 4（256 B）；
//   - 运行期每个 tile 同时驻留 4 个 UCE context（4 dispatch x 1 task/tile）；
//     各 tile 的 EVU 为单服务队列，串行接受服务，4 个 tile 并行；
//   - HBM->L2 prefetch 1 条（X 2097152 B）；L2->HBM store 1 条（1 KiB Y）；
//   - 合并 dispatch depends_on 全部 4 个 leaf output_ready（partial 经
//     context-local L2 直接读取，释放等待全部读写完成）；
//   - 单 nest.context 常驻，active_context_peak = 1。
//
// 运行：bash examples/run.sh reduce-sum-splitk-multiuce
builtin.module {

  // 4 个静态变体：view 的 k_step 固定为 {2d, 2d+1}（K 配对分区）；
  // task_dim = 0 让每 dispatch 的 4 task 切分 4 个 m_task 行块。
  // ---- leaf 变体 0（k_step 0+1；task_dim 0 切 M 行块）----
  tile.program @rs_sk_uce_q0(
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
    %abuf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %abuf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %la = tile.load.async %xa into %abuf0 : !tile.event<"la">
    %lb = tile.load.async %xb into %abuf1 : !tile.event<"lb">
    tile.await %la, %lb
    // 本 task 只有两个 chunk load，await 后即 input_released。
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %abuf0
    tile.free %abuf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 1（k_step 2+3；task_dim 0 切 M 行块）----
  tile.program @rs_sk_uce_q1(
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
    %abuf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %abuf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %la = tile.load.async %xa into %abuf0 : !tile.event<"la">
    %lb = tile.load.async %xb into %abuf1 : !tile.event<"lb">
    tile.await %la, %lb
    // 本 task 只有两个 chunk load，await 后即 input_released。
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %abuf0
    tile.free %abuf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 2（k_step 4+5；task_dim 0 切 M 行块）----
  tile.program @rs_sk_uce_q2(
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
    %abuf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %abuf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %la = tile.load.async %xa into %abuf0 : !tile.event<"la">
    %lb = tile.load.async %xb into %abuf1 : !tile.event<"lb">
    tile.await %la, %lb
    // 本 task 只有两个 chunk load，await 后即 input_released。
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %abuf0
    tile.free %abuf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 变体 3（k_step 6+7；task_dim 0 切 M 行块）----
  tile.program @rs_sk_uce_q3(
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
    %abuf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %abuf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %la = tile.load.async %xa into %abuf0 : !tile.event<"la">
    %lb = tile.load.async %xb into %abuf1 : !tile.event<"lb">
    tile.await %la, %lb
    // 本 task 只有两个 chunk load，await 后即 input_released。
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 65600 : !tile.event<"cc">
    tile.await %cc
    tile.free %abuf0
    tile.free %abuf1
    %p_stored = tile.store.async %acc into %p_v : !tile.event<"p_stored">
    tile.await %p_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 扁平 4 路合并：context-local p0..p3 L2 partials -> 最终 Y[4, 64] ----
  // （单级合并，区别于 reduce_sum_gpu_tree 的二叉树两级合并。）
  tile.program @rs_sk_combine(
      %task : !nest.task,
      %q0_l2 : !nest.l2_buffer<4x64xf32>,
      %q1_l2 : !nest.l2_buffer<4x64xf32>,
      %q2_l2 : !nest.l2_buffer<4x64xf32>,
      %q3_l2 : !nest.l2_buffer<4x64xf32>,
      %y_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %q0_v = tile.subview %q0_l2 task = %task task_dim = 0 offsets = [0, 0] sizes = [1, 64]
      strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %q1_v = tile.subview %q1_l2 task = %task task_dim = 0 offsets = [0, 0] sizes = [1, 64]
      strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %q2_v = tile.subview %q2_l2 task = %task task_dim = 0 offsets = [0, 0] sizes = [1, 64]
      strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %q3_v = tile.subview %q3_l2 task = %task task_dim = 0 offsets = [0, 0] sizes = [1, 64]
      strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0 offsets = [0, 0] sizes = [1, 64]
      strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %qa = tile.alloc shape = [1, 64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<1x64xf32>
    %qb = tile.alloc shape = [1, 64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<1x64xf32>
    %qc = tile.alloc shape = [1, 64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<1x64xf32>
    %qd = tile.alloc shape = [1, 64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<1x64xf32>
    %acc = tile.alloc shape = [1, 64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<1x64xf32>
    %qa_l = tile.load.async %q0_v into %qa : !tile.event<"qa_l">
    %qb_l = tile.load.async %q1_v into %qb : !tile.event<"qb_l">
    %qc_l = tile.load.async %q2_v into %qc : !tile.event<"qc_l">
    %qd_l = tile.load.async %q3_v into %qd : !tile.event<"qd_l">
    tile.await %qa_l, %qb_l, %qc_l, %qd_l
    tile.signal input_released(%task)
    %cc = tile.evu.async "reduce_sum" ops = 320 : !tile.event<"cc">
    tile.await %cc
    tile.free %qa
    tile.free %qb
    tile.free %qc
    tile.free %qd
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @rs_sk_uce_ctx(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 2109440,
          requested_contexts_per_tile = 4> {
    %x_blk = nest.subview %X
        offsets = [0, 0, 0, 0] sizes = [4, 8, 64, 512] strides = [1, 1, 1, 1]
        : !nest.global_view<4x8x64x512xbf16>
    %y_blk = nest.subview %Y
        offsets = [0, 0] sizes = [4, 64] strides = [1, 1]
        : !nest.global_view<4x64xf32>
    %x_l2 = nest.alloc slot = "rs_sk_uce_x" role = "in"
        shape = [4, 8, 64, 512] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x512xbf16>
    %p0_l2 = nest.alloc slot = "rs_sk_uce_p0" role = "out" sharing = "context-local"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p1_l2 = nest.alloc slot = "rs_sk_uce_p1" role = "out" sharing = "context-local"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p2_l2 = nest.alloc slot = "rs_sk_uce_p2" role = "out" sharing = "context-local"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %p3_l2 = nest.alloc slot = "rs_sk_uce_p3" role = "out" sharing = "context-local"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %y_l2 = nest.alloc slot = "rs_sk_uce_y" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    // 一次性把整个 X 装进 L2，4 个 dispatch 只读共享。
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_sk_uce_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @rs_sk_uce_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %p0_l2) ins(%x_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_sk_uce_grid_0">, !nest.event<"rs_sk_uce_read_0">,
           !nest.event<"rs_sk_uce_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @rs_sk_uce_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%x_l2, %p1_l2) ins(%x_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_sk_uce_grid_1">, !nest.event<"rs_sk_uce_read_1">,
           !nest.event<"rs_sk_uce_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @rs_sk_uce_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%x_l2, %p2_l2) ins(%x_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_sk_uce_grid_2">, !nest.event<"rs_sk_uce_read_2">,
           !nest.event<"rs_sk_uce_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @rs_sk_uce_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%x_l2, %p3_l2) ins(%x_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_sk_uce_grid_3">, !nest.event<"rs_sk_uce_read_3">,
           !nest.event<"rs_sk_uce_ready_3">)
    // ---- 扁平 4 路合并：同 context 直读 leaf 的 context-local partial L2 ----
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @rs_sk_combine l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%p0_l2, %p1_l2, %p2_l2, %p3_l2, %y_l2)
        ins(%p0_l2, %p1_l2, %p2_l2, %p3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%ready_0, %ready_1, %ready_2, %ready_3)
        : (!nest.event<"rs_sk_uce_grid_c">, !nest.event<"rs_sk_uce_read_c">,
           !nest.event<"rs_sk_uce_ready_c">)
    nest.release %x_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %x_prefetched)
    // context-local scratch 无需 publish 或 HBM Store，释放等待 writer 和 reader。
    nest.release %p0_l2 depends_on(%read_c, %ready_0)
    nest.release %p1_l2 depends_on(%read_c, %ready_1)
    nest.release %p2_l2 depends_on(%read_c, %ready_2)
    nest.release %p3_l2 depends_on(%read_c, %ready_3)
    %store_y = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"rs_sk_uce_store_y">
    nest.release %y_l2 depends_on(%store_y)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %store_y
    nest.return
  }

  nexus.program @reduce_sum_splitk_multiuce(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>) {
    %done = nexus.submit_context.async @rs_sk_uce_ctx(%X, %Y)
        : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
