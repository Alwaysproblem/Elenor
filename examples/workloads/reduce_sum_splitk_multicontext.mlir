// 256x16384 bf16 Reduce-Sum（沿 reduce axis K 规约），单次规约的三级结构：
// 第 1/2 级是 reduce-axis tiling（跨 context K 分区 + task 内 K chunk），
// 第 3 级是跨 context 部分和合并。本 program 的 per-task L1 Arena（contract
// 69632 B）装不下任何 task 的行块，必须分块经 L2 流水：
//
//   总问题:  Y[256] = sum_k X[256, 16384]          X = 8 MiB bf16
//
//   第 1 级 (跨 context):  K = 16384 = 8 x 2048，按 reduce axis 切成
//     KP = 8 个 producer context（@rs_k0..@rs_k7），每个 context 拥有
//     X 的一个 256x2048 K-分区（1 MiB），产出一个部分和向量。
//     8 个 context 提交后由 device scheduler 按 4 个 device slot 轮转准入，
//     相邻 context 的 prefetch/compute/store 相互流水重叠
//     （full_memory 实测 active_context_peak = 4）；L2 峰值 ≤ 4 x ~1 MiB < 8 MiB。
//
//   第 2 级 (task 内, L1 分块):  每 context 4 task（placement = 15）按行分工
//     （m_task，非 reduce-axis）：单个 context 的 task 只承担 64 行 x 2048 列
//     = 256 KiB；同一 m_task 的切片跨 8 个 context 合计覆盖 64 行 x 16384 列
//     = 2 MiB 完整 K 行。行块远超 L1 Arena contract（逻辑 payload 65792 B = 双缓冲 2 x 32 KiB
//     chunk + 256 B f32 acc，含 stripe padding 后 contract 69632 B）。行块
//     再沿 K 切成 KS = 8 x KC = 256 的 chunk（32 KiB/chunk），双缓冲
//     （buf0/buf1 ping-pong）使 MFE load 与 EVU reduce 流水重叠；跨 chunk
//     的循环携带依赖是常驻 L1 的累加器 %acc[64]——每个 chunk 规约出 64 行
//     部分和并累加进 %acc。
//
//   第 3 级 (跨 context 合并):  各 producer 把 %acc 经 L2 store 写回 HBM
//     部分和 Y_part[8, 4, 64]；combine context（@rs_combine）在全部 8 个
//     producer 的 HBM store 完成后启动（nexus.await 顺序），每 task 把自己的
//     8 份 64 行部分和再次 reduce，得到最终 Y[256]。
//
// HBM 全局输入使用 block-packed（预切）布局，tiling 维都在前导维上，
// 所有 subview / DMA 均为连续 row-major 区间（validator V1 物理约束）：
// DMA 只做线性字节拷贝、不转置，因此 producer 写出的 [k_part, task, row]
// 切片由 combine 用 8 个独立 [4, 64] 输入逐份读回（见 @rs_final_combine）。
//
//   X_tiled  [8, 4, 8, 64, 256] = (k_part, m_task, k_step, R, KC)   8 MiB
//   Y_part   [8, 4, 64]         = (k_part, m_task, R)               8 KiB f32
//   Y        [4, 64]            = (m_task, R)                       1 KiB f32
//
// 数值边界：本 validator 是时间模型。`tile.evu.async` / `tile.boa.async`
// 不携带 buffer operand、不执行任何数值运算，也没有 accumulate 语义，
// `%acc` 的"累加"只是 IR 上的循环携带依赖意图。输入 chunk 为 bf16，
// 累加器与输出为 f32（BF16 reduction 使用 FP32 accumulate，见
// ELENOR_EVU_Design），仅为类型意图，无数值检查。本例验证 tiling 结构、
// 搬运字节数、依赖/生命周期与时序，不证明 tensor 数值正确性。
//
// 每个 producer task 的数据流（时间展开，无循环 IR）：
//
//   load x0->buf0, load x1->buf1          # 预发两个 chunk 的 L2->L1 load
//   await x0; reduce_sum(buf0)->acc        # 首 chunk 初始化累加器
//   s = 1..7:  await x_s; reduce_sum(buf_s)+=acc; 发射 x_{s+1} load
//              （load 与上一 step 的 EVU compute 重叠）
//   await x7 后才 signal input_released    # input_released 不早于最后一次真实 load
//   store acc -> Y_part[k_part, task]；HBM store 由 output_ready 门控
//
// 验证点（trace / report）：
//   - 8 个 producer 的 EVU "reduce_sum" 每 chunk ops = 16448
//     （16384 输入 + 64 输出，与 s11 reduce 节点同口径），每 task 8 次；
//   - producer 按 slot 轮转准入且相邻重叠（active_context_peak = 4），
//     每个 producer 的 input_released >= 最后一次 chunk load 完成；
//   - @rs_combine 的首个 engine 活动晚于全部 8 个 Y_part HBM store 完成；
//   - combine 每 task 一次 reduce_sum ops = 576（8 x 64 输入 + 64 输出），
//     8 份输入按 k_part 逐份 prefetch，无布局重排。
//
// 运行：bash examples/run.sh reduce-sum-splitk-multicontext
builtin.module {

  // ---- producer tile program：8 个 k-part context 共用；task 区分 4 个 64 行块 ----
  tile.program @rs_partial_sum(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x256xbf16>,
      %yp_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 69632> {
    // 本 task 的 8 个 K-chunk 视图（k_step 0..7）。
    %x0 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 2, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 3, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 4, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 5, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 6, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 7, 0, 0] sizes = [1, 1, 64, 256] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x256xbf16>
    %yp_v = tile.subview %yp_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    // L1：双缓冲 chunk（2 x 32 KiB）+ 常驻累加器（跨 k_step 循环携带）。
    %buf0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %buf1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    // 预发两个 chunk 的 load，让 MFE load 与首个 EVU reduce 重叠。
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    // k_step 0：reduce_sum 初始化累加器（16384 输入 + 64 输出）。
    %c0 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    // k_step 1：reduce_sum 累加进 acc；同时发射 k_step 2 的 load 与本步重叠。
    %c1 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    // 最后一次真实 load 完成后才允许 input_released。
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum" ops = 16448 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    // 部分和 [64] 写回本 context 的 Y_part 切片，再由 context 级 HBM store 落盘。
    %yp_stored = tile.store.async %acc into %yp_v : !tile.event<"yp_stored">
    tile.await %yp_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- combine tile program：8 份部分和 -> 最终 Y ----
  // 每份部分和是独立的 [4, 64] L2 输入（与 producer 写出布局逐字节一致；
  // DMA 只做连续拷贝、不转置，因此不能把 [k_part, task, row] 线性区间当作
  // [task, k_part, row] 读回）。task 维由 task_dim = 0 视图切出本 task 的 [1, 64]。
  tile.program @rs_final_combine(
      %task : !nest.task,
      %yp0_l2 : !nest.l2_buffer<4x64xf32>,
      %yp1_l2 : !nest.l2_buffer<4x64xf32>,
      %yp2_l2 : !nest.l2_buffer<4x64xf32>,
      %yp3_l2 : !nest.l2_buffer<4x64xf32>,
      %yp4_l2 : !nest.l2_buffer<4x64xf32>,
      %yp5_l2 : !nest.l2_buffer<4x64xf32>,
      %yp6_l2 : !nest.l2_buffer<4x64xf32>,
      %yp7_l2 : !nest.l2_buffer<4x64xf32>,
      %y_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 9216> {
    %yp0_v = tile.subview %yp0_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp1_v = tile.subview %yp1_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp2_v = tile.subview %yp2_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp3_v = tile.subview %yp3_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp4_v = tile.subview %yp4_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp5_v = tile.subview %yp5_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp6_v = tile.subview %yp6_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %yp7_v = tile.subview %yp7_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    %p0 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p1 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p2 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p3 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p4 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p5 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p6 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %p7 = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    %yp0_l = tile.load.async %yp0_v into %p0 : !tile.event<"yp0_l">
    %yp1_l = tile.load.async %yp1_v into %p1 : !tile.event<"yp1_l">
    %yp2_l = tile.load.async %yp2_v into %p2 : !tile.event<"yp2_l">
    %yp3_l = tile.load.async %yp3_v into %p3 : !tile.event<"yp3_l">
    %yp4_l = tile.load.async %yp4_v into %p4 : !tile.event<"yp4_l">
    %yp5_l = tile.load.async %yp5_v into %p5 : !tile.event<"yp5_l">
    %yp6_l = tile.load.async %yp6_v into %p6 : !tile.event<"yp6_l">
    %yp7_l = tile.load.async %yp7_v into %p7 : !tile.event<"yp7_l">
    tile.await %yp0_l, %yp1_l, %yp2_l, %yp3_l, %yp4_l, %yp5_l, %yp6_l, %yp7_l
    tile.signal input_released(%task)
    // 8 份 [64] 部分和 + 64 输出。
    %cc = tile.evu.async "reduce_sum" ops = 576 : !tile.event<"cc">
    tile.await %cc
    tile.free %p0
    tile.free %p1
    tile.free %p2
    tile.free %p3
    tile.free %p4
    tile.free %p5
    tile.free %p6
    tile.free %p7
    // 最终 [64] 行和写回本 task 的 Y 切片。
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 8 个 producer context（reduce axis 的 K-分区 0..7）----
  // 本 context 的 X 切片（256x2048 = 1 MiB）整体 HBM->L2 prefetch；task 内
  // 再按 k_step 分块进 L1。部分和 [4, 64] 是本 context 唯一输出。

  nest.context @rs_k0(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [0, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [0, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k0_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k0_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k0_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k0, %read_k0, %ready_k0 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k0_grid_done">, !nest.event<"rs_k0_inrel">,
           !nest.event<"rs_k0_out_ready">)
    nest.release %x_l2 depends_on(%read_k0, %x_prefetched)
    %yp_store_k0 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k0) : !nest.event<"rs_k0_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k0)
    nest.await %grid_k0, %yp_store_k0
    nest.return
  }

  nest.context @rs_k1(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [1, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [1, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k1_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k1_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k1_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k1, %read_k1, %ready_k1 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k1_grid_done">, !nest.event<"rs_k1_inrel">,
           !nest.event<"rs_k1_out_ready">)
    nest.release %x_l2 depends_on(%read_k1, %x_prefetched)
    %yp_store_k1 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k1) : !nest.event<"rs_k1_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k1)
    nest.await %grid_k1, %yp_store_k1
    nest.return
  }

  nest.context @rs_k2(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [2, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [2, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k2_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k2_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k2_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k2, %read_k2, %ready_k2 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k2_grid_done">, !nest.event<"rs_k2_inrel">,
           !nest.event<"rs_k2_out_ready">)
    nest.release %x_l2 depends_on(%read_k2, %x_prefetched)
    %yp_store_k2 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k2) : !nest.event<"rs_k2_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k2)
    nest.await %grid_k2, %yp_store_k2
    nest.return
  }

  nest.context @rs_k3(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [3, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [3, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k3_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k3_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k3_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k3, %read_k3, %ready_k3 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k3_grid_done">, !nest.event<"rs_k3_inrel">,
           !nest.event<"rs_k3_out_ready">)
    nest.release %x_l2 depends_on(%read_k3, %x_prefetched)
    %yp_store_k3 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k3) : !nest.event<"rs_k3_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k3)
    nest.await %grid_k3, %yp_store_k3
    nest.return
  }

  nest.context @rs_k4(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [4, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [4, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k4_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k4_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k4_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k4, %read_k4, %ready_k4 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k4_grid_done">, !nest.event<"rs_k4_inrel">,
           !nest.event<"rs_k4_out_ready">)
    nest.release %x_l2 depends_on(%read_k4, %x_prefetched)
    %yp_store_k4 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k4) : !nest.event<"rs_k4_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k4)
    nest.await %grid_k4, %yp_store_k4
    nest.return
  }

  nest.context @rs_k5(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [5, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [5, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k5_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k5_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k5_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k5, %read_k5, %ready_k5 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k5_grid_done">, !nest.event<"rs_k5_inrel">,
           !nest.event<"rs_k5_out_ready">)
    nest.release %x_l2 depends_on(%read_k5, %x_prefetched)
    %yp_store_k5 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k5) : !nest.event<"rs_k5_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k5)
    nest.await %grid_k5, %yp_store_k5
    nest.return
  }

  nest.context @rs_k6(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [6, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [6, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k6_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k6_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k6_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k6, %read_k6, %ready_k6 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k6_grid_done">, !nest.event<"rs_k6_inrel">,
           !nest.event<"rs_k6_out_ready">)
    nest.release %x_l2 depends_on(%read_k6, %x_prefetched)
    %yp_store_k6 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k6) : !nest.event<"rs_k6_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k6)
    nest.await %grid_k6, %yp_store_k6
    nest.return
  }

  nest.context @rs_k7(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 1052672,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [7, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 256] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x256xbf16>
    %yp_blk = nest.subview %Y_part
        offsets = [7, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %x_l2 = nest.alloc slot = "rs_k7_x" role = "in"
        shape = [4, 8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x256xbf16>
    %yp_l2 = nest.alloc slot = "rs_k7_ypart" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_k7_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_k7, %read_k7, %ready_k7 =
        nest.dispatch.tasks.async @rs_partial_sum l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%x_l2, %yp_l2) ins(%x_l2) outs(%yp_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_k7_grid_done">, !nest.event<"rs_k7_inrel">,
           !nest.event<"rs_k7_out_ready">)
    nest.release %x_l2 depends_on(%read_k7, %x_prefetched)
    %yp_store_k7 = nest.dma.store.async %yp_l2 into %yp_blk
        depends_on(%ready_k7) : !nest.event<"rs_k7_yp_store_done">
    nest.release %yp_l2 depends_on(%yp_store_k7)
    nest.await %grid_k7, %yp_store_k7
    nest.return
  }

  // ---- 合并 context：消费全部 8 份部分和，产出最终 Y[256] ----
  // 8 个输入 L2 buffer 一一对应 Y_part[k_part] 的 [4, 64] 连续切片（producer
  // 写出布局），prefetch 不做任何重排；每个 buffer 的 task 维经 task_dim 视图
  // 对应本 task 的 [1, 64] 部分和。

  nest.context @rs_combine(
      %Y_part : !nest.global_memref<8x4x64xf32>,
      %Y : !nest.global_memref<4x64xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 9216,
          requested_contexts_per_tile = 1> {
    %yp0_blk = nest.subview %Y_part
        offsets = [0, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp1_blk = nest.subview %Y_part
        offsets = [1, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp2_blk = nest.subview %Y_part
        offsets = [2, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp3_blk = nest.subview %Y_part
        offsets = [3, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp4_blk = nest.subview %Y_part
        offsets = [4, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp5_blk = nest.subview %Y_part
        offsets = [5, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp6_blk = nest.subview %Y_part
        offsets = [6, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %yp7_blk = nest.subview %Y_part
        offsets = [7, 0, 0] sizes = [1, 4, 64] strides = [1, 1, 1]
        : !nest.global_view<1x4x64xf32>
    %y_blk = nest.subview %Y
        offsets = [0, 0] sizes = [4, 64] strides = [1, 1]
        : !nest.global_view<4x64xf32>
    %yp0_l2 = nest.alloc slot = "rs_combine_yp0" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp1_l2 = nest.alloc slot = "rs_combine_yp1" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp2_l2 = nest.alloc slot = "rs_combine_yp2" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp3_l2 = nest.alloc slot = "rs_combine_yp3" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp4_l2 = nest.alloc slot = "rs_combine_yp4" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp5_l2 = nest.alloc slot = "rs_combine_yp5" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp6_l2 = nest.alloc slot = "rs_combine_yp6" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp7_l2 = nest.alloc slot = "rs_combine_yp7" role = "in"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %y_l2 = nest.alloc slot = "rs_combine_y" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %yp0_prefetched = nest.dma.prefetch.async %yp0_blk into %yp0_l2
        : !nest.event<"rs_combine_yp0_prefetched">
    %yp1_prefetched = nest.dma.prefetch.async %yp1_blk into %yp1_l2
        : !nest.event<"rs_combine_yp1_prefetched">
    %yp2_prefetched = nest.dma.prefetch.async %yp2_blk into %yp2_l2
        : !nest.event<"rs_combine_yp2_prefetched">
    %yp3_prefetched = nest.dma.prefetch.async %yp3_blk into %yp3_l2
        : !nest.event<"rs_combine_yp3_prefetched">
    %yp4_prefetched = nest.dma.prefetch.async %yp4_blk into %yp4_l2
        : !nest.event<"rs_combine_yp4_prefetched">
    %yp5_prefetched = nest.dma.prefetch.async %yp5_blk into %yp5_l2
        : !nest.event<"rs_combine_yp5_prefetched">
    %yp6_prefetched = nest.dma.prefetch.async %yp6_blk into %yp6_l2
        : !nest.event<"rs_combine_yp6_prefetched">
    %yp7_prefetched = nest.dma.prefetch.async %yp7_blk into %yp7_l2
        : !nest.event<"rs_combine_yp7_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @rs_final_combine l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%yp0_l2, %yp1_l2, %yp2_l2, %yp3_l2, %yp4_l2, %yp5_l2, %yp6_l2,
                 %yp7_l2, %y_l2)
        ins(%yp0_l2, %yp1_l2, %yp2_l2, %yp3_l2, %yp4_l2, %yp5_l2, %yp6_l2, %yp7_l2)
        outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%yp0_prefetched, %yp1_prefetched, %yp2_prefetched,
                   %yp3_prefetched, %yp4_prefetched, %yp5_prefetched,
                   %yp6_prefetched, %yp7_prefetched)
        : (!nest.event<"rs_combine_grid_done">, !nest.event<"rs_combine_inrel">,
           !nest.event<"rs_combine_out_ready">)
    nest.release %yp0_l2 depends_on(%read_c, %yp0_prefetched)
    nest.release %yp1_l2 depends_on(%read_c, %yp1_prefetched)
    nest.release %yp2_l2 depends_on(%read_c, %yp2_prefetched)
    nest.release %yp3_l2 depends_on(%read_c, %yp3_prefetched)
    nest.release %yp4_l2 depends_on(%read_c, %yp4_prefetched)
    nest.release %yp5_l2 depends_on(%read_c, %yp5_prefetched)
    nest.release %yp6_l2 depends_on(%read_c, %yp6_prefetched)
    nest.release %yp7_l2 depends_on(%read_c, %yp7_prefetched)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"rs_combine_y_store_done">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_c, %y_store
    nest.return
  }

  // 模型入口：8 个 producer 全部完成（含 HBM store）后才 submit combine。
  nexus.program @reduce_sum_splitk_multicontext(
      %X : !nest.global_memref<8x4x8x64x256xbf16>,
      %Y_part : !nest.global_memref<8x4x64xf32>,
      %Y : !nest.global_memref<4x64xf32>) {
    %k0_done = nexus.submit_context.async @rs_k0(%X, %Y_part)
        : !nexus.event<"k0_done">
    %k1_done = nexus.submit_context.async @rs_k1(%X, %Y_part)
        : !nexus.event<"k1_done">
    %k2_done = nexus.submit_context.async @rs_k2(%X, %Y_part)
        : !nexus.event<"k2_done">
    %k3_done = nexus.submit_context.async @rs_k3(%X, %Y_part)
        : !nexus.event<"k3_done">
    %k4_done = nexus.submit_context.async @rs_k4(%X, %Y_part)
        : !nexus.event<"k4_done">
    %k5_done = nexus.submit_context.async @rs_k5(%X, %Y_part)
        : !nexus.event<"k5_done">
    %k6_done = nexus.submit_context.async @rs_k6(%X, %Y_part)
        : !nexus.event<"k6_done">
    %k7_done = nexus.submit_context.async @rs_k7(%X, %Y_part)
        : !nexus.event<"k7_done">
    nexus.await %k0_done, %k1_done, %k2_done, %k3_done
    nexus.await %k4_done, %k5_done, %k6_done, %k7_done
    %combine_done = nexus.submit_context.async @rs_combine(%Y_part, %Y)
        : !nexus.event<"combine_done">
    nexus.await %combine_done
    nexus.return
  }
}
