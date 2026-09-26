// 256x4096 bf16 Reduce-Sum（沿 reduce axis K 规约），单 context 两层组织：
// task 间按行分工（m_task，非 reduce-axis tiling）；reduce-axis tiling 在
// task 内——K = 4096 切成 8 x 512 chunk。本 program 的 per-task L1 Arena
// （contract 135168 B）装不下任何 task 的 512 KiB K 切片，必须分块经 L2 流水。
//
//   总问题:  Y[256] = sum_k X[256, 4096]            X = 2 MiB bf16
//
//   第 1 层 (task 间按行分工, 非 reduce-axis):  placement = 15，4 个 task
//     （task_dim = 0）各拥有 64 行 x 4096 列 = 512 KiB 完整 K 切片；4 个
//     切片合计 2 MiB，大于 4 x L1 Arena 合计，整体驻留在 L2（2 MiB <
//     8 MiB Group SRAM）。
//
//   第 2 层 (task 内 reduce-axis tiling, L1 分块):  512 KiB 行块远超本
//     program 的 L1 Arena
//     contract（逻辑 payload = 双缓冲 2 x 64 KiB bf16 chunk + 256 B f32 acc
//     = 131328 B，含 stripe padding 后 Arena contract = 135168 B）。行块
//     再沿 K 切成 KS = 8 x KC = 512 的 chunk（64 KiB/chunk），
//     buf0/buf1 双缓冲使 MFE load 与 EVU reduce 流水重叠；跨 chunk 的
//     循环携带依赖是常驻 L1 的累加器 %acc[64]——每 chunk 规约出 64 行
//     部分和并累加进 %acc，最后 store 回 L2 的 [4, 64] 输出切片。
//
// HBM 全局输入使用 block-packed（预切）布局，tiling 维都在前导维上，
// 所有 subview / DMA 均为连续 row-major 区间（validator V1 物理约束）：
//
//   X_tiled [4, 8, 64, 512] = (m_task, k_step, R, KC)     2 MiB bf16
//   Y       [4, 64]         = (m_task, R)                 1 KiB f32
//
// 数值边界：本 validator 是时间模型。`tile.evu.async` 不携带 buffer
// operand、不执行任何数值运算，也没有 accumulate 语义；首步命名
// `reduce_sum`（初始化 acc）、后续 7 步命名 `reduce_sum_accumulate`
// 只是 IR 上的循环携带依赖意图，%acc 并未真的被累加。输入 chunk 保持
// bf16，累加器与输出为 f32（BF16 reduction 使用 FP32 accumulate，
// 见 ELENOR_EVU_Design），仅为类型意图，无数值检查。本例验证 tiling
// 结构、搬运字节数、依赖/生命周期与时序，不证明 tensor 数值正确性。
//
// 每 task 数据流（时间展开，无循环 IR）：
//
//   load x0->buf0, load x1->buf1           # 预发两个 chunk 的 L2->L1 load
//   await x0; reduce_sum(buf0)->acc         # 首 chunk 初始化累加器
//   s = 1..7:  await x_s; reduce_sum_accumulate(buf_s)+=acc;
//              先发射 x_{s+1} load 与本步 EVU compute 重叠
//   await x7 后才 signal input_released     # 不早于最后一次真实 load
//   store acc -> Y[task, 0:64]；HBM store 由 output_ready 门控
//
// 验证点（trace / report）：
//   - EVU 切片 32 = 4 task x 8 K-step；每步 ops = 32832（64x512 输入 + 64 输出）；
//   - MFE_LD0 切片 32，每条 bytes = 65536；
//   - HBM->L2 prefetch 1 条（2097152 B）、L2->HBM store 1 条（1024 B f32）；
//   - 每 task 的 input_released >= 最后一次（第 8 个）chunk load 完成；
//   - 单 context 常驻，active_context_peak = 1。
//
// 运行：bash examples/run.sh reduce-sum-single-context
builtin.module {

  // SPMD tile program：4 个 task 共用；task 区分 4 个 64 行块（task_dim = 0），
  // K = 4096 在 task 内静态展开为 8 个 64x512 chunk，双缓冲流过 L1。
  tile.program @rs_ktiled(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x8x64x512xbf16>,
      %y_l2 : !nest.l2_buffer<4x64xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 135168> {
    // 本 task 的 8 个 K-chunk 视图（k_step 0..7）。
    %x0 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 2, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 3, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 4, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 5, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 6, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 0
        offsets = [0, 7, 0, 0] sizes = [1, 1, 64, 512] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x512xbf16>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xf32>
    // L1：双缓冲 chunk（2 x 64 KiB bf16）+ 常驻 f32 累加器（跨 k_step 循环
    // 携带；BF16 reduction 使用 FP32 accumulate，见 ELENOR_EVU_Design）。
    %buf0 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %buf1 = tile.alloc shape = [64, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x512xbf16>
    %acc = tile.alloc shape = [64] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<64xf32>
    // 预发两个 chunk 的 load，让 MFE load 与首个 EVU reduce 重叠。
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    // k_step 0：reduce_sum 初始化累加器（32768 输入 + 64 输出）。
    %c0 = tile.evu.async "reduce_sum" ops = 32832 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    // k_step 1：reduce_sum_accumulate 累加进 acc；先发射 k_step 2 的 load
    // 与本步 EVU compute 重叠（buf0 在 c0 await 后已可复用）。
    %c1 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    // 最后一次真实 load 完成后才允许 input_released。
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum_accumulate" ops = 32832 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    // 行和 [64] 写回本 task 的 Y 切片，再由 context 级 HBM store 落盘。
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @rs_ctx(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>)
      placement = 15 context = 0
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 4, l2_spm_bytes = 2101248,
          requested_contexts_per_tile = 1> {
    %x_blk = nest.subview %X
        offsets = [0, 0, 0, 0] sizes = [4, 8, 64, 512] strides = [1, 1, 1, 1]
        : !nest.global_view<4x8x64x512xbf16>
    %y_blk = nest.subview %Y
        offsets = [0, 0] sizes = [4, 64] strides = [1, 1]
        : !nest.global_view<4x64xf32>
    %x_l2 = nest.alloc slot = "rs_x" role = "in"
        shape = [4, 8, 64, 512] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x512xbf16>
    %y_l2 = nest.alloc slot = "rs_y" role = "out"
        shape = [4, 64] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x64xf32>
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_done, %read_done, %ready_done =
        nest.dispatch.tasks.async @rs_ktiled l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %y_l2) ins(%x_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_grid_done">, !nest.event<"rs_inrel">,
           !nest.event<"rs_out_ready">)
    nest.release %x_l2 depends_on(%read_done, %x_prefetched)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_done) : !nest.event<"rs_y_store_done">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_done, %y_store
    nest.return
  }

  nexus.program @reduce_sum_ktiled_single_context(
      %X : !nest.global_memref<4x8x64x512xbf16>,
      %Y : !nest.global_memref<4x64xf32>) {
    %done = nexus.submit_context.async @rs_ctx(%X, %Y)
        : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
