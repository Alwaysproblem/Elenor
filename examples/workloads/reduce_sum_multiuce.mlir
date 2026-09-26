// 512x4096 bf16 Reduce-Sum（沿 reduce axis K 规约），非 reduce axis 的
// multiUCE 情况：UCE supertile x task 行块两级 M 分工，4 个 UCE hardware
// context 每 tile 并发驻留，全部 4 个 tile 参与执行：
//
//   总问题:  Y[512] = sum_k X[512, 4096]            X = 4 MiB bf16
//
//   M 维两级分工（都不是 reduce axis）：
//     dim0 = uce   : 4 个静态程序变体 @rs_uce_d0..d3（UCE context pin =
//                    0..3，dispatch d 固定 dim0 = d）——每个 UCE context
//                    拥有一个 128 行 supertile；
//     dim1 = task  :  placement = 15 的每 dispatch 4 task（task_dim = 1）
//                    把 supertile 切成 4 个 32 行块。
//   即 4 uce x 4 task x 32 行 = 512 行，无重复计算；每个 context 规约
//   完整 K = 8 x 512 = 4096（reduce axis 在 context 内部按 k_step 切块
//   流水，不是跨 context 切分——跨 context 的 K 切分见
//   reduce_sum_splitk_multiuce）。
//
//   X_tiled [4, 4, 8, 32, 512] = (uce, task, k_step, R, KC)  4 MiB bf16
//   Y       [4, 4, 32]         = (uce, task, R)              2 KiB f32
//
// 三级流水：X 整块一次 HBM->L2 prefetch（4 MiB 驻留 Group SRAM）；每 task
// 把自己的 8 个 32 KiB k_step chunk 经 L2->L1 双缓冲流过（下一 chunk 的
// load 与当前 EVU reduce 重叠）；EVU 在 L1 内逐步 accumulate 进常驻
// %acc[32] f32。input_released 在第 8 个 chunk load await 之后才发。
//
// context placement = 15、requested_contexts_per_tile = 4（--context-mode
// 4）：运行期每个 tile 同时驻留 4 个 UCE context——每个 dispatch 在每个
// tile 上各有 1 task（4 dispatch x 4 tile = 16 个 task 实例，各占一个
// pin），共享该 tile 的单 EVU 服务队列。
//
// 数值边界：本 validator 是时间模型。`tile.evu.async` 不携带 buffer operand、
// 不执行任何数值运算、也没有 accumulate 语义；首步命名 `reduce_sum`、后续
// `reduce_sum_accumulate` 只是 IR 上的循环携带依赖意图。输入 chunk 为 bf16，
// 累加器与输出为 f32（BF16 reduction 使用 FP32 accumulate，见
// ELENOR_EVU_Design），仅为类型意图，无数值检查。本例验证 tiling 结构、
// 搬运字节数、依赖/生命周期与时序，不证明 tensor 数值正确性。
//
// 验证点（trace / report）：
//   - EVU 切片 128 = 4 dispatch x 4 task x 8 k_step；首步 ops = 16416
//     （16384 输入 + 32 输出），后续同值（accumulate 标注区分）；
//   - MFE_LD0 切片 128 = 4 dispatch x 4 task x 8 chunk，每条 bytes = 32768；
//   - HBM->L2 prefetch 1 条（4194304 B）；L2->HBM store 4 条（各 512 B f32）；
//   - 每 tile 4 个 UCE context 并发驻留（4 dispatch x 1 task/tile），
//     EVU 为各 tile 单服务队列，串行接受服务；
//   - 单 nest.context 常驻，active_context_peak = 1。
//
// 运行：bash examples/run.sh reduce-sum-multiuce
builtin.module {

  // 4 个静态变体：view 的 uce offset 固定为 0/1/2/3；task_dim = 1 由
  // logical task 选 32 行块。每 task：8 个 k_step chunk 双缓冲 + 常驻 acc。
  // ---- 变体 d0（uce 0）----
  tile.program @rs_uce_d0(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x4x8x32x512xbf16>,
      %y_l2 : !nest.l2_buffer<4x32xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 69632> {
    %x0 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 1, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 2, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 3, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 4, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 5, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 6, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [0, 0, 7, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 32] strides = [1, 1]
        : !nest.l2_view<1x32xf32>
    // L1：双缓冲 chunk（2 x 32 KiB bf16）+ 常驻 f32 累加器。
    %buf0 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %buf1 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %acc = tile.alloc shape = [32] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<32xf32>
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    // k_step 0：reduce_sum 初始化累加器（16384 输入 + 32 输出）。
    %c0 = tile.evu.async "reduce_sum" ops = 16416 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    // k_step 1..7：reduce_sum_accumulate；下一 chunk 的 load 与当前 EVU 重叠。
    %c1 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    // 最后一次真实 load 完成后才 input_released。
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    // 行和 [32] f32 写回本 (uce, task) 的输出切片。
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 变体 d1（仅 uce offset = 1）----
  tile.program @rs_uce_d1(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x4x8x32x512xbf16>,
      %y_l2 : !nest.l2_buffer<4x32xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 69632> {
    %x0 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 1, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 2, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 3, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 4, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 5, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 6, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [1, 0, 7, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 32] strides = [1, 1]
        : !nest.l2_view<1x32xf32>
    %buf0 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %buf1 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %acc = tile.alloc shape = [32] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<32xf32>
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    %c0 = tile.evu.async "reduce_sum" ops = 16416 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    %c1 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 变体 d2（仅 uce offset = 2）----
  tile.program @rs_uce_d2(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x4x8x32x512xbf16>,
      %y_l2 : !nest.l2_buffer<4x32xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 69632> {
    %x0 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 1, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 2, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 3, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 4, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 5, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 6, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [2, 0, 7, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 32] strides = [1, 1]
        : !nest.l2_view<1x32xf32>
    %buf0 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %buf1 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %acc = tile.alloc shape = [32] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<32xf32>
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    %c0 = tile.evu.async "reduce_sum" ops = 16416 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    %c1 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- 变体 d3（仅 uce offset = 3）----
  tile.program @rs_uce_d3(
      %task : !nest.task,
      %x_l2 : !nest.l2_buffer<4x4x8x32x512xbf16>,
      %y_l2 : !nest.l2_buffer<4x32xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 69632> {
    %x0 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 0, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x1 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 1, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x2 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 2, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x3 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 3, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x4 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 4, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x5 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 5, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x6 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 6, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %x7 = tile.subview %x_l2 task = %task task_dim = 1
        offsets = [3, 0, 7, 0, 0] sizes = [1, 1, 1, 32, 512]
        strides = [1, 1, 1, 1, 1] : !nest.l2_view<1x1x1x32x512xbf16>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 32] strides = [1, 1]
        : !nest.l2_view<1x32xf32>
    %buf0 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %buf1 = tile.alloc shape = [32, 512] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<32x512xbf16>
    %acc = tile.alloc shape = [32] dtype = "f32"
        alignment = 64 : !tile.l1_buffer<32xf32>
    %x0_l = tile.load.async %x0 into %buf0 : !tile.event<"x0_l">
    %x1_l = tile.load.async %x1 into %buf1 : !tile.event<"x1_l">
    tile.await %x0_l
    %c0 = tile.evu.async "reduce_sum" ops = 16416 : !tile.event<"c0">
    tile.await %c0
    tile.await %x1_l
    %c1 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c1">
    %x2_l = tile.load.async %x2 into %buf0 : !tile.event<"x2_l">
    tile.await %c1
    tile.await %x2_l
    %c2 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c2">
    %x3_l = tile.load.async %x3 into %buf1 : !tile.event<"x3_l">
    tile.await %c2
    tile.await %x3_l
    %c3 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c3">
    %x4_l = tile.load.async %x4 into %buf0 : !tile.event<"x4_l">
    tile.await %c3
    tile.await %x4_l
    %c4 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c4">
    %x5_l = tile.load.async %x5 into %buf1 : !tile.event<"x5_l">
    tile.await %c4
    tile.await %x5_l
    %c5 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c5">
    %x6_l = tile.load.async %x6 into %buf0 : !tile.event<"x6_l">
    tile.await %c5
    tile.await %x6_l
    %c6 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c6">
    %x7_l = tile.load.async %x7 into %buf1 : !tile.event<"x7_l">
    tile.await %c6
    tile.await %x7_l
    tile.signal input_released(%task)
    %c7 = tile.evu.async "reduce_sum_accumulate" ops = 16416 : !tile.event<"c7">
    tile.await %c7
    tile.free %buf0
    tile.free %buf1
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @rs_uce_ctx(
      %X : !nest.global_memref<4x4x8x32x512xbf16>,
      %Y : !nest.global_memref<4x4x32xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 16, l2_spm_bytes = 4198400,
          requested_contexts_per_tile = 4> {
    %x_blk = nest.subview %X
        offsets = [0, 0, 0, 0, 0] sizes = [4, 4, 8, 32, 512] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<4x4x8x32x512xbf16>
    %y0_blk = nest.subview %Y
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.global_view<1x4x32xf32>
    %y1_blk = nest.subview %Y
        offsets = [1, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.global_view<1x4x32xf32>
    %y2_blk = nest.subview %Y
        offsets = [2, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.global_view<1x4x32xf32>
    %y3_blk = nest.subview %Y
        offsets = [3, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.global_view<1x4x32xf32>
    %x_l2 = nest.alloc slot = "rs_uce_x" role = "in"
        shape = [4, 4, 8, 32, 512] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x4x8x32x512xbf16>
    %y0_l2 = nest.alloc slot = "rs_uce_y0" role = "out"
        shape = [4, 32] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x32xf32>
    %y1_l2 = nest.alloc slot = "rs_uce_y1" role = "out"
        shape = [4, 32] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x32xf32>
    %y2_l2 = nest.alloc slot = "rs_uce_y2" role = "out"
        shape = [4, 32] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x32xf32>
    %y3_l2 = nest.alloc slot = "rs_uce_y3" role = "out"
        shape = [4, 32] dtype = "f32" alignment = 64
        : !nest.l2_buffer<4x32xf32>
    // 一次性把整个 X 装进 L2，4 个 dispatch 只读共享。
    %x_prefetched = nest.dma.prefetch.async %x_blk into %x_l2
        : !nest.event<"rs_uce_x_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @rs_uce_d0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%x_l2, %y0_l2) ins(%x_l2) outs(%y0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_uce_grid_0">, !nest.event<"rs_uce_read_0">,
           !nest.event<"rs_uce_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @rs_uce_d1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%x_l2, %y1_l2) ins(%x_l2) outs(%y1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_uce_grid_1">, !nest.event<"rs_uce_read_1">,
           !nest.event<"rs_uce_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @rs_uce_d2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%x_l2, %y2_l2) ins(%x_l2) outs(%y2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_uce_grid_2">, !nest.event<"rs_uce_read_2">,
           !nest.event<"rs_uce_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @rs_uce_d3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%x_l2, %y3_l2) ins(%x_l2) outs(%y3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%x_prefetched)
        : (!nest.event<"rs_uce_grid_3">, !nest.event<"rs_uce_read_3">,
           !nest.event<"rs_uce_ready_3">)
    nest.release %x_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %x_prefetched)
    %store_0 = nest.dma.store.async %y0_l2 into %y0_blk
        depends_on(%ready_0) : !nest.event<"rs_uce_store_0">
    %store_1 = nest.dma.store.async %y1_l2 into %y1_blk
        depends_on(%ready_1) : !nest.event<"rs_uce_store_1">
    %store_2 = nest.dma.store.async %y2_l2 into %y2_blk
        depends_on(%ready_2) : !nest.event<"rs_uce_store_2">
    %store_3 = nest.dma.store.async %y3_l2 into %y3_blk
        depends_on(%ready_3) : !nest.event<"rs_uce_store_3">
    nest.release %y0_l2 depends_on(%store_0)
    nest.release %y1_l2 depends_on(%store_1)
    nest.release %y2_l2 depends_on(%store_2)
    nest.release %y3_l2 depends_on(%store_3)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %store_0, %store_1, %store_2, %store_3
    nest.return
  }

  nexus.program @reduce_sum_multiuce(
      %X : !nest.global_memref<4x4x8x32x512xbf16>,
      %Y : !nest.global_memref<4x4x32xf32>) {
    %done = nexus.submit_context.async @rs_uce_ctx(%X, %Y)
        : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
