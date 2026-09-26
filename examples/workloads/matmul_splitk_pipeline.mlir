// 256x256x512 bf16 Split-K Matmul（reduce axis K tiling + HBM->L2->L1
// 三级流水）：
//
//   C[256, 512] = A[256, 512] x B[512, 512]
//
// reduce axis（K=512）切分为 4 个 quarter（各 128 = 2 x 64），由 4 个
// leaf dispatch（placement = 15，每 dispatch 4 task 按 M 分工到 4 tiles；
// UCE pin = 0..3，每 tile 4 个 UCE context 并发，--context-mode 4）各自
// 完成 C 的 K-quarter 部分：partial C_d = A[:, {2d, 2d+1}] x B[{2d, 2d+1}, :]
// 经 HBM scratch S 写回，最后 1 个 combine dispatch 从 scratch 读回 4 份
// partial 逐元素相加得到 C——split-K GEMM 的标准两级结构。
//
// 三级流水结构（每级都有显式 IR）：
//   第 1 级 HBM->L2：A、B 整块一次 prefetch 驻留 L2
//     （A_tiled [4,8,64,64] = 256 KiB；B_tiled [8,64,256] = 256 KiB）。
//   第 2 级 L2->L1：每 leaf task 把自己的 K quarter 切成 2 个 k_sub x 64，
//     A chunk [64,64] 8 KiB x2 + B chunk [64,256] 32 KiB x2 双缓冲，
//     两步 BOA 64x256x64 accumulate 出本 quarter 的 C partial。
//   第 3 级 L1 compute：BOA accumulate 进常驻 L1 的 C partial acc f32，
//     store 回 L2 并经 HBM scratch 交给 combine。
//
//   A_tiled [4, 8, 64, 64]  = (m_task, k_sub, R, Kc)   256 KiB bf16
//   B_tiled [8, 64, 256]    = (k_sub, Kc, N_blk)       256 KiB bf16
//   S       [4, 4, 64, 256] = (k_q, m_task, R, N_blk)   1 MiB f32 scratch
//   C       [4, 64, 256]    = (m_task, R, N_blk)        256 KiB f32
//
// L1 Arena contract：leaf 逻辑 payload 2 x 8 KiB (A) + 2 x 32 KiB (B) +
// 64 KiB (C partial acc f32) = 132 KiB 逻辑；每 buffer 各占整 stripe
// round 后 contract = 147,456 B。combine（两阶段、双 scratch buffer）：
// 2 x 64 KiB (S 读回) + 64 KiB acc = 196,608 B = contract；combine 的
// allowed_profiles 收窄为 [0, 1]（R=4 x per-bank envelope 在 mode 2 下
// 不满足）。L2 Arena contract：A + B + 4 partial + 4 scratch 读回 +
// Y = 2,883,584 B。
//
// 数值边界：本 validator 是时间模型。`tile.boa.async` / `tile.evu.async`
// 不携带 buffer operand、不执行任何数值运算，`accumulate` 标记只是循环
// 携带依赖意图。输入 bf16、C 累加器与输出 f32（FP32 accumulate，见
// ELENOR_EVU_Design），仅为类型意图，无数值检查。本例验证 tiling 结构、
// 搬运字节数、依赖/生命周期与时序，不证明 tensor 数值正确性。
//
// 验证点（trace / report）：
//   - BOA 切片 32 = 4 leaf x 4 task x 2 k_sub；每步 ops = 2097152
//     （2 x 64 x 256 x 64）；每 task 首步无 accumulate、第二步 accumulate；
//   - EVU 切片 8 = 1 个 combine dispatch x 4 task x 2 阶段
//     （每阶段 ops = 32768，两阶段完成 4 x 16384 元素求和）；
//   - MFE_LD0 切片 80 = 4 leaf x 4 task x (2 A 8 KiB + 2 B 32 KiB)
//     + combine x 4 task x 4（64 KiB f32 读回）；
//   - HBM->L2 prefetch 6 条（A 262144 B + B 262144 B + 4 x 262144 B
//     scratch 读回）；L2->HBM store 5 条（4 x 262144 B partial + Y）；
//   - combine dispatch 在全部 4 个 partial store 完成后才启动；
//   - 单 nest.context 常驻，active_context_peak = 1。
//
// 运行：bash examples/run.sh matmul-splitk-pipeline
builtin.module {

  // ---- leaf 变体 d：K quarter d = k_sub {2d, 2d+1}；task_dim 0 切 M ----
  // ---- leaf 0 ----
  tile.program @mm_sk_q0(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x8x64x64xbf16>,
      %b_l2 : !nest.l2_buffer<8x64x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 147456> {
    %a0 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 0, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %a1 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 1, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %b0 = tile.subview %b_l2
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %b1 = tile.subview %b_l2
        offsets = [1, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %c_v = tile.subview %c_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %abuf0 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %abuf1 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %bbuf0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %bbuf1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    // k_sub 0：BOA 首步覆写 acc。
    %la0 = tile.load.async %a0 into %abuf0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0 into %bbuf0 : !tile.event<"lb0">
    tile.await %la0, %lb0
    %g0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"g0">
    // k_sub 1 的 load 与 k_sub 0 的 BOA 重叠。
    %la1 = tile.load.async %a1 into %abuf1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1 into %bbuf1 : !tile.event<"lb1">
    tile.await %g0
    // 最后一次真实 load 完成后才 input_released。
    tile.await %la1, %lb1
    tile.signal input_released(%task)
    %g1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        accumulate : !tile.event<"g1">
    tile.await %g1
    tile.free %abuf0
    tile.free %abuf1
    tile.free %bbuf0
    tile.free %bbuf1
    %c_stored = tile.store.async %acc into %c_v : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 1（k_sub 2+3）----
  tile.program @mm_sk_q1(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x8x64x64xbf16>,
      %b_l2 : !nest.l2_buffer<8x64x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 147456> {
    %a0 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 2, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %a1 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 3, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %b0 = tile.subview %b_l2
        offsets = [2, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %b1 = tile.subview %b_l2
        offsets = [3, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %c_v = tile.subview %c_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %abuf0 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %abuf1 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %bbuf0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %bbuf1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %la0 = tile.load.async %a0 into %abuf0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0 into %bbuf0 : !tile.event<"lb0">
    tile.await %la0, %lb0
    %g0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"g0">
    %la1 = tile.load.async %a1 into %abuf1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1 into %bbuf1 : !tile.event<"lb1">
    tile.await %g0
    tile.await %la1, %lb1
    tile.signal input_released(%task)
    %g1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        accumulate : !tile.event<"g1">
    tile.await %g1
    tile.free %abuf0
    tile.free %abuf1
    tile.free %bbuf0
    tile.free %bbuf1
    %c_stored = tile.store.async %acc into %c_v : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 2（k_sub 4+5）----
  tile.program @mm_sk_q2(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x8x64x64xbf16>,
      %b_l2 : !nest.l2_buffer<8x64x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 147456> {
    %a0 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 4, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %a1 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 5, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %b0 = tile.subview %b_l2
        offsets = [4, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %b1 = tile.subview %b_l2
        offsets = [5, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %c_v = tile.subview %c_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %abuf0 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %abuf1 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %bbuf0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %bbuf1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %la0 = tile.load.async %a0 into %abuf0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0 into %bbuf0 : !tile.event<"lb0">
    tile.await %la0, %lb0
    %g0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"g0">
    %la1 = tile.load.async %a1 into %abuf1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1 into %bbuf1 : !tile.event<"lb1">
    tile.await %g0
    tile.await %la1, %lb1
    tile.signal input_released(%task)
    %g1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        accumulate : !tile.event<"g1">
    tile.await %g1
    tile.free %abuf0
    tile.free %abuf1
    tile.free %bbuf0
    tile.free %bbuf1
    %c_stored = tile.store.async %acc into %c_v : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- leaf 3（k_sub 6+7）----
  tile.program @mm_sk_q3(
      %task : !nest.task,
      %a_l2 : !nest.l2_buffer<4x8x64x64xbf16>,
      %b_l2 : !nest.l2_buffer<8x64x256xbf16>,
      %c_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 147456> {
    %a0 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 6, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %a1 = tile.subview %a_l2 task = %task task_dim = 0
        offsets = [0, 7, 0, 0] sizes = [1, 1, 64, 64] strides = [1, 1, 1, 1]
        : !nest.l2_view<1x1x64x64xbf16>
    %b0 = tile.subview %b_l2
        offsets = [6, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %b1 = tile.subview %b_l2
        offsets = [7, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xbf16>
    %c_v = tile.subview %c_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %abuf0 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %abuf1 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %bbuf0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %bbuf1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %la0 = tile.load.async %a0 into %abuf0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0 into %bbuf0 : !tile.event<"lb0">
    tile.await %la0, %lb0
    %g0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"g0">
    %la1 = tile.load.async %a1 into %abuf1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1 into %bbuf1 : !tile.event<"lb1">
    tile.await %g0
    tile.await %la1, %lb1
    tile.signal input_released(%task)
    %g1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        accumulate : !tile.event<"g1">
    tile.await %g1
    tile.free %abuf0
    tile.free %abuf1
    tile.free %bbuf0
    tile.free %bbuf1
    %c_stored = tile.store.async %acc into %c_v : !tile.event<"c_stored">
    tile.await %c_stored
    tile.signal output_ready(%task)
    tile.return
  }

  // ---- combine：4 份 K-quarter partial 逐元素相加 -> 最终 C ----
  tile.program @mm_sk_add(
      %task : !nest.task,
      %q0_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q1_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q2_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q3_l2 : !nest.l2_buffer<4x64x256xf32>,
      %y_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 131072> {
    %q0_v = tile.subview %q0_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %q1_v = tile.subview %q1_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %q2_v = tile.subview %q2_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %q3_v = tile.subview %q3_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %y_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x64x256xf32>
    %qabuf0 = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %acc = tile.alloc shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    // 顺序累加：acc = q0；acc += q1；acc += q2；acc += q3（单一 scratch buffer）。
    %qa = tile.load.async %q0_v into %qabuf0 : !tile.event<"qa">
    tile.await %qa
    %cc0 = tile.evu.async "add" ops = 16384 : !tile.event<"cc0">
    tile.await %cc0
    %qb = tile.load.async %q1_v into %qabuf0 : !tile.event<"qb">
    tile.await %qb
    %cc1 = tile.evu.async "add" ops = 32768 : !tile.event<"cc1">
    tile.await %cc1
    %qc = tile.load.async %q2_v into %qabuf0 : !tile.event<"qc">
    tile.await %qc
    %cc2 = tile.evu.async "add" ops = 32768 : !tile.event<"cc2">
    tile.await %cc2
    %qd = tile.load.async %q3_v into %qabuf0 : !tile.event<"qd">
    tile.await %qd
    tile.signal input_released(%task)
    %cc3 = tile.evu.async "add" ops = 32768 : !tile.event<"cc3">
    tile.await %cc3
    %y_stored = tile.store.async %acc into %y_v : !tile.event<"y_stored">
    tile.await %y_stored
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @mm_sk_ctx(
      %A : !nest.global_memref<4x8x64x64xbf16>,
      %B : !nest.global_memref<8x64x256xbf16>,
      %S : !nest.global_memref<4x4x64x256xf32>,
      %C : !nest.global_memref<4x64x256xf32>)
      placement = 15
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 2883584,
          requested_contexts_per_tile = 4> {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0] sizes = [4, 8, 64, 64] strides = [1, 1, 1, 1]
        : !nest.global_view<4x8x64x64xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0] sizes = [8, 64, 256] strides = [1, 1, 1]
        : !nest.global_view<8x64x256xbf16>
    %s0_blk = nest.subview %S
        offsets = [0, 0, 0, 0] sizes = [1, 4, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x4x64x256xf32>
    %s1_blk = nest.subview %S
        offsets = [1, 0, 0, 0] sizes = [1, 4, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x4x64x256xf32>
    %s2_blk = nest.subview %S
        offsets = [2, 0, 0, 0] sizes = [1, 4, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x4x64x256xf32>
    %s3_blk = nest.subview %S
        offsets = [3, 0, 0, 0] sizes = [1, 4, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x4x64x256xf32>
    %y_blk = nest.subview %C
        offsets = [0, 0, 0] sizes = [4, 64, 256] strides = [1, 1, 1]
        : !nest.global_view<4x64x256xf32>
    %a_l2 = nest.alloc slot = "mm_sk_a" role = "in"
        shape = [4, 8, 64, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x64xbf16>
    %b_l2 = nest.alloc slot = "mm_sk_b" role = "in"
        shape = [8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<8x64x256xbf16>
    %c0_l2 = nest.alloc slot = "mm_sk_c0" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %c1_l2 = nest.alloc slot = "mm_sk_c1" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %c2_l2 = nest.alloc slot = "mm_sk_c2" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %c3_l2 = nest.alloc slot = "mm_sk_c3" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %q0_l2 = nest.alloc slot = "mm_sk_q0" role = "in"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %q1_l2 = nest.alloc slot = "mm_sk_q1" role = "in"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %q2_l2 = nest.alloc slot = "mm_sk_q2" role = "in"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %q3_l2 = nest.alloc slot = "mm_sk_q3" role = "in"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %y_l2 = nest.alloc slot = "mm_sk_y" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    // 第 1 级：A、B 一次 HBM->L2 prefetch。
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_sk_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_sk_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    // ---- 第 2/3 级：4 个 split-K leaf 并发 ----
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @mm_sk_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c0_l2) ins(%a_l2, %b_l2) outs(%c0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_grid_0">, !nest.event<"mm_sk_read_0">,
           !nest.event<"mm_sk_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @mm_sk_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c1_l2) ins(%a_l2, %b_l2) outs(%c1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_grid_1">, !nest.event<"mm_sk_read_1">,
           !nest.event<"mm_sk_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @mm_sk_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c2_l2) ins(%a_l2, %b_l2) outs(%c2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_grid_2">, !nest.event<"mm_sk_read_2">,
           !nest.event<"mm_sk_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @mm_sk_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %c3_l2) ins(%a_l2, %b_l2) outs(%c3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_grid_3">, !nest.event<"mm_sk_read_3">,
           !nest.event<"mm_sk_ready_3">)
    // ---- partial 写回 scratch（对应 split-K 把 C partial 写 global）----
    %store_0 = nest.dma.store.async %c0_l2 into %s0_blk
        depends_on(%ready_0) : !nest.event<"mm_sk_store_0">
    %store_1 = nest.dma.store.async %c1_l2 into %s1_blk
        depends_on(%ready_1) : !nest.event<"mm_sk_store_1">
    %store_2 = nest.dma.store.async %c2_l2 into %s2_blk
        depends_on(%ready_2) : !nest.event<"mm_sk_store_2">
    %store_3 = nest.dma.store.async %c3_l2 into %s3_blk
        depends_on(%ready_3) : !nest.event<"mm_sk_store_3">
    // ---- combine：scratch prefetch -> 逐元素相加 -> 最终 C ----
    %pref_0 = nest.dma.prefetch.async %s0_blk into %q0_l2
        depends_on(%store_0) : !nest.event<"mm_sk_pref_0">
    %pref_1 = nest.dma.prefetch.async %s1_blk into %q1_l2
        depends_on(%store_1) : !nest.event<"mm_sk_pref_1">
    %pref_2 = nest.dma.prefetch.async %s2_blk into %q2_l2
        depends_on(%store_2) : !nest.event<"mm_sk_pref_2">
    %pref_3 = nest.dma.prefetch.async %s3_blk into %q3_l2
        depends_on(%store_3) : !nest.event<"mm_sk_pref_3">
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @mm_sk_add l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%q0_l2, %q1_l2, %q2_l2, %q3_l2, %y_l2)
        ins(%q0_l2, %q1_l2, %q2_l2, %q3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%pref_0, %pref_1, %pref_2, %pref_3)
        : (!nest.event<"mm_sk_grid_c">, !nest.event<"mm_sk_read_c">,
           !nest.event<"mm_sk_ready_c">)
    nest.release %a_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %a_prefetched)
    nest.release %b_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %b_prefetched)
    nest.release %c0_l2 depends_on(%store_0)
    nest.release %c1_l2 depends_on(%store_1)
    nest.release %c2_l2 depends_on(%store_2)
    nest.release %c3_l2 depends_on(%store_3)
    nest.release %q0_l2 depends_on(%read_c, %pref_0)
    nest.release %q1_l2 depends_on(%read_c, %pref_1)
    nest.release %q2_l2 depends_on(%read_c, %pref_2)
    nest.release %q3_l2 depends_on(%read_c, %pref_3)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"mm_sk_store_y">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %y_store
    nest.return
  }

  nexus.program @matmul_splitk_tiled_pipeline(
      %A : !nest.global_memref<4x8x64x64xbf16>,
      %B : !nest.global_memref<8x64x256xbf16>,
      %S : !nest.global_memref<4x4x64x256xf32>,
      %C : !nest.global_memref<4x64x256xf32>) {
    %done = nexus.submit_context.async @mm_sk_ctx(%A, %B, %S, %C)
        : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
