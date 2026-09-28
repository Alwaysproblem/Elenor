// 512x512x512 bf16 Split-K Matmul：四个 M/N 分片 context，每个 context 独立拆 K。
//
//   C[512,512] = A[512,512] x B[512,512]
//   context (m,n) 拥有 C 的一个 [256,256] 象限，m,n ∈ {0,1}；
//   每个 context 的 4 个 task 再沿 M 分成 4 个 64 行块（placement=15）。
//   4 个 leaf dispatch 分别负责 K 的 quarter {0,1}, {2,3}, {4,5}, {6,7}：
//   每个 leaf 的两个 K=64 tile 在 f32 L1 acc 中 BOA accumulate，输出 partial。
//   context 内的 combine 读取 4 个 context-local L2 partial 并写最终 C 象限；
//   不需要 HBM scratch，也不跨 device context 汇总 K partial。
//
// HBM 采用 block-packed 布局，确保每个 nest.subview 是连续的 row-major 区间：
//   A [2,4,8,64,64]  = (m_context,m_task,k_sub,M_tile,K_tile) 512 KiB bf16
//   B [2,8,64,256]   = (n_context,k_sub,K_tile,N_tile)       512 KiB bf16
//   C [2,2,4,64,256] = (m_context,n_context,m_task,M_tile,N_tile) 1 MiB f32
// 每 context L2 = A/B 各 256 KiB + 4 partial 各 256 KiB + C 256 KiB
// = 1,835,008 B；4 个并发 context 总计 7 MiB。leaf L1 contract 147,456 B，
// combine L1 contract 131,072 B； --context-mode 4 --device-context-mode 4。
//
// 时间模型：tile.boa.async / tile.evu.async 不携带数值 operand；accumulate
// 只表示循环携带依赖，不能据此证明矩阵乘法的数值正确性。
// trace / report 可检查：128 BOA (4 context x 4 leaf x 4 task x 2 K tile)、
// 96 EVU (4 context x 4 task x 2 半区 x 3 add，combine 双缓冲流水)、
// 8 次 A/B prefetch、4 次最终 C store。
// 运行：bash examples/run.sh matmul-splitk-multicontext-pipeline
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
  // software pipeline：[64,256] 按行分成两个 [32,256] 半区。q0 半区直接 load
  // 进 acc 半区（省去原顺序版的 copy add），q1/q2/q3 的 L2→L1 load 与前一个
  // EVU add 重叠；s0/s1 两个 32 KiB scratch 交替承载相邻 load，半区 b 的
  // 首个 load 提前到半区 a 流水期间发出。4 个缓冲合计 4 x 32 KiB =
  // 131072 B，L1 contract 与 allowed_profiles 保持不变。
  tile.program @mm_sk_add(
      %task : !nest.task,
      %q0_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q1_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q2_l2 : !nest.l2_buffer<4x64x256xf32>,
      %q3_l2 : !nest.l2_buffer<4x64x256xf32>,
      %y_l2 : !nest.l2_buffer<4x64x256xf32>)
                resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 131072> {
    // 半区 a = 行 0..31，半区 b = 行 32..63。
    %q0a_v = tile.subview %q0_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q0b_v = tile.subview %q0_l2 task = %task task_dim = 0
        offsets = [0, 32, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q1a_v = tile.subview %q1_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q1b_v = tile.subview %q1_l2 task = %task task_dim = 0
        offsets = [0, 32, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q2a_v = tile.subview %q2_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q2b_v = tile.subview %q2_l2 task = %task task_dim = 0
        offsets = [0, 32, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q3a_v = tile.subview %q3_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %q3b_v = tile.subview %q3_l2 task = %task task_dim = 0
        offsets = [0, 32, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %ya_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %yb_v = tile.subview %y_l2 task = %task task_dim = 0
        offsets = [0, 32, 0] sizes = [1, 32, 256] strides = [1, 1, 1]
        : !nest.l2_view<1x32x256xf32>
    %acc0 = tile.alloc shape = [32, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<32x256xf32>
    %acc1 = tile.alloc shape = [32, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<32x256xf32>
    %s0 = tile.alloc shape = [32, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<32x256xf32>
    %s1 = tile.alloc shape = [32, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<32x256xf32>
    // ---- 半区 a：acc0 = q0a（直接 load），再依次 += q1a/q2a/q3a ----
    %la0 = tile.load.async %q0a_v into %acc0 : !tile.event<"sk_la0">
    %la1 = tile.load.async %q1a_v into %s0 : !tile.event<"sk_la1">
    tile.await %la0, %la1
    %ca0 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_ca0">
    %la2 = tile.load.async %q2a_v into %s1 : !tile.event<"sk_la2">
    %lb0 = tile.load.async %q0b_v into %acc1 : !tile.event<"sk_lb0">
    tile.await %ca0
    tile.await %la2
    %ca1 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_ca1">
    %la3 = tile.load.async %q3a_v into %s0 : !tile.event<"sk_la3">
    tile.await %ca1
    tile.await %la3
    %ca2 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_ca2">
    tile.await %ca2
    %ys0 = tile.store.async %acc0 into %ya_v : !tile.event<"sk_ys0">
    // ---- 半区 b：复用 s0/s1；q0b 已在半区 a 流水期间提前 load ----
    tile.await %lb0
    %lb1 = tile.load.async %q1b_v into %s0 : !tile.event<"sk_lb1">
    tile.await %lb1
    %cb0 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_cb0">
    %lb2 = tile.load.async %q2b_v into %s1 : !tile.event<"sk_lb2">
    tile.await %cb0
    tile.await %lb2
    %cb1 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_cb1">
    %lb3 = tile.load.async %q3b_v into %s0 : !tile.event<"sk_lb3">
    tile.await %cb1
    tile.await %lb3
    // 最后一次真实 load 完成后才 input_released。
    tile.signal input_released(%task)
    %cb2 = tile.evu.async "add" ops = 16384 : !tile.event<"sk_cb2">
    tile.await %cb2
    %ys1 = tile.store.async %acc1 into %yb_v : !tile.event<"sk_ys1">
    tile.await %ys0, %ys1
    tile.signal output_ready(%task)
    tile.return
  }

  // context (0,0)：输出 C[0,0,:,:,:]，4 个 leaf 在本 context 内合并。
  nest.context @mm_sk_m0n0(
      %A : !nest.global_memref<2x4x8x64x64xbf16>,
      %B : !nest.global_memref<2x8x64x256xbf16>,
      %C : !nest.global_memref<2x2x4x64x256xf32>)
      placement = 15 context = 0
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 1835008,
          requested_contexts_per_tile = 4> {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 64] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x64xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0, 0] sizes = [1, 8, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x8x64x256xbf16>
    %y_blk = nest.subview %C
        offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
        strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xf32>
    %a_l2 = nest.alloc slot = "mm_sk_m0n0_a" role = "in"
        shape = [4, 8, 64, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x64xbf16>
    %b_l2 = nest.alloc slot = "mm_sk_m0n0_b" role = "in"
        shape = [8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<8x64x256xbf16>
    %p0_l2 = nest.alloc slot = "mm_sk_m0n0_p0" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p1_l2 = nest.alloc slot = "mm_sk_m0n0_p1" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p2_l2 = nest.alloc slot = "mm_sk_m0n0_p2" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p3_l2 = nest.alloc slot = "mm_sk_m0n0_p3" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %y_l2 = nest.alloc slot = "mm_sk_m0n0_y" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_sk_m0n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_sk_m0n0_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @mm_sk_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p0_l2) ins(%a_l2, %b_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n0_grid_0">, !nest.event<"mm_sk_m0n0_read_0">,
           !nest.event<"mm_sk_m0n0_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @mm_sk_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p1_l2) ins(%a_l2, %b_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n0_grid_1">, !nest.event<"mm_sk_m0n0_read_1">,
           !nest.event<"mm_sk_m0n0_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @mm_sk_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p2_l2) ins(%a_l2, %b_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n0_grid_2">, !nest.event<"mm_sk_m0n0_read_2">,
           !nest.event<"mm_sk_m0n0_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @mm_sk_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p3_l2) ins(%a_l2, %b_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n0_grid_3">, !nest.event<"mm_sk_m0n0_read_3">,
           !nest.event<"mm_sk_m0n0_ready_3">)
    // 同 context 直读四份 partial；不写回 HBM 中转。
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @mm_sk_add l1_mode = 0
        tasks(%tasks) globals()
        bindings(%p0_l2, %p1_l2, %p2_l2, %p3_l2, %y_l2)
        ins(%p0_l2, %p1_l2, %p2_l2, %p3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%ready_0, %ready_1, %ready_2, %ready_3)
        : (!nest.event<"mm_sk_m0n0_grid_c">, !nest.event<"mm_sk_m0n0_read_c">,
           !nest.event<"mm_sk_m0n0_ready_c">)
    nest.release %a_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %a_prefetched)
    nest.release %b_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %b_prefetched)
    nest.release %p0_l2 depends_on(%ready_0, %read_c)
    nest.release %p1_l2 depends_on(%ready_1, %read_c)
    nest.release %p2_l2 depends_on(%ready_2, %read_c)
    nest.release %p3_l2 depends_on(%ready_3, %read_c)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"mm_sk_m0n0_store_y">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %y_store
    nest.return
  }

  // context (0,1)：输出 C[0,1,:,:,:]，4 个 leaf 在本 context 内合并。
  nest.context @mm_sk_m0n1(
      %A : !nest.global_memref<2x4x8x64x64xbf16>,
      %B : !nest.global_memref<2x8x64x256xbf16>,
      %C : !nest.global_memref<2x2x4x64x256xf32>)
      placement = 15 context = 1
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 1835008,
          requested_contexts_per_tile = 4> {
    %a_blk = nest.subview %A
        offsets = [0, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 64] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x64xbf16>
    %b_blk = nest.subview %B
        offsets = [1, 0, 0, 0] sizes = [1, 8, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x8x64x256xbf16>
    %y_blk = nest.subview %C
        offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
        strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xf32>
    %a_l2 = nest.alloc slot = "mm_sk_m0n1_a" role = "in"
        shape = [4, 8, 64, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x64xbf16>
    %b_l2 = nest.alloc slot = "mm_sk_m0n1_b" role = "in"
        shape = [8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<8x64x256xbf16>
    %p0_l2 = nest.alloc slot = "mm_sk_m0n1_p0" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p1_l2 = nest.alloc slot = "mm_sk_m0n1_p1" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p2_l2 = nest.alloc slot = "mm_sk_m0n1_p2" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p3_l2 = nest.alloc slot = "mm_sk_m0n1_p3" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %y_l2 = nest.alloc slot = "mm_sk_m0n1_y" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_sk_m0n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_sk_m0n1_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @mm_sk_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p0_l2) ins(%a_l2, %b_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n1_grid_0">, !nest.event<"mm_sk_m0n1_read_0">,
           !nest.event<"mm_sk_m0n1_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @mm_sk_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p1_l2) ins(%a_l2, %b_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n1_grid_1">, !nest.event<"mm_sk_m0n1_read_1">,
           !nest.event<"mm_sk_m0n1_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @mm_sk_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p2_l2) ins(%a_l2, %b_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n1_grid_2">, !nest.event<"mm_sk_m0n1_read_2">,
           !nest.event<"mm_sk_m0n1_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @mm_sk_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p3_l2) ins(%a_l2, %b_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m0n1_grid_3">, !nest.event<"mm_sk_m0n1_read_3">,
           !nest.event<"mm_sk_m0n1_ready_3">)
    // 同 context 直读四份 partial；不写回 HBM 中转。
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @mm_sk_add l1_mode = 0
        tasks(%tasks) globals()
        bindings(%p0_l2, %p1_l2, %p2_l2, %p3_l2, %y_l2)
        ins(%p0_l2, %p1_l2, %p2_l2, %p3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%ready_0, %ready_1, %ready_2, %ready_3)
        : (!nest.event<"mm_sk_m0n1_grid_c">, !nest.event<"mm_sk_m0n1_read_c">,
           !nest.event<"mm_sk_m0n1_ready_c">)
    nest.release %a_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %a_prefetched)
    nest.release %b_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %b_prefetched)
    nest.release %p0_l2 depends_on(%ready_0, %read_c)
    nest.release %p1_l2 depends_on(%ready_1, %read_c)
    nest.release %p2_l2 depends_on(%ready_2, %read_c)
    nest.release %p3_l2 depends_on(%ready_3, %read_c)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"mm_sk_m0n1_store_y">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %y_store
    nest.return
  }

  // context (1,0)：输出 C[1,0,:,:,:]，4 个 leaf 在本 context 内合并。
  nest.context @mm_sk_m1n0(
      %A : !nest.global_memref<2x4x8x64x64xbf16>,
      %B : !nest.global_memref<2x8x64x256xbf16>,
      %C : !nest.global_memref<2x2x4x64x256xf32>)
      placement = 15 context = 2
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 1835008,
          requested_contexts_per_tile = 4> {
    %a_blk = nest.subview %A
        offsets = [1, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 64] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x64xbf16>
    %b_blk = nest.subview %B
        offsets = [0, 0, 0, 0] sizes = [1, 8, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x8x64x256xbf16>
    %y_blk = nest.subview %C
        offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
        strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xf32>
    %a_l2 = nest.alloc slot = "mm_sk_m1n0_a" role = "in"
        shape = [4, 8, 64, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x64xbf16>
    %b_l2 = nest.alloc slot = "mm_sk_m1n0_b" role = "in"
        shape = [8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<8x64x256xbf16>
    %p0_l2 = nest.alloc slot = "mm_sk_m1n0_p0" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p1_l2 = nest.alloc slot = "mm_sk_m1n0_p1" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p2_l2 = nest.alloc slot = "mm_sk_m1n0_p2" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p3_l2 = nest.alloc slot = "mm_sk_m1n0_p3" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %y_l2 = nest.alloc slot = "mm_sk_m1n0_y" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_sk_m1n0_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_sk_m1n0_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @mm_sk_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p0_l2) ins(%a_l2, %b_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n0_grid_0">, !nest.event<"mm_sk_m1n0_read_0">,
           !nest.event<"mm_sk_m1n0_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @mm_sk_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p1_l2) ins(%a_l2, %b_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n0_grid_1">, !nest.event<"mm_sk_m1n0_read_1">,
           !nest.event<"mm_sk_m1n0_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @mm_sk_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p2_l2) ins(%a_l2, %b_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n0_grid_2">, !nest.event<"mm_sk_m1n0_read_2">,
           !nest.event<"mm_sk_m1n0_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @mm_sk_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p3_l2) ins(%a_l2, %b_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n0_grid_3">, !nest.event<"mm_sk_m1n0_read_3">,
           !nest.event<"mm_sk_m1n0_ready_3">)
    // 同 context 直读四份 partial；不写回 HBM 中转。
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @mm_sk_add l1_mode = 0
        tasks(%tasks) globals()
        bindings(%p0_l2, %p1_l2, %p2_l2, %p3_l2, %y_l2)
        ins(%p0_l2, %p1_l2, %p2_l2, %p3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%ready_0, %ready_1, %ready_2, %ready_3)
        : (!nest.event<"mm_sk_m1n0_grid_c">, !nest.event<"mm_sk_m1n0_read_c">,
           !nest.event<"mm_sk_m1n0_ready_c">)
    nest.release %a_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %a_prefetched)
    nest.release %b_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %b_prefetched)
    nest.release %p0_l2 depends_on(%ready_0, %read_c)
    nest.release %p1_l2 depends_on(%ready_1, %read_c)
    nest.release %p2_l2 depends_on(%ready_2, %read_c)
    nest.release %p3_l2 depends_on(%ready_3, %read_c)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"mm_sk_m1n0_store_y">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %y_store
    nest.return
  }

  // context (1,1)：输出 C[1,1,:,:,:]，4 个 leaf 在本 context 内合并。
  nest.context @mm_sk_m1n1(
      %A : !nest.global_memref<2x4x8x64x64xbf16>,
      %B : !nest.global_memref<2x8x64x256xbf16>,
      %C : !nest.global_memref<2x2x4x64x256xf32>)
      placement = 15 context = 3
                resource_contract = #nest.context_resources<l2_mode = 0,
          allowed_profiles = [0, 1, 2], logical_tasks = 20, l2_spm_bytes = 1835008,
          requested_contexts_per_tile = 4> {
    %a_blk = nest.subview %A
        offsets = [1, 0, 0, 0, 0] sizes = [1, 4, 8, 64, 64] strides = [1, 1, 1, 1, 1]
        : !nest.global_view<1x4x8x64x64xbf16>
    %b_blk = nest.subview %B
        offsets = [1, 0, 0, 0] sizes = [1, 8, 64, 256] strides = [1, 1, 1, 1]
        : !nest.global_view<1x8x64x256xbf16>
    %y_blk = nest.subview %C
        offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
        strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xf32>
    %a_l2 = nest.alloc slot = "mm_sk_m1n1_a" role = "in"
        shape = [4, 8, 64, 64] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x8x64x64xbf16>
    %b_l2 = nest.alloc slot = "mm_sk_m1n1_b" role = "in"
        shape = [8, 64, 256] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<8x64x256xbf16>
    %p0_l2 = nest.alloc slot = "mm_sk_m1n1_p0" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p1_l2 = nest.alloc slot = "mm_sk_m1n1_p1" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p2_l2 = nest.alloc slot = "mm_sk_m1n1_p2" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %p3_l2 = nest.alloc slot = "mm_sk_m1n1_p3" role = "out" sharing = "context-local"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %y_l2 = nest.alloc slot = "mm_sk_m1n1_y" role = "out"
        shape = [4, 64, 256] dtype = "f32" alignment = 256
        : !nest.l2_buffer<4x64x256xf32>
    %a_prefetched = nest.dma.prefetch.async %a_blk into %a_l2
        : !nest.event<"mm_sk_m1n1_a_prefetched">
    %b_prefetched = nest.dma.prefetch.async %b_blk into %b_l2
        : !nest.event<"mm_sk_m1n1_b_prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_0, %read_0, %ready_0 =
        nest.dispatch.tasks.async @mm_sk_q0 l1_mode = 0 context = 0
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p0_l2) ins(%a_l2, %b_l2) outs(%p0_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n1_grid_0">, !nest.event<"mm_sk_m1n1_read_0">,
           !nest.event<"mm_sk_m1n1_ready_0">)
    %grid_1, %read_1, %ready_1 =
        nest.dispatch.tasks.async @mm_sk_q1 l1_mode = 0 context = 1
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p1_l2) ins(%a_l2, %b_l2) outs(%p1_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n1_grid_1">, !nest.event<"mm_sk_m1n1_read_1">,
           !nest.event<"mm_sk_m1n1_ready_1">)
    %grid_2, %read_2, %ready_2 =
        nest.dispatch.tasks.async @mm_sk_q2 l1_mode = 0 context = 2
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p2_l2) ins(%a_l2, %b_l2) outs(%p2_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n1_grid_2">, !nest.event<"mm_sk_m1n1_read_2">,
           !nest.event<"mm_sk_m1n1_ready_2">)
    %grid_3, %read_3, %ready_3 =
        nest.dispatch.tasks.async @mm_sk_q3 l1_mode = 0 context = 3
        tasks(%tasks) globals()
        bindings(%a_l2, %b_l2, %p3_l2) ins(%a_l2, %b_l2) outs(%p3_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%a_prefetched, %b_prefetched)
        : (!nest.event<"mm_sk_m1n1_grid_3">, !nest.event<"mm_sk_m1n1_read_3">,
           !nest.event<"mm_sk_m1n1_ready_3">)
    // 同 context 直读四份 partial；不写回 HBM 中转。
    %grid_c, %read_c, %ready_c =
        nest.dispatch.tasks.async @mm_sk_add l1_mode = 0
        tasks(%tasks) globals()
        bindings(%p0_l2, %p1_l2, %p2_l2, %p3_l2, %y_l2)
        ins(%p0_l2, %p1_l2, %p2_l2, %p3_l2) outs(%y_l2)
        signal_policy {
          input_released = #nest.aggregate<all_tasks>,
          output_ready = #nest.aggregate<all_tasks>
        }
        depends_on(%ready_0, %ready_1, %ready_2, %ready_3)
        : (!nest.event<"mm_sk_m1n1_grid_c">, !nest.event<"mm_sk_m1n1_read_c">,
           !nest.event<"mm_sk_m1n1_ready_c">)
    nest.release %a_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %a_prefetched)
    nest.release %b_l2 depends_on(%read_0, %read_1, %read_2, %read_3,
                                  %b_prefetched)
    nest.release %p0_l2 depends_on(%ready_0, %read_c)
    nest.release %p1_l2 depends_on(%ready_1, %read_c)
    nest.release %p2_l2 depends_on(%ready_2, %read_c)
    nest.release %p3_l2 depends_on(%ready_3, %read_c)
    %y_store = nest.dma.store.async %y_l2 into %y_blk
        depends_on(%ready_c) : !nest.event<"mm_sk_m1n1_store_y">
    nest.release %y_l2 depends_on(%y_store)
    nest.await %grid_0, %grid_1, %grid_2, %grid_3
    nest.await %grid_c, %y_store
    nest.return
  }

  nexus.program @matmul_splitk_multicontext_pipeline(
      %A : !nest.global_memref<2x4x8x64x64xbf16>,
      %B : !nest.global_memref<2x8x64x256xbf16>,
      %C : !nest.global_memref<2x2x4x64x256xf32>) {
    %m0n0_done = nexus.submit_context.async @mm_sk_m0n0(%A, %B, %C)
        : !nexus.event<"m0n0_done">
    %m0n1_done = nexus.submit_context.async @mm_sk_m0n1(%A, %B, %C)
        : !nexus.event<"m0n1_done">
    %m1n0_done = nexus.submit_context.async @mm_sk_m1n0(%A, %B, %C)
        : !nexus.event<"m1n0_done">
    %m1n1_done = nexus.submit_context.async @mm_sk_m1n1(%A, %B, %C)
        : !nexus.event<"m1n1_done">
    nexus.await %m0n0_done, %m0n1_done, %m1n0_done, %m1n1_done
    nexus.return
  }
}
