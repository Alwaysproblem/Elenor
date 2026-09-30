// Transformer Prefill Attention block (one device root, GQA 16:4, BF16).
//
// Shapes: seq=512, hidden=1024, q_heads=16, kv_heads=4, head_dim=64;
// QUERY_BLOCK=128, KV_BLOCK=128, QKV projection K-chunk=128.
// Tile t (placement 15) owns KV head t and Q heads 4t..4t+3.
//
// TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor
// operands and execute no numerics.  This workload validates Group
// ready-action scheduling, L2 lifetimes (input_released / output_ready),
// HBM->L2->L1 streaming and BOA/EVU/MFE overlap -- NOT numerical
// correctness of Transformer attention.
//
// Pipelined variant: per K-chunk c the X/WQ/WK/WV prefetch (L2
// ping/pong, gated on chunk c-2 input_released) feeds qkv_chunk_init
// (chunk 0) / qkv_chunk_accum (chunks 1..7, read-modify-write of the
// L2 partial sums); attention runs 4 query blocks x 4 KV blocks with
// L1 KV ping/pong (per-head BOA QK + EVU online softmax + BOA PV); the
// Wo prefetch starts at the last chunk's input_released and overlaps
// attention; the N-split output projection consumes O and Wo, then one
// HBM store.  Baseline variant: identical buffers and traffic, but
// every stage is fenced with per-chunk nest.await and tile programs
// issue all loads of a stage before any compute.
//
// Useful MACs: QKV 512x1024x1536; attention 4 tiles x 16 x 4 heads x
// (128x128x64 + 128x64x128); output projection 512x1024x1024.
//
// Run: bash examples/run.sh transformer-prefill-attention

builtin.module {
  tile.program @qkv_chunk_init(
    %task: !nest.task, %x_chunk: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk: !nest.l2_buffer<4x128x256xbf16>, %wk_chunk: !nest.l2_buffer<4x128x64xbf16>,
    %wv_chunk: !nest.l2_buffer<4x128x64xbf16>, %q_l2: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2: !nest.l2_buffer<4x512x64xbf16>, %v_l2: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 622592> {
    %0 = tile.subview %x_chunk offsets = [0, 0] sizes = [512, 128] strides = [1, 1]
      : !nest.l2_view<512x128xbf16>
    %1 = tile.subview %wq_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %2 = tile.subview %wk_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %3 = tile.subview %wv_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %4 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x512x256xbf16>
    %5 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x512x64xbf16>
    %6 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x512x64xbf16>
    %x_buf = tile.alloc shape = [512, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x128xbf16>
    %q_acc = tile.alloc shape = [512, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x256xbf16>
    %wq_buf = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %wk_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %k_acc = tile.alloc shape = [512, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x64xbf16>
    %wv_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc = tile.alloc shape = [512, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x64xbf16>
    %x_loaded = tile.load.async %0 into %x_buf : !tile.event<"x_loaded">
    %wq_loaded = tile.load.async %1 into %wq_buf : !tile.event<"wq_loaded">
    %wk_loaded = tile.load.async %2 into %wk_buf : !tile.event<"wk_loaded">
    %wv_loaded = tile.load.async %3 into %wv_buf : !tile.event<"wv_loaded">
    tile.await %x_loaded, %wq_loaded, %wk_loaded, %wv_loaded
    tile.signal input_released(%task)
    %q_boa = tile.boa.async "matmul" m = 512 n = 256 k = 128 ops = 33554432 : !tile.event<"q_boa">
    tile.await %q_boa
    %q_stored = tile.store.async %q_acc into %4 : !tile.event<"q_stored">
    tile.await %q_stored
    %k_boa = tile.boa.async "matmul" m = 512 n = 64 k = 128 ops = 8388608 : !tile.event<"k_boa">
    tile.await %k_boa
    %k_stored = tile.store.async %k_acc into %5 : !tile.event<"k_stored">
    tile.await %k_stored
    %v_boa = tile.boa.async "matmul" m = 512 n = 64 k = 128 ops = 8388608 : !tile.event<"v_boa">
    tile.await %v_boa
    %v_stored = tile.store.async %v_acc into %6 : !tile.event<"v_stored">
    tile.await %v_stored
    tile.signal output_ready(%task)
    tile.free %x_buf
    tile.free %q_acc
    tile.free %wq_buf
    tile.free %wk_buf
    tile.free %k_acc
    tile.free %wv_buf
    tile.free %v_acc
    tile.return
  }
  tile.program @qkv_chunk_accum(
    %task_1: !nest.task, %x_chunk_1: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk_1: !nest.l2_buffer<4x128x256xbf16>, %wk_chunk_1: !nest.l2_buffer<4x128x64xbf16>,
    %wv_chunk_1: !nest.l2_buffer<4x128x64xbf16>, %q_l2_1: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_1: !nest.l2_buffer<4x512x64xbf16>, %v_l2_1: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 622592> {
    %7 = tile.subview %x_chunk_1 offsets = [0, 0] sizes = [512, 128] strides = [1, 1]
      : !nest.l2_view<512x128xbf16>
    %8 = tile.subview %wq_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %9 = tile.subview %wk_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %10 = tile.subview %wv_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %11 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x512x256xbf16>
    %12 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x512x64xbf16>
    %13 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x512x64xbf16>
    %x_buf_1 = tile.alloc shape = [512, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x128xbf16>
    %q_acc_1 = tile.alloc shape = [512, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x256xbf16>
    %wq_buf_1 = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %wk_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %k_acc_1 = tile.alloc shape = [512, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x64xbf16>
    %wv_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc_1 = tile.alloc shape = [512, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x64xbf16>
    %x_loaded_1 = tile.load.async %7 into %x_buf_1 : !tile.event<"x_loaded">
    %wq_loaded_1 = tile.load.async %8 into %wq_buf_1 : !tile.event<"wq_loaded">
    %wk_loaded_1 = tile.load.async %9 into %wk_buf_1 : !tile.event<"wk_loaded">
    %wv_loaded_1 = tile.load.async %10 into %wv_buf_1 : !tile.event<"wv_loaded">
    %q_partial_loaded = tile.load.async %11 into %q_acc_1 : !tile.event<"q_partial_loaded">
    %k_partial_loaded = tile.load.async %12 into %k_acc_1 : !tile.event<"k_partial_loaded">
    %v_partial_loaded = tile.load.async %13 into %v_acc_1 : !tile.event<"v_partial_loaded">
    tile.await %x_loaded_1, %wq_loaded_1, %wk_loaded_1, %wv_loaded_1, %q_partial_loaded,
      %k_partial_loaded, %v_partial_loaded
    tile.signal input_released(%task_1)
    %q_boa_1 = tile.boa.async "matmul" m = 512 n = 256 k = 128 ops = 33554432 accumulate
      : !tile.event<"q_boa">
    tile.await %q_boa_1
    %q_stored_1 = tile.store.async %q_acc_1 into %11 : !tile.event<"q_stored">
    tile.await %q_stored_1
    %k_boa_1 = tile.boa.async "matmul" m = 512 n = 64 k = 128 ops = 8388608 accumulate
      : !tile.event<"k_boa">
    tile.await %k_boa_1
    %k_stored_1 = tile.store.async %k_acc_1 into %12 : !tile.event<"k_stored">
    tile.await %k_stored_1
    %v_boa_1 = tile.boa.async "matmul" m = 512 n = 64 k = 128 ops = 8388608 accumulate
      : !tile.event<"v_boa">
    tile.await %v_boa_1
    %v_stored_1 = tile.store.async %v_acc_1 into %13 : !tile.event<"v_stored">
    tile.await %v_stored_1
    tile.signal output_ready(%task_1)
    tile.free %x_buf_1
    tile.free %q_acc_1
    tile.free %wq_buf_1
    tile.free %wk_buf_1
    tile.free %k_acc_1
    tile.free %wv_buf_1
    tile.free %v_acc_1
    tile.return
  }
  tile.program @prefill_attention_tile(
    %task_2: !nest.task, %q_l2_2: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_2: !nest.l2_buffer<4x512x64xbf16>, %v_l2_2: !nest.l2_buffer<4x512x64xbf16>,
    %o_l2: !nest.l2_buffer<4x512x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 270336> {
    %q_buf = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %acc = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %m = tile.alloc shape = [128] dtype = "f32" alignment = 256 : !tile.l1_buffer<128xf32>
    %l = tile.alloc shape = [128] dtype = "f32" alignment = 256 : !tile.l1_buffer<128xf32>
    %k0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %k1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %k2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %k3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %14 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %q0_loaded = tile.load.async %14 into %q_buf : !tile.event<"q0_loaded">
    %15 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %16 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded = tile.load.async %15 into %k0 : !tile.event<"k0_loaded">
    %v0_loaded = tile.load.async %16 into %v0 : !tile.event<"v0_loaded">
    %17 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %18 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded = tile.load.async %17 into %k1 : !tile.event<"k1_loaded">
    %v1_loaded = tile.load.async %18 into %v1 : !tile.event<"v1_loaded">
    %19 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %20 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded = tile.load.async %19 into %k2 : !tile.event<"k2_loaded">
    %v2_loaded = tile.load.async %20 into %v2 : !tile.event<"v2_loaded">
    %21 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %22 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded = tile.load.async %21 into %k3 : !tile.event<"k3_loaded">
    %v3_loaded = tile.load.async %22 into %v3 : !tile.event<"v3_loaded">
    tile.await %q0_loaded, %k0_loaded, %v0_loaded, %k1_loaded, %v1_loaded, %k2_loaded, %v2_loaded,
      %k3_loaded, %v3_loaded
    %qk0_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk0_h0">
    %qk0_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk0_h1">
    %qk0_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk0_h2">
    %qk0_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk0_h3">
    tile.await %qk0_h0, %qk0_h1, %qk0_h2, %qk0_h3
    %sm0 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm0">
    tile.await %sm0
    %pv0_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 : !tile.event<"pv0_h0">
    %pv0_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 : !tile.event<"pv0_h1">
    %pv0_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 : !tile.event<"pv0_h2">
    %pv0_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 : !tile.event<"pv0_h3">
    tile.await %pv0_h0, %pv0_h1, %pv0_h2, %pv0_h3
    %qk1_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk1_h0">
    %qk1_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk1_h1">
    %qk1_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk1_h2">
    %qk1_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk1_h3">
    tile.await %qk1_h0, %qk1_h1, %qk1_h2, %qk1_h3
    %sm1 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm1">
    tile.await %sm1
    %pv1_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv1_h0">
    %pv1_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv1_h1">
    %pv1_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv1_h2">
    %pv1_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv1_h3">
    tile.await %pv1_h0, %pv1_h1, %pv1_h2, %pv1_h3
    %qk2_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk2_h0">
    %qk2_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk2_h1">
    %qk2_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk2_h2">
    %qk2_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk2_h3">
    tile.await %qk2_h0, %qk2_h1, %qk2_h2, %qk2_h3
    %sm2 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm2">
    tile.await %sm2
    %pv2_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv2_h0">
    %pv2_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv2_h1">
    %pv2_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv2_h2">
    %pv2_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv2_h3">
    tile.await %pv2_h0, %pv2_h1, %pv2_h2, %pv2_h3
    %qk3_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk3_h0">
    %qk3_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk3_h1">
    %qk3_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk3_h2">
    %qk3_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk3_h3">
    tile.await %qk3_h0, %qk3_h1, %qk3_h2, %qk3_h3
    %sm3 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm3">
    tile.await %sm3
    %pv3_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv3_h0">
    %pv3_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv3_h1">
    %pv3_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv3_h2">
    %pv3_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv3_h3">
    tile.await %pv3_h0, %pv3_h1, %pv3_h2, %pv3_h3
    %23 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %o0_stored = tile.store.async %acc into %23 : !tile.event<"o0_stored">
    tile.await %o0_stored
    %24 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %q1_loaded = tile.load.async %24 into %q_buf : !tile.event<"q1_loaded">
    %25 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %26 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k4_loaded = tile.load.async %25 into %k0 : !tile.event<"k4_loaded">
    %v4_loaded = tile.load.async %26 into %v0 : !tile.event<"v4_loaded">
    %27 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %28 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k5_loaded = tile.load.async %27 into %k1 : !tile.event<"k5_loaded">
    %v5_loaded = tile.load.async %28 into %v1 : !tile.event<"v5_loaded">
    %29 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %30 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k6_loaded = tile.load.async %29 into %k2 : !tile.event<"k6_loaded">
    %v6_loaded = tile.load.async %30 into %v2 : !tile.event<"v6_loaded">
    %31 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %32 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k7_loaded = tile.load.async %31 into %k3 : !tile.event<"k7_loaded">
    %v7_loaded = tile.load.async %32 into %v3 : !tile.event<"v7_loaded">
    tile.await %q1_loaded, %k4_loaded, %v4_loaded, %k5_loaded, %v5_loaded, %k6_loaded, %v6_loaded,
      %k7_loaded, %v7_loaded
    %qk4_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk4_h0">
    %qk4_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk4_h1">
    %qk4_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk4_h2">
    %qk4_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk4_h3">
    tile.await %qk4_h0, %qk4_h1, %qk4_h2, %qk4_h3
    %sm4 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm4">
    tile.await %sm4
    %pv4_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv4_h0">
    %pv4_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv4_h1">
    %pv4_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv4_h2">
    %pv4_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv4_h3">
    tile.await %pv4_h0, %pv4_h1, %pv4_h2, %pv4_h3
    %qk5_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk5_h0">
    %qk5_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk5_h1">
    %qk5_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk5_h2">
    %qk5_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk5_h3">
    tile.await %qk5_h0, %qk5_h1, %qk5_h2, %qk5_h3
    %sm5 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm5">
    tile.await %sm5
    %pv5_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv5_h0">
    %pv5_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv5_h1">
    %pv5_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv5_h2">
    %pv5_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv5_h3">
    tile.await %pv5_h0, %pv5_h1, %pv5_h2, %pv5_h3
    %qk6_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk6_h0">
    %qk6_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk6_h1">
    %qk6_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk6_h2">
    %qk6_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk6_h3">
    tile.await %qk6_h0, %qk6_h1, %qk6_h2, %qk6_h3
    %sm6 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm6">
    tile.await %sm6
    %pv6_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv6_h0">
    %pv6_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv6_h1">
    %pv6_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv6_h2">
    %pv6_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv6_h3">
    tile.await %pv6_h0, %pv6_h1, %pv6_h2, %pv6_h3
    %qk7_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk7_h0">
    %qk7_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk7_h1">
    %qk7_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk7_h2">
    %qk7_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk7_h3">
    tile.await %qk7_h0, %qk7_h1, %qk7_h2, %qk7_h3
    %sm7 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm7">
    tile.await %sm7
    %pv7_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv7_h0">
    %pv7_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv7_h1">
    %pv7_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv7_h2">
    %pv7_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv7_h3">
    tile.await %pv7_h0, %pv7_h1, %pv7_h2, %pv7_h3
    %33 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %o1_stored = tile.store.async %acc into %33 : !tile.event<"o1_stored">
    tile.await %o1_stored
    %34 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %q2_loaded = tile.load.async %34 into %q_buf : !tile.event<"q2_loaded">
    %35 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %36 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k8_loaded = tile.load.async %35 into %k0 : !tile.event<"k8_loaded">
    %v8_loaded = tile.load.async %36 into %v0 : !tile.event<"v8_loaded">
    %37 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %38 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k9_loaded = tile.load.async %37 into %k1 : !tile.event<"k9_loaded">
    %v9_loaded = tile.load.async %38 into %v1 : !tile.event<"v9_loaded">
    %39 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %40 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k10_loaded = tile.load.async %39 into %k2 : !tile.event<"k10_loaded">
    %v10_loaded = tile.load.async %40 into %v2 : !tile.event<"v10_loaded">
    %41 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %42 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k11_loaded = tile.load.async %41 into %k3 : !tile.event<"k11_loaded">
    %v11_loaded = tile.load.async %42 into %v3 : !tile.event<"v11_loaded">
    tile.await %q2_loaded, %k8_loaded, %v8_loaded, %k9_loaded, %v9_loaded, %k10_loaded, %v10_loaded,
      %k11_loaded, %v11_loaded
    %qk8_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk8_h0">
    %qk8_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk8_h1">
    %qk8_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk8_h2">
    %qk8_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk8_h3">
    tile.await %qk8_h0, %qk8_h1, %qk8_h2, %qk8_h3
    %sm8 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm8">
    tile.await %sm8
    %pv8_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv8_h0">
    %pv8_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv8_h1">
    %pv8_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv8_h2">
    %pv8_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv8_h3">
    tile.await %pv8_h0, %pv8_h1, %pv8_h2, %pv8_h3
    %qk9_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk9_h0">
    %qk9_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk9_h1">
    %qk9_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk9_h2">
    %qk9_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk9_h3">
    tile.await %qk9_h0, %qk9_h1, %qk9_h2, %qk9_h3
    %sm9 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm9">
    tile.await %sm9
    %pv9_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv9_h0">
    %pv9_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv9_h1">
    %pv9_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv9_h2">
    %pv9_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv9_h3">
    tile.await %pv9_h0, %pv9_h1, %pv9_h2, %pv9_h3
    %qk10_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk10_h0">
    %qk10_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk10_h1">
    %qk10_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk10_h2">
    %qk10_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk10_h3">
    tile.await %qk10_h0, %qk10_h1, %qk10_h2, %qk10_h3
    %sm10 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm10">
    tile.await %sm10
    %pv10_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv10_h0">
    %pv10_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv10_h1">
    %pv10_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv10_h2">
    %pv10_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv10_h3">
    tile.await %pv10_h0, %pv10_h1, %pv10_h2, %pv10_h3
    %qk11_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk11_h0">
    %qk11_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk11_h1">
    %qk11_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk11_h2">
    %qk11_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk11_h3">
    tile.await %qk11_h0, %qk11_h1, %qk11_h2, %qk11_h3
    %sm11 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm11">
    tile.await %sm11
    %pv11_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv11_h0">
    %pv11_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv11_h1">
    %pv11_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv11_h2">
    %pv11_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv11_h3">
    tile.await %pv11_h0, %pv11_h1, %pv11_h2, %pv11_h3
    %43 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %o2_stored = tile.store.async %acc into %43 : !tile.event<"o2_stored">
    tile.await %o2_stored
    %44 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %q3_loaded = tile.load.async %44 into %q_buf : !tile.event<"q3_loaded">
    %45 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %46 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k12_loaded = tile.load.async %45 into %k0 : !tile.event<"k12_loaded">
    %v12_loaded = tile.load.async %46 into %v0 : !tile.event<"v12_loaded">
    %47 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %48 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k13_loaded = tile.load.async %47 into %k1 : !tile.event<"k13_loaded">
    %v13_loaded = tile.load.async %48 into %v1 : !tile.event<"v13_loaded">
    %49 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %50 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k14_loaded = tile.load.async %49 into %k2 : !tile.event<"k14_loaded">
    %v14_loaded = tile.load.async %50 into %v2 : !tile.event<"v14_loaded">
    %51 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %52 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k15_loaded = tile.load.async %51 into %k3 : !tile.event<"k15_loaded">
    %v15_loaded = tile.load.async %52 into %v3 : !tile.event<"v15_loaded">
    tile.await %q3_loaded, %k12_loaded, %v12_loaded, %k13_loaded, %v13_loaded, %k14_loaded,
      %v14_loaded, %k15_loaded, %v15_loaded
    %qk12_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk12_h0">
    %qk12_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk12_h1">
    %qk12_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk12_h2">
    %qk12_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk12_h3">
    tile.await %qk12_h0, %qk12_h1, %qk12_h2, %qk12_h3
    %sm12 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm12">
    tile.await %sm12
    %pv12_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv12_h0">
    %pv12_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv12_h1">
    %pv12_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv12_h2">
    %pv12_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv12_h3">
    tile.await %pv12_h0, %pv12_h1, %pv12_h2, %pv12_h3
    %qk13_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk13_h0">
    %qk13_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk13_h1">
    %qk13_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk13_h2">
    %qk13_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk13_h3">
    tile.await %qk13_h0, %qk13_h1, %qk13_h2, %qk13_h3
    %sm13 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm13">
    tile.await %sm13
    %pv13_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv13_h0">
    %pv13_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv13_h1">
    %pv13_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv13_h2">
    %pv13_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv13_h3">
    tile.await %pv13_h0, %pv13_h1, %pv13_h2, %pv13_h3
    %qk14_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk14_h0">
    %qk14_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk14_h1">
    %qk14_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk14_h2">
    %qk14_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk14_h3">
    tile.await %qk14_h0, %qk14_h1, %qk14_h2, %qk14_h3
    %sm14 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm14">
    tile.await %sm14
    %pv14_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv14_h0">
    %pv14_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv14_h1">
    %pv14_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv14_h2">
    %pv14_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv14_h3">
    tile.await %pv14_h0, %pv14_h1, %pv14_h2, %pv14_h3
    %qk15_h0 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk15_h0">
    %qk15_h1 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk15_h1">
    %qk15_h2 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk15_h2">
    %qk15_h3 = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"qk15_h3">
    tile.await %qk15_h0, %qk15_h1, %qk15_h2, %qk15_h3
    %sm15 = tile.evu.async "online_softmax_update" ops = 99328 : !tile.event<"sm15">
    tile.await %sm15
    %pv15_h0 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv15_h0">
    %pv15_h1 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv15_h1">
    %pv15_h2 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv15_h2">
    %pv15_h3 = tile.boa.async "matmul" m = 128 n = 64 k = 128 ops = 2097152 accumulate
      : !tile.event<"pv15_h3">
    tile.await %pv15_h0, %pv15_h1, %pv15_h2, %pv15_h3
    %53 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 384, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %o3_stored = tile.store.async %acc into %53 : !tile.event<"o3_stored">
    tile.await %o3_stored
    tile.signal input_released(%task_2)
    tile.signal output_ready(%task_2)
    tile.return
  }
  tile.program @prefill_outproj_tile(
    %task_3: !nest.task, %o_l2_1: !nest.l2_buffer<4x512x256xbf16>,
    %wo_l2: !nest.l2_buffer<4x1024x256xbf16>, %out_l2: !nest.l2_buffer<4x512x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 655360> {
    %acc_1 = tile.alloc shape = [512, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x256xbf16>
    %o0 = tile.alloc shape = [512, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<512x256xbf16>
    %w = tile.alloc shape = [256, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<256x256xbf16>
    %54 = tile.subview %out_l2 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 512, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x512x256xbf16>
    %55 = tile.subview %o_l2_1 offsets = [0, 0, 0] sizes = [1, 512, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x512x256xbf16>
    %56 = tile.subview %wo_l2 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 256, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x256x256xbf16>
    %o0_loaded = tile.load.async %55 into %o0 : !tile.event<"o0_loaded">
    %w0_loaded = tile.load.async %56 into %w : !tile.event<"w0_loaded">
    tile.await %o0_loaded, %w0_loaded
    %pj0 = tile.boa.async "matmul" m = 512 n = 256 k = 256 ops = 67108864 : !tile.event<"pj0">
    tile.await %pj0
    %57 = tile.subview %o_l2_1 offsets = [1, 0, 0] sizes = [1, 512, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x512x256xbf16>
    %58 = tile.subview %wo_l2 task = %task_3 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 256, 256] strides = [1, 1, 1] : !nest.l2_view<1x256x256xbf16>
    %o1_loaded = tile.load.async %57 into %o0 : !tile.event<"o1_loaded">
    %w1_loaded = tile.load.async %58 into %w : !tile.event<"w1_loaded">
    tile.await %o1_loaded, %w1_loaded
    %pj1 = tile.boa.async "matmul" m = 512 n = 256 k = 256 ops = 67108864 accumulate
      : !tile.event<"pj1">
    tile.await %pj1
    %59 = tile.subview %o_l2_1 offsets = [2, 0, 0] sizes = [1, 512, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x512x256xbf16>
    %60 = tile.subview %wo_l2 task = %task_3 task_dim = 0 offsets = [0, 512, 0]
      sizes = [1, 256, 256] strides = [1, 1, 1] : !nest.l2_view<1x256x256xbf16>
    %o2_loaded = tile.load.async %59 into %o0 : !tile.event<"o2_loaded">
    %w2_loaded = tile.load.async %60 into %w : !tile.event<"w2_loaded">
    tile.await %o2_loaded, %w2_loaded
    %pj2 = tile.boa.async "matmul" m = 512 n = 256 k = 256 ops = 67108864 accumulate
      : !tile.event<"pj2">
    tile.await %pj2
    %61 = tile.subview %o_l2_1 offsets = [3, 0, 0] sizes = [1, 512, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x512x256xbf16>
    %62 = tile.subview %wo_l2 task = %task_3 task_dim = 0 offsets = [0, 768, 0]
      sizes = [1, 256, 256] strides = [1, 1, 1] : !nest.l2_view<1x256x256xbf16>
    %o3_loaded = tile.load.async %61 into %o0 : !tile.event<"o3_loaded">
    %w3_loaded = tile.load.async %62 into %w : !tile.event<"w3_loaded">
    tile.await %o3_loaded, %w3_loaded
    %pj3 = tile.boa.async "matmul" m = 512 n = 256 k = 256 ops = 67108864 accumulate
      : !tile.event<"pj3">
    tile.await %pj3
    tile.signal input_released(%task_3)
    %out_stored = tile.store.async %acc_1 into %54 : !tile.event<"out_stored">
    tile.await %out_stored
    tile.signal output_ready(%task_3)
    tile.return
  }
  nest.context @prefill_ctx(
    %X: !nest.global_memref<8x512x128xbf16>, %WQ: !nest.global_memref<8x4x128x256xbf16>,
    %WK: !nest.global_memref<8x4x128x64xbf16>, %WV: !nest.global_memref<8x4x128x64xbf16>,
    %WO: !nest.global_memref<4x1024x256xbf16>, %OUT: !nest.global_memref<4x512x256xbf16>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1],
      logical_tasks = 40, l2_spm_bytes = 6815744, requested_contexts_per_tile = 1> {
    %x_p0 = nest.alloc slot = "x_p0" role = "in" shape = [512, 128] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<512x128xbf16>
    %x_p1 = nest.alloc slot = "x_p1" role = "in" shape = [512, 128] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<512x128xbf16>
    %wq_p0 = nest.alloc slot = "wq_p0" role = "in" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %wq_p1 = nest.alloc slot = "wq_p1" role = "in" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %wk_p0 = nest.alloc slot = "wk_p0" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %wk_p1 = nest.alloc slot = "wk_p1" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %wv_p0 = nest.alloc slot = "wv_p0" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %wv_p1 = nest.alloc slot = "wv_p1" role = "in" shape = [4, 128, 64] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x64xbf16>
    %q_l2_3 = nest.alloc slot = "q_l2" role = "inout" sharing = "context-local"
      shape = [4, 512, 256] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x256xbf16>
    %k_l2_3 = nest.alloc slot = "k_l2" role = "inout" sharing = "context-local" shape = [4, 512, 64]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x64xbf16>
    %v_l2_3 = nest.alloc slot = "v_l2" role = "inout" sharing = "context-local" shape = [4, 512, 64]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x64xbf16>
    %wo_l2_1 = nest.alloc slot = "wo_l2" role = "in" shape = [4, 1024, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x256xbf16>
    %o_l2_2 = nest.alloc slot = "o_l2" role = "inout" sharing = "context-local"
      shape = [4, 512, 256] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x256xbf16>
    %out_l2_1 = nest.alloc slot = "out_l2" role = "out" shape = [4, 512, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x512x256xbf16>
    %63 = nest.task.range from = 0 to = 4 : !nest.task_range
    %64 = nest.subview %X offsets = [0, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %65 = nest.subview %WQ offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %66 = nest.subview %WK offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %67 = nest.subview %WV offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x = nest.dma.prefetch.async %64 into %x_p0 : !nest.event<"pre_x_0">
    %pre_wq = nest.dma.prefetch.async %65 into %wq_p0 : !nest.event<"pre_wq_0">
    %pre_wk = nest.dma.prefetch.async %66 into %wk_p0 : !nest.event<"pre_wk_0">
    %pre_wv = nest.dma.prefetch.async %67 into %wv_p0 : !nest.event<"pre_wv_0">
    nest.await %pre_x, %pre_wq, %pre_wk, %pre_wv
    %d0_grid, %d0_inrel, %d0_out = nest.dispatch.tasks.async @qkv_chunk_init l1_mode = 0 tasks(%63)
      globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x, %pre_wq, %pre_wk, %pre_wv)
      : (!nest.event<"d0_grid">, !nest.event<"d0_inrel">, !nest.event<"d0_out">)
    nest.await %d0_grid
    %68 = nest.subview %X offsets = [1, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %69 = nest.subview %WQ offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %70 = nest.subview %WK offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %71 = nest.subview %WV offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_1 = nest.dma.prefetch.async %68 into %x_p1 : !nest.event<"pre_x_1">
    %pre_wq_1 = nest.dma.prefetch.async %69 into %wq_p1 : !nest.event<"pre_wq_1">
    %pre_wk_1 = nest.dma.prefetch.async %70 into %wk_p1 : !nest.event<"pre_wk_1">
    %pre_wv_1 = nest.dma.prefetch.async %71 into %wv_p1 : !nest.event<"pre_wv_1">
    nest.await %pre_x_1, %pre_wq_1, %pre_wk_1, %pre_wv_1
    %d1_grid, %d1_inrel, %d1_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_1, %pre_wq_1, %pre_wk_1, %pre_wv_1, %d0_out)
      : (!nest.event<"d1_grid">, !nest.event<"d1_inrel">, !nest.event<"d1_out">)
    nest.await %d1_grid
    %72 = nest.subview %X offsets = [2, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %73 = nest.subview %WQ offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %74 = nest.subview %WK offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %75 = nest.subview %WV offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_2 = nest.dma.prefetch.async %72 into %x_p0 : !nest.event<"pre_x_2">
    %pre_wq_2 = nest.dma.prefetch.async %73 into %wq_p0 : !nest.event<"pre_wq_2">
    %pre_wk_2 = nest.dma.prefetch.async %74 into %wk_p0 : !nest.event<"pre_wk_2">
    %pre_wv_2 = nest.dma.prefetch.async %75 into %wv_p0 : !nest.event<"pre_wv_2">
    nest.await %pre_x_2, %pre_wq_2, %pre_wk_2, %pre_wv_2
    %d2_grid, %d2_inrel, %d2_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_2, %pre_wq_2, %pre_wk_2, %pre_wv_2, %d1_out)
      : (!nest.event<"d2_grid">, !nest.event<"d2_inrel">, !nest.event<"d2_out">)
    nest.await %d2_grid
    %76 = nest.subview %X offsets = [3, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %77 = nest.subview %WQ offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %78 = nest.subview %WK offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %79 = nest.subview %WV offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_3 = nest.dma.prefetch.async %76 into %x_p1 : !nest.event<"pre_x_3">
    %pre_wq_3 = nest.dma.prefetch.async %77 into %wq_p1 : !nest.event<"pre_wq_3">
    %pre_wk_3 = nest.dma.prefetch.async %78 into %wk_p1 : !nest.event<"pre_wk_3">
    %pre_wv_3 = nest.dma.prefetch.async %79 into %wv_p1 : !nest.event<"pre_wv_3">
    nest.await %pre_x_3, %pre_wq_3, %pre_wk_3, %pre_wv_3
    %d3_grid, %d3_inrel, %d3_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_3, %pre_wq_3, %pre_wk_3, %pre_wv_3, %d2_out)
      : (!nest.event<"d3_grid">, !nest.event<"d3_inrel">, !nest.event<"d3_out">)
    nest.await %d3_grid
    %80 = nest.subview %X offsets = [4, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %81 = nest.subview %WQ offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %82 = nest.subview %WK offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %83 = nest.subview %WV offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_4 = nest.dma.prefetch.async %80 into %x_p0 : !nest.event<"pre_x_4">
    %pre_wq_4 = nest.dma.prefetch.async %81 into %wq_p0 : !nest.event<"pre_wq_4">
    %pre_wk_4 = nest.dma.prefetch.async %82 into %wk_p0 : !nest.event<"pre_wk_4">
    %pre_wv_4 = nest.dma.prefetch.async %83 into %wv_p0 : !nest.event<"pre_wv_4">
    nest.await %pre_x_4, %pre_wq_4, %pre_wk_4, %pre_wv_4
    %d4_grid, %d4_inrel, %d4_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_4, %pre_wq_4, %pre_wk_4, %pre_wv_4, %d3_out)
      : (!nest.event<"d4_grid">, !nest.event<"d4_inrel">, !nest.event<"d4_out">)
    nest.await %d4_grid
    %84 = nest.subview %X offsets = [5, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %85 = nest.subview %WQ offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %86 = nest.subview %WK offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %87 = nest.subview %WV offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_5 = nest.dma.prefetch.async %84 into %x_p1 : !nest.event<"pre_x_5">
    %pre_wq_5 = nest.dma.prefetch.async %85 into %wq_p1 : !nest.event<"pre_wq_5">
    %pre_wk_5 = nest.dma.prefetch.async %86 into %wk_p1 : !nest.event<"pre_wk_5">
    %pre_wv_5 = nest.dma.prefetch.async %87 into %wv_p1 : !nest.event<"pre_wv_5">
    nest.await %pre_x_5, %pre_wq_5, %pre_wk_5, %pre_wv_5
    %d5_grid, %d5_inrel, %d5_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_5, %pre_wq_5, %pre_wk_5, %pre_wv_5, %d4_out)
      : (!nest.event<"d5_grid">, !nest.event<"d5_inrel">, !nest.event<"d5_out">)
    nest.await %d5_grid
    %88 = nest.subview %X offsets = [6, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %89 = nest.subview %WQ offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %90 = nest.subview %WK offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %91 = nest.subview %WV offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_6 = nest.dma.prefetch.async %88 into %x_p0 : !nest.event<"pre_x_6">
    %pre_wq_6 = nest.dma.prefetch.async %89 into %wq_p0 : !nest.event<"pre_wq_6">
    %pre_wk_6 = nest.dma.prefetch.async %90 into %wk_p0 : !nest.event<"pre_wk_6">
    %pre_wv_6 = nest.dma.prefetch.async %91 into %wv_p0 : !nest.event<"pre_wv_6">
    nest.await %pre_x_6, %pre_wq_6, %pre_wk_6, %pre_wv_6
    %d6_grid, %d6_inrel, %d6_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_6, %pre_wq_6, %pre_wk_6, %pre_wv_6, %d5_out)
      : (!nest.event<"d6_grid">, !nest.event<"d6_inrel">, !nest.event<"d6_out">)
    nest.await %d6_grid
    %92 = nest.subview %X offsets = [7, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %93 = nest.subview %WQ offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %94 = nest.subview %WK offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %95 = nest.subview %WV offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_7 = nest.dma.prefetch.async %92 into %x_p1 : !nest.event<"pre_x_7">
    %pre_wq_7 = nest.dma.prefetch.async %93 into %wq_p1 : !nest.event<"pre_wq_7">
    %pre_wk_7 = nest.dma.prefetch.async %94 into %wk_p1 : !nest.event<"pre_wk_7">
    %pre_wv_7 = nest.dma.prefetch.async %95 into %wv_p1 : !nest.event<"pre_wv_7">
    nest.await %pre_x_7, %pre_wq_7, %pre_wk_7, %pre_wv_7
    %d7_grid, %d7_inrel, %d7_out = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0 tasks(%63)
      globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_3, %k_l2_3, %v_l2_3) outs(%q_l2_3, %k_l2_3, %v_l2_3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_7, %pre_wq_7, %pre_wk_7, %pre_wv_7, %d6_out)
      : (!nest.event<"d7_grid">, !nest.event<"d7_inrel">, !nest.event<"d7_out">)
    nest.await %d7_grid
    %96 = nest.subview %WO offsets = [0, 0, 0] sizes = [4, 1024, 256] strides = [1, 1, 1]
      : !nest.global_view<4x1024x256xbf16>
    %pre_wo = nest.dma.prefetch.async %96 into %wo_l2_1 : !nest.event<"pre_wo">
    nest.await %pre_wo
    %att_grid, %att_inrel, %att_out = nest.dispatch.tasks.async @prefill_attention_tile l1_mode = 0
      tasks(%63) globals() bindings(%q_l2_3, %k_l2_3, %v_l2_3, %o_l2_2)
      ins(%q_l2_3, %k_l2_3, %v_l2_3) outs(%o_l2_2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%d7_out)
      : (!nest.event<"att_grid">, !nest.event<"att_inrel">, !nest.event<"att_out">)
    nest.await %att_grid
    %pj_grid, %pj_inrel, %pj_out = nest.dispatch.tasks.async @prefill_outproj_tile l1_mode = 0
      tasks(%63) globals() bindings(%o_l2_2, %wo_l2_1, %out_l2_1) ins(%o_l2_2, %wo_l2_1)
      outs(%out_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out, %pre_wo)
      : (!nest.event<"pj_grid">, !nest.event<"pj_inrel">, !nest.event<"pj_out">)
    nest.await %pj_grid
    %97 = nest.subview %OUT offsets = [0, 0, 0] sizes = [4, 512, 256] strides = [1, 1, 1]
      : !nest.global_view<4x512x256xbf16>
    %out_store_done = nest.dma.store.async %out_l2_1 into %97 depends_on(%pj_out)
      : !nest.event<"out_store_done">
    nest.await %out_store_done
    nest.release %x_p0
      depends_on(%pre_x, %pre_x_2, %pre_x_4, %pre_x_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %wq_p0 depends_on(
      %pre_wq, %pre_wq_2, %pre_wq_4, %pre_wq_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %wk_p0 depends_on(
      %pre_wk, %pre_wk_2, %pre_wk_4, %pre_wk_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %wv_p0 depends_on(
      %pre_wv, %pre_wv_2, %pre_wv_4, %pre_wv_6, %d0_inrel, %d2_inrel, %d4_inrel, %d6_inrel)
    nest.release %x_p1
      depends_on(%pre_x_1, %pre_x_3, %pre_x_5, %pre_x_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %wq_p1 depends_on(
      %pre_wq_1, %pre_wq_3, %pre_wq_5, %pre_wq_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %wk_p1 depends_on(
      %pre_wk_1, %pre_wk_3, %pre_wk_5, %pre_wk_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %wv_p1 depends_on(
      %pre_wv_1, %pre_wv_3, %pre_wv_5, %pre_wv_7, %d1_inrel, %d3_inrel, %d5_inrel, %d7_inrel)
    nest.release %q_l2_3 depends_on(
      %d1_inrel, %d2_inrel, %d3_inrel, %d4_inrel, %d5_inrel, %d6_inrel, %d7_inrel, %d0_out,
      %d1_out, %d2_out, %d3_out, %d4_out, %d5_out, %d6_out, %d7_out, %att_inrel)
    nest.release %k_l2_3 depends_on(
      %d1_inrel, %d2_inrel, %d3_inrel, %d4_inrel, %d5_inrel, %d6_inrel, %d7_inrel, %d0_out,
      %d1_out, %d2_out, %d3_out, %d4_out, %d5_out, %d6_out, %d7_out, %att_inrel)
    nest.release %v_l2_3 depends_on(
      %d1_inrel, %d2_inrel, %d3_inrel, %d4_inrel, %d5_inrel, %d6_inrel, %d7_inrel, %d0_out,
      %d1_out, %d2_out, %d3_out, %d4_out, %d5_out, %d6_out, %d7_out, %att_inrel)
    nest.release %o_l2_2 depends_on(%att_out, %pj_inrel)
    nest.release %wo_l2_1 depends_on(%pre_wo, %pj_inrel)
    nest.release %out_l2_1 depends_on(%out_store_done)
    nest.await %att_grid, %pj_grid, %out_store_done
    nest.return
  }
  nexus.program @transformer_prefill_attention(
    %X_1: !nest.global_memref<8x512x128xbf16>, %WQ_1: !nest.global_memref<8x4x128x256xbf16>,
    %WK_1: !nest.global_memref<8x4x128x64xbf16>, %WV_1: !nest.global_memref<8x4x128x64xbf16>,
    %WO_1: !nest.global_memref<4x1024x256xbf16>, %OUT_1: !nest.global_memref<4x512x256xbf16>) {
    %prefill_done =
      nexus.submit_context.async @prefill_ctx(%X_1, %WQ_1, %WK_1, %WV_1, %WO_1, %OUT_1)
      : !nexus.event<"prefill_done">
    nexus.await %prefill_done
    nexus.return
  }
}
