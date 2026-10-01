// Transformer Prefill Attention, dispatch-parallel producers (one device root, BF16).
//
// Same logical shapes, packing and attention tail as
// transformer_prefill_attention_multicontext.mlir; the three producer programs
// drop load hoisting and the schedule compensates with more Grids:
//
// No load hoisting inside the producers; every tile.await is sunk to the
// last dependency-safe point:
// - the weight fill co-issues with the first X fill (plus the first
//   accumulated partials), behind one merged await before the first BOA;
// - each store issues as soon as its own BOA has been awaited and overlaps
//   the next independent fill (the K store co-issues with the V BOA);
// - no store ever co-issues with an outstanding BOA and no load is hoisted
//   across the BOA that reads its buffer, so input_released still fires only
//   after the final L2 load and staging recycling cannot start early.
//
// Parallelism comes from dispatch count instead:
// - the Q projection (writes q_l2) and the K/V projection (writes k_l2/v_l2)
//   are separate allocations, so each input-K chunk becomes two Grids that
//   never contend: 16 QKV dispatches form two independent 8-step chains;
// - each query block's output projection is issued twice, over the low and
//   high 64-row halves, each with its own output buffer: 8 independent
//   outproj Grids (32 partial OUT stores cover the head-major packing);
// - attention programs and dispatch structure are unchanged from the baseline.
//
// Traffic: HBM and L2 payload bytes match the baseline; only dispatch count
// grows (20 -> 28 Grids).  A GRID_WINDOW=8 retirement throttle keeps the live
// Grid routes inside the 16-entry Group table without any --sim-override.
//
// TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor operands and
// execute no numerics.  This source validates scheduling, resources and traffic, not
// numerical correctness or hardware performance guarantees.
// Generate: PYTHONPATH=. python
//     examples/generators/generate_transformer_prefill_dispatchparallel.py
//   --contexts-per-tile 4
// Run: bash examples/run.sh transformer-prefill-attention-dispatchparallel

builtin.module {
  tile.program @qkv_q_init(
    %task: !nest.task, %x_chunk: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk: !nest.l2_buffer<4x128x256xbf16>, %q_l2: !nest.l2_buffer<4x512x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 114688> {
    %x_buf = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %q_acc = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %wq_buf = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %0 = tile.subview %wq_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %wq_loaded = tile.load.async %0 into %wq_buf : !tile.event<"wq_loaded">
    %1 = tile.subview %x_chunk offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded = tile.load.async %1 into %x_buf : !tile.event<"x_loaded_0">
    tile.await %x_loaded, %wq_loaded
    %2 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_0">
    tile.await %q_boa
    %q_stored = tile.store.async %q_acc into %2 : !tile.event<"q_stored_0">
    %3 = tile.subview %x_chunk offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_1 = tile.load.async %3 into %x_buf : !tile.event<"x_loaded_1">
    tile.await %x_loaded_1, %q_stored
    %4 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_1 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_1">
    tile.await %q_boa_1
    %q_stored_1 = tile.store.async %q_acc into %4 : !tile.event<"q_stored_1">
    %5 = tile.subview %x_chunk offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_2 = tile.load.async %5 into %x_buf : !tile.event<"x_loaded_2">
    tile.await %x_loaded_2, %q_stored_1
    %6 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_2 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_2">
    tile.await %q_boa_2
    %q_stored_2 = tile.store.async %q_acc into %6 : !tile.event<"q_stored_2">
    %7 = tile.subview %x_chunk offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_3 = tile.load.async %7 into %x_buf : !tile.event<"x_loaded_3">
    tile.await %x_loaded_3, %q_stored_2
    %8 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_3 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_3">
    tile.await %q_boa_3
    %q_stored_3 = tile.store.async %q_acc into %8 : !tile.event<"q_stored_3">
    %9 = tile.subview %x_chunk offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_4 = tile.load.async %9 into %x_buf : !tile.event<"x_loaded_4">
    tile.await %x_loaded_4, %q_stored_3
    %10 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_4 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_4">
    tile.await %q_boa_4
    %q_stored_4 = tile.store.async %q_acc into %10 : !tile.event<"q_stored_4">
    %11 = tile.subview %x_chunk offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_5 = tile.load.async %11 into %x_buf : !tile.event<"x_loaded_5">
    tile.await %x_loaded_5, %q_stored_4
    %12 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_5 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_5">
    tile.await %q_boa_5
    %q_stored_5 = tile.store.async %q_acc into %12 : !tile.event<"q_stored_5">
    %13 = tile.subview %x_chunk offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_6 = tile.load.async %13 into %x_buf : !tile.event<"x_loaded_6">
    tile.await %x_loaded_6, %q_stored_5
    %14 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_6 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_6">
    tile.await %q_boa_6
    %q_stored_6 = tile.store.async %q_acc into %14 : !tile.event<"q_stored_6">
    %15 = tile.subview %x_chunk offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_7 = tile.load.async %15 into %x_buf : !tile.event<"x_loaded_7">
    tile.await %x_loaded_7, %q_stored_6
    tile.signal input_released(%task)
    %16 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_7 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_7">
    tile.await %q_boa_7
    %q_stored_7 = tile.store.async %q_acc into %16 : !tile.event<"q_stored_7">
    tile.await %q_stored_7
    tile.signal output_ready(%task)
    tile.free %x_buf
    tile.free %q_acc
    tile.free %wq_buf
    tile.return
  }
  tile.program @qkv_q_accum(
    %task_1: !nest.task, %x_chunk_1: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk_1: !nest.l2_buffer<4x128x256xbf16>, %q_l2_1: !nest.l2_buffer<4x512x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 114688> {
    %x_buf_1 = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %q_acc_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %wq_buf_1 = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %17 = tile.subview %wq_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %wq_loaded_1 = tile.load.async %17 into %wq_buf_1 : !tile.event<"wq_loaded">
    %18 = tile.subview %x_chunk_1 offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_8 = tile.load.async %18 into %x_buf_1 : !tile.event<"x_loaded_0">
    %19 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded = tile.load.async %19 into %q_acc_1 : !tile.event<"q_partial_loaded_0">
    tile.await %x_loaded_8, %q_partial_loaded, %wq_loaded_1
    %20 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_8 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_0">
    tile.await %q_boa_8
    %q_stored_8 = tile.store.async %q_acc_1 into %20 : !tile.event<"q_stored_0">
    %21 = tile.subview %x_chunk_1 offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_9 = tile.load.async %21 into %x_buf_1 : !tile.event<"x_loaded_1">
    tile.await %q_stored_8
    %22 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_1 = tile.load.async %22 into %q_acc_1 : !tile.event<"q_partial_loaded_1">
    tile.await %x_loaded_9, %q_partial_loaded_1
    %23 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_9 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_1">
    tile.await %q_boa_9
    %q_stored_9 = tile.store.async %q_acc_1 into %23 : !tile.event<"q_stored_1">
    %24 = tile.subview %x_chunk_1 offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_10 = tile.load.async %24 into %x_buf_1 : !tile.event<"x_loaded_2">
    tile.await %q_stored_9
    %25 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_2 = tile.load.async %25 into %q_acc_1 : !tile.event<"q_partial_loaded_2">
    tile.await %x_loaded_10, %q_partial_loaded_2
    %26 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_10 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_2">
    tile.await %q_boa_10
    %q_stored_10 = tile.store.async %q_acc_1 into %26 : !tile.event<"q_stored_2">
    %27 = tile.subview %x_chunk_1 offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_11 = tile.load.async %27 into %x_buf_1 : !tile.event<"x_loaded_3">
    tile.await %q_stored_10
    %28 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_3 = tile.load.async %28 into %q_acc_1 : !tile.event<"q_partial_loaded_3">
    tile.await %x_loaded_11, %q_partial_loaded_3
    %29 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_11 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_3">
    tile.await %q_boa_11
    %q_stored_11 = tile.store.async %q_acc_1 into %29 : !tile.event<"q_stored_3">
    %30 = tile.subview %x_chunk_1 offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_12 = tile.load.async %30 into %x_buf_1 : !tile.event<"x_loaded_4">
    tile.await %q_stored_11
    %31 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_4 = tile.load.async %31 into %q_acc_1 : !tile.event<"q_partial_loaded_4">
    tile.await %x_loaded_12, %q_partial_loaded_4
    %32 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_12 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_4">
    tile.await %q_boa_12
    %q_stored_12 = tile.store.async %q_acc_1 into %32 : !tile.event<"q_stored_4">
    %33 = tile.subview %x_chunk_1 offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_13 = tile.load.async %33 into %x_buf_1 : !tile.event<"x_loaded_5">
    tile.await %q_stored_12
    %34 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_5 = tile.load.async %34 into %q_acc_1 : !tile.event<"q_partial_loaded_5">
    tile.await %x_loaded_13, %q_partial_loaded_5
    %35 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_13 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_5">
    tile.await %q_boa_13
    %q_stored_13 = tile.store.async %q_acc_1 into %35 : !tile.event<"q_stored_5">
    %36 = tile.subview %x_chunk_1 offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_14 = tile.load.async %36 into %x_buf_1 : !tile.event<"x_loaded_6">
    tile.await %q_stored_13
    %37 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_6 = tile.load.async %37 into %q_acc_1 : !tile.event<"q_partial_loaded_6">
    tile.await %x_loaded_14, %q_partial_loaded_6
    %38 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_14 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_6">
    tile.await %q_boa_14
    %q_stored_14 = tile.store.async %q_acc_1 into %38 : !tile.event<"q_stored_6">
    %39 = tile.subview %x_chunk_1 offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_15 = tile.load.async %39 into %x_buf_1 : !tile.event<"x_loaded_7">
    tile.await %q_stored_14
    %40 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_partial_loaded_7 = tile.load.async %40 into %q_acc_1 : !tile.event<"q_partial_loaded_7">
    tile.await %x_loaded_15, %q_partial_loaded_7
    tile.signal input_released(%task_1)
    %41 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q_boa_15 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_7">
    tile.await %q_boa_15
    %q_stored_15 = tile.store.async %q_acc_1 into %41 : !tile.event<"q_stored_7">
    tile.await %q_stored_15
    tile.signal output_ready(%task_1)
    tile.free %x_buf_1
    tile.free %q_acc_1
    tile.free %wq_buf_1
    tile.return
  }
  tile.program @qkv_kv_init(
    %task_2: !nest.task, %x_chunk_2: !nest.l2_buffer<512x128xbf16>,
    %wk_chunk: !nest.l2_buffer<4x128x64xbf16>, %wv_chunk: !nest.l2_buffer<4x128x64xbf16>,
    %k_l2: !nest.l2_buffer<4x512x64xbf16>, %v_l2: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 65536> {
    %x_buf_2 = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %k_acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wk_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wv_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %42 = tile.subview %wk_chunk task = %task_2 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %43 = tile.subview %wv_chunk task = %task_2 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %wk_loaded = tile.load.async %42 into %wk_buf : !tile.event<"wk_loaded">
    %wv_loaded = tile.load.async %43 into %wv_buf : !tile.event<"wv_loaded">
    %44 = tile.subview %x_chunk_2 offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_16 = tile.load.async %44 into %x_buf_2 : !tile.event<"x_loaded_0">
    tile.await %x_loaded_16, %wk_loaded, %wv_loaded
    %45 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_0">
    tile.await %k_boa
    %k_stored = tile.store.async %k_acc into %45 : !tile.event<"k_stored_0">
    %46 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_0">
    tile.await %v_boa
    %v_stored = tile.store.async %v_acc into %46 : !tile.event<"v_stored_0">
    %47 = tile.subview %x_chunk_2 offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_17 = tile.load.async %47 into %x_buf_2 : !tile.event<"x_loaded_1">
    tile.await %x_loaded_17, %k_stored, %v_stored
    %48 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_1">
    tile.await %k_boa_1
    %k_stored_1 = tile.store.async %k_acc into %48 : !tile.event<"k_stored_1">
    %49 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_1">
    tile.await %v_boa_1
    %v_stored_1 = tile.store.async %v_acc into %49 : !tile.event<"v_stored_1">
    %50 = tile.subview %x_chunk_2 offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_18 = tile.load.async %50 into %x_buf_2 : !tile.event<"x_loaded_2">
    tile.await %x_loaded_18, %k_stored_1, %v_stored_1
    %51 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_2">
    tile.await %k_boa_2
    %k_stored_2 = tile.store.async %k_acc into %51 : !tile.event<"k_stored_2">
    %52 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_2">
    tile.await %v_boa_2
    %v_stored_2 = tile.store.async %v_acc into %52 : !tile.event<"v_stored_2">
    %53 = tile.subview %x_chunk_2 offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_19 = tile.load.async %53 into %x_buf_2 : !tile.event<"x_loaded_3">
    tile.await %x_loaded_19, %k_stored_2, %v_stored_2
    %54 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_3">
    tile.await %k_boa_3
    %k_stored_3 = tile.store.async %k_acc into %54 : !tile.event<"k_stored_3">
    %55 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_3">
    tile.await %v_boa_3
    %v_stored_3 = tile.store.async %v_acc into %55 : !tile.event<"v_stored_3">
    %56 = tile.subview %x_chunk_2 offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_20 = tile.load.async %56 into %x_buf_2 : !tile.event<"x_loaded_4">
    tile.await %x_loaded_20, %k_stored_3, %v_stored_3
    %57 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_4 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_4">
    tile.await %k_boa_4
    %k_stored_4 = tile.store.async %k_acc into %57 : !tile.event<"k_stored_4">
    %58 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_4 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_4">
    tile.await %v_boa_4
    %v_stored_4 = tile.store.async %v_acc into %58 : !tile.event<"v_stored_4">
    %59 = tile.subview %x_chunk_2 offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_21 = tile.load.async %59 into %x_buf_2 : !tile.event<"x_loaded_5">
    tile.await %x_loaded_21, %k_stored_4, %v_stored_4
    %60 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_5 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_5">
    tile.await %k_boa_5
    %k_stored_5 = tile.store.async %k_acc into %60 : !tile.event<"k_stored_5">
    %61 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_5 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_5">
    tile.await %v_boa_5
    %v_stored_5 = tile.store.async %v_acc into %61 : !tile.event<"v_stored_5">
    %62 = tile.subview %x_chunk_2 offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_22 = tile.load.async %62 into %x_buf_2 : !tile.event<"x_loaded_6">
    tile.await %x_loaded_22, %k_stored_5, %v_stored_5
    %63 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_6 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_6">
    tile.await %k_boa_6
    %k_stored_6 = tile.store.async %k_acc into %63 : !tile.event<"k_stored_6">
    %64 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_6 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_6">
    tile.await %v_boa_6
    %v_stored_6 = tile.store.async %v_acc into %64 : !tile.event<"v_stored_6">
    %65 = tile.subview %x_chunk_2 offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_23 = tile.load.async %65 into %x_buf_2 : !tile.event<"x_loaded_7">
    tile.await %x_loaded_23, %k_stored_6, %v_stored_6
    tile.signal input_released(%task_2)
    %66 = tile.subview %k_l2 task = %task_2 task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_7 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_7">
    tile.await %k_boa_7
    %k_stored_7 = tile.store.async %k_acc into %66 : !tile.event<"k_stored_7">
    %67 = tile.subview %v_l2 task = %task_2 task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_7 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_7">
    tile.await %v_boa_7
    %v_stored_7 = tile.store.async %v_acc into %67 : !tile.event<"v_stored_7">
    tile.await %k_stored_7, %v_stored_7
    tile.signal output_ready(%task_2)
    tile.free %x_buf_2
    tile.free %k_acc
    tile.free %wk_buf
    tile.free %v_acc
    tile.free %wv_buf
    tile.return
  }
  tile.program @qkv_kv_accum(
    %task_3: !nest.task, %x_chunk_3: !nest.l2_buffer<512x128xbf16>,
    %wk_chunk_1: !nest.l2_buffer<4x128x64xbf16>, %wv_chunk_1: !nest.l2_buffer<4x128x64xbf16>,
    %k_l2_1: !nest.l2_buffer<4x512x64xbf16>, %v_l2_1: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 65536> {
    %x_buf_3 = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %k_acc_1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wk_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc_1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wv_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %68 = tile.subview %wk_chunk_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %69 = tile.subview %wv_chunk_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %wk_loaded_1 = tile.load.async %68 into %wk_buf_1 : !tile.event<"wk_loaded">
    %wv_loaded_1 = tile.load.async %69 into %wv_buf_1 : !tile.event<"wv_loaded">
    %70 = tile.subview %x_chunk_3 offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_24 = tile.load.async %70 into %x_buf_3 : !tile.event<"x_loaded_0">
    %71 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded = tile.load.async %71 into %k_acc_1 : !tile.event<"k_partial_loaded_0">
    %72 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded = tile.load.async %72 into %v_acc_1 : !tile.event<"v_partial_loaded_0">
    tile.await %x_loaded_24, %k_partial_loaded, %v_partial_loaded, %wk_loaded_1, %wv_loaded_1
    %73 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_8 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_0">
    tile.await %k_boa_8
    %k_stored_8 = tile.store.async %k_acc_1 into %73 : !tile.event<"k_stored_0">
    %74 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_8 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_0">
    tile.await %v_boa_8
    %v_stored_8 = tile.store.async %v_acc_1 into %74 : !tile.event<"v_stored_0">
    %75 = tile.subview %x_chunk_3 offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_25 = tile.load.async %75 into %x_buf_3 : !tile.event<"x_loaded_1">
    tile.await %k_stored_8, %v_stored_8
    %76 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_1 = tile.load.async %76 into %k_acc_1 : !tile.event<"k_partial_loaded_1">
    %77 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_1 = tile.load.async %77 into %v_acc_1 : !tile.event<"v_partial_loaded_1">
    tile.await %x_loaded_25, %k_partial_loaded_1, %v_partial_loaded_1
    %78 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_9 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_1">
    tile.await %k_boa_9
    %k_stored_9 = tile.store.async %k_acc_1 into %78 : !tile.event<"k_stored_1">
    %79 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_9 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_1">
    tile.await %v_boa_9
    %v_stored_9 = tile.store.async %v_acc_1 into %79 : !tile.event<"v_stored_1">
    %80 = tile.subview %x_chunk_3 offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_26 = tile.load.async %80 into %x_buf_3 : !tile.event<"x_loaded_2">
    tile.await %k_stored_9, %v_stored_9
    %81 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_2 = tile.load.async %81 into %k_acc_1 : !tile.event<"k_partial_loaded_2">
    %82 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_2 = tile.load.async %82 into %v_acc_1 : !tile.event<"v_partial_loaded_2">
    tile.await %x_loaded_26, %k_partial_loaded_2, %v_partial_loaded_2
    %83 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_10 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_2">
    tile.await %k_boa_10
    %k_stored_10 = tile.store.async %k_acc_1 into %83 : !tile.event<"k_stored_2">
    %84 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_10 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_2">
    tile.await %v_boa_10
    %v_stored_10 = tile.store.async %v_acc_1 into %84 : !tile.event<"v_stored_2">
    %85 = tile.subview %x_chunk_3 offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_27 = tile.load.async %85 into %x_buf_3 : !tile.event<"x_loaded_3">
    tile.await %k_stored_10, %v_stored_10
    %86 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_3 = tile.load.async %86 into %k_acc_1 : !tile.event<"k_partial_loaded_3">
    %87 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_3 = tile.load.async %87 into %v_acc_1 : !tile.event<"v_partial_loaded_3">
    tile.await %x_loaded_27, %k_partial_loaded_3, %v_partial_loaded_3
    %88 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_11 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_3">
    tile.await %k_boa_11
    %k_stored_11 = tile.store.async %k_acc_1 into %88 : !tile.event<"k_stored_3">
    %89 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_11 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_3">
    tile.await %v_boa_11
    %v_stored_11 = tile.store.async %v_acc_1 into %89 : !tile.event<"v_stored_3">
    %90 = tile.subview %x_chunk_3 offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_28 = tile.load.async %90 into %x_buf_3 : !tile.event<"x_loaded_4">
    tile.await %k_stored_11, %v_stored_11
    %91 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_4 = tile.load.async %91 into %k_acc_1 : !tile.event<"k_partial_loaded_4">
    %92 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_4 = tile.load.async %92 into %v_acc_1 : !tile.event<"v_partial_loaded_4">
    tile.await %x_loaded_28, %k_partial_loaded_4, %v_partial_loaded_4
    %93 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_12 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_4">
    tile.await %k_boa_12
    %k_stored_12 = tile.store.async %k_acc_1 into %93 : !tile.event<"k_stored_4">
    %94 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_12 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_4">
    tile.await %v_boa_12
    %v_stored_12 = tile.store.async %v_acc_1 into %94 : !tile.event<"v_stored_4">
    %95 = tile.subview %x_chunk_3 offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_29 = tile.load.async %95 into %x_buf_3 : !tile.event<"x_loaded_5">
    tile.await %k_stored_12, %v_stored_12
    %96 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_5 = tile.load.async %96 into %k_acc_1 : !tile.event<"k_partial_loaded_5">
    %97 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_5 = tile.load.async %97 into %v_acc_1 : !tile.event<"v_partial_loaded_5">
    tile.await %x_loaded_29, %k_partial_loaded_5, %v_partial_loaded_5
    %98 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_13 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_5">
    tile.await %k_boa_13
    %k_stored_13 = tile.store.async %k_acc_1 into %98 : !tile.event<"k_stored_5">
    %99 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_13 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_5">
    tile.await %v_boa_13
    %v_stored_13 = tile.store.async %v_acc_1 into %99 : !tile.event<"v_stored_5">
    %100 = tile.subview %x_chunk_3 offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_30 = tile.load.async %100 into %x_buf_3 : !tile.event<"x_loaded_6">
    tile.await %k_stored_13, %v_stored_13
    %101 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_6 = tile.load.async %101 into %k_acc_1 : !tile.event<"k_partial_loaded_6">
    %102 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_6 = tile.load.async %102 into %v_acc_1 : !tile.event<"v_partial_loaded_6">
    tile.await %x_loaded_30, %k_partial_loaded_6, %v_partial_loaded_6
    %103 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_14 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_6">
    tile.await %k_boa_14
    %k_stored_14 = tile.store.async %k_acc_1 into %103 : !tile.event<"k_stored_6">
    %104 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_14 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_6">
    tile.await %v_boa_14
    %v_stored_14 = tile.store.async %v_acc_1 into %104 : !tile.event<"v_stored_6">
    %105 = tile.subview %x_chunk_3 offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %x_loaded_31 = tile.load.async %105 into %x_buf_3 : !tile.event<"x_loaded_7">
    tile.await %k_stored_14, %v_stored_14
    %106 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_partial_loaded_7 = tile.load.async %106 into %k_acc_1 : !tile.event<"k_partial_loaded_7">
    %107 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_partial_loaded_7 = tile.load.async %107 into %v_acc_1 : !tile.event<"v_partial_loaded_7">
    tile.await %x_loaded_31, %k_partial_loaded_7, %v_partial_loaded_7
    tile.signal input_released(%task_3)
    %108 = tile.subview %k_l2_1 task = %task_3 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %k_boa_15 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_7">
    tile.await %k_boa_15
    %k_stored_15 = tile.store.async %k_acc_1 into %108 : !tile.event<"k_stored_7">
    %109 = tile.subview %v_l2_1 task = %task_3 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 64] strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %v_boa_15 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_7">
    tile.await %v_boa_15
    %v_stored_15 = tile.store.async %v_acc_1 into %109 : !tile.event<"v_stored_7">
    tile.await %k_stored_15, %v_stored_15
    tile.signal output_ready(%task_3)
    tile.free %x_buf_3
    tile.free %k_acc_1
    tile.free %wk_buf_1
    tile.free %v_acc_1
    tile.free %wv_buf_1
    tile.return
  }
  tile.program @prefill_attention_q0(
    %task_4: !nest.task, %q_l2_2: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_2: !nest.l2_buffer<4x512x64xbf16>, %v_l2_2: !nest.l2_buffer<4x512x64xbf16>,
    %o_l2: !nest.l2_buffer<4x128x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0],
      tile_l1_spm_bytes_per_context = 245760> {
    %q0 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %q1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %acc0 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %m0 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %m1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l0 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %score = tile.alloc shape = [64, 128] dtype = "f32" alignment = 256
      : !tile.l1_buffer<64x128xf32>
    %k0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %k1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v0 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %v1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256 : !tile.l1_buffer<128x64xbf16>
    %110 = tile.subview %q_l2_2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded = tile.load.async %110 into %q0 : !tile.event<"q0_loaded">
    %111 = tile.subview %q_l2_2 task = %task_4 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded = tile.load.async %111 into %q1 : !tile.event<"q1_loaded">
    %112 = tile.subview %k_l2_2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %113 = tile.subview %v_l2_2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded = tile.load.async %112 into %k0 : !tile.event<"k0_loaded">
    %v0_loaded = tile.load.async %113 into %v0 : !tile.event<"v0_loaded">
    %114 = tile.subview %k_l2_2 task = %task_4 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %115 = tile.subview %v_l2_2 task = %task_4 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded = tile.load.async %114 into %k1 : !tile.event<"k1_loaded">
    %v1_loaded = tile.load.async %115 into %v1 : !tile.event<"v1_loaded">
    tile.await %q0_loaded, %q1_loaded
    tile.await %k0_loaded, %v0_loaded
    %qk0_p0_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h0">
    tile.await %qk0_p0_h0
    %sm0_p0_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h0">
    tile.await %sm0_p0_h0
    %pv0_p0_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h0">
    tile.await %pv0_p0_h0
    %qk0_p0_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h1">
    tile.await %qk0_p0_h1
    %sm0_p0_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h1">
    tile.await %sm0_p0_h1
    %pv0_p0_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h1">
    tile.await %pv0_p0_h1
    %qk0_p0_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h2">
    tile.await %qk0_p0_h2
    %sm0_p0_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h2">
    tile.await %sm0_p0_h2
    %pv0_p0_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h2">
    tile.await %pv0_p0_h2
    %qk0_p0_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h3">
    tile.await %qk0_p0_h3
    %sm0_p0_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h3">
    tile.await %sm0_p0_h3
    %pv0_p0_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h3">
    tile.await %pv0_p0_h3
    %qk0_p1_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h0">
    tile.await %qk0_p1_h0
    %sm0_p1_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h0">
    tile.await %sm0_p1_h0
    %pv0_p1_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h0">
    tile.await %pv0_p1_h0
    %qk0_p1_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h1">
    tile.await %qk0_p1_h1
    %sm0_p1_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h1">
    tile.await %sm0_p1_h1
    %pv0_p1_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h1">
    tile.await %pv0_p1_h1
    %qk0_p1_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h2">
    tile.await %qk0_p1_h2
    %sm0_p1_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h2">
    tile.await %sm0_p1_h2
    %pv0_p1_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h2">
    tile.await %pv0_p1_h2
    %qk0_p1_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h3">
    tile.await %qk0_p1_h3
    %sm0_p1_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h3">
    tile.await %sm0_p1_h3
    %pv0_p1_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h3">
    tile.await %pv0_p1_h3
    %116 = tile.subview %k_l2_2 task = %task_4 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %117 = tile.subview %v_l2_2 task = %task_4 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded = tile.load.async %116 into %k0 : !tile.event<"k2_loaded">
    %v2_loaded = tile.load.async %117 into %v0 : !tile.event<"v2_loaded">
    tile.await %k1_loaded, %v1_loaded
    %qk1_p0_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h0">
    tile.await %qk1_p0_h0
    %sm1_p0_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h0">
    tile.await %sm1_p0_h0
    %pv1_p0_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h0">
    tile.await %pv1_p0_h0
    %qk1_p0_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h1">
    tile.await %qk1_p0_h1
    %sm1_p0_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h1">
    tile.await %sm1_p0_h1
    %pv1_p0_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h1">
    tile.await %pv1_p0_h1
    %qk1_p0_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h2">
    tile.await %qk1_p0_h2
    %sm1_p0_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h2">
    tile.await %sm1_p0_h2
    %pv1_p0_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h2">
    tile.await %pv1_p0_h2
    %qk1_p0_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h3">
    tile.await %qk1_p0_h3
    %sm1_p0_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h3">
    tile.await %sm1_p0_h3
    %pv1_p0_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h3">
    tile.await %pv1_p0_h3
    %qk1_p1_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h0">
    tile.await %qk1_p1_h0
    %sm1_p1_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h0">
    tile.await %sm1_p1_h0
    %pv1_p1_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h0">
    tile.await %pv1_p1_h0
    %qk1_p1_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h1">
    tile.await %qk1_p1_h1
    %sm1_p1_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h1">
    tile.await %sm1_p1_h1
    %pv1_p1_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h1">
    tile.await %pv1_p1_h1
    %qk1_p1_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h2">
    tile.await %qk1_p1_h2
    %sm1_p1_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h2">
    tile.await %sm1_p1_h2
    %pv1_p1_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h2">
    tile.await %pv1_p1_h2
    %qk1_p1_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h3">
    tile.await %qk1_p1_h3
    %sm1_p1_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h3">
    tile.await %sm1_p1_h3
    %pv1_p1_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h3">
    tile.await %pv1_p1_h3
    %118 = tile.subview %k_l2_2 task = %task_4 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %119 = tile.subview %v_l2_2 task = %task_4 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded = tile.load.async %118 into %k1 : !tile.event<"k3_loaded">
    %v3_loaded = tile.load.async %119 into %v1 : !tile.event<"v3_loaded">
    tile.await %k2_loaded, %v2_loaded
    %qk2_p0_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h0">
    tile.await %qk2_p0_h0
    %sm2_p0_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h0">
    tile.await %sm2_p0_h0
    %pv2_p0_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h0">
    tile.await %pv2_p0_h0
    %qk2_p0_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h1">
    tile.await %qk2_p0_h1
    %sm2_p0_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h1">
    tile.await %sm2_p0_h1
    %pv2_p0_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h1">
    tile.await %pv2_p0_h1
    %qk2_p0_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h2">
    tile.await %qk2_p0_h2
    %sm2_p0_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h2">
    tile.await %sm2_p0_h2
    %pv2_p0_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h2">
    tile.await %pv2_p0_h2
    %qk2_p0_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h3">
    tile.await %qk2_p0_h3
    %sm2_p0_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h3">
    tile.await %sm2_p0_h3
    %pv2_p0_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h3">
    tile.await %pv2_p0_h3
    %qk2_p1_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h0">
    tile.await %qk2_p1_h0
    %sm2_p1_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h0">
    tile.await %sm2_p1_h0
    %pv2_p1_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h0">
    tile.await %pv2_p1_h0
    %qk2_p1_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h1">
    tile.await %qk2_p1_h1
    %sm2_p1_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h1">
    tile.await %sm2_p1_h1
    %pv2_p1_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h1">
    tile.await %pv2_p1_h1
    %qk2_p1_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h2">
    tile.await %qk2_p1_h2
    %sm2_p1_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h2">
    tile.await %sm2_p1_h2
    %pv2_p1_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h2">
    tile.await %pv2_p1_h2
    %qk2_p1_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h3">
    tile.await %qk2_p1_h3
    %sm2_p1_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h3">
    tile.await %sm2_p1_h3
    %pv2_p1_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h3">
    tile.await %pv2_p1_h3
    tile.await %k3_loaded, %v3_loaded
    tile.signal input_released(%task_4)
    %qk3_p0_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h0">
    tile.await %qk3_p0_h0
    %sm3_p0_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h0">
    tile.await %sm3_p0_h0
    %pv3_p0_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h0">
    tile.await %pv3_p0_h0
    %qk3_p0_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h1">
    tile.await %qk3_p0_h1
    %sm3_p0_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h1">
    tile.await %sm3_p0_h1
    %pv3_p0_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h1">
    tile.await %pv3_p0_h1
    %qk3_p0_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h2">
    tile.await %qk3_p0_h2
    %sm3_p0_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h2">
    tile.await %sm3_p0_h2
    %pv3_p0_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h2">
    tile.await %pv3_p0_h2
    %qk3_p0_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h3">
    tile.await %qk3_p0_h3
    %sm3_p0_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h3">
    tile.await %sm3_p0_h3
    %pv3_p0_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h3">
    tile.await %pv3_p0_h3
    %qk3_p1_h0 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h0">
    tile.await %qk3_p1_h0
    %sm3_p1_h0 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h0">
    tile.await %sm3_p1_h0
    %pv3_p1_h0 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h0">
    tile.await %pv3_p1_h0
    %qk3_p1_h1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h1">
    tile.await %qk3_p1_h1
    %sm3_p1_h1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h1">
    tile.await %sm3_p1_h1
    %pv3_p1_h1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h1">
    tile.await %pv3_p1_h1
    %qk3_p1_h2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h2">
    tile.await %qk3_p1_h2
    %sm3_p1_h2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h2">
    tile.await %sm3_p1_h2
    %pv3_p1_h2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h2">
    tile.await %pv3_p1_h2
    %qk3_p1_h3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h3">
    tile.await %qk3_p1_h3
    %sm3_p1_h3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h3">
    tile.await %sm3_p1_h3
    %pv3_p1_h3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h3">
    tile.await %pv3_p1_h3
    %120 = tile.subview %o_l2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored = tile.store.async %acc0 into %120 : !tile.event<"o0_stored">
    %121 = tile.subview %o_l2 task = %task_4 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored = tile.store.async %acc1 into %121 : !tile.event<"o1_stored">
    tile.await %o0_stored, %o1_stored
    tile.signal output_ready(%task_4)
    tile.free %q0
    tile.free %q1
    tile.free %acc0
    tile.free %acc1
    tile.free %m0
    tile.free %m1
    tile.free %l0
    tile.free %l1
    tile.free %score
    tile.free %k0
    tile.free %k1
    tile.free %v0
    tile.free %v1
    tile.return
  }
  tile.program @prefill_attention_q1(
    %task_5: !nest.task, %q_l2_3: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_3: !nest.l2_buffer<4x512x64xbf16>, %v_l2_3: !nest.l2_buffer<4x512x64xbf16>,
    %o_l2_1: !nest.l2_buffer<4x128x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0],
      tile_l1_spm_bytes_per_context = 245760> {
    %q0_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %q1_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc0_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc1_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %m0_1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %m1_1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l0_1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l1_1 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %score_1 = tile.alloc shape = [64, 128] dtype = "f32" alignment = 256
      : !tile.l1_buffer<64x128xf32>
    %k0_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %k1_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v0_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v1_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %122 = tile.subview %q_l2_3 task = %task_5 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_1 = tile.load.async %122 into %q0_1 : !tile.event<"q0_loaded">
    %123 = tile.subview %q_l2_3 task = %task_5 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_1 = tile.load.async %123 into %q1_1 : !tile.event<"q1_loaded">
    %124 = tile.subview %k_l2_3 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %125 = tile.subview %v_l2_3 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_1 = tile.load.async %124 into %k0_1 : !tile.event<"k0_loaded">
    %v0_loaded_1 = tile.load.async %125 into %v0_1 : !tile.event<"v0_loaded">
    %126 = tile.subview %k_l2_3 task = %task_5 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %127 = tile.subview %v_l2_3 task = %task_5 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_1 = tile.load.async %126 into %k1_1 : !tile.event<"k1_loaded">
    %v1_loaded_1 = tile.load.async %127 into %v1_1 : !tile.event<"v1_loaded">
    tile.await %q0_loaded_1, %q1_loaded_1
    tile.await %k0_loaded_1, %v0_loaded_1
    %qk0_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h0">
    tile.await %qk0_p0_h0_1
    %sm0_p0_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h0">
    tile.await %sm0_p0_h0_1
    %pv0_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h0">
    tile.await %pv0_p0_h0_1
    %qk0_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h1">
    tile.await %qk0_p0_h1_1
    %sm0_p0_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h1">
    tile.await %sm0_p0_h1_1
    %pv0_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h1">
    tile.await %pv0_p0_h1_1
    %qk0_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h2">
    tile.await %qk0_p0_h2_1
    %sm0_p0_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h2">
    tile.await %sm0_p0_h2_1
    %pv0_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h2">
    tile.await %pv0_p0_h2_1
    %qk0_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h3">
    tile.await %qk0_p0_h3_1
    %sm0_p0_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h3">
    tile.await %sm0_p0_h3_1
    %pv0_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h3">
    tile.await %pv0_p0_h3_1
    %qk0_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h0">
    tile.await %qk0_p1_h0_1
    %sm0_p1_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h0">
    tile.await %sm0_p1_h0_1
    %pv0_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h0">
    tile.await %pv0_p1_h0_1
    %qk0_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h1">
    tile.await %qk0_p1_h1_1
    %sm0_p1_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h1">
    tile.await %sm0_p1_h1_1
    %pv0_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h1">
    tile.await %pv0_p1_h1_1
    %qk0_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h2">
    tile.await %qk0_p1_h2_1
    %sm0_p1_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h2">
    tile.await %sm0_p1_h2_1
    %pv0_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h2">
    tile.await %pv0_p1_h2_1
    %qk0_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h3">
    tile.await %qk0_p1_h3_1
    %sm0_p1_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h3">
    tile.await %sm0_p1_h3_1
    %pv0_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h3">
    tile.await %pv0_p1_h3_1
    %128 = tile.subview %k_l2_3 task = %task_5 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %129 = tile.subview %v_l2_3 task = %task_5 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_1 = tile.load.async %128 into %k0_1 : !tile.event<"k2_loaded">
    %v2_loaded_1 = tile.load.async %129 into %v0_1 : !tile.event<"v2_loaded">
    tile.await %k1_loaded_1, %v1_loaded_1
    %qk1_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h0">
    tile.await %qk1_p0_h0_1
    %sm1_p0_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h0">
    tile.await %sm1_p0_h0_1
    %pv1_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h0">
    tile.await %pv1_p0_h0_1
    %qk1_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h1">
    tile.await %qk1_p0_h1_1
    %sm1_p0_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h1">
    tile.await %sm1_p0_h1_1
    %pv1_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h1">
    tile.await %pv1_p0_h1_1
    %qk1_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h2">
    tile.await %qk1_p0_h2_1
    %sm1_p0_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h2">
    tile.await %sm1_p0_h2_1
    %pv1_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h2">
    tile.await %pv1_p0_h2_1
    %qk1_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h3">
    tile.await %qk1_p0_h3_1
    %sm1_p0_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h3">
    tile.await %sm1_p0_h3_1
    %pv1_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h3">
    tile.await %pv1_p0_h3_1
    %qk1_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h0">
    tile.await %qk1_p1_h0_1
    %sm1_p1_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h0">
    tile.await %sm1_p1_h0_1
    %pv1_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h0">
    tile.await %pv1_p1_h0_1
    %qk1_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h1">
    tile.await %qk1_p1_h1_1
    %sm1_p1_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h1">
    tile.await %sm1_p1_h1_1
    %pv1_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h1">
    tile.await %pv1_p1_h1_1
    %qk1_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h2">
    tile.await %qk1_p1_h2_1
    %sm1_p1_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h2">
    tile.await %sm1_p1_h2_1
    %pv1_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h2">
    tile.await %pv1_p1_h2_1
    %qk1_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h3">
    tile.await %qk1_p1_h3_1
    %sm1_p1_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h3">
    tile.await %sm1_p1_h3_1
    %pv1_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h3">
    tile.await %pv1_p1_h3_1
    %130 = tile.subview %k_l2_3 task = %task_5 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %131 = tile.subview %v_l2_3 task = %task_5 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_1 = tile.load.async %130 into %k1_1 : !tile.event<"k3_loaded">
    %v3_loaded_1 = tile.load.async %131 into %v1_1 : !tile.event<"v3_loaded">
    tile.await %k2_loaded_1, %v2_loaded_1
    %qk2_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h0">
    tile.await %qk2_p0_h0_1
    %sm2_p0_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h0">
    tile.await %sm2_p0_h0_1
    %pv2_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h0">
    tile.await %pv2_p0_h0_1
    %qk2_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h1">
    tile.await %qk2_p0_h1_1
    %sm2_p0_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h1">
    tile.await %sm2_p0_h1_1
    %pv2_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h1">
    tile.await %pv2_p0_h1_1
    %qk2_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h2">
    tile.await %qk2_p0_h2_1
    %sm2_p0_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h2">
    tile.await %sm2_p0_h2_1
    %pv2_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h2">
    tile.await %pv2_p0_h2_1
    %qk2_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h3">
    tile.await %qk2_p0_h3_1
    %sm2_p0_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h3">
    tile.await %sm2_p0_h3_1
    %pv2_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h3">
    tile.await %pv2_p0_h3_1
    %qk2_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h0">
    tile.await %qk2_p1_h0_1
    %sm2_p1_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h0">
    tile.await %sm2_p1_h0_1
    %pv2_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h0">
    tile.await %pv2_p1_h0_1
    %qk2_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h1">
    tile.await %qk2_p1_h1_1
    %sm2_p1_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h1">
    tile.await %sm2_p1_h1_1
    %pv2_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h1">
    tile.await %pv2_p1_h1_1
    %qk2_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h2">
    tile.await %qk2_p1_h2_1
    %sm2_p1_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h2">
    tile.await %sm2_p1_h2_1
    %pv2_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h2">
    tile.await %pv2_p1_h2_1
    %qk2_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h3">
    tile.await %qk2_p1_h3_1
    %sm2_p1_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h3">
    tile.await %sm2_p1_h3_1
    %pv2_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h3">
    tile.await %pv2_p1_h3_1
    tile.await %k3_loaded_1, %v3_loaded_1
    tile.signal input_released(%task_5)
    %qk3_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h0">
    tile.await %qk3_p0_h0_1
    %sm3_p0_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h0">
    tile.await %sm3_p0_h0_1
    %pv3_p0_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h0">
    tile.await %pv3_p0_h0_1
    %qk3_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h1">
    tile.await %qk3_p0_h1_1
    %sm3_p0_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h1">
    tile.await %sm3_p0_h1_1
    %pv3_p0_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h1">
    tile.await %pv3_p0_h1_1
    %qk3_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h2">
    tile.await %qk3_p0_h2_1
    %sm3_p0_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h2">
    tile.await %sm3_p0_h2_1
    %pv3_p0_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h2">
    tile.await %pv3_p0_h2_1
    %qk3_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h3">
    tile.await %qk3_p0_h3_1
    %sm3_p0_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h3">
    tile.await %sm3_p0_h3_1
    %pv3_p0_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h3">
    tile.await %pv3_p0_h3_1
    %qk3_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h0">
    tile.await %qk3_p1_h0_1
    %sm3_p1_h0_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h0">
    tile.await %sm3_p1_h0_1
    %pv3_p1_h0_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h0">
    tile.await %pv3_p1_h0_1
    %qk3_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h1">
    tile.await %qk3_p1_h1_1
    %sm3_p1_h1_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h1">
    tile.await %sm3_p1_h1_1
    %pv3_p1_h1_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h1">
    tile.await %pv3_p1_h1_1
    %qk3_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h2">
    tile.await %qk3_p1_h2_1
    %sm3_p1_h2_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h2">
    tile.await %sm3_p1_h2_1
    %pv3_p1_h2_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h2">
    tile.await %pv3_p1_h2_1
    %qk3_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h3">
    tile.await %qk3_p1_h3_1
    %sm3_p1_h3_1 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h3">
    tile.await %sm3_p1_h3_1
    %pv3_p1_h3_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h3">
    tile.await %pv3_p1_h3_1
    %132 = tile.subview %o_l2_1 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_1 = tile.store.async %acc0_1 into %132 : !tile.event<"o0_stored">
    %133 = tile.subview %o_l2_1 task = %task_5 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_1 = tile.store.async %acc1_1 into %133 : !tile.event<"o1_stored">
    tile.await %o0_stored_1, %o1_stored_1
    tile.signal output_ready(%task_5)
    tile.free %q0_1
    tile.free %q1_1
    tile.free %acc0_1
    tile.free %acc1_1
    tile.free %m0_1
    tile.free %m1_1
    tile.free %l0_1
    tile.free %l1_1
    tile.free %score_1
    tile.free %k0_1
    tile.free %k1_1
    tile.free %v0_1
    tile.free %v1_1
    tile.return
  }
  tile.program @prefill_attention_q2(
    %task_6: !nest.task, %q_l2_4: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_4: !nest.l2_buffer<4x512x64xbf16>, %v_l2_4: !nest.l2_buffer<4x512x64xbf16>,
    %o_l2_2: !nest.l2_buffer<4x128x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0],
      tile_l1_spm_bytes_per_context = 245760> {
    %q0_2 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %q1_2 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc0_2 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc1_2 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %m0_2 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %m1_2 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l0_2 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l1_2 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %score_2 = tile.alloc shape = [64, 128] dtype = "f32" alignment = 256
      : !tile.l1_buffer<64x128xf32>
    %k0_2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %k1_2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v0_2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v1_2 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %134 = tile.subview %q_l2_4 task = %task_6 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_2 = tile.load.async %134 into %q0_2 : !tile.event<"q0_loaded">
    %135 = tile.subview %q_l2_4 task = %task_6 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_2 = tile.load.async %135 into %q1_2 : !tile.event<"q1_loaded">
    %136 = tile.subview %k_l2_4 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %137 = tile.subview %v_l2_4 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_2 = tile.load.async %136 into %k0_2 : !tile.event<"k0_loaded">
    %v0_loaded_2 = tile.load.async %137 into %v0_2 : !tile.event<"v0_loaded">
    %138 = tile.subview %k_l2_4 task = %task_6 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %139 = tile.subview %v_l2_4 task = %task_6 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_2 = tile.load.async %138 into %k1_2 : !tile.event<"k1_loaded">
    %v1_loaded_2 = tile.load.async %139 into %v1_2 : !tile.event<"v1_loaded">
    tile.await %q0_loaded_2, %q1_loaded_2
    tile.await %k0_loaded_2, %v0_loaded_2
    %qk0_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h0">
    tile.await %qk0_p0_h0_2
    %sm0_p0_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h0">
    tile.await %sm0_p0_h0_2
    %pv0_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h0">
    tile.await %pv0_p0_h0_2
    %qk0_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h1">
    tile.await %qk0_p0_h1_2
    %sm0_p0_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h1">
    tile.await %sm0_p0_h1_2
    %pv0_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h1">
    tile.await %pv0_p0_h1_2
    %qk0_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h2">
    tile.await %qk0_p0_h2_2
    %sm0_p0_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h2">
    tile.await %sm0_p0_h2_2
    %pv0_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h2">
    tile.await %pv0_p0_h2_2
    %qk0_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h3">
    tile.await %qk0_p0_h3_2
    %sm0_p0_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h3">
    tile.await %sm0_p0_h3_2
    %pv0_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h3">
    tile.await %pv0_p0_h3_2
    %qk0_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h0">
    tile.await %qk0_p1_h0_2
    %sm0_p1_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h0">
    tile.await %sm0_p1_h0_2
    %pv0_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h0">
    tile.await %pv0_p1_h0_2
    %qk0_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h1">
    tile.await %qk0_p1_h1_2
    %sm0_p1_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h1">
    tile.await %sm0_p1_h1_2
    %pv0_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h1">
    tile.await %pv0_p1_h1_2
    %qk0_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h2">
    tile.await %qk0_p1_h2_2
    %sm0_p1_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h2">
    tile.await %sm0_p1_h2_2
    %pv0_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h2">
    tile.await %pv0_p1_h2_2
    %qk0_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h3">
    tile.await %qk0_p1_h3_2
    %sm0_p1_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h3">
    tile.await %sm0_p1_h3_2
    %pv0_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h3">
    tile.await %pv0_p1_h3_2
    %140 = tile.subview %k_l2_4 task = %task_6 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %141 = tile.subview %v_l2_4 task = %task_6 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_2 = tile.load.async %140 into %k0_2 : !tile.event<"k2_loaded">
    %v2_loaded_2 = tile.load.async %141 into %v0_2 : !tile.event<"v2_loaded">
    tile.await %k1_loaded_2, %v1_loaded_2
    %qk1_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h0">
    tile.await %qk1_p0_h0_2
    %sm1_p0_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h0">
    tile.await %sm1_p0_h0_2
    %pv1_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h0">
    tile.await %pv1_p0_h0_2
    %qk1_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h1">
    tile.await %qk1_p0_h1_2
    %sm1_p0_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h1">
    tile.await %sm1_p0_h1_2
    %pv1_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h1">
    tile.await %pv1_p0_h1_2
    %qk1_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h2">
    tile.await %qk1_p0_h2_2
    %sm1_p0_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h2">
    tile.await %sm1_p0_h2_2
    %pv1_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h2">
    tile.await %pv1_p0_h2_2
    %qk1_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h3">
    tile.await %qk1_p0_h3_2
    %sm1_p0_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h3">
    tile.await %sm1_p0_h3_2
    %pv1_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h3">
    tile.await %pv1_p0_h3_2
    %qk1_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h0">
    tile.await %qk1_p1_h0_2
    %sm1_p1_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h0">
    tile.await %sm1_p1_h0_2
    %pv1_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h0">
    tile.await %pv1_p1_h0_2
    %qk1_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h1">
    tile.await %qk1_p1_h1_2
    %sm1_p1_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h1">
    tile.await %sm1_p1_h1_2
    %pv1_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h1">
    tile.await %pv1_p1_h1_2
    %qk1_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h2">
    tile.await %qk1_p1_h2_2
    %sm1_p1_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h2">
    tile.await %sm1_p1_h2_2
    %pv1_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h2">
    tile.await %pv1_p1_h2_2
    %qk1_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h3">
    tile.await %qk1_p1_h3_2
    %sm1_p1_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h3">
    tile.await %sm1_p1_h3_2
    %pv1_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h3">
    tile.await %pv1_p1_h3_2
    %142 = tile.subview %k_l2_4 task = %task_6 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %143 = tile.subview %v_l2_4 task = %task_6 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_2 = tile.load.async %142 into %k1_2 : !tile.event<"k3_loaded">
    %v3_loaded_2 = tile.load.async %143 into %v1_2 : !tile.event<"v3_loaded">
    tile.await %k2_loaded_2, %v2_loaded_2
    %qk2_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h0">
    tile.await %qk2_p0_h0_2
    %sm2_p0_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h0">
    tile.await %sm2_p0_h0_2
    %pv2_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h0">
    tile.await %pv2_p0_h0_2
    %qk2_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h1">
    tile.await %qk2_p0_h1_2
    %sm2_p0_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h1">
    tile.await %sm2_p0_h1_2
    %pv2_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h1">
    tile.await %pv2_p0_h1_2
    %qk2_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h2">
    tile.await %qk2_p0_h2_2
    %sm2_p0_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h2">
    tile.await %sm2_p0_h2_2
    %pv2_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h2">
    tile.await %pv2_p0_h2_2
    %qk2_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h3">
    tile.await %qk2_p0_h3_2
    %sm2_p0_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h3">
    tile.await %sm2_p0_h3_2
    %pv2_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h3">
    tile.await %pv2_p0_h3_2
    %qk2_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h0">
    tile.await %qk2_p1_h0_2
    %sm2_p1_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h0">
    tile.await %sm2_p1_h0_2
    %pv2_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h0">
    tile.await %pv2_p1_h0_2
    %qk2_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h1">
    tile.await %qk2_p1_h1_2
    %sm2_p1_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h1">
    tile.await %sm2_p1_h1_2
    %pv2_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h1">
    tile.await %pv2_p1_h1_2
    %qk2_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h2">
    tile.await %qk2_p1_h2_2
    %sm2_p1_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h2">
    tile.await %sm2_p1_h2_2
    %pv2_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h2">
    tile.await %pv2_p1_h2_2
    %qk2_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h3">
    tile.await %qk2_p1_h3_2
    %sm2_p1_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h3">
    tile.await %sm2_p1_h3_2
    %pv2_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h3">
    tile.await %pv2_p1_h3_2
    tile.await %k3_loaded_2, %v3_loaded_2
    tile.signal input_released(%task_6)
    %qk3_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h0">
    tile.await %qk3_p0_h0_2
    %sm3_p0_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h0">
    tile.await %sm3_p0_h0_2
    %pv3_p0_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h0">
    tile.await %pv3_p0_h0_2
    %qk3_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h1">
    tile.await %qk3_p0_h1_2
    %sm3_p0_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h1">
    tile.await %sm3_p0_h1_2
    %pv3_p0_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h1">
    tile.await %pv3_p0_h1_2
    %qk3_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h2">
    tile.await %qk3_p0_h2_2
    %sm3_p0_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h2">
    tile.await %sm3_p0_h2_2
    %pv3_p0_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h2">
    tile.await %pv3_p0_h2_2
    %qk3_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h3">
    tile.await %qk3_p0_h3_2
    %sm3_p0_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h3">
    tile.await %sm3_p0_h3_2
    %pv3_p0_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h3">
    tile.await %pv3_p0_h3_2
    %qk3_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h0">
    tile.await %qk3_p1_h0_2
    %sm3_p1_h0_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h0">
    tile.await %sm3_p1_h0_2
    %pv3_p1_h0_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h0">
    tile.await %pv3_p1_h0_2
    %qk3_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h1">
    tile.await %qk3_p1_h1_2
    %sm3_p1_h1_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h1">
    tile.await %sm3_p1_h1_2
    %pv3_p1_h1_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h1">
    tile.await %pv3_p1_h1_2
    %qk3_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h2">
    tile.await %qk3_p1_h2_2
    %sm3_p1_h2_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h2">
    tile.await %sm3_p1_h2_2
    %pv3_p1_h2_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h2">
    tile.await %pv3_p1_h2_2
    %qk3_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h3">
    tile.await %qk3_p1_h3_2
    %sm3_p1_h3_2 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h3">
    tile.await %sm3_p1_h3_2
    %pv3_p1_h3_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h3">
    tile.await %pv3_p1_h3_2
    %144 = tile.subview %o_l2_2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_2 = tile.store.async %acc0_2 into %144 : !tile.event<"o0_stored">
    %145 = tile.subview %o_l2_2 task = %task_6 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_2 = tile.store.async %acc1_2 into %145 : !tile.event<"o1_stored">
    tile.await %o0_stored_2, %o1_stored_2
    tile.signal output_ready(%task_6)
    tile.free %q0_2
    tile.free %q1_2
    tile.free %acc0_2
    tile.free %acc1_2
    tile.free %m0_2
    tile.free %m1_2
    tile.free %l0_2
    tile.free %l1_2
    tile.free %score_2
    tile.free %k0_2
    tile.free %k1_2
    tile.free %v0_2
    tile.free %v1_2
    tile.return
  }
  tile.program @prefill_attention_q3(
    %task_7: !nest.task, %q_l2_5: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_5: !nest.l2_buffer<4x512x64xbf16>, %v_l2_5: !nest.l2_buffer<4x512x64xbf16>,
    %o_l2_3: !nest.l2_buffer<4x128x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0],
      tile_l1_spm_bytes_per_context = 245760> {
    %q0_3 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %q1_3 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc0_3 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc1_3 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %m0_3 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %m1_3 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l0_3 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %l1_3 = tile.alloc shape = [4, 64] dtype = "f32" alignment = 256 : !tile.l1_buffer<4x64xf32>
    %score_3 = tile.alloc shape = [64, 128] dtype = "f32" alignment = 256
      : !tile.l1_buffer<64x128xf32>
    %k0_3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %k1_3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v0_3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v1_3 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %146 = tile.subview %q_l2_5 task = %task_7 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_3 = tile.load.async %146 into %q0_3 : !tile.event<"q0_loaded">
    %147 = tile.subview %q_l2_5 task = %task_7 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_3 = tile.load.async %147 into %q1_3 : !tile.event<"q1_loaded">
    %148 = tile.subview %k_l2_5 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %149 = tile.subview %v_l2_5 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_3 = tile.load.async %148 into %k0_3 : !tile.event<"k0_loaded">
    %v0_loaded_3 = tile.load.async %149 into %v0_3 : !tile.event<"v0_loaded">
    %150 = tile.subview %k_l2_5 task = %task_7 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %151 = tile.subview %v_l2_5 task = %task_7 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_3 = tile.load.async %150 into %k1_3 : !tile.event<"k1_loaded">
    %v1_loaded_3 = tile.load.async %151 into %v1_3 : !tile.event<"v1_loaded">
    tile.await %q0_loaded_3, %q1_loaded_3
    tile.await %k0_loaded_3, %v0_loaded_3
    %qk0_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h0">
    tile.await %qk0_p0_h0_3
    %sm0_p0_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h0">
    tile.await %sm0_p0_h0_3
    %pv0_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h0">
    tile.await %pv0_p0_h0_3
    %qk0_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h1">
    tile.await %qk0_p0_h1_3
    %sm0_p0_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h1">
    tile.await %sm0_p0_h1_3
    %pv0_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h1">
    tile.await %pv0_p0_h1_3
    %qk0_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h2">
    tile.await %qk0_p0_h2_3
    %sm0_p0_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h2">
    tile.await %sm0_p0_h2_3
    %pv0_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h2">
    tile.await %pv0_p0_h2_3
    %qk0_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p0_h3">
    tile.await %qk0_p0_h3_3
    %sm0_p0_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p0_h3">
    tile.await %sm0_p0_h3_3
    %pv0_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p0_h3">
    tile.await %pv0_p0_h3_3
    %qk0_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h0">
    tile.await %qk0_p1_h0_3
    %sm0_p1_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h0">
    tile.await %sm0_p1_h0_3
    %pv0_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h0">
    tile.await %pv0_p1_h0_3
    %qk0_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h1">
    tile.await %qk0_p1_h1_3
    %sm0_p1_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h1">
    tile.await %sm0_p1_h1_3
    %pv0_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h1">
    tile.await %pv0_p1_h1_3
    %qk0_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h2">
    tile.await %qk0_p1_h2_3
    %sm0_p1_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h2">
    tile.await %sm0_p1_h2_3
    %pv0_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h2">
    tile.await %pv0_p1_h2_3
    %qk0_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk0_p1_h3">
    tile.await %qk0_p1_h3_3
    %sm0_p1_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm0_p1_h3">
    tile.await %sm0_p1_h3_3
    %pv0_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576
      : !tile.event<"pv0_p1_h3">
    tile.await %pv0_p1_h3_3
    %152 = tile.subview %k_l2_5 task = %task_7 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %153 = tile.subview %v_l2_5 task = %task_7 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_3 = tile.load.async %152 into %k0_3 : !tile.event<"k2_loaded">
    %v2_loaded_3 = tile.load.async %153 into %v0_3 : !tile.event<"v2_loaded">
    tile.await %k1_loaded_3, %v1_loaded_3
    %qk1_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h0">
    tile.await %qk1_p0_h0_3
    %sm1_p0_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h0">
    tile.await %sm1_p0_h0_3
    %pv1_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h0">
    tile.await %pv1_p0_h0_3
    %qk1_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h1">
    tile.await %qk1_p0_h1_3
    %sm1_p0_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h1">
    tile.await %sm1_p0_h1_3
    %pv1_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h1">
    tile.await %pv1_p0_h1_3
    %qk1_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h2">
    tile.await %qk1_p0_h2_3
    %sm1_p0_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h2">
    tile.await %sm1_p0_h2_3
    %pv1_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h2">
    tile.await %pv1_p0_h2_3
    %qk1_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p0_h3">
    tile.await %qk1_p0_h3_3
    %sm1_p0_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p0_h3">
    tile.await %sm1_p0_h3_3
    %pv1_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p0_h3">
    tile.await %pv1_p0_h3_3
    %qk1_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h0">
    tile.await %qk1_p1_h0_3
    %sm1_p1_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h0">
    tile.await %sm1_p1_h0_3
    %pv1_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h0">
    tile.await %pv1_p1_h0_3
    %qk1_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h1">
    tile.await %qk1_p1_h1_3
    %sm1_p1_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h1">
    tile.await %sm1_p1_h1_3
    %pv1_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h1">
    tile.await %pv1_p1_h1_3
    %qk1_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h2">
    tile.await %qk1_p1_h2_3
    %sm1_p1_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h2">
    tile.await %sm1_p1_h2_3
    %pv1_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h2">
    tile.await %pv1_p1_h2_3
    %qk1_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk1_p1_h3">
    tile.await %qk1_p1_h3_3
    %sm1_p1_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm1_p1_h3">
    tile.await %sm1_p1_h3_3
    %pv1_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv1_p1_h3">
    tile.await %pv1_p1_h3_3
    %154 = tile.subview %k_l2_5 task = %task_7 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %155 = tile.subview %v_l2_5 task = %task_7 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_3 = tile.load.async %154 into %k1_3 : !tile.event<"k3_loaded">
    %v3_loaded_3 = tile.load.async %155 into %v1_3 : !tile.event<"v3_loaded">
    tile.await %k2_loaded_3, %v2_loaded_3
    %qk2_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h0">
    tile.await %qk2_p0_h0_3
    %sm2_p0_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h0">
    tile.await %sm2_p0_h0_3
    %pv2_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h0">
    tile.await %pv2_p0_h0_3
    %qk2_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h1">
    tile.await %qk2_p0_h1_3
    %sm2_p0_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h1">
    tile.await %sm2_p0_h1_3
    %pv2_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h1">
    tile.await %pv2_p0_h1_3
    %qk2_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h2">
    tile.await %qk2_p0_h2_3
    %sm2_p0_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h2">
    tile.await %sm2_p0_h2_3
    %pv2_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h2">
    tile.await %pv2_p0_h2_3
    %qk2_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p0_h3">
    tile.await %qk2_p0_h3_3
    %sm2_p0_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p0_h3">
    tile.await %sm2_p0_h3_3
    %pv2_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p0_h3">
    tile.await %pv2_p0_h3_3
    %qk2_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h0">
    tile.await %qk2_p1_h0_3
    %sm2_p1_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h0">
    tile.await %sm2_p1_h0_3
    %pv2_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h0">
    tile.await %pv2_p1_h0_3
    %qk2_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h1">
    tile.await %qk2_p1_h1_3
    %sm2_p1_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h1">
    tile.await %sm2_p1_h1_3
    %pv2_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h1">
    tile.await %pv2_p1_h1_3
    %qk2_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h2">
    tile.await %qk2_p1_h2_3
    %sm2_p1_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h2">
    tile.await %sm2_p1_h2_3
    %pv2_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h2">
    tile.await %pv2_p1_h2_3
    %qk2_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk2_p1_h3">
    tile.await %qk2_p1_h3_3
    %sm2_p1_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm2_p1_h3">
    tile.await %sm2_p1_h3_3
    %pv2_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv2_p1_h3">
    tile.await %pv2_p1_h3_3
    tile.await %k3_loaded_3, %v3_loaded_3
    tile.signal input_released(%task_7)
    %qk3_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h0">
    tile.await %qk3_p0_h0_3
    %sm3_p0_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h0">
    tile.await %sm3_p0_h0_3
    %pv3_p0_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h0">
    tile.await %pv3_p0_h0_3
    %qk3_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h1">
    tile.await %qk3_p0_h1_3
    %sm3_p0_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h1">
    tile.await %sm3_p0_h1_3
    %pv3_p0_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h1">
    tile.await %pv3_p0_h1_3
    %qk3_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h2">
    tile.await %qk3_p0_h2_3
    %sm3_p0_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h2">
    tile.await %sm3_p0_h2_3
    %pv3_p0_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h2">
    tile.await %pv3_p0_h2_3
    %qk3_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p0_h3">
    tile.await %qk3_p0_h3_3
    %sm3_p0_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p0_h3">
    tile.await %sm3_p0_h3_3
    %pv3_p0_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p0_h3">
    tile.await %pv3_p0_h3_3
    %qk3_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h0">
    tile.await %qk3_p1_h0_3
    %sm3_p1_h0_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h0">
    tile.await %sm3_p1_h0_3
    %pv3_p1_h0_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h0">
    tile.await %pv3_p1_h0_3
    %qk3_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h1">
    tile.await %qk3_p1_h1_3
    %sm3_p1_h1_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h1">
    tile.await %sm3_p1_h1_3
    %pv3_p1_h1_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h1">
    tile.await %pv3_p1_h1_3
    %qk3_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h2">
    tile.await %qk3_p1_h2_3
    %sm3_p1_h2_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h2">
    tile.await %sm3_p1_h2_3
    %pv3_p1_h2_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h2">
    tile.await %pv3_p1_h2_3
    %qk3_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 128 k = 64 ops = 1048576
      : !tile.event<"qk3_p1_h3">
    tile.await %qk3_p1_h3_3
    %sm3_p1_h3_3 = tile.evu.async "online_softmax_update" ops = 12416 : !tile.event<"sm3_p1_h3">
    tile.await %sm3_p1_h3_3
    %pv3_p1_h3_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"pv3_p1_h3">
    tile.await %pv3_p1_h3_3
    %156 = tile.subview %o_l2_3 task = %task_7 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_3 = tile.store.async %acc0_3 into %156 : !tile.event<"o0_stored">
    %157 = tile.subview %o_l2_3 task = %task_7 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_3 = tile.store.async %acc1_3 into %157 : !tile.event<"o1_stored">
    tile.await %o0_stored_3, %o1_stored_3
    tile.signal output_ready(%task_7)
    tile.free %q0_3
    tile.free %q1_3
    tile.free %acc0_3
    tile.free %acc1_3
    tile.free %m0_3
    tile.free %m1_3
    tile.free %l0_3
    tile.free %l1_3
    tile.free %score_3
    tile.free %k0_3
    tile.free %k1_3
    tile.free %v0_3
    tile.free %v1_3
    tile.return
  }
  tile.program @prefill_outproj_lo(
    %task_8: !nest.task, %o_l2_4: !nest.l2_buffer<4x128x256xbf16>,
    %wo_l2: !nest.l2_buffer<4x1024x256xbf16>, %out_l2: !nest.l2_buffer<4x64x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 98304> {
    %acc = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %o = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %w_buf = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %158 = tile.subview %o_l2_4 offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o0_loaded = tile.load.async %158 into %o : !tile.event<"o0_loaded">
    %159 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_0_loaded = tile.load.async %159 into %w_buf : !tile.event<"w0_0_loaded">
    tile.await %w0_0_loaded, %o0_loaded
    %outproj0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
      : !tile.event<"outproj0_0">
    tile.await %outproj0
    %160 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_1_loaded = tile.load.async %160 into %w_buf : !tile.event<"w0_1_loaded">
    tile.await %w0_1_loaded
    %outproj0_1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_1">
    tile.await %outproj0_1
    %161 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_2_loaded = tile.load.async %161 into %w_buf : !tile.event<"w0_2_loaded">
    tile.await %w0_2_loaded
    %outproj0_2 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_2">
    tile.await %outproj0_2
    %162 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_3_loaded = tile.load.async %162 into %w_buf : !tile.event<"w0_3_loaded">
    tile.await %w0_3_loaded
    %outproj0_3 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_3">
    tile.await %outproj0_3
    %163 = tile.subview %o_l2_4 offsets = [1, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o1_loaded = tile.load.async %163 into %o : !tile.event<"o1_loaded">
    %164 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_0_loaded = tile.load.async %164 into %w_buf : !tile.event<"w1_0_loaded">
    tile.await %w1_0_loaded, %o1_loaded
    %outproj1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_0">
    tile.await %outproj1
    %165 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_1_loaded = tile.load.async %165 into %w_buf : !tile.event<"w1_1_loaded">
    tile.await %w1_1_loaded
    %outproj1_1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_1">
    tile.await %outproj1_1
    %166 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_2_loaded = tile.load.async %166 into %w_buf : !tile.event<"w1_2_loaded">
    tile.await %w1_2_loaded
    %outproj1_2 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_2">
    tile.await %outproj1_2
    %167 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_3_loaded = tile.load.async %167 into %w_buf : !tile.event<"w1_3_loaded">
    tile.await %w1_3_loaded
    %outproj1_3 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_3">
    tile.await %outproj1_3
    %168 = tile.subview %o_l2_4 offsets = [2, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o2_loaded = tile.load.async %168 into %o : !tile.event<"o2_loaded">
    %169 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 512, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_0_loaded = tile.load.async %169 into %w_buf : !tile.event<"w2_0_loaded">
    tile.await %w2_0_loaded, %o2_loaded
    %outproj2 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_0">
    tile.await %outproj2
    %170 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 576, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_1_loaded = tile.load.async %170 into %w_buf : !tile.event<"w2_1_loaded">
    tile.await %w2_1_loaded
    %outproj2_1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_1">
    tile.await %outproj2_1
    %171 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 640, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_2_loaded = tile.load.async %171 into %w_buf : !tile.event<"w2_2_loaded">
    tile.await %w2_2_loaded
    %outproj2_2 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_2">
    tile.await %outproj2_2
    %172 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 704, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_3_loaded = tile.load.async %172 into %w_buf : !tile.event<"w2_3_loaded">
    tile.await %w2_3_loaded
    %outproj2_3 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_3">
    tile.await %outproj2_3
    %173 = tile.subview %o_l2_4 offsets = [3, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o3_loaded = tile.load.async %173 into %o : !tile.event<"o3_loaded">
    %174 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 768, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_0_loaded = tile.load.async %174 into %w_buf : !tile.event<"w3_0_loaded">
    tile.await %w3_0_loaded, %o3_loaded
    %outproj3 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_0">
    tile.await %outproj3
    %175 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 832, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_1_loaded = tile.load.async %175 into %w_buf : !tile.event<"w3_1_loaded">
    tile.await %w3_1_loaded
    %outproj3_1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_1">
    tile.await %outproj3_1
    %176 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 896, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_2_loaded = tile.load.async %176 into %w_buf : !tile.event<"w3_2_loaded">
    tile.await %w3_2_loaded
    %outproj3_2 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_2">
    tile.await %outproj3_2
    %177 = tile.subview %wo_l2 task = %task_8 task_dim = 0 offsets = [0, 960, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_3_loaded = tile.load.async %177 into %w_buf : !tile.event<"w3_3_loaded">
    tile.await %w3_3_loaded
    tile.signal input_released(%task_8)
    %outproj3_3 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_3">
    tile.await %outproj3_3
    %178 = tile.subview %out_l2 task = %task_8 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %out_stored = tile.store.async %acc into %178 : !tile.event<"out_stored">
    tile.await %out_stored
    tile.signal output_ready(%task_8)
    tile.free %acc
    tile.free %o
    tile.free %w_buf
    tile.return
  }
  tile.program @prefill_outproj_hi(
    %task_9: !nest.task, %o_l2_5: !nest.l2_buffer<4x128x256xbf16>,
    %wo_l2_1: !nest.l2_buffer<4x1024x256xbf16>, %out_l2_1: !nest.l2_buffer<4x64x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 98304> {
    %acc_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %o_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %w_buf_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %179 = tile.subview %o_l2_5 offsets = [0, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o0_loaded_1 = tile.load.async %179 into %o_1 : !tile.event<"o0_loaded">
    %180 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_0_loaded_1 = tile.load.async %180 into %w_buf_1 : !tile.event<"w0_0_loaded">
    tile.await %w0_0_loaded_1, %o0_loaded_1
    %outproj0_4 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
      : !tile.event<"outproj0_0">
    tile.await %outproj0_4
    %181 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_1_loaded_1 = tile.load.async %181 into %w_buf_1 : !tile.event<"w0_1_loaded">
    tile.await %w0_1_loaded_1
    %outproj0_5 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_1">
    tile.await %outproj0_5
    %182 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_2_loaded_1 = tile.load.async %182 into %w_buf_1 : !tile.event<"w0_2_loaded">
    tile.await %w0_2_loaded_1
    %outproj0_6 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_2">
    tile.await %outproj0_6
    %183 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_3_loaded_1 = tile.load.async %183 into %w_buf_1 : !tile.event<"w0_3_loaded">
    tile.await %w0_3_loaded_1
    %outproj0_7 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_3">
    tile.await %outproj0_7
    %184 = tile.subview %o_l2_5 offsets = [1, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o1_loaded_1 = tile.load.async %184 into %o_1 : !tile.event<"o1_loaded">
    %185 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_0_loaded_1 = tile.load.async %185 into %w_buf_1 : !tile.event<"w1_0_loaded">
    tile.await %w1_0_loaded_1, %o1_loaded_1
    %outproj1_4 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_0">
    tile.await %outproj1_4
    %186 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_1_loaded_1 = tile.load.async %186 into %w_buf_1 : !tile.event<"w1_1_loaded">
    tile.await %w1_1_loaded_1
    %outproj1_5 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_1">
    tile.await %outproj1_5
    %187 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_2_loaded_1 = tile.load.async %187 into %w_buf_1 : !tile.event<"w1_2_loaded">
    tile.await %w1_2_loaded_1
    %outproj1_6 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_2">
    tile.await %outproj1_6
    %188 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_3_loaded_1 = tile.load.async %188 into %w_buf_1 : !tile.event<"w1_3_loaded">
    tile.await %w1_3_loaded_1
    %outproj1_7 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_3">
    tile.await %outproj1_7
    %189 = tile.subview %o_l2_5 offsets = [2, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o2_loaded_1 = tile.load.async %189 into %o_1 : !tile.event<"o2_loaded">
    %190 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 512, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_0_loaded_1 = tile.load.async %190 into %w_buf_1 : !tile.event<"w2_0_loaded">
    tile.await %w2_0_loaded_1, %o2_loaded_1
    %outproj2_4 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_0">
    tile.await %outproj2_4
    %191 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 576, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_1_loaded_1 = tile.load.async %191 into %w_buf_1 : !tile.event<"w2_1_loaded">
    tile.await %w2_1_loaded_1
    %outproj2_5 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_1">
    tile.await %outproj2_5
    %192 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 640, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_2_loaded_1 = tile.load.async %192 into %w_buf_1 : !tile.event<"w2_2_loaded">
    tile.await %w2_2_loaded_1
    %outproj2_6 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_2">
    tile.await %outproj2_6
    %193 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 704, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_3_loaded_1 = tile.load.async %193 into %w_buf_1 : !tile.event<"w2_3_loaded">
    tile.await %w2_3_loaded_1
    %outproj2_7 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_3">
    tile.await %outproj2_7
    %194 = tile.subview %o_l2_5 offsets = [3, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o3_loaded_1 = tile.load.async %194 into %o_1 : !tile.event<"o3_loaded">
    %195 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 768, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_0_loaded_1 = tile.load.async %195 into %w_buf_1 : !tile.event<"w3_0_loaded">
    tile.await %w3_0_loaded_1, %o3_loaded_1
    %outproj3_4 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_0">
    tile.await %outproj3_4
    %196 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 832, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_1_loaded_1 = tile.load.async %196 into %w_buf_1 : !tile.event<"w3_1_loaded">
    tile.await %w3_1_loaded_1
    %outproj3_5 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_1">
    tile.await %outproj3_5
    %197 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 896, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_2_loaded_1 = tile.load.async %197 into %w_buf_1 : !tile.event<"w3_2_loaded">
    tile.await %w3_2_loaded_1
    %outproj3_6 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_2">
    tile.await %outproj3_6
    %198 = tile.subview %wo_l2_1 task = %task_9 task_dim = 0 offsets = [0, 960, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_3_loaded_1 = tile.load.async %198 into %w_buf_1 : !tile.event<"w3_3_loaded">
    tile.await %w3_3_loaded_1
    tile.signal input_released(%task_9)
    %outproj3_7 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_3">
    tile.await %outproj3_7
    %199 = tile.subview %out_l2_1 task = %task_9 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %out_stored_1 = tile.store.async %acc_1 into %199 : !tile.event<"out_stored">
    tile.await %out_stored_1
    tile.signal output_ready(%task_9)
    tile.free %acc_1
    tile.free %o_1
    tile.free %w_buf_1
    tile.return
  }
  nest.context @prefill_dispatchparallel_ctx(
    %X: !nest.global_memref<8x512x128xbf16>, %WQ: !nest.global_memref<8x4x128x256xbf16>,
    %WK: !nest.global_memref<8x4x128x64xbf16>, %WV: !nest.global_memref<8x4x128x64xbf16>,
    %WO: !nest.global_memref<4x1024x256xbf16>, %OUT: !nest.global_memref<4x2x4x64x256xbf16>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1],
      logical_tasks = 112, l2_spm_bytes = 6815744, requested_contexts_per_tile = 4> {
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
    %q_l2_6 = nest.alloc slot = "q_l2" role = "inout" sharing = "context-local"
      shape = [4, 512, 256] dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x256xbf16>
    %k_l2_6 = nest.alloc slot = "k_l2" role = "inout" sharing = "context-local" shape = [4, 512, 64]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x64xbf16>
    %v_l2_6 = nest.alloc slot = "v_l2" role = "inout" sharing = "context-local" shape = [4, 512, 64]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x512x64xbf16>
    %wo_l2_2 = nest.alloc slot = "wo_l2" role = "in" shape = [4, 1024, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x256xbf16>
    %o_q0 = nest.alloc slot = "o_q0" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q1 = nest.alloc slot = "o_q1" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q2 = nest.alloc slot = "o_q2" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q3 = nest.alloc slot = "o_q3" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %out_q0_lo = nest.alloc slot = "out_q0_lo" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q0_hi = nest.alloc slot = "out_q0_hi" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q1_lo = nest.alloc slot = "out_q1_lo" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q1_hi = nest.alloc slot = "out_q1_hi" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q2_lo = nest.alloc slot = "out_q2_lo" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q2_hi = nest.alloc slot = "out_q2_hi" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q3_lo = nest.alloc slot = "out_q3_lo" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %out_q3_hi = nest.alloc slot = "out_q3_hi" role = "out" shape = [4, 64, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x64x256xbf16>
    %200 = nest.task.range from = 0 to = 4 : !nest.task_range
    %201 = nest.subview %X offsets = [0, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %202 = nest.subview %WQ offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %203 = nest.subview %WK offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %204 = nest.subview %WV offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x = nest.dma.prefetch.async %201 into %x_p0 : !nest.event<"pre_x_0">
    %pre_wq = nest.dma.prefetch.async %202 into %wq_p0 : !nest.event<"pre_wq_0">
    %pre_wk = nest.dma.prefetch.async %203 into %wk_p0 : !nest.event<"pre_wk_0">
    %pre_wv = nest.dma.prefetch.async %204 into %wv_p0 : !nest.event<"pre_wv_0">
    %qkv_q_grid, %qkv_q_inrel, %qkv_q_out = nest.dispatch.tasks.async @qkv_q_init l1_mode = 0
      tasks(%200) globals() bindings(%x_p0, %wq_p0, %q_l2_6) ins(%x_p0, %wq_p0) outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x, %pre_wq)
      : (!nest.event<"qkv_q_grid_0">, !nest.event<"qkv_q_inrel_0">, !nest.event<"qkv_q_out_0">)
    %qkv_kv_grid, %qkv_kv_inrel, %qkv_kv_out = nest.dispatch.tasks.async @qkv_kv_init l1_mode = 0
      tasks(%200) globals() bindings(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wk_p0, %wv_p0) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x, %pre_wk, %pre_wv)
      : (!nest.event<"qkv_kv_grid_0">, !nest.event<"qkv_kv_inrel_0">, !nest.event<"qkv_kv_out_0">)
    %205 = nest.subview %X offsets = [1, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %206 = nest.subview %WQ offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %207 = nest.subview %WK offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %208 = nest.subview %WV offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_1 = nest.dma.prefetch.async %205 into %x_p1 : !nest.event<"pre_x_1">
    %pre_wq_1 = nest.dma.prefetch.async %206 into %wq_p1 : !nest.event<"pre_wq_1">
    %pre_wk_1 = nest.dma.prefetch.async %207 into %wk_p1 : !nest.event<"pre_wk_1">
    %pre_wv_1 = nest.dma.prefetch.async %208 into %wv_p1 : !nest.event<"pre_wv_1">
    %qkv_q_grid_1, %qkv_q_inrel_1, %qkv_q_out_1 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p1, %wq_p1, %q_l2_6) ins(%x_p1, %wq_p1, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_1, %pre_wq_1, %qkv_q_out)
      : (!nest.event<"qkv_q_grid_1">, !nest.event<"qkv_q_inrel_1">, !nest.event<"qkv_q_out_1">)
    %qkv_kv_grid_1, %qkv_kv_inrel_1, %qkv_kv_out_1 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_1, %pre_wk_1, %pre_wv_1, %qkv_kv_out)
      : (!nest.event<"qkv_kv_grid_1">, !nest.event<"qkv_kv_inrel_1">, !nest.event<"qkv_kv_out_1">)
    %209 = nest.subview %X offsets = [2, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %210 = nest.subview %WQ offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %211 = nest.subview %WK offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %212 = nest.subview %WV offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_2 = nest.dma.prefetch.async %209 into %x_p0 depends_on(%qkv_q_inrel, %qkv_kv_inrel)
      : !nest.event<"pre_x_2">
    %pre_wq_2 = nest.dma.prefetch.async %210 into %wq_p0 depends_on(%qkv_q_inrel)
      : !nest.event<"pre_wq_2">
    %pre_wk_2 = nest.dma.prefetch.async %211 into %wk_p0 depends_on(%qkv_kv_inrel)
      : !nest.event<"pre_wk_2">
    %pre_wv_2 = nest.dma.prefetch.async %212 into %wv_p0 depends_on(%qkv_kv_inrel)
      : !nest.event<"pre_wv_2">
    %qkv_q_grid_2, %qkv_q_inrel_2, %qkv_q_out_2 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p0, %wq_p0, %q_l2_6) ins(%x_p0, %wq_p0, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_2, %pre_wq_2, %qkv_q_out_1)
      : (!nest.event<"qkv_q_grid_2">, !nest.event<"qkv_q_inrel_2">, !nest.event<"qkv_q_out_2">)
    %qkv_kv_grid_2, %qkv_kv_inrel_2, %qkv_kv_out_2 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_2, %pre_wk_2, %pre_wv_2, %qkv_kv_out_1)
      : (!nest.event<"qkv_kv_grid_2">, !nest.event<"qkv_kv_inrel_2">, !nest.event<"qkv_kv_out_2">)
    %213 = nest.subview %X offsets = [3, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %214 = nest.subview %WQ offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %215 = nest.subview %WK offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %216 = nest.subview %WV offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_3 = nest.dma.prefetch.async %213 into %x_p1 depends_on(%qkv_q_inrel_1, %qkv_kv_inrel_1)
      : !nest.event<"pre_x_3">
    %pre_wq_3 = nest.dma.prefetch.async %214 into %wq_p1 depends_on(%qkv_q_inrel_1)
      : !nest.event<"pre_wq_3">
    %pre_wk_3 = nest.dma.prefetch.async %215 into %wk_p1 depends_on(%qkv_kv_inrel_1)
      : !nest.event<"pre_wk_3">
    %pre_wv_3 = nest.dma.prefetch.async %216 into %wv_p1 depends_on(%qkv_kv_inrel_1)
      : !nest.event<"pre_wv_3">
    %qkv_q_grid_3, %qkv_q_inrel_3, %qkv_q_out_3 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p1, %wq_p1, %q_l2_6) ins(%x_p1, %wq_p1, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_3, %pre_wq_3, %qkv_q_out_2)
      : (!nest.event<"qkv_q_grid_3">, !nest.event<"qkv_q_inrel_3">, !nest.event<"qkv_q_out_3">)
    %qkv_kv_grid_3, %qkv_kv_inrel_3, %qkv_kv_out_3 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_3, %pre_wk_3, %pre_wv_3, %qkv_kv_out_2)
      : (!nest.event<"qkv_kv_grid_3">, !nest.event<"qkv_kv_inrel_3">, !nest.event<"qkv_kv_out_3">)
    %217 = nest.subview %X offsets = [4, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %218 = nest.subview %WQ offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %219 = nest.subview %WK offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %220 = nest.subview %WV offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_4 = nest.dma.prefetch.async %217 into %x_p0 depends_on(%qkv_q_inrel_2, %qkv_kv_inrel_2)
      : !nest.event<"pre_x_4">
    %pre_wq_4 = nest.dma.prefetch.async %218 into %wq_p0 depends_on(%qkv_q_inrel_2)
      : !nest.event<"pre_wq_4">
    %pre_wk_4 = nest.dma.prefetch.async %219 into %wk_p0 depends_on(%qkv_kv_inrel_2)
      : !nest.event<"pre_wk_4">
    %pre_wv_4 = nest.dma.prefetch.async %220 into %wv_p0 depends_on(%qkv_kv_inrel_2)
      : !nest.event<"pre_wv_4">
    %qkv_q_grid_4, %qkv_q_inrel_4, %qkv_q_out_4 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p0, %wq_p0, %q_l2_6) ins(%x_p0, %wq_p0, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_4, %pre_wq_4, %qkv_q_out_3, %qkv_q_grid)
      : (!nest.event<"qkv_q_grid_4">, !nest.event<"qkv_q_inrel_4">, !nest.event<"qkv_q_out_4">)
    %qkv_kv_grid_4, %qkv_kv_inrel_4, %qkv_kv_out_4 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_4, %pre_wk_4, %pre_wv_4, %qkv_kv_out_3, %qkv_kv_grid)
      : (!nest.event<"qkv_kv_grid_4">, !nest.event<"qkv_kv_inrel_4">, !nest.event<"qkv_kv_out_4">)
    %221 = nest.subview %X offsets = [5, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %222 = nest.subview %WQ offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %223 = nest.subview %WK offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %224 = nest.subview %WV offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_5 = nest.dma.prefetch.async %221 into %x_p1 depends_on(%qkv_q_inrel_3, %qkv_kv_inrel_3)
      : !nest.event<"pre_x_5">
    %pre_wq_5 = nest.dma.prefetch.async %222 into %wq_p1 depends_on(%qkv_q_inrel_3)
      : !nest.event<"pre_wq_5">
    %pre_wk_5 = nest.dma.prefetch.async %223 into %wk_p1 depends_on(%qkv_kv_inrel_3)
      : !nest.event<"pre_wk_5">
    %pre_wv_5 = nest.dma.prefetch.async %224 into %wv_p1 depends_on(%qkv_kv_inrel_3)
      : !nest.event<"pre_wv_5">
    %qkv_q_grid_5, %qkv_q_inrel_5, %qkv_q_out_5 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p1, %wq_p1, %q_l2_6) ins(%x_p1, %wq_p1, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_5, %pre_wq_5, %qkv_q_out_4, %qkv_q_grid_1)
      : (!nest.event<"qkv_q_grid_5">, !nest.event<"qkv_q_inrel_5">, !nest.event<"qkv_q_out_5">)
    %qkv_kv_grid_5, %qkv_kv_inrel_5, %qkv_kv_out_5 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_5, %pre_wk_5, %pre_wv_5, %qkv_kv_out_4, %qkv_kv_grid_1)
      : (!nest.event<"qkv_kv_grid_5">, !nest.event<"qkv_kv_inrel_5">, !nest.event<"qkv_kv_out_5">)
    %225 = nest.subview %X offsets = [6, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %226 = nest.subview %WQ offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %227 = nest.subview %WK offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %228 = nest.subview %WV offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_6 = nest.dma.prefetch.async %225 into %x_p0 depends_on(%qkv_q_inrel_4, %qkv_kv_inrel_4)
      : !nest.event<"pre_x_6">
    %pre_wq_6 = nest.dma.prefetch.async %226 into %wq_p0 depends_on(%qkv_q_inrel_4)
      : !nest.event<"pre_wq_6">
    %pre_wk_6 = nest.dma.prefetch.async %227 into %wk_p0 depends_on(%qkv_kv_inrel_4)
      : !nest.event<"pre_wk_6">
    %pre_wv_6 = nest.dma.prefetch.async %228 into %wv_p0 depends_on(%qkv_kv_inrel_4)
      : !nest.event<"pre_wv_6">
    %qkv_q_grid_6, %qkv_q_inrel_6, %qkv_q_out_6 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p0, %wq_p0, %q_l2_6) ins(%x_p0, %wq_p0, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_6, %pre_wq_6, %qkv_q_out_5, %qkv_q_grid_2)
      : (!nest.event<"qkv_q_grid_6">, !nest.event<"qkv_q_inrel_6">, !nest.event<"qkv_q_out_6">)
    %qkv_kv_grid_6, %qkv_kv_inrel_6, %qkv_kv_out_6 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wk_p0, %wv_p0, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_6, %pre_wk_6, %pre_wv_6, %qkv_kv_out_5, %qkv_kv_grid_2)
      : (!nest.event<"qkv_kv_grid_6">, !nest.event<"qkv_kv_inrel_6">, !nest.event<"qkv_kv_out_6">)
    %229 = nest.subview %X offsets = [7, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %230 = nest.subview %WQ offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %231 = nest.subview %WK offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %232 = nest.subview %WV offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_7 = nest.dma.prefetch.async %229 into %x_p1 depends_on(%qkv_q_inrel_5, %qkv_kv_inrel_5)
      : !nest.event<"pre_x_7">
    %pre_wq_7 = nest.dma.prefetch.async %230 into %wq_p1 depends_on(%qkv_q_inrel_5)
      : !nest.event<"pre_wq_7">
    %pre_wk_7 = nest.dma.prefetch.async %231 into %wk_p1 depends_on(%qkv_kv_inrel_5)
      : !nest.event<"pre_wk_7">
    %pre_wv_7 = nest.dma.prefetch.async %232 into %wv_p1 depends_on(%qkv_kv_inrel_5)
      : !nest.event<"pre_wv_7">
    %qkv_q_grid_7, %qkv_q_inrel_7, %qkv_q_out_7 = nest.dispatch.tasks.async @qkv_q_accum l1_mode = 0
      tasks(%200) globals() bindings(%x_p1, %wq_p1, %q_l2_6) ins(%x_p1, %wq_p1, %q_l2_6)
      outs(%q_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_7, %pre_wq_7, %qkv_q_out_6, %qkv_q_grid_3)
      : (!nest.event<"qkv_q_grid_7">, !nest.event<"qkv_q_inrel_7">, !nest.event<"qkv_q_out_7">)
    %qkv_kv_grid_7, %qkv_kv_inrel_7, %qkv_kv_out_7 = nest.dispatch.tasks.async @qkv_kv_accum
      l1_mode = 0 tasks(%200) globals() bindings(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wk_p1, %wv_p1, %k_l2_6, %v_l2_6) outs(%k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_7, %pre_wk_7, %pre_wv_7, %qkv_kv_out_6, %qkv_kv_grid_3)
      : (!nest.event<"qkv_kv_grid_7">, !nest.event<"qkv_kv_inrel_7">, !nest.event<"qkv_kv_out_7">)
    %233 = nest.subview %WO offsets = [0, 0, 0] sizes = [4, 1024, 256] strides = [1, 1, 1]
      : !nest.global_view<4x1024x256xbf16>
    %pre_wo = nest.dma.prefetch.async %233 into %wo_l2_2 depends_on(%qkv_q_inrel_7, %qkv_kv_inrel_7)
      : !nest.event<"pre_wo">
    %att_grid, %att_inrel, %att_out = nest.dispatch.tasks.async @prefill_attention_q0 l1_mode = 0
      tasks(%200) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q0)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_q_out_7, %qkv_kv_out_7, %qkv_q_grid_4)
      : (!nest.event<"att_grid_0">, !nest.event<"att_inrel_0">, !nest.event<"att_out_0">)
    %att_grid_1, %att_inrel_1, %att_out_1 = nest.dispatch.tasks.async @prefill_attention_q1
      l1_mode = 0 tasks(%200) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q1)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_q_out_7, %qkv_kv_out_7, %qkv_kv_grid_4)
      : (!nest.event<"att_grid_1">, !nest.event<"att_inrel_1">, !nest.event<"att_out_1">)
    %att_grid_2, %att_inrel_2, %att_out_2 = nest.dispatch.tasks.async @prefill_attention_q2
      l1_mode = 0 tasks(%200) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q2)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_q_out_7, %qkv_kv_out_7, %qkv_q_grid_5)
      : (!nest.event<"att_grid_2">, !nest.event<"att_inrel_2">, !nest.event<"att_out_2">)
    %att_grid_3, %att_inrel_3, %att_out_3 = nest.dispatch.tasks.async @prefill_attention_q3
      l1_mode = 0 tasks(%200) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q3)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_q_out_7, %qkv_kv_out_7, %qkv_kv_grid_5)
      : (!nest.event<"att_grid_3">, !nest.event<"att_inrel_3">, !nest.event<"att_out_3">)
    %pj_grid_0_lo, %pj_inrel_0_lo, %pj_out_0_lo = nest.dispatch.tasks.async @prefill_outproj_lo
      l1_mode = 0 tasks(%200) globals() bindings(%o_q0, %wo_l2_2, %out_q0_lo) ins(%o_q0, %wo_l2_2)
      outs(%out_q0_lo)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out, %pre_wo, %qkv_q_grid_6)
      : (!nest.event<"pj_grid_0_lo">, !nest.event<"pj_inrel_0_lo">, !nest.event<"pj_out_0_lo">)
    %pj_grid_0_hi, %pj_inrel_0_hi, %pj_out_0_hi = nest.dispatch.tasks.async @prefill_outproj_hi
      l1_mode = 0 tasks(%200) globals() bindings(%o_q0, %wo_l2_2, %out_q0_hi) ins(%o_q0, %wo_l2_2)
      outs(%out_q0_hi)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out, %pre_wo, %qkv_kv_grid_6)
      : (!nest.event<"pj_grid_0_hi">, !nest.event<"pj_inrel_0_hi">, !nest.event<"pj_out_0_hi">)
    %pj_grid_1_lo, %pj_inrel_1_lo, %pj_out_1_lo = nest.dispatch.tasks.async @prefill_outproj_lo
      l1_mode = 0 tasks(%200) globals() bindings(%o_q1, %wo_l2_2, %out_q1_lo) ins(%o_q1, %wo_l2_2)
      outs(%out_q1_lo)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_1, %pre_wo, %qkv_q_grid_7)
      : (!nest.event<"pj_grid_1_lo">, !nest.event<"pj_inrel_1_lo">, !nest.event<"pj_out_1_lo">)
    %pj_grid_1_hi, %pj_inrel_1_hi, %pj_out_1_hi = nest.dispatch.tasks.async @prefill_outproj_hi
      l1_mode = 0 tasks(%200) globals() bindings(%o_q1, %wo_l2_2, %out_q1_hi) ins(%o_q1, %wo_l2_2)
      outs(%out_q1_hi)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_1, %pre_wo, %qkv_kv_grid_7)
      : (!nest.event<"pj_grid_1_hi">, !nest.event<"pj_inrel_1_hi">, !nest.event<"pj_out_1_hi">)
    %pj_grid_2_lo, %pj_inrel_2_lo, %pj_out_2_lo = nest.dispatch.tasks.async @prefill_outproj_lo
      l1_mode = 0 tasks(%200) globals() bindings(%o_q2, %wo_l2_2, %out_q2_lo) ins(%o_q2, %wo_l2_2)
      outs(%out_q2_lo)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_2, %pre_wo, %att_grid)
      : (!nest.event<"pj_grid_2_lo">, !nest.event<"pj_inrel_2_lo">, !nest.event<"pj_out_2_lo">)
    %pj_grid_2_hi, %pj_inrel_2_hi, %pj_out_2_hi = nest.dispatch.tasks.async @prefill_outproj_hi
      l1_mode = 0 tasks(%200) globals() bindings(%o_q2, %wo_l2_2, %out_q2_hi) ins(%o_q2, %wo_l2_2)
      outs(%out_q2_hi)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_2, %pre_wo, %att_grid_1)
      : (!nest.event<"pj_grid_2_hi">, !nest.event<"pj_inrel_2_hi">, !nest.event<"pj_out_2_hi">)
    %pj_grid_3_lo, %pj_inrel_3_lo, %pj_out_3_lo = nest.dispatch.tasks.async @prefill_outproj_lo
      l1_mode = 0 tasks(%200) globals() bindings(%o_q3, %wo_l2_2, %out_q3_lo) ins(%o_q3, %wo_l2_2)
      outs(%out_q3_lo)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_3, %pre_wo, %att_grid_2)
      : (!nest.event<"pj_grid_3_lo">, !nest.event<"pj_inrel_3_lo">, !nest.event<"pj_out_3_lo">)
    %pj_grid_3_hi, %pj_inrel_3_hi, %pj_out_3_hi = nest.dispatch.tasks.async @prefill_outproj_hi
      l1_mode = 0 tasks(%200) globals() bindings(%o_q3, %wo_l2_2, %out_q3_hi) ins(%o_q3, %wo_l2_2)
      outs(%out_q3_hi)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_3, %pre_wo, %att_grid_3)
      : (!nest.event<"pj_grid_3_hi">, !nest.event<"pj_inrel_3_hi">, !nest.event<"pj_out_3_hi">)
    %234 = nest.subview %OUT offsets = [0, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_0_lo = nest.dma.store.async %out_q0_lo into %234 depends_on(%pj_out_0_lo)
      : !nest.event<"out_store_0_lo">
    %235 = nest.subview %OUT offsets = [0, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_0_hi = nest.dma.store.async %out_q0_hi into %235 depends_on(%pj_out_0_hi)
      : !nest.event<"out_store_0_hi">
    %236 = nest.subview %OUT offsets = [1, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_1_lo = nest.dma.store.async %out_q1_lo into %236 depends_on(%pj_out_1_lo)
      : !nest.event<"out_store_1_lo">
    %237 = nest.subview %OUT offsets = [1, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_1_hi = nest.dma.store.async %out_q1_hi into %237 depends_on(%pj_out_1_hi)
      : !nest.event<"out_store_1_hi">
    %238 = nest.subview %OUT offsets = [2, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_2_lo = nest.dma.store.async %out_q2_lo into %238 depends_on(%pj_out_2_lo)
      : !nest.event<"out_store_2_lo">
    %239 = nest.subview %OUT offsets = [2, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_2_hi = nest.dma.store.async %out_q2_hi into %239 depends_on(%pj_out_2_hi)
      : !nest.event<"out_store_2_hi">
    %240 = nest.subview %OUT offsets = [3, 0, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_3_lo = nest.dma.store.async %out_q3_lo into %240 depends_on(%pj_out_3_lo)
      : !nest.event<"out_store_3_lo">
    %241 = nest.subview %OUT offsets = [3, 1, 0, 0, 0] sizes = [1, 1, 4, 64, 256]
      strides = [1, 1, 1, 1, 1] : !nest.global_view<1x1x4x64x256xbf16>
    %out_store_3_hi = nest.dma.store.async %out_q3_hi into %241 depends_on(%pj_out_3_hi)
      : !nest.event<"out_store_3_hi">
    nest.release %x_p0 depends_on(
      %pre_x, %pre_x_2, %pre_x_4, %pre_x_6, %qkv_q_inrel, %qkv_q_inrel_2, %qkv_q_inrel_4,
      %qkv_q_inrel_6, %qkv_kv_inrel, %qkv_kv_inrel_2, %qkv_kv_inrel_4, %qkv_kv_inrel_6)
    nest.release %wq_p0 depends_on(
      %pre_wq, %pre_wq_2, %pre_wq_4, %pre_wq_6, %qkv_q_inrel, %qkv_q_inrel_2, %qkv_q_inrel_4,
      %qkv_q_inrel_6)
    nest.release %wk_p0 depends_on(
      %pre_wk, %pre_wk_2, %pre_wk_4, %pre_wk_6, %qkv_kv_inrel, %qkv_kv_inrel_2, %qkv_kv_inrel_4,
      %qkv_kv_inrel_6)
    nest.release %wv_p0 depends_on(
      %pre_wv, %pre_wv_2, %pre_wv_4, %pre_wv_6, %qkv_kv_inrel, %qkv_kv_inrel_2, %qkv_kv_inrel_4,
      %qkv_kv_inrel_6)
    nest.release %x_p1 depends_on(
      %pre_x_1, %pre_x_3, %pre_x_5, %pre_x_7, %qkv_q_inrel_1, %qkv_q_inrel_3, %qkv_q_inrel_5,
      %qkv_q_inrel_7, %qkv_kv_inrel_1, %qkv_kv_inrel_3, %qkv_kv_inrel_5, %qkv_kv_inrel_7)
    nest.release %wq_p1 depends_on(
      %pre_wq_1, %pre_wq_3, %pre_wq_5, %pre_wq_7, %qkv_q_inrel_1, %qkv_q_inrel_3, %qkv_q_inrel_5,
      %qkv_q_inrel_7)
    nest.release %wk_p1 depends_on(
      %pre_wk_1, %pre_wk_3, %pre_wk_5, %pre_wk_7, %qkv_kv_inrel_1, %qkv_kv_inrel_3,
      %qkv_kv_inrel_5, %qkv_kv_inrel_7)
    nest.release %wv_p1 depends_on(
      %pre_wv_1, %pre_wv_3, %pre_wv_5, %pre_wv_7, %qkv_kv_inrel_1, %qkv_kv_inrel_3,
      %qkv_kv_inrel_5, %qkv_kv_inrel_7)
    nest.release %q_l2_6 depends_on(
      %qkv_q_inrel_1, %qkv_q_inrel_2, %qkv_q_inrel_3, %qkv_q_inrel_4, %qkv_q_inrel_5,
      %qkv_q_inrel_6, %qkv_q_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3,
      %qkv_q_out, %qkv_q_out_1, %qkv_q_out_2, %qkv_q_out_3, %qkv_q_out_4, %qkv_q_out_5,
      %qkv_q_out_6, %qkv_q_out_7)
    nest.release %k_l2_6 depends_on(
      %qkv_kv_inrel_1, %qkv_kv_inrel_2, %qkv_kv_inrel_3, %qkv_kv_inrel_4, %qkv_kv_inrel_5,
      %qkv_kv_inrel_6, %qkv_kv_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3,
      %qkv_kv_out, %qkv_kv_out_1, %qkv_kv_out_2, %qkv_kv_out_3, %qkv_kv_out_4, %qkv_kv_out_5,
      %qkv_kv_out_6, %qkv_kv_out_7)
    nest.release %v_l2_6 depends_on(
      %qkv_kv_inrel_1, %qkv_kv_inrel_2, %qkv_kv_inrel_3, %qkv_kv_inrel_4, %qkv_kv_inrel_5,
      %qkv_kv_inrel_6, %qkv_kv_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3,
      %qkv_kv_out, %qkv_kv_out_1, %qkv_kv_out_2, %qkv_kv_out_3, %qkv_kv_out_4, %qkv_kv_out_5,
      %qkv_kv_out_6, %qkv_kv_out_7)
    nest.release %wo_l2_2 depends_on(
      %pre_wo, %pj_inrel_0_lo, %pj_inrel_0_hi, %pj_inrel_1_lo, %pj_inrel_1_hi, %pj_inrel_2_lo,
      %pj_inrel_2_hi, %pj_inrel_3_lo, %pj_inrel_3_hi)
    nest.release %o_q0 depends_on(%att_out, %pj_inrel_0_lo, %pj_inrel_0_hi)
    nest.release %o_q1 depends_on(%att_out_1, %pj_inrel_1_lo, %pj_inrel_1_hi)
    nest.release %o_q2 depends_on(%att_out_2, %pj_inrel_2_lo, %pj_inrel_2_hi)
    nest.release %o_q3 depends_on(%att_out_3, %pj_inrel_3_lo, %pj_inrel_3_hi)
    nest.release %out_q0_lo depends_on(%out_store_0_lo)
    nest.release %out_q0_hi depends_on(%out_store_0_hi)
    nest.release %out_q1_lo depends_on(%out_store_1_lo)
    nest.release %out_q1_hi depends_on(%out_store_1_hi)
    nest.release %out_q2_lo depends_on(%out_store_2_lo)
    nest.release %out_q2_hi depends_on(%out_store_2_hi)
    nest.release %out_q3_lo depends_on(%out_store_3_lo)
    nest.release %out_q3_hi depends_on(%out_store_3_hi)
    nest.await %qkv_q_grid, %qkv_kv_grid, %qkv_q_grid_1, %qkv_kv_grid_1, %qkv_q_grid_2,
      %qkv_kv_grid_2, %qkv_q_grid_3, %qkv_kv_grid_3, %qkv_q_grid_4, %qkv_kv_grid_4, %qkv_q_grid_5,
      %qkv_kv_grid_5, %qkv_q_grid_6, %qkv_kv_grid_6, %qkv_q_grid_7, %qkv_kv_grid_7, %att_grid,
      %att_grid_1, %att_grid_2, %att_grid_3, %pj_grid_0_lo, %pj_grid_0_hi, %pj_grid_1_lo,
      %pj_grid_1_hi, %pj_grid_2_lo, %pj_grid_2_hi, %pj_grid_3_lo, %pj_grid_3_hi, %out_store_0_lo,
      %out_store_0_hi, %out_store_1_lo, %out_store_1_hi, %out_store_2_lo, %out_store_2_hi,
      %out_store_3_lo, %out_store_3_hi
    nest.return
  }
  nexus.program @transformer_prefill_dispatchparallel(
    %X_1: !nest.global_memref<8x512x128xbf16>, %WQ_1: !nest.global_memref<8x4x128x256xbf16>,
    %WK_1: !nest.global_memref<8x4x128x64xbf16>, %WV_1: !nest.global_memref<8x4x128x64xbf16>,
    %WO_1: !nest.global_memref<4x1024x256xbf16>, %OUT_1: !nest.global_memref<4x2x4x64x256xbf16>) {
    %prefill_dispatchparallel_done =
      nexus.submit_context.async @prefill_dispatchparallel_ctx(
        %X_1, %WQ_1, %WK_1, %WV_1, %WO_1, %OUT_1)
      : !nexus.event<"prefill_dispatchparallel_done">
    nexus.await %prefill_dispatchparallel_done
    nexus.return
  }
}
