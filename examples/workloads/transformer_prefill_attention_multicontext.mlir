// Transformer Prefill Attention, query-row multicontext schedule (one device root, BF16).
//
// Logical shapes: seq=512, hidden=1024, q_heads=16, kv_heads=4, head_dim=64;
// tile t owns KV head t and Q heads 4t..4t+3 (placement 15).
// Tiles: query block=128, internal/query and QKV row tile=64, KV block=128, QKV input-K=128.
// Four independent query blocks partition rows 0..511; each attends all four KV blocks.
// QKV is projected once across all rows; QKV weights are fetched once per K chunk.
// Each query block has distinct writable L2 O/OUT allocations, avoiding
// whole-buffer write hazards between concurrently submitted blocks.
// R=4 unpinned UCE Tasks per tile; every dispatch uses task range [0,4).
// The R x child per-bank L1 envelope is proved for each advertised L1 mode.
// Default root Arena is 6.5 MiB (L2 modes 0/1); all concurrent roots share the Group L2 pool.
// At R=4, attention child=240 KiB (15 KiB/bank): 60 KiB/bank fits only mode 0's 62 KiB.
// QKV/outproj children are 160 KiB each; all L1 allowed-mode sets are recomputed for R.
//
// Packing and traffic: HBM inputs retain baseline extents (X [8,512,128],
// WQ [8,4,128,256], WK/WV [8,4,128,64], WO [4,1024,256]).
// OUT [4,4,128,256] is query-block-major physical packing of logical OUT [4,512,256]:
// OUT[qblock,head,row,:] maps to logical OUT[head,qblock*128+row,:]. O/OUT are
// split into four 128-row L2 allocations with the same aggregate bytes as baseline.
// Q/K/V and input ping-pong retain baseline bytes; WO is still prefetched from HBM once.
// Each of four query-block output Tasks loads its tile's 512 KiB WO shard from shared L2,
// adding exactly 6 MiB aggregate L2-to-L1 read payload vs baseline (three extra copies).
// All other interface payloads, HBM bytes and logical BOA FLOPs match baseline.
//
// Compute: QKV = 512x1024x1536 useful MACs; attention covers 16 Q heads x 512 queries
// x all 512 keys x 64 features for both QK and PV; output projection = 512x1024x1024
// Online-softmax L1 state is m/l [4,64]xf32 per 64-row microtile plus one reusable
// [64,128]xf32 score tile. Q/K stores overlap the independent following QKV BOAs;
// independent final row stores launch together, then share a completion frontier.
// Every buffer reuse is gated by the preceding consumers/stores; Arena sizes are unchanged.
//
// TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor operands and
// execute no numerics. This source validates scheduling, resources and traffic, not
// numerical correctness or hardware performance guarantees.
// Generate: PYTHONPATH=. python examples/generators/generate_transformer_prefill_multicontext.py
//   --contexts-per-tile 4
// Run: bash examples/run.sh transformer-prefill-attention-multicontext

builtin.module {
  tile.program @qkv_chunk_init(
    %task: !nest.task, %x_chunk: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk: !nest.l2_buffer<4x128x256xbf16>, %wk_chunk: !nest.l2_buffer<4x128x64xbf16>,
    %wv_chunk: !nest.l2_buffer<4x128x64xbf16>, %q_l2: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2: !nest.l2_buffer<4x512x64xbf16>, %v_l2: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 163840> {
    %x_buf = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %q_acc = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %wq_buf = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %k_acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wk_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wv_buf = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %0 = tile.subview %wq_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %1 = tile.subview %wk_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %2 = tile.subview %wv_chunk task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %wq_loaded = tile.load.async %0 into %wq_buf : !tile.event<"wq_loaded">
    %wk_loaded = tile.load.async %1 into %wk_buf : !tile.event<"wk_loaded">
    %wv_loaded = tile.load.async %2 into %wv_buf : !tile.event<"wv_loaded">
    tile.await %wq_loaded, %wk_loaded, %wv_loaded
    %3 = tile.subview %x_chunk offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %4 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %5 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %6 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded = tile.load.async %3 into %x_buf : !tile.event<"x_loaded_0">
    tile.await %x_loaded
    %q_boa = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_0">
    tile.await %q_boa
    %q_stored = tile.store.async %q_acc into %4 : !tile.event<"q_stored_0">
    %k_boa = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_0">
    tile.await %k_boa
    %k_stored = tile.store.async %k_acc into %5 : !tile.event<"k_stored_0">
    %v_boa = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_0">
    tile.await %v_boa
    %v_stored = tile.store.async %v_acc into %6 : !tile.event<"v_stored_0">
    tile.await %q_stored, %k_stored, %v_stored
    %7 = tile.subview %x_chunk offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %8 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %9 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %10 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_1 = tile.load.async %7 into %x_buf : !tile.event<"x_loaded_1">
    tile.await %x_loaded_1
    %q_boa_1 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_1">
    tile.await %q_boa_1
    %q_stored_1 = tile.store.async %q_acc into %8 : !tile.event<"q_stored_1">
    %k_boa_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_1">
    tile.await %k_boa_1
    %k_stored_1 = tile.store.async %k_acc into %9 : !tile.event<"k_stored_1">
    %v_boa_1 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_1">
    tile.await %v_boa_1
    %v_stored_1 = tile.store.async %v_acc into %10 : !tile.event<"v_stored_1">
    tile.await %q_stored_1, %k_stored_1, %v_stored_1
    %11 = tile.subview %x_chunk offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %12 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %13 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %14 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_2 = tile.load.async %11 into %x_buf : !tile.event<"x_loaded_2">
    tile.await %x_loaded_2
    %q_boa_2 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_2">
    tile.await %q_boa_2
    %q_stored_2 = tile.store.async %q_acc into %12 : !tile.event<"q_stored_2">
    %k_boa_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_2">
    tile.await %k_boa_2
    %k_stored_2 = tile.store.async %k_acc into %13 : !tile.event<"k_stored_2">
    %v_boa_2 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_2">
    tile.await %v_boa_2
    %v_stored_2 = tile.store.async %v_acc into %14 : !tile.event<"v_stored_2">
    tile.await %q_stored_2, %k_stored_2, %v_stored_2
    %15 = tile.subview %x_chunk offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %16 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %17 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %18 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_3 = tile.load.async %15 into %x_buf : !tile.event<"x_loaded_3">
    tile.await %x_loaded_3
    %q_boa_3 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_3">
    tile.await %q_boa_3
    %q_stored_3 = tile.store.async %q_acc into %16 : !tile.event<"q_stored_3">
    %k_boa_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_3">
    tile.await %k_boa_3
    %k_stored_3 = tile.store.async %k_acc into %17 : !tile.event<"k_stored_3">
    %v_boa_3 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_3">
    tile.await %v_boa_3
    %v_stored_3 = tile.store.async %v_acc into %18 : !tile.event<"v_stored_3">
    tile.await %q_stored_3, %k_stored_3, %v_stored_3
    %19 = tile.subview %x_chunk offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %20 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %21 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %22 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_4 = tile.load.async %19 into %x_buf : !tile.event<"x_loaded_4">
    tile.await %x_loaded_4
    %q_boa_4 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_4">
    tile.await %q_boa_4
    %q_stored_4 = tile.store.async %q_acc into %20 : !tile.event<"q_stored_4">
    %k_boa_4 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_4">
    tile.await %k_boa_4
    %k_stored_4 = tile.store.async %k_acc into %21 : !tile.event<"k_stored_4">
    %v_boa_4 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_4">
    tile.await %v_boa_4
    %v_stored_4 = tile.store.async %v_acc into %22 : !tile.event<"v_stored_4">
    tile.await %q_stored_4, %k_stored_4, %v_stored_4
    %23 = tile.subview %x_chunk offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %24 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %25 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %26 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_5 = tile.load.async %23 into %x_buf : !tile.event<"x_loaded_5">
    tile.await %x_loaded_5
    %q_boa_5 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_5">
    tile.await %q_boa_5
    %q_stored_5 = tile.store.async %q_acc into %24 : !tile.event<"q_stored_5">
    %k_boa_5 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_5">
    tile.await %k_boa_5
    %k_stored_5 = tile.store.async %k_acc into %25 : !tile.event<"k_stored_5">
    %v_boa_5 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_5">
    tile.await %v_boa_5
    %v_stored_5 = tile.store.async %v_acc into %26 : !tile.event<"v_stored_5">
    tile.await %q_stored_5, %k_stored_5, %v_stored_5
    %27 = tile.subview %x_chunk offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %28 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %29 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %30 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_6 = tile.load.async %27 into %x_buf : !tile.event<"x_loaded_6">
    tile.await %x_loaded_6
    %q_boa_6 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_6">
    tile.await %q_boa_6
    %q_stored_6 = tile.store.async %q_acc into %28 : !tile.event<"q_stored_6">
    %k_boa_6 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_6">
    tile.await %k_boa_6
    %k_stored_6 = tile.store.async %k_acc into %29 : !tile.event<"k_stored_6">
    %v_boa_6 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_6">
    tile.await %v_boa_6
    %v_stored_6 = tile.store.async %v_acc into %30 : !tile.event<"v_stored_6">
    tile.await %q_stored_6, %k_stored_6, %v_stored_6
    %31 = tile.subview %x_chunk offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %32 = tile.subview %q_l2 task = %task task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %33 = tile.subview %k_l2 task = %task task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %34 = tile.subview %v_l2 task = %task task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_7 = tile.load.async %31 into %x_buf : !tile.event<"x_loaded_7">
    tile.await %x_loaded_7
    tile.signal input_released(%task)
    %q_boa_7 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 : !tile.event<"q_boa_7">
    tile.await %q_boa_7
    %q_stored_7 = tile.store.async %q_acc into %32 : !tile.event<"q_stored_7">
    %k_boa_7 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"k_boa_7">
    tile.await %k_boa_7
    %k_stored_7 = tile.store.async %k_acc into %33 : !tile.event<"k_stored_7">
    %v_boa_7 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 : !tile.event<"v_boa_7">
    tile.await %v_boa_7
    %v_stored_7 = tile.store.async %v_acc into %34 : !tile.event<"v_stored_7">
    tile.await %q_stored_7, %k_stored_7, %v_stored_7
    tile.signal output_ready(%task)
    tile.free %x_buf
    tile.free %q_acc
    tile.free %wq_buf
    tile.free %k_acc
    tile.free %wk_buf
    tile.free %v_acc
    tile.free %wv_buf
    tile.return
  }
  tile.program @qkv_chunk_accum(
    %task_1: !nest.task, %x_chunk_1: !nest.l2_buffer<512x128xbf16>,
    %wq_chunk_1: !nest.l2_buffer<4x128x256xbf16>, %wk_chunk_1: !nest.l2_buffer<4x128x64xbf16>,
    %wv_chunk_1: !nest.l2_buffer<4x128x64xbf16>, %q_l2_1: !nest.l2_buffer<4x512x256xbf16>,
    %k_l2_1: !nest.l2_buffer<4x512x64xbf16>, %v_l2_1: !nest.l2_buffer<4x512x64xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 163840> {
    %x_buf_1 = tile.alloc shape = [64, 128] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x128xbf16>
    %q_acc_1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %wq_buf_1 = tile.alloc shape = [128, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x256xbf16>
    %k_acc_1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wk_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %v_acc_1 = tile.alloc shape = [64, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x64xbf16>
    %wv_buf_1 = tile.alloc shape = [128, 64] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<128x64xbf16>
    %35 = tile.subview %wq_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 256] strides = [1, 1, 1] : !nest.l2_view<1x128x256xbf16>
    %36 = tile.subview %wk_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %37 = tile.subview %wv_chunk_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %wq_loaded_1 = tile.load.async %35 into %wq_buf_1 : !tile.event<"wq_loaded">
    %wk_loaded_1 = tile.load.async %36 into %wk_buf_1 : !tile.event<"wk_loaded">
    %wv_loaded_1 = tile.load.async %37 into %wv_buf_1 : !tile.event<"wv_loaded">
    tile.await %wq_loaded_1, %wk_loaded_1, %wv_loaded_1
    %38 = tile.subview %x_chunk_1 offsets = [0, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %39 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %40 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %41 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_8 = tile.load.async %38 into %x_buf_1 : !tile.event<"x_loaded_0">
    %q_partial_loaded = tile.load.async %39 into %q_acc_1 : !tile.event<"q_partial_loaded_0">
    %k_partial_loaded = tile.load.async %40 into %k_acc_1 : !tile.event<"k_partial_loaded_0">
    %v_partial_loaded = tile.load.async %41 into %v_acc_1 : !tile.event<"v_partial_loaded_0">
    tile.await %x_loaded_8, %q_partial_loaded, %k_partial_loaded, %v_partial_loaded
    %q_boa_8 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_0">
    tile.await %q_boa_8
    %q_stored_8 = tile.store.async %q_acc_1 into %39 : !tile.event<"q_stored_0">
    %k_boa_8 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_0">
    tile.await %k_boa_8
    %k_stored_8 = tile.store.async %k_acc_1 into %40 : !tile.event<"k_stored_0">
    %v_boa_8 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_0">
    tile.await %v_boa_8
    %v_stored_8 = tile.store.async %v_acc_1 into %41 : !tile.event<"v_stored_0">
    tile.await %q_stored_8, %k_stored_8, %v_stored_8
    %42 = tile.subview %x_chunk_1 offsets = [64, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %43 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %44 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %45 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_9 = tile.load.async %42 into %x_buf_1 : !tile.event<"x_loaded_1">
    %q_partial_loaded_1 = tile.load.async %43 into %q_acc_1 : !tile.event<"q_partial_loaded_1">
    %k_partial_loaded_1 = tile.load.async %44 into %k_acc_1 : !tile.event<"k_partial_loaded_1">
    %v_partial_loaded_1 = tile.load.async %45 into %v_acc_1 : !tile.event<"v_partial_loaded_1">
    tile.await %x_loaded_9, %q_partial_loaded_1, %k_partial_loaded_1, %v_partial_loaded_1
    %q_boa_9 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_1">
    tile.await %q_boa_9
    %q_stored_9 = tile.store.async %q_acc_1 into %43 : !tile.event<"q_stored_1">
    %k_boa_9 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_1">
    tile.await %k_boa_9
    %k_stored_9 = tile.store.async %k_acc_1 into %44 : !tile.event<"k_stored_1">
    %v_boa_9 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_1">
    tile.await %v_boa_9
    %v_stored_9 = tile.store.async %v_acc_1 into %45 : !tile.event<"v_stored_1">
    tile.await %q_stored_9, %k_stored_9, %v_stored_9
    %46 = tile.subview %x_chunk_1 offsets = [128, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %47 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %48 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %49 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 128, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_10 = tile.load.async %46 into %x_buf_1 : !tile.event<"x_loaded_2">
    %q_partial_loaded_2 = tile.load.async %47 into %q_acc_1 : !tile.event<"q_partial_loaded_2">
    %k_partial_loaded_2 = tile.load.async %48 into %k_acc_1 : !tile.event<"k_partial_loaded_2">
    %v_partial_loaded_2 = tile.load.async %49 into %v_acc_1 : !tile.event<"v_partial_loaded_2">
    tile.await %x_loaded_10, %q_partial_loaded_2, %k_partial_loaded_2, %v_partial_loaded_2
    %q_boa_10 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_2">
    tile.await %q_boa_10
    %q_stored_10 = tile.store.async %q_acc_1 into %47 : !tile.event<"q_stored_2">
    %k_boa_10 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_2">
    tile.await %k_boa_10
    %k_stored_10 = tile.store.async %k_acc_1 into %48 : !tile.event<"k_stored_2">
    %v_boa_10 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_2">
    tile.await %v_boa_10
    %v_stored_10 = tile.store.async %v_acc_1 into %49 : !tile.event<"v_stored_2">
    tile.await %q_stored_10, %k_stored_10, %v_stored_10
    %50 = tile.subview %x_chunk_1 offsets = [192, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %51 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %52 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %53 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 192, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_11 = tile.load.async %50 into %x_buf_1 : !tile.event<"x_loaded_3">
    %q_partial_loaded_3 = tile.load.async %51 into %q_acc_1 : !tile.event<"q_partial_loaded_3">
    %k_partial_loaded_3 = tile.load.async %52 into %k_acc_1 : !tile.event<"k_partial_loaded_3">
    %v_partial_loaded_3 = tile.load.async %53 into %v_acc_1 : !tile.event<"v_partial_loaded_3">
    tile.await %x_loaded_11, %q_partial_loaded_3, %k_partial_loaded_3, %v_partial_loaded_3
    %q_boa_11 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_3">
    tile.await %q_boa_11
    %q_stored_11 = tile.store.async %q_acc_1 into %51 : !tile.event<"q_stored_3">
    %k_boa_11 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_3">
    tile.await %k_boa_11
    %k_stored_11 = tile.store.async %k_acc_1 into %52 : !tile.event<"k_stored_3">
    %v_boa_11 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_3">
    tile.await %v_boa_11
    %v_stored_11 = tile.store.async %v_acc_1 into %53 : !tile.event<"v_stored_3">
    tile.await %q_stored_11, %k_stored_11, %v_stored_11
    %54 = tile.subview %x_chunk_1 offsets = [256, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %55 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %56 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %57 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 256, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_12 = tile.load.async %54 into %x_buf_1 : !tile.event<"x_loaded_4">
    %q_partial_loaded_4 = tile.load.async %55 into %q_acc_1 : !tile.event<"q_partial_loaded_4">
    %k_partial_loaded_4 = tile.load.async %56 into %k_acc_1 : !tile.event<"k_partial_loaded_4">
    %v_partial_loaded_4 = tile.load.async %57 into %v_acc_1 : !tile.event<"v_partial_loaded_4">
    tile.await %x_loaded_12, %q_partial_loaded_4, %k_partial_loaded_4, %v_partial_loaded_4
    %q_boa_12 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_4">
    tile.await %q_boa_12
    %q_stored_12 = tile.store.async %q_acc_1 into %55 : !tile.event<"q_stored_4">
    %k_boa_12 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_4">
    tile.await %k_boa_12
    %k_stored_12 = tile.store.async %k_acc_1 into %56 : !tile.event<"k_stored_4">
    %v_boa_12 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_4">
    tile.await %v_boa_12
    %v_stored_12 = tile.store.async %v_acc_1 into %57 : !tile.event<"v_stored_4">
    tile.await %q_stored_12, %k_stored_12, %v_stored_12
    %58 = tile.subview %x_chunk_1 offsets = [320, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %59 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %60 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %61 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 320, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_13 = tile.load.async %58 into %x_buf_1 : !tile.event<"x_loaded_5">
    %q_partial_loaded_5 = tile.load.async %59 into %q_acc_1 : !tile.event<"q_partial_loaded_5">
    %k_partial_loaded_5 = tile.load.async %60 into %k_acc_1 : !tile.event<"k_partial_loaded_5">
    %v_partial_loaded_5 = tile.load.async %61 into %v_acc_1 : !tile.event<"v_partial_loaded_5">
    tile.await %x_loaded_13, %q_partial_loaded_5, %k_partial_loaded_5, %v_partial_loaded_5
    %q_boa_13 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_5">
    tile.await %q_boa_13
    %q_stored_13 = tile.store.async %q_acc_1 into %59 : !tile.event<"q_stored_5">
    %k_boa_13 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_5">
    tile.await %k_boa_13
    %k_stored_13 = tile.store.async %k_acc_1 into %60 : !tile.event<"k_stored_5">
    %v_boa_13 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_5">
    tile.await %v_boa_13
    %v_stored_13 = tile.store.async %v_acc_1 into %61 : !tile.event<"v_stored_5">
    tile.await %q_stored_13, %k_stored_13, %v_stored_13
    %62 = tile.subview %x_chunk_1 offsets = [384, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %63 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %64 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %65 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 384, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_14 = tile.load.async %62 into %x_buf_1 : !tile.event<"x_loaded_6">
    %q_partial_loaded_6 = tile.load.async %63 into %q_acc_1 : !tile.event<"q_partial_loaded_6">
    %k_partial_loaded_6 = tile.load.async %64 into %k_acc_1 : !tile.event<"k_partial_loaded_6">
    %v_partial_loaded_6 = tile.load.async %65 into %v_acc_1 : !tile.event<"v_partial_loaded_6">
    tile.await %x_loaded_14, %q_partial_loaded_6, %k_partial_loaded_6, %v_partial_loaded_6
    %q_boa_14 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_6">
    tile.await %q_boa_14
    %q_stored_14 = tile.store.async %q_acc_1 into %63 : !tile.event<"q_stored_6">
    %k_boa_14 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_6">
    tile.await %k_boa_14
    %k_stored_14 = tile.store.async %k_acc_1 into %64 : !tile.event<"k_stored_6">
    %v_boa_14 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_6">
    tile.await %v_boa_14
    %v_stored_14 = tile.store.async %v_acc_1 into %65 : !tile.event<"v_stored_6">
    tile.await %q_stored_14, %k_stored_14, %v_stored_14
    %66 = tile.subview %x_chunk_1 offsets = [448, 0] sizes = [64, 128] strides = [1, 1]
      : !nest.l2_view<64x128xbf16>
    %67 = tile.subview %q_l2_1 task = %task_1 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %68 = tile.subview %k_l2_1 task = %task_1 task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %69 = tile.subview %v_l2_1 task = %task_1 task_dim = 0 offsets = [0, 448, 0] sizes = [1, 64, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x64x64xbf16>
    %x_loaded_15 = tile.load.async %66 into %x_buf_1 : !tile.event<"x_loaded_7">
    %q_partial_loaded_7 = tile.load.async %67 into %q_acc_1 : !tile.event<"q_partial_loaded_7">
    %k_partial_loaded_7 = tile.load.async %68 into %k_acc_1 : !tile.event<"k_partial_loaded_7">
    %v_partial_loaded_7 = tile.load.async %69 into %v_acc_1 : !tile.event<"v_partial_loaded_7">
    tile.await %x_loaded_15, %q_partial_loaded_7, %k_partial_loaded_7, %v_partial_loaded_7
    tile.signal input_released(%task_1)
    %q_boa_15 = tile.boa.async "matmul" m = 64 n = 256 k = 128 ops = 4194304 accumulate
      : !tile.event<"q_boa_7">
    tile.await %q_boa_15
    %q_stored_15 = tile.store.async %q_acc_1 into %67 : !tile.event<"q_stored_7">
    %k_boa_15 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"k_boa_7">
    tile.await %k_boa_15
    %k_stored_15 = tile.store.async %k_acc_1 into %68 : !tile.event<"k_stored_7">
    %v_boa_15 = tile.boa.async "matmul" m = 64 n = 64 k = 128 ops = 1048576 accumulate
      : !tile.event<"v_boa_7">
    tile.await %v_boa_15
    %v_stored_15 = tile.store.async %v_acc_1 into %69 : !tile.event<"v_stored_7">
    tile.await %q_stored_15, %k_stored_15, %v_stored_15
    tile.signal output_ready(%task_1)
    tile.free %x_buf_1
    tile.free %q_acc_1
    tile.free %wq_buf_1
    tile.free %k_acc_1
    tile.free %wk_buf_1
    tile.free %v_acc_1
    tile.free %wv_buf_1
    tile.return
  }
  tile.program @prefill_attention_q0(
    %task_2: !nest.task, %q_l2_2: !nest.l2_buffer<4x512x256xbf16>,
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
    %70 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded = tile.load.async %70 into %q0 : !tile.event<"q0_loaded">
    %71 = tile.subview %q_l2_2 task = %task_2 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded = tile.load.async %71 into %q1 : !tile.event<"q1_loaded">
    %72 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %73 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded = tile.load.async %72 into %k0 : !tile.event<"k0_loaded">
    %v0_loaded = tile.load.async %73 into %v0 : !tile.event<"v0_loaded">
    %74 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %75 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded = tile.load.async %74 into %k1 : !tile.event<"k1_loaded">
    %v1_loaded = tile.load.async %75 into %v1 : !tile.event<"v1_loaded">
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
    %76 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %77 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded = tile.load.async %76 into %k0 : !tile.event<"k2_loaded">
    %v2_loaded = tile.load.async %77 into %v0 : !tile.event<"v2_loaded">
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
    %78 = tile.subview %k_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %79 = tile.subview %v_l2_2 task = %task_2 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded = tile.load.async %78 into %k1 : !tile.event<"k3_loaded">
    %v3_loaded = tile.load.async %79 into %v1 : !tile.event<"v3_loaded">
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
    tile.signal input_released(%task_2)
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
    %80 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored = tile.store.async %acc0 into %80 : !tile.event<"o0_stored">
    %81 = tile.subview %o_l2 task = %task_2 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored = tile.store.async %acc1 into %81 : !tile.event<"o1_stored">
    tile.await %o0_stored, %o1_stored
    tile.signal output_ready(%task_2)
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
    %task_3: !nest.task, %q_l2_3: !nest.l2_buffer<4x512x256xbf16>,
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
    %82 = tile.subview %q_l2_3 task = %task_3 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_1 = tile.load.async %82 into %q0_1 : !tile.event<"q0_loaded">
    %83 = tile.subview %q_l2_3 task = %task_3 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_1 = tile.load.async %83 into %q1_1 : !tile.event<"q1_loaded">
    %84 = tile.subview %k_l2_3 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %85 = tile.subview %v_l2_3 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_1 = tile.load.async %84 into %k0_1 : !tile.event<"k0_loaded">
    %v0_loaded_1 = tile.load.async %85 into %v0_1 : !tile.event<"v0_loaded">
    %86 = tile.subview %k_l2_3 task = %task_3 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %87 = tile.subview %v_l2_3 task = %task_3 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_1 = tile.load.async %86 into %k1_1 : !tile.event<"k1_loaded">
    %v1_loaded_1 = tile.load.async %87 into %v1_1 : !tile.event<"v1_loaded">
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
    %88 = tile.subview %k_l2_3 task = %task_3 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %89 = tile.subview %v_l2_3 task = %task_3 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_1 = tile.load.async %88 into %k0_1 : !tile.event<"k2_loaded">
    %v2_loaded_1 = tile.load.async %89 into %v0_1 : !tile.event<"v2_loaded">
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
    %90 = tile.subview %k_l2_3 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %91 = tile.subview %v_l2_3 task = %task_3 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_1 = tile.load.async %90 into %k1_1 : !tile.event<"k3_loaded">
    %v3_loaded_1 = tile.load.async %91 into %v1_1 : !tile.event<"v3_loaded">
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
    tile.signal input_released(%task_3)
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
    %92 = tile.subview %o_l2_1 task = %task_3 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_1 = tile.store.async %acc0_1 into %92 : !tile.event<"o0_stored">
    %93 = tile.subview %o_l2_1 task = %task_3 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_1 = tile.store.async %acc1_1 into %93 : !tile.event<"o1_stored">
    tile.await %o0_stored_1, %o1_stored_1
    tile.signal output_ready(%task_3)
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
    %task_4: !nest.task, %q_l2_4: !nest.l2_buffer<4x512x256xbf16>,
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
    %94 = tile.subview %q_l2_4 task = %task_4 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_2 = tile.load.async %94 into %q0_2 : !tile.event<"q0_loaded">
    %95 = tile.subview %q_l2_4 task = %task_4 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_2 = tile.load.async %95 into %q1_2 : !tile.event<"q1_loaded">
    %96 = tile.subview %k_l2_4 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %97 = tile.subview %v_l2_4 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_2 = tile.load.async %96 into %k0_2 : !tile.event<"k0_loaded">
    %v0_loaded_2 = tile.load.async %97 into %v0_2 : !tile.event<"v0_loaded">
    %98 = tile.subview %k_l2_4 task = %task_4 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %99 = tile.subview %v_l2_4 task = %task_4 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_2 = tile.load.async %98 into %k1_2 : !tile.event<"k1_loaded">
    %v1_loaded_2 = tile.load.async %99 into %v1_2 : !tile.event<"v1_loaded">
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
    %100 = tile.subview %k_l2_4 task = %task_4 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %101 = tile.subview %v_l2_4 task = %task_4 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_2 = tile.load.async %100 into %k0_2 : !tile.event<"k2_loaded">
    %v2_loaded_2 = tile.load.async %101 into %v0_2 : !tile.event<"v2_loaded">
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
    %102 = tile.subview %k_l2_4 task = %task_4 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %103 = tile.subview %v_l2_4 task = %task_4 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_2 = tile.load.async %102 into %k1_2 : !tile.event<"k3_loaded">
    %v3_loaded_2 = tile.load.async %103 into %v1_2 : !tile.event<"v3_loaded">
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
    tile.signal input_released(%task_4)
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
    %104 = tile.subview %o_l2_2 task = %task_4 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_2 = tile.store.async %acc0_2 into %104 : !tile.event<"o0_stored">
    %105 = tile.subview %o_l2_2 task = %task_4 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_2 = tile.store.async %acc1_2 into %105 : !tile.event<"o1_stored">
    tile.await %o0_stored_2, %o1_stored_2
    tile.signal output_ready(%task_4)
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
    %task_5: !nest.task, %q_l2_5: !nest.l2_buffer<4x512x256xbf16>,
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
    %106 = tile.subview %q_l2_5 task = %task_5 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q0_loaded_3 = tile.load.async %106 into %q0_3 : !tile.event<"q0_loaded">
    %107 = tile.subview %q_l2_5 task = %task_5 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %q1_loaded_3 = tile.load.async %107 into %q1_3 : !tile.event<"q1_loaded">
    %108 = tile.subview %k_l2_5 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %109 = tile.subview %v_l2_5 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 128, 64]
      strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k0_loaded_3 = tile.load.async %108 into %k0_3 : !tile.event<"k0_loaded">
    %v0_loaded_3 = tile.load.async %109 into %v0_3 : !tile.event<"v0_loaded">
    %110 = tile.subview %k_l2_5 task = %task_5 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %111 = tile.subview %v_l2_5 task = %task_5 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k1_loaded_3 = tile.load.async %110 into %k1_3 : !tile.event<"k1_loaded">
    %v1_loaded_3 = tile.load.async %111 into %v1_3 : !tile.event<"v1_loaded">
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
    %112 = tile.subview %k_l2_5 task = %task_5 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %113 = tile.subview %v_l2_5 task = %task_5 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k2_loaded_3 = tile.load.async %112 into %k0_3 : !tile.event<"k2_loaded">
    %v2_loaded_3 = tile.load.async %113 into %v0_3 : !tile.event<"v2_loaded">
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
    %114 = tile.subview %k_l2_5 task = %task_5 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %115 = tile.subview %v_l2_5 task = %task_5 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 128, 64] strides = [1, 1, 1] : !nest.l2_view<1x128x64xbf16>
    %k3_loaded_3 = tile.load.async %114 into %k1_3 : !tile.event<"k3_loaded">
    %v3_loaded_3 = tile.load.async %115 into %v1_3 : !tile.event<"v3_loaded">
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
    tile.signal input_released(%task_5)
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
    %116 = tile.subview %o_l2_3 task = %task_5 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o0_stored_3 = tile.store.async %acc0_3 into %116 : !tile.event<"o0_stored">
    %117 = tile.subview %o_l2_3 task = %task_5 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %o1_stored_3 = tile.store.async %acc1_3 into %117 : !tile.event<"o1_stored">
    tile.await %o0_stored_3, %o1_stored_3
    tile.signal output_ready(%task_5)
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
  tile.program @prefill_outproj_tile(
    %task_6: !nest.task, %o_l2_4: !nest.l2_buffer<4x128x256xbf16>,
    %wo_l2: !nest.l2_buffer<4x1024x256xbf16>, %out_l2: !nest.l2_buffer<4x128x256xbf16>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 163840> {
    %acc0_4 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %acc1_4 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %o0 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %o1 = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %w_buf = tile.alloc shape = [64, 256] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<64x256xbf16>
    %118 = tile.subview %o_l2_4 offsets = [0, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o0_p0_loaded = tile.load.async %118 into %o0 : !tile.event<"o0_p0_loaded">
    %119 = tile.subview %o_l2_4 offsets = [0, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o0_p1_loaded = tile.load.async %119 into %o1 : !tile.event<"o0_p1_loaded">
    tile.await %o0_p0_loaded, %o0_p1_loaded
    %120 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_0_loaded = tile.load.async %120 into %w_buf : !tile.event<"w0_0_loaded">
    tile.await %w0_0_loaded
    %outproj0_0_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
      : !tile.event<"outproj0_0_p0">
    tile.await %outproj0_0_p0
    %outproj0_0_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
      : !tile.event<"outproj0_0_p1">
    tile.await %outproj0_0_p1
    %121 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 64, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_1_loaded = tile.load.async %121 into %w_buf : !tile.event<"w0_1_loaded">
    tile.await %w0_1_loaded
    %outproj0_1_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_1_p0">
    tile.await %outproj0_1_p0
    %outproj0_1_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_1_p1">
    tile.await %outproj0_1_p1
    %122 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 128, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_2_loaded = tile.load.async %122 into %w_buf : !tile.event<"w0_2_loaded">
    tile.await %w0_2_loaded
    %outproj0_2_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_2_p0">
    tile.await %outproj0_2_p0
    %outproj0_2_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_2_p1">
    tile.await %outproj0_2_p1
    %123 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 192, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w0_3_loaded = tile.load.async %123 into %w_buf : !tile.event<"w0_3_loaded">
    tile.await %w0_3_loaded
    %outproj0_3_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_3_p0">
    tile.await %outproj0_3_p0
    %outproj0_3_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj0_3_p1">
    tile.await %outproj0_3_p1
    %124 = tile.subview %o_l2_4 offsets = [1, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o1_p0_loaded = tile.load.async %124 into %o0 : !tile.event<"o1_p0_loaded">
    %125 = tile.subview %o_l2_4 offsets = [1, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o1_p1_loaded = tile.load.async %125 into %o1 : !tile.event<"o1_p1_loaded">
    tile.await %o1_p0_loaded, %o1_p1_loaded
    %126 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 256, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_0_loaded = tile.load.async %126 into %w_buf : !tile.event<"w1_0_loaded">
    tile.await %w1_0_loaded
    %outproj1_0_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_0_p0">
    tile.await %outproj1_0_p0
    %outproj1_0_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_0_p1">
    tile.await %outproj1_0_p1
    %127 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 320, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_1_loaded = tile.load.async %127 into %w_buf : !tile.event<"w1_1_loaded">
    tile.await %w1_1_loaded
    %outproj1_1_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_1_p0">
    tile.await %outproj1_1_p0
    %outproj1_1_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_1_p1">
    tile.await %outproj1_1_p1
    %128 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 384, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_2_loaded = tile.load.async %128 into %w_buf : !tile.event<"w1_2_loaded">
    tile.await %w1_2_loaded
    %outproj1_2_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_2_p0">
    tile.await %outproj1_2_p0
    %outproj1_2_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_2_p1">
    tile.await %outproj1_2_p1
    %129 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 448, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w1_3_loaded = tile.load.async %129 into %w_buf : !tile.event<"w1_3_loaded">
    tile.await %w1_3_loaded
    %outproj1_3_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_3_p0">
    tile.await %outproj1_3_p0
    %outproj1_3_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj1_3_p1">
    tile.await %outproj1_3_p1
    %130 = tile.subview %o_l2_4 offsets = [2, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o2_p0_loaded = tile.load.async %130 into %o0 : !tile.event<"o2_p0_loaded">
    %131 = tile.subview %o_l2_4 offsets = [2, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o2_p1_loaded = tile.load.async %131 into %o1 : !tile.event<"o2_p1_loaded">
    tile.await %o2_p0_loaded, %o2_p1_loaded
    %132 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 512, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_0_loaded = tile.load.async %132 into %w_buf : !tile.event<"w2_0_loaded">
    tile.await %w2_0_loaded
    %outproj2_0_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_0_p0">
    tile.await %outproj2_0_p0
    %outproj2_0_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_0_p1">
    tile.await %outproj2_0_p1
    %133 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 576, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_1_loaded = tile.load.async %133 into %w_buf : !tile.event<"w2_1_loaded">
    tile.await %w2_1_loaded
    %outproj2_1_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_1_p0">
    tile.await %outproj2_1_p0
    %outproj2_1_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_1_p1">
    tile.await %outproj2_1_p1
    %134 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 640, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_2_loaded = tile.load.async %134 into %w_buf : !tile.event<"w2_2_loaded">
    tile.await %w2_2_loaded
    %outproj2_2_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_2_p0">
    tile.await %outproj2_2_p0
    %outproj2_2_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_2_p1">
    tile.await %outproj2_2_p1
    %135 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 704, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w2_3_loaded = tile.load.async %135 into %w_buf : !tile.event<"w2_3_loaded">
    tile.await %w2_3_loaded
    %outproj2_3_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_3_p0">
    tile.await %outproj2_3_p0
    %outproj2_3_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj2_3_p1">
    tile.await %outproj2_3_p1
    %136 = tile.subview %o_l2_4 offsets = [3, 0, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o3_p0_loaded = tile.load.async %136 into %o0 : !tile.event<"o3_p0_loaded">
    %137 = tile.subview %o_l2_4 offsets = [3, 64, 0] sizes = [1, 64, 256] strides = [1, 1, 1]
      : !nest.l2_view<1x64x256xbf16>
    %o3_p1_loaded = tile.load.async %137 into %o1 : !tile.event<"o3_p1_loaded">
    tile.await %o3_p0_loaded, %o3_p1_loaded
    %138 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 768, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_0_loaded = tile.load.async %138 into %w_buf : !tile.event<"w3_0_loaded">
    tile.await %w3_0_loaded
    %outproj3_0_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_0_p0">
    tile.await %outproj3_0_p0
    %outproj3_0_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_0_p1">
    tile.await %outproj3_0_p1
    %139 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 832, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_1_loaded = tile.load.async %139 into %w_buf : !tile.event<"w3_1_loaded">
    tile.await %w3_1_loaded
    %outproj3_1_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_1_p0">
    tile.await %outproj3_1_p0
    %outproj3_1_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_1_p1">
    tile.await %outproj3_1_p1
    %140 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 896, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_2_loaded = tile.load.async %140 into %w_buf : !tile.event<"w3_2_loaded">
    tile.await %w3_2_loaded
    %outproj3_2_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_2_p0">
    tile.await %outproj3_2_p0
    %outproj3_2_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_2_p1">
    tile.await %outproj3_2_p1
    %141 = tile.subview %wo_l2 task = %task_6 task_dim = 0 offsets = [0, 960, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %w3_3_loaded = tile.load.async %141 into %w_buf : !tile.event<"w3_3_loaded">
    tile.await %w3_3_loaded
    tile.signal input_released(%task_6)
    %outproj3_3_p0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_3_p0">
    tile.await %outproj3_3_p0
    %outproj3_3_p1 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152 accumulate
      : !tile.event<"outproj3_3_p1">
    tile.await %outproj3_3_p1
    %142 = tile.subview %out_l2 task = %task_6 task_dim = 0 offsets = [0, 0, 0] sizes = [1, 64, 256]
      strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %out0_stored = tile.store.async %acc0_4 into %142 : !tile.event<"out0_stored">
    %143 = tile.subview %out_l2 task = %task_6 task_dim = 0 offsets = [0, 64, 0]
      sizes = [1, 64, 256] strides = [1, 1, 1] : !nest.l2_view<1x64x256xbf16>
    %out1_stored = tile.store.async %acc1_4 into %143 : !tile.event<"out1_stored">
    tile.await %out0_stored, %out1_stored
    tile.signal output_ready(%task_6)
    tile.free %acc0_4
    tile.free %acc1_4
    tile.free %o0
    tile.free %o1
    tile.free %w_buf
    tile.return
  }
  nest.context @prefill_multicontext_ctx(
    %X: !nest.global_memref<8x512x128xbf16>, %WQ: !nest.global_memref<8x4x128x256xbf16>,
    %WK: !nest.global_memref<8x4x128x64xbf16>, %WV: !nest.global_memref<8x4x128x64xbf16>,
    %WO: !nest.global_memref<4x1024x256xbf16>, %OUT: !nest.global_memref<4x4x128x256xbf16>)
    placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1],
      logical_tasks = 64, l2_spm_bytes = 6815744, requested_contexts_per_tile = 4> {
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
    %wo_l2_1 = nest.alloc slot = "wo_l2" role = "in" shape = [4, 1024, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x1024x256xbf16>
    %o_q0 = nest.alloc slot = "o_q0" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q1 = nest.alloc slot = "o_q1" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q2 = nest.alloc slot = "o_q2" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %o_q3 = nest.alloc slot = "o_q3" role = "inout" sharing = "context-local" shape = [4, 128, 256]
      dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %out_q0 = nest.alloc slot = "out_q0" role = "out" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %out_q1 = nest.alloc slot = "out_q1" role = "out" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %out_q2 = nest.alloc slot = "out_q2" role = "out" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %out_q3 = nest.alloc slot = "out_q3" role = "out" shape = [4, 128, 256] dtype = "bf16"
      alignment = 256 : !nest.l2_buffer<4x128x256xbf16>
    %144 = nest.task.range from = 0 to = 4 : !nest.task_range
    %145 = nest.subview %X offsets = [0, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %146 = nest.subview %WQ offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %147 = nest.subview %WK offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %148 = nest.subview %WV offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x = nest.dma.prefetch.async %145 into %x_p0 : !nest.event<"pre_x_0">
    %pre_wq = nest.dma.prefetch.async %146 into %wq_p0 : !nest.event<"pre_wq_0">
    %pre_wk = nest.dma.prefetch.async %147 into %wk_p0 : !nest.event<"pre_wk_0">
    %pre_wv = nest.dma.prefetch.async %148 into %wv_p0 : !nest.event<"pre_wv_0">
    %qkv_grid, %qkv_inrel, %qkv_out = nest.dispatch.tasks.async @qkv_chunk_init l1_mode = 0
      tasks(%144) globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x, %pre_wq, %pre_wk, %pre_wv)
      : (!nest.event<"qkv_grid_0">, !nest.event<"qkv_inrel_0">, !nest.event<"qkv_out_0">)
    %149 = nest.subview %X offsets = [1, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %150 = nest.subview %WQ offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %151 = nest.subview %WK offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %152 = nest.subview %WV offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_1 = nest.dma.prefetch.async %149 into %x_p1 : !nest.event<"pre_x_1">
    %pre_wq_1 = nest.dma.prefetch.async %150 into %wq_p1 : !nest.event<"pre_wq_1">
    %pre_wk_1 = nest.dma.prefetch.async %151 into %wk_p1 : !nest.event<"pre_wk_1">
    %pre_wv_1 = nest.dma.prefetch.async %152 into %wv_p1 : !nest.event<"pre_wv_1">
    %qkv_grid_1, %qkv_inrel_1, %qkv_out_1 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_1, %pre_wq_1, %pre_wk_1, %pre_wv_1, %qkv_out)
      : (!nest.event<"qkv_grid_1">, !nest.event<"qkv_inrel_1">, !nest.event<"qkv_out_1">)
    %153 = nest.subview %X offsets = [2, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %154 = nest.subview %WQ offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %155 = nest.subview %WK offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %156 = nest.subview %WV offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_2 = nest.dma.prefetch.async %153 into %x_p0 depends_on(%qkv_inrel)
      : !nest.event<"pre_x_2">
    %pre_wq_2 = nest.dma.prefetch.async %154 into %wq_p0 depends_on(%qkv_inrel)
      : !nest.event<"pre_wq_2">
    %pre_wk_2 = nest.dma.prefetch.async %155 into %wk_p0 depends_on(%qkv_inrel)
      : !nest.event<"pre_wk_2">
    %pre_wv_2 = nest.dma.prefetch.async %156 into %wv_p0 depends_on(%qkv_inrel)
      : !nest.event<"pre_wv_2">
    %qkv_grid_2, %qkv_inrel_2, %qkv_out_2 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_2, %pre_wq_2, %pre_wk_2, %pre_wv_2, %qkv_out_1)
      : (!nest.event<"qkv_grid_2">, !nest.event<"qkv_inrel_2">, !nest.event<"qkv_out_2">)
    %157 = nest.subview %X offsets = [3, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %158 = nest.subview %WQ offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %159 = nest.subview %WK offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %160 = nest.subview %WV offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_3 = nest.dma.prefetch.async %157 into %x_p1 depends_on(%qkv_inrel_1)
      : !nest.event<"pre_x_3">
    %pre_wq_3 = nest.dma.prefetch.async %158 into %wq_p1 depends_on(%qkv_inrel_1)
      : !nest.event<"pre_wq_3">
    %pre_wk_3 = nest.dma.prefetch.async %159 into %wk_p1 depends_on(%qkv_inrel_1)
      : !nest.event<"pre_wk_3">
    %pre_wv_3 = nest.dma.prefetch.async %160 into %wv_p1 depends_on(%qkv_inrel_1)
      : !nest.event<"pre_wv_3">
    %qkv_grid_3, %qkv_inrel_3, %qkv_out_3 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_3, %pre_wq_3, %pre_wk_3, %pre_wv_3, %qkv_out_2)
      : (!nest.event<"qkv_grid_3">, !nest.event<"qkv_inrel_3">, !nest.event<"qkv_out_3">)
    %161 = nest.subview %X offsets = [4, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %162 = nest.subview %WQ offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %163 = nest.subview %WK offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %164 = nest.subview %WV offsets = [4, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_4 = nest.dma.prefetch.async %161 into %x_p0 depends_on(%qkv_inrel_2)
      : !nest.event<"pre_x_4">
    %pre_wq_4 = nest.dma.prefetch.async %162 into %wq_p0 depends_on(%qkv_inrel_2)
      : !nest.event<"pre_wq_4">
    %pre_wk_4 = nest.dma.prefetch.async %163 into %wk_p0 depends_on(%qkv_inrel_2)
      : !nest.event<"pre_wk_4">
    %pre_wv_4 = nest.dma.prefetch.async %164 into %wv_p0 depends_on(%qkv_inrel_2)
      : !nest.event<"pre_wv_4">
    %qkv_grid_4, %qkv_inrel_4, %qkv_out_4 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_4, %pre_wq_4, %pre_wk_4, %pre_wv_4, %qkv_out_3)
      : (!nest.event<"qkv_grid_4">, !nest.event<"qkv_inrel_4">, !nest.event<"qkv_out_4">)
    %165 = nest.subview %X offsets = [5, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %166 = nest.subview %WQ offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %167 = nest.subview %WK offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %168 = nest.subview %WV offsets = [5, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_5 = nest.dma.prefetch.async %165 into %x_p1 depends_on(%qkv_inrel_3)
      : !nest.event<"pre_x_5">
    %pre_wq_5 = nest.dma.prefetch.async %166 into %wq_p1 depends_on(%qkv_inrel_3)
      : !nest.event<"pre_wq_5">
    %pre_wk_5 = nest.dma.prefetch.async %167 into %wk_p1 depends_on(%qkv_inrel_3)
      : !nest.event<"pre_wk_5">
    %pre_wv_5 = nest.dma.prefetch.async %168 into %wv_p1 depends_on(%qkv_inrel_3)
      : !nest.event<"pre_wv_5">
    %qkv_grid_5, %qkv_inrel_5, %qkv_out_5 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_5, %pre_wq_5, %pre_wk_5, %pre_wv_5, %qkv_out_4)
      : (!nest.event<"qkv_grid_5">, !nest.event<"qkv_inrel_5">, !nest.event<"qkv_out_5">)
    %169 = nest.subview %X offsets = [6, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %170 = nest.subview %WQ offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %171 = nest.subview %WK offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %172 = nest.subview %WV offsets = [6, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_6 = nest.dma.prefetch.async %169 into %x_p0 depends_on(%qkv_inrel_4)
      : !nest.event<"pre_x_6">
    %pre_wq_6 = nest.dma.prefetch.async %170 into %wq_p0 depends_on(%qkv_inrel_4)
      : !nest.event<"pre_wq_6">
    %pre_wk_6 = nest.dma.prefetch.async %171 into %wk_p0 depends_on(%qkv_inrel_4)
      : !nest.event<"pre_wk_6">
    %pre_wv_6 = nest.dma.prefetch.async %172 into %wv_p0 depends_on(%qkv_inrel_4)
      : !nest.event<"pre_wv_6">
    %qkv_grid_6, %qkv_inrel_6, %qkv_out_6 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p0, %wq_p0, %wk_p0, %wv_p0, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_6, %pre_wq_6, %pre_wk_6, %pre_wv_6, %qkv_out_5)
      : (!nest.event<"qkv_grid_6">, !nest.event<"qkv_inrel_6">, !nest.event<"qkv_out_6">)
    %173 = nest.subview %X offsets = [7, 0, 0] sizes = [1, 512, 128] strides = [1, 1, 1]
      : !nest.global_view<1x512x128xbf16>
    %174 = nest.subview %WQ offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %175 = nest.subview %WK offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %176 = nest.subview %WV offsets = [7, 0, 0, 0] sizes = [1, 4, 128, 64] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x64xbf16>
    %pre_x_7 = nest.dma.prefetch.async %173 into %x_p1 depends_on(%qkv_inrel_5)
      : !nest.event<"pre_x_7">
    %pre_wq_7 = nest.dma.prefetch.async %174 into %wq_p1 depends_on(%qkv_inrel_5)
      : !nest.event<"pre_wq_7">
    %pre_wk_7 = nest.dma.prefetch.async %175 into %wk_p1 depends_on(%qkv_inrel_5)
      : !nest.event<"pre_wk_7">
    %pre_wv_7 = nest.dma.prefetch.async %176 into %wv_p1 depends_on(%qkv_inrel_5)
      : !nest.event<"pre_wv_7">
    %qkv_grid_7, %qkv_inrel_7, %qkv_out_7 = nest.dispatch.tasks.async @qkv_chunk_accum l1_mode = 0
      tasks(%144) globals() bindings(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6)
      ins(%x_p1, %wq_p1, %wk_p1, %wv_p1, %q_l2_6, %k_l2_6, %v_l2_6) outs(%q_l2_6, %k_l2_6, %v_l2_6)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%pre_x_7, %pre_wq_7, %pre_wk_7, %pre_wv_7, %qkv_out_6)
      : (!nest.event<"qkv_grid_7">, !nest.event<"qkv_inrel_7">, !nest.event<"qkv_out_7">)
    %177 = nest.subview %WO offsets = [0, 0, 0] sizes = [4, 1024, 256] strides = [1, 1, 1]
      : !nest.global_view<4x1024x256xbf16>
    %pre_wo = nest.dma.prefetch.async %177 into %wo_l2_1 depends_on(%qkv_inrel_7)
      : !nest.event<"pre_wo">
    %att_grid, %att_inrel, %att_out = nest.dispatch.tasks.async @prefill_attention_q0 l1_mode = 0
      tasks(%144) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q0)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_out_7)
      : (!nest.event<"att_grid_0">, !nest.event<"att_inrel_0">, !nest.event<"att_out_0">)
    %att_grid_1, %att_inrel_1, %att_out_1 = nest.dispatch.tasks.async @prefill_attention_q1
      l1_mode = 0 tasks(%144) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q1)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_out_7)
      : (!nest.event<"att_grid_1">, !nest.event<"att_inrel_1">, !nest.event<"att_out_1">)
    %att_grid_2, %att_inrel_2, %att_out_2 = nest.dispatch.tasks.async @prefill_attention_q2
      l1_mode = 0 tasks(%144) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q2)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_out_7)
      : (!nest.event<"att_grid_2">, !nest.event<"att_inrel_2">, !nest.event<"att_out_2">)
    %att_grid_3, %att_inrel_3, %att_out_3 = nest.dispatch.tasks.async @prefill_attention_q3
      l1_mode = 0 tasks(%144) globals() bindings(%q_l2_6, %k_l2_6, %v_l2_6, %o_q3)
      ins(%q_l2_6, %k_l2_6, %v_l2_6) outs(%o_q3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%qkv_out_7)
      : (!nest.event<"att_grid_3">, !nest.event<"att_inrel_3">, !nest.event<"att_out_3">)
    %pj_grid, %pj_inrel, %pj_out = nest.dispatch.tasks.async @prefill_outproj_tile l1_mode = 0
      tasks(%144) globals() bindings(%o_q0, %wo_l2_1, %out_q0) ins(%o_q0, %wo_l2_1) outs(%out_q0)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out, %pre_wo)
      : (!nest.event<"pj_grid_0">, !nest.event<"pj_inrel_0">, !nest.event<"pj_out_0">)
    %pj_grid_1, %pj_inrel_1, %pj_out_1 = nest.dispatch.tasks.async @prefill_outproj_tile l1_mode = 0
      tasks(%144) globals() bindings(%o_q1, %wo_l2_1, %out_q1) ins(%o_q1, %wo_l2_1) outs(%out_q1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_1, %pre_wo)
      : (!nest.event<"pj_grid_1">, !nest.event<"pj_inrel_1">, !nest.event<"pj_out_1">)
    %pj_grid_2, %pj_inrel_2, %pj_out_2 = nest.dispatch.tasks.async @prefill_outproj_tile l1_mode = 0
      tasks(%144) globals() bindings(%o_q2, %wo_l2_1, %out_q2) ins(%o_q2, %wo_l2_1) outs(%out_q2)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_2, %pre_wo)
      : (!nest.event<"pj_grid_2">, !nest.event<"pj_inrel_2">, !nest.event<"pj_out_2">)
    %pj_grid_3, %pj_inrel_3, %pj_out_3 = nest.dispatch.tasks.async @prefill_outproj_tile l1_mode = 0
      tasks(%144) globals() bindings(%o_q3, %wo_l2_1, %out_q3) ins(%o_q3, %wo_l2_1) outs(%out_q3)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%att_out_3, %pre_wo)
      : (!nest.event<"pj_grid_3">, !nest.event<"pj_inrel_3">, !nest.event<"pj_out_3">)
    %178 = nest.subview %OUT offsets = [0, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %out_store = nest.dma.store.async %out_q0 into %178 depends_on(%pj_out)
      : !nest.event<"out_store_0">
    %179 = nest.subview %OUT offsets = [1, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %out_store_1 = nest.dma.store.async %out_q1 into %179 depends_on(%pj_out_1)
      : !nest.event<"out_store_1">
    %180 = nest.subview %OUT offsets = [2, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %out_store_2 = nest.dma.store.async %out_q2 into %180 depends_on(%pj_out_2)
      : !nest.event<"out_store_2">
    %181 = nest.subview %OUT offsets = [3, 0, 0, 0] sizes = [1, 4, 128, 256] strides = [1, 1, 1, 1]
      : !nest.global_view<1x4x128x256xbf16>
    %out_store_3 = nest.dma.store.async %out_q3 into %181 depends_on(%pj_out_3)
      : !nest.event<"out_store_3">
    nest.release %x_p0 depends_on(
      %pre_x, %pre_x_2, %pre_x_4, %pre_x_6, %qkv_inrel, %qkv_inrel_2, %qkv_inrel_4, %qkv_inrel_6)
    nest.release %wq_p0 depends_on(
      %pre_wq, %pre_wq_2, %pre_wq_4, %pre_wq_6, %qkv_inrel, %qkv_inrel_2, %qkv_inrel_4,
      %qkv_inrel_6)
    nest.release %wk_p0 depends_on(
      %pre_wk, %pre_wk_2, %pre_wk_4, %pre_wk_6, %qkv_inrel, %qkv_inrel_2, %qkv_inrel_4,
      %qkv_inrel_6)
    nest.release %wv_p0 depends_on(
      %pre_wv, %pre_wv_2, %pre_wv_4, %pre_wv_6, %qkv_inrel, %qkv_inrel_2, %qkv_inrel_4,
      %qkv_inrel_6)
    nest.release %x_p1 depends_on(
      %pre_x_1, %pre_x_3, %pre_x_5, %pre_x_7, %qkv_inrel_1, %qkv_inrel_3, %qkv_inrel_5,
      %qkv_inrel_7)
    nest.release %wq_p1 depends_on(
      %pre_wq_1, %pre_wq_3, %pre_wq_5, %pre_wq_7, %qkv_inrel_1, %qkv_inrel_3, %qkv_inrel_5,
      %qkv_inrel_7)
    nest.release %wk_p1 depends_on(
      %pre_wk_1, %pre_wk_3, %pre_wk_5, %pre_wk_7, %qkv_inrel_1, %qkv_inrel_3, %qkv_inrel_5,
      %qkv_inrel_7)
    nest.release %wv_p1 depends_on(
      %pre_wv_1, %pre_wv_3, %pre_wv_5, %pre_wv_7, %qkv_inrel_1, %qkv_inrel_3, %qkv_inrel_5,
      %qkv_inrel_7)
    nest.release %q_l2_6 depends_on(
      %qkv_inrel_1, %qkv_inrel_2, %qkv_inrel_3, %qkv_inrel_4, %qkv_inrel_5, %qkv_inrel_6,
      %qkv_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3, %qkv_out, %qkv_out_1,
      %qkv_out_2, %qkv_out_3, %qkv_out_4, %qkv_out_5, %qkv_out_6, %qkv_out_7)
    nest.release %k_l2_6 depends_on(
      %qkv_inrel_1, %qkv_inrel_2, %qkv_inrel_3, %qkv_inrel_4, %qkv_inrel_5, %qkv_inrel_6,
      %qkv_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3, %qkv_out, %qkv_out_1,
      %qkv_out_2, %qkv_out_3, %qkv_out_4, %qkv_out_5, %qkv_out_6, %qkv_out_7)
    nest.release %v_l2_6 depends_on(
      %qkv_inrel_1, %qkv_inrel_2, %qkv_inrel_3, %qkv_inrel_4, %qkv_inrel_5, %qkv_inrel_6,
      %qkv_inrel_7, %att_inrel, %att_inrel_1, %att_inrel_2, %att_inrel_3, %qkv_out, %qkv_out_1,
      %qkv_out_2, %qkv_out_3, %qkv_out_4, %qkv_out_5, %qkv_out_6, %qkv_out_7)
    nest.release %wo_l2_1 depends_on(%pre_wo, %pj_inrel, %pj_inrel_1, %pj_inrel_2, %pj_inrel_3)
    nest.release %o_q0 depends_on(%att_out, %pj_inrel)
    nest.release %out_q0 depends_on(%out_store)
    nest.release %o_q1 depends_on(%att_out_1, %pj_inrel_1)
    nest.release %out_q1 depends_on(%out_store_1)
    nest.release %o_q2 depends_on(%att_out_2, %pj_inrel_2)
    nest.release %out_q2 depends_on(%out_store_2)
    nest.release %o_q3 depends_on(%att_out_3, %pj_inrel_3)
    nest.release %out_q3 depends_on(%out_store_3)
    nest.await %qkv_grid, %qkv_grid_1, %qkv_grid_2, %qkv_grid_3, %qkv_grid_4, %qkv_grid_5,
      %qkv_grid_6, %qkv_grid_7, %att_grid, %att_grid_1, %att_grid_2, %att_grid_3, %pj_grid,
      %pj_grid_1, %pj_grid_2, %pj_grid_3, %out_store, %out_store_1, %out_store_2, %out_store_3
    nest.return
  }
  nexus.program @transformer_prefill_multicontext(
    %X_1: !nest.global_memref<8x512x128xbf16>, %WQ_1: !nest.global_memref<8x4x128x256xbf16>,
    %WK_1: !nest.global_memref<8x4x128x64xbf16>, %WV_1: !nest.global_memref<8x4x128x64xbf16>,
    %WO_1: !nest.global_memref<4x1024x256xbf16>, %OUT_1: !nest.global_memref<4x4x128x256xbf16>) {
    %prefill_multicontext_done =
      nexus.submit_context.async @prefill_multicontext_ctx(%X_1, %WQ_1, %WK_1, %WV_1, %WO_1, %OUT_1)
      : !nexus.event<"prefill_multicontext_done">
    nexus.await %prefill_multicontext_done
    nexus.return
  }
}
